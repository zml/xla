/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/musa/musa_mudnn_api.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include <dlfcn.h>
#include "xla/stream_executor/musa/mudnn_shim/mudnn_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace stream_executor::musa {
namespace {

constexpr char kMuDnnV1GetterSymbol[] = "xla_musa_mudnn_get_api_v1";
constexpr size_t kMaxConfiguredPathBytes = 4096;
constexpr XlaMusaMuDnnVersion kQualifiedMuDnnVersion = {
    /*major=*/2, /*minor=*/8, /*patch=*/0};

absl::Status MuDnnStatus(int32_t status, absl::string_view operation) {
  if (status == XLA_MUSA_MUDNN_STATUS_SUCCESS) return absl::OkStatus();
  const std::string message =
      absl::StrCat("muDNN ", operation, " failed with shim status ", status);
  switch (status) {
    case XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT:
      return absl::InvalidArgumentError(message);
    case XLA_MUSA_MUDNN_STATUS_NOT_SUPPORTED:
      return absl::UnimplementedError(message);
    case XLA_MUSA_MUDNN_STATUS_OUT_OF_RANGE:
      return absl::OutOfRangeError(message);
    case XLA_MUSA_MUDNN_STATUS_RESOURCE_EXHAUSTED:
      return absl::ResourceExhaustedError(message);
    case XLA_MUSA_MUDNN_STATUS_FAILED_PRECONDITION:
      return absl::FailedPreconditionError(message);
    default:
      return absl::InternalError(message);
  }
}

std::string Dirname(absl::string_view path) {
  const size_t slash = path.rfind('/');
  if (slash == absl::string_view::npos) return ".";
  if (slash == 0) return "/";
  return std::string(path.substr(0, slash));
}

std::string JoinPath(absl::string_view directory, absl::string_view basename) {
  return absl::StrCat(
      directory, directory.size() == 1 && directory.front() == '/' ? "" : "/",
      basename);
}

absl::StatusOr<std::vector<std::string>> GetMusaMuDnnShimCandidatesImpl(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  if (configured_path.has_value()) {
    if (configured_path->empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          kMusaMuDnnShimEnvironment, " must not be empty when set"));
    }
    if (configured_path->size() > kMaxConfiguredPathBytes) {
      return absl::InvalidArgumentError(
          absl::StrCat(kMusaMuDnnShimEnvironment, " exceeds ",
                       kMaxConfiguredPathBytes, " bytes"));
    }
    if (configured_path->front() != '/') {
      return absl::InvalidArgumentError(
          absl::StrCat(kMusaMuDnnShimEnvironment,
                       " must be an absolute path: ", *configured_path));
    }
    return std::vector<std::string>{std::string(*configured_path)};
  }

  std::vector<std::string> candidates;
  if (!plugin_path.empty()) {
    candidates.push_back(JoinPath(Dirname(plugin_path), kMusaMuDnnShimSoname));
  }
  candidates.emplace_back(kMusaMuDnnShimSoname);
  return candidates;
}

class ErrorMusaSymbolLoader final : public internal::MusaSymbolLoader {
 public:
  explicit ErrorMusaSymbolLoader(absl::Status status)
      : status_(std::move(status)) {}

  absl::Status Load() override { return status_; }
  absl::StatusOr<void*> Resolve(absl::string_view) const override {
    return status_;
  }
  absl::string_view loaded_path() const override { return {}; }

 private:
  absl::Status status_;
};

std::unique_ptr<internal::MusaSymbolLoader> CreateShimLoader(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  absl::StatusOr<std::vector<std::string>> candidates =
      GetMusaMuDnnShimCandidatesImpl(configured_path, plugin_path);
  if (!candidates.ok()) {
    return std::make_unique<ErrorMusaSymbolLoader>(candidates.status());
  }
  return internal::CreateMusaDsoLoader(
      *std::move(candidates),
      /*fail_if_not_found=*/configured_path.has_value());
}

std::unique_ptr<internal::MusaSymbolLoader> CreateDefaultShimLoader() {
  std::optional<absl::string_view> configured_path;
  if (const char* configured = std::getenv(kMusaMuDnnShimEnvironment);
      configured != nullptr) {
    configured_path = configured;
  }

  std::string plugin_path;
  Dl_info info = {};
  if (dladdr(reinterpret_cast<void*>(&GetMusaMuDnnApi), &info) != 0 &&
      info.dli_fname != nullptr && info.dli_fname[0] != '\0') {
    plugin_path = info.dli_fname;
  }
  return CreateShimLoader(configured_path, plugin_path);
}

absl::Status ValidateApi(const XlaMusaMuDnnApiV1* api,
                         absl::string_view loaded_path) {
  if (api == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muDNN shim ", loaded_path, " returned a null v1 API table"));
  }
  if (api->struct_size < XLA_MUSA_MUDNN_API_V1_MIN_STRUCT_SIZE) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muDNN shim ", loaded_path,
        " v1 table is too small: ", api->struct_size, "; expected at least ",
        XLA_MUSA_MUDNN_API_V1_MIN_STRUCT_SIZE));
  }
  if (api->abi_version != XLA_MUSA_MUDNN_ABI_VERSION_1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muDNN shim ", loaded_path, " has ABI version ", api->abi_version,
        "; expected ", XLA_MUSA_MUDNN_ABI_VERSION_1));
  }
  if ((api->capabilities & XLA_MUSA_MUDNN_CAPABILITIES_V1) !=
      XLA_MUSA_MUDNN_CAPABILITIES_V1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muDNN shim ", loaded_path, " ABI v1 capabilities are incomplete: ",
        api->capabilities, "; required mask ", XLA_MUSA_MUDNN_CAPABILITIES_V1));
  }
  if (api->get_version == nullptr || api->create_handle == nullptr ||
      api->destroy_handle == nullptr || api->set_stream == nullptr ||
      api->set_allow_tf32 == nullptr || api->create_tensor == nullptr ||
      api->destroy_tensor == nullptr || api->configure_tensor == nullptr ||
      api->set_tensor_address == nullptr ||
      api->create_convolution == nullptr ||
      api->destroy_convolution == nullptr ||
      api->configure_convolution == nullptr ||
      api->get_recommended_algorithm == nullptr ||
      api->get_workspace_size == nullptr || api->convolve == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("muDNN shim ", loaded_path,
                     " has a null required function in its v1 API table"));
  }
  return absl::OkStatus();
}

absl::Status ValidateVersion(const XlaMusaMuDnnVersion& version,
                             absl::string_view loaded_path) {
  if (version.major != kQualifiedMuDnnVersion.major ||
      version.minor != kQualifiedMuDnnVersion.minor ||
      version.patch != kQualifiedMuDnnVersion.patch) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muDNN shim ", loaded_path, " reports unqualified muDNN version ",
        version.major, ".", version.minor, ".", version.patch, "; expected ",
        kQualifiedMuDnnVersion.major, ".", kQualifiedMuDnnVersion.minor, ".",
        kQualifiedMuDnnVersion.patch));
  }
  return absl::OkStatus();
}

absl::Status ValidateDataType(XlaMusaMuDnnDataType data_type) {
  switch (data_type) {
    case XLA_MUSA_MUDNN_DATA_TYPE_F16:
    case XLA_MUSA_MUDNN_DATA_TYPE_BF16:
    case XLA_MUSA_MUDNN_DATA_TYPE_F32:
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("unknown normalized muDNN data type ", data_type));
  }
}

absl::Status ValidateTensorFormat(XlaMusaMuDnnTensorFormat format) {
  switch (format) {
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_UNKNOWN:
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NCW:
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NWC:
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW:
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NHWC:
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_HWCN:
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NCDHW:
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NDHWC:
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_DHWCN:
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("unknown normalized muDNN tensor format ", format));
  }
}

absl::Status ValidateConvolutionKind(XlaMusaMuDnnConvolutionKind kind) {
  switch (kind) {
    case XLA_MUSA_MUDNN_CONVOLUTION_FORWARD:
    case XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA:
    case XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER:
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("unknown normalized muDNN convolution kind ", kind));
  }
}

absl::Status ValidateAlgorithm(XlaMusaMuDnnAlgorithm algorithm,
                               bool allow_recommended) {
  switch (algorithm) {
    case XLA_MUSA_MUDNN_ALGORITHM_RECOMMENDED:
      if (allow_recommended) return absl::OkStatus();
      break;
    case XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM:
    case XLA_MUSA_MUDNN_ALGORITHM_DIRECT:
    case XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED:
    case XLA_MUSA_MUDNN_ALGORITHM_GEMM:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("unknown normalized muDNN algorithm ", algorithm));
}

absl::Status ValidateConvolutionArguments(void* handle, void* convolution,
                                          XlaMusaMuDnnConvolutionKind kind,
                                          void* output, const void* data,
                                          const void* filter) {
  if (handle == nullptr || convolution == nullptr) {
    return absl::FailedPreconditionError(
        "muDNN handle and convolution must be non-null");
  }
  if (output == nullptr || data == nullptr || filter == nullptr) {
    return absl::InvalidArgumentError(
        "muDNN convolution requires non-null output, data, and filter "
        "descriptors");
  }
  return ValidateConvolutionKind(kind);
}

absl::Status ValidateWorkspaceAllocator(
    const XlaMusaMuDnnWorkspaceAllocator* workspace_allocator) {
  if (workspace_allocator == nullptr) {
    return absl::InvalidArgumentError(
        "muDNN workspace allocator must be non-null");
  }
  if (workspace_allocator->struct_size <
      XLA_MUSA_MUDNN_WORKSPACE_ALLOCATOR_MIN_STRUCT_SIZE) {
    return absl::InvalidArgumentError(
        absl::StrCat("muDNN workspace allocator is too small: ",
                     workspace_allocator->struct_size, "; expected at least ",
                     XLA_MUSA_MUDNN_WORKSPACE_ALLOCATOR_MIN_STRUCT_SIZE));
  }
  if (workspace_allocator->reserved != 0) {
    return absl::InvalidArgumentError(
        "muDNN workspace allocator reserved field must be zero");
  }
  if (workspace_allocator->allocate == nullptr ||
      workspace_allocator->release == nullptr) {
    return absl::InvalidArgumentError(
        "muDNN workspace allocator callbacks must be non-null");
  }
  return absl::OkStatus();
}

}  // namespace

namespace internal {

absl::StatusOr<std::vector<std::string>> GetMusaMuDnnShimCandidates(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  return GetMusaMuDnnShimCandidatesImpl(configured_path, plugin_path);
}

std::unique_ptr<MusaSymbolLoader> CreateMusaMuDnnShimLoader(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  return CreateShimLoader(configured_path, plugin_path);
}

}  // namespace internal

MusaMuDnnApi::MusaMuDnnApi(std::unique_ptr<internal::MusaSymbolLoader> loader)
    : loader_(std::move(loader)) {}

MusaMuDnnApi::~MusaMuDnnApi() = default;

std::unique_ptr<MusaMuDnnApi> MusaMuDnnApi::CreateForTesting(
    std::unique_ptr<internal::MusaSymbolLoader> loader) {
  return std::make_unique<MusaMuDnnApi>(std::move(loader));
}

absl::Status MusaMuDnnApi::Init() const {
  absl::call_once(load_once_, [this] { load_status_ = Initialize(); });
  return load_status_;
}

absl::Status MusaMuDnnApi::Initialize() const {
  if (loader_ == nullptr) {
    return absl::InternalError("muDNN shim symbol loader is null");
  }
  RETURN_IF_ERROR(loader_->Load());
  absl::StatusOr<void*> getter = loader_->Resolve(kMuDnnV1GetterSymbol);
  if (!getter.ok() || *getter == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("muDNN shim ", loader_->loaded_path(),
                     " does not export a usable ", kMuDnnV1GetterSymbol,
                     getter.ok() ? "" : absl::StrCat(": ", getter.status())));
  }

  auto get_api = reinterpret_cast<XlaMusaMuDnnGetApiV1Fn>(*getter);
  const XlaMusaMuDnnApiV1* api = get_api();
  RETURN_IF_ERROR(ValidateApi(api, loader_->loaded_path()));

  XlaMusaMuDnnVersion version = {};
  RETURN_IF_ERROR(MuDnnStatus(api->get_version(&version), "get_version"));
  RETURN_IF_ERROR(ValidateVersion(version, loader_->loaded_path()));

  api_ = api;
  version_ = version;
  return absl::OkStatus();
}

uint64_t MusaMuDnnApi::capabilities() const {
  return Init().ok() ? api_->capabilities : 0;
}

uint32_t MusaMuDnnApi::abi_version() const {
  return Init().ok() ? api_->abi_version : 0;
}

absl::string_view MusaMuDnnApi::loaded_path() const {
  return loader_ == nullptr ? absl::string_view() : loader_->loaded_path();
}

absl::Status MusaMuDnnApi::GetVersion(XlaMusaMuDnnVersion* version) const {
  RETURN_IF_ERROR(Init());
  if (version == nullptr) {
    return absl::InvalidArgumentError("muDNN version output is null");
  }
  *version = version_;
  return absl::OkStatus();
}

absl::Status MusaMuDnnApi::CreateHandle(int32_t device_ordinal,
                                        void** handle) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::InvalidArgumentError("muDNN handle output is null");
  }
  *handle = nullptr;
  if (device_ordinal < 0) {
    return absl::InvalidArgumentError(
        "muDNN device ordinal must be non-negative");
  }
  RETURN_IF_ERROR(MuDnnStatus(api_->create_handle(device_ordinal, handle),
                              "create_handle"));
  if (*handle == nullptr) {
    return absl::InternalError("muDNN create_handle returned a null handle");
  }
  return absl::OkStatus();
}

absl::Status MusaMuDnnApi::DestroyHandle(void* handle) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) return absl::OkStatus();
  return MuDnnStatus(api_->destroy_handle(handle), "destroy_handle");
}

absl::Status MusaMuDnnApi::SetStream(void* handle, void* stream) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muDNN handle is null");
  }
  return MuDnnStatus(api_->set_stream(handle, stream), "set_stream");
}

absl::Status MusaMuDnnApi::SetAllowTf32(void* handle, bool allow_tf32) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muDNN handle is null");
  }
  return MuDnnStatus(api_->set_allow_tf32(handle, allow_tf32 ? 1 : 0),
                     "set_allow_tf32");
}

absl::Status MusaMuDnnApi::CreateTensor(void** tensor) const {
  RETURN_IF_ERROR(Init());
  if (tensor == nullptr) {
    return absl::InvalidArgumentError("muDNN tensor output is null");
  }
  *tensor = nullptr;
  RETURN_IF_ERROR(MuDnnStatus(api_->create_tensor(tensor), "create_tensor"));
  if (*tensor == nullptr) {
    return absl::InternalError("muDNN create_tensor returned a null tensor");
  }
  return absl::OkStatus();
}

absl::Status MusaMuDnnApi::DestroyTensor(void* tensor) const {
  RETURN_IF_ERROR(Init());
  if (tensor == nullptr) return absl::OkStatus();
  return MuDnnStatus(api_->destroy_tensor(tensor), "destroy_tensor");
}

absl::Status MusaMuDnnApi::ConfigureTensor(void* tensor,
                                           XlaMusaMuDnnDataType data_type,
                                           XlaMusaMuDnnTensorFormat format,
                                           int32_t rank,
                                           const int64_t* dimensions,
                                           const int64_t* strides) const {
  RETURN_IF_ERROR(Init());
  if (tensor == nullptr) {
    return absl::FailedPreconditionError("muDNN tensor is null");
  }
  RETURN_IF_ERROR(ValidateDataType(data_type));
  RETURN_IF_ERROR(ValidateTensorFormat(format));
  if (rank < 3 || rank > 5) {
    return absl::InvalidArgumentError(
        "muDNN ABI v1 supports tensor ranks 3 through 5");
  }
  if (dimensions == nullptr || strides == nullptr) {
    return absl::InvalidArgumentError(
        "muDNN tensor dimensions and strides must be non-null");
  }
  for (int32_t index = 0; index < rank; ++index) {
    if (dimensions[index] <= 0 || strides[index] <= 0) {
      return absl::InvalidArgumentError(
          "muDNN tensor dimensions and strides must be positive");
    }
  }
  return MuDnnStatus(api_->configure_tensor(tensor, data_type, format, rank,
                                            dimensions, strides),
                     "configure_tensor");
}

absl::Status MusaMuDnnApi::SetTensorAddress(void* tensor, void* address) const {
  RETURN_IF_ERROR(Init());
  if (tensor == nullptr) {
    return absl::FailedPreconditionError("muDNN tensor is null");
  }
  if (address == nullptr) {
    return absl::InvalidArgumentError("muDNN tensor address is null");
  }
  return MuDnnStatus(api_->set_tensor_address(tensor, address),
                     "set_tensor_address");
}

absl::Status MusaMuDnnApi::CreateConvolution(void** convolution) const {
  RETURN_IF_ERROR(Init());
  if (convolution == nullptr) {
    return absl::InvalidArgumentError("muDNN convolution output is null");
  }
  *convolution = nullptr;
  RETURN_IF_ERROR(
      MuDnnStatus(api_->create_convolution(convolution), "create_convolution"));
  if (*convolution == nullptr) {
    return absl::InternalError(
        "muDNN create_convolution returned a null convolution");
  }
  return absl::OkStatus();
}

absl::Status MusaMuDnnApi::DestroyConvolution(void* convolution) const {
  RETURN_IF_ERROR(Init());
  if (convolution == nullptr) return absl::OkStatus();
  return MuDnnStatus(api_->destroy_convolution(convolution),
                     "destroy_convolution");
}

absl::Status MusaMuDnnApi::ConfigureConvolution(void* convolution,
                                                int32_t spatial_rank,
                                                const int64_t* padding,
                                                const int64_t* strides,
                                                const int64_t* dilations,
                                                int64_t group_count) const {
  RETURN_IF_ERROR(Init());
  if (convolution == nullptr) {
    return absl::FailedPreconditionError("muDNN convolution is null");
  }
  if (spatial_rank < 1 || spatial_rank > 3) {
    return absl::InvalidArgumentError(
        "muDNN ABI v1 supports convolution spatial ranks 1 through 3");
  }
  if (padding == nullptr || strides == nullptr || dilations == nullptr) {
    return absl::InvalidArgumentError(
        "muDNN convolution geometry arrays must be non-null");
  }
  if (group_count <= 0) {
    return absl::InvalidArgumentError(
        "muDNN convolution group count must be positive");
  }
  for (int32_t index = 0; index < spatial_rank; ++index) {
    if (padding[index] < 0 || strides[index] <= 0 || dilations[index] <= 0) {
      return absl::InvalidArgumentError(
          "muDNN convolution requires non-negative padding and positive "
          "strides and dilations");
    }
  }
  return MuDnnStatus(
      api_->configure_convolution(convolution, spatial_rank, padding, strides,
                                  dilations, group_count),
      "configure_convolution");
}

absl::Status MusaMuDnnApi::GetRecommendedAlgorithm(
    void* handle, void* convolution, XlaMusaMuDnnConvolutionKind kind,
    void* output, const void* data, const void* filter,
    XlaMusaMuDnnAlgorithm* algorithm) const {
  RETURN_IF_ERROR(Init());
  RETURN_IF_ERROR(ValidateConvolutionArguments(handle, convolution, kind,
                                               output, data, filter));
  if (algorithm == nullptr) {
    return absl::InvalidArgumentError("muDNN algorithm output is null");
  }
  *algorithm = XLA_MUSA_MUDNN_ALGORITHM_RECOMMENDED;
  RETURN_IF_ERROR(MuDnnStatus(
      api_->get_recommended_algorithm(handle, convolution, kind, output, data,
                                      filter, algorithm),
      "get_recommended_algorithm"));
  absl::Status algorithm_status =
      ValidateAlgorithm(*algorithm, /*allow_recommended=*/false);
  if (!algorithm_status.ok()) {
    return absl::InternalError(absl::StrCat(
        "muDNN shim returned an invalid recommended algorithm: ", *algorithm));
  }
  return absl::OkStatus();
}

absl::Status MusaMuDnnApi::GetWorkspaceSize(
    void* handle, void* convolution, XlaMusaMuDnnConvolutionKind kind,
    void* output, const void* data, const void* filter,
    XlaMusaMuDnnAlgorithm algorithm, uint64_t* workspace_size_bytes) const {
  RETURN_IF_ERROR(Init());
  RETURN_IF_ERROR(ValidateConvolutionArguments(handle, convolution, kind,
                                               output, data, filter));
  RETURN_IF_ERROR(ValidateAlgorithm(algorithm, /*allow_recommended=*/true));
  if (workspace_size_bytes == nullptr) {
    return absl::InvalidArgumentError("muDNN workspace-size output is null");
  }
  *workspace_size_bytes = 0;
  return MuDnnStatus(
      api_->get_workspace_size(handle, convolution, kind, output, data, filter,
                               algorithm, workspace_size_bytes),
      "get_workspace_size");
}

absl::Status MusaMuDnnApi::Convolve(
    void* handle, void* convolution, XlaMusaMuDnnConvolutionKind kind,
    void* output, const void* data, const void* filter,
    XlaMusaMuDnnAlgorithm algorithm,
    const XlaMusaMuDnnWorkspaceAllocator* workspace_allocator) const {
  RETURN_IF_ERROR(Init());
  RETURN_IF_ERROR(ValidateConvolutionArguments(handle, convolution, kind,
                                               output, data, filter));
  RETURN_IF_ERROR(ValidateAlgorithm(algorithm, /*allow_recommended=*/true));
  RETURN_IF_ERROR(ValidateWorkspaceAllocator(workspace_allocator));
  return MuDnnStatus(api_->convolve(handle, convolution, kind, output, data,
                                    filter, algorithm, workspace_allocator),
                     "convolve");
}

MusaMuDnnApi* GetMusaMuDnnApi() {
  static auto* api = new MusaMuDnnApi(CreateDefaultShimLoader());
  return api;
}

absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaMuDnnOptionalLibraryAbis() {
  MusaMuDnnApi* api = GetMusaMuDnnApi();
  absl::Status status = api->Init();
  if (absl::IsNotFound(status)) {
    return std::vector<MusaOptionalLibraryAbi>{};
  }
  RETURN_IF_ERROR(status);
  return std::vector<MusaOptionalLibraryAbi>{{
      kMusaMuDnnLibraryAbiName,
      kMusaMuDnnLibraryAbiVersion,
      kMusaMuDnnAbiFingerprintV1,
  }};
}

}  // namespace stream_executor::musa
