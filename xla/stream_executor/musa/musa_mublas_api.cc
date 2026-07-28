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

#include "xla/stream_executor/musa/musa_mublas_api.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include <dlfcn.h>
#include "xla/stream_executor/musa/mublas_shim/mublas_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace stream_executor::musa {
namespace {

constexpr char kMuBlasV1GetterSymbol[] = "xla_musa_mublas_get_api_v1";
constexpr char kMuBlasV2GetterSymbol[] = "xla_musa_mublas_get_api_v2";
constexpr size_t kMaxConfiguredPathBytes = 4096;

using GetApiV1Fn = const XlaMusaMuBlasApiV1* (*)();
using GetApiV2Fn = const XlaMusaMuBlasApiV2* (*)();

absl::Status MuBlasStatus(int32_t status, absl::string_view operation) {
  const std::string message =
      absl::StrCat("muBLAS ", operation, " failed with status ", status);
  switch (status) {
    case XLA_MUSA_MUBLAS_STATUS_SUCCESS:
      return absl::OkStatus();
    case XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT:
      return absl::InvalidArgumentError(message);
    case XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED:
      return absl::UnimplementedError(message);
    case XLA_MUSA_MUBLAS_STATUS_OUT_OF_RANGE:
      return absl::OutOfRangeError(message);
    case XLA_MUSA_MUBLAS_STATUS_RESOURCE_EXHAUSTED:
      return absl::ResourceExhaustedError(message);
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

absl::StatusOr<std::vector<std::string>> GetMusaMuBlasShimCandidatesImpl(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  if (configured_path.has_value()) {
    if (configured_path->empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          kMusaMuBlasShimEnvironment, " must not be empty when set"));
    }
    if (configured_path->size() > kMaxConfiguredPathBytes) {
      return absl::InvalidArgumentError(
          absl::StrCat(kMusaMuBlasShimEnvironment, " exceeds ",
                       kMaxConfiguredPathBytes, " bytes"));
    }
    if (configured_path->front() != '/') {
      return absl::InvalidArgumentError(
          absl::StrCat(kMusaMuBlasShimEnvironment,
                       " must be an absolute path: ", *configured_path));
    }
    return std::vector<std::string>{std::string(*configured_path)};
  }

  std::vector<std::string> candidates;
  if (!plugin_path.empty()) {
    candidates.push_back(JoinPath(Dirname(plugin_path), kMusaMuBlasShimSoname));
  }
  candidates.emplace_back(kMusaMuBlasShimSoname);
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
      GetMusaMuBlasShimCandidatesImpl(configured_path, plugin_path);
  if (!candidates.ok()) {
    return std::make_unique<ErrorMusaSymbolLoader>(candidates.status());
  }
  return internal::CreateMusaDsoLoader(
      *std::move(candidates),
      /*fail_if_not_found=*/configured_path.has_value());
}

std::unique_ptr<internal::MusaSymbolLoader> CreateDefaultShimLoader() {
  std::optional<absl::string_view> configured_path;
  if (const char* configured = std::getenv(kMusaMuBlasShimEnvironment);
      configured != nullptr) {
    configured_path = configured;
  }

  std::string plugin_path;
  Dl_info info = {};
  if (dladdr(reinterpret_cast<void*>(&GetMusaMuBlasApi), &info) != 0 &&
      info.dli_fname != nullptr && info.dli_fname[0] != '\0') {
    plugin_path = info.dli_fname;
  }
  return CreateShimLoader(configured_path, plugin_path);
}

absl::Status ValidateApiV1(const XlaMusaMuBlasApiV1* api,
                           absl::string_view loaded_path) {
  if (api == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loaded_path, " returned a null v1 API table"));
  }
  if (api->struct_size < XLA_MUSA_MUBLAS_API_V1_MIN_STRUCT_SIZE) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loaded_path,
        " v1 table is too small: ", api->struct_size, "; expected at least ",
        XLA_MUSA_MUBLAS_API_V1_MIN_STRUCT_SIZE));
  }
  if (api->abi_version != XLA_MUSA_MUBLAS_ABI_VERSION_1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loaded_path, " has ABI version ", api->abi_version,
        "; expected ", XLA_MUSA_MUBLAS_ABI_VERSION_1));
  }
  if ((api->capabilities & XLA_MUSA_MUBLAS_CAPABILITIES_V1) !=
      XLA_MUSA_MUBLAS_CAPABILITIES_V1) {
    return absl::FailedPreconditionError(
        absl::StrCat("muBLAS shim ", loaded_path,
                     " ABI v1 capabilities are incomplete: ", api->capabilities,
                     "; required mask ", XLA_MUSA_MUBLAS_CAPABILITIES_V1));
  }
  if (api->create == nullptr || api->destroy == nullptr ||
      api->set_stream == nullptr || api->get_version == nullptr ||
      api->gemm == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("muBLAS shim ", loaded_path,
                     " has a null required function in its v1 API table"));
  }
  return absl::OkStatus();
}

absl::Status ValidateApiV2(const XlaMusaMuBlasApiV2* api,
                           absl::string_view loaded_path) {
  if (api == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loaded_path, " returned a null v2 API table"));
  }
  if (api->struct_size < XLA_MUSA_MUBLAS_API_V2_MIN_STRUCT_SIZE) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loaded_path,
        " v2 table is too small: ", api->struct_size, "; expected at least ",
        XLA_MUSA_MUBLAS_API_V2_MIN_STRUCT_SIZE));
  }
  if (api->abi_version != XLA_MUSA_MUBLAS_ABI_VERSION_2) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loaded_path, " has ABI version ", api->abi_version,
        "; expected ", XLA_MUSA_MUBLAS_ABI_VERSION_2));
  }
  if ((api->capabilities & XLA_MUSA_MUBLAS_CAPABILITIES_V1) !=
      XLA_MUSA_MUBLAS_CAPABILITIES_V1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loaded_path,
        " ABI v2 base capabilities are incomplete: ", api->capabilities,
        "; required mask ", XLA_MUSA_MUBLAS_CAPABILITIES_V1));
  }
  if ((api->advanced_capabilities & XLA_MUSA_MUBLAS_ADVANCED_CAPABILITIES_V2) !=
      XLA_MUSA_MUBLAS_ADVANCED_CAPABILITIES_V2) {
    return absl::FailedPreconditionError(
        absl::StrCat("muBLAS shim ", loaded_path,
                     " ABI v2 advanced capabilities are incomplete: ",
                     api->advanced_capabilities, "; required mask ",
                     XLA_MUSA_MUBLAS_ADVANCED_CAPABILITIES_V2));
  }
  if (api->create == nullptr || api->destroy == nullptr ||
      api->set_stream == nullptr || api->get_version == nullptr ||
      api->gemm == nullptr || api->set_atomics_mode == nullptr ||
      api->gemm_with_algorithm == nullptr || api->gemm_batched == nullptr ||
      api->gemm_strided_batched == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("muBLAS shim ", loaded_path,
                     " has a null required function in its v2 API table"));
  }
  return absl::OkStatus();
}

bool HasCapability(uint64_t capabilities, uint64_t capability) {
  return (capabilities & capability) == capability;
}

absl::Status ValidateNormalizedAlgorithm(
    XlaMusaMuBlasAlgorithm algorithm, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type) {
  if (algorithm != XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT &&
      algorithm != XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP) {
    return absl::InvalidArgumentError(
        absl::StrCat("unknown normalized muBLAS algorithm ", algorithm));
  }
  if (algorithm == XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP &&
      (input_type != XLA_MUSA_MUBLAS_DATA_TYPE_F32 ||
       output_type != XLA_MUSA_MUBLAS_DATA_TYPE_F32 ||
       compute_type != XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32)) {
    return absl::UnimplementedError(
        "muBLAS tensor-op algorithm is qualified only for homogeneous F32");
  }
  return absl::OkStatus();
}

absl::Status ValidateNormalizedScalType(XlaMusaMuBlasScalType scal_type) {
  switch (scal_type) {
    case XLA_MUSA_MUBLAS_SCAL_TYPE_F32:
    case XLA_MUSA_MUBLAS_SCAL_TYPE_F64:
    case XLA_MUSA_MUBLAS_SCAL_TYPE_C64:
    case XLA_MUSA_MUBLAS_SCAL_TYPE_C128:
    case XLA_MUSA_MUBLAS_SCAL_TYPE_C64_F32:
    case XLA_MUSA_MUBLAS_SCAL_TYPE_C128_F64:
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("unknown normalized muBLAS SCAL type ", scal_type));
  }
}

}  // namespace

namespace internal {

absl::StatusOr<std::vector<std::string>> GetMusaMuBlasShimCandidates(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  return GetMusaMuBlasShimCandidatesImpl(configured_path, plugin_path);
}

std::unique_ptr<MusaSymbolLoader> CreateMusaMuBlasShimLoader(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  return CreateShimLoader(configured_path, plugin_path);
}

}  // namespace internal

MusaMuBlasApi::MusaMuBlasApi(std::unique_ptr<internal::MusaSymbolLoader> loader)
    : loader_(std::move(loader)) {}

MusaMuBlasApi::~MusaMuBlasApi() = default;

std::unique_ptr<MusaMuBlasApi> MusaMuBlasApi::CreateForTesting(
    std::unique_ptr<internal::MusaSymbolLoader> loader) {
  return std::make_unique<MusaMuBlasApi>(std::move(loader));
}

absl::Status MusaMuBlasApi::Init() const {
  absl::call_once(load_once_, [this] { load_status_ = Initialize(); });
  return load_status_;
}

absl::Status MusaMuBlasApi::Initialize() const {
  if (loader_ == nullptr) {
    return absl::InternalError("muBLAS shim symbol loader is null");
  }
  RETURN_IF_ERROR(loader_->Load());
  absl::StatusOr<void*> v2_getter = loader_->Resolve(kMuBlasV2GetterSymbol);
  if (v2_getter.ok() && *v2_getter != nullptr) {
    GetApiV2Fn get_api = reinterpret_cast<GetApiV2Fn>(*v2_getter);
    const XlaMusaMuBlasApiV2* api = get_api();
    RETURN_IF_ERROR(ValidateApiV2(api, loader_->loaded_path()));
    api_v2_ = api;
    return absl::OkStatus();
  }
  if (!v2_getter.ok() && !absl::IsNotFound(v2_getter.status())) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loader_->loaded_path(), " could not resolve ",
        kMuBlasV2GetterSymbol, ": ", v2_getter.status()));
  }

  absl::StatusOr<void*> v1_getter = loader_->Resolve(kMuBlasV1GetterSymbol);
  if (!v1_getter.ok() || *v1_getter == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loader_->loaded_path(), " exports neither a usable ",
        kMuBlasV2GetterSymbol, " nor ", kMuBlasV1GetterSymbol,
        v1_getter.ok() ? "" : absl::StrCat(": ", v1_getter.status())));
  }
  GetApiV1Fn get_api = reinterpret_cast<GetApiV1Fn>(*v1_getter);
  const XlaMusaMuBlasApiV1* api = get_api();
  RETURN_IF_ERROR(ValidateApiV1(api, loader_->loaded_path()));
  api_v1_ = api;
  return absl::OkStatus();
}

uint64_t MusaMuBlasApi::capabilities() const {
  if (!Init().ok()) return 0;
  return api_v2_ != nullptr ? api_v2_->capabilities : api_v1_->capabilities;
}

uint64_t MusaMuBlasApi::advanced_capabilities() const {
  return Init().ok() && api_v2_ != nullptr ? api_v2_->advanced_capabilities : 0;
}

uint32_t MusaMuBlasApi::abi_version() const {
  if (!Init().ok()) return 0;
  return api_v2_ != nullptr ? XLA_MUSA_MUBLAS_ABI_VERSION_2
                            : XLA_MUSA_MUBLAS_ABI_VERSION_1;
}

bool MusaMuBlasApi::SupportsSetAtomicsMode() const {
  return HasCapability(advanced_capabilities(),
                       XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_SET_ATOMICS_MODE);
}

bool MusaMuBlasApi::SupportsGemmWithAlgorithm() const {
  return HasCapability(advanced_capabilities(),
                       XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_GEMM_WITH_ALGORITHM);
}

bool MusaMuBlasApi::SupportsGemmBatched() const {
  return HasCapability(advanced_capabilities(),
                       XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_GEMM_BATCHED);
}

bool MusaMuBlasApi::SupportsGemmStridedBatched() const {
  return HasCapability(
      advanced_capabilities(),
      XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_GEMM_STRIDED_BATCHED);
}

bool MusaMuBlasApi::SupportsTensorOpF32() const {
  return HasCapability(advanced_capabilities(),
                       XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_TENSOR_OP_F32);
}

bool MusaMuBlasApi::UsesZeroExternalWorkspace() const {
  return HasCapability(
      advanced_capabilities(),
      XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_ZERO_EXTERNAL_WORKSPACE);
}

bool MusaMuBlasApi::SupportsScal() const {
  if (!Init().ok() || api_v2_ == nullptr ||
      api_v2_->struct_size < XLA_MUSA_MUBLAS_API_V2_SCAL_STRUCT_SIZE) {
    return false;
  }
  return HasCapability(api_v2_->advanced_capabilities,
                       XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_SCAL) &&
         api_v2_->scal != nullptr;
}

std::string MusaMuBlasApi::advanced_abi_fingerprint() const {
  if (abi_version() != XLA_MUSA_MUBLAS_ABI_VERSION_2) return {};
  return kMusaMuBlasAdvancedAbiFingerprintV2;
}

std::string MusaMuBlasApi::scal_abi_fingerprint() const {
  return SupportsScal() ? kMusaMuBlasScalAbiFingerprintV1 : std::string();
}

absl::string_view MusaMuBlasApi::loaded_path() const {
  return loader_ == nullptr ? absl::string_view() : loader_->loaded_path();
}

absl::Status MusaMuBlasApi::Create(void** handle) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::InvalidArgumentError("muBLAS handle output is null");
  }
  *handle = nullptr;
  const XlaMusaMuBlasCreateFn create =
      api_v2_ != nullptr ? api_v2_->create : api_v1_->create;
  RETURN_IF_ERROR(MuBlasStatus(create(handle), "create"));
  if (*handle == nullptr) {
    return absl::InternalError("muBLAS create returned a null handle");
  }
  return absl::OkStatus();
}

absl::Status MusaMuBlasApi::Destroy(void* handle) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) return absl::OkStatus();
  const XlaMusaMuBlasDestroyFn destroy =
      api_v2_ != nullptr ? api_v2_->destroy : api_v1_->destroy;
  return MuBlasStatus(destroy(handle), "destroy");
}

absl::Status MusaMuBlasApi::SetStream(void* handle, void* stream) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muBLAS handle is null");
  }
  const XlaMusaMuBlasSetStreamFn set_stream =
      api_v2_ != nullptr ? api_v2_->set_stream : api_v1_->set_stream;
  return MuBlasStatus(set_stream(handle, stream), "set_stream");
}

absl::Status MusaMuBlasApi::GetVersion(void* handle, int32_t* version) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr || version == nullptr) {
    return absl::InvalidArgumentError(
        "muBLAS get_version requires non-null handle and output");
  }
  const XlaMusaMuBlasGetVersionFn get_version =
      api_v2_ != nullptr ? api_v2_->get_version : api_v1_->get_version;
  return MuBlasStatus(get_version(handle, version), "get_version");
}

absl::Status MusaMuBlasApi::SetAtomicsMode(void* handle,
                                           bool allow_atomics) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muBLAS handle is null");
  }
  if (!SupportsSetAtomicsMode()) {
    return absl::UnimplementedError(
        "muBLAS shim does not support the v2 atomics-mode contract");
  }
  return MuBlasStatus(api_v2_->set_atomics_mode(
                          handle, allow_atomics ? UINT32_C(1) : UINT32_C(0)),
                      "set_atomics_mode");
}

absl::Status MusaMuBlasApi::Scal(void* handle, XlaMusaMuBlasScalType scal_type,
                                 int64_t n, const void* alpha, void* x,
                                 int64_t incx) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muBLAS handle is null");
  }
  if (alpha == nullptr || x == nullptr) {
    return absl::InvalidArgumentError(
        "muBLAS SCAL requires non-null scalar and vector pointers");
  }
  RETURN_IF_ERROR(ValidateNormalizedScalType(scal_type));
  if (!SupportsScal()) {
    return absl::UnimplementedError(
        "muBLAS shim does not support the optional V2 SCAL contract");
  }
  return MuBlasStatus(api_v2_->scal(handle, scal_type, n, alpha, x, incx),
                      "scal");
}

absl::Status MusaMuBlasApi::Gemm(
    void* handle, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
    XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* a, int64_t lda,
    const void* b, int64_t ldb, const void* beta, void* c, int64_t ldc) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muBLAS handle is null");
  }
  const XlaMusaMuBlasGemmFn gemm =
      api_v2_ != nullptr ? api_v2_->gemm : api_v1_->gemm;
  return MuBlasStatus(
      gemm(handle, static_cast<uint32_t>(input_type),
           static_cast<uint32_t>(output_type),
           static_cast<uint32_t>(compute_type), static_cast<uint32_t>(trans_a),
           static_cast<uint32_t>(trans_b), m, n, k, alpha, a, lda, b, ldb, beta,
           c, ldc),
      "gemm");
}

absl::Status MusaMuBlasApi::GemmWithAlgorithm(
    void* handle, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
    XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* a, int64_t lda,
    const void* b, int64_t ldb, const void* beta, void* c, int64_t ldc,
    XlaMusaMuBlasAlgorithm algorithm) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muBLAS handle is null");
  }
  RETURN_IF_ERROR(ValidateNormalizedAlgorithm(algorithm, input_type,
                                              output_type, compute_type));
  if (!SupportsGemmWithAlgorithm()) {
    if (algorithm == XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT) {
      return Gemm(handle, input_type, output_type, compute_type, trans_a,
                  trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    return absl::UnimplementedError(
        "muBLAS ABI v1 supports only the default GEMM algorithm");
  }
  return MuBlasStatus(
      api_v2_->gemm_with_algorithm(
          handle, input_type, output_type, compute_type, trans_a, trans_b, m, n,
          k, alpha, a, lda, b, ldb, beta, c, ldc, algorithm),
      "gemm_with_algorithm");
}

absl::Status MusaMuBlasApi::GemmBatched(
    void* handle, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
    XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* const* a, int64_t lda,
    const void* const* b, int64_t ldb, const void* beta, void* const* c,
    int64_t ldc, int64_t batch_count, XlaMusaMuBlasAlgorithm algorithm) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muBLAS handle is null");
  }
  RETURN_IF_ERROR(ValidateNormalizedAlgorithm(algorithm, input_type,
                                              output_type, compute_type));
  if (!SupportsGemmBatched()) {
    return absl::UnimplementedError(
        "muBLAS shim does not support pointer-array batched GEMM");
  }
  return MuBlasStatus(
      api_v2_->gemm_batched(handle, input_type, output_type, compute_type,
                            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, batch_count, algorithm),
      "gemm_batched");
}

absl::Status MusaMuBlasApi::GemmStridedBatched(
    void* handle, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
    XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* a, int64_t lda,
    int64_t stride_a, const void* b, int64_t ldb, int64_t stride_b,
    const void* beta, void* c, int64_t ldc, int64_t stride_c,
    int64_t batch_count, XlaMusaMuBlasAlgorithm algorithm) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muBLAS handle is null");
  }
  RETURN_IF_ERROR(ValidateNormalizedAlgorithm(algorithm, input_type,
                                              output_type, compute_type));
  if (!SupportsGemmStridedBatched()) {
    return absl::UnimplementedError(
        "muBLAS shim does not support strided-batched GEMM");
  }
  return MuBlasStatus(
      api_v2_->gemm_strided_batched(
          handle, input_type, output_type, compute_type, trans_a, trans_b, m, n,
          k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
          batch_count, algorithm),
      "gemm_strided_batched");
}

MusaMuBlasApi* GetMusaMuBlasApi() {
  static auto* api = new MusaMuBlasApi(CreateDefaultShimLoader());
  return api;
}

absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaMuBlasOptionalLibraryAbis() {
  MusaMuBlasApi* api = GetMusaMuBlasApi();
  absl::Status status = api->Init();
  if (absl::IsNotFound(status)) {
    return std::vector<MusaOptionalLibraryAbi>{};
  }
  RETURN_IF_ERROR(status);
  // The public optional-library ABI remains v1. Advanced executables opt into
  // the nonempty v2 contract fingerprint; basic v1 executables require no
  // fingerprint and remain compatible with either table.
  std::vector<MusaOptionalLibraryAbi> libraries = {
      {kMusaMuBlasLibraryAbiName, kMusaMuBlasLibraryAbiVersion,
       api->advanced_abi_fingerprint()}};
  if (api->SupportsScal()) {
    libraries.push_back({kMusaMuBlasScalLibraryAbiName,
                         kMusaMuBlasScalLibraryAbiVersion,
                         api->scal_abi_fingerprint()});
  }
  return libraries;
}

}  // namespace stream_executor::musa
