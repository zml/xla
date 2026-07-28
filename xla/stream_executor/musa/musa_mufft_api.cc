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

#include "xla/stream_executor/musa/musa_mufft_api.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include <limits.h>
#include <sys/stat.h>
#include "xla/stream_executor/musa/mufft_shim/mufft_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace stream_executor::musa {
namespace {

constexpr char kMuFftV1GetterSymbol[] = "xla_musa_mufft_get_api_v1";
constexpr size_t kMaxConfiguredPathBytes = 4096;
constexpr XlaMusaMuFftVersion kQualifiedMuFftVersion = {
    /*major=*/1, /*minor=*/6, /*patch=*/0};

using GetApiV1Fn = const XlaMusaMuFftApiV1* (*)();

absl::Status MuFftStatus(int32_t status, absl::string_view operation) {
  if (status == XLA_MUSA_MUFFT_STATUS_SUCCESS) return absl::OkStatus();
  const std::string message =
      absl::StrCat("muFFT ", operation, " failed with shim status ", status);
  switch (status) {
    case XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT:
      return absl::InvalidArgumentError(message);
    case XLA_MUSA_MUFFT_STATUS_NOT_SUPPORTED:
      return absl::UnimplementedError(message);
    case XLA_MUSA_MUFFT_STATUS_OUT_OF_RANGE:
      return absl::OutOfRangeError(message);
    case XLA_MUSA_MUFFT_STATUS_RESOURCE_EXHAUSTED:
      return absl::ResourceExhaustedError(message);
    case XLA_MUSA_MUFFT_STATUS_FAILED_PRECONDITION:
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

absl::StatusOr<std::vector<std::string>> GetMusaMuFftAdjacentClosurePathsImpl(
    absl::string_view shim_path) {
  if (shim_path.empty() || shim_path.front() != '/') {
    return absl::InvalidArgumentError(
        absl::StrCat("muFFT shim path must be absolute: ", shim_path));
  }
  const std::string directory = Dirname(shim_path);
  return std::vector<std::string>{
      JoinPath(directory, "libmusart.so.1.5"),
      JoinPath(directory, "libmtfft-device-1.so"),
      JoinPath(directory, "libmtfft-device-2.so"),
      JoinPath(directory, "libmtfft-device-3.so"),
      JoinPath(directory, "libmtfft-device-0.so"),
  };
}

absl::StatusOr<std::optional<std::string>> FindExistingCanonicalShimCandidate(
    const std::vector<std::string>& candidates) {
  for (const std::string& candidate : candidates) {
    if (candidate.find('/') == std::string::npos) continue;
    struct stat candidate_stat;
    int result;
    do {
      result = lstat(candidate.c_str(), &candidate_stat);
    } while (result != 0 && errno == EINTR);
    if (result != 0) {
      const int error = errno;
      if (error == ENOENT || error == ENOTDIR) continue;
      return absl::FailedPreconditionError(
          absl::StrCat("Could not inspect muFFT shim candidate ", candidate,
                       ": ", std::strerror(error)));
    }
    char canonical_path[PATH_MAX];
    if (realpath(candidate.c_str(), canonical_path) == nullptr) {
      const int error = errno;
      return absl::FailedPreconditionError(
          absl::StrCat("Could not canonicalize existing muFFT shim candidate ",
                       candidate, ": ", std::strerror(error)));
    }
    return std::optional<std::string>(canonical_path);
  }
  return std::optional<std::string>();
}

absl::Status PreloadMusaDso(absl::string_view path, int flags,
                            absl::string_view shim_path) {
  dlerror();
  void* handle = dlopen(std::string(path).c_str(), flags);
  const char* error = dlerror();
  if (handle == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Could not preload required muFFT sibling ", path, " for shim ",
        shim_path, ": ", error == nullptr ? "unknown dlopen error" : error));
  }
  // Intentionally never call dlclose: vendor runtime objects are process-wide.
  return absl::OkStatus();
}

absl::Status PreloadMusaMuFftAdjacentClosure(absl::string_view shim_path) {
  std::unique_ptr<internal::MusaSymbolLoader> driver =
      internal::CreateMusaDriverDsoLoader();
  absl::Status driver_status = driver->Load();
  if (!driver_status.ok()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muFFT shim ", shim_path,
        " exists, but its libmusart dependency requires an unavailable MUSA "
        "driver: ",
        driver_status));
  }
  absl::StatusOr<std::vector<std::string>> closure =
      GetMusaMuFftAdjacentClosurePathsImpl(shim_path);
  if (!closure.ok()) return closure.status();
  for (size_t index = 0; index < closure->size(); ++index) {
    // Device DSOs contain unresolved MUSA runtime symbols, so libmusart must
    // enter the global lookup scope. Device DSOs themselves remain local.
    const int flags = RTLD_NOW | (index == 0 ? RTLD_GLOBAL : RTLD_LOCAL);
    absl::Status status = PreloadMusaDso((*closure)[index], flags, shim_path);
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> GetMusaMuFftShimCandidatesImpl(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  if (configured_path.has_value()) {
    if (configured_path->empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          kMusaMuFftShimEnvironment, " must not be empty when set"));
    }
    if (configured_path->size() > kMaxConfiguredPathBytes) {
      return absl::InvalidArgumentError(
          absl::StrCat(kMusaMuFftShimEnvironment, " exceeds ",
                       kMaxConfiguredPathBytes, " bytes"));
    }
    if (configured_path->front() != '/') {
      return absl::InvalidArgumentError(
          absl::StrCat(kMusaMuFftShimEnvironment,
                       " must be an absolute path: ", *configured_path));
    }
    return std::vector<std::string>{std::string(*configured_path)};
  }

  std::vector<std::string> candidates;
  if (!plugin_path.empty()) {
    candidates.push_back(JoinPath(Dirname(plugin_path), kMusaMuFftShimSoname));
  }
  candidates.emplace_back(kMusaMuFftShimSoname);
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
      GetMusaMuFftShimCandidatesImpl(configured_path, plugin_path);
  if (!candidates.ok()) {
    return std::make_unique<ErrorMusaSymbolLoader>(candidates.status());
  }
  absl::StatusOr<std::optional<std::string>> existing_shim =
      FindExistingCanonicalShimCandidate(*candidates);
  if (!existing_shim.ok()) {
    return std::make_unique<ErrorMusaSymbolLoader>(existing_shim.status());
  }
  if (existing_shim->has_value()) {
    absl::Status preload_status =
        PreloadMusaMuFftAdjacentClosure(**existing_shim);
    if (!preload_status.ok()) {
      return std::make_unique<ErrorMusaSymbolLoader>(preload_status);
    }
  }
  return internal::CreateMusaDsoLoader(
      *std::move(candidates),
      /*fail_if_not_found=*/configured_path.has_value());
}

std::unique_ptr<internal::MusaSymbolLoader> CreateDefaultShimLoader() {
  std::optional<absl::string_view> configured_path;
  if (const char* configured = std::getenv(kMusaMuFftShimEnvironment);
      configured != nullptr) {
    configured_path = configured;
  }

  std::string plugin_path;
  Dl_info info = {};
  if (dladdr(reinterpret_cast<void*>(&GetMusaMuFftApi), &info) != 0 &&
      info.dli_fname != nullptr && info.dli_fname[0] != '\0') {
    plugin_path = info.dli_fname;
  }
  return CreateShimLoader(configured_path, plugin_path);
}

absl::Status ValidateApi(const XlaMusaMuFftApiV1* api,
                         absl::string_view loaded_path) {
  if (api == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muFFT shim ", loaded_path, " returned a null v1 API table"));
  }
  if (api->struct_size < XLA_MUSA_MUFFT_API_V1_MIN_STRUCT_SIZE) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muFFT shim ", loaded_path,
        " v1 table is too small: ", api->struct_size, "; expected at least ",
        XLA_MUSA_MUFFT_API_V1_MIN_STRUCT_SIZE));
  }
  if (api->abi_version != XLA_MUSA_MUFFT_ABI_VERSION_1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muFFT shim ", loaded_path, " has ABI version ", api->abi_version,
        "; expected ", XLA_MUSA_MUFFT_ABI_VERSION_1));
  }
  if ((api->capabilities & XLA_MUSA_MUFFT_CAPABILITIES_V1) !=
      XLA_MUSA_MUFFT_CAPABILITIES_V1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muFFT shim ", loaded_path, " ABI v1 capabilities are incomplete: ",
        api->capabilities, "; required mask ", XLA_MUSA_MUFFT_CAPABILITIES_V1));
  }
  if (api->get_version == nullptr || api->create == nullptr ||
      api->destroy == nullptr || api->make_plan_many == nullptr ||
      api->set_work_area == nullptr || api->set_stream == nullptr ||
      api->exec_c2c == nullptr || api->exec_r2c == nullptr ||
      api->exec_c2r == nullptr || api->exec_z2z == nullptr ||
      api->exec_d2z == nullptr || api->exec_z2d == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("muFFT shim ", loaded_path,
                     " has a null required function in its v1 API table"));
  }
  return absl::OkStatus();
}

absl::Status ValidateVersion(const XlaMusaMuFftVersion& version,
                             absl::string_view loaded_path) {
  if (version.major != kQualifiedMuFftVersion.major ||
      version.minor != kQualifiedMuFftVersion.minor ||
      version.patch != kQualifiedMuFftVersion.patch) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muFFT shim ", loaded_path, " reports unqualified muFFT version ",
        version.major, ".", version.minor, ".", version.patch, "; expected ",
        kQualifiedMuFftVersion.major, ".", kQualifiedMuFftVersion.minor, ".",
        kQualifiedMuFftVersion.patch));
  }
  return absl::OkStatus();
}

absl::Status ValidateType(XlaMusaMuFftType type) {
  switch (type) {
    case XLA_MUSA_MUFFT_TYPE_C2C:
    case XLA_MUSA_MUFFT_TYPE_R2C:
    case XLA_MUSA_MUFFT_TYPE_C2R:
    case XLA_MUSA_MUFFT_TYPE_Z2Z:
    case XLA_MUSA_MUFFT_TYPE_D2Z:
    case XLA_MUSA_MUFFT_TYPE_Z2D:
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("unknown normalized muFFT type ", type));
  }
}

absl::Status ValidateDirection(XlaMusaMuFftDirection direction) {
  if (direction == XLA_MUSA_MUFFT_DIRECTION_FORWARD ||
      direction == XLA_MUSA_MUFFT_DIRECTION_INVERSE) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("unknown normalized muFFT direction ", direction));
}

absl::Status ValidateExecutionArguments(void* plan, void* input, void* output) {
  if (plan == nullptr) {
    return absl::FailedPreconditionError("muFFT plan is null");
  }
  if (input == nullptr || output == nullptr) {
    return absl::InvalidArgumentError(
        "muFFT execution requires non-null input and output");
  }
  return absl::OkStatus();
}

}  // namespace

namespace internal {

absl::StatusOr<std::vector<std::string>> GetMusaMuFftShimCandidates(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  return GetMusaMuFftShimCandidatesImpl(configured_path, plugin_path);
}

absl::StatusOr<std::vector<std::string>> GetMusaMuFftAdjacentClosurePaths(
    absl::string_view absolute_shim_path) {
  return GetMusaMuFftAdjacentClosurePathsImpl(absolute_shim_path);
}

std::unique_ptr<MusaSymbolLoader> CreateMusaMuFftShimLoader(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path) {
  return CreateShimLoader(configured_path, plugin_path);
}

}  // namespace internal

MusaMuFftApi::MusaMuFftApi(std::unique_ptr<internal::MusaSymbolLoader> loader)
    : loader_(std::move(loader)) {}

MusaMuFftApi::~MusaMuFftApi() = default;

std::unique_ptr<MusaMuFftApi> MusaMuFftApi::CreateForTesting(
    std::unique_ptr<internal::MusaSymbolLoader> loader) {
  return std::make_unique<MusaMuFftApi>(std::move(loader));
}

absl::Status MusaMuFftApi::Init() const {
  absl::call_once(load_once_, [this] { load_status_ = Initialize(); });
  return load_status_;
}

absl::Status MusaMuFftApi::Initialize() const {
  if (loader_ == nullptr) {
    return absl::InternalError("muFFT shim symbol loader is null");
  }
  RETURN_IF_ERROR(loader_->Load());
  absl::StatusOr<void*> getter = loader_->Resolve(kMuFftV1GetterSymbol);
  if (!getter.ok() || *getter == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("muFFT shim ", loader_->loaded_path(),
                     " does not export a usable ", kMuFftV1GetterSymbol,
                     getter.ok() ? "" : absl::StrCat(": ", getter.status())));
  }

  GetApiV1Fn get_api = reinterpret_cast<GetApiV1Fn>(*getter);
  const XlaMusaMuFftApiV1* api = get_api();
  RETURN_IF_ERROR(ValidateApi(api, loader_->loaded_path()));

  XlaMusaMuFftVersion version = {};
  RETURN_IF_ERROR(MuFftStatus(api->get_version(&version), "get_version"));
  RETURN_IF_ERROR(ValidateVersion(version, loader_->loaded_path()));

  api_ = api;
  version_ = version;
  return absl::OkStatus();
}

uint64_t MusaMuFftApi::capabilities() const {
  return Init().ok() ? api_->capabilities : 0;
}

uint32_t MusaMuFftApi::abi_version() const {
  return Init().ok() ? api_->abi_version : 0;
}

absl::string_view MusaMuFftApi::loaded_path() const {
  return loader_ == nullptr ? absl::string_view() : loader_->loaded_path();
}

absl::Status MusaMuFftApi::GetVersion(XlaMusaMuFftVersion* version) const {
  RETURN_IF_ERROR(Init());
  if (version == nullptr) {
    return absl::InvalidArgumentError("muFFT version output is null");
  }
  *version = version_;
  return absl::OkStatus();
}

absl::Status MusaMuFftApi::Create(void** plan) const {
  RETURN_IF_ERROR(Init());
  if (plan == nullptr) {
    return absl::InvalidArgumentError("muFFT plan output is null");
  }
  *plan = nullptr;
  RETURN_IF_ERROR(MuFftStatus(api_->create(plan), "create"));
  if (*plan == nullptr) {
    return absl::InternalError("muFFT create returned a null plan");
  }
  return absl::OkStatus();
}

absl::Status MusaMuFftApi::Destroy(void* plan) const {
  RETURN_IF_ERROR(Init());
  if (plan == nullptr) return absl::OkStatus();
  return MuFftStatus(api_->destroy(plan), "destroy");
}

absl::Status MusaMuFftApi::MakePlanMany(
    void* plan, int32_t rank, const uint64_t* element_count,
    const uint64_t* input_embed, uint64_t input_stride, uint64_t input_distance,
    const uint64_t* output_embed, uint64_t output_stride,
    uint64_t output_distance, XlaMusaMuFftType type, uint64_t batch_count,
    uint64_t* workspace_size_bytes) const {
  RETURN_IF_ERROR(Init());
  if (plan == nullptr) {
    return absl::FailedPreconditionError("muFFT plan is null");
  }
  if (element_count == nullptr || workspace_size_bytes == nullptr) {
    return absl::InvalidArgumentError(
        "muFFT plan requires element counts and a workspace-size output");
  }
  *workspace_size_bytes = 0;
  if (rank <= 0) {
    return absl::InvalidArgumentError("muFFT rank must be positive");
  }
  if (rank > 3) {
    return absl::UnimplementedError("muFFT ABI v1 supports ranks 1 through 3");
  }
  RETURN_IF_ERROR(ValidateType(type));
  return MuFftStatus(
      api_->make_plan_many(plan, rank, element_count, input_embed, input_stride,
                           input_distance, output_embed, output_stride,
                           output_distance, type, batch_count,
                           workspace_size_bytes),
      "make_plan_many");
}

absl::Status MusaMuFftApi::SetWorkArea(void* plan, void* workspace) const {
  RETURN_IF_ERROR(Init());
  if (plan == nullptr) {
    return absl::FailedPreconditionError("muFFT plan is null");
  }
  return MuFftStatus(api_->set_work_area(plan, workspace), "set_work_area");
}

absl::Status MusaMuFftApi::SetStream(void* plan, void* stream) const {
  RETURN_IF_ERROR(Init());
  if (plan == nullptr) {
    return absl::FailedPreconditionError("muFFT plan is null");
  }
  return MuFftStatus(api_->set_stream(plan, stream), "set_stream");
}

absl::Status MusaMuFftApi::ExecC2C(void* plan, void* input, void* output,
                                   XlaMusaMuFftDirection direction) const {
  RETURN_IF_ERROR(Init());
  RETURN_IF_ERROR(ValidateExecutionArguments(plan, input, output));
  RETURN_IF_ERROR(ValidateDirection(direction));
  return MuFftStatus(api_->exec_c2c(plan, input, output, direction),
                     "exec_c2c");
}

absl::Status MusaMuFftApi::ExecR2C(void* plan, void* input,
                                   void* output) const {
  RETURN_IF_ERROR(Init());
  RETURN_IF_ERROR(ValidateExecutionArguments(plan, input, output));
  return MuFftStatus(api_->exec_r2c(plan, input, output), "exec_r2c");
}

absl::Status MusaMuFftApi::ExecC2R(void* plan, void* input,
                                   void* output) const {
  RETURN_IF_ERROR(Init());
  RETURN_IF_ERROR(ValidateExecutionArguments(plan, input, output));
  return MuFftStatus(api_->exec_c2r(plan, input, output), "exec_c2r");
}

absl::Status MusaMuFftApi::ExecZ2Z(void* plan, void* input, void* output,
                                   XlaMusaMuFftDirection direction) const {
  RETURN_IF_ERROR(Init());
  RETURN_IF_ERROR(ValidateExecutionArguments(plan, input, output));
  RETURN_IF_ERROR(ValidateDirection(direction));
  return MuFftStatus(api_->exec_z2z(plan, input, output, direction),
                     "exec_z2z");
}

absl::Status MusaMuFftApi::ExecD2Z(void* plan, void* input,
                                   void* output) const {
  RETURN_IF_ERROR(Init());
  RETURN_IF_ERROR(ValidateExecutionArguments(plan, input, output));
  return MuFftStatus(api_->exec_d2z(plan, input, output), "exec_d2z");
}

absl::Status MusaMuFftApi::ExecZ2D(void* plan, void* input,
                                   void* output) const {
  RETURN_IF_ERROR(Init());
  RETURN_IF_ERROR(ValidateExecutionArguments(plan, input, output));
  return MuFftStatus(api_->exec_z2d(plan, input, output), "exec_z2d");
}

MusaMuFftApi* GetMusaMuFftApi() {
  static auto* api = new MusaMuFftApi(CreateDefaultShimLoader());
  return api;
}

absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaMuFftOptionalLibraryAbis() {
  MusaMuFftApi* api = GetMusaMuFftApi();
  absl::Status status = api->Init();
  if (absl::IsNotFound(status)) {
    return std::vector<MusaOptionalLibraryAbi>{};
  }
  RETURN_IF_ERROR(status);
  return std::vector<MusaOptionalLibraryAbi>{{
      kMusaMuFftLibraryAbiName,
      kMusaMuFftLibraryAbiVersion,
      kMusaMuFftAbiFingerprintV1,
  }};
}

}  // namespace stream_executor::musa
