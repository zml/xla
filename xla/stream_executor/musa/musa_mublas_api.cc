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

constexpr char kMuBlasGetterSymbol[] = "xla_musa_mublas_get_api_v1";
constexpr size_t kMaxConfiguredPathBytes = 4096;

using GetApiV1Fn = const XlaMusaMuBlasApiV1* (*)();

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

absl::Status ValidateApi(const XlaMusaMuBlasApiV1* api,
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
  absl::StatusOr<void*> getter = loader_->Resolve(kMuBlasGetterSymbol);
  if (!getter.ok() || *getter == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "muBLAS shim ", loader_->loaded_path(),
        " is missing its only required export ", kMuBlasGetterSymbol,
        getter.ok() ? "" : absl::StrCat(": ", getter.status())));
  }
  GetApiV1Fn get_api = reinterpret_cast<GetApiV1Fn>(*getter);
  const XlaMusaMuBlasApiV1* api = get_api();
  RETURN_IF_ERROR(ValidateApi(api, loader_->loaded_path()));
  api_ = api;
  return absl::OkStatus();
}

uint64_t MusaMuBlasApi::capabilities() const {
  return Init().ok() ? api_->capabilities : 0;
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
  RETURN_IF_ERROR(MuBlasStatus(api_->create(handle), "create"));
  if (*handle == nullptr) {
    return absl::InternalError("muBLAS create returned a null handle");
  }
  return absl::OkStatus();
}

absl::Status MusaMuBlasApi::Destroy(void* handle) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) return absl::OkStatus();
  return MuBlasStatus(api_->destroy(handle), "destroy");
}

absl::Status MusaMuBlasApi::SetStream(void* handle, void* stream) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muBLAS handle is null");
  }
  return MuBlasStatus(api_->set_stream(handle, stream), "set_stream");
}

absl::Status MusaMuBlasApi::GetVersion(void* handle, int32_t* version) const {
  RETURN_IF_ERROR(Init());
  if (handle == nullptr || version == nullptr) {
    return absl::InvalidArgumentError(
        "muBLAS get_version requires non-null handle and output");
  }
  return MuBlasStatus(api_->get_version(handle, version), "get_version");
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
  return MuBlasStatus(
      api_->gemm(handle, static_cast<uint32_t>(input_type),
                 static_cast<uint32_t>(output_type),
                 static_cast<uint32_t>(compute_type),
                 static_cast<uint32_t>(trans_a), static_cast<uint32_t>(trans_b),
                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
      "gemm");
}

MusaMuBlasApi* GetMusaMuBlasApi() {
  static auto* api = new MusaMuBlasApi(CreateDefaultShimLoader());
  return api;
}

absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaOptionalLibraryAbis() {
  MusaMuBlasApi* api = GetMusaMuBlasApi();
  absl::Status status = api->Init();
  if (absl::IsNotFound(status)) {
    return std::vector<MusaOptionalLibraryAbi>{};
  }
  RETURN_IF_ERROR(status);
  // ABI v1 is the compatibility boundary. The concrete vendor DSO hash is
  // recorded as artifact provenance, but is intentionally not an ABI key.
  return std::vector<MusaOptionalLibraryAbi>{
      {kMusaMuBlasLibraryAbiName, kMusaMuBlasLibraryAbiVersion, ""}};
}

}  // namespace stream_executor::musa
