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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_MUBLAS_API_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_MUBLAS_API_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/musa/mublas_shim/mublas_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace stream_executor::musa {

inline constexpr char kMusaMuBlasShimEnvironment[] =
    "XLA_MUSA_MUBLAS_SHIM_PATH";
inline constexpr char kMusaMuBlasShimSoname[] = "libxla_musa_mublas_shim.so.1";

namespace internal {

// Applies the optional-shim search policy. An explicitly configured path is
// fail-closed: it must be absolute and is the only candidate. Defaults use an
// adjacent DSO first and the versioned SONAME second. Qualified packaging uses
// the adjacent DSO; bare-SONAME discovery is compatibility fallback only.
absl::StatusOr<std::vector<std::string>> GetMusaMuBlasShimCandidates(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path);

// Creates the production shim loader for an explicit or adjacent-discovery
// policy. An explicit path is fail-closed even when the file is absent.
std::unique_ptr<MusaSymbolLoader> CreateMusaMuBlasShimLoader(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path);

}  // namespace internal

// SDK-free, typed interface to the separately linked muBLAS shim. The shim is
// deliberately loaded through one versioned getter, so libpjrt_musa has no
// dynamic dependency on the vendor muBLAS DSO and never resolves vendor C++
// symbols itself.
class MusaMuBlasApi {
 public:
  static std::unique_ptr<MusaMuBlasApi> CreateForTesting(
      std::unique_ptr<internal::MusaSymbolLoader> loader);

  explicit MusaMuBlasApi(std::unique_ptr<internal::MusaSymbolLoader> loader);
  MusaMuBlasApi(const MusaMuBlasApi&) = delete;
  MusaMuBlasApi& operator=(const MusaMuBlasApi&) = delete;
  ~MusaMuBlasApi();

  absl::Status Init() const;
  bool IsLoaded() const { return Init().ok(); }

  uint64_t capabilities() const;
  absl::string_view loaded_path() const;

  absl::Status Create(void** handle) const;
  absl::Status Destroy(void* handle) const;
  absl::Status SetStream(void* handle, void* stream) const;
  absl::Status GetVersion(void* handle, int32_t* version) const;
  absl::Status Gemm(void* handle, XlaMusaMuBlasDataType input_type,
                    XlaMusaMuBlasDataType output_type,
                    XlaMusaMuBlasComputeType compute_type,
                    XlaMusaMuBlasOperation trans_a,
                    XlaMusaMuBlasOperation trans_b, int64_t m, int64_t n,
                    int64_t k, const void* alpha, const void* a, int64_t lda,
                    const void* b, int64_t ldb, const void* beta, void* c,
                    int64_t ldc) const;

 private:
  absl::Status Initialize() const;

  std::unique_ptr<internal::MusaSymbolLoader> loader_;
  mutable absl::once_flag load_once_;
  mutable absl::Status load_status_ =
      absl::UnknownError("muBLAS shim not loaded");
  mutable const XlaMusaMuBlasApiV1* api_ = nullptr;
};

// Returns the process-wide API instance. Its DSO is intentionally pinned by
// the shared MUSA symbol-loader policy.
MusaMuBlasApi* GetMusaMuBlasApi();

// Discovers optional vendor-library ABIs for both Platform runtime snapshots
// and executable-load validation. A missing shim is a supported configuration
// and produces an empty list. A present but malformed shim is a packaging
// error and is returned to the caller.
absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaOptionalLibraryAbis();

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_MUBLAS_API_H_
