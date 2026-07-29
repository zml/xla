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
// Canonical readable identity for the normalized v2 contract and its
// lower-case SHA-256. Executable ABI metadata stores the hash because the
// shared MUSA envelope requires fingerprints to be exactly 64 hex digits.
inline constexpr char kMusaMuBlasAdvancedAbiContractV2[] =
    "xla-musa-mublas;abi=2;base=7;advanced=63;workspace=0";
inline constexpr char kMusaMuBlasAdvancedAbiFingerprintV2[] =
    "097f516c7b70c49b3873926b6b20e39bafd74a80687a5e9e66a2927433dc1a68";
inline constexpr char kMusaMuBlasScalAbiContractV1[] =
    "xla-musa-mublas-scal;abi=1;routes=sscal,dscal,cscal,zscal,csscal,zdscal";
inline constexpr char kMusaMuBlasScalAbiFingerprintV1[] =
    "aee8bd3fc6ac91980f38d634b83a7789633d2573eb962de231ab5f129ae22560";
inline constexpr char kMusaMuBlasTrsmAbiContractV1[] =
    "xla-musa-mublas-trsm;abi=1;routes=strsm,dtrsm,ctrsm,ztrsm,"
    "strsm-batched,dtrsm-batched,ctrsm-batched,ztrsm-batched;"
    "workspace=internal;stream=bound";
inline constexpr char kMusaMuBlasTrsmAbiFingerprintV1[] =
    "cce7da268bd7096df25f1d1c8a7ef2d1b33b5756df7693d9f3022d188739f8e6";

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
  uint64_t advanced_capabilities() const;
  uint32_t abi_version() const;
  bool SupportsSetAtomicsMode() const;
  bool SupportsGemmWithAlgorithm() const;
  bool SupportsGemmBatched() const;
  bool SupportsGemmStridedBatched() const;
  bool SupportsTensorOpF32() const;
  bool UsesZeroExternalWorkspace() const;
  bool SupportsScal() const;
  bool SupportsTrsm() const;
  bool SupportsTrsmBatched() const;
  // Empty for a v1 shim. For v2 this is a deterministic cache/serialization
  // identity for operations that depend on the advanced ABI contract.
  std::string advanced_abi_fingerprint() const;
  // Empty unless the optional V2 SCAL tail is complete. Kept separate from
  // the C16 fingerprint so old V2 shims and executables remain compatible.
  std::string scal_abi_fingerprint() const;
  // Empty unless both optional V2 TRSM functions and their capability bits
  // are present. Kept separate from the C16 and C17 identities.
  std::string trsm_abi_fingerprint() const;
  absl::string_view loaded_path() const;

  absl::Status Create(void** handle) const;
  absl::Status Destroy(void* handle) const;
  absl::Status SetStream(void* handle, void* stream) const;
  absl::Status GetVersion(void* handle, int32_t* version) const;
  absl::Status SetAtomicsMode(void* handle, bool allow_atomics) const;
  absl::Status Scal(void* handle, XlaMusaMuBlasScalType scal_type, int64_t n,
                    const void* alpha, void* x, int64_t incx) const;
  absl::Status Trsm(void* handle, XlaMusaMuBlasTrsmType trsm_type,
                    XlaMusaMuBlasSide side, XlaMusaMuBlasFill fill,
                    XlaMusaMuBlasOperation trans_a,
                    XlaMusaMuBlasDiagonal diagonal, int64_t m, int64_t n,
                    const void* alpha, const void* a, int64_t lda, void* b,
                    int64_t ldb) const;
  absl::Status TrsmBatched(void* handle, XlaMusaMuBlasTrsmType trsm_type,
                           XlaMusaMuBlasSide side, XlaMusaMuBlasFill fill,
                           XlaMusaMuBlasOperation trans_a,
                           XlaMusaMuBlasDiagonal diagonal, int64_t m, int64_t n,
                           const void* alpha, const void* const* a, int64_t lda,
                           void* const* b, int64_t ldb,
                           int64_t batch_count) const;
  absl::Status Gemm(void* handle, XlaMusaMuBlasDataType input_type,
                    XlaMusaMuBlasDataType output_type,
                    XlaMusaMuBlasComputeType compute_type,
                    XlaMusaMuBlasOperation trans_a,
                    XlaMusaMuBlasOperation trans_b, int64_t m, int64_t n,
                    int64_t k, const void* alpha, const void* a, int64_t lda,
                    const void* b, int64_t ldb, const void* beta, void* c,
                    int64_t ldc) const;
  absl::Status GemmWithAlgorithm(
      void* handle, XlaMusaMuBlasDataType input_type,
      XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
      XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
      int64_t n, int64_t k, const void* alpha, const void* a, int64_t lda,
      const void* b, int64_t ldb, const void* beta, void* c, int64_t ldc,
      XlaMusaMuBlasAlgorithm algorithm) const;
  absl::Status GemmBatched(void* handle, XlaMusaMuBlasDataType input_type,
                           XlaMusaMuBlasDataType output_type,
                           XlaMusaMuBlasComputeType compute_type,
                           XlaMusaMuBlasOperation trans_a,
                           XlaMusaMuBlasOperation trans_b, int64_t m, int64_t n,
                           int64_t k, const void* alpha, const void* const* a,
                           int64_t lda, const void* const* b, int64_t ldb,
                           const void* beta, void* const* c, int64_t ldc,
                           int64_t batch_count,
                           XlaMusaMuBlasAlgorithm algorithm) const;
  absl::Status GemmStridedBatched(
      void* handle, XlaMusaMuBlasDataType input_type,
      XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
      XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
      int64_t n, int64_t k, const void* alpha, const void* a, int64_t lda,
      int64_t stride_a, const void* b, int64_t ldb, int64_t stride_b,
      const void* beta, void* c, int64_t ldc, int64_t stride_c,
      int64_t batch_count, XlaMusaMuBlasAlgorithm algorithm) const;

 private:
  absl::Status Initialize() const;

  std::unique_ptr<internal::MusaSymbolLoader> loader_;
  mutable absl::once_flag load_once_;
  mutable absl::Status load_status_ =
      absl::UnknownError("muBLAS shim not loaded");
  mutable const XlaMusaMuBlasApiV1* api_v1_ = nullptr;
  mutable const XlaMusaMuBlasApiV2* api_v2_ = nullptr;
};

// Returns the process-wide API instance. Its DSO is intentionally pinned by
// the shared MUSA symbol-loader policy.
MusaMuBlasApi* GetMusaMuBlasApi();

// Discovers optional vendor-library ABIs for both Platform runtime snapshots
// and executable-load validation. A missing shim is a supported configuration
// and produces an empty list. A present but malformed shim is a packaging
// error and is returned to the caller.
absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaMuBlasOptionalLibraryAbis();

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_MUBLAS_API_H_
