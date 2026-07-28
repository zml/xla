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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_MUFFT_API_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_MUFFT_API_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/musa/mufft_shim/mufft_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace stream_executor::musa {

inline constexpr char kMusaMuFftShimEnvironment[] = "XLA_MUSA_MUFFT_SHIM_PATH";
inline constexpr char kMusaMuFftShimSoname[] = "libxla_musa_mufft_shim.so.1";
inline constexpr char kMusaMuFftLibraryAbiName[] = "mufft";
inline constexpr char kMusaMuFftLibraryAbiVersion[] = "1";

// Canonical readable identity for the normalized v1 contract and its
// lower-case SHA-256. The qualified contract deliberately records the vendor
// version and raw inverse semantics because both affect serialized
// executable compatibility and numerical behavior.
inline constexpr char kMusaMuFftAbiContractV1[] =
    "xla-musa-mufft;abi=1;capabilities=511;rank=1-3;layout=i32;"
    "workspace=external;stream=bound;inverse=unnormalized;mufft=1.6.0";
inline constexpr char kMusaMuFftAbiFingerprintV1[] =
    "51a66971dfa8bc0259ac291a5c7f3d1674822f453ed3bf405aa63a28bdc6df27";

namespace internal {

// Applies the optional-shim search policy. An explicitly configured path is
// fail-closed: it must be absolute and is the only candidate. Defaults use an
// adjacent DSO first and the versioned SONAME second. Qualified packaging uses
// the adjacent DSO; bare-SONAME discovery is compatibility fallback only.
absl::StatusOr<std::vector<std::string>> GetMusaMuFftShimCandidates(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path);

// Returns the dependency-aware absolute preload order for a canonical
// adjacent shim path. libmusart is first; device 1/2/3 precede aggregate
// device 0.
absl::StatusOr<std::vector<std::string>> GetMusaMuFftAdjacentClosurePaths(
    absl::string_view absolute_shim_path);

// Creates the production shim loader for an explicit or adjacent-discovery
// policy. An explicit path is fail-closed even when the file is absent.
std::unique_ptr<MusaSymbolLoader> CreateMusaMuFftShimLoader(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path);

}  // namespace internal

// SDK-free, typed interface to the separately linked muFFT shim. The shim is
// deliberately loaded through one versioned getter, so libpjrt_musa has no
// dynamic dependency on the vendor muFFT DSO and never resolves vendor
// symbols itself.
class MusaMuFftApi {
 public:
  static std::unique_ptr<MusaMuFftApi> CreateForTesting(
      std::unique_ptr<internal::MusaSymbolLoader> loader);

  explicit MusaMuFftApi(std::unique_ptr<internal::MusaSymbolLoader> loader);
  MusaMuFftApi(const MusaMuFftApi&) = delete;
  MusaMuFftApi& operator=(const MusaMuFftApi&) = delete;
  ~MusaMuFftApi();

  absl::Status Init() const;
  bool IsLoaded() const { return Init().ok(); }

  uint64_t capabilities() const;
  uint32_t abi_version() const;
  absl::string_view loaded_path() const;

  absl::Status GetVersion(XlaMusaMuFftVersion* version) const;
  absl::Status Create(void** plan) const;
  absl::Status Destroy(void* plan) const;
  absl::Status MakePlanMany(void* plan, int32_t rank,
                            const uint64_t* element_count,
                            const uint64_t* input_embed, uint64_t input_stride,
                            uint64_t input_distance,
                            const uint64_t* output_embed,
                            uint64_t output_stride, uint64_t output_distance,
                            XlaMusaMuFftType type, uint64_t batch_count,
                            uint64_t* workspace_size_bytes) const;
  absl::Status SetWorkArea(void* plan, void* workspace) const;
  absl::Status SetStream(void* plan, void* stream) const;
  absl::Status ExecC2C(void* plan, void* input, void* output,
                       XlaMusaMuFftDirection direction) const;
  absl::Status ExecR2C(void* plan, void* input, void* output) const;
  absl::Status ExecC2R(void* plan, void* input, void* output) const;
  absl::Status ExecZ2Z(void* plan, void* input, void* output,
                       XlaMusaMuFftDirection direction) const;
  absl::Status ExecD2Z(void* plan, void* input, void* output) const;
  absl::Status ExecZ2D(void* plan, void* input, void* output) const;

 private:
  absl::Status Initialize() const;

  std::unique_ptr<internal::MusaSymbolLoader> loader_;
  mutable absl::once_flag load_once_;
  mutable absl::Status load_status_ =
      absl::UnknownError("muFFT shim not loaded");
  mutable const XlaMusaMuFftApiV1* api_ = nullptr;
  mutable XlaMusaMuFftVersion version_ = {};
};

// Returns the process-wide API instance. Its DSO is intentionally pinned by
// the shared MUSA symbol-loader policy.
MusaMuFftApi* GetMusaMuFftApi();

// Discovers the optional muFFT ABI for Platform runtime snapshots and
// executable-load validation. A missing shim is a supported configuration and
// produces an empty list. A present but malformed or unqualified shim is a
// packaging error and is returned to the caller.
absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaMuFftOptionalLibraryAbis();

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_MUFFT_API_H_
