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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_MUDNN_API_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_MUDNN_API_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/musa/mudnn_shim/mudnn_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace stream_executor::musa {

inline constexpr char kMusaMuDnnShimEnvironment[] = "XLA_MUSA_MUDNN_SHIM_PATH";
inline constexpr char kMusaMuDnnShimSoname[] = "libxla_musa_mudnn_shim.so.1";
inline constexpr char kMusaMuDnnLibraryAbiName[] = "mudnn";
inline constexpr char kMusaMuDnnLibraryAbiVersion[] = "1";

// Canonical readable identity for the normalized v1 contract and its
// lower-case SHA-256. The vendor version is part of executable compatibility:
// the C++ muDNN ABI is intentionally isolated behind the separately linked
// shim, and only this exact version is qualified.
inline constexpr char kMusaMuDnnAbiContractV1[] =
    "xla-musa-mudnn;abi=1;capabilities=32765;types=f16,f32;abi-rank=3-5;"
    "convolution-rank=4-5;layout=contiguous-channels-first;"
    "workspace=callback;stream=bound;tf32=controlled;"
    "algorithms=implicit-gemm,winograd-nonfused,gemm;"
    "convolution=forward,backward-data,backward-filter;mudnn=2.8.0";
inline constexpr char kMusaMuDnnAbiFingerprintV1[] =
    "dfaab657ef752c2f591b8dd38c1310c39c0d6eecf785341f141942fa439a719a";

namespace internal {

// Applies the optional-shim search policy. An explicitly configured path is
// fail-closed: it must be absolute and is the only candidate. Defaults use an
// adjacent DSO first and the versioned SONAME second.
absl::StatusOr<std::vector<std::string>> GetMusaMuDnnShimCandidates(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path);

// Creates the production shim loader for an explicit or adjacent-discovery
// policy. An explicit path is fail-closed even when the file is absent.
std::unique_ptr<MusaSymbolLoader> CreateMusaMuDnnShimLoader(
    std::optional<absl::string_view> configured_path,
    absl::string_view plugin_path);

}  // namespace internal

// SDK-free, typed interface to the separately linked muDNN shim. The shim is
// loaded through one versioned getter, so libpjrt_musa has no dynamic
// dependency on the vendor muDNN DSO and never resolves vendor symbols itself.
class MusaMuDnnApi {
 public:
  static std::unique_ptr<MusaMuDnnApi> CreateForTesting(
      std::unique_ptr<internal::MusaSymbolLoader> loader);

  explicit MusaMuDnnApi(std::unique_ptr<internal::MusaSymbolLoader> loader);
  MusaMuDnnApi(const MusaMuDnnApi&) = delete;
  MusaMuDnnApi& operator=(const MusaMuDnnApi&) = delete;
  ~MusaMuDnnApi();

  absl::Status Init() const;
  bool IsLoaded() const { return Init().ok(); }

  uint64_t capabilities() const;
  uint32_t abi_version() const;
  absl::string_view loaded_path() const;

  absl::Status GetVersion(XlaMusaMuDnnVersion* version) const;
  absl::Status CreateHandle(int32_t device_ordinal, void** handle) const;
  absl::Status DestroyHandle(void* handle) const;
  absl::Status SetStream(void* handle, void* stream) const;
  absl::Status SetAllowTf32(void* handle, bool allow_tf32) const;

  absl::Status CreateTensor(void** tensor) const;
  absl::Status DestroyTensor(void* tensor) const;
  absl::Status ConfigureTensor(void* tensor, XlaMusaMuDnnDataType data_type,
                               XlaMusaMuDnnTensorFormat format, int32_t rank,
                               const int64_t* dimensions,
                               const int64_t* strides) const;
  absl::Status SetTensorAddress(void* tensor, void* address) const;

  absl::Status CreateConvolution(void** convolution) const;
  absl::Status DestroyConvolution(void* convolution) const;
  absl::Status ConfigureConvolution(void* convolution, int32_t spatial_rank,
                                    const int64_t* padding,
                                    const int64_t* strides,
                                    const int64_t* dilations,
                                    int64_t group_count) const;
  absl::Status GetRecommendedAlgorithm(void* handle, void* convolution,
                                       XlaMusaMuDnnConvolutionKind kind,
                                       void* output, const void* data,
                                       const void* filter,
                                       XlaMusaMuDnnAlgorithm* algorithm) const;
  absl::Status GetWorkspaceSize(void* handle, void* convolution,
                                XlaMusaMuDnnConvolutionKind kind, void* output,
                                const void* data, const void* filter,
                                XlaMusaMuDnnAlgorithm algorithm,
                                uint64_t* workspace_size_bytes) const;
  absl::Status Convolve(
      void* handle, void* convolution, XlaMusaMuDnnConvolutionKind kind,
      void* output, const void* data, const void* filter,
      XlaMusaMuDnnAlgorithm algorithm,
      const XlaMusaMuDnnWorkspaceAllocator* workspace_allocator) const;

 private:
  absl::Status Initialize() const;

  std::unique_ptr<internal::MusaSymbolLoader> loader_;
  mutable absl::once_flag load_once_;
  mutable absl::Status load_status_ =
      absl::UnknownError("muDNN shim not loaded");
  mutable const XlaMusaMuDnnApiV1* api_ = nullptr;
  mutable XlaMusaMuDnnVersion version_ = {};
};

// Returns the process-wide API instance. Its DSO is intentionally pinned by
// the shared MUSA symbol-loader policy.
MusaMuDnnApi* GetMusaMuDnnApi();

// Discovers the optional muDNN ABI for Platform runtime snapshots and
// executable-load validation. A missing shim is a supported configuration and
// produces an empty list. A present but malformed or unqualified shim is a
// packaging error and is returned to the caller.
absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaMuDnnOptionalLibraryAbis();

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_MUDNN_API_H_
