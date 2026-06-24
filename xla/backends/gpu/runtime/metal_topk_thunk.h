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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_TOPK_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_TOPK_THUNK_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// GPU bucket/radix-select TopK for the Metal backend (replaces the shared
// single-pass kernel, which scans the whole vocab on ONE threadgroup — ~1/40 of an
// M4 Max for batch==1 decode). Four dispatches over a value-bucket histogram:
//   1. hist   : histogram the TOP-16-bits of each logit's sortable key (65536
//               buckets, global atomics).
//   2. scan   : per row, one threadgroup finds the threshold prefix `pth` (the
//               top-16 of the k-th-largest element) and clears the histogram.
//   3. gather : collect candidates whose top-16 >= pth (a small superset of top-k).
//   4. select : exact top-k of the candidate set on the FULL key (single-thread
//               insertion — the candidate set is tiny). Self-clears the counter.
// Identical result to a full top-k (Descending: value desc, index asc on ties);
// for bf16/f16 the top-16 IS the whole key (exact); for f32 the select on the full
// 32-bit key orders the (slightly larger) candidate set exactly.
//
// Batched: row b = grid.y, per-row buffer offsets. For batch>1 the per-row scan
// and select parallelize across rows. dtype: BF16/F16 -> 16-bit path, F32 -> 32-bit.
class MetalTopKThunk : public Thunk {
 public:
  MetalTopKThunk(ThunkInfo thunk_info, BufferAllocation::Slice data,
                 BufferAllocation::Slice top_vals,
                 BufferAllocation::Slice top_idxs, PrimitiveType dtype,
                 int64_t batch_size, int64_t num_elements, int64_t k);

  MetalTopKThunk(const MetalTopKThunk&) = delete;
  MetalTopKThunk& operator=(const MetalTopKThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

  // Compile-time prewarm (MetalGpuCompiler::RunHloPasses): compile the radix
  // metallib (the dominant first-request compile cost) + create the 4 PSOs into
  // Apple's driver pipeline cache, so the first sample's Ensure is a cache hit.
  // dtype selects the 16/32-bit kernels; k selects the templated select kernel.
  // Best-effort — failures are swallowed (the thunk rebuilds at execute).
  static void Prewarm(stream_executor::StreamExecutor* executor,
                      PrimitiveType dtype, int64_t k);

 private:
  // Lazily compile the metallib + build the 4 pipelines for this dtype/k and
  // allocate the per-row scratch (histograms, threshold, candidate lists). Must
  // hold mu_.
  absl::Status Ensure(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice data_, top_vals_, top_idxs_;
  const PrimitiveType dtype_;
  const int64_t batch_;
  const int64_t n_;
  const int64_t k_;
  const int64_t k_rounded_;  // bit_ceil(k) in {1,2,4,8,16,32} -> select template

  // Candidate-list cap per row. Real logits put only a handful of elements in the
  // threshold's top-16 bucket; 16384 is far beyond that. (A pathological >16384-way
  // top-16 tie would clamp the gather — never happens for model logits.)
  static constexpr int64_t kCap = 16384;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  std::unique_ptr<stream_executor::Kernel> hist_pso_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> scan_pso_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> gather_pso_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> select_pso_ ABSL_GUARDED_BY(mu_);
  // Per-row scratch (allocated once per executor; hist + ccount self-clear after
  // each token via the scan / select, so they're zeroed only at setup).
  stream_executor::DeviceAddressBase hist_ ABSL_GUARDED_BY(mu_);     // B*65536 u32
  stream_executor::DeviceAddressBase thresh_ ABSL_GUARDED_BY(mu_);   // B u32
  stream_executor::DeviceAddressBase ccount_ ABSL_GUARDED_BY(mu_);   // B u32
  stream_executor::DeviceAddressBase cok_ ABSL_GUARDED_BY(mu_);      // B*kCap u32
  stream_executor::DeviceAddressBase cix_ ABSL_GUARDED_BY(mu_);      // B*kCap u32
  stream_executor::DeviceAddressBase args_ ABSL_GUARDED_BY(mu_);     // {n,k,cap}
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_TOPK_THUNK_H_
