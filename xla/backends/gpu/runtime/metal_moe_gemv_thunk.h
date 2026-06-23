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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_MOE_GEMV_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_MOE_GEMV_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// The mixture-of-experts arm of the Metal block-scaled FP8 GEMM family: a
// grouped fused FP8 (block-scaled) GEMV (`custom/fp8_moe_gemv.metal`). Where
// MetalFp8GemvThunk runs one dense matmul, this runs one matmul per routed row
// against that row's expert weight matrix, so top-k MoE expert projections need
// only the selected experts' weights touched (not all E of them).
//
// The model (Moe.forward on Metal) emits this via the __metal$moe_gemm$f8
// custom call after top-k routing: it forms R = num_tokens * top_k rows (token
// t routed to its k experts contributes k rows) and an expert id per row.
// There is no dot for GemmRewriter to match, so unlike __metal$gemm$f8 this
// target is model-emitted only.
//
// One threadgroup (256 threads) per (n, r) output element reduces over K, with
// the weight/scale base offset by expert_id[r].
//
// Operand contract (positional):
//   0 x         bf16     [R, K]               (row-major; K contraction dim)
//   1 w         f8e4m3fn [E, N, K]            (row-major; expert-major)
//   2 scale     bf16     [E, N/128, K/128]
//   3 expert_id s32      [R]                  (expert each output row uses)
//   -> 0 out    bf16     [R, N]
class MetalMoeGemvThunk : public Thunk {
 public:
  MetalMoeGemvThunk(ThunkInfo thunk_info, BufferAllocation::Slice x,
                    Shape x_shape, BufferAllocation::Slice w, Shape w_shape,
                    BufferAllocation::Slice scale, Shape scale_shape,
                    BufferAllocation::Slice expert_id, Shape expert_id_shape,
                    BufferAllocation::Slice out, Shape out_shape, int64_t r,
                    int64_t k, int64_t n);

  MetalMoeGemvThunk(const MetalMoeGemvThunk&) = delete;
  MetalMoeGemvThunk& operator=(const MetalMoeGemvThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  // Lazily, on the first execute for a given executor: compile the embedded
  // metallib (process-cached), load the kernel, and stage the packed
  // (R, K, N, K/128) dims into a small device buffer. Must hold mu_.
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice x_, w_, scale_, expert_id_, out_;
  const Shape x_shape_, w_shape_, scale_shape_, expert_id_shape_, out_shape_;
  const int64_t r_, k_, n_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  // R >= kSortedMinR (prefill: many rows per expert) takes the sorted gather
  // path; below it (decode: ~0.5 rows/expert) takes the per-row GEMV.
  static constexpr int64_t kSortedMinR = 1024;

  // Decode: per-row x-caching GEMV (kernel_). Prefill: counting-sort the rows by
  // expert (kernel_argsort_), gather them into that order (kernel_gather_), run
  // the MLX Steel gather q-GEMM (kernel_steel_, which reuses each expert's
  // weight across its now-contiguous run of rows), then scatter the result back
  // to the original order (kernel_scatter_).
  bool sorted_path_ ABSL_GUARDED_BY(mu_) = false;
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_steel_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_argsort_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_gather_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_scatter_ ABSL_GUARDED_BY(mu_);

  // Packed {R, K, N, K/128} int4, bound as fp8_moe_gemv's `constant int4& dims`
  // (buffer 5); p_dims_steel_ is the {R, N, K} int3 fp8_gather_qmm_rhs wants.
  stream_executor::DeviceAddressBase p_dims_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_dims_steel_ ABSL_GUARDED_BY(mu_);
  // Sorted-prefill scratch (allocated only when sorted_path_): the permutation
  // (order), the grouped expert ids, the expert-sorted x and the GEMM output;
  // plus the int2 {R,*} dims the argsort / gather / scatter kernels read.
  stream_executor::DeviceAddressBase p_order_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_idx_sorted_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_x_sorted_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_out_sorted_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_argsort_dims_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_gx_dims_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_gout_dims_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_MOE_GEMV_THUNK_H_
