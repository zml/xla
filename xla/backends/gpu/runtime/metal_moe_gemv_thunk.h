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

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Grouped MoE GEMV on Metal: one matmul per routed row against that row's
// expert weight matrix (top-k MoE expert projections touch only selected
// experts). Dispatches by weight dtype:
//
//   bf16     -> custom/bf16_moe_gemv.metal        (__metal$moe_gemm)
//   f8e4m3fn -> custom/fp8_moe_gemv.metal         (__metal$moe_gemm$f8)
//               + steel fp8_gather_qmm_rhs (prefill)
//   f4e2m1   -> MLX nvfp4_gather_qmv              (__metal$moe_gemm$f4)
//               (mlx_fp4_qmv.h; same metallib as dense nvfp4_qmv)
//
// The model (Moe.forward on Metal) emits the custom call after top-k routing:
// R = num_tokens * top_k rows, expert id per row. No dot for GemmRewriter.
//
// Operand contracts (positional):
//   bf16:  {x[R,K] bf16, w[E,N,K] bf16, expert_id[R] s32} -> out[R,N] bf16
//   fp8:   {x[R,K] bf16, w[E,N,K] f8e4m3fn, scale[E,N/128,K/128] bf16,
//           expert_id[R] s32} -> out[R,N] bf16
//   nvfp4: {x[R,K] bf16, w[E,N,K] f4e2m1, scale[E,N,K/16] f8e4m3fn,
//           expert_id[R] s32, w_global_scale[E] f32 (optional)}
//           -> out[R,N] bf16
//           (w_global_scale[e] is the compressed-tensors weight-global encode
//            divisor g_ct; both nvfp4 kernels fold its reciprocal into the
//            weight's group scale. Folding into the *weight* keeps the f32
//            accumulator at output magnitude, unlike pre-scaling x -- which
//            costs a full read/write pass over x -- or dividing the output.
//            Absent: no buffer is bound and the kernels are unchanged.)
//
// Dispatch (MLX GatherQMM-aligned for NVFP4):
//   small R / decode: per-row GEMV (nvfp4_gather_qmv / fp8_moe_gemv /
//                     bf16_moe_gemv)
//   large R prefill:  sort-by-expert + Steel gather
//     - bf16:     R >= 1024
//     - fp8:      always per-row fp8_moe_gemv for now (steel fp8_gather_qmm_rhs
//                 sorted path temporarily disabled until golden-checked)
//     - nvfp4:    R >= 16 && R/E >= 4 (MLX gather_qmm_rhs gate; for E=128 this
//                 is R >= 512 — prefill, not continuous-batch decode)
//     - all sorted paths require 1 <= E <= 256 (moe_argsort bucket capacity);
//       unsupported shapes stay on their per-row GEMV path
class MetalMoeGemvThunk : public Thunk {
 public:
  // `global_scale` is bound only when `has_global_scale` (nvfp4 only).
  MetalMoeGemvThunk(ThunkInfo thunk_info, BufferAllocation::Slice x,
                    Shape x_shape, BufferAllocation::Slice w, Shape w_shape,
                    BufferAllocation::Slice scale, Shape scale_shape,
                    BufferAllocation::Slice expert_id, Shape expert_id_shape,
                    BufferAllocation::Slice out, Shape out_shape,
                    BufferAllocation::Slice workspace, Shape workspace_shape,
                    BufferAllocation::Slice global_scale, Shape
                    global_scale_shape, bool has_global_scale, int64_t r,
                    int64_t k, int64_t n);
  ~MetalMoeGemvThunk() override;

  MetalMoeGemvThunk(const MetalMoeGemvThunk&) = delete;
  MetalMoeGemvThunk& operator=(const MetalMoeGemvThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  struct LoadedState;

  // Lazily, on the first execute for a given executor: compile the embedded
  // metallib (process-cached), load the selected kernels, and prepare immutable
  // inline launch constants. Mutable scratch comes from workspace_. Must hold
  // mu_.
  absl::StatusOr<std::shared_ptr<const LoadedState>> EnsureLoaded(
      stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice x_, w_, scale_, expert_id_, out_, workspace_,
      global_scale_;
  const Shape x_shape_, w_shape_, scale_shape_, expert_id_shape_, out_shape_,
      workspace_shape_, global_scale_shape_;
  const bool has_global_scale_;
  const int64_t r_, k_, n_;

  absl::Mutex mu_;
  // Fully initialized immutable state per executor. Building is transactional;
  // shared ownership keeps an executor's state alive while a concurrent launch
  // encodes it, without A/B executor calls evicting each other's pipelines.
  absl::flat_hash_map<stream_executor::StreamExecutor*,
                      std::shared_ptr<const LoadedState>>
      states_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_MOE_GEMV_THUNK_H_
