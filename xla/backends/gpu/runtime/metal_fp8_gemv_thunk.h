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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_FP8_GEMV_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_FP8_GEMV_THUNK_H_

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

// The FP8 arm of the __metal$gemm custom call on the Metal backend: a fused FP8
// (block-scaled) GEMV / thin GEMM (`custom/fp8_gemv.metal`) used for decode.
// GemmRewriter (TryRewriteMetalBlockScaledGemv) reroutes a decode-shaped dot
// over a block-scaled f8e4m3fn weight to a 3-operand {x, w, scale} __metal$gemm,
// which EmitMetalGemmThunk dispatches here by weight element type.
//
// Qwen3.6-FP8 stores the big projections as f8e4m3fn weights with companion
// 128x128 bf16 block scales. In the graph the model dequantizes
// (`w.convert(bf16) * scale`) before `dot`, which for M==1 decode the Metal
// backend lowers to a generic `input_reduce_fusion` running at ~30% of memory
// bandwidth. This kernel instead reads the 1-byte weight directly and
// dequantizes inline, so a decode matmul touches DRAM ~once over the f8 weight.
//
// One threadgroup (256 threads) per (n, b) output element reduces over K.
//
// Operand contract (positional):
//   0 x      bf16     [B, K]          (row-major; K is the contraction dim)
//   1 w      f8e4m3fn [N, K]          (row-major)
//   2 scale  bf16     [N/128, K/128]
//   -> 0 out bf16     [B, N]
class MetalFp8GemvThunk : public Thunk {
 public:
  MetalFp8GemvThunk(ThunkInfo thunk_info, BufferAllocation::Slice x,
                    Shape x_shape, BufferAllocation::Slice w, Shape w_shape,
                    BufferAllocation::Slice scale, Shape scale_shape,
                    BufferAllocation::Slice out, Shape out_shape, int64_t b,
                    int64_t k, int64_t n);

  MetalFp8GemvThunk(const MetalFp8GemvThunk&) = delete;
  MetalFp8GemvThunk& operator=(const MetalFp8GemvThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  // Lazily, on the first execute for a given executor: compile the embedded
  // metallib (process-cached), load the kernel, and stage the packed
  // (B, K, N, K/128) dims into a small device buffer. Must hold mu_.
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice x_, w_, scale_, out_;
  const Shape x_shape_, w_shape_, scale_shape_, out_shape_;
  const int64_t b_, k_, n_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  // b==1: the per-row GEMV (optimal single-stream). b>1: the MLX Steel
  // simdgroup_matrix tiled q-GEMM (kernel_steel_), which reads each f8 weight
  // byte once and reuses it across both rows and columns -- the per-row GEMV's
  // DRAM traffic scales with batch and collapses at b=16. (kernel_tiled_ was an
  // intermediate 1D tile that only reused across rows; kept loaded for A/B.)
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_tiled_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_steel_ ABSL_GUARDED_BY(mu_);

  // Packed {B, K, N, K/128} int4, bound as the kernel's `constant int4& dims`
  // arg (buffer 4), allocated once per executor.
  stream_executor::DeviceAddressBase p_dims_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_FP8_GEMV_THUNK_H_
