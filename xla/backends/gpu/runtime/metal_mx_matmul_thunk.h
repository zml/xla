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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_MX_MATMUL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_MX_MATMUL_THUNK_H_

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

// Metal MX arm of `zml$scaled_matmul` (e8m0 group-32 / legacy u32 pack).
// x bf16[M,K], w f8/f4[N,K] (or legacy u32 pack), scales E8M0[N,K/32] → bf16[M,N].
class MetalMxMatmulThunk : public Thunk {
 public:
  MetalMxMatmulThunk(ThunkInfo thunk_info, BufferAllocation::Slice x,
                        Shape x_shape, BufferAllocation::Slice w, Shape w_shape,
                        BufferAllocation::Slice scales, Shape scales_shape,
                        BufferAllocation::Slice out, Shape out_shape, int64_t m,
                        int64_t k, int64_t n, int64_t bits, int64_t group_size);

  MetalMxMatmulThunk(const MetalMxMatmulThunk&) = delete;
  MetalMxMatmulThunk& operator=(const MetalMxMatmulThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  // Lazily (per executor): compile the vendored MLX metallibs, load the
  // per-bit-width kernels, and stage {M, K, N, K/group_size} into a device
  // buffer. Must hold mu_.
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice x_, w_, scales_, out_;
  const Shape x_shape_, w_shape_, scales_shape_, out_shape_;
  const int64_t m_, k_, n_, bits_, group_size_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  // Decode (M small): MLX qmv / qmv_fast (bandwidth-bound matvec). Prefill
  // (M large): MLX qmm_t (Steel tiled GEMM, reuses the weight across rows). All
  // share one arg ABI {x, w, scales, out, dims}.
  std::unique_ptr<stream_executor::Kernel> kernel_qmv_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_qmv_fast_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_qmm_ ABSL_GUARDED_BY(mu_);
  // Packed {M, K, N, K/group_size} int4, bound as `constant int4& dims`.
  stream_executor::DeviceAddressBase p_dims_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_MX_MATMUL_THUNK_H_
