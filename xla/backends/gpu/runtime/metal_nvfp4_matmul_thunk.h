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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_NVFP4_MATMUL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_NVFP4_MATMUL_THUNK_H_

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

// NVFP4 arm of zml$scaled_matmul on Metal: fused weight-only matmul for
// compressed-tensors NVFP4 (f4e2m1 weights + e4m3 group-16 scales). Caller
// pre-scales x by 1/global; this thunk runs the vendored MLX nvfp4_qmv kernel
// for any M (decode or prefill).
//
// Operand contract:
//   0 x      bf16    [M, K]
//   1 w      f4e2m1  [N, K]   (packed 2 values/byte; K minor)
//   2 scale  e4m3    [N, K/16]
//   -> out   bf16    [M, N]
class MetalNvfp4MatmulThunk : public Thunk {
 public:
  MetalNvfp4MatmulThunk(ThunkInfo thunk_info, BufferAllocation::Slice x,
                        Shape x_shape, BufferAllocation::Slice w, Shape w_shape,
                        BufferAllocation::Slice scale, Shape scale_shape,
                        BufferAllocation::Slice out, Shape out_shape, int64_t m,
                        int64_t k, int64_t n);

  MetalNvfp4MatmulThunk(const MetalNvfp4MatmulThunk&) = delete;
  MetalNvfp4MatmulThunk& operator=(const MetalNvfp4MatmulThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice x_, w_, scale_, out_;
  const Shape x_shape_, w_shape_, scale_shape_, out_shape_;
  const int64_t m_, k_, n_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);
  // Packed {M, K, N, 0} bound as constant int4& dims.
  stream_executor::DeviceAddressBase p_dims_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_NVFP4_MATMUL_THUNK_H_
