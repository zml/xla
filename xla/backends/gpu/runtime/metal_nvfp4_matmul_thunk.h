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

// NVFP4 arm of zml$scaled_matmul on Metal: fused weight-only matmul for
// compressed-tensors NVFP4 (f4e2m1 weights + e4m3 group-16 scales). Caller
// pre-scales x by 1/g_ct, where the saved weight-global value is the
// compressed-tensors encode divisor. MLX's global_scale_w instead means amax;
// the conversion for a future verbatim MLX call is 2688/g_ct.
//
// Dispatch (MLX QuantizedMatmul / dispatch_qmv / qmm_splitk for transposed
// weight-only NVFP4):
//   M == 1                         -> vendored MLX nvfp4_qmv
//   2 <= M < vector_limit          -> vendored MLX nvfp4_qmv_wide_{2..5}
//   M >= vector_limit, split_k<=1  -> steel nvfp4_qmm_t[_alN]
//   M >= vector_limit, split_k>1   -> steel nvfp4_qmm_t_splitk[_alN] + sum
// vector_limit = GetNvfp4QmvBatchLimit(K, N) (MLX get_qmv_batch_limit table).
// split_k = ComputeNvfp4QmmSplitK(M, N, K) (MLX qmm_splitk formula).
//
// Layout/packing (MLX row-contiguous):
//   x bf16[M,K], w f4 packed[N,K/2], scale e4m3[N,K/16], out bf16[M,N]
// align_N: the current 16x64 body uses N%64==0 for both non-split and split-K
// `_alN` entry points (the split-count heuristic still uses MLX's 32x32 tiles).
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
                        BufferAllocation::Slice out, Shape out_shape,
                        BufferAllocation::Slice workspace,
                        Shape workspace_shape, int64_t m, int64_t k, int64_t n,
                        char arch_size, int arch_gen);
  ~MetalNvfp4MatmulThunk() override;

  MetalNvfp4MatmulThunk(const MetalNvfp4MatmulThunk&) = delete;
  MetalNvfp4MatmulThunk& operator=(const MetalNvfp4MatmulThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  struct LoadedState;

  absl::StatusOr<std::shared_ptr<const LoadedState>> EnsureLoaded(
      stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice x_, w_, scale_, out_, workspace_;
  const Shape x_shape_, w_shape_, scale_shape_, out_shape_, workspace_shape_;
  const int64_t m_, k_, n_;
  // Compile-time target architecture. Runtime selection deliberately uses
  // these values rather than the executing device so it cannot disagree with
  // the workspace shape chosen before buffer assignment.
  const char arch_size_;
  const int arch_gen_;

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

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_NVFP4_MATMUL_THUNK_H_
