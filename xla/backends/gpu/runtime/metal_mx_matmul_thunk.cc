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

#include "xla/backends/gpu/runtime/metal_mx_matmul_thunk.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/metalblas_shaders.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

MetalMxMatmulThunk::MetalMxMatmulThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice x, Shape x_shape,
    BufferAllocation::Slice w, Shape w_shape, BufferAllocation::Slice scales,
    Shape scales_shape, BufferAllocation::Slice out, Shape out_shape, int64_t m,
    int64_t k, int64_t n, int64_t bits, int64_t group_size)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      x_(x),
      w_(w),
      scales_(scales),
      out_(out),
      x_shape_(std::move(x_shape)),
      w_shape_(std::move(w_shape)),
      scales_shape_(std::move(scales_shape)),
      out_shape_(std::move(out_shape)),
      m_(m),
      k_(k),
      n_(n),
      bits_(bits),
      group_size_(group_size) {}

absl::Status MetalMxMatmulThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (kernel_qmv_ != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);

  // All kernels share the same 5-arg ABI: 0..3 device buffers (x, w, scales,
  // out) + 4 the packed `constant int4& dims`. Decode uses MLX's qmv /
  // qmv_fast (vendored/mlx/mlx_mxfp_qmv.h); prefill uses MLX's Steel qmm_t
  // (vendored/mlx/mlx_steel_qgemm.h).
  const bool f4 = (bits_ == 4);
  {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_mlx_mxfp_qmv()));
    TF_ASSIGN_OR_RETURN(kernel_qmv_,
                        metal_exec->LoadKernelWithConstants(
                            lib, f4 ? "mxfp4_qmv" : "mxfp8_qmv",
                            /*arity=*/5, {}));
    TF_ASSIGN_OR_RETURN(kernel_qmv_fast_,
                        metal_exec->LoadKernelWithConstants(
                            lib, f4 ? "mxfp4_qmv_fast" : "mxfp8_qmv_fast",
                            /*arity=*/5, {}));
  }
  {
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> lib,
        CompileMetalSourceToMetallibCached(get_mlx_steel_qgemm()));
    TF_ASSIGN_OR_RETURN(kernel_qmm_,
                        metal_exec->LoadKernelWithConstants(
                            lib, f4 ? "mxfp4_qmm_t" : "mxfp8_qmm_t",
                            /*arity=*/5, {}));
  }

  // Stage {M, K, N, K/group_size} into a 16-byte device buffer.
  const int32_t dims[4] = {static_cast<int32_t>(m_), static_cast<int32_t>(k_),
                           static_cast<int32_t>(n_),
                           static_cast<int32_t>(k_ / group_size_)};
  p_dims_ = executor->Allocate(sizeof(dims), 0);
  if (p_dims_.opaque() == nullptr) {
    return absl::ResourceExhaustedError(
        "zml$scaled_matmul (MX): dims alloc failed.");
  }
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&p_dims_, dims, sizeof(dims)));

  return absl::OkStatus();
}

absl::Status MetalMxMatmulThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::MutexLock lock(&mu_);
  if (executor_ != executor) {
    kernel_qmv_ = nullptr;
    kernel_qmv_fast_ = nullptr;
    kernel_qmm_ = nullptr;
    TF_RETURN_IF_ERROR(EnsureLoaded(executor));
    executor_ = executor;
  }

  se::KernelArgsPackedArray args(/*num_args=*/5);
  args.add_argument(allocs.GetDeviceAddress(x_));       // 0  x       bf16 [M,K]
  args.add_argument(allocs.GetDeviceAddress(w_));       // 1  w       MX (f8/f4 or packed u32)
  args.add_argument(allocs.GetDeviceAddress(scales_));  // 2  scales  E8M0 (f8e8m0 or u8)
  args.add_argument(allocs.GetDeviceAddress(out_));     // 3  out     bf16 [M,N]
  args.add_argument(p_dims_);                           // 4  dims    int4

  // Decode (thin M): MLX qmv. Each threadgroup (2 simdgroups = 64 threads)
  // computes 8 output rows for one input vector; grid = (M, ceil(N/8), 1). Use
  // the fast path when the shape is aligned (no bounds guards), else the guarded
  // qmv. pack_factor = 32/bits, fast block = pack_factor * 2 * 32.
  constexpr int64_t kQmvMaxBatch = 8;
  if (m_ < kQmvMaxBatch) {
    const int64_t pack_factor = 32 / bits_;
    const int64_t fast_block = pack_factor * 2 * 32;
    const bool aligned = (n_ % 8 == 0) && (k_ % fast_block == 0);
    se::Kernel* qmv = aligned ? kernel_qmv_fast_.get() : kernel_qmv_.get();
    return qmv->Launch(
        se::ThreadDim(32, 2, 1),
        se::BlockDim(static_cast<uint64_t>(m_),
                     static_cast<uint64_t>((n_ + 7) / 8), 1),
        stream, args);
  }

  // Prefill (large M): MLX qmm_t Steel tiled GEMM. One threadgroup per
  // (BN=64 cols, BM=16 rows) output tile, WM=WN=2 (=128 threads laid out
  // 32x2x2); each weight tile is loaded into threadgroup memory once and reused
  // across the BM rows -- the weight-reuse the per-row GEMV lacks.
  //
  // TODO: on M5+ (Apple GPU gen >= 17, macOS >= 26.2), route prefill to MLX's
  // qmm_nax kernel (fp_quantized_nax.h, mpp::tensor_ops::matmul2d on the Neural
  // Accelerators) when the transpose/K%64==0/non-f32 preconditions hold. Gate on
  // a nax-capability query (MetalComputeCapability::gpu_family, not yet wired) --
  // deferred because it cannot be validated without M5 hardware.
  constexpr int64_t kQmmBM = 16, kQmmBN = 64;
  return kernel_qmm_->Launch(
      se::ThreadDim(32, 2, 2),
      se::BlockDim(static_cast<uint64_t>((n_ + kQmmBN - 1) / kQmmBN),
                   static_cast<uint64_t>((m_ + kQmmBM - 1) / kQmmBM), 1),
      stream, args);
}

Thunk::BufferUses MetalMxMatmulThunk::buffer_uses() const {
  return {
      BufferUse::Read(x_, x_shape_),
      BufferUse::Read(w_, w_shape_),
      BufferUse::Read(scales_, scales_shape_),
      BufferUse::Write(out_, out_shape_),
  };
}

absl::StatusOr<ThunkProto> MetalMxMatmulThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalMxMatmulThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
