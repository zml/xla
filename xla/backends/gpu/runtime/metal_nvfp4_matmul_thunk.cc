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

#include "xla/backends/gpu/runtime/metal_nvfp4_matmul_thunk.h"

#include <cstdint>
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

MetalNvfp4MatmulThunk::MetalNvfp4MatmulThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice x, Shape x_shape,
    BufferAllocation::Slice w, Shape w_shape, BufferAllocation::Slice scale,
    Shape scale_shape, BufferAllocation::Slice out, Shape out_shape, int64_t m,
    int64_t k, int64_t n)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      x_(x),
      w_(w),
      scale_(scale),
      out_(out),
      x_shape_(std::move(x_shape)),
      w_shape_(std::move(w_shape)),
      scale_shape_(std::move(scale_shape)),
      out_shape_(std::move(out_shape)),
      m_(m),
      k_(k),
      n_(n) {}

absl::Status MetalNvfp4MatmulThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (kernel_ != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);

  // Vendored MLX fp_qmv (f4e2m1 + e4m3 group-16). ABI: (w, scales, x, y, dims).
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                      CompileMetalSourceToMetallibCached(get_mlx_fp4_qmv()));
  TF_ASSIGN_OR_RETURN(
      kernel_,
      metal_exec->LoadKernelWithConstants(lib, "nvfp4_qmv", /*arity=*/5, {}));

  const int32_t dims[4] = {static_cast<int32_t>(m_), static_cast<int32_t>(k_),
                           static_cast<int32_t>(n_), 0};
  p_dims_ = executor->Allocate(sizeof(dims), 0);
  if (p_dims_.opaque() == nullptr) {
    return absl::ResourceExhaustedError(
        "zml$scaled_matmul (NVFP4): dims alloc failed.");
  }
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&p_dims_, dims, sizeof(dims)));
  return absl::OkStatus();
}

absl::Status MetalNvfp4MatmulThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::MutexLock lock(&mu_);
  if (executor_ != executor) {
    kernel_ = nullptr;
    TF_RETURN_IF_ERROR(EnsureLoaded(executor));
    executor_ = executor;
  }

  // One threadgroup (2 simdgroups x 32) owns 8 output rows and reduces over K;
  // grid.x = M, grid.y = ceil(N/8).
  se::KernelArgsPackedArray nargs(/*num_args=*/5);
  nargs.add_argument(allocs.GetDeviceAddress(w_));      // 0 w (packed f4)
  nargs.add_argument(allocs.GetDeviceAddress(scale_));  // 1 scales (e4m3)
  nargs.add_argument(allocs.GetDeviceAddress(x_));      // 2 x
  nargs.add_argument(allocs.GetDeviceAddress(out_));    // 3 y
  nargs.add_argument(p_dims_);                          // 4 dims
  constexpr int64_t kRowsPerTg = 8;
  return kernel_->Launch(
      se::ThreadDim(64, 1, 1),
      se::BlockDim(static_cast<uint64_t>(m_),
                   static_cast<uint64_t>((n_ + kRowsPerTg - 1) / kRowsPerTg), 1),
      stream, nargs);
}

Thunk::BufferUses MetalNvfp4MatmulThunk::buffer_uses() const {
  return {
      BufferUse::Read(x_, x_shape_),
      BufferUse::Read(w_, w_shape_),
      BufferUse::Read(scale_, scale_shape_),
      BufferUse::Write(out_, out_shape_),
  };
}

absl::StatusOr<ThunkProto> MetalNvfp4MatmulThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalNvfp4MatmulThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
