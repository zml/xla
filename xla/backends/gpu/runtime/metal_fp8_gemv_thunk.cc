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

#include "xla/backends/gpu/runtime/metal_fp8_gemv_thunk.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/fp8_gemv.h"
#include "xla/service/gpu/metal_kernels/fp8_gemv_pc.h"
#include "xla/service/gpu/metal_kernels/mlx_kernels.h"
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

MetalFp8GemvThunk::MetalFp8GemvThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice x, Shape x_shape,
    BufferAllocation::Slice w, Shape w_shape, BufferAllocation::Slice scale,
    Shape scale_shape, BufferAllocation::Slice out, Shape out_shape, int64_t b,
    int64_t k, int64_t n, bool per_channel)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      x_(x),
      w_(w),
      scale_(scale),
      out_(out),
      x_shape_(std::move(x_shape)),
      w_shape_(std::move(w_shape)),
      scale_shape_(std::move(scale_shape)),
      out_shape_(std::move(out_shape)),
      b_(b),
      k_(k),
      n_(n),
      per_channel_(per_channel) {}

absl::Status MetalFp8GemvThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (kernel_ != nullptr && kernel_steel_ != nullptr && kernel_pc_ != nullptr &&
      kernel_steel64_ != nullptr && kernel_pc_qmm_ != nullptr &&
      kernel_pc_qmm64_ != nullptr)
    return absl::OkStatus();

  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);

  // 5 buffer args: 0..3 device buffers (x, w, scale, out) + 4 packed dims.
  {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_fp8_gemv()));
    TF_ASSIGN_OR_RETURN(
        kernel_,
        metal_exec->LoadKernelWithConstants(lib, "fp8_gemv", /*arity=*/5, {}));
  }
  {
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> lib,
        CompileMetalSourceToMetallibCached(get_mlx_steel_qgemm()));
    TF_ASSIGN_OR_RETURN(kernel_steel_,
                        metal_exec->LoadKernelWithConstants(
                            lib, "fp8_qmm_t", /*arity=*/5, {}));
    TF_ASSIGN_OR_RETURN(kernel_steel64_,
                        metal_exec->LoadKernelWithConstants(
                            lib, "fp8_qmm_t_bm64", /*arity=*/5, {}));
    TF_ASSIGN_OR_RETURN(kernel_pc_qmm_,
                        metal_exec->LoadKernelWithConstants(
                            lib, "fp8_qmm_t_pc", /*arity=*/5, {}));
    TF_ASSIGN_OR_RETURN(kernel_pc_qmm64_,
                        metal_exec->LoadKernelWithConstants(
                            lib, "fp8_qmm_t_pc_bm64", /*arity=*/5, {}));
  }
  {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib_pc,
                        CompileMetalSourceToMetallibCached(get_fp8_gemv_pc()));
    TF_ASSIGN_OR_RETURN(kernel_pc_, metal_exec->LoadKernelWithConstants(
                                        lib_pc, "fp8_gemv_pc", /*arity=*/5, {}));
  }

  const int32_t dims[4] = {static_cast<int32_t>(b_), static_cast<int32_t>(k_),
                           static_cast<int32_t>(n_),
                           static_cast<int32_t>(k_ / 128)};
  p_dims_ = executor->Allocate(sizeof(dims), 0);
  if (p_dims_.opaque() == nullptr) {
    return absl::ResourceExhaustedError(
        "zml$scaled_matmul (FP8): dims alloc failed.");
  }
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&p_dims_, dims, sizeof(dims)));

  return absl::OkStatus();
}

absl::Status MetalFp8GemvThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::MutexLock lock(&mu_);
  if (executor_ != executor) {
    kernel_ = nullptr;
    kernel_steel_ = nullptr;
    kernel_pc_ = nullptr;
    kernel_steel64_ = nullptr;
    kernel_pc_qmm_ = nullptr;
    kernel_pc_qmm64_ = nullptr;
    TF_RETURN_IF_ERROR(EnsureLoaded(executor));
    executor_ = executor;
  }

  se::KernelArgsPackedArray args(/*num_args=*/5);
  args.add_argument(allocs.GetDeviceAddress(x_));      // 0  x
  args.add_argument(allocs.GetDeviceAddress(w_));      // 1  w (f8)
  args.add_argument(allocs.GetDeviceAddress(scale_));  // 2  scale
  args.add_argument(allocs.GetDeviceAddress(out_));    // 3  out
  args.add_argument(p_dims_);                          // 4  dims (int4)

  if (per_channel_) {
    if (b_ == 1) {
      return kernel_pc_->Launch(se::ThreadDim(256, 1, 1),
                                se::BlockDim(static_cast<uint64_t>(n_), 1, 1),
                                stream, args);
    }
    const bool large_m = b_ > 16;
    se::Kernel* kpc = large_m ? kernel_pc_qmm64_.get() : kernel_pc_qmm_.get();
    const int64_t kPcBM = large_m ? 64 : 16;
    constexpr int64_t kPcBN = 64;
    return kpc->Launch(
        se::ThreadDim(32, 2, 2),
        se::BlockDim(static_cast<uint64_t>((n_ + kPcBN - 1) / kPcBN),
                     static_cast<uint64_t>((b_ + kPcBM - 1) / kPcBM), 1),
        stream, args);
  }
  if (b_ == 1) {
    return kernel_->Launch(se::ThreadDim(256, 1, 1),
                           se::BlockDim(static_cast<uint64_t>(n_), 1, 1), stream,
                           args);
  }
  const bool large_m = b_ > 16;
  stream_executor::Kernel* ksteel =
      large_m ? kernel_steel64_.get() : kernel_steel_.get();
  const int64_t kSteelBM = large_m ? 64 : 16;
  constexpr int64_t kSteelBN = 64;
  return ksteel->Launch(
      se::ThreadDim(32, 2, 2),
      se::BlockDim(static_cast<uint64_t>((n_ + kSteelBN - 1) / kSteelBN),
                   static_cast<uint64_t>((b_ + kSteelBM - 1) / kSteelBM), 1),
      stream, args);
}

Thunk::BufferUses MetalFp8GemvThunk::buffer_uses() const {
  return {
      BufferUse::Read(x_, x_shape_),
      BufferUse::Read(w_, w_shape_),
      BufferUse::Read(scale_, scale_shape_),
      BufferUse::Write(out_, out_shape_),
  };
}

absl::StatusOr<ThunkProto> MetalFp8GemvThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalFp8GemvThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
