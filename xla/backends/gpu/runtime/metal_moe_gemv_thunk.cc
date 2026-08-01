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

#include "xla/backends/gpu/runtime/metal_moe_gemv_thunk.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/bf16_moe_gemv.h"
#include "xla/service/gpu/metal_kernels/fp8_moe_gemv.h"
#include "xla/service/gpu/metal_kernels/metalblas_shaders.h"
#include "xla/service/gpu/metal_kernels/moe_argsort.h"
#include "xla/service/gpu/metal_kernels/permute_rows.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_kernel.h"
#include "xla/stream_executor/metal/metal_runtime.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

MetalMoeGemvThunk::MetalMoeGemvThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice x, Shape x_shape,
    BufferAllocation::Slice w, Shape w_shape, BufferAllocation::Slice scale,
    Shape scale_shape, BufferAllocation::Slice expert_id, Shape expert_id_shape,
    BufferAllocation::Slice out, Shape out_shape, int64_t r, int64_t k,
    int64_t n, BufferAllocation::Slice num_tokens, Shape num_tokens_shape,
    int64_t top_k)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      x_(x),
      w_(w),
      scale_(scale),
      expert_id_(expert_id),
      out_(out),
      x_shape_(std::move(x_shape)),
      w_shape_(std::move(w_shape)),
      scale_shape_(std::move(scale_shape)),
      expert_id_shape_(std::move(expert_id_shape)),
      out_shape_(std::move(out_shape)),
      r_(r),
      k_(k),
      n_(n),
      num_tokens_(num_tokens),
      num_tokens_shape_(std::move(num_tokens_shape)),
      top_k_(top_k),
      has_num_tokens_(num_tokens.allocation() != nullptr && top_k > 0) {}

absl::Status MetalMoeGemvThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (kernel_ != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  using FC = se::metal::MetalFunctionConstant;
  const bool is_fp8 = (w_shape_.element_type() == F8E4M3FN);

  sorted_path_ = (r_ >= kSortedMinR);

  if (is_fp8) {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_fp8_moe_gemv()));
    TF_ASSIGN_OR_RETURN(kernel_, metal_exec->LoadKernelWithConstants(
                                     lib, "fp8_moe_gemv", /*arity=*/6, {}));
  } else {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_bf16_moe_gemv()));
    TF_ASSIGN_OR_RETURN(kernel_, metal_exec->LoadKernelWithConstants(
                                     lib, "bf16_moe_gemv", /*arity=*/5, {}));
  }
  if (sorted_path_) {
    const int32_t align_n = (n_ % 32 == 0) ? 1 : 0;
    const int32_t align_k = (k_ % 32 == 0) ? 1 : 0;
    const FC fc[] = {{200, FC::Kind::kBool, 0},
                     {201, FC::Kind::kBool, align_n},
                     {202, FC::Kind::kBool, align_k}};
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> lib,
        CompileMetalSourceToMetallibCached(get_mlx_steel_qgemm()));
    if (is_fp8) {
      TF_ASSIGN_OR_RETURN(kernel_steel_,
                          metal_exec->LoadKernelWithConstants(
                              lib, "fp8_gather_qmm_rhs", /*arity=*/7, fc));
    } else {
      TF_ASSIGN_OR_RETURN(kernel_steel_,
                          metal_exec->LoadKernelWithConstants(
                              lib, "bf16_gather_mm_rhs", /*arity=*/6, fc));
    }
  }

  const int32_t dims[4] = {static_cast<int32_t>(r_), static_cast<int32_t>(k_),
                           static_cast<int32_t>(n_),
                           static_cast<int32_t>(k_ / 128)};
  p_dims_ = executor->Allocate(sizeof(dims), 0);
  if (p_dims_.opaque() == nullptr) {
    return absl::ResourceExhaustedError(
        "__metal$moe_gemm$f8: dims alloc failed.");
  }
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&p_dims_, dims, sizeof(dims)));

  const int32_t top_k_eff = has_num_tokens_ ? static_cast<int32_t>(top_k_) : 1;

  const int32_t mnk[4] = {static_cast<int32_t>(r_), static_cast<int32_t>(n_),
                          static_cast<int32_t>(k_), top_k_eff};
  p_dims_steel_ = executor->Allocate(sizeof(mnk), 0);
  if (p_dims_steel_.opaque() == nullptr) {
    return absl::ResourceExhaustedError(
        "__metal$moe_gemm$f8: steel dims alloc failed.");
  }
  TF_RETURN_IF_ERROR(
      executor->SynchronousMemcpy(&p_dims_steel_, mnk, sizeof(mnk)));

  if (!sorted_path_) return absl::OkStatus();

  const int32_t kE = static_cast<int32_t>(w_shape_.dimensions(0));
  if (kE > 256) {
    return absl::UnimplementedError(
        "__metal$moe_gemm$f8: moe_argsort supports <= 256 experts.");
  }
  {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_moe_argsort()));
    TF_ASSIGN_OR_RETURN(kernel_argsort_, metal_exec->LoadKernelWithConstants(
                                             lib, "moe_argsort", /*arity=*/5,
                                             {}));
    TF_ASSIGN_OR_RETURN(kernel_steel_grid_,
                        metal_exec->LoadKernelWithConstants(
                            lib, "moe_steel_grid", /*arity=*/3, {}));
  }
  {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_permute_rows()));
    TF_ASSIGN_OR_RETURN(kernel_gather_, metal_exec->LoadKernelWithConstants(
                                            lib, "gather_rows", /*arity=*/5,
                                            {}));
    TF_ASSIGN_OR_RETURN(kernel_scatter_, metal_exec->LoadKernelWithConstants(
                                             lib, "scatter_rows", /*arity=*/5,
                                             {}));
  }

  auto alloc = [&](se::DeviceAddressBase* p, int64_t bytes,
                   const char* what) -> absl::Status {
    *p = executor->Allocate(bytes, 0);
    if (p->opaque() == nullptr) {
      return absl::ResourceExhaustedError(
          absl::StrCat("__metal$moe_gemm$f8: ", what, " alloc failed."));
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(alloc(&p_order_, r_ * sizeof(int32_t), "order"));
  TF_RETURN_IF_ERROR(alloc(&p_idx_sorted_, r_ * sizeof(int32_t), "idx_sorted"));
  TF_RETURN_IF_ERROR(alloc(&p_x_sorted_, r_ * k_ * sizeof(uint16_t), "x_sorted"));
  TF_RETURN_IF_ERROR(
      alloc(&p_out_sorted_, r_ * n_ * sizeof(uint16_t), "out_sorted"));

  auto stage_dims = [&](se::DeviceAddressBase* p, int32_t a,
                        int32_t b) -> absl::Status {
    const int32_t d[4] = {a, b, top_k_eff, 0};
    *p = executor->Allocate(sizeof(d), 0);
    if (p->opaque() == nullptr) {
      return absl::ResourceExhaustedError("__metal$moe_gemm$f8: dims alloc.");
    }
    return executor->SynchronousMemcpy(p, d, sizeof(d));
  };
  TF_RETURN_IF_ERROR(stage_dims(&p_argsort_dims_, static_cast<int32_t>(r_), kE));
  TF_RETURN_IF_ERROR(
      stage_dims(&p_gx_dims_, static_cast<int32_t>(r_), static_cast<int32_t>(k_)));
  TF_RETURN_IF_ERROR(stage_dims(&p_gout_dims_, static_cast<int32_t>(r_),
                                static_cast<int32_t>(n_)));

  {
    const int32_t nt = static_cast<int32_t>(r_);
    p_num_tokens_fallback_ = executor->Allocate(sizeof(nt), 0);
    if (p_num_tokens_fallback_.opaque() == nullptr) {
      return absl::ResourceExhaustedError(
          "__metal$moe_gemm$f8: num_tokens fallback alloc failed.");
    }
    TF_RETURN_IF_ERROR(
        executor->SynchronousMemcpy(&p_num_tokens_fallback_, &nt, sizeof(nt)));
  }

  constexpr int32_t kBM = 16, kBN = 32;
  const int32_t steel_args[4] = {static_cast<int32_t>(r_),
                                 static_cast<int32_t>((n_ + kBN - 1) / kBN),
                                 top_k_eff, kBM};
  p_steel_grid_args_ = executor->Allocate(sizeof(steel_args), 0);
  if (p_steel_grid_args_.opaque() == nullptr) {
    return absl::ResourceExhaustedError(
        "__metal$moe_gemm$f8: steel grid args alloc failed.");
  }
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&p_steel_grid_args_, steel_args,
                                                 sizeof(steel_args)));
  p_steel_grid_ = executor->Allocate(4 * sizeof(uint32_t), 0);  // {gx,gy,gz,pad}
  if (p_steel_grid_.opaque() == nullptr) {
    return absl::ResourceExhaustedError(
        "__metal$moe_gemm$f8: steel grid buffer alloc failed.");
  }

  return absl::OkStatus();
}

absl::Status MetalMoeGemvThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::MutexLock lock(&mu_);
  if (executor_ != executor) {
    kernel_ = nullptr;
    kernel_steel_ = nullptr;
    kernel_argsort_ = nullptr;
    kernel_gather_ = nullptr;
    kernel_scatter_ = nullptr;
    kernel_steel_grid_ = nullptr;
    TF_RETURN_IF_ERROR(EnsureLoaded(executor));
    executor_ = executor;
  }

  if (sorted_path_) {
    const bool clamp = has_num_tokens_;
    const se::DeviceAddressBase nt =
        clamp ? allocs.GetDeviceAddress(num_tokens_) : p_num_tokens_fallback_;

    se::KernelArgsPackedArray a_sort(/*num_args=*/5);
    a_sort.add_argument(allocs.GetDeviceAddress(expert_id_));
    a_sort.add_argument(p_order_);
    a_sort.add_argument(p_idx_sorted_);
    a_sort.add_argument(p_argsort_dims_);
    a_sort.add_argument(nt);
    TF_RETURN_IF_ERROR(kernel_argsort_->Launch(
        se::ThreadDim(256, 1, 1), se::BlockDim(1, 1, 1), stream, a_sort));

    se::KernelArgsPackedArray a_gx(/*num_args=*/5);
    a_gx.add_argument(allocs.GetDeviceAddress(x_));
    a_gx.add_argument(p_order_);
    a_gx.add_argument(p_x_sorted_);
    a_gx.add_argument(p_gx_dims_);
    a_gx.add_argument(nt);
    const uint64_t kcols = (static_cast<uint64_t>(k_) + 3) / 4;
    TF_RETURN_IF_ERROR(kernel_gather_->Launch(
        se::ThreadDim(64, 1, 1),
        se::BlockDim((kcols + 63) / 64, static_cast<uint64_t>(r_), 1), stream,
        a_gx));

    const bool is_fp8 = (w_shape_.element_type() == F8E4M3FN);
    se::KernelArgsPackedArray a_mm(/*num_args=*/is_fp8 ? 7 : 6);
    a_mm.add_argument(p_x_sorted_);
    a_mm.add_argument(allocs.GetDeviceAddress(w_));
    if (is_fp8) {
      a_mm.add_argument(allocs.GetDeviceAddress(scale_));
    }
    a_mm.add_argument(p_idx_sorted_);
    a_mm.add_argument(p_out_sorted_);
    a_mm.add_argument(p_dims_steel_);
    a_mm.add_argument(nt);
    constexpr int64_t kBM = 16, kBN = 32;

    se::KernelArgsPackedArray a_grid(/*num_args=*/3);
    a_grid.add_argument(nt);
    a_grid.add_argument(p_steel_grid_args_);
    a_grid.add_argument(p_steel_grid_);
    TF_RETURN_IF_ERROR(kernel_steel_grid_->Launch(
        se::ThreadDim(1, 1, 1), se::BlockDim(1, 1, 1), stream, a_grid));

    auto* steel_metal = static_cast<se::metal::MetalKernel*>(kernel_steel_.get());
    TF_RETURN_IF_ERROR(steel_metal->LaunchIndirect(
        se::ThreadDim(32, 2, 1),
        se::BlockDim(static_cast<uint64_t>((n_ + kBN - 1) / kBN),
                     static_cast<uint64_t>((r_ + kBM - 1) / kBM), 1),
        p_steel_grid_.opaque(), stream, a_mm));

    se::KernelArgsPackedArray a_sc(/*num_args=*/5);
    a_sc.add_argument(p_out_sorted_);
    a_sc.add_argument(p_order_);
    a_sc.add_argument(allocs.GetDeviceAddress(out_));
    a_sc.add_argument(p_gout_dims_);
    a_sc.add_argument(nt);
    const uint64_t ncols = (static_cast<uint64_t>(n_) + 3) / 4;
    return kernel_scatter_->Launch(
        se::ThreadDim(64, 1, 1),
        se::BlockDim((ncols + 63) / 64, static_cast<uint64_t>(r_), 1), stream,
        a_sc);
  }

  const bool is_fp8 = (w_shape_.element_type() == F8E4M3FN);
  constexpr int64_t kMoeGemvTN = 8;  // must match {fp8,bf16}_moe_gemv.metal TN
  se::KernelArgsPackedArray args(/*num_args=*/is_fp8 ? 6 : 5);
  args.add_argument(allocs.GetDeviceAddress(x_));  // 0  x
  args.add_argument(allocs.GetDeviceAddress(w_));  // 1  w ([E,N,K])
  if (is_fp8) {
    args.add_argument(allocs.GetDeviceAddress(scale_));  // 2  scale (fp8 only)
  }
  args.add_argument(allocs.GetDeviceAddress(expert_id_));  // expert_id
  args.add_argument(allocs.GetDeviceAddress(out_));        // out
  args.add_argument(p_dims_);                              // dims (int4)
  return kernel_->Launch(
      se::ThreadDim(256, 1, 1),
      se::BlockDim(static_cast<uint64_t>((n_ + kMoeGemvTN - 1) / kMoeGemvTN),
                   static_cast<uint64_t>(r_), 1),
      stream, args);
}

Thunk::BufferUses MetalMoeGemvThunk::buffer_uses() const {
  Thunk::BufferUses uses = {
      BufferUse::Read(x_, x_shape_),
      BufferUse::Read(w_, w_shape_),
  };
  if (w_shape_.element_type() == F8E4M3FN) {
    uses.push_back(BufferUse::Read(scale_, scale_shape_));  // fp8 only
  }
  uses.push_back(BufferUse::Read(expert_id_, expert_id_shape_));
  if (has_num_tokens_) {
    uses.push_back(BufferUse::Read(num_tokens_, num_tokens_shape_));
  }
  uses.push_back(BufferUse::Write(out_, out_shape_));
  return uses;
}

absl::StatusOr<ThunkProto> MetalMoeGemvThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalMoeGemvThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
