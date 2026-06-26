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
    int64_t n)
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
      n_(n) {}

absl::Status MetalMoeGemvThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (kernel_ != nullptr) return absl::OkStatus();
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  using FC = se::metal::MetalFunctionConstant;
  const bool is_fp8 = (w_shape_.element_type() == F8E4M3FN);

  // Large R (prefill) routes many rows to each expert, so it pays to sort the
  // rows by expert and use the weight-reuse MLX gather q-GEMM; small R (decode,
  // ~0.5 rows/expert) has no reuse, so the per-row x-caching GEMV is better.
  sorted_path_ = (r_ >= kSortedMinR);

  // Decode / small-R path: per-row x-caching GEMV. fp8 takes 6 buffer args
  // (x, w, scale, expert_id, out) + dims; bf16 takes 5 (no scale).
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
  // MLX Steel gather q-GEMM for the sorted (prefill, R>=1024) path: fp8 takes
  // (x, w, scale, indices, out, mnk); bf16 drops scale. align_M=false -> the
  // safe partial-row path (R = tokens*top_k is arbitrary). The steel tile is
  // BN=BK=32, so align_N/K is only valid when N,K are multiples of 32 (Gemma4's
  // K=704 and N=2816/1408 all are). Set them from the real dims so a future
  // non-32 shape takes the bounds-checked path instead of silently corrupting.
  // Skip the compile for decode (R<1024).
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
                              lib, "fp8_gather_qmm_rhs", /*arity=*/6, fc));
    } else {
      TF_ASSIGN_OR_RETURN(kernel_steel_,
                          metal_exec->LoadKernelWithConstants(
                              lib, "bf16_gather_mm_rhs", /*arity=*/5, fc));
    }
  }

  // fp8_moe_gemv dims {R, K, N, K/128} (int4).
  const int32_t dims[4] = {static_cast<int32_t>(r_), static_cast<int32_t>(k_),
                           static_cast<int32_t>(n_),
                           static_cast<int32_t>(k_ / 128)};
  p_dims_ = executor->Allocate(sizeof(dims), 0);
  if (p_dims_.opaque() == nullptr) {
    return absl::ResourceExhaustedError(
        "__metal$moe_gemm$f8: dims alloc failed.");
  }
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&p_dims_, dims, sizeof(dims)));

  // fp8_gather_qmm_rhs dims {R, N, K} as a 16-byte int3 (Metal int3 is
  // 16-byte aligned; the 4th word is padding).
  const int32_t mnk[4] = {static_cast<int32_t>(r_), static_cast<int32_t>(n_),
                          static_cast<int32_t>(k_), 0};
  p_dims_steel_ = executor->Allocate(sizeof(mnk), 0);
  if (p_dims_steel_.opaque() == nullptr) {
    return absl::ResourceExhaustedError(
        "__metal$moe_gemm$f8: steel dims alloc failed.");
  }
  TF_RETURN_IF_ERROR(
      executor->SynchronousMemcpy(&p_dims_steel_, mnk, sizeof(mnk)));

  if (!sorted_path_) return absl::OkStatus();

  // --- Sorted-prefill path: counting-sort + row gather/scatter kernels --------
  const int32_t kE = static_cast<int32_t>(w_shape_.dimensions(0));
  if (kE > 256) {
    return absl::UnimplementedError(
        "__metal$moe_gemm$f8: moe_argsort supports <= 256 experts.");
  }
  {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_moe_argsort()));
    TF_ASSIGN_OR_RETURN(kernel_argsort_, metal_exec->LoadKernelWithConstants(
                                             lib, "moe_argsort", /*arity=*/4,
                                             {}));
  }
  {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_permute_rows()));
    TF_ASSIGN_OR_RETURN(kernel_gather_, metal_exec->LoadKernelWithConstants(
                                            lib, "gather_rows", /*arity=*/4,
                                            {}));
    TF_ASSIGN_OR_RETURN(kernel_scatter_, metal_exec->LoadKernelWithConstants(
                                             lib, "scatter_rows", /*arity=*/4,
                                             {}));
  }

  // Scratch: the permutation, the gathered ids, and the sorted x / out buffers.
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

  // int2 dims for the helper kernels (16-byte alloc; kernels read the first 2).
  auto stage_dims = [&](se::DeviceAddressBase* p, int32_t a,
                        int32_t b) -> absl::Status {
    const int32_t d[4] = {a, b, 0, 0};
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
    TF_RETURN_IF_ERROR(EnsureLoaded(executor));
    executor_ = executor;
  }

  // ---- Sorted-prefill path (large R): sort rows by expert, then the MLX gather
  // q-GEMM reuses each expert's weight across its now-contiguous run of rows.
  // Each kernel is enqueued on the same stream, so the writes of step N are
  // visible to step N+1 (no explicit barriers needed).
  if (sorted_path_) {
    // 1. Counting-sort the expert ids -> order (sorted pos -> orig row) and the
    //    grouped ids idx_sorted (single threadgroup, 256 threads).
    se::KernelArgsPackedArray a_sort(/*num_args=*/4);
    a_sort.add_argument(allocs.GetDeviceAddress(expert_id_));
    a_sort.add_argument(p_order_);
    a_sort.add_argument(p_idx_sorted_);
    a_sort.add_argument(p_argsort_dims_);
    TF_RETURN_IF_ERROR(kernel_argsort_->Launch(
        se::ThreadDim(256, 1, 1), se::BlockDim(1, 1, 1), stream, a_sort));

    // 2. Gather x rows into expert-sorted order: x_sorted[pos] = x[order[pos]].
    se::KernelArgsPackedArray a_gx(/*num_args=*/4);
    a_gx.add_argument(allocs.GetDeviceAddress(x_));
    a_gx.add_argument(p_order_);
    a_gx.add_argument(p_x_sorted_);
    a_gx.add_argument(p_gx_dims_);
    const uint64_t kcols = (static_cast<uint64_t>(k_) + 3) / 4;
    TF_RETURN_IF_ERROR(kernel_gather_->Launch(
        se::ThreadDim(64, 1, 1),
        se::BlockDim((kcols + 63) / 64, static_cast<uint64_t>(r_), 1), stream,
        a_gx));

    // 3. Gather q-GEMM on the sorted rows -> out_sorted (BM=16, BN=32). fp8
    //    carries a scale operand; bf16 does not.
    const bool is_fp8 = (w_shape_.element_type() == F8E4M3FN);
    se::KernelArgsPackedArray a_mm(/*num_args=*/is_fp8 ? 6 : 5);
    a_mm.add_argument(p_x_sorted_);
    a_mm.add_argument(allocs.GetDeviceAddress(w_));
    if (is_fp8) {
      a_mm.add_argument(allocs.GetDeviceAddress(scale_));
    }
    a_mm.add_argument(p_idx_sorted_);
    a_mm.add_argument(p_out_sorted_);
    a_mm.add_argument(p_dims_steel_);
    constexpr int64_t kBM = 16, kBN = 32;
    TF_RETURN_IF_ERROR(kernel_steel_->Launch(
        se::ThreadDim(32, 2, 1),
        se::BlockDim(static_cast<uint64_t>((n_ + kBN - 1) / kBN),
                     static_cast<uint64_t>((r_ + kBM - 1) / kBM), 1),
        stream, a_mm));

    // 4. Scatter out_sorted back to original row order: out[order[pos]] = ...
    se::KernelArgsPackedArray a_sc(/*num_args=*/4);
    a_sc.add_argument(p_out_sorted_);
    a_sc.add_argument(p_order_);
    a_sc.add_argument(allocs.GetDeviceAddress(out_));
    a_sc.add_argument(p_gout_dims_);
    const uint64_t ncols = (static_cast<uint64_t>(n_) + 3) / 4;
    return kernel_scatter_->Launch(
        se::ThreadDim(64, 1, 1),
        se::BlockDim((ncols + 63) / 64, static_cast<uint64_t>(r_), 1), stream,
        a_sc);
  }

  // ---- Decode path (small R): per-row x-caching tiled GEMV. One threadgroup
  // owns kMoeGemvTN output columns for a row (grid.x = ceil(N/TN)), TN
  // simdgroups computing one column each -- TN-x fewer threadgroups than the
  // per-(n,row) layout, amortizing the per-threadgroup fixed cost (256-entry f8
  // decode LUT + barriers + short-K reduction). At MoE decode density
  // (~0.5 rows/expert) there is no cross-row weight reuse to exploit, so the
  // sorted gather is not worth its sort+scatter overhead here.
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
  uses.push_back(BufferUse::Write(out_, out_shape_));
  return uses;
}

absl::StatusOr<ThunkProto> MetalMoeGemvThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalMoeGemvThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
