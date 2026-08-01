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

#include <array>
#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/metal_nvfp4_dispatch.h"
#include "xla/backends/gpu/runtime/metal_workspace.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/bf16_moe_gemv.h"
#include "xla/service/gpu/metal_kernels/fp8_moe_gemv.h"
#include "xla/service/gpu/metal_kernels/mlx_kernels.h"
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

namespace {

absl::StatusOr<se::DeviceAddressBase> GetMoeWorkspaceRegion(
    const se::DeviceAddressBase& workspace, const MetalWorkspaceRegion& region,
    uint64_t expected_size, const char* name) {
  if (region.offset < 0 || region.size < 0 ||
      static_cast<uint64_t>(region.size) != expected_size ||
      (region.size != 0 && region.offset % kMetalWorkspaceAlignment != 0)) {
    return absl::InternalError(absl::StrCat(
        "__metal$moe_gemm: invalid ", name,
        " workspace region {offset=", region.offset, ", size=", region.size,
        "}; expected ", expected_size, " aligned bytes."));
  }
  const uint64_t offset = static_cast<uint64_t>(region.offset);
  const uint64_t size = static_cast<uint64_t>(region.size);
  if (offset > workspace.size() || size > workspace.size() - offset) {
    return absl::InternalError(absl::StrCat(
        "__metal$moe_gemm: ", name, " workspace region at byte offset ", offset,
        " with length ", size, " exceeds workspace size ", workspace.size(),
        "."));
  }
  return workspace.GetByteSlice(offset, size);
}

}  // namespace

struct MetalMoeGemvThunk::LoadedState {
  explicit LoadedState(se::StreamExecutor* executor) : executor(executor) {}

  se::StreamExecutor* const executor;
  bool sorted_path = false;

  std::unique_ptr<se::Kernel> kernel;
  std::unique_ptr<se::Kernel> kernel_steel;
  std::unique_ptr<se::Kernel> kernel_argsort;
  std::unique_ptr<se::Kernel> kernel_gather;
  std::unique_ptr<se::Kernel> kernel_scatter;

  std::array<int32_t, 4> dims{};
  std::array<int32_t, 4> dims_steel{};
  // Every inline dimension payload is a full int4: Metal aligns three-lane
  // vectors to four, so an int3 payload is not a stable setBytes ABI.
  std::array<int32_t, 4> argsort_dims{};
  std::array<int32_t, 4> gx_dims{};
  std::array<int32_t, 4> gout_dims{};
  MetalMoeWorkspaceLayout workspace_layout;
};

MetalMoeGemvThunk::MetalMoeGemvThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice x, Shape x_shape,
    BufferAllocation::Slice w, Shape w_shape, BufferAllocation::Slice scale,
    Shape scale_shape, BufferAllocation::Slice expert_id, Shape expert_id_shape,
    BufferAllocation::Slice out, Shape out_shape,
    BufferAllocation::Slice workspace, Shape workspace_shape,
    BufferAllocation::Slice global_scale, Shape global_scale_shape,
    bool has_global_scale, int64_t r, int64_t k, int64_t n)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      x_(x),
      w_(w),
      scale_(scale),
      expert_id_(expert_id),
      out_(out),
      workspace_(workspace),
      global_scale_(global_scale),
      x_shape_(std::move(x_shape)),
      w_shape_(std::move(w_shape)),
      scale_shape_(std::move(scale_shape)),
      expert_id_shape_(std::move(expert_id_shape)),
      out_shape_(std::move(out_shape)),
      workspace_shape_(std::move(workspace_shape)),
      global_scale_shape_(std::move(global_scale_shape)),
      has_global_scale_(has_global_scale),
      r_(r),
      k_(k),
      n_(n) {}

MetalMoeGemvThunk::~MetalMoeGemvThunk() {
  absl::MutexLock lock(&mu_);
  states_.clear();
}

absl::StatusOr<std::shared_ptr<const MetalMoeGemvThunk::LoadedState>>
MetalMoeGemvThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (auto it = states_.find(executor); it != states_.end()) {
    return it->second;
  }

  auto next = std::make_shared<LoadedState>(executor);
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  using FC = se::metal::MetalFunctionConstant;
  const bool is_fp8 = (w_shape_.element_type() == F8E4M3FN);
  const bool is_nvfp4 = (w_shape_.element_type() == F4E2M1FN);

  const int64_t num_experts = w_shape_.dimensions(0);
  if (num_experts <= 0) {
    return absl::InvalidArgumentError(
        "__metal$moe_gemm requires at least one expert.");
  }
  TF_ASSIGN_OR_RETURN(
      const int64_t expected_workspace_bytes,
      GetMetalMoeWorkspaceBytes(r_, num_experts, k_, n_, is_nvfp4));
  if (expected_workspace_bytes > 0 &&
      (!workspace_shape_.IsArray() || workspace_shape_.element_type() != S8 ||
       workspace_shape_.dimensions().size() != 1 ||
       workspace_shape_.dimensions(0) != expected_workspace_bytes)) {
    return absl::InvalidArgumentError(
        absl::StrCat("__metal$moe_gemm: expected s8[", expected_workspace_bytes,
                     "] workspace; got ", workspace_shape_.ToString(), "."));
  }
  next->sorted_path =
      !is_fp8 && ShouldUseMetalMoeSortedPath(r_, num_experts, k_, n_, is_nvfp4);
  if (is_fp8) {
    LOG(INFO) << "__metal$moe_gemm$f8 EnsureLoaded: R=" << r_ << " K=" << k_
              << " N=" << n_ << " E=" << num_experts
              << " path=fp8_moe_gemv (sorted_steel disabled)";
  }

  if (!next->sorted_path) {
    if (is_fp8) {
      TF_ASSIGN_OR_RETURN(
          std::vector<uint8_t> lib,
          CompileMetalSourceToMetallibCached(get_fp8_moe_gemv()));
      TF_ASSIGN_OR_RETURN(
          next->kernel, metal_exec->LoadKernelWithConstants(lib, "fp8_moe_gemv",
                                                            /*arity=*/6, {}));
      LOG(INFO) << "__metal$moe_gemm$f8: loaded fp8_moe_gemv metallib ("
                << lib.size() << " bytes)";
    } else if (is_nvfp4) {
      TF_ASSIGN_OR_RETURN(
          std::vector<uint8_t> lib,
          CompileMetalSourceToMetallibCached(get_mlx_fp4_qmv()));
      const FC gemv_fc[] = {{440, FC::Kind::kBool, has_global_scale_}};
      TF_ASSIGN_OR_RETURN(
          next->kernel,
          metal_exec->LoadKernelWithConstants(
              lib, "nvfp4_gather_qmv",
              /*arity=*/has_global_scale_ ? 7 : 6, gemv_fc));
    } else {
      TF_ASSIGN_OR_RETURN(
          std::vector<uint8_t> lib,
          CompileMetalSourceToMetallibCached(get_bf16_moe_gemv()));
      TF_ASSIGN_OR_RETURN(next->kernel,
                          metal_exec->LoadKernelWithConstants(
                              lib, "bf16_moe_gemv", /*arity=*/5, {}));
    }
    next->dims = {static_cast<int32_t>(r_), static_cast<int32_t>(k_),
                  static_cast<int32_t>(n_), static_cast<int32_t>(num_experts)};

    std::shared_ptr<const LoadedState> published = std::move(next);
    states_.emplace(executor, published);
    return published;
  }

  // align_M stays false: the Steel row bound is what keeps a partial final BM
  // tile from reading idx_sorted past R.
  const int32_t align_m = 0;
  const int32_t align_n = (n_ % 32 == 0) ? 1 : 0;
  const int32_t align_k = (k_ % 32 == 0) ? 1 : 0;
  const FC fc[] = {{200, FC::Kind::kBool, align_m},
                   {201, FC::Kind::kBool, align_n},
                   {202, FC::Kind::kBool, align_k}};
  TF_ASSIGN_OR_RETURN(
      std::vector<uint8_t> steel_lib,
      CompileMetalSourceToMetallibCached(get_mlx_steel_qgemm()));
  if (is_fp8) {
    TF_ASSIGN_OR_RETURN(next->kernel_steel,
                        metal_exec->LoadKernelWithConstants(
                            steel_lib, "fp8_gather_qmm_rhs", /*arity=*/6, fc));
  } else if (is_nvfp4) {
    const FC fc_nvfp4[] = {{200, FC::Kind::kBool, align_m},
                           {201, FC::Kind::kBool, align_n},
                           {202, FC::Kind::kBool, align_k},
                           {440, FC::Kind::kBool, has_global_scale_}};
    TF_ASSIGN_OR_RETURN(
        next->kernel_steel,
        metal_exec->LoadKernelWithConstants(steel_lib, "nvfp4_gather_qmm_rhs",
                                            /*arity=*/has_global_scale_ ? 7 : 6,
                                            fc_nvfp4));
  } else {
    TF_ASSIGN_OR_RETURN(next->kernel_steel,
                        metal_exec->LoadKernelWithConstants(
                            steel_lib, "bf16_gather_mm_rhs", /*arity=*/5, fc));
  }

  next->dims_steel = {static_cast<int32_t>(r_), static_cast<int32_t>(n_),
                      static_cast<int32_t>(k_), 0};

  const int32_t kE = static_cast<int32_t>(num_experts);
  {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_moe_argsort()));
    TF_ASSIGN_OR_RETURN(next->kernel_argsort,
                        metal_exec->LoadKernelWithConstants(lib, "moe_argsort",
                                                            /*arity=*/4, {}));
  }
  {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_permute_rows()));
    TF_ASSIGN_OR_RETURN(next->kernel_gather,
                        metal_exec->LoadKernelWithConstants(lib, "gather_rows",
                                                            /*arity=*/5, {}));
    TF_ASSIGN_OR_RETURN(next->kernel_scatter,
                        metal_exec->LoadKernelWithConstants(lib, "scatter_rows",
                                                            /*arity=*/5, {}));
  }

  TF_ASSIGN_OR_RETURN(next->workspace_layout,
                      GetMetalMoeWorkspaceLayout(r_, k_, n_));

  next->argsort_dims = {static_cast<int32_t>(r_), kE, 0, 0};
  next->gx_dims = {static_cast<int32_t>(r_), static_cast<int32_t>(k_), kE, 0};
  next->gout_dims = {static_cast<int32_t>(r_), static_cast<int32_t>(n_), kE, 0};

  std::shared_ptr<const LoadedState> published = std::move(next);
  states_.emplace(executor, published);
  return published;
}

absl::Status MetalMoeGemvThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  std::shared_ptr<const LoadedState> state;
  {
    absl::MutexLock lock(&mu_);
    absl::StatusOr<std::shared_ptr<const LoadedState>> loaded =
        EnsureLoaded(executor);
    if (!loaded.ok()) {
      LOG(ERROR) << "__metal$moe_gemm EnsureLoaded failed: " << loaded.status()
                 << " w=" << w_shape_.ToString() << " R=" << r_ << " K=" << k_
                 << " N=" << n_;
      return loaded.status();
    }
    state = *std::move(loaded);
  }

  const bool is_fp8 = (w_shape_.element_type() == F8E4M3FN);
  const bool is_nvfp4 = (w_shape_.element_type() == F4E2M1FN);
  const bool has_scale = is_fp8 || is_nvfp4;
  const se::DeviceAddressBase output = allocs.GetDeviceAddress(out_);

  if (state->sorted_path) {
    const se::DeviceAddressBase workspace = allocs.GetDeviceAddress(workspace_);
    const MetalMoeWorkspaceLayout& layout = state->workspace_layout;
    if (workspace.size() != static_cast<uint64_t>(layout.total_bytes)) {
      return absl::InternalError(absl::StrCat(
          "__metal$moe_gemm: execution workspace has ", workspace.size(),
          " bytes; expected ", layout.total_bytes, "."));
    }
    const uint64_t rows = static_cast<uint64_t>(r_);
    TF_ASSIGN_OR_RETURN(const se::DeviceAddressBase order,
                        GetMoeWorkspaceRegion(workspace, layout.order,
                                              rows * sizeof(int32_t), "order"));
    TF_ASSIGN_OR_RETURN(
        const se::DeviceAddressBase idx_sorted,
        GetMoeWorkspaceRegion(workspace, layout.expert_ids,
                              rows * sizeof(int32_t), "expert-id"));
    TF_ASSIGN_OR_RETURN(
        const se::DeviceAddressBase x_sorted,
        GetMoeWorkspaceRegion(
            workspace, layout.x_sorted,
            rows * static_cast<uint64_t>(k_) * sizeof(uint16_t), "sorted-x"));
    TF_ASSIGN_OR_RETURN(const se::DeviceAddressBase out_sorted,
                        GetMoeWorkspaceRegion(
                            workspace, layout.out_sorted,
                            rows * static_cast<uint64_t>(n_) * sizeof(uint16_t),
                            "sorted-output"));

    se::KernelArgsPackedArray a_sort(/*num_args=*/4);
    a_sort.add_argument(allocs.GetDeviceAddress(expert_id_));
    a_sort.add_argument(order);
    a_sort.add_argument(idx_sorted);
    a_sort.add_argument(state->argsort_dims);
    TF_RETURN_IF_ERROR(state->kernel_argsort->Launch(
        se::ThreadDim(256, 1, 1), se::BlockDim(1, 1, 1), stream, a_sort));

    se::KernelArgsPackedArray a_gx(/*num_args=*/5);
    a_gx.add_argument(allocs.GetDeviceAddress(x_));
    a_gx.add_argument(order);
    a_gx.add_argument(x_sorted);
    a_gx.add_argument(state->gx_dims);
    a_gx.add_argument(allocs.GetDeviceAddress(expert_id_));
    const uint64_t kcols = (static_cast<uint64_t>(k_) + 3) / 4;
    TF_RETURN_IF_ERROR(state->kernel_gather->Launch(
        se::ThreadDim(64, 1, 1),
        se::BlockDim((kcols + 63) / 64, rows, 1), stream, a_gx));

    se::KernelArgsPackedArray a_mm(
        /*num_args=*/(has_scale ? 6 : 5) + (has_global_scale_ ? 1 : 0));
    a_mm.add_argument(x_sorted);
    a_mm.add_argument(allocs.GetDeviceAddress(w_));
    if (has_scale) {
      a_mm.add_argument(allocs.GetDeviceAddress(scale_));
    }
    a_mm.add_argument(idx_sorted);
    a_mm.add_argument(out_sorted);
    a_mm.add_argument(state->dims_steel);
    if (has_global_scale_) {
      a_mm.add_argument(allocs.GetDeviceAddress(global_scale_));
    }
    constexpr int64_t kBM = 16, kBN = 32;
    TF_RETURN_IF_ERROR(state->kernel_steel->Launch(
        se::ThreadDim(32, 2, 1),
        se::BlockDim(static_cast<uint64_t>((n_ + kBN - 1) / kBN),
                     static_cast<uint64_t>((r_ + kBM - 1) / kBM), 1),
        stream, a_mm));

    se::KernelArgsPackedArray a_sc(/*num_args=*/5);
    a_sc.add_argument(out_sorted);
    a_sc.add_argument(order);
    a_sc.add_argument(output);
    a_sc.add_argument(state->gout_dims);
    a_sc.add_argument(allocs.GetDeviceAddress(expert_id_));
    const uint64_t ncols = (static_cast<uint64_t>(n_) + 3) / 4;
    return state->kernel_scatter->Launch(
        se::ThreadDim(64, 1, 1),
        se::BlockDim((ncols + 63) / 64, rows, 1), stream, a_sc);
  }

  constexpr int64_t kMoeGemvTN = 8;
  se::KernelArgsPackedArray args(
      /*num_args=*/(has_scale ? 6 : 5) + (has_global_scale_ ? 1 : 0));
  args.add_argument(allocs.GetDeviceAddress(x_));
  args.add_argument(allocs.GetDeviceAddress(w_));
  if (has_scale) {
    args.add_argument(allocs.GetDeviceAddress(scale_));
  }
  args.add_argument(allocs.GetDeviceAddress(expert_id_));
  args.add_argument(output);
  args.add_argument(state->dims);
  if (has_global_scale_) {
    args.add_argument(allocs.GetDeviceAddress(global_scale_));
  }
  if (is_nvfp4) {
    return state->kernel->Launch(
        se::ThreadDim(32, 2, 1),
        se::BlockDim(static_cast<uint64_t>(r_),
                     static_cast<uint64_t>((n_ + kMoeGemvTN - 1) / kMoeGemvTN),
                     1),
        stream, args);
  }
  return state->kernel->Launch(
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
  if (w_shape_.element_type() == F8E4M3FN ||
      w_shape_.element_type() == F4E2M1FN) {
    uses.push_back(BufferUse::Read(scale_, scale_shape_));
  }
  uses.push_back(BufferUse::Read(expert_id_, expert_id_shape_));
  if (has_global_scale_) {
    uses.push_back(BufferUse::Read(global_scale_, global_scale_shape_));
  }
  uses.push_back(BufferUse::Write(out_, out_shape_));
  if (workspace_.size() != 0) {
    uses.push_back(BufferUse::Scratch(workspace_, workspace_shape_));
  }
  return uses;
}

absl::StatusOr<ThunkProto> MetalMoeGemvThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalMoeGemvThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
