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

#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/metal_nvfp4_dispatch.h"
#include "xla/backends/gpu/runtime/metal_workspace.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metal_air_toolchain.h"
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

struct MetalNvfp4MatmulThunk::LoadedState {
  explicit LoadedState(se::StreamExecutor* executor) : executor(executor) {}

  se::StreamExecutor* const executor;
  Nvfp4DensePath path = Nvfp4DensePath::kQmv;
  int split_k = 1;

  std::unique_ptr<se::Kernel> kernel_qmv;
  // Index zero/one unused; vecs_per_tg=2..5 occupy their natural slots.
  std::unique_ptr<se::Kernel> kernel_qmv_wide[6];
  std::unique_ptr<se::Kernel> kernel_qmm;
  std::unique_ptr<se::Kernel> kernel_splitk;
  std::unique_ptr<se::Kernel> kernel_splitk_sum;

  // Stable 16-byte inline setBytes payloads.
  std::array<int32_t, 4> dims{};
  std::array<uint32_t, 4> split_control{};
  MetalNvfp4WorkspaceLayout workspace_layout;
};

MetalNvfp4MatmulThunk::MetalNvfp4MatmulThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice x, Shape x_shape,
    BufferAllocation::Slice w, Shape w_shape, BufferAllocation::Slice scale,
    Shape scale_shape, BufferAllocation::Slice out, Shape out_shape,
    BufferAllocation::Slice workspace, Shape workspace_shape, int64_t m,
    int64_t k, int64_t n, char arch_size, int arch_gen)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      x_(x),
      w_(w),
      scale_(scale),
      out_(out),
      workspace_(workspace),
      x_shape_(std::move(x_shape)),
      w_shape_(std::move(w_shape)),
      scale_shape_(std::move(scale_shape)),
      out_shape_(std::move(out_shape)),
      workspace_shape_(std::move(workspace_shape)),
      m_(m),
      k_(k),
      n_(n),
      arch_size_(arch_size),
      arch_gen_(arch_gen) {}

MetalNvfp4MatmulThunk::~MetalNvfp4MatmulThunk() {
  absl::MutexLock lock(&mu_);
  states_.clear();
}

absl::StatusOr<std::shared_ptr<const MetalNvfp4MatmulThunk::LoadedState>>
MetalNvfp4MatmulThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (auto it = states_.find(executor); it != states_.end()) {
    return it->second;
  }

  auto next = std::make_shared<LoadedState>(executor);
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);

  TF_ASSIGN_OR_RETURN(
      next->workspace_layout,
      GetMetalNvfp4WorkspaceLayout(m_, k_, n_, arch_size_, arch_gen_));
  // Only split-K owns scratch; every other path is emitted without a workspace
  // tuple element at all.
  if (next->workspace_layout.total_bytes > 0 &&
      (!workspace_shape_.IsArray() || workspace_shape_.element_type() != S8 ||
       workspace_shape_.dimensions().size() != 1 ||
       workspace_shape_.dimensions(0) != next->workspace_layout.total_bytes)) {
    return absl::InvalidArgumentError(
        absl::StrCat("zml$scaled_matmul (NVFP4): expected s8[",
                     next->workspace_layout.total_bytes, "] workspace; got ",
                     workspace_shape_.ToString(), "."));
  }

  next->path = SelectNvfp4DensePath(m_, k_, n_, arch_size_, arch_gen_);
  next->dims = {static_cast<int32_t>(m_), static_cast<int32_t>(k_),
                static_cast<int32_t>(n_), static_cast<int32_t>(k_ / 16)};

  if (next->path == Nvfp4DensePath::kQmv) {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_mlx_fp4_qmv()));
    const bool qmv_fast = n_ % 8 == 0 && k_ % 512 == 0;
    TF_ASSIGN_OR_RETURN(next->kernel_qmv,
                        metal_exec->LoadKernelWithConstants(
                            lib, qmv_fast ? "nvfp4_qmv_fast" : "nvfp4_qmv",
                            /*arity=*/5, {}));
  } else if (next->path == Nvfp4DensePath::kQmvWide) {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                        CompileMetalSourceToMetallibCached(get_mlx_fp4_qmv()));
    const int vecs = Nvfp4QmvWideVecsPerTg(static_cast<int>(m_));
    const char* name = Nvfp4QmvWideKernelName(vecs);
    if (name == nullptr) {
      return absl::InternalError(
          absl::StrCat("zml$scaled_matmul (NVFP4): bad qmv_wide vecs_per_tg=",
                       vecs, " for M=", m_));
    }
    TF_ASSIGN_OR_RETURN(
        next->kernel_qmv_wide[vecs],
        metal_exec->LoadKernelWithConstants(lib, name, /*arity=*/5, {}));
  } else if (next->path == Nvfp4DensePath::kQmm) {
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> lib,
        CompileMetalSourceToMetallibCached(get_mlx_steel_qgemm()));
    TF_ASSIGN_OR_RETURN(next->kernel_qmm,
                        metal_exec->LoadKernelWithConstants(
                            lib, Nvfp4QmmKernelName(static_cast<int>(n_)),
                            /*arity=*/5, {}));
  } else {
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> lib,
        CompileMetalSourceToMetallibCached(get_mlx_steel_qgemm()));
    TF_ASSIGN_OR_RETURN(
        next->kernel_splitk,
        metal_exec->LoadKernelWithConstants(
            lib, Nvfp4QmmSplitkKernelName(static_cast<int>(n_)), /*arity=*/6,
            {}));
    TF_ASSIGN_OR_RETURN(next->kernel_splitk_sum,
                        metal_exec->LoadKernelWithConstants(
                            lib, "nvfp4_splitk_sum", /*arity=*/3, {}));
    next->split_k = ComputeNvfp4QmmSplitK(
        static_cast<int>(m_), static_cast<int>(n_), static_cast<int>(k_));
    if (next->split_k <= 1) {
      return absl::InternalError(
          "zml$scaled_matmul (NVFP4): invalid split-K plan.");
    }
    next->split_control = {static_cast<uint32_t>(m_),
                           static_cast<uint32_t>(next->split_k),
                           static_cast<uint32_t>(k_ / next->split_k),
                           static_cast<uint32_t>(m_ * n_)};
  }

  std::shared_ptr<const LoadedState> published = std::move(next);
  states_.emplace(executor, published);
  return published;
}

absl::Status MetalNvfp4MatmulThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  std::shared_ptr<const LoadedState> state;
  {
    absl::MutexLock lock(&mu_);
    TF_ASSIGN_OR_RETURN(state, EnsureLoaded(executor));
  }

  const se::DeviceAddressBase output = allocs.GetDeviceAddress(out_);

  if (state->path == Nvfp4DensePath::kQmv) {
    se::KernelArgsPackedArray args(/*num_args=*/5);
    args.add_argument(allocs.GetDeviceAddress(w_));
    args.add_argument(allocs.GetDeviceAddress(scale_));
    args.add_argument(allocs.GetDeviceAddress(x_));
    args.add_argument(output);
    args.add_argument(state->dims);
    return state->kernel_qmv->Launch(
        se::ThreadDim(32, 2, 1),
        se::BlockDim(static_cast<uint64_t>(m_),
                     static_cast<uint64_t>((n_ + 7) / 8), 1),
        stream, args);
  }

  if (state->path == Nvfp4DensePath::kQmvWide) {
    const int vecs = Nvfp4QmvWideVecsPerTg(static_cast<int>(m_));
    if (vecs < 2 || vecs > 5 || state->kernel_qmv_wide[vecs] == nullptr) {
      return absl::InternalError(absl::StrCat(
          "zml$scaled_matmul (NVFP4): missing qmv_wide_", vecs, " for M=", m_));
    }
    se::KernelArgsPackedArray args(/*num_args=*/5);
    args.add_argument(allocs.GetDeviceAddress(w_));
    args.add_argument(allocs.GetDeviceAddress(scale_));
    args.add_argument(allocs.GetDeviceAddress(x_));
    args.add_argument(output);
    args.add_argument(state->dims);
    return state->kernel_qmv_wide[vecs]->Launch(
        se::ThreadDim(32, 2, 1),
        se::BlockDim(static_cast<uint64_t>((m_ + vecs - 1) / vecs),
                     static_cast<uint64_t>((n_ + 3) / 4), 1),
        stream, args);
  }

  if (state->path == Nvfp4DensePath::kQmm) {
    se::KernelArgsPackedArray args(/*num_args=*/5);
    args.add_argument(allocs.GetDeviceAddress(x_));
    args.add_argument(allocs.GetDeviceAddress(w_));
    args.add_argument(allocs.GetDeviceAddress(scale_));
    args.add_argument(output);
    args.add_argument(state->dims);
    return state->kernel_qmm->Launch(
        se::ThreadDim(32, 2, 2),
        se::BlockDim(static_cast<uint64_t>((n_ + kNvfp4QmmBN - 1) /
                                           kNvfp4QmmBN),
                     static_cast<uint64_t>((m_ + kNvfp4QmmBM - 1) /
                                           kNvfp4QmmBM),
                     1),
        stream, args);
  }

  // Split-K: partial planes into the workspace, then reduce them into out.
  const se::DeviceAddressBase workspace = allocs.GetDeviceAddress(workspace_);
  if (workspace.size() != state->workspace_layout.total_bytes) {
    return absl::InternalError(
        absl::StrCat("zml$scaled_matmul (NVFP4): execution workspace has ",
                     workspace.size(), " bytes; expected ",
                     state->workspace_layout.total_bytes, "."));
  }
  const se::DeviceAddressBase staging =
      workspace.GetByteSlice(state->workspace_layout.staging.offset,
                             state->workspace_layout.staging.size);
  se::KernelArgsPackedArray split_args(/*num_args=*/6);
  split_args.add_argument(allocs.GetDeviceAddress(x_));
  split_args.add_argument(allocs.GetDeviceAddress(w_));
  split_args.add_argument(allocs.GetDeviceAddress(scale_));
  split_args.add_argument(staging);
  split_args.add_argument(state->dims);
  split_args.add_argument(state->split_control);
  TF_RETURN_IF_ERROR(state->kernel_splitk->Launch(
      se::ThreadDim(32, 2, 2),
      se::BlockDim(
          static_cast<uint64_t>((n_ + kNvfp4SplitkBN - 1) / kNvfp4SplitkBN),
          static_cast<uint64_t>((m_ + kNvfp4SplitkBM - 1) / kNvfp4SplitkBM),
          static_cast<uint64_t>(state->split_k)),
      stream, split_args));
  se::KernelArgsPackedArray sum_args(/*num_args=*/3);
  sum_args.add_argument(staging);
  sum_args.add_argument(output);
  sum_args.add_argument(state->split_control);
  const uint64_t plane_stride = static_cast<uint64_t>(state->split_control[3]);
  return state->kernel_splitk_sum->Launch(
      se::ThreadDim(256, 1, 1), se::BlockDim((plane_stride + 255) / 256, 1, 1),
      stream, sum_args);
}

Thunk::BufferUses MetalNvfp4MatmulThunk::buffer_uses() const {
  Thunk::BufferUses uses = {
      BufferUse::Read(x_, x_shape_),
      BufferUse::Read(w_, w_shape_),
      BufferUse::Read(scale_, scale_shape_),
      BufferUse::Write(out_, out_shape_),
  };
  if (workspace_.size() != 0) {
    uses.push_back(BufferUse::Scratch(workspace_, workspace_shape_));
  }
  return uses;
}

absl::StatusOr<ThunkProto> MetalNvfp4MatmulThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalNvfp4MatmulThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
