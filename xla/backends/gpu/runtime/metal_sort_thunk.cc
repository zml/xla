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

#include "xla/backends/gpu/runtime/metal_sort_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/metalblas_shaders.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
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

namespace {

// Kernel-name dtype token (matches the instantiations in mlx_sort.h).
const char* SortDTypeName(PrimitiveType t) {
  switch (t) {
    case F32:
      return "float";
    case F16:
      return "half";
    case BF16:
      return "bfloat16_t";
    case S32:
      return "int";
    case S16:
      return "short";
    case S8:
      return "char";
    case U32:
      return "uint";
    case U16:
      return "ushort";
    case U8:
      return "uchar";
    default:
      return nullptr;
  }
}

int SortElemSize(PrimitiveType t) {
  switch (t) {
    case F32:
    case S32:
    case U32:
      return 4;
    case F16:
    case BF16:
    case S16:
    case U16:
      return 2;
    case S8:
    case U8:
      return 1;
    default:
      return 4;
  }
}

}  // namespace

MetalSortThunk::MetalSortThunk(ThunkInfo thunk_info,
                               BufferAllocation::Slice data,
                               BufferAllocation::Slice out_vals,
                               BufferAllocation::Slice out_idxs,
                               PrimitiveType dtype, int64_t rows,
                               int64_t num_elements, bool descending)
    : Thunk(Kind::kCustomKernel, std::move(thunk_info)),
      data_(data),
      out_vals_(out_vals),
      out_idxs_(out_idxs),
      dtype_(dtype),
      rows_(rows),
      n_(num_elements),
      descending_(descending),
      n_blocks_((num_elements + kPerBlock - 1) / kPerBlock) {}

void MetalSortThunk::Prewarm(se::StreamExecutor* executor, PrimitiveType dtype,
                             bool descending) {
  const char* dt = SortDTypeName(dtype);
  if (dt == nullptr) return;
  auto lib = CompileMetalSourceToMetallibCached(get_mlx_sort());
  if (!lib.ok()) return;
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  const char* dir = descending ? "desc" : "asc";
  std::vector<se::metal::MetalFunctionConstant> no_fc;
  metal_exec
      ->LoadKernelWithConstants(
          *lib, absl::StrCat("xla_sort_block_", dt, "_", dir), 4, no_fc)
      .IgnoreError();
  metal_exec
      ->LoadKernelWithConstants(
          *lib, absl::StrCat("xla_sort_part_", dt, "_", dir), 6, no_fc)
      .IgnoreError();
  metal_exec
      ->LoadKernelWithConstants(
          *lib, absl::StrCat("xla_sort_merge_", dt, "_", dir), 8, no_fc)
      .IgnoreError();
}

absl::Status MetalSortThunk::Ensure(se::StreamExecutor* executor) {
  if (executor_ == executor && block_pso_ != nullptr) return absl::OkStatus();
  executor_ = executor;

  const char* dt = SortDTypeName(dtype_);
  if (dt == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("metal$sort: unsupported dtype ", dtype_));
  }
  if (n_blocks_ > kMaxBlocks) {
    return absl::UnimplementedError(absl::StrCat(
        "metal$sort: sort axis too large (n=", n_, ", n_blocks=", n_blocks_,
        " > ", kMaxBlocks, ")."));
  }
  const char* dir = descending_ ? "desc" : "asc";

  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                      CompileMetalSourceToMetallibCached(get_mlx_sort()));
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  std::vector<se::metal::MetalFunctionConstant> no_fc;
  auto load = [&](const std::string& name, int arity)
      -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
    return metal_exec->LoadKernelWithConstants(lib, name, arity, no_fc);
  };

  TF_ASSIGN_OR_RETURN(block_pso_,
                      load(absl::StrCat("xla_sort_block_", dt, "_", dir), 4));
  TF_ASSIGN_OR_RETURN(part_pso_,
                      load(absl::StrCat("xla_sort_part_", dt, "_", dir), 6));
  TF_ASSIGN_OR_RETURN(merge_pso_,
                      load(absl::StrCat("xla_sort_merge_", dt, "_", dir), 8));

  // Stage the kernel's `constant int&` scalars into device buffers -- Metal
  // binds add_argument scalars only as device buffers, never by value.
  const int32_t n32 = static_cast<int32_t>(n_);
  args_n_ = executor->Allocate(sizeof(int32_t), 0);
  if (args_n_.opaque() == nullptr) {
    return absl::ResourceExhaustedError("metal$sort: args alloc failed.");
  }
  TF_RETURN_IF_ERROR(
      executor->SynchronousMemcpy(&args_n_, &n32, sizeof(int32_t)));

  if (n_blocks_ > 1) {
    const uint64_t vbytes =
        static_cast<uint64_t>(rows_) * n_ * SortElemSize(dtype_);
    const uint64_t ibytes = static_cast<uint64_t>(rows_) * n_ * 4;
    const uint64_t pbytes = static_cast<uint64_t>(rows_) * (n_blocks_ + 1) * 4;
    for (int i = 0; i < 2; ++i) {
      dev_vals_[i] = executor->Allocate(vbytes, 0);
      dev_idxs_[i] = executor->Allocate(ibytes, 0);
    }
    parts_ = executor->Allocate(pbytes, 0);
    // One merge_tiles value per merge pass (the p-th merge binds args_mt_ at
    // offset p*4). Same sequence the loop in ExecuteOnStream walks.
    std::vector<int32_t> mts;
    for (int mt = 2; (mt / 2) < n_blocks_; mt *= 2) mts.push_back(mt);
    const int32_t nb32 = static_cast<int32_t>(n_blocks_);
    args_nb_ = executor->Allocate(sizeof(int32_t), 0);
    args_mt_ = executor->Allocate(mts.size() * sizeof(int32_t), 0);
    if (dev_vals_[0].opaque() == nullptr || dev_vals_[1].opaque() == nullptr ||
        dev_idxs_[0].opaque() == nullptr || dev_idxs_[1].opaque() == nullptr ||
        parts_.opaque() == nullptr || args_nb_.opaque() == nullptr ||
        args_mt_.opaque() == nullptr) {
      return absl::ResourceExhaustedError("metal$sort: scratch alloc failed.");
    }
    TF_RETURN_IF_ERROR(
        executor->SynchronousMemcpy(&args_nb_, &nb32, sizeof(int32_t)));
    TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(
        &args_mt_, mts.data(), mts.size() * sizeof(int32_t)));
  }
  return absl::OkStatus();
}

absl::Status MetalSortThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::MutexLock lock(&mu_);
  if (executor_ != executor || block_pso_ == nullptr) {
    TF_RETURN_IF_ERROR(Ensure(executor));
  }

  const se::DeviceAddressBase in = allocs.GetDeviceAddress(data_);
  const se::DeviceAddressBase outv = allocs.GetDeviceAddress(out_vals_);
  const se::DeviceAddressBase outi = allocs.GetDeviceAddress(out_idxs_);

  // Single block: the whole row fits one threadgroup -- sort straight to output.
  // Grid = (1, rows, 1), group = (512, 1, 1).
  if (n_blocks_ == 1) {
    se::KernelArgsPackedArray args(4);
    args.add_argument(in);
    args.add_argument(outv);
    args.add_argument(outi);
    args.add_argument(args_n_);
    return block_pso_->Launch(se::ThreadDim(kBlockThreads, 1, 1),
                              se::BlockDim(1, rows_, 1), stream, args);
  }

  // Multi-block: per-block sort into dev[0], then ceil(log2(n_blocks)) merge
  // passes (partition + merge), the last merge writing straight to the output.
  // The stream's serial-dispatch encoder orders these true dependencies.
  {
    se::KernelArgsPackedArray args(4);
    args.add_argument(in);
    args.add_argument(dev_vals_[0]);
    args.add_argument(dev_idxs_[0]);
    args.add_argument(args_n_);
    TF_RETURN_IF_ERROR(block_pso_->Launch(
        se::ThreadDim(kBlockThreads, 1, 1),
        se::BlockDim(static_cast<uint64_t>(n_blocks_), rows_, 1), stream, args));
  }

  const int ntpg =
      (n_blocks_ + 1) < 1024 ? static_cast<int>(n_blocks_ + 1) : 1024;
  int cur = 0, pass = 0;
  for (int mt = 2; (mt / 2) < n_blocks_; mt *= 2, ++pass) {
    const int a = cur, b = 1 - cur;
    const bool last = (mt >= n_blocks_);  // the loop's final iteration
    // args_mt_ holds one int per pass; bind this pass's merge_tiles at pass*4.
    const se::DeviceAddressBase mt_p(
        static_cast<char*>(args_mt_.opaque()) + pass * sizeof(int32_t),
        sizeof(int32_t));

    // Partition. Grid = (1, rows, 1), group = (min(n_blocks+1, 1024), 1, 1).
    {
      se::KernelArgsPackedArray args(6);
      args.add_argument(parts_);
      args.add_argument(dev_vals_[a]);
      args.add_argument(dev_idxs_[a]);
      args.add_argument(args_n_);
      args.add_argument(mt_p);
      args.add_argument(args_nb_);
      TF_RETURN_IF_ERROR(part_pso_->Launch(se::ThreadDim(ntpg, 1, 1),
                                           se::BlockDim(1, rows_, 1), stream,
                                           args));
    }
    // Merge. Grid = (n_blocks, rows, 1), group = (512, 1, 1). The last pass
    // writes to the real output; earlier passes ping-pong through dev[b].
    {
      const se::DeviceAddressBase mv = last ? outv : dev_vals_[b];
      const se::DeviceAddressBase mi = last ? outi : dev_idxs_[b];
      se::KernelArgsPackedArray args(8);
      args.add_argument(parts_);
      args.add_argument(dev_vals_[a]);
      args.add_argument(dev_idxs_[a]);
      args.add_argument(mv);
      args.add_argument(mi);
      args.add_argument(args_n_);
      args.add_argument(mt_p);
      args.add_argument(args_nb_);
      TF_RETURN_IF_ERROR(merge_pso_->Launch(
          se::ThreadDim(kBlockThreads, 1, 1),
          se::BlockDim(static_cast<uint64_t>(n_blocks_), rows_, 1), stream,
          args));
    }
    cur = b;
  }
  return absl::OkStatus();
}

Thunk::BufferUses MetalSortThunk::buffer_uses() const {
  const std::vector<int64_t> dims = {rows_, n_};
  return {BufferUse::Read(data_, ShapeUtil::MakeShape(dtype_, dims)),
          BufferUse::Write(out_vals_, ShapeUtil::MakeShape(dtype_, dims)),
          BufferUse::Write(out_idxs_, ShapeUtil::MakeShape(S32, dims))};
}

absl::StatusOr<ThunkProto> MetalSortThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalSortThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
