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

#include "xla/backends/gpu/runtime/metal_topk_thunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/topk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
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

struct ArgsHost {
  uint32_t n;
  uint32_t k;
  uint32_t cap;
  uint32_t pad;
};

int64_t RoundK(int64_t k) {
  return std::min<int64_t>(std::max<int64_t>(absl::bit_ceil(static_cast<uint64_t>(k)), 1), 32);
}

}  // namespace

MetalTopKThunk::MetalTopKThunk(ThunkInfo thunk_info, BufferAllocation::Slice data,
                               BufferAllocation::Slice top_vals,
                               BufferAllocation::Slice top_idxs,
                               PrimitiveType dtype, int64_t batch_size,
                               int64_t num_elements, int64_t k)
    : Thunk(Kind::kCustomKernel, std::move(thunk_info)),
      data_(data),
      top_vals_(top_vals),
      top_idxs_(top_idxs),
      dtype_(dtype),
      batch_(batch_size),
      n_(num_elements),
      k_(k),
      k_rounded_(RoundK(k)) {}

void MetalTopKThunk::Prewarm(se::StreamExecutor* executor, PrimitiveType dtype,
                             int64_t k) {
  auto lib = CompileMetalSourceToMetallibCached(get_topk());
  if (!lib.ok()) return;
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  const bool b16 = (dtype != F32);
  const int64_t k_rounded = RoundK(k);
  std::vector<se::metal::MetalFunctionConstant> no_fc;
  auto load = [&](const std::string& name, int arity) {
    metal_exec->LoadKernelWithConstants(*lib, name, arity, no_fc).IgnoreError();
  };
  // Same four kernels (and names) EnsureLoaded loads — warming the metallib
  // cache + each PSO. No scratch is allocated here; Ensure does that at execute
  // (cheap, not the first-request bottleneck).
  load(b16 ? "radix_hist16" : "radix_hist32", 3);
  load("radix_scan", 3);
  load(b16 ? "radix_gather16" : "radix_gather32", 6);
  load(absl::StrCat("radix_sel", b16 ? "16" : "32", "_k", k_rounded), 6);
}

absl::Status MetalTopKThunk::Ensure(se::StreamExecutor* executor) {
  if (executor_ == executor && hist_.opaque() != nullptr) return absl::OkStatus();
  executor_ = executor;
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  const bool b16 = (dtype_ != F32);  // BF16/F16 -> 16-bit path
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib, CompileMetalSourceToMetallibCached(get_topk()));
  std::vector<se::metal::MetalFunctionConstant> no_fc;
  auto load = [&](const std::string& name, int arity)
      -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
    return metal_exec->LoadKernelWithConstants(lib, name, arity, no_fc);
  };
  TF_ASSIGN_OR_RETURN(hist_pso_, load(b16 ? "radix_hist16" : "radix_hist32", 3));
  TF_ASSIGN_OR_RETURN(scan_pso_, load("radix_scan", 3));
  TF_ASSIGN_OR_RETURN(gather_pso_, load(b16 ? "radix_gather16" : "radix_gather32", 6));
  TF_ASSIGN_OR_RETURN(
      select_pso_, load(absl::StrCat("radix_sel", b16 ? "16" : "32", "_k", k_rounded_), 6));

  hist_ = executor->Allocate(static_cast<uint64_t>(batch_) * 16384 * 4, 0);
  thresh_ = executor->Allocate(static_cast<uint64_t>(batch_) * 4, 0);
  ccount_ = executor->Allocate(static_cast<uint64_t>(batch_) * 2 * 4, 0);
  cok_ = executor->Allocate(static_cast<uint64_t>(batch_) * kCap * 4, 0);
  cix_ = executor->Allocate(static_cast<uint64_t>(batch_) * kCap * 4, 0);
  args_ = executor->Allocate(sizeof(ArgsHost), 0);
  if (hist_.opaque() == nullptr || thresh_.opaque() == nullptr ||
      ccount_.opaque() == nullptr || cok_.opaque() == nullptr ||
      cix_.opaque() == nullptr || args_.opaque() == nullptr) {
    return absl::ResourceExhaustedError("Metal TopK radix: scratch alloc failed.");
  }
  // hist + ccount must start zeroed (the histogram does atomic_add; the scan and
  // select self-clear them after each token, so this is a one-time setup).
  std::vector<char> zeros(static_cast<size_t>(batch_) * 16384 * 4, 0);
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&hist_, zeros.data(),
                                                 static_cast<uint64_t>(batch_) * 16384 * 4));
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&ccount_, zeros.data(),
                                                 static_cast<uint64_t>(batch_) * 2 * 4));
  ArgsHost a = {static_cast<uint32_t>(n_), static_cast<uint32_t>(k_),
                static_cast<uint32_t>(kCap), 0};
  TF_RETURN_IF_ERROR(executor->SynchronousMemcpy(&args_, &a, sizeof(a)));
  return absl::OkStatus();
}

absl::Status MetalTopKThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;

  absl::MutexLock lock(&mu_);
  if (executor_ != executor || hist_.opaque() == nullptr) {
    TF_RETURN_IF_ERROR(Ensure(executor));
  }
  const se::DeviceAddressBase data = allocs.GetDeviceAddress(data_);
  const se::DeviceAddressBase outv = allocs.GetDeviceAddress(top_vals_);
  const se::DeviceAddressBase outi = allocs.GetDeviceAddress(top_idxs_);
  const int64_t B = batch_;
  const int kGx = 64;  // x-threadgroups for the histogram/gather grid-stride

  // Pass 1: histogram. Grid (kGx, B, 1) x 256 threads.
  {
    se::KernelArgsPackedArray args(3);
    args.add_argument(data);
    args.add_argument(args_);
    args.add_argument(hist_);
    TF_RETURN_IF_ERROR(hist_pso_->Launch(se::ThreadDim(256, 1, 1),
                                     se::BlockDim(kGx, B, 1), stream, args));
  }
  // Pass 2: scan. Grid (1, B, 1) x 1024 threads.
  {
    se::KernelArgsPackedArray args(3);
    args.add_argument(hist_);
    args.add_argument(args_);
    args.add_argument(thresh_);
    TF_RETURN_IF_ERROR(scan_pso_->Launch(se::ThreadDim(1024, 1, 1),
                                     se::BlockDim(1, B, 1), stream, args));
  }
  // Pass 3: gather. Grid (kGx, B, 1) x 256 threads.
  {
    se::KernelArgsPackedArray args(6);
    args.add_argument(data);
    args.add_argument(args_);
    args.add_argument(thresh_);
    args.add_argument(ccount_);
    args.add_argument(cok_);
    args.add_argument(cix_);
    TF_RETURN_IF_ERROR(gather_pso_->Launch(se::ThreadDim(256, 1, 1),
                                       se::BlockDim(kGx, B, 1), stream, args));
  }
  // Pass 4: select. Grid (1, B, 1) x 256 threads, 16KB scratch.
  {
    se::KernelArgsPackedArray args(6);
    args.add_argument(cok_);
    args.add_argument(cix_);
    args.add_argument(ccount_);
    args.add_argument(args_);
    args.add_argument(outv);
    args.add_argument(outi);
    args.add_shared_bytes(16384);
    TF_RETURN_IF_ERROR(select_pso_->Launch(se::ThreadDim(256, 1, 1),
                                       se::BlockDim(1, B, 1), stream, args));
  }
  return absl::OkStatus();
}

Thunk::BufferUses MetalTopKThunk::buffer_uses() const {
  std::vector<int64_t> dims;
  if (batch_ > 1) dims = {batch_, n_};
  else dims = {n_};
  std::vector<int64_t> odims = (batch_ > 1) ? std::vector<int64_t>{batch_, k_}
                                            : std::vector<int64_t>{k_};
  return {BufferUse::Read(data_, ShapeUtil::MakeShape(dtype_, dims)),
          BufferUse::Write(top_vals_, ShapeUtil::MakeShape(dtype_, odims)),
          BufferUse::Write(top_idxs_, ShapeUtil::MakeShape(S32, odims))};
}

absl::StatusOr<ThunkProto> MetalTopKThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalTopKThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
