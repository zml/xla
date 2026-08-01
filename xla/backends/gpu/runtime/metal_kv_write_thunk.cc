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

#include "xla/backends/gpu/runtime/metal_kv_write_thunk.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/kv_write.h"
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

MetalKvWriteThunk::MetalKvWriteThunk(
    ThunkInfo thunk_info, BufferAllocation::Slice k_cache, Shape k_cache_shape,
    BufferAllocation::Slice k_new, Shape k_new_shape,
    BufferAllocation::Slice v_cache, Shape v_cache_shape,
    BufferAllocation::Slice v_new, Shape v_new_shape,
    BufferAllocation::Slice slot, Shape slot_shape,
    BufferAllocation::Slice pos, Shape pos_shape,
    BufferAllocation::Slice freq, Shape freq_shape, int64_t num_slots,
    int64_t kv_heads, int64_t head_dim)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      k_cache_(k_cache),
      k_new_(k_new),
      v_cache_(v_cache),
      v_new_(v_new),
      slot_(slot),
      pos_(pos),
      freq_(freq),
      k_cache_shape_(std::move(k_cache_shape)),
      k_new_shape_(std::move(k_new_shape)),
      v_cache_shape_(std::move(v_cache_shape)),
      v_new_shape_(std::move(v_new_shape)),
      slot_shape_(std::move(slot_shape)),
      pos_shape_(std::move(pos_shape)),
      freq_shape_(std::move(freq_shape)),
      num_slots_(num_slots),
      kv_heads_(kv_heads),
      head_dim_(head_dim) {}

void MetalKvWriteThunk::Prewarm(se::StreamExecutor* executor,
                                int64_t num_slots, int64_t kv_heads,
                                int64_t head_dim) {
  const std::string source =
      absl::Substitute(get_kv_write(), kv_heads, head_dim, num_slots);
  auto lib = CompileMetalSourceToMetallibCached(source);
  if (!lib.ok()) return;
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  metal_exec->LoadKernelWithConstants(*lib, "kv_write", /*arity=*/7, {})
      .IgnoreError();
}

absl::Status MetalKvWriteThunk::EnsureLoaded(se::StreamExecutor* executor) {
  if (kernel_ != nullptr) return absl::OkStatus();
  const std::string source =
      absl::Substitute(get_kv_write(), kv_heads_, head_dim_, num_slots_);
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                      CompileMetalSourceToMetallibCached(source));
  auto* metal_exec = static_cast<se::metal::MetalExecutor*>(executor);
  TF_ASSIGN_OR_RETURN(
      kernel_, metal_exec->LoadKernelWithConstants(lib, "kv_write",
                                                   /*arity=*/7, {}));
  return absl::OkStatus();
}

absl::Status MetalKvWriteThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::Kernel* kernel = nullptr;
  {
    // kernel_ is write-once; the raw pointer stays valid after unlock.
    absl::MutexLock lock(&mu_);
    TF_RETURN_IF_ERROR(EnsureLoaded(stream->parent()));
    kernel = kernel_.get();
  }
  const BufferAllocations& allocs = *params.buffer_allocations;
  se::KernelArgsPackedArray args(/*num_args=*/7);
  args.add_argument(allocs.GetDeviceAddress(k_cache_));  // 0
  args.add_argument(allocs.GetDeviceAddress(k_new_));    // 1
  args.add_argument(allocs.GetDeviceAddress(v_cache_));  // 2
  args.add_argument(allocs.GetDeviceAddress(v_new_));    // 3
  args.add_argument(allocs.GetDeviceAddress(slot_));     // 4
  args.add_argument(allocs.GetDeviceAddress(pos_));      // 5
  args.add_argument(allocs.GetDeviceAddress(freq_));     // 6
  const int64_t elems = kv_heads_ * head_dim_;
  const int64_t threads = 256;
  const int64_t blocks = (elems + threads - 1) / threads;
  return kernel->Launch(se::ThreadDim(threads, 1, 1),
                        se::BlockDim(blocks, 1, 1), stream, args);
}

Thunk::BufferUses MetalKvWriteThunk::buffer_uses() const {
  return {
      BufferUse::Write(k_cache_, k_cache_shape_),
      BufferUse::Read(k_new_, k_new_shape_),
      BufferUse::Write(v_cache_, v_cache_shape_),
      BufferUse::Read(v_new_, v_new_shape_),
      BufferUse::Read(slot_, slot_shape_),
      BufferUse::Read(pos_, pos_shape_),
      BufferUse::Read(freq_, freq_shape_),
  };
}

absl::StatusOr<ThunkProto> MetalKvWriteThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalKvWriteThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
