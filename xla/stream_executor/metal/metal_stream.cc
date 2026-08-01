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

#include "xla/stream_executor/metal/metal_stream.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_event.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/metal/metal_runtime.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::metal {
namespace {

void* ReadPackedPointer(void* packed_arg_address) {
  return *static_cast<void**>(packed_arg_address);
}

void FlushProducerThunk(void* ctx, uint64_t value) {
  static_cast<MetalExecutor*>(ctx)->CommitOpenBufferThrough(value);
}

}  // namespace

MetalStream::MetalStream(
    MetalExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority)
    : StreamCommon(executor, priority), executor_(executor) {
  executor_->RegisterStream(this);
}

MetalStream::~MetalStream() {
  executor_->UnregisterStream(this);
  auto status = BlockHostUntilDone();
  if (!status.ok()) {
    LOG(ERROR) << "Metal stream failed while draining: " << status;
  }
}

Stream::PlatformSpecificHandle MetalStream::platform_specific_handle() const {
  return {executor_->command_queue()};
}

absl::Status MetalStream::WaitFor(Stream* other) {
  if (other == this) return absl::OkStatus();
  return other->BlockHostUntilDone();
}

namespace {
bool BatchDbgEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("METAL_KPROF");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}
struct BatchDbg {
  int executes = 0;       // signal-carrying executes this token
  int commits = 0;        // command buffers committed this token
  int dry_commit = 0;     // RecordEvent GPU-went-dry (self-clocked) commits
  int cap_commit = 0;     // RecordEvent kMaxSignalsPerBuffer safety-cap commits
  int foic_commit = 0;    // FlushOpenBufferIfCarrying (cross-stream) commits
  int nowait_commit = 0;  // CommitOpenBufferNoWait commits (token-boundary etc.)
  int w_inbuf = 0;        // WaitFor case1 in-buffer elide
  int w_signaled = 0;     // WaitFor case2 already-signaled elide
  int w_covered = 0;      // WaitFor case3 covered-by-earlier (+commit-through)
  int w_xbuf = 0;         // WaitFor case4 cross-buffer (commit + GPU wait)
  int w_hostsync = 0;     // WaitFor host Synchronize fallback
};
BatchDbg g_bdbg;
}  // namespace

absl::Status MetalStream::RecordEvent(Event* event) {
  auto* metal_event = dynamic_cast<MetalEvent*>(event);
  if (metal_event == nullptr) {
    return absl::InvalidArgumentError("Expected a MetalEvent.");
  }
  if (command_buffer_ != nullptr) {
    const uint64_t value = executor_->NextEventValue();
    metal::EncodeSignalSharedEvent(command_buffer_, executor_->shared_event(),
                                   value);
    pending_signal_high_ = value;
    metal_event->SetCommandBuffer(nullptr);
    metal_event->SetSignal(executor_->shared_event(), value,
                           &FlushProducerThunk, executor_);
    if (BatchDbgEnabled()) ++g_bdbg.executes;
    ++signals_since_commit_;
    const bool gpu_dry =
        metal::SharedEventSignaledValue(executor_->shared_event()) >=
        last_signaled_value_;
    if (gpu_dry || signals_since_commit_ >= kMaxSignalsPerBuffer) {
      void* committed = metal::CommitBatchCommandBuffer(command_buffer_);
      ReleaseObject(command_buffer_);
      command_buffer_ = nullptr;
      if (committed != nullptr) ReleaseObject(committed);
      last_signaled_value_ = value;
      pending_signal_high_ = 0;
      signals_since_commit_ = 0;
      waited_value_high_ = 0;
      if (BatchDbgEnabled()) {
        ++g_bdbg.commits;
        ++(gpu_dry ? g_bdbg.dry_commit : g_bdbg.cap_commit);
      }
    }
  } else {
    const uint64_t high =
        pending_signal_high_ != 0 ? pending_signal_high_ : last_signaled_value_;
    metal_event->SetCommandBuffer(nullptr);
    metal_event->SetSignal(high != 0 ? executor_->shared_event() : nullptr, high,
                           &FlushProducerThunk, executor_);
  }
  return absl::OkStatus();
}

absl::Status MetalStream::WaitFor(Event* event) {
  if (event == nullptr) return absl::OkStatus();
  auto* metal_event = dynamic_cast<MetalEvent*>(event);
  if (metal_event != nullptr && metal_event->shared_event() != nullptr &&
      metal_event->signal_value() != 0) {
    const uint64_t value = metal_event->signal_value();
    if (metal_event->shared_event() == executor_->shared_event() &&
        command_buffer_ != nullptr && value > last_signaled_value_ &&
        value <= pending_signal_high_) {
      if (BatchDbgEnabled()) ++g_bdbg.w_inbuf;
      return absl::OkStatus();
    }
    if (metal_event->shared_event() == executor_->shared_event() &&
        metal::SharedEventSignaledValue(metal_event->shared_event()) >= value) {
      if (BatchDbgEnabled()) ++g_bdbg.w_signaled;
      return absl::OkStatus();
    }
    if (metal_event->shared_event() == executor_->shared_event() &&
        command_buffer_ != nullptr && value <= waited_value_high_) {
      if (BatchDbgEnabled()) ++g_bdbg.w_covered;
      executor_->CommitOpenBufferThrough(value);
      return absl::OkStatus();
    }
    if (BatchDbgEnabled()) ++g_bdbg.w_xbuf;
    executor_->CommitOpenBufferThrough(value);
    EnsureOpenCommandBuffer();
    metal::EncodeWaitForSharedEvent(command_buffer_,
                                    metal_event->shared_event(), value);
    if (metal_event->shared_event() == executor_->shared_event() &&
        value > waited_value_high_) {
      waited_value_high_ = value;
    }
    return absl::OkStatus();
  }
  if (BatchDbgEnabled()) ++g_bdbg.w_hostsync;
  return event->Synchronize();
}

absl::Status MetalStream::Memset32(DeviceAddressBase* location,
                                   uint32_t pattern, uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("Metal Memset32 size is not 4-byte aligned.");
  }
  const uint8_t b = static_cast<uint8_t>(pattern & 0xff);
  const bool byte_uniform = pattern == (static_cast<uint32_t>(b) * 0x01010101u);
  if (byte_uniform) {
    auto alloc = executor_->ResolveAllocation(location->opaque());
    if (alloc.ok()) {
      const uint64_t offset =
          reinterpret_cast<const char*>(location->opaque()) -
          reinterpret_cast<const char*>(alloc->contents);
      EnsureOpenCommandBuffer();
      return metal::EncodeBlitFill(command_buffer_, alloc->buffer, offset, size,
                                   b);
    }
  }
  TF_RETURN_IF_ERROR(BlockHostUntilDone());
  auto* values = static_cast<uint32_t*>(location->opaque());
  for (uint64_t i = 0; i < size / sizeof(uint32_t); ++i) {
    values[i] = pattern;
  }
  return absl::OkStatus();
}

absl::Status MetalStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  if (size == 0) return absl::OkStatus();
  auto alloc = executor_->ResolveAllocation(location->opaque());
  if (alloc.ok()) {
    const uint64_t offset =
        reinterpret_cast<const char*>(location->opaque()) -
        reinterpret_cast<const char*>(alloc->contents);
    EnsureOpenCommandBuffer();
    return metal::EncodeBlitFill(command_buffer_, alloc->buffer, offset, size,
                                 /*value=*/0);
  }
  TF_RETURN_IF_ERROR(BlockHostUntilDone());
  std::memset(location->opaque(), 0, size);
  return absl::OkStatus();
}

absl::Status MetalStream::Memcpy(DeviceAddressBase* device_dst,
                                 const void* host_src, uint64_t size) {
  // Drain first: the GPU may still read this buffer, and host-side readers of
  // staged scalars rely on the copy being complete on return.
  TF_RETURN_IF_ERROR(BlockHostUntilDone());
  std::memcpy(device_dst->opaque(), host_src, size);
  return absl::OkStatus();
}

absl::Status MetalStream::Memcpy(void* host_dst,
                                 const DeviceAddressBase& device_src,
                                 uint64_t size) {
  TF_RETURN_IF_ERROR(BlockHostUntilDone());
  std::memcpy(host_dst, device_src.opaque(), size);
  return absl::OkStatus();
}

absl::Status MetalStream::Memcpy(DeviceAddressBase* device_dst,
                                 const DeviceAddressBase& device_src,
                                 uint64_t size) {
  if (size == 0) return absl::OkStatus();
  auto dst = executor_->ResolveAllocation(device_dst->opaque());
  auto src = executor_->ResolveAllocation(device_src.opaque());
  if (dst.ok() && src.ok()) {
    const uint64_t dst_off =
        reinterpret_cast<const char*>(device_dst->opaque()) -
        reinterpret_cast<const char*>(dst->contents);
    const uint64_t src_off =
        reinterpret_cast<const char*>(device_src.opaque()) -
        reinterpret_cast<const char*>(src->contents);
    const bool same_buf = dst->buffer == src->buffer;
    const bool overlap =
        same_buf && dst_off < src_off + size && src_off < dst_off + size;
    if (!overlap) {
      EnsureOpenCommandBuffer();
      return metal::EncodeBlitCopy(command_buffer_, dst->buffer, dst_off,
                                   src->buffer, src_off, size);
    }
  }
  TF_RETURN_IF_ERROR(BlockHostUntilDone());
  std::memmove(device_dst->opaque(), device_src.opaque(), size);
  return absl::OkStatus();
}

absl::Status MetalStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  auto run = [cb = std::move(callback)]() mutable {
    absl::Status status = std::move(cb)();
    if (!status.ok()) {
      LOG(ERROR) << "Metal host callback returned an error: " << status;
    }
  };

  void* shared_event = executor_->shared_event();
  void* listener = executor_->shared_event_listener();
  const uint64_t value =
      pending_signal_high_ != 0 ? pending_signal_high_ : last_signaled_value_;
  if (shared_event != nullptr && listener != nullptr && value != 0) {
    metal::NotifySharedEventListener(listener, shared_event, value,
                                     std::move(run));
    return absl::OkStatus();
  }

  if (command_buffer_ != nullptr) {
    void* batch = command_buffer_;
    command_buffer_ = nullptr;
    pending_signal_high_ = 0;
    signals_since_commit_ = 0;
    waited_value_high_ = 0;
    metal::CommitBatchCommandBufferWithCompletion(batch, std::move(run));
    ReleaseObject(batch);
    return absl::OkStatus();
  }
  run();
  return absl::OkStatus();
}

absl::Status MetalStream::BlockHostUntilDone() {
  if (command_buffer_ == nullptr) {
    return absl::OkStatus();
  }
  void* committed = metal::CommitBatchCommandBuffer(command_buffer_);
  ReleaseObject(command_buffer_);
  command_buffer_ = nullptr;
  if (pending_signal_high_ != 0) last_signaled_value_ = pending_signal_high_;
  pending_signal_high_ = 0;
  signals_since_commit_ = 0;
  waited_value_high_ = 0;
  absl::Status status = WaitUntilCompleted(committed);
  ReleaseObject(committed);
  return status;
}

absl::Status MetalStream::FlushBatchedWork() {
  if (BatchDbgEnabled()) {
    static int tok = 0;
    if ((tok++ % 32) == 0) {
      LOG(INFO) << "METAL_KPROF batch/token: executes=" << g_bdbg.executes
                << " commits=" << g_bdbg.commits
                << " (dry=" << g_bdbg.dry_commit << " cap=" << g_bdbg.cap_commit
                << " foic=" << g_bdbg.foic_commit
                << " nowait=" << g_bdbg.nowait_commit << ")"
                << " | WaitFor inbuf=" << g_bdbg.w_inbuf
                << " signaled=" << g_bdbg.w_signaled
                << " covered=" << g_bdbg.w_covered << " xbuf=" << g_bdbg.w_xbuf
                << " hostsync=" << g_bdbg.w_hostsync;
    }
    g_bdbg = BatchDbg{};
  }
  CommitOpenBufferNoWait();
  return absl::OkStatus();
}

absl::Status MetalStream::CommitBatchedWorkNoWait() {
  CommitOpenBufferNoWait();
  return absl::OkStatus();
}

void MetalStream::CommitOpenBufferNoWait() {
  if (command_buffer_ == nullptr) return;
  void* committed = metal::CommitBatchCommandBuffer(command_buffer_);
  ReleaseObject(command_buffer_);
  command_buffer_ = nullptr;
  if (pending_signal_high_ != 0) last_signaled_value_ = pending_signal_high_;
  pending_signal_high_ = 0;
  signals_since_commit_ = 0;
  waited_value_high_ = 0;
  if (committed != nullptr) ReleaseObject(committed);
  if (BatchDbgEnabled()) { ++g_bdbg.commits; ++g_bdbg.nowait_commit; }
}

void MetalStream::FlushOpenBufferIfCarrying(uint64_t value) {
  if (command_buffer_ == nullptr) return;
  if (pending_signal_high_ < value || value <= last_signaled_value_) return;
  void* committed = metal::CommitBatchCommandBuffer(command_buffer_);
  ReleaseObject(command_buffer_);
  command_buffer_ = nullptr;
  if (committed != nullptr) ReleaseObject(committed);
  last_signaled_value_ = pending_signal_high_;
  pending_signal_high_ = 0;
  signals_since_commit_ = 0;  // match FlushBatchedWork: keep cadence accounting consistent.
  waited_value_high_ = 0;
  if (BatchDbgEnabled()) { ++g_bdbg.commits; ++g_bdbg.foic_commit; }
}

absl::Status MetalStream::LaunchMetalKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* pipeline,
    void* function, bool use_argument_buffer, absl::string_view name,
    void** args, absl::Span<const KernelArgumentMetadata> arg_metadata,
    int64_t shmem_bytes, bool use_pdl) {
  if (cluster_dims.has_value()) {
    return absl::UnimplementedError("Metal cluster launches are not supported.");
  }
  if (use_pdl) {
    return absl::UnimplementedError(
        "Metal programmatic dependent launch is not supported.");
  }
  executor_->FlushResidency();

  std::vector<MetalKernelArgument> arguments;
  if (args != nullptr) {
    auto** packed_arg_addresses = reinterpret_cast<void**>(args[0]);
    size_t arg_count = *reinterpret_cast<size_t*>(args[1]);
    if (arg_metadata.size() != arg_count) {
      return absl::InternalError(absl::StrCat(
          "Metal kernel argument-metadata count ", arg_metadata.size(),
          " does not match argument count ", arg_count, "."));
    }
    arguments.reserve(arg_count);
    for (size_t i = 0; i < arg_count; ++i) {
      const int64_t arg_size = arg_metadata[i].size;
      if (arg_size <= 0) {
        return absl::InternalError(absl::StrCat(
            "Metal kernel argument ", i, " has invalid size ", arg_size,
            "."));
      }

      if (arg_metadata[i].kind == KernelArgumentMetadata::Kind::kDeviceAddress) {
        if (arg_size != sizeof(void*)) {
          return absl::InternalError(absl::StrCat(
              "Metal device-address argument ", i, " has size ", arg_size,
              ", expected ", sizeof(void*), "."));
        }
        void* value = ReadPackedPointer(packed_arg_addresses[i]);
        if (value == nullptr) {
          arguments.push_back(MetalKernelArgument{
              MetalKernelArgument::Kind::kBuffer, nullptr, 0, nullptr, 0});
          continue;
        }
        auto allocation = executor_->ResolveAllocation(value);
        if (allocation.ok()) {
          auto base = reinterpret_cast<uintptr_t>(allocation->contents);
          auto ptr = reinterpret_cast<uintptr_t>(value);
          arguments.push_back(MetalKernelArgument{
              MetalKernelArgument::Kind::kBuffer, allocation->buffer,
              static_cast<uint64_t>(ptr - base), nullptr, 0});
          continue;
        }
        return absl::InternalError(absl::StrCat(
            "Metal kernel device-address argument ", i,
            " does not resolve to an executor allocation."));
      }
      arguments.push_back(MetalKernelArgument{
          MetalKernelArgument::Kind::kBytes, nullptr, 0,
          packed_arg_addresses[i], static_cast<size_t>(arg_size)});
    }
  }

  EnsureOpenCommandBuffer();
  return metal::EncodeKernel(command_buffer_, pipeline, function,
                             use_argument_buffer, arguments, name, thread_dims,
                             block_dims, shmem_bytes);
}

void MetalStream::EnsureOpenCommandBuffer() {
  if (command_buffer_ == nullptr) {
    command_buffer_ = metal::NewBatchCommandBuffer(executor_->command_queue());
  }
}

}  // namespace stream_executor::metal
