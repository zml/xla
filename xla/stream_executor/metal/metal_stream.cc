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
#include <cstring>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

}  // namespace

MetalStream::MetalStream(
    MetalExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority)
    : StreamCommon(executor, priority), executor_(executor) {}

MetalStream::~MetalStream() {
  auto status = BlockHostUntilDone();
  if (!status.ok()) {
    LOG(ERROR) << "Metal stream failed while draining: " << status;
  }
  if (last_command_buffer_ != nullptr) {
    ReleaseObject(last_command_buffer_);
  }
}

Stream::PlatformSpecificHandle MetalStream::platform_specific_handle() const {
  return {executor_->command_queue()};
}

absl::Status MetalStream::WaitFor(Stream* other) {
  if (other == this) return absl::OkStatus();
  return other->BlockHostUntilDone();
}

absl::Status MetalStream::RecordEvent(Event* event) {
  auto* metal_event = dynamic_cast<MetalEvent*>(event);
  if (metal_event == nullptr) {
    return absl::InvalidArgumentError("Expected a MetalEvent.");
  }
  metal_event->SetCommandBuffer(RetainObject(last_command_buffer_));
  return absl::OkStatus();
}

absl::Status MetalStream::WaitFor(Event* event) {
  if (event == nullptr) return absl::OkStatus();
  return event->Synchronize();
}

absl::Status MetalStream::Memset32(DeviceAddressBase* location,
                                   uint32_t pattern, uint64_t size) {
  TF_RETURN_IF_ERROR(BlockHostUntilDone());
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("Metal Memset32 size is not 4-byte aligned.");
  }
  auto* values = static_cast<uint32_t*>(location->opaque());
  for (uint64_t i = 0; i < size / sizeof(uint32_t); ++i) {
    values[i] = pattern;
  }
  return absl::OkStatus();
}

absl::Status MetalStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  TF_RETURN_IF_ERROR(BlockHostUntilDone());
  std::memset(location->opaque(), 0, size);
  return absl::OkStatus();
}

absl::Status MetalStream::Memcpy(DeviceAddressBase* device_dst,
                                 const void* host_src, uint64_t size) {
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
  TF_RETURN_IF_ERROR(BlockHostUntilDone());
  std::memmove(device_dst->opaque(), device_src.opaque(), size);
  return absl::OkStatus();
}

absl::Status MetalStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  TF_RETURN_IF_ERROR(BlockHostUntilDone());
  return std::move(callback)();
}

absl::Status MetalStream::BlockHostUntilDone() {
  if (last_command_buffer_ == nullptr) {
    return absl::OkStatus();
  }
  absl::Status status = WaitUntilCompleted(last_command_buffer_);
  ReleaseObject(last_command_buffer_);
  last_command_buffer_ = nullptr;
  return status;
}

absl::Status MetalStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes, bool use_pdl) {
  if (cluster_dims.has_value()) {
    return absl::UnimplementedError("Metal cluster launches are not supported.");
  }
  if (use_pdl) {
    return absl::UnimplementedError(
        "Metal programmatic dependent launch is not supported.");
  }

  std::vector<MetalKernelArgument> arguments;
  if (args != nullptr) {
    auto** packed_arg_addresses = reinterpret_cast<void**>(args[0]);
    size_t arg_count = *reinterpret_cast<size_t*>(args[1]);
    arguments.reserve(arg_count);
    for (size_t i = 0; i < arg_count; ++i) {
      void* value = ReadPackedPointer(packed_arg_addresses[i]);
      auto allocation = executor_->ResolveAllocation(value);
      if (allocation.ok()) {
        auto base = reinterpret_cast<uintptr_t>(allocation->contents);
        auto ptr = reinterpret_cast<uintptr_t>(value);
        arguments.push_back(MetalKernelArgument{
            allocation->buffer, static_cast<uint64_t>(ptr - base), nullptr, 0});
      } else {
        arguments.push_back(MetalKernelArgument{
            nullptr, 0, packed_arg_addresses[i], sizeof(uint64_t)});
      }
    }
  }

  TF_ASSIGN_OR_RETURN(
      void* command_buffer,
      metal::Launch(executor_->command_queue(), function, arguments,
                    thread_dims, block_dims, shmem_bytes));
  SetLastCommandBuffer(command_buffer);
  return absl::OkStatus();
}

void MetalStream::SetLastCommandBuffer(void* command_buffer) {
  if (last_command_buffer_ != nullptr) {
    ReleaseObject(last_command_buffer_);
  }
  last_command_buffer_ = command_buffer;
}

}  // namespace stream_executor::metal
