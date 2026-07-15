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

#include "xla/stream_executor/musa/musa_stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_runtime.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

MusaStream::MusaStream(
    StreamExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority,
    void* stream_handle)
    : StreamCommon(executor, priority),
      executor_(executor),
      stream_handle_(stream_handle) {}

absl::StatusOr<std::unique_ptr<MusaStream>> MusaStream::Create(
    StreamExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  auto stream = MusaRuntime::Get()->StreamCreate();
  if (!stream.ok()) return stream.status();
  return std::unique_ptr<MusaStream>(
      new MusaStream(executor, priority, *stream));
}

MusaStream::~MusaStream() {
  (void)MusaRuntime::Get()->StreamSynchronize(stream_handle_);
  parent()->DeallocateStream(this);
  (void)MusaRuntime::Get()->StreamDestroy(stream_handle_);
  stream_handle_ = nullptr;
}

absl::Status MusaStream::WaitFor(Stream* other) {
  return other->BlockHostUntilDone();
}

absl::Status MusaStream::RecordEvent(Event* event) {
  auto* musa_event = dynamic_cast<MusaEvent*>(event);
  if (musa_event == nullptr) {
    return absl::InvalidArgumentError("Expected a MUSA event.");
  }
  return MusaRuntime::Get()->EventRecord(musa_event->handle(), stream_handle_);
}

absl::Status MusaStream::WaitFor(Event* event) {
  return event->WaitForEventOnExternalStream(
      reinterpret_cast<std::intptr_t>(stream_handle_));
}

absl::Status MusaStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const void* host_src, uint64_t size) {
  return MusaRuntime::Get()->MemcpyAsync(gpu_dst->opaque(), host_src, size,
                                         MusaMemcpyKind::kHostToDevice,
                                         stream_handle_);
}

absl::Status MusaStream::Memcpy(void* host_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  return MusaRuntime::Get()->MemcpyAsync(host_dst, gpu_src.opaque(), size,
                                         MusaMemcpyKind::kDeviceToHost,
                                         stream_handle_);
}

absl::Status MusaStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  return MusaRuntime::Get()->MemcpyAsync(gpu_dst->opaque(), gpu_src.opaque(),
                                         size,
                                         MusaMemcpyKind::kDeviceToDevice,
                                         stream_handle_);
}

absl::Status MusaStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  return MusaRuntime::Get()->MemsetAsync(location->opaque(), 0, size,
                                         stream_handle_);
}

absl::Status MusaStream::Memset32(DeviceAddressBase* location,
                                  uint32_t pattern, uint64_t size) {
  if (pattern == 0) {
    return MemZero(location, size);
  }
  return absl::UnimplementedError("MUSA Memset32 is not implemented yet.");
}

absl::Status MusaStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  absl::Status status = BlockHostUntilDone();
  if (!status.ok()) return status;
  return std::move(callback)();
}

absl::Status MusaStream::BlockHostUntilDone() {
  return MusaRuntime::Get()->StreamSynchronize(stream_handle_);
}

absl::Status MusaStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes, bool use_pdl) {
  return absl::UnimplementedError(
      "MUSA kernel launch is not implemented in PJRT GPU v1.");
}

}  // namespace stream_executor::musa
