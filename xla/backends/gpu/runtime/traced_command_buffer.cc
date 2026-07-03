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

#include "xla/backends/gpu/runtime/traced_command_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"

namespace xla::gpu {

TracedCommandBuffer::TracedCommandBuffer(const Command* trace_cmd,
                                         Command::BufferUses buffers,
                                         int64_t capacity)
    : trace_cmd_(trace_cmd), capacity_(static_cast<size_t>(capacity)) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";  // NOLINT
  // Keep allocation indices sorted to make the address key stable and compact.
  allocs_indices_.reserve(buffers.size());
  for (auto& buffer : buffers) {
    allocs_indices_.push_back(buffer.slice().index());
  }
  absl::c_sort(allocs_indices_);
  allocs_indices_.erase(
      std::unique(allocs_indices_.begin(), allocs_indices_.end()),
      allocs_indices_.end());
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::GetOrTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
    se::StreamPriority priority) {
  // Collect memory addresses for relevant allocations.
  AddressKey key;
  key.allocs.reserve(allocs_indices_.size());
  for (auto& index : allocs_indices_) {
    key.allocs.emplace_back(buffer_allocation->GetDeviceAddress(index));
  }

  if (auto it = entries_by_key_.find(key);
      ABSL_PREDICT_TRUE(it != entries_by_key_.end())) {
    auto entry = it->second;
    if (capacity_ <= 64) {
      entries_.splice(entries_.begin(), entries_, it->second);
      it->second = entries_.begin();
      entry = entries_.begin();
    }
    VLOG(6) << "Command buffer trace cache hit for command "
            << trace_cmd_->ToString(0);
    return entry->command_buffer.get();
  }

  if (entries_.size() == capacity_) {
    entries_by_key_.erase(entries_.back().key);
    entries_.pop_back();
  }

  Entry entry;
  entry.key = std::move(key);
  TF_ASSIGN_OR_RETURN(
      entry.command_buffer,
      se::TraceCommandBufferFactory::Create(executor, stream, trace));
  if (priority != se::StreamPriority::Default) {
    TF_RETURN_IF_ERROR(entry.command_buffer->SetPriority(priority));
  }

  entries_.push_front(std::move(entry));
  entries_by_key_.emplace(entries_.front().key, entries_.begin());

  VLOG(6) << "Command buffer trace cache create new item for command "
          << trace_cmd_->ToString(0);
  return entries_.front().command_buffer.get();
}

bool TracedCommandBuffer::RecordedChildIsCurrent(
    const se::CommandBuffer::Command* command,
    se::CommandBuffer* nested) const {
  auto it = recorded_children_.find(command);
  return it != recorded_children_.end() && it->second == nested;
}

void TracedCommandBuffer::SetRecordedChild(
    const se::CommandBuffer::Command* command, se::CommandBuffer* nested) {
  recorded_children_[command] = nested;
}

}  // namespace xla::gpu
