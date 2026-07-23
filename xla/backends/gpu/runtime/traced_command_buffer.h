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

#ifndef XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_BUFFER_H_
#define XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_BUFFER_H_

#include <cstdint>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

// A cache for traced command buffers that will re-trace on change in buffer
// allocations that are relevant for `buffers` passed to constructor. We use a
// very simple most-recently-used cache of traced command buffers as in practice
// subsequent calls to XLA executable tend to reuse the same allocations.
class TracedCommandBuffer : public CommandState {
 public:
  explicit TracedCommandBuffer(const Command* trace_cmd,
                               Command::BufferUses buffers,
                               int64_t capacity = 16);

  // Returns cached command buffer traced using the same buffer addresses or
  // traces and caches a new command buffer using user provided callback.
  absl::StatusOr<se::CommandBuffer*> GetOrTraceCommandBuffer(
      const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
      se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
      se::StreamPriority priority = se::StreamPriority::Default);

  bool RecordedChildIsCurrent(const se::CommandBuffer::Command* command,
                              se::CommandBuffer* nested) const;
  void SetRecordedChild(const se::CommandBuffer::Command* command,
                        se::CommandBuffer* nested);

 private:
  std::vector<BufferAllocation::Index> allocs_indices_;

  struct AddressKey {
    absl::InlinedVector<se::DeviceAddressBase, 32> allocs;

    bool operator==(const AddressKey& other) const {
      return allocs == other.allocs;
    }

    template <typename H>
    friend H AbslHashValue(H h, const AddressKey& key) {
      h = H::combine(std::move(h), key.allocs.size());
      for (const se::DeviceAddressBase& alloc : key.allocs) {
        h = H::combine(std::move(h), alloc);
      }
      return h;
    }
  };

  struct Entry {
    AddressKey key;
    std::unique_ptr<se::CommandBuffer> command_buffer;
  };

  using EntryList = std::list<Entry>;

  const Command* trace_cmd_;
  size_t capacity_;
  EntryList entries_;
  absl::flat_hash_map<AddressKey, EntryList::iterator> entries_by_key_;
  absl::flat_hash_map<const se::CommandBuffer::Command*, se::CommandBuffer*>
      recorded_children_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_BUFFER_H_
