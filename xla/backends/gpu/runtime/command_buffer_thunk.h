/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

class CommandBufferThunk : public Thunk {
 public:
  CommandBufferThunk(CommandExecutor commands, ThunkInfo thunk_info,
                     std::unique_ptr<SequentialThunk> thunks = nullptr,
                     bool enable_command_buffers_during_profiling = false,
                     int64_t command_buffer_cache_size = 1);

  const std::unique_ptr<SequentialThunk>& thunks() const { return thunks_; }

  // Returns buffer allocation indices referenced by commands in this thunk.
  absl::Span<const BufferAllocation::Index> allocs_indices() const {
    return commands_.allocs_indices();
  }

  // Walks all commands in this thunk, invoking the callback for each.
  absl::Status WalkCommands(
      absl::FunctionRef<absl::Status(const Command*)> callback) const {
    return commands_.Walk(callback);
  }

  absl::Status Prepare(const PrepareParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  // Return the allocation address that was lazilly allocated inside command
  // buffer. This API is required when the buffers are allocated inside command
  // buffer but will be consumed by non-command buffer operations.
  absl::StatusOr<se::DeviceAddressBase> GetCommandBufferAllocationAddress(
      const ExecuteParams& params, int64_t index);

  BufferUses buffer_uses() const override { return {}; }

  absl::Status WalkNested(Walker callback) override;

  std::string ToString(int indent) const override;

  absl::StatusOr<ThunkProto> ToProto() const override;

  // Returns whether command buffers are enabled during profiling.
  // When this is false, and there's an active profiler session, the thunks will
  // be evaluated as a regular thunk sequence.
  bool IsEnabledDuringProfiling() const {
    return enable_command_buffers_during_profiling_;
  }

  int64_t command_buffer_cache_size() const {
    return command_buffer_cache_size_;
  }

 private:
  // Command buffer specialized for one set of device allocation addresses.
  struct CommandBufferEntry {
    explicit CommandBufferEntry(
        std::unique_ptr<se::CommandBuffer> command_buffer);

    std::unique_ptr<se::CommandBuffer> command_buffer;

    // A manager for an external state attached by commands in a command
    // sequence to a command buffer.
    CommandStateManager state;

    // Mapping from buffer allocation index to the device memory passed at
    // that index to the last call of `commands_.Record(...)` for
    // `command_buffer`. We can just use a vector instead of map because
    // `BufferAllocation::Index` is a unique identifier assigned
    // contiguously and thus can be used as array index.
    //
    // If no device memory addresses changed from a previous call to
    // `Record`, we can skip command buffer update and simply submit it for
    // execution on a stream. All other pieces of information (like thread
    // and block sizes) captured by commands at construction time and do not
    // change.
    std::vector<se::DeviceAddressBase> recorded_allocs;

    // True if persistent allocation information was valid when the command
    // buffer was recorded. We track only validity because the execution
    // parameter contract guarantees that the indices remain unchanged once
    // present.
    bool persistent_allocs_info_was_valid = false;

    // Number of command buffer executions since last update.
    int64_t num_executions = 0;
  };

  enum class CacheLookup {
    kHit,
    kMiss,
    kEviction,
  };

  struct CommandBufferSelection {
    CommandBufferEntry* entry;
    CacheLookup lookup;
    std::vector<se::DeviceAddressBase> recorded_allocs;
    std::vector<BufferAllocation::Index> updated_allocs;
    bool persistent_allocs_info_is_changed;
  };

  // MRU cache of command buffers instantiated on one StreamExecutor.
  struct ExecutorCommandBufferCache {
    explicit ExecutorCommandBufferCache(int64_t capacity);

    absl::StatusOr<CommandBufferSelection> Select(
        se::StreamExecutor* executor, const CommandExecutor& commands,
        const Thunk::ExecuteParams& params)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex);

    void Discard(CommandBufferEntry* entry)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex);

    static bool HasDynamicAllocations(
        const CommandExecutor& commands,
        std::optional<absl::Span<const BufferAllocation::Index>>
            persistent_alloc_indices);

    // se::CommandBuffer is not thread safe. This mutex guards all cache entries
    // so lookup, graph mutation and submission are serialized.
    absl::Mutex mutex;
    const int64_t capacity;
    std::vector<std::unique_ptr<CommandBufferEntry>> entries
        ABSL_GUARDED_BY(mutex);

    int64_t cache_hits ABSL_GUARDED_BY(mutex) = 0;
    int64_t cache_misses ABSL_GUARDED_BY(mutex) = 0;
    int64_t cache_evictions ABSL_GUARDED_BY(mutex) = 0;

    // For GPU backend, NCCL may call cuda-graph un-supported host side API
    // during graph capturing (e.g. cuCtxEnablePeerAccess), this will break XLA
    // cuda graph run. To work around the issue, this PR introduces a warm up
    // iteration for command buffer thunk, during warm up iteration, command
    // buffer thunk are executed through normal thunks. The warm up iteration
    // will do the proper NCCL setup, so later iterations running through
    // command buffer does not need to call NCCL setup APIs.
    bool warmup_done ABSL_GUARDED_BY(mutex) = false;
  };

  // Command buffer thunk owns one command buffer cache for each executor.
  struct State {
    absl::Mutex mutex;
    absl::flat_hash_map<se::StreamExecutor*,
                        std::shared_ptr<ExecutorCommandBufferCache>>
        command_buffer_caches ABSL_GUARDED_BY(mutex);
  };

  // Returns a command buffer cache for `executor` or creates a new one.
  std::shared_ptr<ExecutorCommandBufferCache> GetOrCreateCommandBufferCache(
      se::StreamExecutor* executor);

  // Each individual command buffer allocates state on device (CUDA graph) and
  // it adds up pretty quickly. To prevent OOM errors we proactively evict
  // command buffers from device by clearing command buffer thunk state. We use
  // global state to track all command buffer thunks in a process and coordinate
  // command buffer eviction.
  struct GlobalState;

  // Returns a global state of tracked command buffers thunks.
  static GlobalState* GetGlobalState();

  // Adds command buffer thunk state for tracking.
  static void TrackCommandBuffers(std::weak_ptr<State> state);

  // Evicts all previously instantiated command buffers.
  static void EvictCommandBuffers();

  // Commands executor that initializes command buffers on each stream executor.
  CommandExecutor commands_;

  // Thunk sequence that executes the same commands as in `commands_` but using
  // thunk mechanism. We use it as a fallback mechanism to work around CUPTI
  // bugs that lead to memory corruption when CUPTI traces CUDA graph execution.
  std::unique_ptr<SequentialThunk> thunks_;

  // When true, allows command buffers to be used while profiling active.
  // TODO(b/355487968): Remove this option when validation complete.
  bool enable_command_buffers_during_profiling_;

  // Maximum number of address-specialized command buffers retained per
  // StreamExecutor. Collective command buffers use an effective capacity of 1.
  int64_t command_buffer_cache_size_;

  // Command buffer thunk state allocated in heap to allow global (per-process)
  // management of instantiated command buffers.
  std::shared_ptr<State> state_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_THUNK_H_
