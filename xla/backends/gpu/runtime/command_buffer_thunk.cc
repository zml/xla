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

#include "xla/backends/gpu/runtime/command_buffer_thunk.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::gpu {

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

//===----------------------------------------------------------------------===//
// CommandBufferThunk
//===----------------------------------------------------------------------===//

CommandBufferThunk::CommandBufferEntry::CommandBufferEntry(
    std::unique_ptr<se::CommandBuffer> command_buffer)
    : command_buffer(std::move(command_buffer)) {}

CommandBufferThunk::ExecutorCommandBufferCache::ExecutorCommandBufferCache(
    int64_t capacity)
    : capacity(capacity) {
  CHECK_GT(capacity, 0);
}

bool CommandBufferThunk::ExecutorCommandBufferCache::HasDynamicAllocations(
    const CommandExecutor& commands,
    std::optional<absl::Span<const BufferAllocation::Index>>
        persistent_alloc_indices) {
  if (!persistent_alloc_indices.has_value()) {
    return true;
  }

  DCHECK(absl::c_is_sorted(commands.allocs_indices()));
  DCHECK(absl::c_is_sorted(*persistent_alloc_indices));
  return !absl::c_includes(*persistent_alloc_indices,
                           commands.allocs_indices());
}

absl::StatusOr<CommandBufferThunk::CommandBufferSelection>
CommandBufferThunk::ExecutorCommandBufferCache::Select(
    se::StreamExecutor* executor, const CommandExecutor& commands,
    const Thunk::ExecuteParams& params) {
  std::vector<BufferAllocation::Index> dynamic_alloc_indices;
  absl::Span<const BufferAllocation::Index> allocs_to_check =
      commands.allocs_indices();
  if (const auto& persistent_alloc_indices = params.persistent_alloc_indices) {
    DCHECK(absl::c_is_sorted(commands.allocs_indices()));
    DCHECK(absl::c_is_sorted(*persistent_alloc_indices));
    absl::c_set_difference(commands.allocs_indices(), *persistent_alloc_indices,
                           std::back_inserter(dynamic_alloc_indices));
    allocs_to_check = dynamic_alloc_indices;
  }

  std::vector<se::DeviceAddressBase> current_allocs;
  for (BufferAllocation::Index index : allocs_to_check) {
    if (current_allocs.size() <= index) {
      current_allocs.resize(index + 1);
    }
    current_allocs[index] = params.buffer_allocations->GetDeviceAddress(index);
  }

  auto move_to_front = [&](size_t index) -> CommandBufferEntry* {
    if (index != 0) {
      std::unique_ptr<CommandBufferEntry> entry = std::move(entries[index]);
      do {
        entries[index] = std::move(entries[index - 1]);
      } while (--index > 0);
      entries[0] = std::move(entry);
    }
    return entries[0].get();
  };

  bool persistent_allocs_info_is_valid =
      params.persistent_alloc_indices.has_value();
  for (size_t i = 0; i < entries.size(); ++i) {
    if (entries[i]->persistent_allocs_info_was_valid ==
            persistent_allocs_info_is_valid &&
        absl::c_equal(entries[i]->recorded_allocs, current_allocs)) {
      ++cache_hits;
      CommandBufferEntry* entry = move_to_front(i);
      XLA_VLOG_DEVICE(3, executor->device_ordinal())
          << "Command buffer cache hit"
          << "; cache=" << this << "; entry=" << entry
          << "; entries=" << entries.size() << "; capacity=" << capacity
          << "; hits=" << cache_hits << "; misses=" << cache_misses
          << "; evictions=" << cache_evictions;
      TraceMe trace([&] {
        mutex.AssertHeld();
        return TraceMeEncode(
            "command_buffer::cache_hit",
            {{"device", executor->device_ordinal()},
             {"entries", entries.size()},
             {"capacity", capacity}});
      });
      return CommandBufferSelection{entry, CacheLookup::kHit,
                                    std::move(current_allocs), {}, false};
    }
  }

  ++cache_misses;
  CacheLookup lookup = CacheLookup::kMiss;
  CommandBufferEntry* entry;
  if (static_cast<int64_t>(entries.size()) < capacity) {
    ASSIGN_OR_RETURN(
        std::unique_ptr<se::CommandBuffer> command_buffer,
        executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
    entries.insert(entries.begin(), std::make_unique<CommandBufferEntry>(
                                        std::move(command_buffer)));
    entry = entries.front().get();
  } else {
    lookup = CacheLookup::kEviction;
    ++cache_evictions;
    entry = move_to_front(entries.size() - 1);
  }

  bool persistent_allocs_info_is_changed =
      entry->command_buffer->state() != se::CommandBuffer::State::kCreate &&
      entry->persistent_allocs_info_was_valid !=
          persistent_allocs_info_is_valid;
  std::vector<BufferAllocation::Index> updated_allocs;
  for (BufferAllocation::Index index : allocs_to_check) {
    if (entry->recorded_allocs.size() <= index ||
        !entry->recorded_allocs[index].IsSameAs(current_allocs[index])) {
      updated_allocs.push_back(index);
    }
  }

  const char* event_name = lookup == CacheLookup::kEviction
                               ? "command_buffer::cache_evict"
                               : "command_buffer::cache_miss";
  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << (lookup == CacheLookup::kEviction
              ? "Command buffer cache miss (LRU replacement)"
              : "Command buffer cache miss (empty slot)")
      << "; cache=" << this << "; entry=" << entry
      << "; entries=" << entries.size() << "; capacity=" << capacity
      << "; hits=" << cache_hits << "; misses=" << cache_misses
      << "; evictions=" << cache_evictions
      << "; updated_allocs=" << updated_allocs.size()
      << "; updated_alloc_indices=[" << absl::StrJoin(updated_allocs, ",")
      << "]";
  TraceMe trace([&] {
    mutex.AssertHeld();
    return TraceMeEncode(event_name,
                         {{"device", executor->device_ordinal()},
                          {"entries", entries.size()},
                          {"capacity", capacity},
                          {"updated_allocs", updated_allocs.size()}});
  });

  return CommandBufferSelection{entry, lookup, std::move(current_allocs),
                                std::move(updated_allocs),
                                persistent_allocs_info_is_changed};
}

void CommandBufferThunk::ExecutorCommandBufferCache::Discard(
    CommandBufferEntry* entry) {
  auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& item) {
    return item.get() == entry;
  });
  if (it != entries.end()) {
    entries.erase(it);
  }
}

CommandBufferThunk::CommandBufferThunk(
    CommandExecutor commands, ThunkInfo thunk_info,
    std::unique_ptr<SequentialThunk> thunks,
    bool enable_command_buffers_during_profiling,
    int64_t command_buffer_cache_size)
    : Thunk(Thunk::kCommandBuffer, std::move(thunk_info)),
      commands_(std::move(commands)),
      thunks_(std::move(thunks)),
      enable_command_buffers_during_profiling_(
          enable_command_buffers_during_profiling),
      command_buffer_cache_size_(command_buffer_cache_size),
      state_(std::make_shared<State>()) {
  CHECK_GT(command_buffer_cache_size_, 0);
  bool contains_collective = false;
  commands_
      .Walk([&](const Command* command) {
        contains_collective |= command->IsCollective();
        return absl::OkStatus();
      })
      .IgnoreError();
  if (contains_collective && command_buffer_cache_size_ > 1) {
    VLOG(1) << "Command buffer cache capacity is limited to 1 for a command "
               "buffer containing collectives; requested capacity="
            << command_buffer_cache_size_;
    command_buffer_cache_size_ = 1;
  }
  if (VLOG_IS_ON(5)) {
    absl::StatusOr<std::string> graph = commands_.RenderExecutionGraph();
    if (graph.ok()) {
      VLOG(5) << "Rendered command buffer execution graph: " << *graph;
    } else {
      VLOG(5) << "Failed to render command buffer execution graph: "
              << graph.status();
    }
  }

  // When we create a new command buffer thunk (which happens when we
  // instantiate a new Gpu executable) we evict command buffers for all
  // previously instantiated executables. If previously instantiated executable
  // will be executed again, it will simply reconstruct command buffer from
  // a command buffer cmd sequence which is not terribly expensive (few
  // milliseconds for large command buffers). With this approach we keep command
  // buffers (CUDA graphs) resident in device memory only for executable that
  // are actually used.
  //
  // In a perfect world higher level framework (JAX, Tensorflow, PyTorch) would
  // be more aggressive with destroying unused executables, however today they
  // all have a pretty large LRU cache for keeping O(1000) XLA executables.
  EvictCommandBuffers();
  TrackCommandBuffers(state_);
}

absl::Status CommandBufferThunk::Prepare(const PrepareParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // Always prepare thunks if they are present so we are ready to fall back
  // on them if we detect profiling activity.
  if (thunks_) {
    RETURN_IF_ERROR(thunks_->Prepare(params));
  }

  // TODO(b/290773547): Disabled CUDA graphs when profiling is active because of
  // memory corruption.
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Prepare command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TraceMe trace("WARNING: CommandBuffer disabled when profiling");
    return absl::OkStatus();
  }

  RETURN_IF_ERROR(commands_.Prepare(params));

  return absl::OkStatus();
}

absl::Status CommandBufferThunk::Initialize(const InitializeParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // Initialize commands.
  RETURN_IF_ERROR(commands_.Initialize(params));

  // Always initialize thunks if they are present so we are ready to fall back
  // on them if we detect profiling activity.
  if (thunks_) {
    RETURN_IF_ERROR(thunks_->Initialize(params));
  }

  // TODO(b/290773547): Disabled CUDA graphs when profiling is active because of
  // memory corruption.
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Initialize command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TraceMe trace("WARNING: CommandBuffer disabled when profiling");
    return absl::OkStatus();
  }

  std::shared_ptr<ExecutorCommandBufferCache> cache =
      GetOrCreateCommandBufferCache(params.executor);
  absl::MutexLock lock(cache->mutex);

  // If there are no thunks, or command buffer does not require warmup,
  // we can mark warm up as done immediately.
  if (!thunks_ || !commands_.requires_warmup()) {
    cache->warmup_done = true;
  }

  // Construct ExecuteParams with empty fields for everything that is not needed
  // for recording commands.
  Thunk::ExecuteParams execute_params(
      params.buffer_allocations, params.stream,
      params.command_buffer_trace_stream, params.collective_params,
      params.collective_cliques, params.collective_memory,
      /*device_to_host_stream=*/nullptr,
      /*host_to_device_stream=*/nullptr,
      /*send_device_memory_function=*/nullptr,
      /*recv_device_memory_function=*/nullptr, params.ffi_execution_context,
      /*additional_compute_streams=*/{}, params.execution_scoped_state,
      /*mock_collectives=*/false, /*execution_id=*/0,
      /*rng_seed=*/0, params.persistent_alloc_indices);

  if (!cache->warmup_done) {
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(CommandBufferSelection selection,
                   cache->Select(params.executor, commands_, execute_params));
  CommandBufferEntry* cmd_buffer = selection.entry;

  // If command buffer is in `kCreate` state it means that command buffer
  // sequence was never recorded into it. We initialize all command buffers
  // before execution, because command buffers when instantiated will allocate
  // memory on device and this might lead to deadlocks when we have concurrent
  // NCCL operations in flight.

  // If commands require an update during initialization (and VA remapping is
  // not enabled), we also record them into the command buffer before execution.
  // This is required to guarantee that collective commands are recorded on all
  // participating ranks to avoid deadlocks. We also update an existing command
  // buffer when persistent allocation indices become available, because that
  // changes which commands can safely retain their recorded addresses.
  bool has_dynamic_allocations =
      ExecutorCommandBufferCache::HasDynamicAllocations(
          commands_, params.persistent_alloc_indices);
  bool is_first_record =
      cmd_buffer->command_buffer->state() == se::CommandBuffer::State::kCreate;
  bool persistent_allocs_info_is_valid =
      params.persistent_alloc_indices.has_value();
  if (is_first_record || selection.lookup == CacheLookup::kEviction ||
      selection.persistent_allocs_info_is_changed ||
      (has_dynamic_allocations && commands_.requires_update_on_initialize())) {
    const char* initialization_reason =
        is_first_record
            ? "new_cache_entry"
            : selection.persistent_allocs_info_is_changed
                  ? "persistent_allocation_policy"
                  : selection.lookup == CacheLookup::kEviction
                        ? "cache_eviction"
                        : "requires_update_on_initialize";
    VLOG(3) << "Initialize command buffer on device #"
            << params.executor->device_ordinal()
            << " by recoding command buffer cmd sequence"
            << "; num_commands=" << commands_.size()
            << "; persistent_allocs_info_is_valid="
            << persistent_allocs_info_is_valid
            << "; reason=" << initialization_reason;

    TraceMe trace([&] {
      return TraceMeEncode("command_buffer::initialize",
                           {{"device", params.executor->device_ordinal()},
                            {"num_commands", commands_.size()},
                            {"reason", initialization_reason}});
    });

    uint64_t start_micros = tsl::Env::Default()->NowMicros();

    std::optional<std::vector<BufferAllocation::Index>> updated_allocs =
        std::move(selection.updated_allocs);

    // Force all commands to update when the persistent-allocation policy
    // changes so none retain addresses recorded under the previous policy.
    if (selection.persistent_allocs_info_is_changed) {
      updated_allocs.reset();
    }

    Command::RecordParams record_params = {cmd_buffer->state,
                                           std::move(updated_allocs),
                                           /*is_initialization=*/true};
    absl::Status record_status = commands_.Record(
        execute_params, record_params, cmd_buffer->command_buffer.get());
    if (!record_status.ok()) {
      cache->Discard(cmd_buffer);
      return record_status;
    }
    cmd_buffer->recorded_allocs = std::move(selection.recorded_allocs);
    cmd_buffer->persistent_allocs_info_was_valid =
        persistent_allocs_info_is_valid;

    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    VLOG(3) << (is_first_record ? "Recorded" : "Updated")
            << " command buffer during initialization on device #"
            << params.executor->device_ordinal() << " in "
            << (end_micros - start_micros)
            << " μs; num_commands=" << commands_.size()
            << "; reason=" << initialization_reason;
    cmd_buffer->num_executions = 0;
  }
  return absl::OkStatus();
}

absl::Status CommandBufferThunk::ExecuteOnStream(const ExecuteParams& params) {
  // We might end up with empty command sequence if all of the captured fusions
  // are no-op (e.g. memcpy of size 0) and we have no emitted thunks for them.
  if (commands_.empty()) {
    return absl::OkStatus();
  }

  // TODO(b/290773547): Profiler (CUPTI) + CUDA graphs lead to memory
  // corruption. As a work around disable command buffers (CUDA graphs) and run
  // everything in op-by-op mode.
  if (tsl::profiler::ProfilerLock::HasActiveSession() && thunks_ &&
      !enable_command_buffers_during_profiling_) {
    VLOG(1) << "Execute command buffer thunk as a regular thunk sequence "
               "because we detected active profiling session";
    TraceMe trace("WARNING: CommandBuffer disabled when profiling");
    return thunks_->ExecuteOnStream(params);
  }

  se::StreamExecutor* executor = params.stream->parent();
  std::shared_ptr<ExecutorCommandBufferCache> cache =
      GetOrCreateCommandBufferCache(executor);

  absl::MutexLock lock(cache->mutex);

  // warm up iteration, run through thunks if they are present.
  if (!cache->warmup_done && thunks_) {
    VLOG(2) << "Executing warm up iteration of command buffer thunk";
    RETURN_IF_ERROR(thunks_->ExecuteOnStream(params));
    cache->warmup_done = true;
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(CommandBufferSelection selection,
                   cache->Select(executor, commands_, params));
  CommandBufferEntry* cmd_buffer = selection.entry;

  bool is_first_record =
      cmd_buffer->command_buffer->state() == se::CommandBuffer::State::kCreate;
  // Executions initialized on opposite sides of the one-way policy transition
  // can interleave, so the persistent allocation information recorded in the
  // shared command buffer can have different validity from the current run.
  bool persistent_allocs_info_is_valid =
      params.persistent_alloc_indices.has_value();
  bool persistent_allocs_info_is_changed =
      selection.persistent_allocs_info_is_changed;

  std::vector<BufferAllocation::Index> updated_allocs =
      std::move(selection.updated_allocs);

  bool has_dynamic_allocations =
      ExecutorCommandBufferCache::HasDynamicAllocations(
          commands_, params.persistent_alloc_indices);
  bool needs_update = persistent_allocs_info_is_changed ||
                      commands_.requires_update_on_execute() ||
                      (has_dynamic_allocations && !updated_allocs.empty());

  if (is_first_record || needs_update) {
    XLA_VLOG_DEVICE(3, executor->device_ordinal())
        << "Create/Update command buffer"
        << " by recoding command buffer cmd sequence after "
        << cmd_buffer->num_executions << " executions since last update"
        << "; num_commands=" << commands_.size()
        << "; updated_allocs=" << updated_allocs.size()
        << "; is_first_record=" << is_first_record
        << "; persistent_allocs_info_is_changed="
        << persistent_allocs_info_is_changed
        << "; needs_update=" << needs_update;

    const char* update_reason =
        is_first_record
            ? "new_cache_entry"
            : persistent_allocs_info_is_changed
                  ? "persistent_allocation_policy"
                  : selection.lookup == CacheLookup::kEviction
                        ? "cache_eviction"
                        : commands_.requires_update_on_execute()
                              ? "requires_update_on_execute"
                              : "allocation_addresses";
    TraceMe trace([&] {
      cache->mutex.AssertHeld();
      return TraceMeEncode(
          is_first_record ? "command_buffer::record" : "command_buffer::update",
          {{"device", executor->device_ordinal()},
           {"num_commands", commands_.size()},
           {"reason", update_reason}});
    });

    uint64_t start_micros = tsl::Env::Default()->NowMicros();

    std::optional<std::vector<BufferAllocation::Index>> allocs_to_update =
        std::move(updated_allocs);
    // Force all commands to update when the recorded information validity does
    // not match this execution. A filtered update could skip a command whose
    // allocations are all persistent in one of the interleaved executions.
    if (persistent_allocs_info_is_changed) {
      allocs_to_update.reset();
    }

    Command::RecordParams record_params = {
        cmd_buffer->state, std::move(allocs_to_update),
        /*is_initialization=*/is_first_record && !has_dynamic_allocations};
    absl::Status record_status = commands_.Record(
        params, record_params, cmd_buffer->command_buffer.get());
    if (!record_status.ok()) {
      cache->Discard(cmd_buffer);
      return record_status;
    }
    cmd_buffer->recorded_allocs = std::move(selection.recorded_allocs);
    cmd_buffer->persistent_allocs_info_was_valid =
        persistent_allocs_info_is_valid;

    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    XLA_VLOG_DEVICE(3, executor->device_ordinal())
        << (is_first_record ? "Recorded" : "Updated")
        << " command buffer in "
        << (end_micros - start_micros)
        << " μs; num_commands=" << commands_.size()
        << "; reason=" << update_reason;
    cmd_buffer->num_executions = 0;
  }

  ++cmd_buffer->num_executions;

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "Execute command buffer"
      << "; num_executions=" << cmd_buffer->num_executions;

  TraceMe trace([&] {
    cache->mutex.AssertHeld();
    return TraceMeEncode("command_buffer::execute",
                         {{"device", executor->device_ordinal()},
                          {"num_commands", commands_.size()},
                          {"num_executions", cmd_buffer->num_executions},
                          {"cache_entries", cache->entries.size()},
                          {"cache_capacity", cache->capacity}});
  });

  return cmd_buffer->command_buffer->Submit(params.stream);
}

std::shared_ptr<CommandBufferThunk::ExecutorCommandBufferCache>
CommandBufferThunk::GetOrCreateCommandBufferCache(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(state_->mutex);
  if (auto it = state_->command_buffer_caches.find(executor);
      it != state_->command_buffer_caches.end()) {
    return it->second;
  }

  auto emplaced = state_->command_buffer_caches.emplace(
      executor, std::make_shared<ExecutorCommandBufferCache>(
                    command_buffer_cache_size_));
  return emplaced.first->second;
}

//===----------------------------------------------------------------------===//
// Command buffer eviction
//===----------------------------------------------------------------------===//

struct CommandBufferThunk::GlobalState {
  absl::Mutex mutex;
  std::vector<std::weak_ptr<CommandBufferThunk::State>> state
      ABSL_GUARDED_BY(mutex);
};

CommandBufferThunk::GlobalState* CommandBufferThunk::GetGlobalState() {
  static auto* const global_state = new GlobalState();
  return global_state;
}

void CommandBufferThunk::TrackCommandBuffers(
    std::weak_ptr<CommandBufferThunk::State> state) {
  auto* global_state = GetGlobalState();
  absl::MutexLock global_state_lock(global_state->mutex);
  global_state->state.push_back(state);
}

void CommandBufferThunk::EvictCommandBuffers() {
  TraceMe trace("EvictCommandBuffers");

  auto* global_state = GetGlobalState();
  absl::MutexLock global_state_lock(global_state->mutex);
  VLOG(3) << "Evict command buffer thunk command buffers; tracked thunks = "
          << global_state->state.size();

  // Erase state for already destroyed thunks.
  global_state->state.erase(
      std::remove_if(global_state->state.begin(), global_state->state.end(),
                     [](auto& weak_ptr) { return weak_ptr.expired(); }),
      global_state->state.end());

  // Evict command buffers for all tracked thunks.
  int64_t num_evicted = 0;
  for (auto& weak_ptr : global_state->state) {
    auto ptr = weak_ptr.lock();
    if (!ptr) {
      continue;
    }

    // Evict all command buffer caches and count their graph entries.
    absl::MutexLock state_lock(ptr->mutex);
    for (const auto& item : ptr->command_buffer_caches) {
      const std::shared_ptr<ExecutorCommandBufferCache>& cache = item.second;
      absl::MutexLock cache_lock(cache->mutex);
      num_evicted += cache->entries.size();
    }
    ptr->command_buffer_caches.clear();
  }

  if (num_evicted > 0) {
    VLOG(3) << "Evicted " << num_evicted
            << " command buffer thunk command buffers";
  }
}

absl::Status CommandBufferThunk::WalkNested(Walker callback) {
  if (thunks_ != nullptr) {
    RETURN_IF_ERROR(thunks_->Walk(callback));
  }
  return absl::OkStatus();
}

std::string CommandBufferThunk::ToString(int indent) const {
  std::string result = "\n";
  absl::StrAppend(&result, thunks_->ToString(indent + 1));
  return result;
}
absl::StatusOr<ThunkProto> CommandBufferThunk::ToProto() const {
  return absl::InvalidArgumentError("CommandBufferThunk can't be serialized.");
}

}  // namespace xla::gpu
