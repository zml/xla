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

// Force-commits the open command buffer carrying `value` so its shared-event
// signal can fire. Passed to MetalEvent::SetSignal so a host waiter can unblock
// a value still batched in an open buffer (deferred commits). ctx is the
// MetalExecutor (a raw fn ptr avoids a circular metal_executor BUILD dep).
void FlushProducerThunk(void* ctx, uint64_t value) {
  static_cast<MetalExecutor*>(ctx)->CommitOpenBufferThrough(value);
}

}  // namespace

MetalStream::MetalStream(
    MetalExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority)
    : StreamCommon(executor, priority), executor_(executor) {
  executor_->RegisterStream(this);
  // Start at the commit-cadence ceiling; the first token boundary recomputes the
  // adaptive per-token threshold for this model (see adaptive_k_).
  adaptive_k_ = commit_every_;
}

MetalStream::~MetalStream() {
  // Stop receiving cross-stream flush calls before tearing the buffer down.
  executor_->UnregisterStream(this);
  // Commits + waits + releases the open command buffer (if any).
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

absl::Status MetalStream::RecordEvent(Event* event) {
  auto* metal_event = dynamic_cast<MetalEvent*>(event);
  if (metal_event == nullptr) {
    return absl::InvalidArgumentError("Expected a MetalEvent.");
  }
  // Encode a GPU-side signal at the END of this execute's work (after all
  // kernels) so a dependent execute can order on it WITHOUT a host block. The
  // event is value-backed (MetalEvent polls the shared-event value), so it does
  // NOT need a committed command buffer — we can keep the buffer OPEN and commit
  // only every `commit_every_` executes (partial batching). Same-stream
  // consumers order in-buffer by program order (WaitFor's in-buffer fast path);
  // cross-stream/cross-batch consumers force-commit via CommitOpenBufferThrough.
  if (command_buffer_ != nullptr) {
    const uint64_t value = executor_->NextEventValue();
    metal::EncodeSignalSharedEvent(command_buffer_, executor_->shared_event(),
                                   value);
    pending_signal_high_ = value;
    metal_event->SetCommandBuffer(nullptr);
    metal_event->SetSignal(executor_->shared_event(), value,
                           &FlushProducerThunk, executor_);
    // Commit at the batch boundary: starts this batch's GPU work and bounds the
    // open buffer to ~adaptive_k_ executes (the per-token, model-derived
    // threshold). Releasing the committed cb is safe — the event is value-backed,
    // not cb-backed.
    ++executes_this_token_;  // for the next token-boundary recompute.
    if (++signals_since_commit_ >= adaptive_k_) {
      void* committed = metal::CommitBatchCommandBuffer(command_buffer_);
      ReleaseObject(command_buffer_);
      command_buffer_ = nullptr;
      if (committed != nullptr) ReleaseObject(committed);
      last_signaled_value_ = value;
      pending_signal_high_ = 0;
      signals_since_commit_ = 0;
      waited_value_high_ = 0;
    }
  } else {
    // No open command buffer: the batch already committed at an adaptive-K /
    // token boundary, or this execute encoded no GPU work of its own. Point at
    // the highest value this stream has signaled (committed or pending) so a
    // consumer still orders against that work. 0 only before the first signal
    // (no prior work to wait on).
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
    // IN-BUFFER FAST PATH: if this value was signaled into THIS stream's still-
    // open command buffer (same shared event, not yet committed), the consumer's
    // kernels will be encoded into the SAME buffer AFTER the signal, so the
    // single-queue program order already orders consumer-after-producer. No
    // commit and no GPU wait needed — this is what removes the per-execute
    // command-buffer transition gap when commit_every_ > 1. (Decode has no
    // mid-execute command-buffer splits — D2D copies are in-buffer blits, no
    // MPSGraph splits — so the whole open batch is one ordered segment.)
    if (metal_event->shared_event() == executor_->shared_event() &&
        command_buffer_ != nullptr && value > last_signaled_value_ &&
        value <= pending_signal_high_) {
      return absl::OkStatus();
    }
    // ALREADY-SIGNALED fast path: if the GPU has ALREADY crossed `value`
    // (shared-event signaledValue), the producer's work is fully complete —
    // this consumer's kernels are encoded after that observation and commit
    // after it, so no ordering is needed at all and the wait is elided.
    // This is unconditionally safe (it orders against the PAST), unlike
    // eliding on commit-order alone, which measurably broke the CLI's decode:
    // a committed-but-still-running producer can overlap the consumer's
    // buffer on the queue, and cross-buffer hazard tracking does NOT
    // reliably order them. The encodeWaitForEvent this elides is a
    // firmware-level stall (~0.1 ms each, even for an already-signaled
    // value); a serving loop that re-waits every layer-execute on the
    // PREVIOUS token's KV-cache events (complete long ago, but past the
    // in-buffer fast path) pays ~57 waits ≈ 3 ms/token for nothing (llmd:
    // 49.5 -> 62 tok/s at batch 1).
    if (metal_event->shared_event() == executor_->shared_event() &&
        metal::SharedEventSignaledValue(metal_event->shared_event()) >= value) {
      return absl::OkStatus();
    }
    // COVERED-BY-EARLIER-WAIT fast path: the open buffer already carries a
    // GPU wait for >= `value`; everything encoded after that wait is ordered
    // behind it by program order, so a second wait adds nothing but its
    // firmware stall. (Per-layer executes all waiting the same per-token
    // H2D-staged inputs hit this 27x/token.) Still force-commit a possibly
    // still-open OTHER-stream buffer carrying `value` — the earlier wait only
    // orders against signals that actually get committed.
    if (metal_event->shared_event() == executor_->shared_event() &&
        command_buffer_ != nullptr && value <= waited_value_high_) {
      executor_->CommitOpenBufferThrough(value);
      return absl::OkStatus();
    }
    // GPU-side wait across buffers: encode a wait for the producer's signal at
    // the START of this stream's open buffer; the kernels encoded next won't run
    // until the producer's buffer signals — NO host block, so executes pipeline.
    // Cross-stream/cross-batch: the producer's buffer may still be OPEN; on the
    // single shared queue a waiter committed before its signaler deadlocks, so
    // force-commit the producer's open buffer carrying this value FIRST.
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
  // No GPU signal associated (empty producer, or a non-Metal event) — host sync.
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
  // Unified memory: a host->device copy is a plain host write into the buffer's
  // contents, but the GPU may still be reading that buffer from prior work, so drain
  // first to order this write after it. This also leaves the value HOST-coherent
  // immediately — the prefill grid clamps (metal_flash_attn_thunk / _gemm_thunk)
  // read input scalars like num_tokens host-side at encode and rely on that.
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
  // Device-to-device: encode an async GPU blit into the open command buffer
  // instead of draining the queue + a CPU memmove. The drain serialized decode
  // token-to-token — the only transfer thunk in the decode chain is the lm_head
  // RNG-state copy (copy.5), ~once/token, and its BlockHostUntilDone drained the
  // whole token's queue (~13 ms), so the host could not run ahead → ~18% GPU
  // idle. The blit lets the host pipeline ahead; the result is consumed on the
  // GPU, ordered by in-buffer hazard tracking + the stream's shared-event signal
  // across executes (like CUDA/AMD's async cuMemcpyAsync). Falls back to the
  // synchronous path if either pointer isn't a resolvable Metal allocation, or
  // for an overlapping same-buffer copy (blit copyFromBuffer is undefined on
  // overlap; memmove is safe).
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
  // ASYNC host callback (do NOT block the host). Drives PJRT's per-execute
  // definition event (SetStateConcrete, routed through ThenExecuteCallback ->
  // DoHostCallback) on REAL GPU completion while the host keeps enqueueing the
  // next execution — true host/GPU pipelining instead of a BlockHostUntilDone
  // per execute.
  //
  // Step 3 mechanism: encode a shared-event SIGNAL into the open command buffer
  // and register a host LISTENER at that value (the Metal analog of
  // cuLaunchHostFunc) to run `callback` when the GPU crosses it. This decouples
  // the callback from a per-buffer addCompletedHandler — the prerequisite for
  // dropping the per-execute commit later. For now we STILL commit here, so the
  // commit count/timing are unchanged and the listener can never be starved
  // (its carrying buffer is committed immediately). Single serial listener
  // queue ⇒ callbacks run in value (== commit) order, matching the old handler.
  auto run = [cb = std::move(callback)]() mutable {
    absl::Status status = std::move(cb)();
    if (!status.ok()) {
      LOG(ERROR) << "Metal host callback returned an error: " << status;
    }
  };

  // Drive the callback (PJRT SetStateConcrete) off this execute's signal value
  // via a host listener (the Metal analog of cuLaunchHostFunc) WITHOUT committing
  // the open command buffer — this is what lets the buffer keep batching across
  // executes. The value is the signal RecordEvent encoded for this execute:
  // pending (still in the open buffer) or the last committed one. The listener
  // fires when the GPU crosses that value, which happens once the carrying buffer
  // commits — guaranteed within commit_every_ executes by the RecordEvent batch
  // commit, or sooner by a cross-stream readback / BlockHostUntilDone force-
  // commit (CommitOpenBufferThrough) — so it is never starved. Single serial
  // listener queue ⇒ callbacks run in value (== completion) order, exactly-once.
  void* shared_event = executor_->shared_event();
  void* listener = executor_->shared_event_listener();
  const uint64_t value =
      pending_signal_high_ != 0 ? pending_signal_high_ : last_signaled_value_;
  if (shared_event != nullptr && listener != nullptr && value != 0) {
    metal::NotifySharedEventListener(listener, shared_event, value,
                                     std::move(run));
    return absl::OkStatus();
  }

  // Fallback: there is uncommitted work but no shared-event signal (an execute
  // with kernels and no RecordEvent), or no listener — commit with the legacy
  // per-buffer completion handler so the callback still fires on real
  // completion. Otherwise (no GPU work at all) run inline ("already-past").
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
  // Commit the open command buffer, then wait its final committed segment (which
  // drains all prior work on the queue). Release both regardless of wait status.
  void* committed = metal::CommitBatchCommandBuffer(command_buffer_);
  ReleaseObject(command_buffer_);
  command_buffer_ = nullptr;
  // Match FlushBatchedWork's reset discipline. This commit advances the stream's
  // signaled-value bookkeeping; leaving it stale is unsafe: a stale
  // pending_signal_high_ survives into the NEXT open buffer (EnsureOpenCommandBuffer
  // does not clear it), where WaitFor's in-buffer fast path (value <=
  // pending_signal_high_) could skip a cross-buffer wait for a signal that was
  // never encoded into that buffer; a stale signals_since_commit_ skews the
  // adaptive commit cadence. The work is committed + about to be waited, so
  // advancing last_signaled_value_ to it is correct.
  if (pending_signal_high_ != 0) last_signaled_value_ = pending_signal_high_;
  pending_signal_high_ = 0;
  signals_since_commit_ = 0;
  waited_value_high_ = 0;
  absl::Status status = WaitUntilCompleted(committed);
  // (GPU profiling resolves inside CommitBatchCommandBuffer — the one commit
  // funnel shared by this path and RecordEvent — so nothing to do here.)
  ReleaseObject(committed);
  return status;
}

absl::Status MetalStream::FlushBatchedWork() {
  // A host transfer (D2H readback) is a synchronization point. With deferred
  // commits (commit_every_ > 1) the open command buffer may carry the execute
  // whose output is being read; its PJRT definition event (SetStateConcrete,
  // driven by the shared-event listener registered in DoHostCallback) fires
  // only once that buffer commits. Commit it now — WITHOUT waiting — so the
  // listener can fire and the host transfer's wait on the definition event
  // resolves; the transfer itself then waits for completion. This is the MLX
  // model: batch across executes, but flush the open buffer at a sync point.
  // No-op when nothing is open (already committed / nothing batched), which is
  // always at commit_every_ == 1 — so the K=1 baseline is unchanged.
  // Single-host-thread stream-op model (like the rest of this class): in decode
  // the readback runs on the same thread that issued the executes, after the
  // last one, so command_buffer_ is stable here.
  //
  // This D2H readback is the per-token boundary: recompute the adaptive commit
  // threshold from the executes-per-token THIS model just ran, so K auto-scales
  // (no fixed per-model constant). Round to ~target_commits_per_token_ commits,
  // clamp to [1, commit_every_ ceiling]. Done unconditionally (before the
  // open-buffer check) — the boundary stands even if the tail already committed.
  // ONLY here: a cross-stream FlushOpenBufferIfCarrying is not a token boundary.
  if (executes_this_token_ > 0) {
    const int t = target_commits_per_token_;
    int k = (executes_this_token_ + t / 2) / t;  // round(executes / target)
    if (k < 1) k = 1;
    if (k > commit_every_) k = commit_every_;
    adaptive_k_ = k;
    executes_this_token_ = 0;
  }
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
}

void MetalStream::FlushOpenBufferIfCarrying(uint64_t value) {
  // Commit (without waiting) this stream's OPEN command buffer iff it actually
  // carries the awaited signal `value` — i.e. `value` was encoded into it
  // (pending_signal_high_ >= value) and is not already committed
  // (value > last_signaled_value_). Committing lets the encoded signal fire so
  // a cross-stream waiter on `value` can be committed safely after it. No-op
  // otherwise — which is ALWAYS until per-execute commits are deferred, since
  // RecordEvent leaves pending_signal_high_ == 0. Single-host-thread stream-op
  // model (like the rest of this class); the executor serializes callers.
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
}

absl::Status MetalStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes, bool use_pdl) {
  return LaunchMetalKernel(thread_dims, block_dims, cluster_dims,
                           /*pipeline=*/function, /*function=*/nullptr,
                           /*use_argument_buffer=*/false, name, args,
                           shmem_bytes, use_pdl);
}

absl::Status MetalStream::LaunchMetalKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* pipeline,
    void* function, bool use_argument_buffer, absl::string_view name,
    void** args, int64_t shmem_bytes, bool use_pdl) {
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

  // Encode into the stream's open command buffer (#62): the whole execution is
  // one command buffer, committed + waited once by BlockHostUntilDone, instead
  // of one tiny command buffer per kernel.
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
