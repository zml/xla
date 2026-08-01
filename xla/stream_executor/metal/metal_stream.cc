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

// === Env-gated decode batching diagnostics (METAL_KPROF=1) =================
// Counts, per decode token, how the cross-execute sync resolves: commits and
// which WaitFor branch each consumer hit. Logged + reset at the token boundary
// (FlushBatchedWork). Pinpoints why the self-clocked batch commits collapse to
// one-per-execute on a given model. Single-host-thread stream model -> plain
// ints. Zero overhead unless METAL_KPROF is set.
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
  // Encode a GPU-side signal at the END of this execute's work (after all
  // kernels) so a dependent execute can order on it WITHOUT a host block. The
  // event is value-backed (MetalEvent polls the shared-event value), so it does
  // NOT need a committed command buffer — we can keep the buffer OPEN and defer
  // the commit to the self-clocked batch boundary below (partial batching).
  // Same-stream consumers order in-buffer by program order (WaitFor's in-buffer
  // fast path); cross-stream/cross-batch consumers force-commit via
  // CommitOpenBufferThrough.
  if (command_buffer_ != nullptr) {
    const uint64_t value = executor_->NextEventValue();
    metal::EncodeSignalSharedEvent(command_buffer_, executor_->shared_event(),
                                   value);
    pending_signal_high_ = value;
    metal_event->SetCommandBuffer(nullptr);
    metal_event->SetSignal(executor_->shared_event(), value,
                           &FlushProducerThunk, executor_);
    // Self-clocking commit (event-driven, no tuned cadence): commit the open
    // buffer iff the GPU has drained everything this stream previously
    // committed (shared-event signaledValue crossed the last committed signal)
    // — i.e. exactly when the pipeline would otherwise go dry. While committed
    // work is still running the buffer keeps batching, so each buffer
    // accumulates for about as long as its predecessor executes: balanced
    // host/GPU pipelining with the cross-cb gap amortized over the batch (see
    // metal_stream.h for the two extremes). The signal cap only bounds
    // encoded-command growth when the GPU stays busy for a long stretch.
    // Releasing the committed cb is safe — the event is value-backed, not
    // cb-backed.
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
    // No open command buffer: the batch already committed at a self-clocked /
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
    // command-buffer transition gap while commits are deferred. (Decode has no
    // mid-execute command-buffer splits — D2D copies are in-buffer blits, no
    // MPSGraph splits — so the whole open batch is one ordered segment.)
    if (metal_event->shared_event() == executor_->shared_event() &&
        command_buffer_ != nullptr && value > last_signaled_value_ &&
        value <= pending_signal_high_) {
      if (BatchDbgEnabled()) ++g_bdbg.w_inbuf;
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
      if (BatchDbgEnabled()) ++g_bdbg.w_signaled;
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
      if (BatchDbgEnabled()) ++g_bdbg.w_covered;
      executor_->CommitOpenBufferThrough(value);
      return absl::OkStatus();
    }
    if (BatchDbgEnabled()) ++g_bdbg.w_xbuf;
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
  if (BatchDbgEnabled()) ++g_bdbg.w_hostsync;
  return event->Synchronize();
}

absl::Status MetalStream::Memset32(DeviceAddressBase* location,
                                   uint32_t pattern, uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("Metal Memset32 size is not 4-byte aligned.");
  }
  // Byte-uniform patterns (0x00000000, 0xFFFFFFFF, ...) can use the async GPU
  // fill (fillBuffer is byte-granular) — same no-drain path as MemZero. Other
  // patterns keep the drain + host loop (not seen on any hot path).
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
  // Encode an async GPU fill into the open command buffer instead of draining
  // the queue + a host memset. The drain serialized tiled prefill layer-by-layer
  // — paged-attn's per-layer output MemZero fired 28x/request on Llama-3B, each
  // waiting the whole queue (~54 ms/request host-block, and no host run-ahead
  // between prefill layers). The fill is encoded into the buffer like
  // EncodeBlitCopy (the open compute encoder is ended, then the blit runs as its
  // own encoder — ordered after prior encoders' accesses, before later reads);
  // cross-execute consumers order via the stream's shared-event signal.
  // Falls back to drain + host memset when the pointer isn't a resolvable
  // Metal allocation (also keeps the value host-coherent for host readers).
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
  // commits — guaranteed by RecordEvent's self-clocked commit (committed work
  // drains in finite time, after which the next signal commits; the per-buffer
  // signal cap bounds the batch meanwhile), or sooner by a cross-stream readback
  // / BlockHostUntilDone force-commit (CommitOpenBufferThrough) — so it is never
  // starved. Single serial
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
  // per-buffer signal cap. The work is committed + about to be waited, so
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
  // commits the open command buffer may carry the execute
  // whose output is being read; its PJRT definition event (SetStateConcrete,
  // driven by the shared-event listener registered in DoHostCallback) fires
  // only once that buffer commits. Commit it now — WITHOUT waiting — so the
  // listener can fire and the host transfer's wait on the definition event
  // resolves; the transfer itself then waits for completion. This is the MLX
  // model: batch across executes, but flush the open buffer at a sync point.
  // No-op when nothing is open (already committed / nothing batched).
  // Single-host-thread stream-op model (like the rest of this class): in decode
  // the readback runs on the same thread that issued the executes, after the
  // last one, so command_buffer_ is stable here. No cadence state to recompute
  // — the commit cadence is self-clocked off GPU progress in RecordEvent.
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
  // Commit the open buffer WITHOUT waiting. Used by the PJRT execute path when
  // a host-awaited completion future is requested (returned futures): such an
  // isolated execute may have no following execute to reach a self-clocked
  // commit and no D2H readback to flush, so its open command buffer (carrying
  // the shared-event signal the definition-event listener waits on) would never
  // commit, the listener would never fire, and the host await would deadlock.
  // Committing here fires the listener. It only flushes (there is no cadence
  // state to skew), so it is safe even when interleaved mid-token with batched
  // decode executes on this same stream.
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
  if (BatchDbgEnabled()) { ++g_bdbg.commits; ++g_bdbg.foic_commit; }
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
    void** args, int64_t shmem_bytes, bool use_pdl,
    void* indirect_grid_device_ptr) {
  if (cluster_dims.has_value()) {
    return absl::UnimplementedError("Metal cluster launches are not supported.");
  }
  if (use_pdl) {
    return absl::UnimplementedError(
        "Metal programmatic dependent launch is not supported.");
  }

  // Resolve the optional indirect-grid device pointer to its backing MTLBuffer +
  // byte offset (same resolution path as the kernel arguments below).
  void* indirect_grid_buffer = nullptr;
  uint64_t indirect_grid_offset = 0;
  if (indirect_grid_device_ptr != nullptr) {
    auto allocation = executor_->ResolveAllocation(indirect_grid_device_ptr);
    if (!allocation.ok()) {
      return absl::InternalError(
          "Metal indirect dispatch: could not resolve grid buffer allocation.");
    }
    indirect_grid_buffer = allocation->buffer;
    indirect_grid_offset = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(indirect_grid_device_ptr) -
        reinterpret_cast<uintptr_t>(allocation->contents));
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
                             block_dims, shmem_bytes, indirect_grid_buffer,
                             indirect_grid_offset);
}

void MetalStream::EnsureOpenCommandBuffer() {
  if (command_buffer_ == nullptr) {
    command_buffer_ = metal::NewBatchCommandBuffer(executor_->command_queue());
  }
}

}  // namespace stream_executor::metal
