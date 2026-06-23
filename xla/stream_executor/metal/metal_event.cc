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

#include "xla/stream_executor/metal/metal_event.h"

#include "absl/status/status.h"
#include "xla/stream_executor/metal/metal_runtime.h"

namespace stream_executor::metal {

MetalEvent::~MetalEvent() {
  if (command_buffer_ != nullptr) {
    ReleaseObject(command_buffer_);
  }
}

void MetalEvent::FlushProducerForWait() {
  // With deferred commits (METAL_COMMIT_EVERY>1) the producer's signal fires
  // only when its command buffer commits. A host waiter on a value still batched
  // in an open buffer must force-commit it first, or it waits forever (the
  // blocked host can't issue the executes that would trigger the batch commit).
  // No-op if the value is already committed/signaled. K=1 → always already
  // committed → no-op (behavior unchanged).
  if (flush_fn_ != nullptr && signal_value_ != 0) {
    flush_fn_(flush_ctx_, signal_value_);
  }
}

Event::Status MetalEvent::PollForStatus() {
  // Value-backed: complete iff the per-device shared event has reached this
  // event's signal value. This decouples the event from owning a committed
  // command buffer (the structural reason RecordEvent had to commit) — the
  // first step toward making per-execute commits free. signal_value_ == 0 means
  // no GPU work was associated (matches CUDA "a newly created event is deemed
  // already-past", buffer_sequencing_event.cc) and the old null-cb behavior.
  // signal_value_ == 0: no GPU work associated. shared_event_ == nullptr: the
  // executor's shared event failed to initialize (degraded fallback, see
  // MetalExecutor::Init) so no signal was ever encoded — treat as already-complete
  // rather than spinning kPending forever (a poller can never advance a value that
  // was never signaled). Mirrors Synchronize(), which already no-ops on a null
  // shared event via WaitUntilSignaledValue. Neither holds on the supported decode
  // path (shared_event_ is always created), so this is byte-identical there.
  if (signal_value_ == 0 || shared_event_ == nullptr) {
    return Event::Status::kComplete;
  }
  if (SharedEventSignaledValue(shared_event_) >= signal_value_) {
    return Event::Status::kComplete;
  }
  // Not yet reached — someone is polling readiness. Force-commit the producer's
  // open buffer so the signal can fire (else a deferred batch would never
  // commit while a poller spins). Re-read after flushing.
  FlushProducerForWait();
  return SharedEventSignaledValue(shared_event_) >= signal_value_
             ? Event::Status::kComplete
             : Event::Status::kPending;
}

absl::Status MetalEvent::Synchronize() {
  // Host-block on the shared-event value rather than a committed cb (see
  // PollForStatus). signal_value_ == 0 ⇒ nothing to wait on.
  if (signal_value_ == 0) return absl::OkStatus();
  FlushProducerForWait();  // commit the producer's open batch before blocking.
  return WaitUntilSignaledValue(shared_event_, signal_value_);
}

void MetalEvent::SetCommandBuffer(void* command_buffer) {
  if (command_buffer_ != nullptr) {
    ReleaseObject(command_buffer_);
  }
  command_buffer_ = command_buffer;
}

}  // namespace stream_executor::metal
