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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_EVENT_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_EVENT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/stream_executor/event.h"

namespace stream_executor::metal {

class MetalEvent : public Event {
 public:
  // Force-commits the open command buffer carrying `value` (ctx = the executor;
  // body calls CommitOpenBufferThrough). A raw fn ptr, not std::function, to
  // avoid any per-RecordEvent allocation and a circular metal_executor BUILD dep.
  using FlushProducerFn = void (*)(void* ctx, uint64_t value);

  MetalEvent() = default;
  ~MetalEvent() override;

  Status PollForStatus() override;
  absl::Status Synchronize() override;

  void SetCommandBuffer(void* command_buffer);

  // Records that the producer command buffer signals `value` on `shared_event`
  // (the per-device MTLSharedEvent, NOT owned here). Lets a consumer stream emit
  // a GPU-side wait instead of a host block. See MetalStream::WaitFor(Event*).
  // flush_fn/flush_ctx let a HOST waiter (Poll/Synchronize) force-commit the
  // producer's still-open command buffer carrying `value` — with deferred
  // commits (METAL_COMMIT_EVERY>1) the signal fires only on commit, so a host
  // wait on a value still batched in an open buffer would otherwise deadlock
  // (the blocked host can't issue the executes that trigger the batch commit).
  void SetSignal(void* shared_event, uint64_t value, FlushProducerFn flush_fn,
                 void* flush_ctx) {
    shared_event_ = shared_event;
    signal_value_ = value;
    flush_fn_ = flush_fn;
    flush_ctx_ = flush_ctx;
  }
  void* shared_event() const { return shared_event_; }
  uint64_t signal_value() const { return signal_value_; }

 private:
  // Force-commit the open buffer carrying signal_value_ so the signal can fire.
  void FlushProducerForWait();

  void* command_buffer_ = nullptr;
  // Borrowed (owned by MetalExecutor); 0 value == no GPU signal associated.
  void* shared_event_ = nullptr;
  uint64_t signal_value_ = 0;
  FlushProducerFn flush_fn_ = nullptr;
  void* flush_ctx_ = nullptr;  // borrowed (the MetalExecutor).
};

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_EVENT_H_
