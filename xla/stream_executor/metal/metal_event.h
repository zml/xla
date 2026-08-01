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
  using FlushProducerFn = void (*)(void* ctx, uint64_t value);

  MetalEvent() = default;
  ~MetalEvent() override;

  Status PollForStatus() override;
  absl::Status Synchronize() override;

  void SetCommandBuffer(void* command_buffer);

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
  void FlushProducerForWait();

  void* command_buffer_ = nullptr;
  void* shared_event_ = nullptr;
  uint64_t signal_value_ = 0;
  FlushProducerFn flush_fn_ = nullptr;
  void* flush_ctx_ = nullptr;  // borrowed (the MetalExecutor).
};

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_EVENT_H_
