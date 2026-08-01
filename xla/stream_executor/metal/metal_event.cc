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
  if (flush_fn_ != nullptr && signal_value_ != 0) {
    flush_fn_(flush_ctx_, signal_value_);
  }
}

Event::Status MetalEvent::PollForStatus() {
  if (signal_value_ == 0 || shared_event_ == nullptr) {
    return Event::Status::kComplete;
  }
  if (SharedEventSignaledValue(shared_event_) >= signal_value_) {
    return Event::Status::kComplete;
  }
  FlushProducerForWait();
  return SharedEventSignaledValue(shared_event_) >= signal_value_
             ? Event::Status::kComplete
             : Event::Status::kPending;
}

absl::Status MetalEvent::Synchronize() {
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
