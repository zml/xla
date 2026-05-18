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

Event::Status MetalEvent::PollForStatus() {
  return PollCommandBufferStatus(command_buffer_);
}

absl::Status MetalEvent::Synchronize() {
  return WaitUntilCompleted(command_buffer_);
}

void MetalEvent::SetCommandBuffer(void* command_buffer) {
  if (command_buffer_ != nullptr) {
    ReleaseObject(command_buffer_);
  }
  command_buffer_ = command_buffer;
}

}  // namespace stream_executor::metal
