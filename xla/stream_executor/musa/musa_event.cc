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

#include "xla/stream_executor/musa/musa_event.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/musa/musa_runtime.h"

namespace stream_executor::musa {

Event::Status MusaEvent::PollForStatus() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  int result = MusaRuntime::Get()->EventQuery(handle_);
  if (result == musaSuccess) {
    return Event::Status::kComplete;
  }
  if (result == musaErrorNotReady) {
    return Event::Status::kPending;
  }
  return Event::Status::kError;
}

absl::Status MusaEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return MusaRuntime::Get()->StreamWaitEvent(reinterpret_cast<void*>(stream),
                                             handle_);
}

absl::Status MusaEvent::Synchronize() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return MusaRuntime::Get()->EventSynchronize(handle_);
}

absl::StatusOr<MusaEvent> MusaEvent::Create(StreamExecutor* executor,
                                            bool enable_timing) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  auto event = MusaRuntime::Get()->EventCreate(enable_timing);
  if (!event.ok()) return event.status();
  return MusaEvent(executor, *event);
}

MusaEvent::~MusaEvent() {
  if (handle_ != nullptr) {
    std::unique_ptr<ActivateContext> activation = executor_->Activate();
    (void)MusaRuntime::Get()->EventDestroy(handle_);
    handle_ = nullptr;
  }
}

MusaEvent::MusaEvent(MusaEvent&& other)
    : executor_(other.executor_), handle_(other.handle_) {
  other.executor_ = nullptr;
  other.handle_ = nullptr;
}

MusaEvent& MusaEvent::operator=(MusaEvent&& other) {
  if (this != &other) {
    if (handle_ != nullptr) {
      std::unique_ptr<ActivateContext> activation = executor_->Activate();
      (void)MusaRuntime::Get()->EventDestroy(handle_);
    }
    executor_ = other.executor_;
    handle_ = other.handle_;
    other.executor_ = nullptr;
    other.handle_ = nullptr;
  }
  return *this;
}

}  // namespace stream_executor::musa
