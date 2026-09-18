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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

class MusaEvent : public Event {
 public:
  Event::Status PollForStatus() override;
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream) override;
  absl::Status Synchronize() override;

  static absl::StatusOr<MusaEvent> Create(StreamExecutor* executor,
                                          bool enable_timing = false);

  void* handle() const { return handle_; }

  ~MusaEvent() override;
  MusaEvent(const MusaEvent&) = delete;
  MusaEvent& operator=(const MusaEvent&) = delete;
  MusaEvent(MusaEvent&& other);
  MusaEvent& operator=(MusaEvent&& other);

 private:
  MusaEvent(StreamExecutor* executor, void* handle)
      : executor_(executor), handle_(handle) {}

  StreamExecutor* executor_;
  void* handle_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_
