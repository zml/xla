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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_TIMER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_TIMER_H_

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/musa/musa_event.h"

namespace stream_executor {

class Stream;
class StreamExecutor;

namespace musa {

class MusaTimer final : public EventBasedTimer {
 public:
  MusaTimer(MusaTimer&&) = default;
  MusaTimer& operator=(MusaTimer&&) = default;

  absl::StatusOr<absl::Duration> GetElapsedDuration() override;

  static absl::StatusOr<MusaTimer> Create(StreamExecutor* executor,
                                          Stream* stream);

 private:
  MusaTimer(StreamExecutor* executor, Stream* stream, MusaEvent start_event,
            MusaEvent stop_event);

  bool is_stopped_ = false;
  StreamExecutor* executor_;
  Stream* stream_;
  MusaEvent start_event_;
  MusaEvent stop_event_;
};

}  // namespace musa
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_TIMER_H_
