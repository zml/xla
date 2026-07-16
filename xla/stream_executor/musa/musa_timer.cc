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

#include "xla/stream_executor/musa/musa_timer.h"

#include <cmath>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_runtime.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

MusaTimer::MusaTimer(StreamExecutor* executor, Stream* stream,
                     MusaEvent start_event, MusaEvent stop_event)
    : executor_(executor),
      stream_(stream),
      start_event_(std::move(start_event)),
      stop_event_(std::move(stop_event)) {}

absl::StatusOr<absl::Duration> MusaTimer::GetElapsedDuration() {
  if (is_stopped_) {
    return absl::FailedPreconditionError("Measuring inactive MUSA timer");
  }
  RETURN_IF_ERROR(stream_->RecordEvent(&stop_event_));
  RETURN_IF_ERROR(stop_event_.Synchronize());

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  TF_ASSIGN_OR_RETURN(float elapsed_milliseconds,
                      MusaRuntime::Get()->EventElapsedTime(
                          start_event_.handle(), stop_event_.handle()));
  if (!std::isfinite(elapsed_milliseconds) || elapsed_milliseconds < 0.0f) {
    return absl::InternalError(
        "musaEventElapsedTime returned an invalid duration");
  }
  is_stopped_ = true;
  return absl::Milliseconds(elapsed_milliseconds);
}

absl::StatusOr<MusaTimer> MusaTimer::Create(StreamExecutor* executor,
                                            Stream* stream) {
  if (executor == nullptr || stream == nullptr) {
    return absl::InvalidArgumentError(
        "MUSA timer executor and stream must not be null");
  }
  TF_ASSIGN_OR_RETURN(MusaEvent start_event,
                      MusaEvent::Create(executor, /*enable_timing=*/true));
  TF_ASSIGN_OR_RETURN(MusaEvent stop_event,
                      MusaEvent::Create(executor, /*enable_timing=*/true));
  RETURN_IF_ERROR(stream->RecordEvent(&start_event));
  return MusaTimer(executor, stream, std::move(start_event),
                   std::move(stop_event));
}

}  // namespace stream_executor::musa
