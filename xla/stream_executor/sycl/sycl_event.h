/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_EVENT_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_EVENT_H_

#include <sycl/sycl.hpp>

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::sycl {

// This class implements the Event class for SYCL devices.
// It is not thread-safe and should be used in a single-threaded context.
class SyclEvent : public Event {
 public:
  Event::Status PollForStatus() override;

  // Submits a dependency on event to stream_handle so subsequent work
  // on that stream waits for event to complete. If event is already
  // complete at call time, returns immediately without submitting any work
  // to stream_handle.
  // TODO(intel-tf): Remove unused executor parameter.
  static absl::Status WaitStreamOnEvent(StreamExecutor* executor,
                                        ::sycl::queue* stream_handle,
                                        const SyclEvent& event);

  // Waits for the event to complete on an external stream.
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream) override;

  // Creates an unrecorded SyclEvent instance. Unrecorded events poll complete
  // and waits on them are no-ops to preserve StreamExecutor compatibility.
  static absl::StatusOr<SyclEvent> Create(StreamExecutor* executor);

  // Returns the recorded underlying SYCL event, or an error if this event has
  // not been recorded yet. Not thread-safe.
  absl::StatusOr<::sycl::event> GetRecordedEvent() const;

  bool IsRecorded() const { return event_.has_value(); }

  // Blocks the host until this event completes.
  absl::Status Wait();

  // Blocks the host until this event completes.
  absl::Status Synchronize() override { return Wait(); }

  // Sets the underlying SYCL event and records the queue provenance used for
  // future waits. Not thread-safe.
  absl::Status SetRecordedEvent(const ::sycl::event& event,
                                const ::sycl::queue* queue,
                                bool is_barrier_marker);

  // We don't need a destructor for ::sycl::event since it is handled by the
  // SYCL runtime.
  ~SyclEvent() = default;

  // Ensure SyclEvent is moveable but not copyable.
  SyclEvent(const SyclEvent&) = delete;
  SyclEvent& operator=(const SyclEvent&) = delete;
  SyclEvent(SyclEvent&& other) noexcept;
  SyclEvent& operator=(SyclEvent&& other) noexcept;

 private:
  struct RecordedEventMetadata {
    StreamExecutor* executor;
    int device_ordinal;
    std::uintptr_t queue_identity;
    ::sycl::context context;
    ::sycl::device device;
    ::sycl::backend backend;
    bool is_barrier_marker;
  };

  explicit SyclEvent(StreamExecutor* executor) : executor_(executor) {}

  // The Executor used to which this object and ::sycl::event are bound.
  StreamExecutor* executor_;

  // The underlying SYCL event. Empty means this StreamExecutor event has never
  // been recorded.
  std::optional<::sycl::event> event_;
  std::optional<RecordedEventMetadata> metadata_;
};

}  // namespace stream_executor::sycl

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_EVENT_H_
