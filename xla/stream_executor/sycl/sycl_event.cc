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

#include "xla/stream_executor/sycl/sycl_event.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::sycl {

Event::Status SyclEvent::PollForStatus() {
  if (!event_.has_value()) {
    return Event::Status::kComplete;
  }

  try {
    auto event_status =
        event_->get_info<::sycl::info::event::command_execution_status>();
    switch (event_status) {
      case ::sycl::info::event_command_status::submitted: {
        VLOG(2)
            << "Command is submitted to the queue but not yet running on the "
               "device.";
        return Event::Status::kPending;
      }
      case ::sycl::info::event_command_status::running: {
        VLOG(2) << "Command has started running on the device but has not yet "
                   "completed.";
        return Event::Status::kPending;
      }
      case ::sycl::info::event_command_status::complete: {
        VLOG(2) << "Command has finished running on the device.";
        return Event::Status::kComplete;
      }
      default: {
        LOG(ERROR) << "Event status is unknown: "
                   << static_cast<int>(event_status);
        return Event::Status::kUnknown;
      }
    }
  } catch (const ::sycl::exception& e) {
    LOG(ERROR) << "SYCL exception while polling event status: " << e.what()
               << " (error code: " << e.code() << ")";
    return Event::Status::kError;
  }
}

absl::Status SyclEvent::WaitStreamOnEvent(StreamExecutor* executor,
                                          ::sycl::queue* stream_handle,
                                          const SyclEvent& event) {
  // No need to call executor->Activate() since the SYCL context need not
  // be activated explicitly.
  if (stream_handle == nullptr) {
    return absl::InternalError(
        "WaitStreamOnEvent: Stream handle is not initialized.");
  }
  if (!event.event_.has_value()) {
    return absl::OkStatus();
  }
  if (!event.metadata_.has_value()) {
    return absl::InternalError(
        "WaitStreamOnEvent: Recorded event is missing metadata.");
  }

  const RecordedEventMetadata& metadata = *event.metadata_;
  try {
    if (stream_handle->get_backend() != metadata.backend) {
      return absl::InvalidArgumentError(
          "WaitStreamOnEvent: Target stream backend does not match recorded "
          "event backend.");
    }

    ::sycl::context target_context = stream_handle->get_context();
    if (target_context != metadata.context) {
      return absl::InvalidArgumentError(
          "WaitStreamOnEvent: Target stream context does not match recorded "
          "event context.");
    }

    ::sycl::device target_device = stream_handle->get_device();
    if (target_device != metadata.device) {
      return absl::UnimplementedError(
          "WaitStreamOnEvent: Cross-device SYCL event waits are not enabled.");
    }

    if (executor != nullptr &&
        executor->device_ordinal() != metadata.device_ordinal) {
      return absl::InvalidArgumentError(absl::StrCat(
          "WaitStreamOnEvent: Target executor ordinal ",
          executor->device_ordinal(), " does not match recorded event ordinal ",
          metadata.device_ordinal, "."));
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        "WaitStreamOnEvent: Failed to validate target stream before waiting: " +
        std::string(e.what()));
  }

  try {
    auto event_status =
        event.event_->get_info<::sycl::info::event::command_execution_status>();
    if (event_status == ::sycl::info::event_command_status::complete) {
      VLOG(2) << "Event is already complete; no need to wait.";
      return absl::OkStatus();
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        "WaitStreamOnEvent: Failed to check event status before waiting: " +
        std::string(e.what()));
  }
  try {
    stream_handle->ext_oneapi_submit_barrier(
        std::vector<::sycl::event>{*event.event_});
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        "WaitStreamOnEvent: Failed to submit event barrier: " +
        std::string(e.what()));
  }
  return absl::OkStatus();
}

absl::Status SyclEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  if (stream == 0) {
    return absl::InvalidArgumentError(
        "WaitForEventOnExternalStream: external stream is null.");
  }
  auto* queue_ptr = reinterpret_cast<::sycl::queue*>(stream);
  return WaitStreamOnEvent(executor_, queue_ptr, *this);
}

absl::Status SyclEvent::Wait() {
  if (!event_.has_value()) {
    return absl::OkStatus();
  }
  return SyclEventSynchronize(*event_, executor_->device_ordinal(),
                              "SyclEvent::Wait");
}

absl::StatusOr<SyclEvent> SyclEvent::Create(StreamExecutor* executor) {
  // SYCL reports synchronous (host-side) errors via exceptions, so we catch
  // them and return an error status.
  try {
    return SyclEvent(executor);
  } catch (const ::sycl::exception& e) {
    LOG(ERROR) << "SYCL exception while creating event: " << e.what()
               << " (error code: " << e.code() << ")";
    return absl::InternalError(
        absl::StrCat("Failed to create SYCL event: ", e.what()));
  }
}

absl::StatusOr<::sycl::event> SyclEvent::GetRecordedEvent() const {
  if (!event_.has_value()) {
    return absl::FailedPreconditionError(
        "SyclEvent::GetRecordedEvent: event has not been recorded.");
  }
  return *event_;
}

absl::Status SyclEvent::SetRecordedEvent(const ::sycl::event& event,
                                         const ::sycl::queue* queue,
                                         bool is_barrier_marker) {
  if (queue == nullptr) {
    return absl::InvalidArgumentError(
        "SyclEvent::SetRecordedEvent: queue is null.");
  }

  try {
    metadata_ = RecordedEventMetadata{
        executor_,
        executor_->device_ordinal(),
        reinterpret_cast<std::uintptr_t>(queue),
        queue->get_context(),
        queue->get_device(),
        queue->get_backend(),
        is_barrier_marker,
    };
    event_ = event;
    return absl::OkStatus();
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("SyclEvent::SetRecordedEvent: failed to record event "
                     "metadata: ",
                     e.what()));
  }
}

SyclEvent::SyclEvent(SyclEvent&& other) noexcept
    : executor_(other.executor_),
      event_(std::move(other.event_)),
      metadata_(std::move(other.metadata_)) {
  other.executor_ = nullptr;
}

SyclEvent& SyclEvent::operator=(SyclEvent&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  executor_ = other.executor_;
  event_ = std::move(other.event_);
  metadata_ = std::move(other.metadata_);
  other.executor_ = nullptr;
  return *this;
}

}  // namespace stream_executor::sycl
