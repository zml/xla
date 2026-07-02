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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor::sycl {
namespace {

const int kDefaultDeviceOrdinal = 0;

class SyclEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        Platform * platform,
        stream_executor::PlatformManager::PlatformWithId(kSyclPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(executor_,
                            platform->ExecutorForDevice(kDefaultDeviceOrdinal));
  }

  StreamExecutor* executor_;
};

// TODO (intel-tf): Add a test for events with dependencies once SyclStream
// class is supported.
TEST_F(SyclEventTest, CreateEvent) {
  TF_ASSERT_OK_AND_ASSIGN(SyclEvent event, SyclEvent::Create(executor_));

  EXPECT_FALSE(event.IsRecorded());
  EXPECT_THAT(event.GetRecordedEvent(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  // Unrecorded events are treated as complete to preserve StreamExecutor
  // compatibility.
  EXPECT_EQ(event.PollForStatus(), Event::Status::kComplete);
  EXPECT_THAT(event.Synchronize(), absl_testing::IsOk());
}

TEST_F(SyclEventTest, MoveEvent) {
  TF_ASSERT_OK_AND_ASSIGN(SyclEvent orig_sycl_event,
                          SyclEvent::Create(executor_));

  // Move the event to a new SyclEvent instance.
  SyclEvent moved_sycl_event = std::move(orig_sycl_event);

  EXPECT_FALSE(moved_sycl_event.IsRecorded());
  EXPECT_EQ(moved_sycl_event.PollForStatus(), Event::Status::kComplete);
}

TEST_F(SyclEventTest, WaitReturnsStoredAsyncError) {
  TF_ASSERT_OK_AND_ASSIGN(SyclEvent event, SyclEvent::Create(executor_));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor_->CreateStream());
  TF_ASSERT_OK(stream->RecordEvent(&event));
  SyclRecordAsyncErrorForTesting(
      kDefaultDeviceOrdinal,
      absl::InternalError("injected async event failure"));

  EXPECT_THAT(event.Wait(),
              absl_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  ::testing::HasSubstr("injected async event failure")));
}

TEST_F(SyclEventTest, SynchronizeReturnsStoredAsyncError) {
  TF_ASSERT_OK_AND_ASSIGN(SyclEvent event, SyclEvent::Create(executor_));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor_->CreateStream());
  TF_ASSERT_OK(stream->RecordEvent(&event));
  Event* base_event = &event;
  SyclRecordAsyncErrorForTesting(
      kDefaultDeviceOrdinal,
      absl::InternalError("injected async base event failure"));

  EXPECT_THAT(base_event->Synchronize(),
              absl_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  ::testing::HasSubstr("injected async base event failure")));
}

TEST_F(SyclEventTest, ExternalStreamNullIsRejected) {
  TF_ASSERT_OK_AND_ASSIGN(SyclEvent event, SyclEvent::Create(executor_));

  EXPECT_THAT(event.WaitForEventOnExternalStream(0),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SyclEventTest, ExternalStreamSameContextSucceeds) {
  TF_ASSERT_OK_AND_ASSIGN(SyclEvent event, SyclEvent::Create(executor_));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor_->CreateStream());
  TF_ASSERT_OK(stream->RecordEvent(&event));

  auto handle = stream->platform_specific_handle();
  ASSERT_NE(handle.stream, nullptr);
  EXPECT_THAT(event.WaitForEventOnExternalStream(
                  reinterpret_cast<std::intptr_t>(handle.stream)),
              absl_testing::IsOk());
}

}  // namespace

}  // namespace stream_executor::sycl
