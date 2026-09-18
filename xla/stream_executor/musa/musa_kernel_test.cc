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

#include "xla/stream_executor/musa/musa_kernel.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/musa/musa_module_reaper.h"
#include "xla/stream_executor/musa/musa_module_use_tracker.h"

namespace stream_executor::musa {
namespace {

using ::testing::_;
using ::testing::Return;
using ::testing::ReturnRef;

class MockTrackedMusaStream : public MockStream, public MusaModuleUseTracker {
 public:
  MOCK_METHOD(absl::Status, RecordModuleUse,
              (std::shared_ptr<MusaModule> module), (override));
  MOCK_METHOD(void, OrphanModuleUse, (std::shared_ptr<MusaModule> module),
              (override));
};

TEST(MusaKernelTest, DestructorReleasesParentModuleOnReaperThread) {
  MusaModuleReaper reaper(/*device_ordinal=*/0);
  absl::Notification released;
  std::atomic<bool> module_released = false;
  std::thread::id release_thread;
  const std::thread::id caller_thread = std::this_thread::get_id();
  auto* module_pointer = reinterpret_cast<MusaModule*>(uintptr_t{0x1234});
  std::shared_ptr<MusaModule> module(
      module_pointer,
      [&released, &module_released, &release_thread](MusaModule*) {
        release_thread = std::this_thread::get_id();
        module_released = true;
        released.Notify();
      });
  std::weak_ptr<MusaModule> weak_module = module;
  MockStreamExecutor executor;
  auto function = reinterpret_cast<MUfunction>(uintptr_t{0x5678});

  {
    MusaKernel kernel(&executor, function, module, &reaper, /*arity=*/3);
    module.reset();

    EXPECT_EQ(kernel.Arity(), 3);
    EXPECT_EQ(kernel.function(), function);
    EXPECT_EQ(kernel.module().get(), module_pointer);
    EXPECT_FALSE(weak_module.expired());
    EXPECT_FALSE(module_released);
  }

  ASSERT_TRUE(released.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_TRUE(weak_module.expired());
  EXPECT_TRUE(module_released);
  EXPECT_NE(release_thread, caller_thread);
}

TEST(MusaKernelTest, PacksScalarsWithoutSynchronizingInDestructor) {
  MusaModuleReaper reaper(/*device_ordinal=*/0);
  std::atomic<bool> module_released = false;
  auto* module_pointer = reinterpret_cast<MusaModule*>(uintptr_t{0x1234});
  std::shared_ptr<MusaModule> module(
      module_pointer,
      [&module_released](MusaModule*) { module_released = true; });
  std::weak_ptr<MusaModule> weak_module = module;
  MockStreamExecutor executor;
  MockTrackedMusaStream stream;
  auto function = reinterpret_cast<MUfunction>(uintptr_t{0x5678});
  DeviceDescription description;
  description.set_shared_memory_per_block(64 * 1024);

  EXPECT_CALL(stream,
              LaunchKernel(_, _, _, function, "scalar_kernel", _, 0, false))
      .WillOnce([](const ThreadDim& threads, const BlockDim& blocks,
                   const std::optional<ClusterDim>&, void*, absl::string_view,
                   void** arguments, int64_t, bool) {
        EXPECT_EQ(threads, ThreadDim(8, 2, 1));
        EXPECT_EQ(blocks, BlockDim(3, 4, 5));
        EXPECT_EQ(*static_cast<int32_t*>(arguments[0]), 7);
        EXPECT_EQ(*static_cast<int64_t*>(arguments[1]), 11);
        EXPECT_FLOAT_EQ(*static_cast<float*>(arguments[2]), 1.5f);
        return absl::OkStatus();
      });
  EXPECT_CALL(stream, parent).WillOnce(testing::Return(&executor));
  EXPECT_CALL(executor, GetDeviceDescription).WillOnce(ReturnRef(description));
  EXPECT_CALL(stream, RecordModuleUse(_)).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(stream, OrphanModuleUse).Times(0);
  EXPECT_CALL(executor, SynchronizeAllActivity).Times(0);

  {
    MusaKernel kernel(&executor, function, module, &reaper, /*arity=*/3);
    kernel.set_name("scalar_kernel");
    module.reset();
    KernelArgsPackedTuple<int32_t, int64_t, float> arguments(
        7, 11, 1.5f, /*shared_memory_bytes=*/0);

    EXPECT_TRUE(kernel
                    .Launch(ThreadDim(8, 2, 1), BlockDim(3, 4, 5),
                            /*cluster_dims=*/std::nullopt, &stream, arguments)
                    .ok());
  }

  reaper.WaitUntilIdleForTesting();
  EXPECT_TRUE(weak_module.expired());
  EXPECT_TRUE(module_released);
}

TEST(MusaKernelTest, RejectsWrongArityWithoutCallingStream) {
  MusaModuleReaper reaper(/*device_ordinal=*/0);
  auto* module_pointer = reinterpret_cast<MusaModule*>(uintptr_t{0x1234});
  std::shared_ptr<MusaModule> module(module_pointer, [](MusaModule*) {});
  MockStreamExecutor executor;
  MockTrackedMusaStream stream;
  MusaKernel kernel(&executor, reinterpret_cast<MUfunction>(uintptr_t{0x5678}),
                    module, &reaper,
                    /*arity=*/2);
  kernel.set_name("arity_kernel");
  KernelArgsPackedTuple<int32_t> arguments(7, /*shared_memory_bytes=*/0);

  EXPECT_CALL(stream, LaunchKernel).Times(0);
  EXPECT_CALL(stream, RecordModuleUse).Times(0);
  EXPECT_CALL(stream, OrphanModuleUse).Times(0);
  EXPECT_CALL(stream, parent).WillOnce(testing::Return(&executor));
  absl::Status status =
      kernel.Launch(ThreadDim(), BlockDim(), std::nullopt, &stream, arguments);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(MusaKernelTest, AmbiguousLaunchFailureOrphansModule) {
  MusaModuleReaper reaper(/*device_ordinal=*/0);
  auto* module_pointer = reinterpret_cast<MusaModule*>(uintptr_t{0x1234});
  std::shared_ptr<MusaModule> module(module_pointer, [](MusaModule*) {});
  MockStreamExecutor executor;
  MockTrackedMusaStream stream;
  MusaKernel kernel(&executor, reinterpret_cast<MUfunction>(uintptr_t{0x5678}),
                    module, &reaper,
                    /*arity=*/1);
  kernel.set_name("failing_kernel");
  KernelArgsPackedTuple<int32_t> arguments(7, /*shared_memory_bytes=*/0);
  DeviceDescription description;
  description.set_shared_memory_per_block(64 * 1024);

  EXPECT_CALL(stream, parent).WillOnce(Return(&executor));
  EXPECT_CALL(executor, GetDeviceDescription).WillOnce(ReturnRef(description));
  EXPECT_CALL(stream, LaunchKernel)
      .WillOnce(Return(absl::InternalError("ambiguous driver failure")));
  EXPECT_CALL(stream, RecordModuleUse).Times(0);
  EXPECT_CALL(stream, OrphanModuleUse(_)).Times(1);

  absl::Status status =
      kernel.Launch(ThreadDim(), BlockDim(), std::nullopt, &stream, arguments);
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
}

}  // namespace
}  // namespace stream_executor::musa
