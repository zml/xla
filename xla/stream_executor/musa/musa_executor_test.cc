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

#include "xla/stream_executor/musa/musa_executor.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(MusaExecutorTest, S80DeviceDescriptionUsesLiveQueries) {
  MusaDriver& driver = MusaDriver::Instance();
  TF_ASSERT_OK_AND_ASSIGN(MUcontext context_before, driver.CurrentContext());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceDescription> description,
                          MusaExecutor::CreateDeviceDescription(0));
  TF_ASSERT_OK_AND_ASSIGN(MUcontext context_after, driver.CurrentContext());
  EXPECT_EQ(context_after, context_before);

  EXPECT_EQ(description->name(), "MTT S80");
  EXPECT_EQ(description->device_vendor(), "Moore Threads");
  EXPECT_FALSE(description->pci_bus_id().empty());
  EXPECT_GT(description->runtime_version(), (SemanticVersion{0, 0, 0}));
  EXPECT_GT(description->driver_version(), (SemanticVersion{0, 0, 0}));
  EXPECT_EQ(description->compile_time_toolkit_version(),
            (SemanticVersion{4, 0, 1}));
  EXPECT_GE(description->thread_dim_limit().x, 1024);
  EXPECT_GE(description->thread_dim_limit().y, 1);
  EXPECT_GE(description->thread_dim_limit().z, 1);
  EXPECT_EQ(description->threads_per_block_limit(), 1024);
  EXPECT_GT(description->threads_per_core_limit(), 0);
  EXPECT_EQ(description->threads_per_warp(), 128);
  EXPECT_GT(description->core_count(), 0);
  EXPECT_GT(description->registers_per_block_limit(), 0);
  EXPECT_GT(description->registers_per_core_limit(), 0);
  EXPECT_GT(description->shared_memory_per_block(), 0);
  EXPECT_GT(description->shared_memory_per_core(), 0);
  EXPECT_GT(description->l2_cache_size(), 0);
  EXPECT_GT(description->device_memory_size(), 0);
  EXPECT_GT(description->memory_bandwidth(), 0);

  const MusaComputeCapability* capability =
      description->gpu_compute_capability().musa_compute_capability();
  ASSERT_NE(capability, nullptr);
  EXPECT_EQ(capability->architecture(), "mp_21");
  EXPECT_EQ(capability->major(), 2);
  EXPECT_EQ(capability->minor(), 1);
  EXPECT_EQ(capability->hardware_warp_size(), 128);
  EXPECT_EQ(capability->logical_subgroup_size(), 32);

  EXPECT_THAT(DeviceDescription::FromProto(description->ToProto()),
              IsOkAndHolds(*description));
}

TEST(MusaExecutorTest, S80PrimaryContextInitializesActivatesAndTearsDown) {
  MusaDriver& driver = MusaDriver::Instance();
  {
    MusaExecutor executor(/*platform=*/nullptr, /*device_ordinal=*/0);
    absl::Status status = executor.Init();
    ASSERT_TRUE(status.ok()) << status;

    std::unique_ptr<ActivateContext> activation = executor.Activate();
    auto current_context = driver.CurrentContext();
    ASSERT_TRUE(current_context.ok()) << current_context.status();
    EXPECT_NE(*current_context, nullptr);

    auto current_device = driver.CurrentDevice();
    ASSERT_TRUE(current_device.ok()) << current_device.status();
    EXPECT_EQ(*current_device, 0);
    EXPECT_TRUE(executor.SynchronizeAllActivity());

    int64_t free_bytes = 0;
    int64_t total_bytes = 0;
    EXPECT_TRUE(executor.DeviceMemoryUsage(&free_bytes, &total_bytes));
    EXPECT_GT(free_bytes, 0);
    EXPECT_GT(total_bytes, 0);
    EXPECT_LE(free_bytes, total_bytes);

    constexpr int kThreadCount = 4;
    constexpr int kIterations = 25;
    std::atomic<bool> all_workers_activated = true;
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; ++i) {
      threads.emplace_back([&] {
        for (int j = 0; j < kIterations; ++j) {
          std::unique_ptr<ActivateContext> worker_activation =
              executor.Activate();
          auto worker_context = driver.CurrentContext();
          auto worker_device = driver.CurrentDevice();
          if (!worker_context.ok() || *worker_context == nullptr ||
              !worker_device.ok() || *worker_device != 0) {
            all_workers_activated = false;
          }
        }
      });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
    EXPECT_TRUE(all_workers_activated);
  }

  // libmusart retains the process primary context and the S80 driver can report
  // it as current again after our driver reference is released. The hermetic
  // MusaContext tests verify the exact clear/release call balance; here we
  // verify that teardown leaves the driver usable.
  EXPECT_TRUE(driver.CurrentContext().ok());
}

TEST(MusaExecutorTest, DeviceAllocationRetainsContextPastExecutorLifetime) {
  std::unique_ptr<MemoryAllocator> allocator;
  std::unique_ptr<MemoryAllocation> allocation;
  {
    MusaExecutor executor(/*platform=*/nullptr, /*device_ordinal=*/0);
    ASSERT_TRUE(executor.Init().ok());
    TF_ASSERT_OK_AND_ASSIGN(
        allocator, executor.CreateMemoryAllocator(MemorySpace::kDevice));
    TF_ASSERT_OK_AND_ASSIGN(allocation, allocator->Allocate(256));
    EXPECT_FALSE(allocation->address().is_null());
  }

  // Both callbacks own the shared primary context, so this free remains valid
  // after the executor itself has been destroyed.
  allocation.reset();
  allocator.reset();
  EXPECT_TRUE(MusaDriver::Instance().CurrentContext().ok());
}

}  // namespace
}  // namespace stream_executor::musa
