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

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/core/status_test_util.h"
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
  // NUMA discovery returns -1 when sysfs topology is unavailable.
  EXPECT_GE(description->numa_node(), -1);
  EXPECT_GE(description->device_interconnect_info().active_links, 0);
  EXPECT_TRUE(
      description->device_interconnect_info().cluster_uuid.empty());
  EXPECT_TRUE(description->device_interconnect_info().clique_id.empty());
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

TEST(MusaExecutorTest, SelfPeerTopologyIsStableAndIdempotent) {
  MusaExecutor executor(/*platform=*/nullptr, /*device_ordinal=*/0);
  TF_ASSERT_OK_AND_ASSIGN(int device_count,
                          MusaDriver::Instance().DeviceCount());
  TF_ASSERT_OK(executor.Init());

  EXPECT_TRUE(executor.CanEnablePeerAccessTo(/*other_device_ordinal=*/0));
  EXPECT_TRUE(executor.CanEnablePeerAccessTo(&executor));
  EXPECT_FALSE(executor.CanEnablePeerAccessTo(/*other_device_ordinal=*/-1));
  EXPECT_FALSE(executor.CanEnablePeerAccessTo(device_count));
  TF_ASSERT_OK(executor.EnablePeerAccessTo(&executor));
  TF_ASSERT_OK(executor.EnablePeerAccessTo(&executor));

  TF_ASSERT_OK_AND_ASSIGN(std::string status,
                          executor.GetInterconnectStatus());
  EXPECT_EQ(status.find("MUSA peer topology v1: source=0"), 0);
  EXPECT_NE(status.find(
                "; peer=0,access=1,details=0,performance_rank=0,"
                "native_atomic=0,musa_array=0,mtlink_ports=0"),
            std::string::npos);
  EXPECT_EQ(executor.numa_node(),
            executor.GetDeviceDescription().numa_node());
}

TEST(MusaExecutorTest, CrossExecutorEventOrdersSameDeviceD2DCopy) {
  MusaExecutor source_executor(/*platform=*/nullptr, /*device_ordinal=*/0);
  MusaExecutor destination_executor(/*platform=*/nullptr,
                                    /*device_ordinal=*/0);
  TF_ASSERT_OK(source_executor.Init());
  TF_ASSERT_OK(destination_executor.Init());
  TF_ASSERT_OK(destination_executor.EnablePeerAccessTo(&source_executor));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> producer,
                          source_executor.CreateStream(std::nullopt));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> consumer,
                          destination_executor.CreateStream(std::nullopt));

  constexpr uint64_t kBytes = 8 * sizeof(uint32_t);
  DeviceAddressBase source =
      source_executor.Allocate(kBytes, /*memory_space=*/0);
  DeviceAddressBase destination =
      destination_executor.Allocate(kBytes, /*memory_space=*/0);
  ASSERT_FALSE(source.is_null());
  ASSERT_FALSE(destination.is_null());

  const std::array<uint32_t, 8> input = {
      0x01020304, 0x11121314, 0x21222324, 0x31323334,
      0x41424344, 0x51525354, 0x61626364, 0x71727374,
  };
  TF_ASSERT_OK_AND_ASSIGN(MUcontext context_before,
                          MusaDriver::Instance().CurrentContext());
  TF_ASSERT_OK(producer->Memcpy(&source, input.data(), kBytes));
  TF_ASSERT_OK(consumer->WaitFor(producer.get()));
  TF_ASSERT_OK(consumer->Memcpy(&destination, source, kBytes));
  TF_ASSERT_OK(consumer->BlockHostUntilDone());
  TF_ASSERT_OK_AND_ASSIGN(MUcontext context_after,
                          MusaDriver::Instance().CurrentContext());
  EXPECT_EQ(context_after, context_before);

  std::array<uint32_t, 8> output = {};
  std::array<uint32_t, 8> preserved_source = {};
  TF_ASSERT_OK(destination_executor.SynchronousMemcpy(
      output.data(), destination, kBytes));
  TF_ASSERT_OK(source_executor.SynchronousMemcpy(
      preserved_source.data(), source, kBytes));
  EXPECT_EQ(output, input);
  EXPECT_EQ(preserved_source, input);

  source_executor.Deallocate(&source);
  destination_executor.Deallocate(&destination);
}

TEST(MusaExecutorTest, BidirectionalPeerCopyWhenTwoDevicesAvailable) {
  MusaDriver& driver = MusaDriver::Instance();
  TF_ASSERT_OK_AND_ASSIGN(int device_count, driver.DeviceCount());
  if (device_count < 2) {
    GTEST_SKIP() << "Two MUSA devices are required for physical peer-copy "
                    "qualification";
  }
  {
    MusaExecutor first(/*platform=*/nullptr, /*device_ordinal=*/0);
    MusaExecutor second(/*platform=*/nullptr, /*device_ordinal=*/1);
    TF_ASSERT_OK(first.Init());
    TF_ASSERT_OK(second.Init());
    std::unique_ptr<ActivateContext> anchor = first.Activate();
    TF_ASSERT_OK_AND_ASSIGN(MUcontext anchor_context,
                            driver.CurrentContext());

    if (!first.CanEnablePeerAccessTo(&second) ||
        !second.CanEnablePeerAccessTo(&first)) {
      GTEST_SKIP() << "Visible MUSA devices do not expose bidirectional peer "
                      "access";
    }
    TF_ASSERT_OK(first.EnablePeerAccessTo(&second));
    TF_ASSERT_OK(first.EnablePeerAccessTo(&second));
    TF_ASSERT_OK(second.EnablePeerAccessTo(&first));
    TF_ASSERT_OK(second.EnablePeerAccessTo(&first));

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> first_stream,
                            first.CreateStream(std::nullopt));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> second_stream,
                            second.CreateStream(std::nullopt));

    constexpr uint64_t kBytes = 8 * sizeof(uint32_t);
    DeviceAddressBase source_on_first = first.Allocate(kBytes, 0);
    DeviceAddressBase destination_on_first = first.Allocate(kBytes, 0);
    DeviceAddressBase source_on_second = second.Allocate(kBytes, 0);
    DeviceAddressBase destination_on_second = second.Allocate(kBytes, 0);
    ASSERT_FALSE(source_on_first.is_null());
    ASSERT_FALSE(destination_on_first.is_null());
    ASSERT_FALSE(source_on_second.is_null());
    ASSERT_FALSE(destination_on_second.is_null());

    const std::array<uint32_t, 8> first_pattern = {
        0x00112233, 0x10213243, 0x20314253, 0x30415263,
        0x40516273, 0x50617283, 0x60718293, 0x708192a3,
    };
    const std::array<uint32_t, 8> second_pattern = {
        0xffeeddcc, 0xefdecdbc, 0xdfcebdac, 0xcfbead9c,
        0xbfae9d8c, 0xaf9e8d7c, 0x9f8e7d6c, 0x8f7e6d5c,
    };
    TF_ASSERT_OK(
        first_stream->Memcpy(&source_on_first, first_pattern.data(), kBytes));
    TF_ASSERT_OK(second_stream->Memcpy(&source_on_second,
                                       second_pattern.data(), kBytes));

    TF_ASSERT_OK(first_stream->WaitFor(second_stream.get()));
    TF_ASSERT_OK(first_stream->Memcpy(&destination_on_first, source_on_second,
                                      kBytes));
    TF_ASSERT_OK(second_stream->WaitFor(first_stream.get()));
    TF_ASSERT_OK(second_stream->Memcpy(&destination_on_second, source_on_first,
                                       kBytes));
    TF_ASSERT_OK(first_stream->BlockHostUntilDone());
    TF_ASSERT_OK(second_stream->BlockHostUntilDone());

    std::array<uint32_t, 8> copied_to_first = {};
    std::array<uint32_t, 8> copied_to_second = {};
    std::array<uint32_t, 8> preserved_first = {};
    std::array<uint32_t, 8> preserved_second = {};
    TF_ASSERT_OK(first.SynchronousMemcpy(copied_to_first.data(),
                                         destination_on_first, kBytes));
    TF_ASSERT_OK(second.SynchronousMemcpy(copied_to_second.data(),
                                          destination_on_second, kBytes));
    TF_ASSERT_OK(first.SynchronousMemcpy(preserved_first.data(),
                                         source_on_first, kBytes));
    TF_ASSERT_OK(second.SynchronousMemcpy(preserved_second.data(),
                                          source_on_second, kBytes));
    EXPECT_EQ(copied_to_first, second_pattern);
    EXPECT_EQ(copied_to_second, first_pattern);
    EXPECT_EQ(preserved_first, first_pattern);
    EXPECT_EQ(preserved_second, second_pattern);

    first.Deallocate(&source_on_first);
    first.Deallocate(&destination_on_first);
    second.Deallocate(&source_on_second);
    second.Deallocate(&destination_on_second);
    TF_ASSERT_OK_AND_ASSIGN(MUcontext context_after_operations,
                            driver.CurrentContext());
    EXPECT_EQ(context_after_operations, anchor_context);
  }
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

TEST(MusaExecutorTest, HostMemoryAllocatorReturnsBfcAlignedMemory) {
  MusaExecutor executor(/*platform=*/nullptr, /*device_ordinal=*/0);
  ASSERT_TRUE(executor.Init().ok());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocator> allocator,
                          executor.CreateMemoryAllocator(MemorySpace::kHost));

  for (uint64_t size : {UINT64_C(1), UINT64_C(4096), UINT64_C(1) << 20}) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                            allocator->Allocate(size));
    ASSERT_FALSE(allocation->address().is_null());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(allocation->address().opaque()) &
                  (uintptr_t{256} - 1),
              0);
    EXPECT_EQ(allocation->address().size(), size);
  }
}

}  // namespace
}  // namespace stream_executor::musa
