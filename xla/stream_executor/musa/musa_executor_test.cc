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

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(MusaExecutorTest, S80DeviceDescriptionUsesLiveQueries) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceDescription> description,
                          MusaExecutor::CreateDeviceDescription(0));

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

}  // namespace
}  // namespace stream_executor::musa
