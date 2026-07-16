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

#include "xla/stream_executor/musa/musa_device_description.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_device_properties.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

MusaDeviceProperties S80Properties() {
  MusaDeviceProperties properties;
  properties.name = "MTT S80";
  properties.pci_bus_id = "0000:01:00.0";
  properties.total_memory_bytes = 17089384448ULL;
  properties.compute_capability_major = 2;
  properties.compute_capability_minor = 1;
  properties.max_threads_per_block = 1024;
  properties.max_block_dim_x = 1024;
  properties.max_block_dim_y = 1024;
  properties.max_block_dim_z = 1024;
  properties.max_grid_dim_x = 2147483647;
  properties.max_grid_dim_y = 2147483647;
  properties.max_grid_dim_z = 2147483647;
  properties.max_shared_memory_per_block = 28672;
  properties.max_shared_memory_per_multiprocessor = 28672;
  properties.max_shared_memory_per_block_optin = 28672;
  properties.reserved_shared_memory_per_block = 0;
  properties.max_registers_per_block = 262144;
  properties.max_registers_per_multiprocessor = 131072;
  properties.max_threads_per_multiprocessor = 6144;
  properties.max_blocks_per_multiprocessor = 1024;
  properties.multiprocessor_count = 32;
  properties.hardware_warp_size = 128;
  properties.clock_rate_khz = 1800000;
  properties.memory_clock_rate_khz = 7000000;
  properties.memory_bus_width_bits = 256;
  properties.l2_cache_size_bytes = 25165824;
  properties.texture_alignment_bytes = 128;
  properties.texture_pitch_alignment_bytes = 128;
  properties.total_constant_memory_bytes = 8192;
  return properties;
}

TEST(MusaDeviceDescriptionTest, BuildsMeasuredS80Contract) {
  MusaDeviceVersions versions{
      .runtime_api = 10504,
      .driver_api = 10504,
      .compile_time_toolkit = 40001,
      .kernel_mode_driver = SemanticVersion{3, 0, 0},
  };
  ASSERT_OK_AND_ASSIGN(DeviceDescription description,
                       BuildMusaDeviceDescription(S80Properties(), versions));

  EXPECT_EQ(description.name(), "MTT S80");
  EXPECT_EQ(description.pci_bus_id(), "0000:01:00.0");
  EXPECT_EQ(description.runtime_version(), (SemanticVersion{1, 5, 4}));
  EXPECT_EQ(description.driver_version(), (SemanticVersion{1, 5, 4}));
  EXPECT_EQ(description.kernel_mode_driver_version(),
            (SemanticVersion{3, 0, 0}));
  EXPECT_EQ(description.compile_time_toolkit_version(),
            (SemanticVersion{4, 0, 1}));
  EXPECT_EQ(description.thread_dim_limit(), ThreadDim(1024, 1024, 1024));
  EXPECT_EQ(description.block_dim_limit(),
            BlockDim(2147483647, 2147483647, 2147483647));
  EXPECT_EQ(description.threads_per_core_limit(), 6144);
  EXPECT_EQ(description.threads_per_warp(), 128);
  EXPECT_EQ(description.max_blocks_per_multiprocessor(), -1);
  EXPECT_EQ(description.core_count(), 32);
  EXPECT_EQ(description.shared_memory_per_block(), 28672);
  EXPECT_EQ(description.registers_per_block_limit(), 262144);
  EXPECT_EQ(description.registers_per_core_limit(), 131072);
  EXPECT_EQ(description.l2_cache_size(), 25165824);
  EXPECT_EQ(description.memory_bandwidth(), 448000000000LL);
  const MusaComputeCapability* capability =
      description.gpu_compute_capability().musa_compute_capability();
  ASSERT_NE(capability, nullptr);
  EXPECT_EQ(capability->architecture(), "mp_21");
  EXPECT_EQ(capability->hardware_warp_size(), 128);
  EXPECT_EQ(capability->logical_subgroup_size(), 32);

  EXPECT_THAT(DeviceDescription::FromProto(description.ToProto()),
              IsOkAndHolds(description));
}

TEST(MusaDeviceDescriptionTest, RejectsMissingRequiredFacts) {
  MusaDeviceProperties properties = S80Properties();
  properties.multiprocessor_count = 0;
  EXPECT_THAT(
      BuildMusaDeviceDescription(
          properties, MusaDeviceVersions{10504, 10504, 40001, std::nullopt}),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(MusaDeviceDescriptionTest, RejectsUnqualifiedDeviceIdentity) {
  MusaDeviceProperties properties = S80Properties();
  properties.name = "Another mp_21 device";
  EXPECT_THAT(
      BuildMusaDeviceDescription(
          properties, MusaDeviceVersions{10504, 10504, 40001, std::nullopt}),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(MusaDeviceDescriptionTest, RejectsUnqualifiedHardwareWidth) {
  MusaDeviceProperties properties = S80Properties();
  properties.hardware_warp_size = 32;
  EXPECT_THAT(
      BuildMusaDeviceDescription(
          properties, MusaDeviceVersions{10504, 10504, 40001, std::nullopt}),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(MusaDeviceDescriptionTest, RejectsUnqualifiedToolkit) {
  EXPECT_THAT(BuildMusaDeviceDescription(
                  S80Properties(),
                  MusaDeviceVersions{10504, 10504, 40300, std::nullopt}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(MusaDeviceDescriptionTest, RejectsInvalidComputeCapability) {
  EXPECT_THAT(MusaArchitectureFromComputeCapability(0, 0),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(MusaArchitectureFromComputeCapability(2, 1),
              IsOkAndHolds("mp_21"));
}

}  // namespace
}  // namespace stream_executor::musa
