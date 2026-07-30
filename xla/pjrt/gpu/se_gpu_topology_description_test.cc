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

#include "xla/pjrt/gpu/se_gpu_topology_description.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/se/pjrt_stream_executor_device_description.h"
#include "xla/service/gpu_topology.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

void CheckDeviceDescription(const PjRtDeviceDescription& device_desc,
                            int global_device_id, int process_index,
                            const std::vector<int>& coords) {
  EXPECT_EQ(device_desc.id(), global_device_id);
  EXPECT_EQ(device_desc.process_index(), process_index);
  const auto& gpu_device_desc =
      dynamic_cast<const PjRtStreamExecutorDeviceDescription&>(device_desc);
  EXPECT_THAT(gpu_device_desc.coords(),
              ElementsAre(coords[0], coords[1], coords[2]));
}

TEST(StreamExecutorGpuTopologyDescriptionTest, SymmetricTopology) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/3,
          /*num_hosts_per_partition=*/2, /*num_devices_per_host=*/4);

  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  EXPECT_EQ(topology_desc.platform_id(), xla::CudaId());
  EXPECT_EQ(topology_desc.platform_name(), xla::CudaName());
  EXPECT_EQ(topology_desc.platform_version(), "12.3");

  const auto device_descs = topology_desc.DeviceDescriptions();
  EXPECT_EQ(device_descs.size(), 24);
  CheckDeviceDescription(*device_descs[0], 0, 0, {0, 0, 0});
  CheckDeviceDescription(*device_descs[1], 1, 0, {0, 0, 1});
  CheckDeviceDescription(*device_descs[2], 2, 0, {0, 0, 2});
  CheckDeviceDescription(*device_descs[3], 3, 0, {0, 0, 3});
  CheckDeviceDescription(*device_descs[4], 4, 1, {0, 1, 0});
  CheckDeviceDescription(*device_descs[5], 5, 1, {0, 1, 1});
  CheckDeviceDescription(*device_descs[6], 6, 1, {0, 1, 2});
  CheckDeviceDescription(*device_descs[7], 7, 1, {0, 1, 3});
  CheckDeviceDescription(*device_descs[8], 8, 2, {1, 0, 0});
  CheckDeviceDescription(*device_descs[9], 9, 2, {1, 0, 1});
  CheckDeviceDescription(*device_descs[10], 10, 2, {1, 0, 2});
  CheckDeviceDescription(*device_descs[11], 11, 2, {1, 0, 3});
  CheckDeviceDescription(*device_descs[12], 12, 3, {1, 1, 0});
  CheckDeviceDescription(*device_descs[13], 13, 3, {1, 1, 1});
  CheckDeviceDescription(*device_descs[14], 14, 3, {1, 1, 2});
  CheckDeviceDescription(*device_descs[15], 15, 3, {1, 1, 3});
  CheckDeviceDescription(*device_descs[16], 16, 4, {2, 0, 0});
  CheckDeviceDescription(*device_descs[17], 17, 4, {2, 0, 1});
  CheckDeviceDescription(*device_descs[18], 18, 4, {2, 0, 2});
  CheckDeviceDescription(*device_descs[19], 19, 4, {2, 0, 3});
  CheckDeviceDescription(*device_descs[20], 20, 5, {2, 1, 0});
  CheckDeviceDescription(*device_descs[21], 21, 5, {2, 1, 1});
  CheckDeviceDescription(*device_descs[22], 22, 5, {2, 1, 2});
  CheckDeviceDescription(*device_descs[23], 23, 5, {2, 1, 3});
}


TEST(StreamExecutorGpuTopologyDescriptionTest,
     MusaTargetConfigPreservesMetadataAndAssignments) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"MUSA 4.0.1", /*num_partitions=*/2,
          /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/2);

  stream_executor::GpuTargetConfigProto target_config;
  stream_executor::GpuDeviceInfoProto* device_info =
      target_config.mutable_gpu_device_info();
  device_info->set_device_vendor("Moore Threads");
  device_info->set_core_count(32);
  device_info->set_threads_per_warp(128);
  device_info->set_device_memory_size(INT64_C(17089384448));
  device_info->set_shared_memory_per_block_optin(65536);
  stream_executor::MusaComputeCapabilityProto* capability =
      device_info->mutable_musa_compute_capability();
  capability->set_architecture("mp_21");
  capability->set_major(2);
  capability->set_minor(1);
  capability->set_hardware_warp_size(128);
  capability->set_logical_subgroup_size(32);

  StreamExecutorGpuTopologyDescription topology_desc(
      xla::MusaId(), xla::MusaName(), gpu_topology,
      /*attributes=*/{}, target_config);

  const auto device_descs = topology_desc.DeviceDescriptions();
  ASSERT_EQ(device_descs.size(), 4);
  CheckDeviceDescription(*device_descs[0], 0, 0, {0, 0, 0});
  CheckDeviceDescription(*device_descs[1], 1, 0, {0, 0, 1});
  CheckDeviceDescription(*device_descs[2], 2, 1, {1, 0, 0});
  CheckDeviceDescription(*device_descs[3], 3, 1, {1, 0, 1});

  const auto& first_attributes = device_descs[0]->Attributes();
  EXPECT_EQ(std::get<std::string>(first_attributes.at("device_vendor")),
            "Moore Threads");
  EXPECT_EQ(std::get<std::string>(first_attributes.at("compute_capability")),
            "mp_21");
  EXPECT_EQ(std::get<int64_t>(first_attributes.at("core_count")), 32);
  EXPECT_EQ(std::get<int64_t>(
                first_attributes.at("device_memory_bytes_limit")),
            INT64_C(17089384448));
  EXPECT_EQ(std::get<int64_t>(first_attributes.at("partition_index")), 0);
  EXPECT_TRUE(
      std::get<std::string>(first_attributes.at("fabric_uuid")).empty());

  const auto& last_attributes = device_descs[3]->Attributes();
  EXPECT_EQ(std::get<int64_t>(last_attributes.at("partition_index")), 1);
  TF_ASSERT_OK_AND_ASSIGN(uint64_t first_fingerprint,
                          topology_desc.Fingerprint());
  TF_ASSERT_OK_AND_ASSIGN(uint64_t second_fingerprint,
                          topology_desc.Fingerprint());
  EXPECT_EQ(first_fingerprint, second_fingerprint);

  TF_ASSERT_OK_AND_ASSIGN(PjRtTopologyDescriptionProto serialized,
                          topology_desc.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StreamExecutorGpuTopologyDescription> roundtrip,
      StreamExecutorGpuTopologyDescription::FromProto(serialized));
  const auto roundtrip_devices = roundtrip->DeviceDescriptions();
  ASSERT_EQ(roundtrip_devices.size(), 4);
  const auto& roundtrip_attributes = roundtrip_devices[3]->Attributes();
  EXPECT_EQ(std::get<std::string>(
                roundtrip_attributes.at("compute_capability")),
            "mp_21");
  EXPECT_EQ(std::get<int64_t>(
                roundtrip_attributes.at("partition_index")),
            1);
  TF_ASSERT_OK_AND_ASSIGN(uint64_t roundtrip_fingerprint,
                          roundtrip->Fingerprint());
  EXPECT_EQ(roundtrip_fingerprint, first_fingerprint);
  EXPECT_EQ(*roundtrip, topology_desc);

  stream_executor::GpuTargetConfigProto other_target_config = target_config;
  other_target_config.mutable_gpu_device_info()
      ->mutable_musa_compute_capability()
      ->set_architecture("mp_31");
  other_target_config.mutable_gpu_device_info()
      ->mutable_musa_compute_capability()
      ->set_major(3);
  other_target_config.mutable_gpu_device_info()
      ->mutable_musa_compute_capability()
      ->set_minor(1);
  StreamExecutorGpuTopologyDescription other_topology_desc(
      xla::MusaId(), xla::MusaName(), gpu_topology,
      /*attributes=*/{}, std::move(other_target_config));
  TF_ASSERT_OK_AND_ASSIGN(uint64_t other_fingerprint,
                          other_topology_desc.Fingerprint());
  EXPECT_NE(other_fingerprint, first_fingerprint);
  EXPECT_FALSE(other_topology_desc == topology_desc);
}

TEST(StreamExecutorGpuTopologyDescriptionTest, AsymmetricTopology) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/-1,
          /*num_hosts_per_partition=*/-1, /*num_devices_per_host=*/-1);

  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  EXPECT_EQ(topology_desc.platform_id(), xla::CudaId());
  EXPECT_EQ(topology_desc.platform_name(), xla::CudaName());
  EXPECT_EQ(topology_desc.platform_version(), "12.3");

  const auto device_descs = topology_desc.DeviceDescriptions();
  EXPECT_EQ(device_descs.size(), 0);
}

TEST(PjRtTopologyUtilsGPUTest, GetDeviceCoords) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/1,
          /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/4);
  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(1)));
  auto [device_coords, core_id] = std::move(device_core);
  ASSERT_EQ(device_coords, (PjRtDeviceDimensions{0, 0, 1}));
  ASSERT_EQ(core_id, 0);
}

TEST(PjRtTopologyUtilsGPUTest, GetDeviceCoordsSingleHostScopedPartition) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/4,
          /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/4);
  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core1,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(1)));
  auto [device_coords1, core_id1] = std::move(device_core1);
  ASSERT_EQ(device_coords1, (PjRtDeviceDimensions{0, 0, 1}));
  ASSERT_EQ(core_id1, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core2,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(6)));
  auto [device_coords2, core_id2] = std::move(device_core2);
  ASSERT_EQ(device_coords2, (PjRtDeviceDimensions{1, 0, 2}));
  ASSERT_EQ(core_id2, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core3,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(10)));
  auto [device_coords3, core_id3] = std::move(device_core3);
  ASSERT_EQ(device_coords3, (PjRtDeviceDimensions{2, 0, 2}));
  ASSERT_EQ(core_id3, 0);
}

TEST(PjRtTopologyUtilsGPUTest, GetDeviceCoordsMultipleHostScopedPartition) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"12.3", /*num_partitions=*/1,
          /*num_hosts_per_partition=*/4, /*num_devices_per_host=*/4);
  StreamExecutorGpuTopologyDescription topology_desc(
      xla::CudaId(), xla::CudaName(), gpu_topology);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core1,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(1)));
  auto [device_coords1, core_id1] = std::move(device_core1);
  ASSERT_EQ(device_coords1, (PjRtDeviceDimensions{0, 0, 1}));
  ASSERT_EQ(core_id1, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core2,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(6)));
  auto [device_coords2, core_id2] = std::move(device_core2);
  ASSERT_EQ(device_coords2, (PjRtDeviceDimensions{0, 1, 2}));
  ASSERT_EQ(core_id2, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_core3,
      topology_desc.ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
          GlobalDeviceId(10)));
  auto [device_coords3, core_id3] = std::move(device_core3);
  ASSERT_EQ(device_coords3, (PjRtDeviceDimensions{0, 2, 2}));
  ASSERT_EQ(core_id3, 0);
}

}  // namespace
}  // namespace xla
