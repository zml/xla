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

#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/service/gpu_topology.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

TEST(StreamExecutorGpuCompilerTopologyTest,
     DeserializesMatchingGpuTopology) {
  std::shared_ptr<xla::GpuTopology> gpu_topology =
      std::make_shared<xla::GpuTopology>(
          /*platform_version=*/"MTT S80", /*num_partitions=*/1,
          /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/1);
  StreamExecutorGpuTopologyDescription topology(
      xla::MusaId(), xla::MusaName(), gpu_topology);
  StreamExecutorGpuCompiler compiler(xla::MusaId());
  TF_ASSERT_OK_AND_ASSIGN(PjRtTopologyDescriptionProto proto,
                          topology.ToProto());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtTopologyDescription> deserialized,
      compiler.DeserializePjRtTopologyDescription(proto.SerializeAsString()));
  const auto* deserialized_gpu =
      dynamic_cast<const StreamExecutorGpuTopologyDescription*>(
          deserialized.get());
  ASSERT_NE(deserialized_gpu, nullptr);
  EXPECT_EQ(*deserialized_gpu, topology);

  absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> malformed =
      compiler.DeserializePjRtTopologyDescription("not a proto");
  ASSERT_FALSE(malformed.ok());
  EXPECT_EQ(malformed.status().code(), absl::StatusCode::kInvalidArgument);

  proto.set_platform_id(xla::RocmId());
  absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> wrong_platform =
      compiler.DeserializePjRtTopologyDescription(proto.SerializeAsString());
  ASSERT_FALSE(wrong_platform.ok());
  EXPECT_EQ(wrong_platform.status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace xla
