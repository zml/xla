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

#include "xla/backends/gpu/target_config/target_config.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "google/protobuf/text_format.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

struct GpuTargetConfigTestCase {
  std::string test_name;
  GpuModel gpu_model;
  bool expect_ok;
};

using GetGpuTargetConfigTest =
    ::testing::TestWithParam<GpuTargetConfigTestCase>;

TEST_P(GetGpuTargetConfigTest, TestProtoRetrieval) {
  const GpuTargetConfigTestCase& test_case = GetParam();
  auto config = GetGpuTargetConfig(test_case.gpu_model);

  if (test_case.expect_ok) {
    ASSERT_THAT(config, absl_testing::IsOk());
    EXPECT_TRUE(config->has_gpu_device_info());
    EXPECT_GT(config->gpu_device_info().threads_per_block_limit(), 0);
  } else {
    EXPECT_THAT(config,
                absl_testing::StatusIs(absl::StatusCode::kNotFound,
                                       HasSubstr("Embedded file not found")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    GetGpuTargetConfigTests, GetGpuTargetConfigTest,
    ::testing::ValuesIn<GpuTargetConfigTestCase>({
        {"A100_PCIE_80", GpuModel::A100_PCIE_80, true},
        {"A100_SXM_40", GpuModel::A100_SXM_40, true},
        {"A100_SXM_80", GpuModel::A100_SXM_80, true},
        {"A6000", GpuModel::A6000, true},
        {"B200", GpuModel::B200, true},
        {"B300", GpuModel::B300, true},
        {"BMG_G21", GpuModel::BMG_G21, true},
        {"H100_PCIE", GpuModel::H100_PCIE, true},
        {"H100_SXM", GpuModel::H100_SXM, true},
        {"H200", GpuModel::H200, true},
        {"MI200", GpuModel::MI200, true},
        {"P100", GpuModel::P100, true},
        {"PVC", GpuModel::PVC, true},
        {"S80", GpuModel::S80, true},
        {"V100", GpuModel::V100, true},
        {"GB200", GpuModel::GB200, true},
        {"GB300", GpuModel::GB300, true},
        {"RTX6000PRO", GpuModel::RTX6000PRO, true},
    }),
    [](const ::testing::TestParamInfo<GetGpuTargetConfigTest::ParamType>&
           info) { return info.param.test_name; });

TEST(TargetConfigTest, RetrievesMeasuredMusaS80Snapshot) {
  ASSERT_OK_AND_ASSIGN(stream_executor::GpuTargetConfigProto config,
                       GetGpuTargetConfig(GpuModel::S80));

  EXPECT_EQ(config.platform_name(), "MUSA");
  EXPECT_EQ(config.device_description_str(), "MTT S80");
  ASSERT_TRUE(config.has_gpu_device_info());
  const stream_executor::GpuDeviceInfoProto& device = config.gpu_device_info();

  EXPECT_EQ(device.device_vendor(), "Moore Threads");
  EXPECT_EQ(device.name(), "MTT S80");
  EXPECT_EQ(device.device_address_bits(), 64);
  EXPECT_EQ(device.device_memory_size(), 17089384448LL);
  EXPECT_EQ(device.thread_dim_limit_x(), 1024);
  EXPECT_EQ(device.thread_dim_limit_y(), 1024);
  EXPECT_EQ(device.thread_dim_limit_z(), 1024);
  EXPECT_EQ(device.block_dim_limit_x(), 2147483647);
  EXPECT_EQ(device.block_dim_limit_y(), 2147483647);
  EXPECT_EQ(device.block_dim_limit_z(), 2147483647);
  EXPECT_EQ(device.threads_per_block_limit(), 1024);
  EXPECT_EQ(device.threads_per_core_limit(), 6144);
  EXPECT_EQ(device.threads_per_warp(), 128);
  EXPECT_EQ(device.core_count(), 32);
  EXPECT_EQ(device.shared_memory_per_block(), 28672);
  EXPECT_EQ(device.shared_memory_per_block_optin(), 28672);
  EXPECT_EQ(device.shared_memory_per_core(), 28672);
  EXPECT_EQ(device.registers_per_block_limit(), 262144);
  EXPECT_EQ(device.registers_per_core_limit(), 131072);
  EXPECT_EQ(device.l2_cache_size(), 25165824);
  EXPECT_EQ(device.memory_bandwidth(), 448000000000LL);
  EXPECT_FLOAT_EQ(device.clock_rate_ghz(), 1.8f);
  EXPECT_FLOAT_EQ(device.mem_clock_ghz(), 7.0f);

  ASSERT_TRUE(device.has_musa_compute_capability());
  const stream_executor::MusaComputeCapabilityProto& capability =
      device.musa_compute_capability();
  EXPECT_EQ(capability.architecture(), "mp_21");
  EXPECT_EQ(capability.major(), 2);
  EXPECT_EQ(capability.minor(), 1);
  EXPECT_EQ(capability.hardware_warp_size(), 128);
  EXPECT_EQ(capability.logical_subgroup_size(), 32);

  EXPECT_EQ(device.driver_version(), "1.5.4");
  EXPECT_EQ(device.kernel_mode_driver_version(), "3.0.0");
  EXPECT_EQ(device.runtime_version(), "1.5.4");
  EXPECT_EQ(device.compile_time_toolkit_version(), "4.0.1");
  ASSERT_TRUE(config.has_runtime_version());
  EXPECT_EQ(config.runtime_version().major(), 1);
  EXPECT_EQ(config.runtime_version().minor(), 5);
  EXPECT_EQ(config.runtime_version().patch(), 4);

  ASSERT_OK_AND_ASSIGN(GpuTargetConfig parsed,
                       GpuTargetConfig::FromProto(config));
  EXPECT_TRUE(parsed.device_description.gpu_compute_capability().IsMusa());
  EXPECT_EQ(parsed.device_description.musa_compute_capability().architecture(),
            "mp_21");
  EXPECT_EQ(parsed.ToProto().gpu_device_info().threads_per_warp(), 128);
}

TEST(TargetConfigTest, CompareEqualFromSameProto) {
  stream_executor::GpuTargetConfigProto config_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        platform_name: "platform"
        dnn_version_info { major: 2 }
        runtime_version { major: 12 }
        gpu_device_info { threads_per_block_limit: 5 }
        device_description_str: "foo"
      )pb",
      &config_proto));

  ASSERT_OK_AND_ASSIGN(auto config1, GpuTargetConfig::FromProto(config_proto));
  ASSERT_OK_AND_ASSIGN(auto config2, GpuTargetConfig::FromProto(config_proto));
  EXPECT_THAT(config1, ::testing::Eq(config2));
}

TEST(TargetConfigTest, GetTargetConfigFromFile) {
  std::string filename =
      tsl::io::JoinPath(testing::TempDir(), "target_config.textproto");
  std::string proto_content = R"pb(
    platform_name: "platform"
    gpu_device_info { threads_per_block_limit: 5 }
  )pb";
  TF_ASSERT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), filename, proto_content));

  ASSERT_OK_AND_ASSIGN(GpuTargetConfig config,
                       GetTargetConfigFromFile(filename));
  EXPECT_EQ(config.platform_name, "platform");
  EXPECT_EQ(config.device_description.threads_per_block_limit(), 5);
}

}  // namespace
}  // namespace xla::gpu
