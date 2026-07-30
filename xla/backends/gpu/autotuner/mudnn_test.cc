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

#include "xla/backends/gpu/autotuner/mudnn.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::tsl::proto_testing::EqualsProto;

constexpr char kForwardCustomCallHlo[] = R"hlo(
HloModule mudnn_autotune

ENTRY main {
  input = f32[3,56,56,16]{2,1,0,3} parameter(0)
  filter = f32[3,3,3,64]{2,1,0,3} parameter(1)
  ROOT conv = (f32[54,54,16,64]{1,0,3,2}, u8[0]{0})
    custom-call(input, filter), custom_call_target="__cudnn$convForward",
    window={size=3x3}, dim_labels=f01b_i01o->01bf,
    backend_config={"cudnn_conv_backend_config":{
      "activation_mode":"kNone",
      "conv_result_scale":1,
      "side_input_scale":0,
      "leakyrelu_alpha":0
    }}
}
)hlo";

constexpr char kFusedCustomCallHlo[] = R"hlo(
HloModule mudnn_fused_unsupported

ENTRY main {
  input = f32[3,56,56,16]{2,1,0,3} parameter(0)
  filter = f32[3,3,3,64]{2,1,0,3} parameter(1)
  bias = f32[64]{0} parameter(2)
  ROOT conv = (f32[54,54,16,64]{1,0,3,2}, u8[0]{0})
    custom-call(input, filter, bias),
    custom_call_target="__cudnn$convBiasActivationForward",
    window={size=3x3}, dim_labels=f01b_i01o->01bf,
    backend_config={"cudnn_conv_backend_config":{
      "activation_mode":"kRelu",
      "conv_result_scale":1,
      "side_input_scale":0,
      "leakyrelu_alpha":0
    }}
}
)hlo";

class MudnnBackendTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    debug_options_.set_xla_gpu_autotune_level(0);
    ASSERT_OK_AND_ASSIGN(se::GpuTargetConfigProto proto,
                         GetGpuTargetConfig(GpuModel::S80));
    ASSERT_OK_AND_ASSIGN(Compiler::GpuTargetConfig target_config,
                         Compiler::GpuTargetConfig::FromProto(proto));
    target_config_ =
        std::make_unique<Compiler::GpuTargetConfig>(std::move(target_config));
    backend_ = std::make_unique<MudnnBackend>(
        /*stream_executor=*/nullptr, &debug_options_, /*compiler=*/nullptr,
        target_config_.get(), /*allocator=*/nullptr);
  }

  DebugOptions debug_options_;
  std::unique_ptr<Compiler::GpuTargetConfig> target_config_;
  std::unique_ptr<MudnnBackend> backend_;
};

TEST_F(MudnnBackendTest, HasTruthfulBackendIdentityAndVersion) {
  EXPECT_EQ(backend_->backend(), autotuner::Backend::MUDNN);
  EXPECT_EQ(backend_->name(), "MUDNN");
  EXPECT_THAT(backend_->version(),
              HasSubstr("mudnn_algorithm_selection=live_runner"));
  EXPECT_THAT(
      backend_->version(),
      HasSubstr(
          "dfaab657ef752c2f591b8dd38c1310c39c0d6eecf785341f141942fa439a719a"));
  EXPECT_THAT(backend_->version(), HasSubstr("mudnn_deterministic_ops=0"));
}

TEST_F(MudnnBackendTest, DefaultWithoutLiveDeviceFailsActionably) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kForwardCustomCallHlo));
  HloInstruction* conv = module->entry_computation()->root_instruction();
  EXPECT_THAT(backend_->GetDefaultConfig(*conv),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("live MUSA executor and allocator")));
}

TEST_F(MudnnBackendTest, LevelZeroWithoutLiveDeviceFailsActionably) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kForwardCustomCallHlo));
  HloInstruction* conv = module->entry_computation()->root_instruction();
  EXPECT_THAT(backend_->GetSupportedConfigs(*conv),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("live MUSA executor and allocator")));
}

TEST_F(MudnnBackendTest, RejectsFusedConvolution) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kFusedCustomCallHlo));
  HloInstruction* fused = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_->GetSupportedConfigs(*fused));
  EXPECT_TRUE(configs.empty());
  EXPECT_THAT(
      backend_->GetDefaultConfig(*fused),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unfused")));
}

TEST_F(MudnnBackendTest, ApplyConfigWritesSharedConvolutionBackendConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kForwardCustomCallHlo));
  HloInstruction* conv = module->entry_computation()->root_instruction();
  BackendConfig config;
  config.mutable_algorithm()->set_algo_id(3);
  config.mutable_algorithm()->set_math_type(
      se::dnn::AlgorithmProto::DEFAULT_MATH);
  config.mutable_algorithm()->mutable_workspace_size()->set_value(0);
  ASSERT_OK(backend_->ApplyConfig(*conv, config));

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       conv->backend_config<GpuBackendConfig>());
  EXPECT_THAT(gpu_config.cudnn_conv_backend_config().algorithm(),
              EqualsProto(config.algorithm()));
}

TEST_F(MudnnBackendTest, ApplyConfigResizesSerializedWorkspace) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kForwardCustomCallHlo));
  HloInstruction* conv = module->entry_computation()->root_instruction();
  BackendConfig config;
  config.mutable_algorithm()->set_algo_id(2);
  config.mutable_algorithm()->mutable_workspace_size()->set_value(4096);
  ASSERT_OK(backend_->ApplyConfig(*conv, config));

  HloInstruction* replaced =
      module->entry_computation()->GetInstructionWithName("conv");
  ASSERT_NE(replaced, nullptr);
  ASSERT_TRUE(replaced->shape().IsTuple());
  EXPECT_EQ(replaced->shape().tuple_shapes(1).dimensions(0), 4096);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       replaced->backend_config<GpuBackendConfig>());
  EXPECT_THAT(gpu_config.cudnn_conv_backend_config().algorithm(),
              EqualsProto(config.algorithm()));
}

TEST_F(MudnnBackendTest, RejectsMissingOrOutOfRangeAlgorithm) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kForwardCustomCallHlo));
  HloInstruction* conv = module->entry_computation()->root_instruction();

  BackendConfig missing;
  EXPECT_THAT(backend_->ApplyConfig(*conv, missing),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("AlgorithmProto")));

  BackendConfig negative;
  negative.mutable_algorithm()->set_algo_id(-1);
  EXPECT_THAT(backend_->ApplyConfig(*conv, negative),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be one of 0, 2, or 3")));

  BackendConfig direct;
  direct.mutable_algorithm()->set_algo_id(1);
  EXPECT_THAT(backend_->ApplyConfig(*conv, direct),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be one of 0, 2, or 3")));

  BackendConfig unknown;
  unknown.mutable_algorithm()->set_algo_id(4);
  EXPECT_THAT(backend_->ApplyConfig(*conv, unknown),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be one of 0, 2, or 3")));

  BackendConfig unknown_workspace;
  unknown_workspace.mutable_algorithm()->set_algo_id(0);
  EXPECT_THAT(backend_->ApplyConfig(*conv, unknown_workspace),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("explicit workspace size")));

  BackendConfig knobs;
  knobs.mutable_algorithm()->set_algo_id(0);
  knobs.mutable_algorithm()->mutable_workspace_size()->set_value(0);
  (*knobs.mutable_algorithm()->mutable_tuning_knobs())[1] = 2;
  EXPECT_THAT(backend_->ApplyConfig(*conv, knobs),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("do not accept tuning knobs")));

  BackendConfig frontend;
  frontend.mutable_algorithm()->set_algo_id(0);
  frontend.mutable_algorithm()->mutable_workspace_size()->set_value(0);
  frontend.mutable_algorithm()->set_is_cudnn_frontend(true);
  EXPECT_THAT(backend_->ApplyConfig(*conv, frontend),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("do not accept cuDNN frontend state")));
}

TEST_F(MudnnBackendTest, VersionSeparatesDeterminismModes) {
  const std::string default_version = backend_->version();
  debug_options_.set_xla_gpu_deterministic_ops(true);
  const std::string deterministic_version = backend_->version();
  EXPECT_NE(default_version, deterministic_version);
  EXPECT_THAT(deterministic_version, HasSubstr("mudnn_deterministic_ops=1"));
}

}  // namespace
}  // namespace xla::gpu
