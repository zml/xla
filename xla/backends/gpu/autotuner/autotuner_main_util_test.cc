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

#include "xla/backends/gpu/autotuner/autotuner_main_util.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOk;

class AutotunerMainUtilTest : public HloHardwareIndependentTestBase {};

TEST_F(AutotunerMainUtilTest, ClearsOnlyTritonBlockLevelConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

triton_computation {
  p0 = f32[16,64]{1,0} parameter(0)
  ROOT copy = f32[16,64]{1,0} copy(p0)
}

fly_computation {
  p0 = f32[16,64]{1,0} parameter(0)
  ROOT copy = f32[16,64]{1,0} copy(p0)
}

plain_computation {
  p0 = f32[16,64]{1,0} parameter(0)
  ROOT copy = f32[16,64]{1,0} copy(p0)
}

ENTRY main {
  p0 = f32[16,64]{1,0} parameter(0)
  triton = f32[16,64]{1,0} fusion(p0), kind=kCustom,
    calls=triton_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["4","16"]}],
      "num_warps":"2","num_ctas":"1","num_stages":"1"}}}
  fly = f32[16,64]{1,0} fusion(p0), kind=kCustom, calls=fly_computation,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["8","8"]}],
      "num_warps":"4","num_ctas":"1","num_stages":"1"}}}
  plain = f32[16,64]{1,0} fusion(p0), kind=kInput,
    calls=plain_computation
  ROOT result = (f32[16,64]{1,0}, f32[16,64]{1,0}, f32[16,64]{1,0})
    tuple(triton, fly, plain)
}
)"));

  ASSERT_THAT(ClearPreexistingBlockLevelConfigs(*module), IsOk());

  HloInstruction* result = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(
      GpuBackendConfig triton_config,
      result->operand(0)->backend_config<GpuBackendConfig>());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig fly_config,
                          result->operand(1)->backend_config<GpuBackendConfig>());
  TF_ASSERT_OK_AND_ASSIGN(
      GpuBackendConfig plain_config,
      result->operand(2)->backend_config<GpuBackendConfig>());

  EXPECT_EQ(triton_config.fusion_backend_config().kind(), kTritonFusionKind);
  EXPECT_FALSE(triton_config.fusion_backend_config()
                   .has_block_level_fusion_config());
  EXPECT_EQ(fly_config.fusion_backend_config().kind(), kFlyFusionKind);
  EXPECT_TRUE(
      fly_config.fusion_backend_config().has_block_level_fusion_config());
  EXPECT_FALSE(
      plain_config.fusion_backend_config().has_block_level_fusion_config());
}

}  // namespace
}  // namespace xla::gpu
