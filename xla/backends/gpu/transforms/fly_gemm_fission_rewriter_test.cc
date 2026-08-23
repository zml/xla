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

#include "xla/backends/gpu/transforms/fly_gemm_fission_rewriter.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

class FlyGemmFissionRewriterTest : public HloHardwareIndependentTestBase {};

TEST_F(FlyGemmFissionRewriterTest, IsolatesDotFromDequantizationProducers) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      lhs = bf16[64,32]{1,0} parameter(0)
      lhs_scale = bf16[64,32]{1,0} parameter(1)
      scaled_lhs = bf16[64,32]{1,0} multiply(lhs, lhs_scale)
      rhs = bf16[32,128]{1,0} parameter(2)
      rhs_scale = bf16[32,128]{1,0} parameter(3)
      scaled_rhs = bf16[32,128]{1,0} multiply(rhs, rhs_scale)
      ROOT dot = f32[64,128]{1,0} dot(scaled_lhs, scaled_rhs),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  FlyGemmFissionRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kCustom);
  ASSERT_EQ(fusion->operand_count(), 2);
  EXPECT_EQ(fusion->operand(0)->opcode(), HloOpcode::kMultiply);
  EXPECT_EQ(fusion->operand(1)->opcode(), HloOpcode::kMultiply);

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(config.fusion_backend_config().kind(), kFlyGemmFusionKind);

  const HloComputation* body = fusion->fused_instructions_computation();
  EXPECT_EQ(body->instruction_count(), 3);
  EXPECT_EQ(body->root_instruction()->opcode(), HloOpcode::kDot);
  EXPECT_EQ(body->root_instruction()->operand(0)->opcode(),
            HloOpcode::kParameter);
  EXPECT_EQ(body->root_instruction()->operand(1)->opcode(),
            HloOpcode::kParameter);
  const auto verified = ParseAndReturnVerifiedModule(module->ToString());
  ASSERT_TRUE(verified.ok()) << verified.status();
}

TEST_F(FlyGemmFissionRewriterTest, LeavesModulesWithoutDotsUnchanged) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      lhs = bf16[64,32]{1,0} parameter(0)
      rhs = bf16[64,32]{1,0} parameter(1)
      ROOT add = bf16[64,32]{1,0} add(lhs, rhs)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  FlyGemmFissionRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kAdd);
}

}  // namespace
}  // namespace xla::gpu
