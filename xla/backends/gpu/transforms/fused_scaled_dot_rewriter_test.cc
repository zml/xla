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

#include "xla/backends/gpu/transforms/fused_scaled_dot_rewriter.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::HasSubstr;

constexpr absl::string_view kScaledDotModule = R"(
    HloModule m
    ENTRY main {
      lhs = bf16[32,64] parameter(0)
      rhs = bf16[16,64] parameter(1)
      lhs_scale = bf16[] parameter(2)
      rhs_scale = bf16[16,2] parameter(3)
      ROOT d = f32[32,16] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }
  )";

FusedScaledDotArm ClaimingArm(std::string target) {
  return [target](HloComputation* computation,
                  HloScaledDotInstruction* dot) -> absl::StatusOr<HloInstruction*> {
    return computation->AddInstruction(HloInstruction::CreateCustomCall(
        dot->shape(), dot->operands(), target));
  };
}

FusedScaledDotArm DecliningArm() {
  return [](HloComputation*, HloScaledDotInstruction*) -> absl::StatusOr<HloInstruction*> {
    return nullptr;
  };
}

FusedScaledDotArm FailingArm() {
  return [](HloComputation*, HloScaledDotInstruction*) -> absl::StatusOr<HloInstruction*> {
    return absl::InternalError("arm failed");
  };
}

using FusedScaledDotRewriterTest = HloHardwareIndependentTestBase;

TEST_F(FusedScaledDotRewriterTest, NoArmsLeavesTheModuleAlone) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kScaledDotModule));
  FusedScaledDotRewriter rewriter({});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kScaledDot);
}

TEST_F(FusedScaledDotRewriterTest, AClaimReplacesTheDot) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kScaledDotModule));
  FusedScaledDotRewriter rewriter({ClaimingArm("backend$matmul")});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), "backend$matmul");
  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kScaledDot);
  }
}

TEST_F(FusedScaledDotRewriterTest, ArmsAreTriedInOrderAndTheFirstClaimWins) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kScaledDotModule));
  FusedScaledDotRewriter rewriter(
      {DecliningArm(), ClaimingArm("second$arm"), ClaimingArm("third$arm")});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->custom_call_target(),
            "second$arm");
}

TEST_F(FusedScaledDotRewriterTest, AnArmErrorPropagates) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kScaledDotModule));
  FusedScaledDotRewriter rewriter({FailingArm(), ClaimingArm("never$reached")});
  absl::Status status = rewriter.Run(module.get()).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(std::string(status.message()), HasSubstr("arm failed"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
