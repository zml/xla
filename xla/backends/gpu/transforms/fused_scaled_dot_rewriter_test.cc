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
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/transforms/fused_scaled_dot_arms_cuda.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
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

class Nvfp4SplitKTest : public HloHardwareIndependentTestBase {
 protected:
  std::string Projection(int64_t weight_rows, int64_t k, int64_t batch = 1) {
    return absl::Substitute(R"(
      HloModule m
      ENTRY main {
        x = f4e2m1fn[$2,$1]{1,0:E(4)} parameter(0)
        w = f4e2m1fn[$0,$1]{1,0:E(4)} parameter(1)
        xs = f8e4m3fn[$2,$3]{1,0} parameter(2)
        ws = f8e4m3fn[$0,$3]{1,0} parameter(3)
        ROOT d = bf16[$2,$0]{1,0} scaled-dot(x, w, xs, ws),
            lhs_contracting_dims={1}, rhs_contracting_dims={1}
      })",
                            weight_rows, k, batch, k / 16);
  }

  int64_t SplitFactor(const HloModule& module) {
    const HloInstruction* root = module.entry_computation()->root_instruction();
    for (const HloInstruction* instr :
         module.entry_computation()->instructions()) {
      if (instr->opcode() != HloOpcode::kFusion) continue;
      const Shape& shape = instr->shape();
      return shape.dimensions().size() > 2 ? shape.dimensions(0) : 1;
    }
    ADD_FAILURE() << "no fusion in " << root->parent()->ToString();
    return 0;
  }

  absl::StatusOr<int64_t> SplitFor(absl::string_view hlo, int major, int minor,
                                   int core_count) {
    se::DeviceDescription device;
    device.set_gpu_compute_capability(
        se::GpuComputeCapability(se::CudaComputeCapability(major, minor)));
    device.set_core_count(core_count);
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo));
    FusedScaledDotRewriter rewriter({Nvfp4DecodeDotArm(device)});
    TF_ASSIGN_OR_RETURN(bool changed, rewriter.Run(module.get()));
    if (!changed) return 1;
    return SplitFactor(*module);
  }
};

TEST_F(Nvfp4SplitKTest, SplitsOnlyTheProjectionThatCannotNarrow) {
  // sm_103: down is tile-starved, gate/up is not.
  EXPECT_THAT(SplitFor(Projection(/*weight_rows=*/5120, /*k=*/17408), 10, 3,
                       /*core_count=*/148),
              absl_testing::IsOkAndHolds(2));
  EXPECT_THAT(SplitFor(Projection(/*weight_rows=*/34816, /*k=*/5120), 10, 3,
                       /*core_count=*/148),
              absl_testing::IsOkAndHolds(1));
}

TEST_F(Nvfp4SplitKTest, DoesNotSplitWhereTheTileCanNarrowInstead) {
  // sm_120 at TP=2: the floor is low enough that nothing is starved.
  EXPECT_THAT(SplitFor(Projection(/*weight_rows=*/2560, /*k=*/8704), 12, 0,
                       /*core_count=*/170),
              absl_testing::IsOkAndHolds(1));
  EXPECT_THAT(SplitFor(Projection(/*weight_rows=*/17408, /*k=*/5120), 12, 0,
                       /*core_count=*/170),
              absl_testing::IsOkAndHolds(1));
}

TEST_F(Nvfp4SplitKTest, RefusesASplitThatEmptiesTheBlockKLadder) {
  // Halving leaves a k no block_k divides.
  EXPECT_THAT(SplitFor(Projection(/*weight_rows=*/2048, /*k=*/640), 10, 3,
                       /*core_count=*/148),
              absl_testing::IsOkAndHolds(1));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
