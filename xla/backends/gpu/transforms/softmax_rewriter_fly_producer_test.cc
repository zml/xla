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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/flydsl/softmax_support.h"
#include "xla/backends/gpu/transforms/softmax_rewriter_fly.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOkAndHolds;

class SoftmaxRewriterFlyProducerTest : public HloHardwareIndependentTestBase {};

TEST_F(SoftmaxRewriterFlyProducerTest, FormsSoftmaxAroundNonParameterProducer) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_softmax_producer

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

add_reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = bf16[8,64]{1,0} parameter(0)
  p1 = bf16[8,64]{1,0} parameter(1)
  producer = bf16[8,64]{1,0} add(p0, p1)
  converted = f32[8,64]{1,0} convert(producer)
  minus_inf = f32[] constant(-inf)
  row_max = f32[8]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[8,64]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[8,64]{1,0} subtract(converted, broadcast_max)
  exponential = f32[8,64]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[8]{0} reduce(exponential, zero), dimensions={1},
    to_apply=add_reducer
  broadcast_sum = f32[8,64]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[8,64]{1,0} divide(exponential, broadcast_sum)
  ROOT result = bf16[8,64]{1,0} convert(normalized)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  ASSERT_EQ(fusion->operand_count(), 1);
  EXPECT_EQ(fusion->operand(0)->opcode(), HloOpcode::kAdd);
  EXPECT_TRUE(flydsl::IsFlySoftmaxRoot(
      *fusion->fused_instructions_computation()->root_instruction()));
  for (const HloInstruction* instruction :
       fusion->fused_instructions_computation()->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kAdd);
  }
}

TEST_F(SoftmaxRewriterFlyProducerTest, FormsNarrowedSoftmaxAroundF32Producer) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_mixed_softmax_producer

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

add_reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = f32[8,64]{1,0} parameter(0)
  scale = f32[] constant(0.125)
  scales = f32[8,64]{1,0} broadcast(scale), dimensions={}
  producer = f32[8,64]{1,0} multiply(p0, scales)
  minus_inf = f32[] constant(-inf)
  row_max = f32[8]{0} reduce(producer, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[8,64]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[8,64]{1,0} subtract(producer, broadcast_max)
  exponential = f32[8,64]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[8]{0} reduce(exponential, zero), dimensions={1},
    to_apply=add_reducer
  broadcast_sum = f32[8,64]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[8,64]{1,0} divide(exponential, broadcast_sum)
  ROOT result = bf16[8,64]{1,0} convert(normalized)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(true));
  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  ASSERT_EQ(fusion->operand_count(), 1);
  EXPECT_EQ(fusion->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(fusion->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(fusion->shape().element_type(), BF16);
  EXPECT_TRUE(flydsl::IsFlySoftmaxRoot(
      *fusion->fused_instructions_computation()->root_instruction()));
  EXPECT_DOUBLE_EQ(
      flydsl::GetFlySoftmaxInputScale(
          *fusion->fused_instructions_computation()->root_instruction()),
      0.125);
}

TEST_F(SoftmaxRewriterFlyProducerTest, PreservesSharedTrainingIntermediates) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_shared_softmax_intermediate

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

add_reducer {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = f32[8,64]{1,0} parameter(0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[8]{0} reduce(p0, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[8,64]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[8,64]{1,0} subtract(p0, broadcast_max)
  exponential = f32[8,64]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[8]{0} reduce(exponential, zero), dimensions={1},
    to_apply=add_reducer
  broadcast_sum = f32[8,64]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[8,64]{1,0} divide(exponential, broadcast_sum)
  narrowed = bf16[8,64]{1,0} convert(normalized)
  ROOT result = (bf16[8,64]{1,0}, f32[8]{0}) tuple(narrowed, row_sum)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kTuple);
}

}  // namespace
}  // namespace xla::gpu
