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
#include <optional>

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

class SoftmaxRewriterFlyBackwardTest : public HloHardwareIndependentTestBase {};

TEST_F(SoftmaxRewriterFlyBackwardTest,
       RewritesForwardAndBackwardTrainingDiamonds) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_softmax_backward

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
  scores = f32[8,64]{1,0} parameter(0)
  upstream = f32[8,64]{1,0} parameter(1)
  scale = f32[] constant(0.125)
  scales = f32[8,64]{1,0} broadcast(scale), dimensions={}
  logits = f32[8,64]{1,0} multiply(scores, scales)
  minus_inf = f32[] constant(-inf)
  row_max = f32[8]{0} reduce(logits, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[8,64]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[8,64]{1,0} subtract(logits, broadcast_max)
  exponential = f32[8,64]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[8]{0} reduce(exponential, zero), dimensions={1},
    to_apply=add_reducer
  broadcast_sum = f32[8,64]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[8,64]{1,0} divide(exponential, broadcast_sum)
  forward = bf16[8,64]{1,0} convert(normalized)

  direct = f32[8,64]{1,0} divide(upstream, broadcast_sum)
  ones = f32[] constant(1)
  row_ones = f32[8]{0} broadcast(ones), dimensions={}
  sum_squared = f32[8]{0} multiply(row_sum, row_sum)
  reciprocal_squared_sum = f32[8]{0} divide(row_ones, sum_squared)
  reciprocal_broadcast = f32[8,64]{1,0}
    broadcast(reciprocal_squared_sum), dimensions={0}
  weighted_upstream = f32[8,64]{1,0}
    multiply(upstream, reciprocal_broadcast)
  weighted_exponential = f32[8,64]{1,0}
    multiply(weighted_upstream, exponential)
  weighted_sum = f32[8]{0} reduce(weighted_exponential, zero), dimensions={1},
    to_apply=add_reducer
  negative_weighted_sum = f32[8]{0} negate(weighted_sum)
  correction_broadcast = f32[8,64]{1,0}
    broadcast(negative_weighted_sum), dimensions={0}
  correction = f32[8,64]{1,0} add(direct, correction_broadcast)
  scaled_derivative = f32[8,64]{1,0} multiply(correction, exponential)
  raw_score_derivative = f32[8,64]{1,0}
    multiply(scaled_derivative, scales)
  backward = bf16[8,64]{1,0} convert(raw_score_derivative)
  ROOT result = (bf16[8,64]{1,0}, bf16[8,64]{1,0})
    tuple(forward, backward)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const HloInstruction* tuple = module->entry_computation()->root_instruction();
  ASSERT_EQ(tuple->opcode(), HloOpcode::kTuple);
  ASSERT_EQ(tuple->operand_count(), 2);
  const auto* forward = Cast<const HloFusionInstruction>(tuple->operand(0));
  const auto* backward = Cast<const HloFusionInstruction>(tuple->operand(1));
  EXPECT_THAT(forward->name(), testing::HasSubstr("fly_softmax"));
  EXPECT_THAT(backward->name(), testing::HasSubstr("fly_softmax_backward"));
  EXPECT_EQ(forward->operand_count(), 1);
  EXPECT_EQ(backward->operand_count(), 2);

  const HloInstruction* backward_root =
      backward->fused_instructions_computation()->root_instruction();
  std::optional<flydsl::FlySoftmaxBackwardDescriptor> descriptor =
      flydsl::GetFlySoftmaxBackwardDescriptor(*backward_root);
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->scores->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(descriptor->upstream_gradient->opcode(), HloOpcode::kParameter);
  EXPECT_DOUBLE_EQ(descriptor->scale, 0.125);
}

}  // namespace
}  // namespace xla::gpu
