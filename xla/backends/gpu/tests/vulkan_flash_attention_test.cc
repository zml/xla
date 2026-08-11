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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla::gpu {
namespace {

using VulkanFlashAttentionTest =
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(VulkanFlashAttentionTest, DecodeGqa) {
  constexpr absl::string_view kCustomCallHlo = R"(
    HloModule flash_attention

    ENTRY main {
      q = bf16[2,1,16]{2,1,0} parameter(0)
      k = bf16[1,5,16]{2,1,0} parameter(1)
      v = bf16[1,5,16]{2,1,0} parameter(2)
      token = s32[] constant(4)
      ROOT attention = bf16[2,1,16]{2,1,0} custom-call(q, k, v, token),
        custom_call_target="zml$flash_attn"
    })";

  constexpr absl::string_view kReferenceHlo = R"(
    HloModule flash_attention_reference

    maximum {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT max = f32[] maximum(lhs, rhs)
    }

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    ENTRY main {
      q_bf16 = bf16[2,1,16]{2,1,0} parameter(0)
      k_bf16 = bf16[1,5,16]{2,1,0} parameter(1)
      v_bf16 = bf16[1,5,16]{2,1,0} parameter(2)
      q = f32[2,1,16]{2,1,0} convert(q_bf16)
      k = f32[1,5,16]{2,1,0} convert(k_bf16)
      v = f32[1,5,16]{2,1,0} convert(v_bf16)
      k4 = f32[1,2,5,16]{3,2,1,0} broadcast(k), dimensions={0,2,3}
      k_heads = f32[2,5,16]{2,1,0} reshape(k4)
      scores = f32[2,1,5]{2,1,0} dot(q, k_heads),
        lhs_batch_dims={0}, lhs_contracting_dims={2},
        rhs_batch_dims={0}, rhs_contracting_dims={2}
      scale = f32[] constant(0.25)
      scale_broadcast = f32[2,1,5]{2,1,0} broadcast(scale), dimensions={}
      scaled = f32[2,1,5]{2,1,0} multiply(scores, scale_broadcast)
      minus_inf = f32[] constant(-inf)
      row_max = f32[2,1]{1,0} reduce(scaled, minus_inf), dimensions={2},
        to_apply=maximum
      row_max_broadcast = f32[2,1,5]{2,1,0} broadcast(row_max),
        dimensions={0,1}
      shifted = f32[2,1,5]{2,1,0} subtract(scaled, row_max_broadcast)
      exponentials = f32[2,1,5]{2,1,0} exponential(shifted)
      zero = f32[] constant(0)
      denominator = f32[2,1]{1,0} reduce(exponentials, zero),
        dimensions={2}, to_apply=add
      denominator_broadcast = f32[2,1,5]{2,1,0} broadcast(denominator),
        dimensions={0,1}
      probabilities = f32[2,1,5]{2,1,0} divide(
        exponentials, denominator_broadcast)
      v4 = f32[1,2,5,16]{3,2,1,0} broadcast(v), dimensions={0,2,3}
      v_heads = f32[2,5,16]{2,1,0} reshape(v4)
      output = f32[2,1,16]{2,1,0} dot(probabilities, v_heads),
        lhs_batch_dims={0}, lhs_contracting_dims={2},
        rhs_batch_dims={0}, rhs_contracting_dims={1}
      ROOT result = bf16[2,1,16]{2,1,0} convert(output)
    })";

  EXPECT_TRUE(RunAndCompareTwoModules(
      kCustomCallHlo, kReferenceHlo,
      ErrorSpec{/*aabs=*/0.03, /*arel=*/0.03}));
}

}  // namespace
}  // namespace xla::gpu
