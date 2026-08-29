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

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"

namespace xla::gpu {
namespace {

class FlySoftmaxBackwardDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {};

TEST_F(FlySoftmaxBackwardDeviceTest, Bf16TransformerRows) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_softmax_backward_device

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

fly_softmax_backward {
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
  direct = f32[8,64]{1,0} divide(upstream, broadcast_sum)
  one = f32[] constant(1)
  row_ones = f32[8]{0} broadcast(one), dimensions={}
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
  ROOT result = bf16[8,64]{1,0} convert(raw_score_derivative)
}

ENTRY main {
  scores = f32[8,64]{1,0} parameter(0)
  upstream = f32[8,64]{1,0} parameter(1)
  ROOT fusion = bf16[8,64]{1,0} fusion(scores, upstream),
    kind=kCustom, calls=fly_softmax_backward,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4","64"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.03}));
}

}  // namespace
}  // namespace xla::gpu
