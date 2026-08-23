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

class FlyFusionDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {};

TEST_F(FlyFusionDeviceTest, Bf16Softmax64x4096) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_softmax

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

softmax {
  p0 = bf16[64,4096]{1,0} parameter(0)
  converted = f32[64,4096]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[64]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[64,4096]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[64,4096]{1,0} subtract(converted, broadcast_max)
  exponential = f32[64,4096]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[64,4096]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[64,4096]{1,0} divide(exponential, broadcast_sum)
  ROOT result = bf16[64,4096]{1,0} convert(normalized)
}

ENTRY main {
  p0 = bf16[64,4096]{1,0} parameter(0)
  ROOT fusion = bf16[64,4096]{1,0} fusion(p0), kind=kCustom,
    calls=softmax,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","4096"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.01}));
}

TEST_F(FlyFusionDeviceTest, Bf16Transpose128x192) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose

transpose {
  p0 = bf16[128,192]{1,0} parameter(0)
  ROOT result = bf16[192,128]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[128,192]{1,0} parameter(0)
  ROOT fusion = bf16[192,128]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64","64"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, Bf16TransposePartialTiles) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose_partial_tiles

transpose {
  p0 = bf16[65,127]{1,0} parameter(0)
  ROOT result = bf16[127,65]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[65,127]{1,0} parameter(0)
  ROOT fusion = bf16[127,65]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, GenericBf16Elementwise) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_bf16_elementwise

elementwise {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  add = bf16[128,192]{1,0} add(p0, p1)
  scale = bf16[] constant(1.5)
  broadcast = bf16[128,192]{1,0} broadcast(scale), dimensions={}
  multiply = bf16[128,192]{1,0} multiply(add, broadcast)
  ROOT result = bf16[128,192]{1,0} maximum(multiply, p0)
}

ENTRY main {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  ROOT fusion = bf16[128,192]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, GenericF32RowReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_f32_row_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = f32[64,256]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT result = f32[64]{0} reduce(p0, zero), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[64,256]{1,0} parameter(0)
  ROOT fusion = f32[64]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(FlyFusionDeviceTest, GenericDynamicSliceBitcast) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_dynamic_slice_bitcast

slice {
  p0 = f32[32,16]{1,0} parameter(0)
  offset = s32[] parameter(1)
  dynamic_slice = f32[16,16]{1,0} dynamic-slice(p0, offset, offset),
    dynamic_slice_sizes={16,16}
  ROOT result = f32[256]{0} bitcast(dynamic_slice)
}

ENTRY main {
  p0 = f32[32,16]{1,0} parameter(0)
  offset = s32[] parameter(1)
  ROOT fusion = f32[256]{0} fusion(p0, offset), kind=kCustom, calls=slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, GenericBf16Concatenate) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_bf16_concatenate

concatenate {
  p0 = bf16[33]{0} parameter(0)
  p1 = bf16[31]{0} parameter(1)
  abs = bf16[33]{0} abs(p0)
  negate = bf16[31]{0} negate(p1)
  ROOT result = bf16[64]{0} concatenate(abs, negate), dimensions={0}
}

ENTRY main {
  p0 = bf16[33]{0} parameter(0)
  p1 = bf16[31]{0} parameter(1)
  ROOT fusion = bf16[64]{0} fusion(p0, p1), kind=kCustom,
    calls=concatenate,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, GenericF32HighPadding) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_f32_high_padding

padding {
  p0 = f32[17,137]{1,0} parameter(0)
  one = f32[] constant(1)
  ROOT result = f32[32,138]{1,0} pad(p0, one), padding=0_15x0_1
}

ENTRY main {
  p0 = f32[17,137]{1,0} parameter(0)
  ROOT fusion = f32[32,138]{1,0} fusion(p0), kind=kCustom,
    calls=padding,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

}  // namespace
}  // namespace xla::gpu
