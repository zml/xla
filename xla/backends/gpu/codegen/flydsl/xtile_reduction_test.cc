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

#include "xla/backends/gpu/codegen/flydsl/xtile_reduction.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {
namespace {

class FlyXTileReductionTest : public HloHardwareIndependentTestBase {
 protected:
  bool IsSupported(const std::string& hlo) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(
        *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
    return IsFlyXTileRowReductionFusion(analysis);
  }
};

TEST_F(FlyXTileReductionTest, RecognizesF32MinorAddReduction) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_f32_row_reduction

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
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesDynamicInitRowReduction) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_f32_dynamic_init_row_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = f32[64,256]{1,0} parameter(0)
  init = f32[] parameter(1)
  ROOT result = f32[64]{0} reduce(p0, init), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[64,256]{1,0} parameter(0)
  init = f32[] parameter(1)
  ROOT fusion = f32[64]{0} fusion(p0, init), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesConvertedBf16MaximumReduction) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_bf16_row_maximum

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(lhs, rhs)
}

reduction {
  p0 = bf16[64,256]{1,0} parameter(0)
  converted = f32[64,256]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  ROOT result = f32[64]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = bf16[64,256]{1,0} parameter(0)
  ROOT fusion = f32[64]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesDirectLowPrecisionReductions) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_direct_f16_row_add

add {
  lhs = f16[] parameter(0)
  rhs = f16[] parameter(1)
  ROOT sum = f16[] add(lhs, rhs)
}

reduction {
  p0 = f16[127,259]{1,0} parameter(0)
  zero = f16[] constant(0)
  ROOT result = f16[127]{0} reduce(p0, zero), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f16[127,259]{1,0} parameter(0)
  ROOT fusion = f16[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));

  EXPECT_TRUE(IsSupported(R"(
HloModule native_direct_f16_row_maximum

maximum {
  lhs = f16[] parameter(0)
  rhs = f16[] parameter(1)
  ROOT max = f16[] maximum(lhs, rhs)
}

reduction {
  p0 = f16[127,259]{1,0} parameter(0)
  minus_inf = f16[] constant(-inf)
  ROOT result = f16[127]{0} reduce(p0, minus_inf), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = f16[127,259]{1,0} parameter(0)
  ROOT fusion = f16[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));

  EXPECT_TRUE(IsSupported(R"(
HloModule native_direct_bf16_row_minimum

minimum {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT min = bf16[] minimum(lhs, rhs)
}

reduction {
  p0 = bf16[127,259]{1,0} parameter(0)
  plus_inf = bf16[] constant(inf)
  ROOT result = bf16[127]{0} reduce(p0, plus_inf), dimensions={1},
    to_apply=minimum
}

ENTRY main {
  p0 = bf16[127,259]{1,0} parameter(0)
  ROOT fusion = bf16[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesDirectIntegerReductions) {
  for (absl::string_view hlo : {
           R"(
HloModule native_direct_pred_row_maximum

maximum {
  lhs = pred[] parameter(0)
  rhs = pred[] parameter(1)
  ROOT max = pred[] maximum(lhs, rhs)
}

reduction {
  p0 = pred[127,259]{1,0} parameter(0)
  init = pred[] constant(false)
  ROOT result = pred[127]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = pred[127,259]{1,0} parameter(0)
  ROOT fusion = pred[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)",
           R"(
HloModule native_direct_s8_row_add

add {
  lhs = s8[] parameter(0)
  rhs = s8[] parameter(1)
  ROOT sum = s8[] add(lhs, rhs)
}

reduction {
  p0 = s8[127,259]{1,0} parameter(0)
  init = s8[] constant(0)
  ROOT result = s8[127]{0} reduce(p0, init), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = s8[127,259]{1,0} parameter(0)
  ROOT fusion = s8[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)",
           R"(
HloModule native_direct_s16_row_minimum

minimum {
  lhs = s16[] parameter(0)
  rhs = s16[] parameter(1)
  ROOT min = s16[] minimum(lhs, rhs)
}

reduction {
  p0 = s16[127,259]{1,0} parameter(0)
  init = s16[] constant(32767)
  ROOT result = s16[127]{0} reduce(p0, init), dimensions={1},
    to_apply=minimum
}

ENTRY main {
  p0 = s16[127,259]{1,0} parameter(0)
  ROOT fusion = s16[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)",
           R"(
HloModule native_direct_s32_row_maximum

maximum {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT max = s32[] maximum(lhs, rhs)
}

reduction {
  p0 = s32[127,259]{1,0} parameter(0)
  init = s32[] constant(-2147483648)
  ROOT result = s32[127]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = s32[127,259]{1,0} parameter(0)
  ROOT fusion = s32[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)",
           R"(
HloModule native_direct_s32_row_xor

xor {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT result = s32[] xor(lhs, rhs)
}

reduction {
  p0 = s32[127,259]{1,0} parameter(0)
  init = s32[] constant(0)
  ROOT result = s32[127]{0} reduce(p0, init), dimensions={1}, to_apply=xor
}

ENTRY main {
  p0 = s32[127,259]{1,0} parameter(0)
  ROOT fusion = s32[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)",
           R"(
HloModule native_direct_s64_row_maximum

maximum {
  lhs = s64[] parameter(0)
  rhs = s64[] parameter(1)
  ROOT max = s64[] maximum(lhs, rhs)
}

reduction {
  p0 = s64[31,67]{1,0} parameter(0)
  init = s64[] constant(-9223372036854775808)
  ROOT result = s64[31]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = s64[31,67]{1,0} parameter(0)
  ROOT fusion = s64[31]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)",
           R"(
HloModule native_direct_f64_row_maximum

maximum {
  lhs = f64[] parameter(0)
  rhs = f64[] parameter(1)
  ROOT max = f64[] maximum(lhs, rhs)
}

reduction {
  p0 = f64[31,67]{1,0} parameter(0)
  init = f64[] constant(-inf)
  ROOT result = f64[31]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = f64[31,67]{1,0} parameter(0)
  ROOT fusion = f64[31]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"}) {
    EXPECT_TRUE(IsSupported(std::string(hlo)));
  }
}

TEST_F(FlyXTileReductionTest, RecognizesFusedInputAndOutputGraphs) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_fused_row_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = bf16[64,256]{1,0} parameter(0)
  p1 = bf16[64,256]{1,0} parameter(1)
  lhs = f32[64,256]{1,0} convert(p0)
  rhs = f32[64,256]{1,0} convert(p1)
  difference = f32[64,256]{1,0} subtract(lhs, rhs)
  square = f32[64,256]{1,0} multiply(difference, difference)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(square, zero), dimensions={1}, to_apply=add
  scale = f32[] constant(0.25)
  scales = f32[64]{0} broadcast(scale), dimensions={}
  ROOT result = f32[64]{0} multiply(row_sum, scales)
}

ENTRY main {
  p0 = bf16[64,256]{1,0} parameter(0)
  p1 = bf16[64,256]{1,0} parameter(1)
  ROOT fusion = f32[64]{0} fusion(p0, p1), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesExternalRowBroadcastInInputGraph) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_external_row_broadcast_input_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  input = f32[64,256]{1,0} parameter(0)
  row_offset = f32[64]{0} parameter(1)
  row_offsets = f32[64,256]{1,0} broadcast(row_offset), dimensions={0}
  shifted = f32[64,256]{1,0} subtract(input, row_offsets)
  exponentials = f32[64,256]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(exponentials, zero), dimensions={1},
    to_apply=add
  row_sums = f32[64,256]{1,0} broadcast(row_sum), dimensions={0}
  ROOT result = f32[64,256]{1,0} divide(exponentials, row_sums)
}

ENTRY main {
  input = f32[64,256]{1,0} parameter(0)
  row_offset = f32[64]{0} parameter(1)
  ROOT fusion = f32[64,256]{1,0} fusion(input, row_offset), kind=kCustom,
    calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesExternalRowScaleInOutputGraph) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_external_row_scale_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  input = bf16[65,256]{1,0} parameter(0)
  row_scale = f32[65]{0} parameter(1)
  converted = f32[65,256]{1,0} convert(input)
  zero = f32[] constant(0)
  row_sum = f32[65]{0} reduce(converted, zero), dimensions={1}, to_apply=add
  scaled = f32[65]{0} multiply(row_sum, row_scale)
  ROOT result = bf16[65]{0} convert(scaled)
}

ENTRY main {
  input = bf16[65,256]{1,0} parameter(0)
  row_scale = f32[65]{0} parameter(1)
  ROOT fusion = bf16[65]{0} fusion(input, row_scale), kind=kCustom,
    calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesNarrowingBf16Output) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_narrowing_bf16_row_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = bf16[64,256]{1,0} parameter(0)
  converted = f32[64,256]{1,0} convert(p0)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(converted, zero), dimensions={1}, to_apply=add
  scale = f32[] constant(0.25)
  scales = f32[64]{0} broadcast(scale), dimensions={}
  scaled = f32[64]{0} multiply(row_sum, scales)
  ROOT result = bf16[64]{0} convert(scaled)
}

ENTRY main {
  p0 = bf16[64,256]{1,0} parameter(0)
  ROOT fusion = bf16[64]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesNarrowingF16Output) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_narrowing_f16_row_reduction

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

reduction {
  p0 = f16[64,256]{1,0} parameter(0)
  converted = f32[64,256]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[64]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  ROOT result = f16[64]{0} convert(row_max)
}

ENTRY main {
  p0 = f16[64,256]{1,0} parameter(0)
  ROOT fusion = f16[64]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesRaggedBf16RowWidth) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_ragged_bf16_row_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = bf16[64,259]{1,0} parameter(0)
  converted = f32[64,259]{1,0} convert(p0)
  zero = f32[] constant(0)
  ROOT result = f32[64]{0} reduce(converted, zero), dimensions={1},
    to_apply=add
}

ENTRY main {
  p0 = bf16[64,259]{1,0} parameter(0)
  ROOT fusion = f32[64]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesRank3Bf16RmsNorm) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_rank3_bf16_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[2,17,259]{2,1,0} parameter(0)
  converted = f32[2,17,259]{2,1,0} convert(p0)
  squared = f32[2,17,259]{2,1,0} multiply(converted, converted)
  zero = f32[] constant(0)
  row_sum = f32[2,17]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[2,17]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,17]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,17]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,17]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,17]{1,0} rsqrt(variance)
  scales = f32[2,17,259]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,17,259]{2,1,0} multiply(converted, scales)
  ROOT result = bf16[2,17,259]{2,1,0} convert(normalized)
}

ENTRY main {
  p0 = bf16[2,17,259]{2,1,0} parameter(0)
  ROOT fusion = bf16[2,17,259]{2,1,0} fusion(p0), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesBf16ColumnScaleRmsNorm) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_bf16_column_scale_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[4,259]{1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  converted = f32[4,259]{1,0} convert(p0)
  squared = f32[4,259]{1,0} multiply(converted, converted)
  zero = f32[] constant(0)
  row_sum = f32[4]{0} reduce(squared, zero), dimensions={1}, to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[4]{0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[4]{0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[4]{0} broadcast(epsilon), dimensions={}
  variance = f32[4]{0} add(mean_square, epsilons)
  reciprocal_stddev = f32[4]{0} rsqrt(variance)
  scales = f32[4,259]{1,0} broadcast(reciprocal_stddev), dimensions={0}
  normalized = f32[4,259]{1,0} multiply(converted, scales)
  weights = bf16[4,259]{1,0} broadcast(weight), dimensions={1}
  weights_f32 = f32[4,259]{1,0} convert(weights)
  weighted = f32[4,259]{1,0} multiply(normalized, weights_f32)
  ROOT result = bf16[4,259]{1,0} convert(weighted)
}

ENTRY main {
  p0 = bf16[4,259]{1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  ROOT fusion = bf16[4,259]{1,0} fusion(p0, weight), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest,
       RecognizesBitcastBf16ColumnScaleRmsNorm) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_bitcast_bf16_column_scale_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[4,1,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  converted = f32[4,1,259]{2,1,0} convert(p0)
  squared = f32[4,1,259]{2,1,0} multiply(converted, converted)
  flat_square = f32[4,259]{1,0} bitcast(squared)
  zero = f32[] constant(0)
  row_sum = f32[4]{0} reduce(flat_square, zero), dimensions={1}, to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[4]{0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[4]{0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[4]{0} broadcast(epsilon), dimensions={}
  variance = f32[4]{0} add(mean_square, epsilons)
  reciprocal_stddev = f32[4]{0} rsqrt(variance)
  scales = f32[4,1,259]{2,1,0} broadcast(reciprocal_stddev), dimensions={0}
  normalized = f32[4,1,259]{2,1,0} multiply(converted, scales)
  narrowed = bf16[4,1,259]{2,1,0} convert(normalized)
  normalized_view = bf16[4,259]{1,0} bitcast(narrowed)
  normalized_f32 = f32[4,259]{1,0} convert(normalized_view)
  weights = bf16[4,259]{1,0} broadcast(weight), dimensions={1}
  weights_f32 = f32[4,259]{1,0} convert(weights)
  weighted = f32[4,259]{1,0} multiply(normalized_f32, weights_f32)
  ROOT result = bf16[4,259]{1,0} convert(weighted)
}

ENTRY main {
  p0 = bf16[4,1,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  ROOT fusion = bf16[4,259]{1,0} fusion(p0, weight), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest,
       RecognizesFlattenedRank3Bf16ColumnScaleRmsNorm) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_flattened_rank3_bf16_column_scale_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[2,3,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  converted = f32[2,3,259]{2,1,0} convert(p0)
  squared = f32[2,3,259]{2,1,0} multiply(converted, converted)
  zero = f32[] constant(0)
  row_sum = f32[2,3]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[2,3]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,3]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,3]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,3]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,3]{1,0} rsqrt(variance)
  scales = f32[2,3,259]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,3,259]{2,1,0} multiply(converted, scales)
  narrowed = bf16[2,3,259]{2,1,0} convert(normalized)
  normalized_view = bf16[6,259]{1,0} bitcast(narrowed)
  normalized_f32 = f32[6,259]{1,0} convert(normalized_view)
  weights = bf16[6,259]{1,0} broadcast(weight), dimensions={1}
  weights_f32 = f32[6,259]{1,0} convert(weights)
  weighted = f32[6,259]{1,0} multiply(normalized_f32, weights_f32)
  ROOT result = bf16[6,259]{1,0} convert(weighted)
}

ENTRY main {
  p0 = bf16[2,3,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  ROOT fusion = bf16[6,259]{1,0} fusion(p0, weight), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesResidualRmsNormAfterSplitKReduction) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_residual_rms_norm_after_split_k

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  residual = bf16[2,17,259]{2,1,0} parameter(0)
  partials = f32[2,34,259]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  projected = f32[34,259]{1,0} reduce(partials, zero), dimensions={0},
    to_apply=add
  projected_bf16 = bf16[34,259]{1,0} convert(projected)
  projected_view = bf16[2,17,259]{2,1,0} bitcast(projected_bf16)
  residual_f32 = f32[2,17,259]{2,1,0} convert(residual)
  projected_f32 = f32[2,17,259]{2,1,0} convert(projected_view)
  added = f32[2,17,259]{2,1,0} add(residual_f32, projected_f32)
  squared = f32[2,17,259]{2,1,0} multiply(added, added)
  row_sum = f32[2,17]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[2,17]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,17]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,17]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,17]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,17]{1,0} rsqrt(variance)
  scales = f32[2,17,259]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,17,259]{2,1,0} multiply(added, scales)
  ROOT result = bf16[2,17,259]{2,1,0} convert(normalized)
}

ENTRY main {
  residual = bf16[2,17,259]{2,1,0} parameter(0)
  partials = f32[2,34,259]{2,1,0} parameter(1)
  ROOT fusion = bf16[2,17,259]{2,1,0} fusion(residual, partials),
    kind=kCustom, calls=rms_norm,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RecognizesBitcastResidualRmsStatistic) {
  EXPECT_TRUE(IsSupported(R"(
HloModule native_bitcast_residual_rms_statistic

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

statistic {
  flat_residual = bf16[256,1024]{1,0} parameter(0)
  input = bf16[2,128,1024]{2,1,0} parameter(1)
  residual = bf16[2,128,1024]{2,1,0} bitcast(flat_residual)
  input_f32 = f32[2,128,1024]{2,1,0} convert(input)
  residual_f32 = f32[2,128,1024]{2,1,0} convert(residual)
  added = f32[2,128,1024]{2,1,0} add(input_f32, residual_f32)
  squared = f32[2,128,1024]{2,1,0} multiply(added, added)
  zero = f32[] constant(0)
  row_sum = f32[2,128]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.0009765625)
  widths = f32[2,128]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,128]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,128]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,128]{1,0} add(mean_square, epsilons)
  ROOT result = f32[2,128]{1,0} rsqrt(variance)
}

ENTRY main {
  flat_residual = bf16[256,1024]{1,0} parameter(0)
  input = bf16[2,128,1024]{2,1,0} parameter(1)
  ROOT fusion = f32[2,128]{1,0} fusion(flat_residual, input),
    kind=kCustom, calls=statistic,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RejectsMajorDimensionReduction) {
  EXPECT_FALSE(IsSupported(R"(
HloModule column_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = f32[64,256]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT result = f32[256]{0} reduce(p0, zero), dimensions={0}, to_apply=add
}

ENTRY main {
  p0 = f32[64,256]{1,0} parameter(0)
  ROOT fusion = f32[256]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly"}}
}
)"));
}

TEST_F(FlyXTileReductionTest, RejectsNonAssociativeReducer) {
  EXPECT_FALSE(IsSupported(R"(
HloModule subtract_reduction

subtract {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT difference = f32[] subtract(lhs, rhs)
}

reduction {
  p0 = f32[64,256]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT result = f32[64]{0} reduce(p0, zero), dimensions={1},
    to_apply=subtract
}

ENTRY main {
  p0 = f32[64,256]{1,0} parameter(0)
  ROOT fusion = f32[64]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly"}}
}
)"));
}

}  // namespace
}  // namespace xla::gpu::flydsl
