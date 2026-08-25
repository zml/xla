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

#include "xla/backends/gpu/codegen/flydsl/xtile_elementwise.h"

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

class FlyXTileElementwiseTest : public HloHardwareIndependentTestBase {
 protected:
  bool IsSupported(const std::string& hlo) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(
        *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
    return IsFlyXTileElementwiseFusion(analysis);
  }
};

TEST_F(FlyXTileElementwiseTest, RecognizesVectorizedBf16Dag) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  add = bf16[128,64]{1,0} add(p0, p1)
  scale = bf16[] constant(1.5)
  scale_broadcast = bf16[128,64]{1,0} broadcast(scale), dimensions={}
  scaled = bf16[128,64]{1,0} multiply(add, scale_broadcast)
  ROOT result = bf16[128,64]{1,0} maximum(scaled, p0)
}

ENTRY entry {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesMultiOutputBf16Dag) {
  EXPECT_TRUE(IsSupported(R"(
HloModule multi_output_elementwise

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  sum = bf16[128,64]{1,0} add(p0, p1)
  product = bf16[128,64]{1,0} multiply(sum, p0)
  ROOT tuple = (bf16[128,64]{1,0}, bf16[128,64]{1,0}) tuple(sum, product)
}

ENTRY entry {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = (bf16[128,64]{1,0}, bf16[128,64]{1,0})
    fusion(p0, p1), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RejectsMismatchedMultiOutputShapes) {
  EXPECT_FALSE(IsSupported(R"(
HloModule mismatched_multi_output_elementwise

elementwise {
  p0 = f32[128,64]{1,0} parameter(0)
  negated = f32[128,64]{1,0} negate(p0)
  reshaped = f32[8192]{0} bitcast(negated)
  ROOT tuple = (f32[128,64]{1,0}, f32[8192]{0}) tuple(negated, reshaped)
}

ENTRY entry {
  p0 = f32[128,64]{1,0} parameter(0)
  ROOT fusion = (f32[128,64]{1,0}, f32[8192]{0}) fusion(p0),
    kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesBf16DagNormalizedToF32) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_normalized

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p0_f32 = f32[128,64]{1,0} convert(p0)
  p1 = bf16[128,64]{1,0} parameter(1)
  p1_f32 = f32[128,64]{1,0} convert(p1)
  add = f32[128,64]{1,0} add(p0_f32, p1_f32)
  scale = bf16[] constant(1.5)
  scale_broadcast = bf16[128,64]{1,0} broadcast(scale), dimensions={}
  scale_f32 = f32[128,64]{1,0} convert(scale_broadcast)
  scaled = f32[128,64]{1,0} multiply(add, scale_f32)
  maximum = f32[128,64]{1,0} maximum(scaled, p0_f32)
  ROOT result = bf16[128,64]{1,0} convert(maximum)
}

ENTRY entry {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesSmallSplitKResidualReductions) {
  EXPECT_TRUE(IsSupported(R"(
HloModule small_split_k_residual

add_reduce {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

residual {
  base = bf16[2,128,1024]{2,1,0} parameter(0)
  base_f32 = f32[2,128,1024]{2,1,0} convert(base)
  partial4 = f32[4,256,1024]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  sum4 = f32[256,1024]{1,0} reduce(partial4, zero), dimensions={0},
      to_apply=add_reduce
  rounded4 = bf16[256,1024]{1,0} convert(sum4)
  view4 = bf16[2,128,1024]{2,1,0} bitcast(rounded4)
  value4 = f32[2,128,1024]{2,1,0} convert(view4)
  partial2 = f32[2,256,1024]{2,1,0} parameter(2)
  sum2 = f32[256,1024]{1,0} reduce(partial2, zero), dimensions={0},
      to_apply=add_reduce
  rounded2 = bf16[256,1024]{1,0} convert(sum2)
  view2 = bf16[2,128,1024]{2,1,0} bitcast(rounded2)
  value2 = f32[2,128,1024]{2,1,0} convert(view2)
  first = f32[2,128,1024]{2,1,0} add(base_f32, value2)
  total = f32[2,128,1024]{2,1,0} add(first, value4)
  ROOT result = bf16[2,128,1024]{2,1,0} convert(total)
}

ENTRY entry {
  base = bf16[2,128,1024]{2,1,0} parameter(0)
  partial4 = f32[4,256,1024]{2,1,0} parameter(1)
  partial2 = f32[2,256,1024]{2,1,0} parameter(2)
  ROOT fusion = bf16[2,128,1024]{2,1,0}
      fusion(base, partial4, partial2), kind=kCustom, calls=residual,
      backend_config={"fusion_backend_config":{
        "kind":"__fly",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1",
          "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRank4DilatedReduceWindow) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_rank4_reduce_window

maximum {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT result = bf16[] maximum(lhs, rhs)
}

elementwise {
  p0 = bf16[2,5,7,11]{3,2,1,0} parameter(0)
  negative_infinity = bf16[] constant(-inf)
  window = bf16[2,3,3,11]{3,2,1,0} reduce-window(p0, negative_infinity),
    window={size=1x3x3x1 stride=1x2x2x1
      pad=0_0x1_1x2_1x0_0 rhs_dilate=1x1x2x1}, to_apply=maximum
  ROOT result = bf16[2,3,3,11]{3,2,1,0} abs(window)
}

ENTRY entry {
  p0 = bf16[2,5,7,11]{3,2,1,0} parameter(0)
  ROOT fusion = bf16[2,3,3,11]{3,2,1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RejectsOversizedReduceWindow) {
  EXPECT_FALSE(IsSupported(R"(
HloModule elementwise_oversized_reduce_window

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] add(lhs, rhs)
}

elementwise {
  p0 = f32[1024]{0} parameter(0)
  zero = f32[] constant(0)
  ROOT result = f32[1024]{0} reduce-window(p0, zero),
    window={size=129 pad=64_64}, to_apply=add
}

ENTRY entry {
  p0 = f32[1024]{0} parameter(0)
  ROOT fusion = f32[1024]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesCompareSelectClampDag) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_select

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  absolute = bf16[128,64]{1,0} abs(p0)
  zero = bf16[] constant(0)
  zero_broadcast = bf16[128,64]{1,0} broadcast(zero), dimensions={}
  compare = pred[128,64]{1,0} compare(p1, zero_broadcast), direction=GT
  lower = bf16[] constant(-1)
  lower_broadcast = bf16[128,64]{1,0} broadcast(lower), dimensions={}
  upper = bf16[] constant(1)
  upper_broadcast = bf16[128,64]{1,0} broadcast(upper), dimensions={}
  clamped = bf16[128,64]{1,0} clamp(lower_broadcast, p1, upper_broadcast)
  ROOT result = bf16[128,64]{1,0} select(compare, absolute, clamped)
}

ENTRY entry {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesMixedPrecisionSigmoidDag) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_sigmoid

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  converted = f32[128,64]{1,0} convert(p0)
  negated = f32[128,64]{1,0} negate(converted)
  exponential = f32[128,64]{1,0} exponential(negated)
  one = f32[] constant(1)
  one_broadcast = f32[128,64]{1,0} broadcast(one), dimensions={}
  denominator = f32[128,64]{1,0} add(exponential, one_broadcast)
  sigmoid = f32[128,64]{1,0} divide(one_broadcast, denominator)
  ROOT result = bf16[128,64]{1,0} convert(sigmoid)
}

ENTRY entry {
  p0 = bf16[128,64]{1,0} parameter(0)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesNonVectorizedTail) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_tail

elementwise {
  p0 = f32[127]{0} parameter(0)
  ROOT result = f32[127]{0} negate(p0)
}

ENTRY entry {
  p0 = f32[127]{0} parameter(0)
  ROOT fusion = f32[127]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest,
       RecognizesUnalignedContiguousConcatenateDag) {
  constexpr char kHlo[] = R"(
HloModule indexed_concatenate

concatenate {
  p0 = bf16[33]{0} parameter(0)
  p1 = bf16[31]{0} parameter(1)
  absolute = bf16[33]{0} abs(p0)
  negated = bf16[31]{0} negate(p1)
  ROOT result = bf16[64]{0} concatenate(absolute, negated), dimensions={0}
}

ENTRY entry {
  p0 = bf16[33]{0} parameter(0)
  p1 = bf16[31]{0} parameter(1)
  ROOT fusion = bf16[64]{0} fusion(p0, p1), kind=kCustom,
    calls=concatenate,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();
  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  EXPECT_TRUE(IsFlyXTileElementwiseFusion(analysis));
  EXPECT_TRUE(IsFlyXTileIndexedFusion(analysis));
}

TEST_F(FlyXTileElementwiseTest, RecognizesTrailingDimensionBroadcast) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_broadcast

elementwise {
  p0 = f32[128,64]{1,0} parameter(0)
  row = f32[64]{0} parameter(1)
  row_broadcast = f32[128,64]{1,0} broadcast(row), dimensions={1}
  ROOT result = f32[128,64]{1,0} add(p0, row_broadcast)
}

ENTRY entry {
  p0 = f32[128,64]{1,0} parameter(0)
  row = f32[64]{0} parameter(1)
  ROOT fusion = f32[128,64]{1,0} fusion(p0, row), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RejectsNonTrailingDimensionBroadcast) {
  EXPECT_FALSE(IsSupported(R"(
HloModule elementwise_non_trailing_broadcast

elementwise {
  p0 = f32[128,64]{1,0} parameter(0)
  column = f32[128]{0} parameter(1)
  column_broadcast = f32[128,64]{1,0} broadcast(column), dimensions={0}
  ROOT result = f32[128,64]{1,0} add(p0, column_broadcast)
}

ENTRY entry {
  p0 = f32[128,64]{1,0} parameter(0)
  column = f32[128]{0} parameter(1)
  ROOT fusion = f32[128,64]{1,0} fusion(p0, column), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesContiguousSlice) {
  constexpr char kHlo[] = R"(
HloModule elementwise_contiguous_slice

elementwise {
  p0 = bf16[19,67]{1,0} parameter(0)
  absolute = bf16[19,67]{1,0} abs(p0)
  ROOT result = bf16[15,67]{1,0} slice(absolute),
    slice={[2:17], [0:67]}
}

ENTRY entry {
  p0 = bf16[19,67]{1,0} parameter(0)
  ROOT fusion = bf16[15,67]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();
  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  EXPECT_TRUE(IsFlyXTileElementwiseFusion(analysis));
  EXPECT_TRUE(IsFlyXTileIndexedFusion(analysis));
}

TEST_F(FlyXTileElementwiseTest, RecognizesGappedRectangularSlice) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_gapped_slice

elementwise {
  p0 = bf16[19,67]{1,0} parameter(0)
  ROOT result = bf16[15,65]{1,0} slice(p0),
    slice={[2:17], [1:66]}
}

ENTRY entry {
  p0 = bf16[19,67]{1,0} parameter(0)
  ROOT fusion = bf16[15,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRank4RectangularSlice) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_rank4_slice

elementwise {
  p0 = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  absolute = bf16[4,6,8,11]{3,2,1,0} abs(p0)
  ROOT result = bf16[2,4,6,9]{3,2,1,0} slice(absolute),
    slice={[1:3], [1:5], [1:7], [1:10]}
}

ENTRY entry {
  p0 = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  ROOT fusion = bf16[2,4,6,9]{3,2,1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RejectsStridedSlice) {
  EXPECT_FALSE(IsSupported(R"(
HloModule elementwise_strided_slice

elementwise {
  p0 = bf16[19,67]{1,0} parameter(0)
  ROOT result = bf16[9,67]{1,0} slice(p0),
    slice={[0:18:2], [0:67]}
}

ENTRY entry {
  p0 = bf16[19,67]{1,0} parameter(0)
  ROOT fusion = bf16[9,67]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesDynamicRectangularSlice) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_dynamic_slice

elementwise {
  p0 = bf16[19,67]{1,0} parameter(0)
  row = s32[] parameter(1)
  column = s32[] parameter(2)
  ROOT result = bf16[15,65]{1,0} dynamic-slice(p0, row, column),
    dynamic_slice_sizes={15,65}
}

ENTRY entry {
  p0 = bf16[19,67]{1,0} parameter(0)
  row = s32[] parameter(1)
  column = s32[] parameter(2)
  ROOT fusion = bf16[15,65]{1,0} fusion(p0, row, column), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesS64DynamicRectangularSlice) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_s64_dynamic_slice

elementwise {
  p0 = bf16[19,67]{1,0} parameter(0)
  row = s64[] parameter(1)
  column = s64[] parameter(2)
  ROOT result = bf16[15,65]{1,0} dynamic-slice(p0, row, column),
    dynamic_slice_sizes={15,65}
}

ENTRY entry {
  p0 = bf16[19,67]{1,0} parameter(0)
  row = s64[] parameter(1)
  column = s64[] parameter(2)
  ROOT fusion = bf16[15,65]{1,0} fusion(p0, row, column), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRank4DynamicSlice) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_rank4_dynamic_slice

elementwise {
  p0 = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  start0 = s64[] parameter(1)
  start1 = s64[] parameter(2)
  start2 = s64[] parameter(3)
  start3 = s64[] parameter(4)
  absolute = bf16[4,6,8,11]{3,2,1,0} abs(p0)
  ROOT result = bf16[2,4,6,9]{3,2,1,0} dynamic-slice(
    absolute, start0, start1, start2, start3),
    dynamic_slice_sizes={2,4,6,9}
}

ENTRY entry {
  p0 = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  start0 = s64[] parameter(1)
  start1 = s64[] parameter(2)
  start2 = s64[] parameter(3)
  start3 = s64[] parameter(4)
  ROOT fusion = bf16[2,4,6,9]{3,2,1,0} fusion(
    p0, start0, start1, start2, start3), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesDynamicRectangularUpdate) {
  constexpr char kHlo[] = R"(
HloModule elementwise_dynamic_update_slice

elementwise {
  input = bf16[19,67]{1,0} parameter(0)
  update = bf16[15,65]{1,0} parameter(1)
  row = s32[] parameter(2)
  column = s32[] parameter(3)
  scale = bf16[] constant(1.25)
  broadcast = bf16[15,65]{1,0} broadcast(scale), dimensions={}
  scaled = bf16[15,65]{1,0} multiply(update, broadcast)
  ROOT result = bf16[19,67]{1,0} dynamic-update-slice(
    input, scaled, row, column)
}

ENTRY entry {
  input = bf16[19,67]{1,0} parameter(0)
  update = bf16[15,65]{1,0} parameter(1)
  row = s32[] parameter(2)
  column = s32[] parameter(3)
  ROOT fusion = bf16[19,67]{1,0}
    fusion(input, update, row, column), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();
  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  EXPECT_TRUE(IsFlyXTileElementwiseFusion(analysis));
  EXPECT_TRUE(IsFlyXTileIndexedFusion(analysis));
}

TEST_F(FlyXTileElementwiseTest, RecognizesS64DynamicRectangularUpdate) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_s64_dynamic_update_slice

elementwise {
  input = bf16[19,67]{1,0} parameter(0)
  update = bf16[15,65]{1,0} parameter(1)
  row = s64[] parameter(2)
  column = s64[] parameter(3)
  ROOT result = bf16[19,67]{1,0} dynamic-update-slice(
    input, update, row, column)
}

ENTRY entry {
  input = bf16[19,67]{1,0} parameter(0)
  update = bf16[15,65]{1,0} parameter(1)
  row = s64[] parameter(2)
  column = s64[] parameter(3)
  ROOT fusion = bf16[19,67]{1,0}
    fusion(input, update, row, column), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRank4DynamicUpdateSlice) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_rank4_dynamic_update_slice

elementwise {
  input = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  update = bf16[2,4,6,9]{3,2,1,0} parameter(1)
  start0 = s64[] parameter(2)
  start1 = s64[] parameter(3)
  start2 = s64[] parameter(4)
  start3 = s64[] parameter(5)
  absolute = bf16[2,4,6,9]{3,2,1,0} abs(update)
  ROOT result = bf16[4,6,8,11]{3,2,1,0} dynamic-update-slice(
    input, absolute, start0, start1, start2, start3)
}

ENTRY entry {
  input = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  update = bf16[2,4,6,9]{3,2,1,0} parameter(1)
  start0 = s64[] parameter(2)
  start1 = s64[] parameter(3)
  start2 = s64[] parameter(4)
  start3 = s64[] parameter(5)
  ROOT fusion = bf16[4,6,8,11]{3,2,1,0} fusion(
    input, update, start0, start1, start2, start3), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRank1DynamicUpdateSlice) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_rank1_dynamic_update_slice

elementwise {
  input = bf16[67]{0} parameter(0)
  update = bf16[65]{0} parameter(1)
  start = s32[] parameter(2)
  absolute = bf16[65]{0} abs(update)
  ROOT result = bf16[67]{0} dynamic-update-slice(input, absolute, start)
}

ENTRY entry {
  input = bf16[67]{0} parameter(0)
  update = bf16[65]{0} parameter(1)
  start = s32[] parameter(2)
  ROOT fusion = bf16[67]{0} fusion(input, update, start), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesFlatReverse) {
  constexpr char kHlo[] = R"(
HloModule elementwise_flat_reverse

elementwise {
  p0 = bf16[3,67]{1,0} parameter(0)
  absolute = bf16[3,67]{1,0} abs(p0)
  ROOT result = bf16[3,67]{1,0} reverse(absolute), dimensions={0,1}
}

ENTRY entry {
  p0 = bf16[3,67]{1,0} parameter(0)
  ROOT fusion = bf16[3,67]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();
  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  EXPECT_TRUE(IsFlyXTileElementwiseFusion(analysis));
  EXPECT_TRUE(IsFlyXTileIndexedFusion(analysis));
}

TEST_F(FlyXTileElementwiseTest, RecognizesMinorDimensionReverse) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_minor_reverse

elementwise {
  p0 = bf16[3,67]{1,0} parameter(0)
  ROOT result = bf16[3,67]{1,0} reverse(p0), dimensions={1}
}

ENTRY entry {
  p0 = bf16[3,67]{1,0} parameter(0)
  ROOT fusion = bf16[3,67]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesMajorDimensionReverse) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_major_reverse

elementwise {
  p0 = bf16[3,67]{1,0} parameter(0)
  ROOT result = bf16[3,67]{1,0} reverse(p0), dimensions={0}
}

ENTRY entry {
  p0 = bf16[3,67]{1,0} parameter(0)
  ROOT fusion = bf16[3,67]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRank4PartialReverse) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_rank4_partial_reverse

elementwise {
  p0 = bf16[3,5,7,11]{3,2,1,0} parameter(0)
  absolute = bf16[3,5,7,11]{3,2,1,0} abs(p0)
  ROOT result = bf16[3,5,7,11]{3,2,1,0} reverse(absolute),
    dimensions={0,2}
}

ENTRY entry {
  p0 = bf16[3,5,7,11]{3,2,1,0} parameter(0)
  ROOT fusion = bf16[3,5,7,11]{3,2,1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesFlatEdgePad) {
  constexpr char kHlo[] = R"(
HloModule elementwise_flat_edge_pad

elementwise {
  p0 = bf16[15,67]{1,0} parameter(0)
  absolute = bf16[15,67]{1,0} abs(p0)
  zero = bf16[] constant(0)
  ROOT result = bf16[19,67]{1,0} pad(absolute, zero),
    padding=2_2x0_0
}

ENTRY entry {
  p0 = bf16[15,67]{1,0} parameter(0)
  ROOT fusion = bf16[19,67]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kHlo).value();
  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  EXPECT_TRUE(IsFlyXTileElementwiseFusion(analysis));
  EXPECT_TRUE(IsFlyXTileIndexedFusion(analysis));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRectangularEdgePad) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_gapped_pad

elementwise {
  p0 = bf16[15,67]{1,0} parameter(0)
  zero = bf16[] constant(0)
  ROOT result = bf16[15,71]{1,0} pad(p0, zero), padding=0_0x2_2
}

ENTRY entry {
  p0 = bf16[15,67]{1,0} parameter(0)
  ROOT fusion = bf16[15,71]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRank4InteriorAndNegativeEdgePad) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_rank4_pad

elementwise {
  p0 = bf16[4,5,6,7]{3,2,1,0} parameter(0)
  absolute = bf16[4,5,6,7]{3,2,1,0} abs(p0)
  padding_value = bf16[] constant(-0.5)
  ROOT result = bf16[5,9,7,15]{3,2,1,0} pad(absolute, padding_value),
    padding=-1_2x1_-1_1x0_1x2_0_1
}

ENTRY entry {
  p0 = bf16[4,5,6,7]{3,2,1,0} parameter(0)
  ROOT fusion = bf16[5,9,7,15]{3,2,1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRank1InteriorAndNegativeEdgePad) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_rank1_pad

elementwise {
  p0 = bf16[7]{0} parameter(0)
  padding_value = bf16[] constant(2)
  ROOT result = bf16[14]{0} pad(p0, padding_value), padding=-1_2_1
}

ENTRY entry {
  p0 = bf16[7]{0} parameter(0)
  ROOT fusion = bf16[14]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesRank1NegativeEdgePad) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_rank1_negative_edge_pad

elementwise {
  p0 = bf16[7]{0} parameter(0)
  padding_value = bf16[] constant(2)
  ROOT result = bf16[6]{0} pad(p0, padding_value), padding=-2_1
}

ENTRY entry {
  p0 = bf16[7]{0} parameter(0)
  ROOT fusion = bf16[6]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesInteriorPad) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_interior_pad

elementwise {
  p0 = bf16[15,67]{1,0} parameter(0)
  zero = bf16[] constant(0)
  ROOT result = bf16[15,133]{1,0} pad(p0, zero), padding=0_0x0_0_1
}

ENTRY entry {
  p0 = bf16[15,67]{1,0} parameter(0)
  ROOT fusion = bf16[15,133]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

}  // namespace
}  // namespace xla::gpu::flydsl
