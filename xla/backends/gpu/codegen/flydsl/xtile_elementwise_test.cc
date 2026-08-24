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

TEST_F(FlyXTileElementwiseTest, RejectsNonScalarBroadcast) {
  EXPECT_FALSE(IsSupported(R"(
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

}  // namespace
}  // namespace xla::gpu::flydsl
