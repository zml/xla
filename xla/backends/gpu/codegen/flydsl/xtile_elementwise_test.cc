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

  FlyXTileMemoryPolicy MemoryPolicy(const std::string& hlo) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(
        *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
    return GetFlyXTileMemoryPolicy(analysis);
  }
};

TEST_F(FlyXTileElementwiseTest, UsesNonTemporalMemoryForLargeOnePassFusion) {
  EXPECT_EQ(MemoryPolicy(R"(
HloModule large_streaming_fp8_conversion

elementwise {
  p0 = f32[67108864]{0} parameter(0)
  ROOT result = f8e5m2fnuz[67108864]{0} convert(p0)
}

ENTRY entry {
  p0 = f32[67108864]{0} parameter(0)
  ROOT fusion = f8e5m2fnuz[67108864]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
}
)"),
            FlyXTileMemoryPolicy::kNonTemporal);
}

TEST_F(FlyXTileElementwiseTest, KeepsSmallFusionOnNormalCachePath) {
  EXPECT_EQ(MemoryPolicy(R"(
HloModule small_fp8_conversion

elementwise {
  p0 = f32[1024]{0} parameter(0)
  ROOT result = f8e5m2fnuz[1024]{0} convert(p0)
}

ENTRY entry {
  p0 = f32[1024]{0} parameter(0)
  ROOT fusion = f8e5m2fnuz[1024]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
}
)"),
            FlyXTileMemoryPolicy::kCached);
}

TEST_F(FlyXTileElementwiseTest, KeepsBroadcastReuseOnNormalCachePath) {
  EXPECT_EQ(MemoryPolicy(R"(
HloModule large_broadcast_reuse

elementwise {
  p0 = f32[8192]{0} parameter(0)
  broadcast = f32[8192,8192]{1,0} broadcast(p0), dimensions={1}
  ROOT result = f32[8192,8192]{1,0} abs(broadcast)
}

ENTRY entry {
  p0 = f32[8192]{0} parameter(0)
  ROOT fusion = f32[8192,8192]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"),
            FlyXTileMemoryPolicy::kCached);
}

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

TEST_F(FlyXTileElementwiseTest, RecognizesTritonPredicateSurface) {
  EXPECT_TRUE(IsSupported(R"(
HloModule triton_predicate_surface

elementwise {
  p0 = pred[127,65]{1,0} parameter(0)
  p1 = pred[127,65]{1,0} parameter(1)
  sum = pred[127,65]{1,0} add(p0, p1)
  product = pred[127,65]{1,0} multiply(p0, p1)
  maximum = pred[127,65]{1,0} maximum(sum, product)
  minimum = pred[127,65]{1,0} minimum(maximum, p0)
  lower = pred[] constant(false)
  lowers = pred[127,65]{1,0} broadcast(lower), dimensions={}
  upper = pred[] constant(true)
  uppers = pred[127,65]{1,0} broadcast(upper), dimensions={}
  clamped = pred[127,65]{1,0} clamp(lowers, minimum, uppers)
  equal = pred[127,65]{1,0} compare(p0, p1), direction=EQ
  ROOT result = pred[127,65]{1,0} select(equal, clamped, sum)
}

ENTRY entry {
  p0 = pred[127,65]{1,0} parameter(0)
  p1 = pred[127,65]{1,0} parameter(1)
  ROOT fusion = pred[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesFp8StorageAndConversions) {
  EXPECT_TRUE(IsSupported(R"(
HloModule fp8_input_conversion

elementwise {
  p0 = f8e4m3fnuz[65]{0} parameter(0)
  converted = f32[65]{0} convert(p0)
  ROOT result = f32[65]{0} abs(converted)
}

ENTRY entry {
  p0 = f8e4m3fnuz[65]{0} parameter(0)
  ROOT fusion = f32[65]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));

  EXPECT_TRUE(IsSupported(R"(
HloModule fp8_output_conversion

elementwise {
  p0 = f32[65]{0} parameter(0)
  converted = f8e5m2fnuz[65]{0} convert(p0)
  ROOT result = f8e5m2fnuz[65]{0} abs(converted)
}

ENTRY entry {
  p0 = f32[65]{0} parameter(0)
  ROOT fusion = f8e5m2fnuz[65]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
}
)"));

  EXPECT_TRUE(IsSupported(R"(
HloModule ocp_fp8_conversion

elementwise {
  p0 = f8e4m3fn[65]{0} parameter(0)
  ROOT result = bf16[65]{0} convert(p0)
}

ENTRY entry {
  p0 = f8e4m3fn[65]{0} parameter(0)
  ROOT fusion = bf16[65]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));

  EXPECT_TRUE(IsSupported(R"(
HloModule fp8_native_value_ops

elementwise {
  predicate = pred[65]{0} parameter(0)
  p0 = f8e4m3fnuz[65]{0} parameter(1)
  reduced = f8e4m3fnuz[65]{0} reduce-precision(p0),
    exponent_bits=3, mantissa_bits=1
  fallback = f8e4m3fnuz[] constant(-1.5)
  fallbacks = f8e4m3fnuz[65]{0} broadcast(fallback), dimensions={}
  selected = f8e4m3fnuz[65]{0} select(predicate, reduced, fallbacks)
  ROOT result = f8e4m3fnuz[65]{0} abs(selected)
}

ENTRY entry {
  predicate = pred[65]{0} parameter(0)
  p0 = f8e4m3fnuz[65]{0} parameter(1)
  ROOT fusion = f8e4m3fnuz[65]{0} fusion(predicate, p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesDirectPackedS4InputConversion) {
  EXPECT_TRUE(IsSupported(R"(
HloModule packed_s4_input_conversion

elementwise {
  p0 = s4[65]{0:E(4)} parameter(0)
  widened = s8[65]{0} convert(p0)
  converted = bf16[65]{0} convert(widened)
  scale = bf16[] constant(0.5)
  scales = bf16[65]{0} broadcast(scale), dimensions={}
  ROOT result = bf16[65]{0} multiply(converted, scales)
}

ENTRY entry {
  p0 = s4[65]{0:E(4)} parameter(0)
  ROOT fusion = bf16[65]{0} fusion(p0), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesPackedS4PhysicalViews) {
  EXPECT_TRUE(IsSupported(R"(
HloModule packed_s4_physical_views

elementwise {
  p0 = s4[13,5]{1,0:E(4)} parameter(0)
  transposed = s4[5,13]{0,1:E(4)} transpose(p0), dimensions={1,0}
  widened = s8[5,13]{0,1} convert(transposed)
  ROOT result = bf16[5,13]{0,1} convert(widened)
}

ENTRY entry {
  p0 = s4[13,5]{1,0:E(4)} parameter(0)
  ROOT fusion = bf16[5,13]{0,1} fusion(p0), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesPackedS4OddOffsetSlice) {
  EXPECT_TRUE(IsSupported(R"(
HloModule packed_s4_slice

elementwise {
  p0 = s4[66]{0:E(4)} parameter(0)
  widened = s8[66]{0} convert(p0)
  ROOT result = s8[65]{0} slice(widened), slice={[1:66]}
}

ENTRY entry {
  p0 = s4[66]{0:E(4)} parameter(0)
  ROOT fusion = s8[65]{0} fusion(p0), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesDirectPackedS4OddOffsetSlice) {
  EXPECT_TRUE(IsSupported(R"(
HloModule direct_packed_s4_slice

elementwise {
  p0 = s4[66]{0:E(4)} parameter(0)
  sliced = s4[65]{0:E(4)} slice(p0), slice={[1:66]}
  widened = s8[65]{0} convert(sliced)
  ROOT result = bf16[65]{0} convert(widened)
}

ENTRY entry {
  p0 = s4[66]{0:E(4)} parameter(0)
  ROOT fusion = bf16[65]{0} fusion(p0), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesPackedS4DataMovingGraphs) {
  EXPECT_TRUE(IsSupported(R"(
HloModule packed_s4_data_moving

elementwise {
  p0 = s4[65]{0:E(4)} parameter(0)
  reversed = s4[65]{0:E(4)} reverse(p0), dimensions={0}
  widened = s8[65]{0} convert(reversed)
  converted = bf16[65]{0} convert(widened)
  zero = bf16[] constant(0)
  padded = bf16[67]{0} pad(converted, zero),
    padding=1_1
  ROOT result = bf16[134]{0} concatenate(padded, padded), dimensions={0}
}

ENTRY entry {
  p0 = s4[65]{0:E(4)} parameter(0)
  ROOT fusion = bf16[134]{0} fusion(p0), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesPackedS4BitcastConvert) {
  EXPECT_TRUE(IsSupported(R"(
HloModule packed_s4_bitcast_convert

elementwise {
  p0 = s4[65,4]{1,0:E(4)} parameter(0)
  widened = s8[65,4]{1,0} convert(p0)
  ROOT result = s32[65]{0} bitcast-convert(widened)
}

ENTRY entry {
  p0 = s4[65,4]{1,0:E(4)} parameter(0)
  ROOT fusion = s32[65]{0} fusion(p0), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesPackedS4ReduceWindow) {
  EXPECT_TRUE(IsSupported(R"(
HloModule packed_s4_reduce_window

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT result = bf16[] add(lhs, rhs)
}

elementwise {
  p0 = s4[67]{0:E(4)} parameter(0)
  widened = s8[67]{0} convert(p0)
  converted = bf16[67]{0} convert(widened)
  zero = bf16[] constant(0)
  ROOT result = bf16[67]{0} reduce-window(converted, zero),
    window={size=3 pad=1_1}, to_apply=add
}

ENTRY entry {
  p0 = s4[67]{0:E(4)} parameter(0)
  ROOT fusion = bf16[67]{0} fusion(p0), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesPackedS4OutputConversion) {
  EXPECT_TRUE(IsSupported(R"(
HloModule packed_s4_output_conversion

elementwise {
  p0 = s8[65]{0} parameter(0)
  ROOT result = s4[65]{0:E(4)} convert(p0)
}

ENTRY entry {
  p0 = s8[65]{0} parameter(0)
  ROOT fusion = s4[65]{0:E(4)} fusion(p0), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesShapeChangingPhysicalViews) {
  EXPECT_TRUE(IsSupported(R"(
HloModule reshape_view_elementwise

elementwise {
  p0 = s32[64,16]{1,0} parameter(0)
  reshaped = s32[16,64]{1,0} reshape(p0)
  ROOT result = s32[16,64]{1,0} negate(reshaped)
}

ENTRY entry {
  p0 = s32[64,16]{1,0} parameter(0)
  ROOT fusion = s32[16,64]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));

  EXPECT_TRUE(IsSupported(R"(
HloModule effective_transpose_view_elementwise

elementwise {
  p0 = f32[10,5]{1,0} parameter(0)
  transposed = f32[5,10]{0,1} transpose(p0), dimensions={1,0}
  ROOT result = f32[5,10]{0,1} abs(transposed)
}

ENTRY entry {
  p0 = f32[10,5]{1,0} parameter(0)
  ROOT fusion = f32[5,10]{0,1} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RejectsDataMovingTransposeView) {
  EXPECT_FALSE(IsSupported(R"(
HloModule data_moving_transpose_elementwise

elementwise {
  p0 = f32[10,5]{1,0} parameter(0)
  transposed = f32[5,10]{1,0} transpose(p0), dimensions={1,0}
  ROOT result = f32[5,10]{1,0} abs(transposed)
}

ENTRY entry {
  p0 = f32[10,5]{1,0} parameter(0)
  ROOT fusion = f32[5,10]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesTypeChangingBitcastViews) {
  EXPECT_TRUE(IsSupported(R"(
HloModule same_width_type_bitcast_elementwise

elementwise {
  p0 = f32[65]{0} parameter(0)
  bits = s32[65]{0} bitcast(p0)
  ROOT result = s32[65]{0} not(bits)
}

ENTRY entry {
  p0 = f32[65]{0} parameter(0)
  ROOT fusion = s32[65]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));

  EXPECT_TRUE(IsSupported(R"(
HloModule narrowing_bitcast_convert_elementwise

elementwise {
  p0 = s32[65]{0} parameter(0)
  bytes = s8[65,4]{1,0} bitcast-convert(p0)
  ROOT result = s8[65,4]{1,0} not(bytes)
}

ENTRY entry {
  p0 = s32[65]{0} parameter(0)
  ROOT fusion = s8[65,4]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));

  EXPECT_TRUE(IsSupported(R"(
HloModule widening_bitcast_convert_elementwise

elementwise {
  p0 = s8[65,4]{1,0} parameter(0)
  words = s32[65]{0} bitcast-convert(p0)
  ROOT result = s32[65]{0} not(words)
}

ENTRY entry {
  p0 = s8[65,4]{1,0} parameter(0)
  ROOT fusion = s32[65]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RejectsTypeChangingPredicateBitcast) {
  EXPECT_FALSE(IsSupported(R"(
HloModule predicate_type_bitcast_elementwise

elementwise {
  p0 = pred[65]{0} parameter(0)
  ROOT result = s8[65]{0} bitcast(p0)
}

ENTRY entry {
  p0 = pred[65]{0} parameter(0)
  ROOT fusion = s8[65]{0} fusion(p0), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest,
       RecognizesScalarExpressionBroadcastNormalizedToF32) {
  EXPECT_TRUE(IsSupported(R"(
HloModule elementwise_scalar_expression_broadcast

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p0_f32 = f32[128,64]{1,0} convert(p0)
  scale = bf16[] constant(1.5)
  scale_f32 = f32[] convert(scale)
  scales = f32[128,64]{1,0} broadcast(scale_f32), dimensions={}
  scaled = f32[128,64]{1,0} multiply(p0_f32, scales)
  ROOT result = bf16[128,64]{1,0} convert(scaled)
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

TEST_F(FlyXTileElementwiseTest, RecognizesCompleteFloatingUnaryMathSurface) {
  EXPECT_TRUE(IsSupported(R"(
HloModule floating_unary_math

elementwise {
  p0 = f32[33,68]{1,0} parameter(0)
  acos = f32[33,68]{1,0} acos(p0)
  acosh = f32[33,68]{1,0} acosh(p0)
  asin = f32[33,68]{1,0} asin(p0)
  asinh = f32[33,68]{1,0} asinh(p0)
  atanh = f32[33,68]{1,0} atanh(p0)
  cbrt = f32[33,68]{1,0} cbrt(p0)
  ceil = f32[33,68]{1,0} ceil(p0)
  cosine = f32[33,68]{1,0} cosine(p0)
  cosh = f32[33,68]{1,0} cosh(p0)
  erf = f32[33,68]{1,0} erf(p0)
  expm1 = f32[33,68]{1,0} exponential-minus-one(p0)
  floor = f32[33,68]{1,0} floor(p0)
  log1p = f32[33,68]{1,0} log-plus-one(p0)
  round = f32[33,68]{1,0} round-nearest-even(p0)
  sine = f32[33,68]{1,0} sine(p0)
  sinh = f32[33,68]{1,0} sinh(p0)
  tan = f32[33,68]{1,0} tan(p0)
  reduced = f32[33,68]{1,0} reduce-precision(p0),
    exponent_bits=5, mantissa_bits=10
  ROOT tuple = (f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0})
    tuple(acos, acosh, asin, asinh, atanh, cbrt, ceil, cosine, cosh, erf,
      expm1, floor, log1p, round, sine, sinh, tan, reduced)
}

ENTRY entry {
  p0 = f32[33,68]{1,0} parameter(0)
  ROOT fusion = (f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0},
    f32[33,68]{1,0}, f32[33,68]{1,0}, f32[33,68]{1,0})
    fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesCompleteFloatingBinaryMathSurface) {
  EXPECT_TRUE(IsSupported(R"(
HloModule floating_binary_math

elementwise {
  p0 = bf16[33,68]{1,0} parameter(0)
  p1 = bf16[33,68]{1,0} parameter(1)
  atan2 = bf16[33,68]{1,0} atan2(p0, p1)
  power = bf16[33,68]{1,0} power(p0, p1)
  remainder = bf16[33,68]{1,0} remainder(p0, p1)
  ROOT tuple = (bf16[33,68]{1,0}, bf16[33,68]{1,0}, bf16[33,68]{1,0})
    tuple(atan2, power, remainder)
}

ENTRY entry {
  p0 = bf16[33,68]{1,0} parameter(0)
  p1 = bf16[33,68]{1,0} parameter(1)
  ROOT fusion = (bf16[33,68]{1,0}, bf16[33,68]{1,0},
    bf16[33,68]{1,0}) fusion(p0, p1), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest,
       RecognizesCompleteSignedIntegerAndPredicateSurface) {
  EXPECT_TRUE(IsSupported(R"(
HloModule signed_integer_elementwise

elementwise {
  p0 = s32[33,68]{1,0} parameter(0)
  p1 = s32[33,68]{1,0} parameter(1)
  absolute = s32[33,68]{1,0} abs(p0)
  negated = s32[33,68]{1,0} negate(p1)
  inverted = s32[33,68]{1,0} not(p0)
  sum = s32[33,68]{1,0} add(absolute, negated)
  difference = s32[33,68]{1,0} subtract(sum, p1)
  product = s32[33,68]{1,0} multiply(difference, p0)
  one = s32[] constant(1)
  ones = s32[33,68]{1,0} broadcast(one), dimensions={}
  abs_denominator = s32[33,68]{1,0} abs(p1)
  denominator = s32[33,68]{1,0} maximum(abs_denominator, ones)
  quotient = s32[33,68]{1,0} divide(product, denominator)
  remainder = s32[33,68]{1,0} remainder(product, denominator)
  minimum = s32[33,68]{1,0} minimum(quotient, remainder)
  anded = s32[33,68]{1,0} and(minimum, inverted)
  ored = s32[33,68]{1,0} or(anded, p1)
  xored = s32[33,68]{1,0} xor(ored, p0)
  less = pred[33,68]{1,0} compare(sum, difference), direction=LT
  equal = pred[33,68]{1,0} compare(quotient, remainder), direction=EQ
  not_equal = pred[33,68]{1,0} not(equal)
  both = pred[33,68]{1,0} and(less, not_equal)
  either = pred[33,68]{1,0} or(less, equal)
  decision = pred[33,68]{1,0} xor(both, either)
  selected = s32[33,68]{1,0} select(decision, xored, product)
  lower = s32[] constant(-1000)
  lowers = s32[33,68]{1,0} broadcast(lower), dimensions={}
  upper = s32[] constant(1000)
  uppers = s32[33,68]{1,0} broadcast(upper), dimensions={}
  ROOT result = s32[33,68]{1,0} clamp(lowers, selected, uppers)
}

ENTRY entry {
  p0 = s32[33,68]{1,0} parameter(0)
  p1 = s32[33,68]{1,0} parameter(1)
  ROOT fusion = s32[33,68]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest,
       RecognizesExternalPredicateBuffersAndConversions) {
  EXPECT_TRUE(IsSupported(R"(
HloModule external_predicate_elementwise

elementwise {
  p0 = pred[33,68]{1,0} parameter(0)
  p1 = s32[33,68]{1,0} parameter(1)
  p0_s32 = s32[33,68]{1,0} convert(p0)
  sum = s32[33,68]{1,0} add(p0_s32, p1)
  sum_pred = pred[33,68]{1,0} convert(sum)
  inverted = pred[33,68]{1,0} not(p0)
  not_sum = pred[33,68]{1,0} not(sum_pred)
  both = pred[33,68]{1,0} and(sum_pred, p0)
  neither = pred[33,68]{1,0} and(not_sum, inverted)
  ROOT result = pred[33,68]{1,0} or(both, neither)
}

ENTRY entry {
  p0 = pred[33,68]{1,0} parameter(0)
  p1 = s32[33,68]{1,0} parameter(1)
  ROOT fusion = pred[33,68]{1,0} fusion(p0, p1), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesMultidimensionalIotaDag) {
  EXPECT_TRUE(IsSupported(R"(
HloModule multidimensional_iota_elementwise

elementwise {
  iota = s32[3,4,67,5]{3,2,1,0} iota(), iota_dimension=2
  converted = f32[3,4,67,5]{3,2,1,0} convert(iota)
  half = f32[] constant(0.5)
  half_broadcast = f32[3,4,67,5]{3,2,1,0} broadcast(half), dimensions={}
  ROOT result = f32[3,4,67,5]{3,2,1,0}
    multiply(converted, half_broadcast)
}

ENTRY entry {
  ROOT fusion = f32[3,4,67,5]{3,2,1,0} fusion(), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesMixedIntegerFloatingConversions) {
  EXPECT_TRUE(IsSupported(R"(
HloModule mixed_integer_floating_conversions

elementwise {
  p0 = s32[33,68]{1,0} parameter(0)
  p1 = f32[33,68]{1,0} parameter(1)
  converted = f32[33,68]{1,0} convert(p0)
  sum = f32[33,68]{1,0} add(converted, p1)
  lower = f32[] constant(-1000)
  lowers = f32[33,68]{1,0} broadcast(lower), dimensions={}
  upper = f32[] constant(1000)
  uppers = f32[33,68]{1,0} broadcast(upper), dimensions={}
  bounded = f32[33,68]{1,0} clamp(lowers, sum, uppers)
  ROOT result = s32[33,68]{1,0} convert(bounded)
}

ENTRY entry {
  p0 = s32[33,68]{1,0} parameter(0)
  p1 = f32[33,68]{1,0} parameter(1)
  ROOT fusion = s32[33,68]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)"));

  EXPECT_TRUE(IsSupported(R"(
HloModule wide_f64_to_narrow_s8_conversion

elementwise {
  p0 = f64[65]{0} parameter(0)
  offset = f64[] constant(37)
  offsets = f64[65]{0} broadcast(offset), dimensions={}
  shifted = f64[65]{0} add(p0, offsets)
  lower = f64[] constant(-100)
  lowers = f64[65]{0} broadcast(lower), dimensions={}
  upper = f64[] constant(100)
  uppers = f64[65]{0} broadcast(upper), dimensions={}
  bounded = f64[65]{0} clamp(lowers, shifted, uppers)
  ROOT result = s8[65]{0} convert(bounded)
}

ENTRY entry {
  p0 = f64[65]{0} parameter(0)
  ROOT fusion = s8[65]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"16"}}}
}
)"));
}

TEST_F(FlyXTileElementwiseTest, RecognizesF64ElementwiseDag) {
  EXPECT_TRUE(IsSupported(R"(
HloModule f64_elementwise

elementwise {
  p0 = f64[33,68]{1,0} parameter(0)
  p1 = f64[33,68]{1,0} parameter(1)
  absolute = f64[33,68]{1,0} abs(p0)
  sum = f64[33,68]{1,0} add(absolute, p1)
  sine = f64[33,68]{1,0} sine(sum)
  ROOT result = f64[33,68]{1,0} maximum(sine, p0)
}

ENTRY entry {
  p0 = f64[33,68]{1,0} parameter(0)
  p1 = f64[33,68]{1,0} parameter(1)
  ROOT fusion = f64[33,68]{1,0} fusion(p0, p1), kind=kCustom,
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

TEST_F(FlyXTileElementwiseTest, RecognizesGeneralLeadingReduction) {
  EXPECT_TRUE(IsSupported(R"(
HloModule general_leading_reduction

add_reduce {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

body {
  p0 = f32[257,259]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT result = f32[259]{0} reduce(p0, zero), dimensions={0},
      to_apply=add_reduce
}

ENTRY entry {
  p0 = f32[257,259]{1,0} parameter(0)
  ROOT fusion = f32[259]{0} fusion(p0), kind=kCustom, calls=body,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
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

TEST_F(FlyXTileElementwiseTest, RecognizesUnalignedContiguousConcatenateDag) {
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
  const HloInstruction* root = module->entry_computation()->root_instruction();
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

TEST_F(FlyXTileElementwiseTest, RecognizesLeadingDimensionBroadcast) {
  EXPECT_TRUE(IsSupported(R"(
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
  const HloInstruction* root = module->entry_computation()->root_instruction();
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
  const HloInstruction* root = module->entry_computation()->root_instruction();
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
  const HloInstruction* root = module->entry_computation()->root_instruction();
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
  const HloInstruction* root = module->entry_computation()->root_instruction();
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
