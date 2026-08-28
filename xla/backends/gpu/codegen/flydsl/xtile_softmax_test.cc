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

#include "xla/backends/gpu/codegen/flydsl/xtile_softmax.h"

#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "xla/backends/gpu/codegen/flydsl/layer_norm_support.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {
namespace {

class FlyXTileSoftmaxTest : public HloHardwareIndependentTestBase {
 protected:
  bool IsSupported(std::string max_identity,
                   std::string interface_type = "bf16",
                   int64_t columns = 4096) {
    constexpr char kSoftmaxHlo[] = R"(
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
  p0 = __TYPE__[64,__COLUMNS__]{1,0} parameter(0)
  __CONVERSION__
  max_identity = f32[] constant(__MAX_IDENTITY__)
  row_max = f32[64]{0} reduce(__COMPUTE_VALUE__, max_identity), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[64,__COLUMNS__]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[64,__COLUMNS__]{1,0} subtract(__COMPUTE_VALUE__, broadcast_max)
  exponential = f32[64,__COLUMNS__]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[64,__COLUMNS__]{1,0} broadcast(row_sum), dimensions={0}
  __RESULT__
}

ENTRY entry {
  p0 = __TYPE__[64,__COLUMNS__]{1,0} parameter(0)
  ROOT fusion = __TYPE__[64,__COLUMNS__]{1,0} fusion(p0), kind=kInput,
    calls=softmax
}
)";
    const std::string column_string = std::to_string(columns);
    const bool f32_interface = interface_type == "f32";
    const std::string conversion =
        f32_interface
            ? ""
            : absl::StrCat("converted = f32[64,", column_string,
                           "]{1,0} convert(p0)");
    const std::string compute_value = f32_interface ? "p0" : "converted";
    const std::string result =
        f32_interface
            ? absl::StrCat("ROOT result = f32[64,", column_string,
                           "]{1,0} divide(exponential, broadcast_sum)")
            : absl::StrCat(
                  "normalized = f32[64,", column_string,
                  "]{1,0} divide(exponential, broadcast_sum)\n"
                  "  ROOT result = ",
                  interface_type, "[64,", column_string,
                  "]{1,0} convert(normalized)");
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(absl::StrReplaceAll(
            kSoftmaxHlo,
            {{"__TYPE__", interface_type},
             {"__COLUMNS__", column_string},
             {"__CONVERSION__", conversion},
             {"__COMPUTE_VALUE__", compute_value},
             {"__MAX_IDENTITY__", max_identity},
             {"__RESULT__", result}}))
            .value();
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(
        *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
    return IsFlySoftmaxFusion(analysis);
  }
};

TEST_F(FlyXTileSoftmaxTest, RecognizesCanonicalBf16Softmax) {
  EXPECT_TRUE(IsSupported("-inf"));
}

TEST_F(FlyXTileSoftmaxTest, RecognizesF16SoftmaxWithTail) {
  EXPECT_TRUE(IsSupported("-inf", "f16", 125));
}

TEST_F(FlyXTileSoftmaxTest, RecognizesF32SoftmaxWithTail) {
  EXPECT_TRUE(IsSupported("-inf", "f32", 125));
}

TEST_F(FlyXTileSoftmaxTest, RecognizesExternalRowOffsetSoftmax) {
  constexpr char kSoftmaxHlo[] = R"(
add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

softmax {
  input = f32[64,4096]{1,0} parameter(0)
  row_offset = f32[64]{0} parameter(1)
  row_offsets = f32[64,4096]{1,0} broadcast(row_offset), dimensions={0}
  shifted = f32[64,4096]{1,0} subtract(input, row_offsets)
  exponential = f32[64,4096]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[64,4096]{1,0} broadcast(row_sum), dimensions={0}
  ROOT result = f32[64,4096]{1,0} divide(exponential, broadcast_sum)
}

ENTRY entry {
  input = f32[64,4096]{1,0} parameter(0)
  row_offset = f32[64]{0} parameter(1)
  ROOT fusion = f32[64,4096]{1,0} fusion(input, row_offset), kind=kInput,
    calls=softmax
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kSoftmaxHlo).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  EXPECT_TRUE(IsFlySoftmaxFusion(analysis));
}

TEST_F(FlyXTileSoftmaxTest,
       RecognizesStabilizedExternalRowOffsetSoftmax) {
  constexpr char kSoftmaxHlo[] = R"(
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
  input = bf16[4,31,125]{2,1,0} parameter(0)
  converted = f32[4,31,125]{2,1,0} convert(input)
  row_offset = f32[4,31]{1,0} parameter(1)
  row_offsets = f32[4,31,125]{2,1,0} broadcast(row_offset),
    dimensions={0,1}
  pre_shift = f32[4,31,125]{2,1,0} subtract(converted, row_offsets)
  minus_inf = f32[] constant(-inf)
  row_max = f32[4,31]{1,0} reduce(pre_shift, minus_inf), dimensions={2},
    to_apply=maximum
  row_maxes = f32[4,31,125]{2,1,0} broadcast(row_max), dimensions={0,1}
  shifted = f32[4,31,125]{2,1,0} subtract(pre_shift, row_maxes)
  exponential = f32[4,31,125]{2,1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[4,31]{1,0} reduce(exponential, zero), dimensions={2},
    to_apply=add
  row_sums = f32[4,31,125]{2,1,0} broadcast(row_sum), dimensions={0,1}
  normalized = f32[4,31,125]{2,1,0} divide(exponential, row_sums)
  ROOT result = bf16[4,31,125]{2,1,0} convert(normalized)
}

ENTRY entry {
  input = bf16[4,31,125]{2,1,0} parameter(0)
  row_offset = f32[4,31]{1,0} parameter(1)
  ROOT fusion = bf16[4,31,125]{2,1,0} fusion(input, row_offset),
    kind=kInput, calls=softmax
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kSoftmaxHlo).value();
  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  EXPECT_TRUE(IsFlySoftmaxFusion(analysis));
  EXPECT_TRUE(FlySoftmaxRecomputesMaximumAfterExternalRowOffset(
      *root->fused_expression_root()));
}

TEST_F(FlyXTileSoftmaxTest, RejectsIncorrectMaximumIdentity) {
  EXPECT_FALSE(IsSupported("0"));
}

TEST_F(FlyXTileSoftmaxTest,
       RecognizesRank4DoubleStabilizedBf16Softmax) {
  constexpr char kSoftmaxHlo[] = R"(
maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

maximum.1 {
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
  p0 = bf16[32,128,128]{2,1,0} parameter(0)
  converted_flat = f32[32,128,128]{2,1,0} convert(p0)
  converted = f32[2,16,128,128]{3,2,1,0} bitcast(converted_flat)
  minus_inf = f32[] constant(-inf)
  row_max_flat = f32[32,128]{1,0} reduce(converted_flat, minus_inf),
    dimensions={2}, to_apply=maximum
  row_max.0 = f32[2,16,128]{2,1,0} bitcast(row_max_flat)
  broadcast_max.0 = f32[2,16,128,128]{3,2,1,0}
    broadcast(row_max.0), dimensions={0,1,2}
  shifted.0 = f32[2,16,128,128]{3,2,1,0}
    subtract(converted, broadcast_max.0)
  row_max.1 = f32[2,16,128]{2,1,0} reduce(shifted.0, minus_inf),
    dimensions={3}, to_apply=maximum.1
  broadcast_max.1 = f32[2,16,128,128]{3,2,1,0}
    broadcast(row_max.1), dimensions={0,1,2}
  shifted.1 = f32[2,16,128,128]{3,2,1,0}
    subtract(shifted.0, broadcast_max.1)
  exponential = f32[2,16,128,128]{3,2,1,0} exponential(shifted.1)
  zero = f32[] constant(0)
  row_sum = f32[2,16,128]{2,1,0} reduce(exponential, zero),
    dimensions={3}, to_apply=add
  broadcast_sum = f32[2,16,128,128]{3,2,1,0}
    broadcast(row_sum), dimensions={0,1,2}
  normalized = f32[2,16,128,128]{3,2,1,0}
    divide(exponential, broadcast_sum)
  ROOT result = bf16[2,16,128,128]{3,2,1,0} convert(normalized)
}

ENTRY entry {
  p0 = bf16[32,128,128]{2,1,0} parameter(0)
  ROOT fusion = bf16[2,16,128,128]{3,2,1,0} fusion(p0), kind=kInput,
    calls=softmax
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kSoftmaxHlo).value();
  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  EXPECT_TRUE(IsFlySoftmaxFusion(analysis));
}

TEST_F(FlyXTileSoftmaxTest, RecognizesAffineTrainingBf16LayerNorm) {
  constexpr char kLayerNormHlo[] = R"(
add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

layer_norm {
  input = bf16[256,4096]{1,0} parameter(0)
  gamma = bf16[4096]{0} parameter(1)
  beta = bf16[4096]{0} parameter(2)
  input_f32 = f32[256,4096]{1,0} convert(input)
  zero = f32[] constant(0)
  sum = f32[256]{0} reduce(input_f32, zero), dimensions={1}, to_apply=add
  reciprocal = f32[] constant(0.000244140625)
  reciprocals = f32[256]{0} broadcast(reciprocal), dimensions={}
  mean = f32[256]{0} multiply(sum, reciprocals)
  means = f32[256,4096]{1,0} broadcast(mean), dimensions={0}
  centered = f32[256,4096]{1,0} subtract(input_f32, means)
  squared = f32[256,4096]{1,0} multiply(centered, centered)
  square_sum = f32[256]{0} reduce(squared, zero), dimensions={1}, to_apply=add
  variance = f32[256]{0} multiply(square_sum, reciprocals)
  epsilon = f32[] constant(1e-5)
  epsilons = f32[256]{0} broadcast(epsilon), dimensions={}
  variance_epsilon = f32[256]{0} add(variance, epsilons)
  reciprocal_stddev = f32[256]{0} rsqrt(variance_epsilon)
  reciprocal_stddev_broadcast = f32[256,4096]{1,0}
      broadcast(reciprocal_stddev), dimensions={0}
  normalized = f32[256,4096]{1,0}
      multiply(centered, reciprocal_stddev_broadcast)
  gamma_f32 = f32[4096]{0} convert(gamma)
  gamma_broadcast = f32[256,4096]{1,0} broadcast(gamma_f32), dimensions={1}
  scaled = f32[256,4096]{1,0} multiply(normalized, gamma_broadcast)
  beta_f32 = f32[4096]{0} convert(beta)
  beta_broadcast = f32[256,4096]{1,0} broadcast(beta_f32), dimensions={1}
  shifted = f32[256,4096]{1,0} add(scaled, beta_broadcast)
  result = bf16[256,4096]{1,0} convert(shifted)
  ROOT outputs = (bf16[256,4096]{1,0}, f32[256]{0}, f32[256]{0})
      tuple(result, mean, reciprocal_stddev)
}

ENTRY entry {
  input = bf16[256,4096]{1,0} parameter(0)
  gamma = bf16[4096]{0} parameter(1)
  beta = bf16[4096]{0} parameter(2)
  ROOT fusion = (bf16[256,4096]{1,0}, f32[256]{0}, f32[256]{0})
      fusion(input, gamma, beta), kind=kInput, calls=layer_norm
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kLayerNormHlo).value();
  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  std::optional<FlyLayerNormDescriptor> descriptor =
      GetFlyLayerNormDescriptor(analysis);
  ASSERT_TRUE(descriptor.has_value());
  ASSERT_NE(descriptor->gamma, nullptr);
  ASSERT_NE(descriptor->beta, nullptr);
  EXPECT_EQ(descriptor->gamma->parameter_number(), 1);
  EXPECT_EQ(descriptor->beta->parameter_number(), 2);
  EXPECT_EQ(descriptor->output_count, 3);
}

TEST_F(FlyXTileSoftmaxTest, RecognizesMomentsTrainingBf16LayerNorm) {
  constexpr char kLayerNormHlo[] = R"(
add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

layer_norm {
  input = bf16[256,4096]{1,0} parameter(0)
  gamma = bf16[4096]{0} parameter(1)
  beta = bf16[4096]{0} parameter(2)
  input_f32 = f32[256,4096]{1,0} convert(input)
  input_square = f32[256,4096]{1,0} multiply(input_f32, input_f32)
  zero = f32[] constant(0)
  square_sum = f32[256]{0} reduce(input_square, zero), dimensions={1},
      to_apply=add
  reciprocal_columns = f32[] constant(0.000244140625)
  reciprocal_columns_broadcast = f32[256]{0}
      broadcast(reciprocal_columns), dimensions={}
  square_mean = f32[256]{0}
      multiply(square_sum, reciprocal_columns_broadcast)
  sum = f32[256]{0} reduce(input_f32, zero), dimensions={1}, to_apply=add
  mean = f32[256]{0} multiply(sum, reciprocal_columns_broadcast)
  mean_square = f32[256]{0} multiply(mean, mean)
  variance = f32[256]{0} subtract(square_mean, mean_square)
  epsilon = f32[] constant(1e-5)
  epsilons = f32[256]{0} broadcast(epsilon), dimensions={}
  variance_epsilon = f32[256]{0} add(variance, epsilons)
  reciprocal_stddev = f32[256]{0} rsqrt(variance_epsilon)
  reciprocal_stddev_broadcast = f32[256,4096]{1,0}
      broadcast(reciprocal_stddev), dimensions={0}
  means = f32[256,4096]{1,0} broadcast(mean), dimensions={0}
  centered = f32[256,4096]{1,0} subtract(input_f32, means)
  normalized = f32[256,4096]{1,0}
      multiply(centered, reciprocal_stddev_broadcast)
  gamma_f32 = f32[4096]{0} convert(gamma)
  gamma_broadcast = f32[256,4096]{1,0} broadcast(gamma_f32), dimensions={1}
  scaled = f32[256,4096]{1,0} multiply(normalized, gamma_broadcast)
  beta_f32 = f32[4096]{0} convert(beta)
  beta_broadcast = f32[256,4096]{1,0} broadcast(beta_f32), dimensions={1}
  shifted = f32[256,4096]{1,0} add(scaled, beta_broadcast)
  result = bf16[256,4096]{1,0} convert(shifted)
  reciprocal_stddev_cube = f32[256]{0}
      divide(reciprocal_stddev, variance_epsilon)
  ROOT outputs = (bf16[256,4096]{1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})
      tuple(result, mean, reciprocal_stddev, reciprocal_stddev_cube)
}

ENTRY entry {
  input = bf16[256,4096]{1,0} parameter(0)
  gamma = bf16[4096]{0} parameter(1)
  beta = bf16[4096]{0} parameter(2)
  ROOT fusion = (bf16[256,4096]{1,0}, f32[256]{0}, f32[256]{0}, f32[256]{0})
      fusion(input, gamma, beta), kind=kInput, calls=layer_norm
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(kLayerNormHlo).value();
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *fusion, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  std::optional<FlyLayerNormDescriptor> descriptor =
      GetFlyLayerNormDescriptor(analysis);
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_TRUE(IsFlyLayerNormFusion(analysis));
  EXPECT_EQ(descriptor->output_count, 4);
  EXPECT_TRUE(descriptor->uses_moments_variance);
  EXPECT_NE(descriptor->reciprocal_stddev_cube, nullptr);
}

}  // namespace
}  // namespace xla::gpu::flydsl
