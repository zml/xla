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
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class FlyFusionDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {};

class FlyFusionPipelineDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloPjRtGpuTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_flydsl_fusion(true);
    debug_options.set_xla_gpu_experimental_enable_fusion_autotuner(false);
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }
};

class FlyFusionAutotuningPipelineDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloPjRtGpuTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_flydsl_fusion(true);
    debug_options.set_xla_gpu_experimental_enable_fusion_autotuner(true);
    debug_options.set_xla_gpu_autotune_level(3);
    debug_options.clear_xla_gpu_experimental_autotune_backends();
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::FLY_FUSION);
    return debug_options;
  }
};

TEST_F(FlyFusionPipelineDeviceTest, FormsAndExecutesFlyFusionEndToEnd) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fusion_pipeline

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

ENTRY main {
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
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloComputation* entry = optimized->entry_computation();
  EXPECT_EQ(entry->instruction_count(), 2) << optimized->ToString();
  const HloInstruction* root = entry->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(
      RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.01}));
}

TEST_F(FlyFusionPipelineDeviceTest,
       FormsAndExecutesRank4DoubleStabilizedSoftmax) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_rank4_double_stabilized_softmax_pipeline

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

ENTRY main {
  p0 = bf16[32,128,128]{2,1,0} parameter(0)
  view = bf16[2,16,128,128]{3,2,1,0} reshape(p0)
  converted = f32[2,16,128,128]{3,2,1,0} convert(view)
  minus_inf = f32[] constant(-inf)
  row_max.0 = f32[2,16,128]{2,1,0} reduce(converted, minus_inf),
    dimensions={3}, to_apply=maximum
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
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(
      RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesNativeElementwiseFusionEndToEnd) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_elementwise_pipeline

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  add = bf16[128,64]{1,0} add(p0, p1)
  scale = bf16[] constant(1.5)
  scale_broadcast = bf16[128,64]{1,0} broadcast(scale), dimensions={}
  scaled = bf16[128,64]{1,0} multiply(add, scale_broadcast)
  ROOT result = bf16[128,64]{1,0} maximum(scaled, p0)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo,
                            ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesRaggedNativeElementwiseFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ragged_native_elementwise_pipeline

ENTRY main {
  p0 = bf16[127,65]{1,0} parameter(0)
  p1 = bf16[127,65]{1,0} parameter(1)
  add = bf16[127,65]{1,0} add(p0, p1)
  scale = bf16[] constant(1.5)
  scale_broadcast = bf16[127,65]{1,0} broadcast(scale), dimensions={}
  scaled = bf16[127,65]{1,0} multiply(add, scale_broadcast)
  ROOT result = bf16[127,65]{1,0} maximum(scaled, p0)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo,
                            ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesNarrowingBf16RowReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_narrowing_row_reduction_pipeline

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = bf16[65,256]{1,0} parameter(0)
  converted = f32[65,256]{1,0} convert(p0)
  zero = f32[] constant(0)
  row_sum = f32[65]{0} reduce(converted, zero), dimensions={1}, to_apply=add
  scale = f32[] constant(0.00390625)
  scales = f32[65]{0} broadcast(scale), dimensions={}
  mean = f32[65]{0} multiply(row_sum, scales)
  ROOT result = bf16[65]{0} convert(mean)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_EQ(root->fused_expression_root()->opcode(), HloOpcode::kConvert);
  bool contains_reduce = false;
  for (const HloInstruction* instruction :
       root->fused_instructions_computation()->instructions()) {
    contains_reduce |= instruction->opcode() == HloOpcode::kReduce;
  }
  EXPECT_TRUE(contains_reduce) << optimized->ToString();
  EXPECT_TRUE(
      RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesRaggedBf16RowReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_ragged_row_reduction_pipeline

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = bf16[65,259]{1,0} parameter(0)
  converted = f32[65,259]{1,0} convert(p0)
  zero = f32[] constant(0)
  ROOT row_sum = f32[65]{0} reduce(converted, zero), dimensions={1},
    to_apply=add
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config =
      backend_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind) << optimized->ToString();
  EXPECT_GE(fusion_config.block_level_fusion_config().vector_size_bits(), 64)
      << optimized->ToString();
  EXPECT_TRUE(
      RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesRank3Bf16RmsNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank3_bf16_rms_norm_pipeline

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
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
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config =
      backend_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind) << optimized->ToString();
  EXPECT_GE(fusion_config.block_level_fusion_config().vector_size_bits(), 64)
      << optimized->ToString();
  bool contains_reduce = false;
  bool contains_rsqrt = false;
  for (const HloInstruction* instruction :
       root->fused_instructions_computation()->instructions()) {
    contains_reduce |= instruction->opcode() == HloOpcode::kReduce;
    contains_rsqrt |= instruction->opcode() == HloOpcode::kRsqrt;
  }
  EXPECT_TRUE(contains_reduce) << optimized->ToString();
  EXPECT_TRUE(contains_rsqrt) << optimized->ToString();
  EXPECT_TRUE(
      RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesResidualRmsNormAfterSplitKReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_residual_rms_norm_after_split_k_pipeline

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  residual = bf16[2,17,259]{2,1,0} parameter(0)
  partials = f32[2,34,259]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  projected = f32[34,259]{1,0} reduce(partials, zero), dimensions={0},
    to_apply=add
  projected_bf16 = bf16[34,259]{1,0} convert(projected)
  projected_view = bf16[2,17,259]{2,1,0} reshape(projected_bf16)
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
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(
      RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.02, /*arel=*/0.02}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsNativeMultiOutputWithoutTritonFlag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_multi_output_pipeline

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  sum = bf16[128,64]{1,0} add(p0, p1)
  product = bf16[128,64]{1,0} multiply(sum, p0)
  ROOT result = (bf16[128,64]{1,0}, bf16[128,64]{1,0})
    tuple(sum, product)
})";
  ASSERT_FALSE(GetDebugOptionsForTest()
                   .xla_gpu_unsupported_enable_triton_multi_output_fusion());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* fly_fusion = nullptr;
  for (const HloInstruction* instruction :
       optimized->entry_computation()->instructions()) {
    if (instruction->opcode() != HloOpcode::kFusion ||
        !instruction->shape().IsTuple()) {
      continue;
    }
    ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                         instruction->backend_config<GpuBackendConfig>());
    if (backend_config.fusion_backend_config().kind() == kFlyFusionKind) {
      fly_fusion = instruction;
      break;
    }
  }
  ASSERT_NE(fly_fusion, nullptr) << optimized->ToString();
  EXPECT_TRUE(
      RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionPipelineDeviceTest,
       ExecutesLateFlyLoopFusionWithoutAutotuning) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_late_loop_fusion_pipeline

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  sum = bf16[128,64]{1,0} add(p0, p1)
  ROOT result = bf16[128,64]{1,0} multiply(sum, p0)
}

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0, p1), kind=kLoop,
    calls=elementwise
})";

  EXPECT_TRUE(
      RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

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

TEST_F(FlyFusionDeviceTest, NativeBf16ElementwiseDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_native_elementwise

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  add = bf16[128,64]{1,0} add(p0, p1)
  scale = bf16[] constant(1.5)
  scale_broadcast = bf16[128,64]{1,0} broadcast(scale), dimensions={}
  scaled = bf16[128,64]{1,0} multiply(add, scale_broadcast)
  ROOT result = bf16[128,64]{1,0} maximum(scaled, p0)
}

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRaggedF16MixedPrecisionElementwiseDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ragged_f16_native_elementwise

elementwise {
  p0 = f16[127,65]{1,0} parameter(0)
  p1 = f16[127,65]{1,0} parameter(1)
  p0_f32 = f32[127,65]{1,0} convert(p0)
  p1_f32 = f32[127,65]{1,0} convert(p1)
  sum = f32[127,65]{1,0} add(p0_f32, p1_f32)
  zero = f32[] constant(0)
  zeros = f32[127,65]{1,0} broadcast(zero), dimensions={}
  positive = pred[127,65]{1,0} compare(sum, zeros), direction=GT
  absolute = f32[127,65]{1,0} abs(sum)
  negated = f32[127,65]{1,0} negate(absolute)
  selected = f32[127,65]{1,0} select(positive, absolute, negated)
  ROOT result = f16[127,65]{1,0} convert(selected)
}

ENTRY main {
  p0 = f16[127,65]{1,0} parameter(0)
  p1 = f16[127,65]{1,0} parameter(1)
  ROOT fusion = f16[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionDeviceTest, NativeTinyF32ElementwiseTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_tiny_f32_native_elementwise

elementwise {
  p0 = f32[3]{0} parameter(0)
  absolute = f32[3]{0} abs(p0)
  one = f32[] constant(1)
  ones = f32[3]{0} broadcast(one), dimensions={}
  ROOT result = f32[3]{0} add(absolute, ones)
}

ENTRY main {
  p0 = f32[3]{0} parameter(0)
  ROOT fusion = f32[3]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"1", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(FlyFusionDeviceTest, NativeMultiOutputBf16ElementwiseDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_native_multi_output_elementwise

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  sum = bf16[128,64]{1,0} add(p0, p1)
  product = bf16[128,64]{1,0} multiply(sum, p0)
  ROOT tuple = (bf16[128,64]{1,0}, bf16[128,64]{1,0}) tuple(sum, product)
}

ENTRY main {
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
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16CompareSelectClampDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_compare_select_clamp

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

ENTRY main {
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
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16MixedPrecisionSigmoidDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_sigmoid

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

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionDeviceTest, NativeF32TranscendentalDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_transcendentals

elementwise {
  p0 = f32[128,64]{1,0} parameter(0)
  absolute = f32[128,64]{1,0} abs(p0)
  one = f32[] constant(1)
  one_broadcast = f32[128,64]{1,0} broadcast(one), dimensions={}
  positive = f32[128,64]{1,0} add(absolute, one_broadcast)
  logarithm = f32[128,64]{1,0} log(positive)
  square_root = f32[128,64]{1,0} sqrt(positive)
  reciprocal_square_root = f32[128,64]{1,0} rsqrt(positive)
  hyperbolic_tangent = f32[128,64]{1,0} tanh(logarithm)
  sum = f32[128,64]{1,0} add(square_root, reciprocal_square_root)
  ROOT result = f32[128,64]{1,0} add(sum, hyperbolic_tangent)
}

ENTRY main {
  p0 = f32[128,64]{1,0} parameter(0)
  ROOT fusion = f32[128,64]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(FlyFusionDeviceTest, F16Softmax31x125Tail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_softmax_tail

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
  p0 = f16[31,125]{1,0} parameter(0)
  converted = f32[31,125]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[31]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[31,125]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[31,125]{1,0} subtract(converted, broadcast_max)
  exponential = f32[31,125]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[31]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[31,125]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[31,125]{1,0} divide(exponential, broadcast_sum)
  ROOT result = f16[31,125]{1,0} convert(normalized)
}

ENTRY main {
  p0 = f16[31,125]{1,0} parameter(0)
  ROOT fusion = f16[31,125]{1,0} fusion(p0), kind=kCustom,
    calls=softmax,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","125"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.01}));
}

TEST_F(FlyFusionDeviceTest, F32Softmax31x125Tail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_softmax_tail

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
  p0 = f32[31,125]{1,0} parameter(0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[31]{0} reduce(p0, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[31,125]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[31,125]{1,0} subtract(p0, broadcast_max)
  exponential = f32[31,125]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[31]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[31,125]{1,0} broadcast(row_sum), dimensions={0}
  ROOT result = f32[31,125]{1,0} divide(exponential, broadcast_sum)
}

ENTRY main {
  p0 = f32[31,125]{1,0} parameter(0)
  ROOT fusion = f32[31,125]{1,0} fusion(p0), kind=kCustom,
    calls=softmax,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","125"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/0.0001, /*arel=*/0.001}));
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

TEST_F(FlyFusionDeviceTest, Bf16Transpose32Tile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose_32_tile

transpose {
  p0 = bf16[96,160]{1,0} parameter(0)
  ROOT result = bf16[160,96]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[96,160]{1,0} parameter(0)
  ROOT fusion = bf16[160,96]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32","32"]}],
        "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, Bf16Transpose128Tile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose_128_tile

transpose {
  p0 = bf16[256,384]{1,0} parameter(0)
  ROOT result = bf16[384,256]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[256,384]{1,0} parameter(0)
  ROOT fusion = bf16[384,256]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["128","128"]}],
        "num_stages":"1", "num_warps":"16", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, TransformerQkvSliceTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_transformer_qkv_slice_transpose

transpose {
  p0 = bf16[256,3072]{1,0} parameter(0)
  view = bf16[2,128,3,16,64]{4,3,2,1,0} bitcast(p0)
  q = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [1:2], [0:16], [0:64]}
  ROOT result = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(q),
    dimensions={0,2,3,4,1}
}

ENTRY main {
  p0 = bf16[256,3072]{1,0} parameter(0)
  ROOT fusion = bf16[2,1,16,64,128]{4,3,2,1,0} fusion(p0),
    kind=kCustom, calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64","64"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, MultiOutputTransformerQkvTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_multi_output_transformer_qkv_transpose

transpose {
  p0 = bf16[256,3072]{1,0} parameter(0)
  view = bf16[2,128,3,16,64]{4,3,2,1,0} bitcast(p0)
  q_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [0:1], [0:16], [0:64]}
  q = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(q_slice),
    dimensions={0,2,3,4,1}
  k_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [1:2], [0:16], [0:64]}
  k = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(k_slice),
    dimensions={0,2,3,4,1}
  v_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [2:3], [0:16], [0:64]}
  v = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(v_slice),
    dimensions={0,2,3,4,1}
  ROOT result = (bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0}) tuple(q, k, v)
}

ENTRY main {
  p0 = bf16[256,3072]{1,0} parameter(0)
  ROOT fusion = (bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0}) fusion(p0), kind=kCustom,
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

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesMultiOutputTransformerQkvTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_multi_output_transformer_qkv_transpose_pipeline

ENTRY main {
  p0 = bf16[256,3072]{1,0} parameter(0)
  view = bf16[2,128,3,16,64]{4,3,2,1,0} reshape(p0)
  q_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [0:1], [0:16], [0:64]}
  q = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(q_slice),
    dimensions={0,2,3,4,1}
  k_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [1:2], [0:16], [0:64]}
  k = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(k_slice),
    dimensions={0,2,3,4,1}
  v_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [2:3], [0:16], [0:64]}
  v = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(v_slice),
    dimensions={0,2,3,4,1}
  ROOT result = (bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0}) tuple(q, k, v)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* qkv_fusion = nullptr;
  for (const HloInstruction* instruction :
       optimized->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kFusion &&
        instruction->IsMultiOutputFusion()) {
      qkv_fusion = instruction;
    }
  }
  ASSERT_NE(qkv_fusion, nullptr) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       qkv_fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_EQ(backend_config.fusion_backend_config()
                .block_level_fusion_config()
                .output_tiles(0)
                .sizes_size(),
            2)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, TransformerContextTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_transformer_context_transpose

transpose {
  p0 = bf16[32,64,128]{2,1,0} parameter(0)
  view = bf16[2,16,64,128]{3,2,1,0} bitcast(p0)
  ROOT result = bf16[2,128,16,64]{3,2,1,0} transpose(view),
    dimensions={0,3,1,2}
}

ENTRY main {
  p0 = bf16[32,64,128]{2,1,0} parameter(0)
  ROOT fusion = bf16[2,128,16,64]{3,2,1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["128","128"]}],
        "num_stages":"1", "num_warps":"16", "num_ctas":"1"}}}
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

TEST_F(FlyFusionDeviceTest, NativeF32RowReduction) {
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
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(FlyFusionDeviceTest, NativeConvertedBf16RowMaximum) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_bf16_row_maximum

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(lhs, rhs)
}

reduction {
  p0 = bf16[65,512]{1,0} parameter(0)
  converted = f32[65,512]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  ROOT result = f32[65]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = bf16[65,512]{1,0} parameter(0)
  ROOT fusion = f32[65]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeNarrowingF16RowMaximum) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_narrowing_f16_row_maximum

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(lhs, rhs)
}

reduction {
  p0 = f16[65,256]{1,0} parameter(0)
  converted = f32[65,256]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[65]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  ROOT result = f16[65]{0} convert(row_max)
}

ENTRY main {
  p0 = f16[65,256]{1,0} parameter(0)
  ROOT fusion = f16[65]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeFusedBf16SquaredDifferenceReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_fused_bf16_squared_difference_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = bf16[65,512]{1,0} parameter(0)
  p1 = bf16[65,512]{1,0} parameter(1)
  lhs = f32[65,512]{1,0} convert(p0)
  rhs = f32[65,512]{1,0} convert(p1)
  difference = f32[65,512]{1,0} subtract(lhs, rhs)
  square = f32[65,512]{1,0} multiply(difference, difference)
  zero = f32[] constant(0)
  row_sum = f32[65]{0} reduce(square, zero), dimensions={1}, to_apply=add
  scale = f32[] constant(0.25)
  scales = f32[65]{0} broadcast(scale), dimensions={}
  ROOT result = f32[65]{0} multiply(row_sum, scales)
}

ENTRY main {
  p0 = bf16[65,512]{1,0} parameter(0)
  p1 = bf16[65,512]{1,0} parameter(1)
  ROOT fusion = f32[65]{0} fusion(p0, p1), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(FlyFusionDeviceTest, NativeRank3Bf16RmsNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank3_bf16_rms_norm

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
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
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

TEST_F(FlyFusionDeviceTest, GenericInPlaceDynamicUpdateSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_in_place_dynamic_update_slice

update_slice {
  input = f32[64,96]{1,0} parameter(0)
  update = f32[17,31]{1,0} parameter(1)
  row = s32[] parameter(2)
  column = s32[] parameter(3)
  scale = f32[] constant(1.25)
  broadcast = f32[17,31]{1,0} broadcast(scale), dimensions={}
  scaled = f32[17,31]{1,0} multiply(update, broadcast)
  ROOT result = f32[64,96]{1,0} dynamic-update-slice(
      input, scaled, row, column)
}

ENTRY main {
  input = f32[64,96]{1,0} parameter(0)
  update = f32[17,31]{1,0} parameter(1)
  row = s32[] parameter(2)
  column = s32[] parameter(3)
  ROOT fusion = f32[64,96]{1,0} fusion(input, update, row, column),
    kind=kCustom, calls=update_slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
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
