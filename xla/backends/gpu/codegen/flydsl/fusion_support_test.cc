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

#include "xla/backends/gpu/codegen/flydsl/fusion_support.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/backends/gpu/codegen/custom.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_elementwise.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_reduction.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {
namespace {

class FlyFusionSupportTest : public HloHardwareIndependentTestBase {};

TEST_F(FlyFusionSupportTest, DistinguishesNativeAndGenericFlyRoutes) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> native_module,
                       ParseAndReturnVerifiedModule(R"(
HloModule fly_native_route

body {
  p0 = f32[7,11]{1,0} parameter(0)
  ROOT result = f32[7,11]{1,0} negate(p0)
}

ENTRY main {
  p0 = f32[7,11]{1,0} parameter(0)
  ROOT fusion = f32[7,11]{1,0} fusion(p0), kind=kCustom, calls=body,
      backend_config={"fusion_backend_config":{"kind":"__fly"}}
}
)"));
  auto device = TestGpuDeviceInfo::AMDMI210DeviceInfo();
  HloFusionAnalysis native_analysis = HloFusionAnalysis::Create(
      *native_module->entry_computation()->root_instruction(), device);
  EXPECT_EQ(ClassifyFlyFusion(native_analysis), FlyFusionRoute::kElementwise);
  EXPECT_TRUE(IsNativeFlyFusionRoute(ClassifyFlyFusion(native_analysis)));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> generic_module,
                       ParseAndReturnVerifiedModule(R"(
HloModule fly_generic_route

body {
  p0 = u32[17,11]{1,0} parameter(0)
  ROOT result = u32[17,11]{1,0} copy(p0)
}

ENTRY main {
  p0 = u32[17,11]{1,0} parameter(0)
  ROOT fusion = u32[17,11]{1,0} fusion(p0), kind=kCustom, calls=body,
      backend_config={"fusion_backend_config":{"kind":"__fly"}}
}
)"));
  HloFusionAnalysis generic_analysis = HloFusionAnalysis::Create(
      *generic_module->entry_computation()->root_instruction(), device);
  EXPECT_EQ(ClassifyFlyFusion(generic_analysis), FlyFusionRoute::kGenericXla);
  EXPECT_EQ(FlyFusionRouteName(ClassifyFlyFusion(generic_analysis)),
            "generic-xla-emitter");
  EXPECT_FALSE(IsNativeFlyFusionRoute(ClassifyFlyFusion(generic_analysis)));
}

TEST_F(FlyFusionSupportTest, PrefersNativeRowReductionOverElementwiseFallback) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule fly_rms_norm_route

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

body {
  p0 = bf16[2,128,1024]{2,1,0} parameter(0)
  converted = f32[2,128,1024]{2,1,0} convert(p0)
  squared = f32[2,128,1024]{2,1,0} multiply(converted, converted)
  zero = f32[] constant(0)
  row_sum = f32[2,128]{1,0} reduce(squared, zero), dimensions={2},
      to_apply=add
  reciprocal_width = f32[] constant(0.0009765625)
  widths = f32[2,128]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,128]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,128]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,128]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,128]{1,0} rsqrt(variance)
  scales = f32[2,128,1024]{2,1,0} broadcast(reciprocal_stddev),
      dimensions={0,1}
  normalized = f32[2,128,1024]{2,1,0} multiply(converted, scales)
  ROOT result = bf16[2,128,1024]{2,1,0} convert(normalized)
}

ENTRY main {
  p0 = bf16[2,128,1024]{2,1,0} parameter(0)
  ROOT fusion = bf16[2,128,1024]{2,1,0} fusion(p0), kind=kCustom,
      calls=body,
      backend_config={"fusion_backend_config":{"kind":"__fly"}}
}
)"));
  auto device = TestGpuDeviceInfo::AMDMI210DeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *module->entry_computation()->root_instruction(), device);

  // This overlap is intentional: elementwise remains a broad correctness
  // fallback, while the row matcher identifies the efficient launch domain.
  EXPECT_TRUE(IsFlyXTileElementwiseFusion(analysis));
  EXPECT_TRUE(IsFlyXTileRowReductionFusion(analysis));
  EXPECT_EQ(ClassifyFlyFusion(analysis), FlyFusionRoute::kRowReduction);
}

TEST_F(FlyFusionSupportTest, RoutesOpaqueCustomCallToErrorEmitter) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule fly_opaque_custom_call_routing

body {
  p0 = f32[8]{0} parameter(0)
  ROOT unsupported_call = f32[8]{0} custom-call(p0),
      custom_call_target="__opaque$unsupported"
}

ENTRY main {
  p0 = f32[8]{0} parameter(0)
  ROOT fusion = f32[8]{0} fusion(p0), kind=kCustom, calls=body,
      backend_config={"fusion_backend_config":{"kind":"__fly"}}
}
)"));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_TRUE(ContainsUnsupportedCustomCall(*fusion));

  auto device = TestGpuDeviceInfo::AMDMI210DeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device);
  EXPECT_TRUE(ContainsUnsupportedCustomCall(analysis));
  std::unique_ptr<FusionInterface> emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo(analysis));
  EXPECT_NE(dynamic_cast<CustomFusion*>(emitter.get()), nullptr);
}

TEST_F(FlyFusionSupportTest,
       CreatesNativeMultiOutputEmitterFromSymbolicTilingConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule fly_multi_output_symbolic_tiles

body {
  p0 = bf16[128,256]{1,0} parameter(0)
  p1 = bf16[128,256]{1,0} parameter(1)
  sum = bf16[128,256]{1,0} add(p0, p1)
  product = bf16[128,256]{1,0} multiply(sum, p0)
  ROOT result = (bf16[128,256]{1,0}, bf16[128,256]{1,0})
      tuple(sum, product)
}

ENTRY main {
  p0 = bf16[128,256]{1,0} parameter(0)
  p1 = bf16[128,256]{1,0} parameter(1)
  ROOT fusion = (bf16[128,256]{1,0}, bf16[128,256]{1,0})
      fusion(p0, p1), kind=kCustom, calls=body,
      backend_config={"fusion_backend_config":{"kind":"__fly",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","256"]},
                          {"sizes":["1","256"]}],
          "num_warps":"1","num_ctas":1,"num_stages":1}}}
}
)"));
  auto device = TestGpuDeviceInfo::AMDMI210DeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *module->entry_computation()->root_instruction(), device);
  EXPECT_EQ(ClassifyFlyFusion(analysis), FlyFusionRoute::kElementwise);

  std::unique_ptr<FusionInterface> emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo(analysis));
  EXPECT_NE(emitter, nullptr);
}

TEST_F(FlyFusionSupportTest, CreatesNativeEmitterForLargeElementwiseDag) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule fly_large_elementwise_dag

body {
  p0 = f32[128,1024]{1,0} parameter(0)
  n00 = f32[128,1024]{1,0} negate(p0)
  n01 = f32[128,1024]{1,0} negate(n00)
  n02 = f32[128,1024]{1,0} negate(n01)
  n03 = f32[128,1024]{1,0} negate(n02)
  n04 = f32[128,1024]{1,0} negate(n03)
  n05 = f32[128,1024]{1,0} negate(n04)
  n06 = f32[128,1024]{1,0} negate(n05)
  n07 = f32[128,1024]{1,0} negate(n06)
  n08 = f32[128,1024]{1,0} negate(n07)
  n09 = f32[128,1024]{1,0} negate(n08)
  n10 = f32[128,1024]{1,0} negate(n09)
  n11 = f32[128,1024]{1,0} negate(n10)
  n12 = f32[128,1024]{1,0} negate(n11)
  n13 = f32[128,1024]{1,0} negate(n12)
  n14 = f32[128,1024]{1,0} negate(n13)
  n15 = f32[128,1024]{1,0} negate(n14)
  n16 = f32[128,1024]{1,0} negate(n15)
  n17 = f32[128,1024]{1,0} negate(n16)
  n18 = f32[128,1024]{1,0} negate(n17)
  n19 = f32[128,1024]{1,0} negate(n18)
  n20 = f32[128,1024]{1,0} negate(n19)
  n21 = f32[128,1024]{1,0} negate(n20)
  n22 = f32[128,1024]{1,0} negate(n21)
  n23 = f32[128,1024]{1,0} negate(n22)
  n24 = f32[128,1024]{1,0} negate(n23)
  n25 = f32[128,1024]{1,0} negate(n24)
  n26 = f32[128,1024]{1,0} negate(n25)
  n27 = f32[128,1024]{1,0} negate(n26)
  n28 = f32[128,1024]{1,0} negate(n27)
  n29 = f32[128,1024]{1,0} negate(n28)
  n30 = f32[128,1024]{1,0} negate(n29)
  ROOT result = f32[128,1024]{1,0} negate(n30)
}

ENTRY main {
  p0 = f32[128,1024]{1,0} parameter(0)
  ROOT fusion = f32[128,1024]{1,0} fusion(p0), kind=kCustom, calls=body,
      backend_config={"fusion_backend_config":{"kind":"__fly",
        "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
          "num_warps":"4","num_ctas":1,"num_stages":1,
          "vector_size_bits":"128"}}}
}
)"));
  auto device = TestGpuDeviceInfo::AMDMI210DeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *module->entry_computation()->root_instruction(), device);
  EXPECT_EQ(ClassifyFlyFusion(analysis), FlyFusionRoute::kElementwise);

  std::unique_ptr<FusionInterface> emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo(analysis));
  EXPECT_NE(emitter, nullptr);
}

}  // namespace
}  // namespace xla::gpu::flydsl
