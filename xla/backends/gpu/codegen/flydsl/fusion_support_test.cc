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
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {
namespace {

class FlyFusionSupportTest : public HloHardwareIndependentTestBase {};

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
  HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  EXPECT_TRUE(ContainsUnsupportedCustomCall(*fusion));

  auto device = TestGpuDeviceInfo::AMDMI210DeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device);
  EXPECT_TRUE(ContainsUnsupportedCustomCall(analysis));
  std::unique_ptr<FusionInterface> emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo(analysis));
  EXPECT_NE(dynamic_cast<CustomFusion*>(emitter.get()), nullptr);
}

}  // namespace
}  // namespace xla::gpu::flydsl
