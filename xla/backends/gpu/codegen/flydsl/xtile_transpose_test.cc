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

#include "xla/backends/gpu/codegen/flydsl/xtile_transpose.h"

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

class FlyXTileTransposeTest : public HloHardwareIndependentTestBase {
 protected:
  bool IsSupported(const std::string& hlo) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(
        *root, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
    return IsFlyXTileTransposeFusion(analysis);
  }
};

TEST_F(FlyXTileTransposeTest, RecognizesTiledBf16Transpose) {
  EXPECT_TRUE(IsSupported(R"(
HloModule transpose

transpose {
  p0 = bf16[128,192]{1,0} parameter(0)
  ROOT result = bf16[192,128]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY entry {
  p0 = bf16[128,192]{1,0} parameter(0)
  ROOT fusion = bf16[192,128]{1,0} fusion(p0), kind=kInput,
    calls=transpose
}
)"));
}

TEST_F(FlyXTileTransposeTest, RejectsNonBf16Transpose) {
  EXPECT_FALSE(IsSupported(R"(
HloModule transpose

transpose {
  p0 = f32[128,192]{1,0} parameter(0)
  ROOT result = f32[192,128]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY entry {
  p0 = f32[128,192]{1,0} parameter(0)
  ROOT fusion = f32[192,128]{1,0} fusion(p0), kind=kInput,
    calls=transpose
}
)"));
}

TEST_F(FlyXTileTransposeTest, RejectsPartialTiles) {
  EXPECT_FALSE(IsSupported(R"(
HloModule transpose

transpose {
  p0 = bf16[96,200]{1,0} parameter(0)
  ROOT result = bf16[200,96]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY entry {
  p0 = bf16[96,200]{1,0} parameter(0)
  ROOT fusion = bf16[200,96]{1,0} fusion(p0), kind=kInput,
    calls=transpose
}
)"));
}

}  // namespace
}  // namespace xla::gpu::flydsl
