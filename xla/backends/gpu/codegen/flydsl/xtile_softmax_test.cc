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
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {
namespace {

class FlyXTileSoftmaxTest : public HloHardwareIndependentTestBase {
 protected:
  bool IsSupported(std::string max_identity) {
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
  p0 = bf16[64,4096]{1,0} parameter(0)
  converted = f32[64,4096]{1,0} convert(p0)
  max_identity = f32[] constant(__MAX_IDENTITY__)
  row_max = f32[64]{0} reduce(converted, max_identity), dimensions={1},
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

ENTRY entry {
  p0 = bf16[64,4096]{1,0} parameter(0)
  ROOT fusion = bf16[64,4096]{1,0} fusion(p0), kind=kInput, calls=softmax
}
)";
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(
            absl::StrReplaceAll(kSoftmaxHlo,
                                {{"__MAX_IDENTITY__", max_identity}}))
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

TEST_F(FlyXTileSoftmaxTest, RejectsIncorrectMaximumIdentity) {
  EXPECT_FALSE(IsSupported("0"));
}

}  // namespace
}  // namespace xla::gpu::flydsl
