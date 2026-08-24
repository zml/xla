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
#include "absl/strings/str_cat.h"
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
  view = bf16[2,16,128,128]{3,2,1,0} bitcast(p0)
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

}  // namespace
}  // namespace xla::gpu::flydsl
