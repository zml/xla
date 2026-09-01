/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/splitk_rewriter.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

se::DeviceDescription GetDeviceDescription() {
  return se::DeviceDescription::FromProto(
             ParseTextProto<stream_executor::GpuDeviceInfoProto>(
                 "core_count: 132")
                 .value())
      .value();
}

class SplitkRewriterTest : public HloHardwareIndependentTestBase {
 public:
  SplitkRewriterTest() : rewriter_(GetDeviceDescription()) {}

 protected:
  SplitkRewriter rewriter_;
};

TEST_F(SplitkRewriterTest, SmallNonContractingDimensionCauseSplitK) {
  const char* hlo_string = R"(
HloModule module

ENTRY test {
  lhs = f32[16,10240]{1,0} parameter(0)
  rhs = f32[10240,128]{1,0} parameter(1)
  ROOT dot = f32[16,128]{1,0} dot(lhs, rhs),
                              lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: dot({{.*}}), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
CHECK: ROOT {{.*}} = f32[16,128]{1,0} reduce
  )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, PaddingIsInserted) {
  // Huge K dimension to trigger 128 which is the largest possible splitK
  // (hoping to make the test less fragile as heuristic changes).
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    lhs = f32[16,102401]{1,0} parameter(0)
    rhs = f32[102401,128]{1,0} parameter(1)
    ROOT dot = f32[16,128]{1,0} dot(lhs, rhs),
                                lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: f32[16,102912]{1,0} pad(%lhs, %constant), padding=0_0x0_511
    )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, AccumulatorTypeIsDifferentFromOutputType) {
  // Huge K dimension to trigger 128 which is the largest possible splitK
  // (hoping to make the test less fragile as heuristic changes).
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    lhs = bf16[16,102400]{1,0} parameter(0)
    rhs = bf16[102400,128]{1,0} parameter(1)
    ROOT dot = bf16[16,128]{1,0} dot(lhs, rhs),
                                lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: f32{{.*}} dot(
CHECK: f32{{.*}} reduce(
CHECK: bf16[16,128]{1,0} convert(
)")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, NoSplitKIfEnoughWork) {
  // Small K is not profitable to split.
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[1024,512]{1,0} parameter(0)
      rhs = f32[512,2048]{1,0} parameter(1)
      ROOT dot = f32[1024,2048]{1,0} dot(lhs, rhs),
                             lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SplitkRewriterTest, DoNotSplitKS32) {
  // We would split K otherwise, but for s32 operands we don't, because neither
  // cuBLAS nor Triton support it.
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = s32[16,10240]{1,0} parameter(0)
      rhs = s32[10240,128]{1,0} parameter(1)
      ROOT dot = s32[16,128]{1,0} dot(lhs, rhs),
                    lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SplitkRewriterTest, B200FailingModuleTest) {
  se::DeviceDescription b200_desc = TestGpuDeviceInfo::B200SXMDeviceInfo();
  SplitkRewriter splitk_rewriter(b200_desc);

  const char* hlo_string = R"(
HloModule jit_fwd

ENTRY %main.2 (broadcast: f32[2,128,128], b.1: f32[128,2,128]) -> f32[2,128,128] {
  %broadcast = f32[2,128,128]{2,1,0} parameter(0)
  %b.1 = f32[128,2,128]{2,1,0} parameter(1)
  ROOT %dot.1 = f32[2,128,128]{2,1,0} dot(%broadcast, %b.1), lhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CHECK_OK(splitk_rewriter.HloModulePass::Run(module.get()));
}

TEST_F(SplitkRewriterTest, ForceSplitK) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[1024,512]{1,0} parameter(0)
      rhs = f32[512,2048]{1,0} parameter(1)
      ROOT dot = f32[1024,2048]{1,0} dot(lhs, rhs),
                             lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_force_split_k(2);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: dot({{.*}}), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
CHECK: ROOT {{.*}} = f32[1024,2048]{1,0} reduce
  )")
                  .value_or(false));
}

constexpr absl::string_view kUnderOccupiedFp8Projection = R"(
    HloModule module

    ENTRY test {
      w = f8e4m3fn[5120,6144]{1,0} parameter(0)
      s = bf16[5120]{0} parameter(1)
      a = bf16[6144,1]{1,0} parameter(2)
      cw = bf16[5120,6144]{1,0} convert(w)
      bs = bf16[5120,6144]{1,0} broadcast(s), dimensions={0}
      m = bf16[5120,6144]{1,0} multiply(cw, bs)
      ROOT d = bf16[5120,1]{1,0} dot(m, a),
                             lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

int64_t SplitKOf(const HloModule& module) {
  for (const HloInstruction* instr :
       module.entry_computation()->instructions()) {
    if (instr->opcode() != HloOpcode::kDot) {
      continue;
    }
    const auto& dnums = instr->dot_dimension_numbers();
    if (dnums.lhs_batch_dimensions().empty()) {
      return 0;
    }
    return instr->operand(0)->shape().dimensions(
        dnums.lhs_batch_dimensions(0));
  }
  return 0;
}

TEST_F(SplitkRewriterTest, TargetWavesNeverSplitsLessThanTheCostModel) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> floored,
      ParseAndReturnVerifiedModule(kUnderOccupiedFp8Projection));
  CHECK_OK(rewriter_.HloModulePass::Run(floored.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> plain,
      ParseAndReturnVerifiedModule(kUnderOccupiedFp8Projection));
  CHECK_OK(rewriter_.HloModulePass::Run(plain.get()));

  EXPECT_GE(SplitKOf(*floored), SplitKOf(*plain));
  EXPECT_GT(SplitKOf(*floored), 1);
}

TEST_F(SplitkRewriterTest, TargetWavesSplitsAnUnderOccupiedDot) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kUnderOccupiedFp8Projection));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: dot({{.*}}), lhs_batch_dims={1}
CHECK: reduce
  )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, TargetWavesLeavesAWellOccupiedDotAlone) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[16384,512]{1,0} parameter(0)
      rhs = f32[512,16384]{1,0} parameter(1)
      ROOT dot = f32[16384,16384]{1,0} dot(lhs, rhs),
                             lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> with_floor,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed_with_floor,
                          rewriter_.HloModulePass::Run(with_floor.get()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> without_floor,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed_without_floor,
                          rewriter_.HloModulePass::Run(without_floor.get()));

  EXPECT_EQ(changed_with_floor, changed_without_floor);
  EXPECT_EQ(with_floor->ToString(), without_floor->ToString());
}

TEST_F(SplitkRewriterTest, TargetWavesLeavesADotJustUnderOneWaveAlone) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      x = bf16[16,5120]{1,0} parameter(0)
      w = f8e4m3fn[5120,16384]{1,0} parameter(1)
      cw = bf16[5120,16384]{1,0} convert(w)
      ROOT d = bf16[16,16384]{1,0} dot(x, cw),
                             lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_FALSE(changed);
}

constexpr absl::string_view kNvfp4DownProjection = R"(
    HloModule module

    ENTRY test {
      lhs = f4e2m1fn[16,17408]{1,0:E(4)} parameter(0)
      rhs = f4e2m1fn[5120,17408]{1,0:E(4)} parameter(1)
      ls = f8e4m3fn[16,1088]{1,0} parameter(2)
      rs = f8e4m3fn[5120,1088]{1,0} parameter(3)
      ROOT sd = bf16[16,5120]{1,0} scaled-dot(lhs, rhs, ls, rs),
                             lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })";

TEST_F(SplitkRewriterTest, SplitsAnUnderOccupiedScaledDotUnasked) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kNvfp4DownProjection));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(SplitkRewriterTest, SplitsAScaledDotAndItsScalesTogether) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kNvfp4DownProjection));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  ASSERT_TRUE(changed);

  const HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kScaledDot);
  ASSERT_NE(dot, nullptr);
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  ASSERT_EQ(dnums.lhs_batch_dimensions_size(), 1);
  ASSERT_EQ(dnums.rhs_batch_dimensions_size(), 1);
  const int64_t lhs_k = dnums.lhs_contracting_dimensions(0);
  const int64_t rhs_k = dnums.rhs_contracting_dimensions(0);
  const int64_t split_k =
      dot->operand(0)->shape().dimensions(dnums.lhs_batch_dimensions(0));
  EXPECT_GT(split_k, 1);
  EXPECT_EQ(dot->operand(0)->shape().dimensions(lhs_k) /
                dot->operand(2)->shape().dimensions(lhs_k),
            16);
  EXPECT_EQ(dot->operand(1)->shape().dimensions(rhs_k) /
                dot->operand(3)->shape().dimensions(rhs_k),
            16);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: reduce
CHECK: ROOT {{.*}} = bf16[16,5120]{1,0} convert
  )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, DeclinesAScaledDotWhoseScalesDoNotDivide) {
  constexpr absl::string_view kIndivisible = R"(
    HloModule m
    ENTRY e {
      lhs = f4e2m1fn[16,17360]{1,0:E(4)} parameter(0)
      rhs = f4e2m1fn[5120,17360]{1,0:E(4)} parameter(1)
      ls = f8e4m3fn[16,1085]{1,0} parameter(2)
      rs = f8e4m3fn[5120,1085]{1,0} parameter(3)
      ROOT sd = bf16[16,5120]{1,0} scaled-dot(lhs, rhs, ls, rs),
                             lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kIndivisible));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
