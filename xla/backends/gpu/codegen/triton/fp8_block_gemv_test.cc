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

#include "xla/backends/gpu/codegen/triton/fp8_block_gemv.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

se::GpuComputeCapability Sm(int major, int minor) {
  return se::GpuComputeCapability(se::CudaComputeCapability(major, minor));
}

std::string W8A8Hlo(int m, int n, int k,
                    absl::string_view scale_type = "f32") {
  return absl::Substitute(R"(
    HloModule m
    ENTRY main {
      x = f8e4m3fn[$0,$2]{1,0} parameter(0)
      w = f8e4m3fn[$1,$2]{1,0} parameter(1)
      xs = $4[$0,$3]{1,0} parameter(2)
      ws = $4[$5,$3]{1,0} parameter(3)
      ROOT d = bf16[$0,$1]{1,0} scaled-dot(x, w, xs, ws),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })",
                          m, n, k, k / 128, scale_type, n / 128);
}

std::string W8A16Hlo(int m, int n, int k,
                     absl::string_view scale_type = "bf16") {
  return absl::Substitute(R"(
    HloModule m
    ENTRY main {
      x = bf16[$0,$2]{1,0} parameter(0)
      w = f8e4m3fn[$1,$2]{1,0} parameter(1)
      xs = bf16[1,1]{1,0} constant({{1}})
      ws = $4[$5,$3]{1,0} parameter(2)
      ROOT d = bf16[$0,$1]{1,0} scaled-dot(x, w, xs, ws),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })",
                          m, n, k, k / 128, scale_type, n / 128);
}

class Fp8BlockGemvTest : public HloHardwareIndependentTestBase {
 protected:
  const HloScaledDotInstruction& RootDot(const HloModule& module) {
    return *Cast<HloScaledDotInstruction>(
        module.entry_computation()->root_instruction());
  }
};

TEST_F(Fp8BlockGemvTest, ClaimsAW8A8ProjectionAtEveryBatch) {
  // 24 is the batch only the CUTLASS rung tiles.
  for (int m : {1, 16, 24, 64, 2048}) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(W8A8Hlo(m, 5120, 512)));
    EXPECT_TRUE(Fp8BlockGemvSupportsScaledDot(RootDot(*module), Sm(10, 3)))
        << "m = " << m;
  }
}

TEST_F(Fp8BlockGemvTest, AnUntileableBatchNeedsTheCutlassRung) {
  EXPECT_FALSE(Fp8BlockGemvBatchNeedsCutlass(1));
  EXPECT_FALSE(Fp8BlockGemvBatchNeedsCutlass(16));
  EXPECT_FALSE(Fp8BlockGemvBatchNeedsCutlass(64));
  EXPECT_TRUE(Fp8BlockGemvBatchNeedsCutlass(24));

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(W8A8Hlo(24, 5120, 512)));
  EXPECT_FALSE(Fp8BlockGemvSupportsScaledDot(RootDot(*module), Sm(9, 0)));
  EXPECT_TRUE(Fp8BlockGemvSupportsScaledDot(RootDot(*module), Sm(10, 3)));
}

TEST_F(Fp8BlockGemvTest, ConsumerBlackwellIsTwelvePointZeroOnly) {
  EXPECT_TRUE(HasCutlassBlockGemm(Sm(10, 0)));
  EXPECT_TRUE(HasCutlassBlockGemm(Sm(10, 3)));
  EXPECT_TRUE(HasCutlassBlockGemm(Sm(12, 0)));
  EXPECT_FALSE(HasCutlassBlockGemm(Sm(12, 1)));
  EXPECT_FALSE(HasCutlassBlockGemm(Sm(9, 0)));
}

TEST_F(Fp8BlockGemvTest, ClaimsAWeightOnlyProjectionUpToSixteenRows) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(W8A16Hlo(1, 5120, 512)));
  EXPECT_TRUE(Fp8BlockGemvSupportsScaledDot(RootDot(*module), Sm(10, 3)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto prefill, ParseAndReturnVerifiedModule(W8A16Hlo(512, 5120, 512)));
  EXPECT_FALSE(Fp8BlockGemvSupportsScaledDot(RootDot(*prefill), Sm(10, 3)));
}

TEST_F(Fp8BlockGemvTest, TakesEveryScaleSpelling) {
  for (absl::string_view scale : {"bf16", "f32", "f8e8m0fnu"}) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        ParseAndReturnVerifiedModule(W8A16Hlo(1, 5120, 512, scale)));
    EXPECT_TRUE(Fp8BlockGemvSupportsScaledDot(RootDot(*module), Sm(10, 3)))
        << scale;
  }
}

TEST_F(Fp8BlockGemvTest, DeclinesAScaleGridThatIsNotBlock128) {
  const std::string mxfp8 = R"(
    HloModule m
    ENTRY main {
      x = f8e4m3fn[1,512]{1,0} parameter(0)
      w = f8e4m3fn[5120,512]{1,0} parameter(1)
      xs = f8e8m0fnu[1,16]{1,0} parameter(2)
      ws = f8e8m0fnu[5120,16]{1,0} parameter(3)
      ROOT d = bf16[1,5120]{1,0} scaled-dot(x, w, xs, ws),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(mxfp8));
  EXPECT_FALSE(Fp8BlockGemvSupportsScaledDot(RootDot(*module), Sm(10, 3)));
}

TEST_F(Fp8BlockGemvTest, DeclinesABatchedDot) {
  const std::string batched = R"(
    HloModule m
    ENTRY main {
      x = f8e4m3fn[4,1,128]{2,1,0} parameter(0)
      w = f8e4m3fn[4,5120,128]{2,1,0} parameter(1)
      xs = f32[4,1,1]{2,1,0} parameter(2)
      ws = f32[4,40,1]{2,1,0} parameter(3)
      ROOT d = f32[4,1,5120]{2,1,0} scaled-dot(x, w, xs, ws),
          lhs_batch_dims={0}, lhs_contracting_dims={2},
          rhs_batch_dims={0}, rhs_contracting_dims={2}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(batched));
  EXPECT_FALSE(Fp8BlockGemvSupportsScaledDot(RootDot(*module), Sm(10, 3)));
}

}  // namespace
}  // namespace xla::gpu
