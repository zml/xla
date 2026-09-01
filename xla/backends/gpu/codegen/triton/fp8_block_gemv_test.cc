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

// A block-128 FP8 projection with quantized activations: an f8e4m3fn row and
// its own [m, k/128] scale against an f8e4m3fn weight on a [n/128, k/128] grid.
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

// The weight-only form the same matcher serves: a bf16 activation against an
// identity scale, which the frontend spells as an all-ones constant.
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
  // 1 reduces, 16 mmas, and 24 is neither -- the batch only the CUTLASS rung
  // can tile, which is why the answer depends on the capability.
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
  // sm_90 has no CUTLASS blockwise collective here, so nothing can tile 24.
  EXPECT_FALSE(Fp8BlockGemvSupportsScaledDot(RootDot(*module), Sm(9, 0)));
  EXPECT_TRUE(Fp8BlockGemvSupportsScaledDot(RootDot(*module), Sm(10, 3)));
}

// The build targets 12.0a; a plain 12.1 cubin has TMA compiled out, so the
// collective must not be advertised there or the arm claims a dot every config
// then declines.
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

  // Above sixteen rows the weight-only form is a prefill shape, where the
  // dequantize-and-cuBLAS path wins; the arm leaves it alone.
  TF_ASSERT_OK_AND_ASSIGN(
      auto prefill, ParseAndReturnVerifiedModule(W8A16Hlo(512, 5120, 512)));
  EXPECT_FALSE(Fp8BlockGemvSupportsScaledDot(RootDot(*prefill), Sm(10, 3)));
}

// bf16, f32 and ue8m0 are all accepted spellings of a block scale. The rungs
// differ in what they can read, but the matcher itself does not filter.
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
  // MXFP8: fp8 x fp8 on a 32-element block. A different format, a different
  // emitter, and not this arm's to claim.
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
  // What SplitkRewriter used to hand it: the same projection with a split-K
  // batch dimension inserted, which the matcher rejects on its dnums.
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
