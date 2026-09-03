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

#include "xla/backends/gpu/codegen/triton/nvfp4_decode_dot.h"

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
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

std::string DecodeDotHlo(int rows, int n, int k) {
  return absl::Substitute(R"(
    HloModule m
    ENTRY main {
      x = f4e2m1fn[$0,$2] parameter(0)
      w = f4e2m1fn[$1,$2] parameter(1)
      xs = f8e4m3fn[$0,$3] parameter(2)
      ws = f8e4m3fn[$1,$3] parameter(3)
      ROOT d = bf16[$0,$1] scaled-dot(x, w, xs, ws),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })",
                          rows, n, k, k / 16);
}

class Nvfp4DecodeDotTest : public HloHardwareIndependentTestBase {
 protected:
  const HloScaledDotInstruction& RootDot(const HloModule& module) {
    return *Cast<HloScaledDotInstruction>(
        module.entry_computation()->root_instruction());
  }
};

TEST_F(Nvfp4DecodeDotTest, LimitsAreKeyedOnTheArchitecture) {
  const Nvfp4DecodeLimits& sm103 = Nvfp4DecodeLimitsFor(Sm(10, 3));
  EXPECT_TRUE(sm103.claim);
  EXPECT_TRUE(sm103.swap);
  EXPECT_EQ(sm103.min_weight_tile, 128);  // the tcgen05 MMA's rows

  EXPECT_TRUE(Nvfp4DecodeLimitsFor(Sm(10, 0)).claim);

  const Nvfp4DecodeLimits& sm120 = Nvfp4DecodeLimitsFor(Sm(12, 0));
  EXPECT_TRUE(sm120.claim);
  EXPECT_TRUE(sm120.swap);
  EXPECT_EQ(sm120.min_weight_tile, 16);
  EXPECT_EQ(sm120.min_batch_tile, 16);

  EXPECT_FALSE(Nvfp4DecodeLimitsFor(Sm(9, 0)).claim);
  EXPECT_FALSE(Nvfp4DecodeLimitsFor(Sm(8, 9)).claim);
  EXPECT_FALSE(
      Nvfp4DecodeLimitsFor(se::GpuComputeCapability(se::RocmComputeCapability(
                               "gfx942")))
          .claim);
}

TEST_F(Nvfp4DecodeDotTest, ContractingTileLegalityIsOnePlace) {
  EXPECT_EQ(WidestNvfp4BlockK(17408), 512);
  EXPECT_EQ(WidestNvfp4BlockK(640), 128);  // 256 and 512 do not divide it
  EXPECT_EQ(WidestNvfp4BlockK(144), 0);    // nothing does
  EXPECT_TRUE(HasNvfp4BlockK(1536));
  EXPECT_FALSE(HasNvfp4BlockK(144));

  EXPECT_EQ(Nvfp4BlockKAtMost(4096, 256), 256);
  EXPECT_EQ(Nvfp4BlockKAtMost(128, 256), 128);
  EXPECT_EQ(Nvfp4BlockKAtMost(640, 256), 128);
  EXPECT_EQ(Nvfp4BlockKAtMost(17408, 256), 256);  // capped, not widest
}

TEST_F(Nvfp4DecodeDotTest, TheSeedAlwaysDividesTheContraction) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           DecodeDotHlo(16, 5120, 640)));
  std::optional<Nvfp4DecodeDotConfig> seed =
      Nvfp4DecodeDotConfigFor(RootDot(*module), Sm(10, 3));
  ASSERT_TRUE(seed.has_value());
  EXPECT_EQ(seed->block_k, 128);
  EXPECT_EQ(640 % seed->block_k, 0);
}

TEST_F(Nvfp4DecodeDotTest, DeclinesAContractionNoTileServes) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           DecodeDotHlo(16, 5120, 144)));
  EXPECT_FALSE(MatchNvfp4DecodeDot(RootDot(*module), Sm(10, 3)).has_value());
}

TEST_F(Nvfp4DecodeDotTest, ClaimsADecodeProjectionOnSm103) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           DecodeDotHlo(16, 5120, 4096)));
  std::optional<Nvfp4DecodeDotSpec> spec =
      MatchNvfp4DecodeDot(RootDot(*module), Sm(10, 3));
  ASSERT_TRUE(spec.has_value());
  EXPECT_EQ(spec->batch, 16);
  EXPECT_EQ(spec->weight_rows, 5120);
  EXPECT_EQ(spec->k, 4096);
  EXPECT_FALSE(spec->weight_on_lhs);
}

TEST_F(Nvfp4DecodeDotTest, DeclinesWhatIsNotADecodeProjection) {
  TF_ASSERT_OK_AND_ASSIGN(auto prefill, ParseAndReturnVerifiedModule(
                                            DecodeDotHlo(512, 5120, 4096)));
  EXPECT_FALSE(MatchNvfp4DecodeDot(RootDot(*prefill), Sm(10, 3)).has_value());
  TF_ASSERT_OK_AND_ASSIGN(auto narrow, ParseAndReturnVerifiedModule(
                                           DecodeDotHlo(16, 64, 4096)));
  EXPECT_FALSE(MatchNvfp4DecodeDot(RootDot(*narrow), Sm(10, 3)).has_value());
  TF_ASSERT_OK_AND_ASSIGN(auto hopper, ParseAndReturnVerifiedModule(
                                           DecodeDotHlo(16, 5120, 4096)));
  EXPECT_FALSE(MatchNvfp4DecodeDot(RootDot(*hopper), Sm(9, 0)).has_value());
}

TEST_F(Nvfp4DecodeDotTest, ClaimsOnSm120Too) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           DecodeDotHlo(16, 5120, 4096)));
  std::optional<Nvfp4DecodeDotConfig> seed =
      Nvfp4DecodeDotConfigFor(RootDot(*module), Sm(12, 0));
  ASSERT_TRUE(seed.has_value());
  EXPECT_TRUE(seed->swap);
  EXPECT_EQ(seed->num_warps, 4);
}

TEST_F(Nvfp4DecodeDotTest, TheSeedFollowsTheArchitecture) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           DecodeDotHlo(16, 5120, 4096)));
  std::optional<Nvfp4DecodeDotConfig> sm103 =
      Nvfp4DecodeDotConfigFor(RootDot(*module), Sm(10, 3));
  ASSERT_TRUE(sm103.has_value());
  EXPECT_TRUE(sm103->swap);
  EXPECT_EQ(sm103->weight_tile, 128);
  EXPECT_EQ(sm103->batch_tile, 16);
  EXPECT_EQ(sm103->block_k, 256);

  TF_ASSERT_OK_AND_ASSIGN(auto small, ParseAndReturnVerifiedModule(
                                          DecodeDotHlo(4, 5120, 128)));
  std::optional<Nvfp4DecodeDotConfig> seed =
      Nvfp4DecodeDotConfigFor(RootDot(*small), Sm(10, 3));
  ASSERT_TRUE(seed.has_value());
  EXPECT_EQ(seed->batch_tile, 16);
  EXPECT_EQ(seed->block_k, 128);
}

constexpr absl::string_view kClaimedFusion = R"(
    HloModule m
    nvfp4_decode_dot_d {
      p0 = f4e2m1fn[$0,4096] parameter(0)
      p1 = f4e2m1fn[$1,4096] parameter(1)
      p2 = f8e4m3fn[$0,256] parameter(2)
      p3 = f8e4m3fn[$1,256] parameter(3)
      ROOT d = bf16[$0,$1] scaled-dot(p0, p1, p2, p3),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }
    ENTRY main {
      a = f4e2m1fn[$0,4096] parameter(0)
      b = f4e2m1fn[$1,4096] parameter(1)
      c = f8e4m3fn[$0,256] parameter(2)
      e = f8e4m3fn[$1,256] parameter(3)
      ROOT f = bf16[$0,$1] fusion(a, b, c, e), kind=kCustom,
          calls=nvfp4_decode_dot_d
    })";

TEST_F(Nvfp4DecodeDotTest, TheFusionMatcherRecoversTheOrientation) {
  TF_ASSERT_OK_AND_ASSIGN(auto swapped, ParseAndReturnVerifiedModule(absl::Substitute(
                                            kClaimedFusion, 5120, 16)));
  std::optional<Nvfp4DecodeDotSpec> s = MatchNvfp4DecodeDotFusion(
      *Cast<HloFusionInstruction>(swapped->entry_computation()->root_instruction()));
  ASSERT_TRUE(s.has_value());
  EXPECT_TRUE(s->weight_on_lhs);
  EXPECT_EQ(s->weight_rows, 5120);
  EXPECT_EQ(s->batch, 16);

  TF_ASSERT_OK_AND_ASSIGN(auto plain, ParseAndReturnVerifiedModule(absl::Substitute(
                                          kClaimedFusion, 16, 5120)));
  std::optional<Nvfp4DecodeDotSpec> p = MatchNvfp4DecodeDotFusion(
      *Cast<HloFusionInstruction>(plain->entry_computation()->root_instruction()));
  ASSERT_TRUE(p.has_value());
  EXPECT_FALSE(p->weight_on_lhs);
  EXPECT_EQ(p->weight_rows, 5120);
  EXPECT_EQ(p->batch, 16);
}

}  // namespace
}  // namespace xla::gpu
