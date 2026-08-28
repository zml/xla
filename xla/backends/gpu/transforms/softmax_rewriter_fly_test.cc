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

#include "xla/backends/gpu/transforms/softmax_rewriter_fly.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/flydsl/layer_norm_support.h"
#include "xla/backends/gpu/codegen/flydsl/softmax_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;

constexpr absl::string_view kSoftmaxHlo = R"(
HloModule fly_softmax_rewriter

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
  p0 = __TYPE__[64,__COLUMNS__]{1,0} parameter(0)
  __INPUT_CONVERT__
  minus_inf = f32[] constant(-inf)
  row_max = f32[64]{0} reduce(__COMPUTE_INPUT__, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[64,__COLUMNS__]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[64,__COLUMNS__]{1,0} subtract(__COMPUTE_INPUT__, broadcast_max)
  exponential = f32[64,__COLUMNS__]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[64,__COLUMNS__]{1,0} broadcast(row_sum), dimensions={0}
  __RESULT__
})";

std::string SoftmaxHlo(absl::string_view type, int64_t columns) {
  const bool is_f32 = type == "f32";
  return absl::StrReplaceAll(
      kSoftmaxHlo,
      {{"__TYPE__", type},
       {"__COLUMNS__", std::to_string(columns)},
       {"__INPUT_CONVERT__",
        is_f32 ? "" : absl::StrCat("converted = f32[64,", columns,
                                    "]{1,0} convert(p0)")},
       {"__COMPUTE_INPUT__", is_f32 ? "p0" : "converted"},
       {"__RESULT__",
        is_f32
            ? absl::StrCat("ROOT result = f32[64,", columns,
                           "]{1,0} divide(exponential, broadcast_sum)")
            : absl::StrCat(
                  "normalized = f32[64,", columns,
                  "]{1,0} divide(exponential, broadcast_sum)\n"
                  "  ROOT result = ",
                  type, "[64,", columns, "]{1,0} convert(normalized)")}});
}

class SoftmaxRewriterFlyTest : public HloHardwareIndependentTestBase {};

TEST_F(SoftmaxRewriterFlyTest, FormsCanonicalBf16SoftmaxDirectlyForFly) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(SoftmaxHlo("bf16", 4096)));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_THAT(root->name(), HasSubstr("fly_softmax"));
  const auto* fusion = Cast<const HloFusionInstruction>(root);
  EXPECT_TRUE(flydsl::IsFlySoftmaxRoot(
      *fusion->fused_instructions_computation()->root_instruction()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config =
      gpu_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind);
  ASSERT_TRUE(fusion_config.has_block_level_fusion_config());
  const BlockLevelFusionConfig& block =
      fusion_config.block_level_fusion_config();
  ASSERT_EQ(block.output_tiles_size(), 1);
  EXPECT_THAT(block.output_tiles(0).sizes(),
              ::testing::ElementsAre(1, 4096));
  EXPECT_EQ(block.num_warps(), 4);
}

TEST_F(SoftmaxRewriterFlyTest, ChoosesValidTailConfiguration) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(SoftmaxHlo("f16", 125)));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(true));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .num_warps(),
            4);
}

TEST_F(SoftmaxRewriterFlyTest, FormsCanonicalF32Softmax) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(SoftmaxHlo("f32", 4093)));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(true));
  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(flydsl::IsFlySoftmaxRoot(
      *fusion->fused_instructions_computation()->root_instruction()));
}

TEST_F(SoftmaxRewriterFlyTest, IncreasesOccupancyForMaximumSupportedRow) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(SoftmaxHlo("bf16", 65536)));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(true));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       module->entry_computation()
                           ->root_instruction()
                           ->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .num_warps(),
            16);
}

TEST_F(SoftmaxRewriterFlyTest, FormsDependentLayerNormAsOneNativeFlyFusion) {
  constexpr absl::string_view kLayerNormHlo = R"(
HloModule fly_layer_norm_rewriter

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = bf16[256,4096]{1,0} parameter(0)
  converted = f32[256,4096]{1,0} convert(p0)
  zero = f32[] constant(0)
  sum = f32[256]{0} reduce(converted, zero), dimensions={1}, to_apply=add
  reciprocal = f32[] constant(0.000244140625)
  reciprocals = f32[256]{0} broadcast(reciprocal), dimensions={}
  mean = f32[256]{0} multiply(sum, reciprocals)
  means = f32[256,4096]{1,0} broadcast(mean), dimensions={0}
  centered = f32[256,4096]{1,0} subtract(converted, means)
  squared = f32[256,4096]{1,0} multiply(centered, centered)
  square_sum = f32[256]{0} reduce(squared, zero), dimensions={1}, to_apply=add
  variance = f32[256]{0} multiply(square_sum, reciprocals)
  epsilon = f32[] constant(1e-5)
  epsilons = f32[256]{0} broadcast(epsilon), dimensions={}
  variance_epsilon = f32[256]{0} add(variance, epsilons)
  reciprocal_stddev = f32[256]{0} rsqrt(variance_epsilon)
  scales = f32[256,4096]{1,0} broadcast(reciprocal_stddev), dimensions={0}
  normalized = f32[256,4096]{1,0} multiply(centered, scales)
  ROOT result = bf16[256,4096]{1,0} convert(normalized)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kLayerNormHlo));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(fusion->name(), HasSubstr("fly_layer_norm"));
  EXPECT_TRUE(flydsl::IsFlyLayerNormRoot(
      *fusion->fused_instructions_computation()->root_instruction()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config = gpu_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind);
  EXPECT_THAT(fusion_config.block_level_fusion_config().output_tiles(0).sizes(),
              ::testing::ElementsAre(1, 4096));
  EXPECT_EQ(fusion_config.block_level_fusion_config().num_warps(), 4);
}

TEST_F(SoftmaxRewriterFlyTest, FormsRank4DoubleStabilizedTransformerSoftmax) {
  constexpr absl::string_view kTransformerSoftmaxHlo = R"(
HloModule fly_rank4_double_stabilized_softmax

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
})";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kTransformerSoftmaxHlo));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(flydsl::IsFlySoftmaxRoot(
      *fusion->fused_instructions_computation()->root_instruction()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  const BlockLevelFusionConfig& block =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  EXPECT_THAT(block.output_tiles(0).sizes(),
              ::testing::ElementsAre(1, 1, 4, 128));
  EXPECT_EQ(block.num_warps(), 4);
}

TEST_F(SoftmaxRewriterFlyTest, DoesNotUsePartialNormalizationDiamonds) {
  constexpr absl::string_view kPartialHlo = R"(
HloModule partial_normalization

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

ENTRY main {
  p0 = f32[64,4096]{1,0} parameter(0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[64]{0} reduce(p0, minus_inf), dimensions={1}, to_apply=maximum
  broadcast_max = f32[64,4096]{1,0} broadcast(row_max), dimensions={0}
  ROOT shifted = f32[64,4096]{1,0} subtract(p0, broadcast_max)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kPartialHlo));

  EXPECT_THAT(SoftmaxRewriterFly().Run(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kSubtract);
}

}  // namespace
}  // namespace xla::gpu
