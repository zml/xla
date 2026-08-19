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
#include "xla/backends/gpu/transforms/fused_scaled_dot_arms_metal.h"

#include "xla/backends/gpu/transforms/fused_scaled_dot_rewriter.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/metal_custom_calls.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class MetalFusedScaledDotRewriterTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunMetalRewrite(HloModule* module) {
    FusedScaledDotRewriter fused_rewriter(MetalFusedScaledDotArms());
    TF_ASSIGN_OR_RETURN(bool fused, fused_rewriter.Run(module));
    ScaledDotRewriter generic_rewriter;
    TF_RETURN_IF_ERROR(generic_rewriter.Run(module).status());
    return fused;
  }

  void ExpectMetalScaledMatmul(absl::string_view hlo_string,
                               bool expect_fused,
                               bool expect_lhs_scale_multiply = false) {
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnUnverifiedModule(hlo_string));
    ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
    EXPECT_EQ(changed, expect_fused);

    auto verified = ParseAndReturnVerifiedModule(module->ToString());
    ASSERT_TRUE(verified.ok()) << verified.status();

    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    if (expect_fused) {
      ASSERT_EQ(root->opcode(), HloOpcode::kCustomCall);
      EXPECT_EQ(root->custom_call_target(), "zml$scaled_matmul");
      EXPECT_EQ(root->operand_count(), 3);
    } else {
      ASSERT_EQ(root->opcode(), HloOpcode::kDot);
      if (expect_lhs_scale_multiply) {
        ASSERT_EQ(root->operand(0)->opcode(), HloOpcode::kMultiply);
        EXPECT_EQ(root->operand(0)->operand(1)->opcode(),
                  HloOpcode::kBroadcast);
      }
    }
  }

  static HloCustomCallInstruction* FindMetalScaledMatmul(HloModule* module) {
    for (HloInstruction* instruction :
         module->entry_computation()->instructions()) {
      auto* call = DynCast<HloCustomCallInstruction>(instruction);
      if (call != nullptr &&
          call->custom_call_target() == "zml$scaled_matmul") {
        return call;
      }
    }
    return nullptr;
  }
};

TEST_F(MetalFusedScaledDotRewriterTest, ScalarOneConstantIsIdentityScale) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[] constant(1)
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/true);
}

TEST_F(MetalFusedScaledDotRewriterTest, ScalarNonOneConstantIsNotIdentityScale) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[] constant(2)
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/false, /*expect_lhs_scale_multiply=*/true);
}

TEST_F(MetalFusedScaledDotRewriterTest, ScalarParameterIsNotIdentityScale) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[] parameter(2)
      rhs_scale = f8e4m3fn[8,2] parameter(3)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/false, /*expect_lhs_scale_multiply=*/true);
}

TEST_F(MetalFusedScaledDotRewriterTest, DenseAllOnesConstantIsIdentityScale) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/true);
}

TEST(MetalScaledMatmulSchemeTest, ClassifiesTheThreeImplementedSchemes) {
  EXPECT_EQ(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F4E2M1FN, {8, 32}),
                                      ShapeUtil::MakeShape(F8E4M3FN, {8, 2})),
            MetalScaledMatmulScheme::kNvfp4Group16);
  EXPECT_EQ(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {8, 32}),
                                      ShapeUtil::MakeShape(BF16, {8, 1})),
            MetalScaledMatmulScheme::kFp8PerChannel);
  EXPECT_EQ(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {256, 256}),
                                      ShapeUtil::MakeShape(BF16, {2, 2})),
            MetalScaledMatmulScheme::kFp8Block128);
}

TEST(MetalScaledMatmulSchemeTest, ClassifiesPerTensorAndF32Scales) {
  EXPECT_EQ(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {8, 32}),
                                      ShapeUtil::MakeShape(F32, {1, 1})),
            MetalScaledMatmulScheme::kFp8PerChannel);
  EXPECT_EQ(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {8, 32}),
                                      ShapeUtil::MakeShape(BF16, {1, 1})),
            MetalScaledMatmulScheme::kFp8PerChannel);
  EXPECT_EQ(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {8, 32}),
                                      ShapeUtil::MakeShape(F32, {8, 1})),
            MetalScaledMatmulScheme::kFp8PerChannel);
  EXPECT_FALSE(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {8, 24}),
                                         ShapeUtil::MakeShape(F32, {1, 1}))
                   .has_value());
}

TEST(MetalScaledMatmulSchemeTest, BlockOneTwentyEightWinsTheAmbiguousShapes) {
  EXPECT_EQ(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {128, 128}),
                                      ShapeUtil::MakeShape(BF16, {1, 1})),
            MetalScaledMatmulScheme::kFp8Block128);
  EXPECT_EQ(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {1, 32}),
                                      ShapeUtil::MakeShape(BF16, {1, 1})),
            MetalScaledMatmulScheme::kFp8PerChannel);
}

TEST(MetalScaledMatmulSchemeTest, BlockOneTwentyEightStaysBf16Only) {
  EXPECT_FALSE(
      ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {256, 256}),
                                ShapeUtil::MakeShape(F32, {2, 2}))
          .has_value());
}

TEST(MetalScaledMatmulSchemeTest, E8m0ScalesAreNeverClassified) {
  EXPECT_FALSE(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F4E2M1FN, {8, 32}),
                                         ShapeUtil::MakeShape(F8E8M0FNU, {8, 2}))
                   .has_value());
  EXPECT_FALSE(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {8, 32}),
                                         ShapeUtil::MakeShape(F8E8M0FNU, {8, 1}))
                   .has_value());
  EXPECT_FALSE(
      ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {256, 256}),
                                ShapeUtil::MakeShape(F8E8M0FNU, {2, 2}))
          .has_value());
}

TEST(MetalScaledMatmulSchemeTest, RejectsDegenerateAndOffGridShapes) {
  EXPECT_FALSE(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F4E2M1FN, {8, 0}),
                                         ShapeUtil::MakeShape(F8E4M3FN, {8, 0}))
                   .has_value());
  EXPECT_FALSE(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F4E2M1FN, {8, 24}),
                                         ShapeUtil::MakeShape(F8E4M3FN, {8, 2}))
                   .has_value());
  EXPECT_FALSE(
      ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F4E2M1FN, {2, 8, 32}),
                                ShapeUtil::MakeShape(F8E4M3FN, {2, 8, 2}))
          .has_value());
  EXPECT_FALSE(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(BF16, {8, 32}),
                                         ShapeUtil::MakeShape(BF16, {8, 1}))
                   .has_value());
}

TEST(MetalScaledMatmulSchemeTest, RejectsPerChannelWithUnalignedContraction) {
  EXPECT_EQ(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {8, 64}),
                                      ShapeUtil::MakeShape(BF16, {8, 1})),
            MetalScaledMatmulScheme::kFp8PerChannel);
  EXPECT_FALSE(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {8, 20}),
                                         ShapeUtil::MakeShape(BF16, {8, 1}))
                   .has_value());
  EXPECT_FALSE(ClassifyMetalScaledMatmul(ShapeUtil::MakeShape(F8E4M3FN, {8, 30}),
                                         ShapeUtil::MakeShape(BF16, {8, 1}))
                   .has_value());
}

TEST_F(MetalFusedScaledDotRewriterTest, Fp8Block128IsFused) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,256] parameter(0)
      rhs = f8e4m3fn[256,256] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = bf16[2,2] parameter(2)
      ROOT dot = bf16[2,256] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/true);
}

TEST_F(MetalFusedScaledDotRewriterTest, Fp8PerChannelIsFused) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f8e4m3fn[8,32] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = bf16[8,1] parameter(2)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/true);
}

TEST_F(MetalFusedScaledDotRewriterTest, Fp8UnrecognizedScaleGridIsNotFused) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f8e4m3fn[8,32] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = bf16[8,2] parameter(2)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, MxGroup32HasNoThunkAndIsNotFused) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,64] parameter(0)
      rhs = f8e4m3fn[8,64] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = f8e8m0fnu[8,2] parameter(2)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, MxGroup32Fp4HasNoThunkAndIsNotFused) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,64] parameter(0)
      rhs = f4e2m1fn[8,64] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = f8e8m0fnu[8,2] parameter(2)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, DenseNonOneConstantIsNotIdentityScale) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[1,1] constant({{2}})
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, BroadcastOneIsNotFoldedAsConstant) {
  ExpectMetalScaledMatmul(R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      one = bf16[] constant(1)
      lhs_scale = bf16[1,1] broadcast(one), dimensions={}
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = bf16[2,8] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, FlattensRankOneLhsLikeMlx) {
  const std::string hlo_string = R"(
    HloModule rank_one

    ENTRY main {
      lhs = bf16[32]{0} parameter(0)
      rhs = f4e2m1fn[8,32]{1,0:E(4)} parameter(1)
      lhs_scale = bf16[] constant(1)
      rhs_scale = f8e4m3fn[8,2]{1,0} parameter(2)
      ROOT dot = bf16[8]{0}
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={0}, rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
  ASSERT_TRUE(changed);

  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kReshape);
  EXPECT_THAT(root->shape().dimensions(), ::testing::ElementsAre(8));

  HloCustomCallInstruction* call = FindMetalScaledMatmul(module.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(root->operand(0), call);
  EXPECT_THAT(call->shape().dimensions(), ::testing::ElementsAre(1, 8));
  ASSERT_EQ(call->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_THAT(call->operand(0)->shape().dimensions(),
              ::testing::ElementsAre(1, 32));
  ASSERT_TRUE(call->layout_constrained());
  EXPECT_THAT(call->shape().layout().minor_to_major(),
              ::testing::ElementsAre(1, 0));
  EXPECT_EQ(call->operand_shapes_with_layout()[1]
                .layout()
                .element_size_in_bits(),
            4);

  auto verified = ParseAndReturnVerifiedModule(module->ToString());
  ASSERT_TRUE(verified.ok()) << verified.status();
}

TEST_F(MetalFusedScaledDotRewriterTest, FlattensRankThreeLhsAndConstrainsLayout) {
  const std::string hlo_string = R"(
    HloModule rank_three

    ENTRY main {
      lhs = bf16[2,3,32]{0,2,1} parameter(0)
      rhs = f4e2m1fn[8,32]{0,1:E(4)} parameter(1)
      lhs_scale = bf16[1,1,1]{2,1,0} constant({{{1}}})
      rhs_scale = f8e4m3fn[8,2]{0,1} parameter(2)
      ROOT dot = bf16[2,3,8]{2,1,0}
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
  ASSERT_TRUE(changed);

  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kReshape);
  EXPECT_THAT(root->shape().dimensions(), ::testing::ElementsAre(2, 3, 8));
  EXPECT_THAT(root->shape().layout().minor_to_major(),
              ::testing::ElementsAre(2, 1, 0));

  HloCustomCallInstruction* call = FindMetalScaledMatmul(module.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(root->operand(0), call);
  EXPECT_THAT(call->shape().dimensions(), ::testing::ElementsAre(6, 8));
  ASSERT_EQ(call->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_THAT(call->operand(0)->shape().dimensions(),
              ::testing::ElementsAre(6, 32));
  ASSERT_TRUE(call->layout_constrained());
  EXPECT_THAT(call->shape().layout().minor_to_major(),
              ::testing::ElementsAre(1, 0));
  ASSERT_EQ(call->operand_shapes_with_layout().size(), 3);
  for (const Shape& shape : call->operand_shapes_with_layout()) {
    EXPECT_THAT(shape.layout().minor_to_major(),
                ::testing::ElementsAre(1, 0));
  }
  EXPECT_EQ(call->operand_shapes_with_layout()[1]
                .layout()
                .element_size_in_bits(),
            4);

  auto verified = ParseAndReturnVerifiedModule(module->ToString());
  ASSERT_TRUE(verified.ok()) << verified.status();
}

TEST_F(MetalFusedScaledDotRewriterTest, FlattensRankFourLhsAndRestoresShape) {
  const std::string hlo_string = R"(
    HloModule rank_four

    ENTRY main {
      lhs = bf16[2,3,5,32]{3,2,1,0} parameter(0)
      rhs = f4e2m1fn[8,32]{1,0:E(4)} parameter(1)
      lhs_scale = bf16[1,1,1,1]{3,2,1,0} constant({{{{1}}}})
      rhs_scale = f8e4m3fn[8,2]{1,0} parameter(2)
      ROOT dot = bf16[2,3,5,8]{3,2,1,0}
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={3}, rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
  ASSERT_TRUE(changed);

  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kReshape);
  EXPECT_THAT(root->shape().dimensions(),
              ::testing::ElementsAre(2, 3, 5, 8));
  HloCustomCallInstruction* call = FindMetalScaledMatmul(module.get());
  ASSERT_NE(call, nullptr);
  EXPECT_THAT(call->shape().dimensions(), ::testing::ElementsAre(30, 8));
  EXPECT_THAT(call->operand(0)->shape().dimensions(),
              ::testing::ElementsAre(30, 32));

  auto verified = ParseAndReturnVerifiedModule(module->ToString());
  ASSERT_TRUE(verified.ok()) << verified.status();
}

TEST_F(MetalFusedScaledDotRewriterTest, TrueBatchDimensionsFallBack) {
  ExpectMetalScaledMatmul(R"(
    HloModule true_batch

    ENTRY main {
      lhs = bf16[2,3,32] parameter(0)
      rhs = f4e2m1fn[2,8,32] parameter(1)
      lhs_scale = bf16[1,1,1] constant({{{1}}})
      rhs_scale = f8e4m3fn[2,8,2] parameter(2)
      ROOT dot = bf16[2,3,8]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, NonLastLhsContractingDimensionFallsBack) {
  ExpectMetalScaledMatmul(R"(
    HloModule non_last_contracting

    ENTRY main {
      lhs = bf16[32,2,3] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[1,1,1] constant({{{1}}})
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = bf16[2,3,8]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={0}, rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, RankThreeWeightsFallBack) {
  ExpectMetalScaledMatmul(R"(
    HloModule rank_three_weights

    ENTRY main {
      lhs = bf16[2,3,32] parameter(0)
      rhs = f4e2m1fn[4,8,32] parameter(1)
      lhs_scale = bf16[1,1,1] constant({{{1}}})
      rhs_scale = f8e4m3fn[4,8,2] parameter(2)
      ROOT dot = bf16[2,3,4,8]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, NonMinorRhsContractingDimensionFallsBack) {
  ExpectMetalScaledMatmul(R"(
    HloModule non_minor_rhs_contracting

    ENTRY main {
      lhs = bf16[2,3,32] parameter(0)
      rhs = f4e2m1fn[32,8] parameter(1)
      lhs_scale = bf16[1,1,1] constant({{{1}}})
      rhs_scale = f8e4m3fn[2,8] parameter(2)
      ROOT dot = bf16[2,3,8]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={2}, rhs_contracting_dims={0}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, F32ResultFallsBack) {
  ExpectMetalScaledMatmul(R"(
    HloModule f32_result

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = f32[2,8]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }
  )", /*expect_fused=*/false);
}

TEST_F(MetalFusedScaledDotRewriterTest, MismatchedResultShapeFallsBack) {
  const std::string hlo_string = R"(
    HloModule mismatched_result

    ENTRY main {
      lhs = bf16[2,3,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[1,1,1] constant({{{1}}})
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = bf16[2,3,7]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalFusedScaledDotRewriterTest, DynamicResultShapeFallsBack) {
  const std::string hlo_string = R"(
    HloModule dynamic_result

    ENTRY main {
      lhs = bf16[2,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = bf16[<=2,8]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalFusedScaledDotRewriterTest, RankTwoInt32OverflowFallsBack) {
  const std::string hlo_string = R"(
    HloModule int32_overflow

    ENTRY main {
      lhs = bf16[2147483648,16] parameter(0)
      rhs = f4e2m1fn[8,16] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = f8e4m3fn[8,1] parameter(2)
      ROOT dot = bf16[2147483648,8]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalFusedScaledDotRewriterTest, FlattenedRowProductOverflowFallsBack) {
  const std::string hlo_string = R"(
    HloModule flattened_product_overflow

    ENTRY main {
      lhs = bf16[46341,46341,16] parameter(0)
      rhs = f4e2m1fn[8,16] parameter(1)
      lhs_scale = bf16[1,1,1] constant({{{1}}})
      rhs_scale = f8e4m3fn[8,1] parameter(2)
      ROOT dot = bf16[46341,46341,8]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalFusedScaledDotRewriterTest, DynamicRankTwoLhsFallsBack) {
  const std::string hlo_string = R"(
    HloModule dynamic_lhs

    ENTRY main {
      lhs = bf16[<=2,32] parameter(0)
      rhs = f4e2m1fn[8,32] parameter(1)
      lhs_scale = bf16[1,1] constant({{1}})
      rhs_scale = f8e4m3fn[8,2] parameter(2)
      ROOT dot = bf16[<=2,8]
        scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalFusedScaledDotRewriterTest, FusedCallConstrainsRowMajorLayouts) {
  const std::string hlo_string = R"(
    HloModule module

    ENTRY main {
      lhs = bf16[2,32]{0,1} parameter(0)
      rhs = f4e2m1fn[8,32]{0,1:E(4)} parameter(1)
      lhs_scale = bf16[] constant(1)
      rhs_scale = f8e4m3fn[8,2]{0,1} parameter(2)
      ROOT dot = bf16[2,8]{0,1} scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
  ASSERT_TRUE(changed);

  const auto* custom_call = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());
  ASSERT_TRUE(custom_call->layout_constrained());
  EXPECT_THAT(custom_call->shape().layout().minor_to_major(),
              ::testing::ElementsAre(1, 0));
  ASSERT_EQ(custom_call->operand_shapes_with_layout().size(), 3);
  for (const Shape& shape : custom_call->operand_shapes_with_layout()) {
    EXPECT_THAT(shape.layout().minor_to_major(),
                ::testing::ElementsAre(1, 0));
  }
  EXPECT_EQ(custom_call->operand_shapes_with_layout()[1]
                .layout()
                .element_size_in_bits(),
            4);

  auto verified = ParseAndReturnVerifiedModule(module->ToString());
  ASSERT_TRUE(verified.ok()) << verified.status();
}

}  // namespace
}  // namespace gpu
}  // namespace xla
