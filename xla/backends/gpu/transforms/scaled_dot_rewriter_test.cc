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
#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tests/restricted/hlo_test_base_legacy.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct ScaledDotRewriterTestCase {
  PrimitiveType operand_type;
  PrimitiveType scale_type;
};

using ScaledDotRewriterTest =
    ::testing::WithParamInterface<ScaledDotRewriterTestCase>;

class ScaledDotRewriterTestFixture : public HloTestBaseLegacy,
                                     public ScaledDotRewriterTest {
 public:
  void SetUp() override {}
};

TEST_P(ScaledDotRewriterTestFixture, ScaledDot) {
  const ScaledDotRewriterTestCase& test_case = GetParam();

  for (auto output_type :
       {PrimitiveType::F32, PrimitiveType::BF16, PrimitiveType::F16}) {
    // lhs_scale should have two dim
    const std::string hlo_string = absl::Substitute(
        R"(
        HloModule module

        ENTRY main {
          lhs = $0[1024,512] parameter(0)
          rhs = $0[64,512] parameter(1)
          lhs_scale = $1[32,2] parameter(2)
          rhs_scale = $1[64,2] parameter(3)
          ROOT dot = $2[1024,64] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
            lhs_contracting_dims={1},
            rhs_contracting_dims={1}
        }
      )",
        absl::AsciiStrToLower(PrimitiveType_Name(test_case.operand_type)),
        absl::AsciiStrToLower(PrimitiveType_Name(test_case.scale_type)),
        absl::AsciiStrToLower(PrimitiveType_Name(output_type)));
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnUnverifiedModule(hlo_string));

    ScaledDotRewriter rewriter;
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);

    // Verify that the module is still valid after the rewrite.
    auto status_or_module = ParseAndReturnVerifiedModule(module->ToString());
    EXPECT_TRUE(status_or_module.status().ok()) << status_or_module.status();

    const HloInstruction* root =
        module->entry_computation()->root_instruction();

    if (output_type == PrimitiveType::F16) {
      EXPECT_EQ(root->opcode(), HloOpcode::kConvert);
      root = root->operand(0);
    }

    EXPECT_EQ(root->opcode(), HloOpcode::kDot);
    for (const HloInstruction* operand : root->operands()) {
      std::vector<HloOpcode> actual_op_codes{};
      while (operand->opcode() != HloOpcode::kParameter) {
        actual_op_codes.push_back(operand->opcode());
        if (operand->opcode() == HloOpcode::kMultiply) {
          operand = operand->operand(1);
        } else {
          operand = operand->operand(0);
        }
      }
      actual_op_codes = std::vector<HloOpcode>(actual_op_codes.rbegin(),
                                               actual_op_codes.rend());

      if (test_case.scale_type == PrimitiveType::BF16) {
        const std::vector<HloOpcode> expected_op_codes{
            HloOpcode::kBroadcast, HloOpcode::kReshape, HloOpcode::kMultiply};
        EXPECT_THAT(actual_op_codes, expected_op_codes);
      } else {
        const std::vector<HloOpcode> expected_op_codes_with_convert{
            HloOpcode::kConvert, HloOpcode::kBroadcast, HloOpcode::kReshape,
            HloOpcode::kMultiply};
        EXPECT_THAT(actual_op_codes, expected_op_codes_with_convert);
      }
    }

    EXPECT_TRUE(RunAndCompare(std::move(module),
                              ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
  }
}

using ScaledDotRewriterElementSizeTest = HloHardwareIndependentTestBase;

class MetalScaledDotRewriterTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunMetalRewrite(HloModule* module) {
    ScaledDotRewriter rewriter{
        se::GpuComputeCapability(se::MetalComputeCapability("Apple9"))};
    return rewriter.Run(module);
  }

  void ExpectMetalScaledMatmul(absl::string_view hlo_string,
                               bool expect_fused,
                               bool expect_lhs_scale_multiply = false) {
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnUnverifiedModule(hlo_string));
    ASSERT_OK_AND_ASSIGN(bool changed, RunMetalRewrite(module.get()));
    EXPECT_TRUE(changed);

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

TEST_F(MetalScaledDotRewriterTest, ScalarOneConstantIsIdentityScale) {
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

TEST_F(MetalScaledDotRewriterTest, ScalarNonOneConstantIsNotIdentityScale) {
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

TEST_F(MetalScaledDotRewriterTest, ScalarParameterIsNotIdentityScale) {
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

TEST_F(MetalScaledDotRewriterTest, DenseAllOnesConstantIsIdentityScale) {
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

TEST_F(MetalScaledDotRewriterTest, DenseNonOneConstantIsNotIdentityScale) {
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

TEST_F(MetalScaledDotRewriterTest, BroadcastOneIsNotFoldedAsConstant) {
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

TEST_F(MetalScaledDotRewriterTest, FlattensRankOneLhsLikeMlx) {
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

TEST_F(MetalScaledDotRewriterTest, FlattensRankThreeLhsAndConstrainsLayout) {
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

TEST_F(MetalScaledDotRewriterTest, FlattensRankFourLhsAndRestoresShape) {
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

TEST_F(MetalScaledDotRewriterTest, TrueBatchDimensionsFallBack) {
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

TEST_F(MetalScaledDotRewriterTest, NonLastLhsContractingDimensionFallsBack) {
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

TEST_F(MetalScaledDotRewriterTest, RankThreeWeightsFallBack) {
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

TEST_F(MetalScaledDotRewriterTest, NonMinorRhsContractingDimensionFallsBack) {
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

TEST_F(MetalScaledDotRewriterTest, F32ResultFallsBack) {
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

TEST_F(MetalScaledDotRewriterTest, MismatchedResultShapeFallsBack) {
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
  EXPECT_TRUE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalScaledDotRewriterTest, DynamicResultShapeFallsBack) {
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
  EXPECT_TRUE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalScaledDotRewriterTest, RankTwoInt32OverflowFallsBack) {
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
  EXPECT_TRUE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalScaledDotRewriterTest, FlattenedRowProductOverflowFallsBack) {
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
  EXPECT_TRUE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalScaledDotRewriterTest, DynamicRankTwoLhsFallsBack) {
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
  EXPECT_TRUE(changed);
  EXPECT_EQ(FindMetalScaledMatmul(module.get()), nullptr);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kDot);
}

TEST_F(MetalScaledDotRewriterTest, FusedCallConstrainsRowMajorLayouts) {
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

// Upcasting an fp4 operand (whose layout carries element_size_in_bits=4) to
// BF16 must drop the custom sub-byte element size; otherwise the rewriter emits
// an illegal bf16[...]{:E(4)} shape that the CpuGpuShapeVerifier rejects.
TEST_F(ScaledDotRewriterElementSizeTest, Fp4OperandDropsSubByteElementSize) {
  const std::string hlo_string = R"(
    HloModule module

    ENTRY main {
      lhs = f4e2m1fn[1024,512]{1,0:E(4)} parameter(0)
      rhs = f8e4m3fn[64,512] parameter(1)
      lhs_scale = f8e8m0fnu[1024,16] parameter(2)
      rhs_scale = f8e8m0fnu[64,16] parameter(3)
      ROOT dot = f32[1024,64] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_string));

  ScaledDotRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      const Shape& shape = instruction->shape();
      if (!shape.IsArray() || !shape.has_layout()) {
        continue;
      }
      if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
        continue;
      }
      EXPECT_EQ(shape.layout().element_size_in_bits(), 0)
          << "Non-sub-byte instruction retains custom element size: "
          << instruction->ToString();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ScaledDotRewriterTests, ScaledDotRewriterTestFixture,
    ::testing::ValuesIn<ScaledDotRewriterTestCase>({
        {PrimitiveType::F8E4M3FN, PrimitiveType::F8E8M0FNU},
        {PrimitiveType::F8E5M2, PrimitiveType::F8E8M0FNU},
        {PrimitiveType::BF16, PrimitiveType::BF16},
        {PrimitiveType::S4, PrimitiveType::BF16},
    }),
    [](const ::testing::TestParamInfo<ScaledDotRewriterTest::ParamType>& info) {
      return absl::StrCat(PrimitiveType_Name(info.param.operand_type), "_",
                          PrimitiveType_Name(info.param.scale_type));
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
