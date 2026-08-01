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

#include "xla/backends/gpu/transforms/metal_workspace_rewriter.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "xla/backends/gpu/runtime/metal_workspace.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class MetalWorkspaceRewriterTest : public HloHardwareIndependentTestBase {
 protected:
  // A path that owns scratch is wrapped in tuple(result, s8[bytes]) behind a
  // GTE(0); a path that owns none keeps its plain array result.
  void ExpectWorkspace(std::unique_ptr<HloModule> module,
                       int64_t expected_bytes,
                       char arch_size = kNvfp4DefaultArchSize,
                       int arch_gen = kNvfp4DefaultArchGen) {
    MetalWorkspaceRewriter pass(arch_size, arch_gen);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    EXPECT_TRUE(changed);

    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    ASSERT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_EQ(root->tuple_index(), 0);
    const auto* call = Cast<HloCustomCallInstruction>(root->operand(0));
    ASSERT_TRUE(call->shape().IsTuple());
    ASSERT_EQ(call->shape().tuple_shapes().size(), 2);
    EXPECT_EQ(call->shape().tuple_shapes(0), root->shape());
    const Shape& workspace = call->shape().tuple_shapes(1);
    EXPECT_EQ(workspace.element_type(), S8);
    ASSERT_EQ(workspace.dimensions().size(), 1);
    EXPECT_EQ(workspace.dimensions(0), expected_bytes);

    // The original array result remains the computation ABI, and a second pass
    // run must not wrap the already-tuple call again.
    EXPECT_TRUE(
        module->entry_computation()->root_instruction()->shape().IsArray());
    TF_ASSERT_OK_AND_ASSIGN(bool changed_again,
                            RunHloPass(&pass, module.get()));
    EXPECT_FALSE(changed_again);
  }

  void ExpectNoWorkspace(std::unique_ptr<HloModule> module,
                         char arch_size = kNvfp4DefaultArchSize,
                         int arch_gen = kNvfp4DefaultArchGen) {
    MetalWorkspaceRewriter pass(arch_size, arch_gen);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    EXPECT_FALSE(changed);
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
    EXPECT_TRUE(root->shape().IsArray());
  }
};

TEST_F(MetalWorkspaceRewriterTest, DenseSplitKGetsExactWorkspace) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule dense_splitk

    ENTRY main {
      x = bf16[16,2816]{1,0} parameter(0)
      w = f4e2m1fn[2816,2816]{1,0:E(4)} parameter(1)
      scale = f8e4m3fn[2816,176]{1,0} parameter(2)
      ROOT call = bf16[16,2816]{1,0} custom-call(x, w, scale),
        custom_call_target="zml$scaled_matmul"
    }
  )"));
  // M=16 selects split_k=4, staging bf16[4,16,2816].
  ExpectWorkspace(std::move(module), 4 * 16 * 2816 * 2);
}

TEST_F(MetalWorkspaceRewriterTest, DenseNonSplitIsNotWrapped) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule dense_qmv

    ENTRY main {
      x = bf16[1,2816]{1,0} parameter(0)
      w = f4e2m1fn[2816,2816]{1,0:E(4)} parameter(1)
      scale = f8e4m3fn[2816,176]{1,0} parameter(2)
      ROOT call = bf16[1,2816]{1,0} custom-call(x, w, scale),
        custom_call_target="zml$scaled_matmul"
    }
  )"));
  ExpectNoWorkspace(std::move(module));
}

TEST_F(MetalWorkspaceRewriterTest, RejectsDenseDimensionOutsideKernelInt32) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule dense_dimension_overflow

    ENTRY main {
      x = bf16[1,2147483648]{1,0} parameter(0)
      w = f4e2m1fn[1,2147483648]{1,0:E(4)} parameter(1)
      scale = f8e4m3fn[1,134217728]{1,0} parameter(2)
      ROOT call = bf16[1,1]{1,0} custom-call(x, w, scale),
        custom_call_target="zml$scaled_matmul"
    }
  )"));
  MetalWorkspaceRewriter pass(kNvfp4DefaultArchSize,
                              kNvfp4DefaultArchGen);
  auto result = RunHloPass(&pass, module.get());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(MetalWorkspaceRewriterTest, RejectsMoeDimensionOutsideKernelInt32) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule moe_dimension_overflow

    ENTRY main {
      x = bf16[2147483648,16]{1,0} parameter(0)
      w = f4e2m1fn[1,1,16]{2,1,0:E(4)} parameter(1)
      scale = f8e4m3fn[1,1,1]{2,1,0} parameter(2)
      expert = s32[2147483648]{0} parameter(3)
      ROOT call = bf16[2147483648,1]{1,0} custom-call(x, w, scale, expert),
        custom_call_target="__metal$moe_gemm$f4"
    }
  )"));
  MetalWorkspaceRewriter pass(kNvfp4DefaultArchSize,
                              kNvfp4DefaultArchGen);
  auto result = RunHloPass(&pass, module.get());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(MetalWorkspaceRewriterTest,
       DenseWorkspaceUsesCompileTargetArchitecture) {
  constexpr char kHlo[] = R"(
    HloModule dense_arch_boundary

    ENTRY main {
      x = bf16[20,2048]{1,0} parameter(0)
      w = f4e2m1fn[2048,2048]{1,0:E(4)} parameter(1)
      scale = f8e4m3fn[2048,128]{1,0} parameter(2)
      ROOT call = bf16[20,2048]{1,0} custom-call(x, w, scale),
        custom_call_target="zml$scaled_matmul"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto fallback, ParseAndReturnUnverifiedModule(kHlo));
  // Fallback limit 18 -> split_k=8 -> staging bf16[8,20,2048].
  ExpectWorkspace(std::move(fallback), 8 * 20 * 2048 * 2);

  TF_ASSERT_OK_AND_ASSIGN(auto gen14_ultra,
                          ParseAndReturnUnverifiedModule(kHlo));
  // Gen-14 Ultra limit 32 -> QMV-wide, which owns no scratch.
  ExpectNoWorkspace(std::move(gen14_ultra), 'd', 14);
}

TEST_F(MetalWorkspaceRewriterTest, OtherScaledMatmulSchemeIsNotWrapped) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule dense_fp8

    ENTRY main {
      x = bf16[1,128]{1,0} parameter(0)
      w = f8e4m3fn[128,128]{1,0} parameter(1)
      scale = bf16[1,1]{1,0} parameter(2)
      ROOT call = bf16[1,128]{1,0} custom-call(x, w, scale),
        custom_call_target="zml$scaled_matmul"
    }
  )"));
  MetalWorkspaceRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kCustomCall);
}

TEST_F(MetalWorkspaceRewriterTest, Nvfp4SortedMoeGetsExactWorkspace) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule nvfp4_moe

    ENTRY main {
      x = bf16[512,2816]{1,0} parameter(0)
      w = f4e2m1fn[128,704,2816]{2,1,0:E(4)} parameter(1)
      scale = f8e4m3fn[128,704,176]{2,1,0} parameter(2)
      expert = s32[512]{0} parameter(3)
      ROOT call = bf16[512,704]{1,0}
        custom-call(x, w, scale, expert),
        custom_call_target="__metal$moe_gemm$f4"
    }
  )"));
  // order + expert ids + sorted x + sorted output, each 16-byte aligned.
  ExpectWorkspace(std::move(module), 3608576);
}

TEST_F(MetalWorkspaceRewriterTest, Nvfp4DecodeMoeIsNotWrapped) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule nvfp4_moe_decode

    ENTRY main {
      x = bf16[128,2816]{1,0} parameter(0)
      w = f4e2m1fn[128,704,2816]{2,1,0:E(4)} parameter(1)
      scale = f8e4m3fn[128,704,176]{2,1,0} parameter(2)
      expert = s32[128]{0} parameter(3)
      ROOT call = bf16[128,704]{1,0}
        custom-call(x, w, scale, expert),
        custom_call_target="__metal$moe_gemm$f4"
    }
  )"));
  // R/E = 1 misses the MLX reuse gate: per-row GEMV owns no scratch.
  ExpectNoWorkspace(std::move(module));
}

TEST_F(MetalWorkspaceRewriterTest, Bf16MoeUsesSharedLegacySortedGate) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule bf16_moe

    ENTRY main {
      x = bf16[1024,32]{1,0} parameter(0)
      w = bf16[8,32,32]{2,1,0} parameter(1)
      expert = s32[1024]{0} parameter(2)
      ROOT call = bf16[1024,32]{1,0} custom-call(x, w, expert),
        custom_call_target="__metal$moe_gemm"
    }
  )"));
  ExpectWorkspace(std::move(module), 139264);
}

TEST_F(MetalWorkspaceRewriterTest,
       ExistingTupleAliasIsClearedAndThenIdempotent) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule already_rewritten

    ENTRY main {
      x = bf16[1,32]{1,0} parameter(0)
      w = f4e2m1fn[8,32]{1,0:E(4)} parameter(1)
      scale = f8e4m3fn[8,2]{1,0} parameter(2)
      call = (bf16[1,8]{1,0}, s8[0]{0})
        custom-call(x, w, scale),
        custom_call_target="zml$scaled_matmul",
        output_to_operand_aliasing={{0}: (0, {})}
      ROOT result = bf16[1,8]{1,0} get-tuple-element(call), index=0
    }
  )"));
  MetalWorkspaceRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const auto* call = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction()->operand(0));
  EXPECT_TRUE(call->output_operand_aliasing().empty());

  TF_ASSERT_OK_AND_ASSIGN(bool changed_again, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed_again);
}

TEST_F(MetalWorkspaceRewriterTest, ArrayResultAliasIsClearedDuringRewrite) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule aliased_moe

    ENTRY main {
      x = bf16[1024,32]{1,0} parameter(0)
      w = bf16[8,32,32]{2,1,0} parameter(1)
      expert = s32[1024]{0} parameter(2)
      ROOT call = bf16[1024,32]{1,0}
        custom-call(x, w, expert),
        custom_call_target="__metal$moe_gemm",
        output_to_operand_aliasing={{}: (0, {})}
    }
  )"));
  MetalWorkspaceRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  ASSERT_TRUE(changed);
  const auto* call = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction()->operand(0));
  EXPECT_TRUE(call->output_operand_aliasing().empty());
}

TEST_F(MetalWorkspaceRewriterTest,
       NonWorkspaceScaledMatmulAliasIsClearedWithoutWrapping) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule aliased_fp8

    ENTRY main {
      x = bf16[1,128]{1,0} parameter(0)
      w = f8e4m3fn[128,128]{1,0} parameter(1)
      scale = bf16[1,1]{1,0} parameter(2)
      ROOT call = bf16[1,128]{1,0} custom-call(x, w, scale),
        custom_call_target="zml$scaled_matmul",
        output_to_operand_aliasing={{}: (0, {})}
    }
  )"));
  MetalWorkspaceRewriter pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const auto* call = Cast<HloCustomCallInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(call->shape().IsArray());
  EXPECT_TRUE(call->output_operand_aliasing().empty());

  TF_ASSERT_OK_AND_ASSIGN(bool changed_again, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed_again);
}

TEST(MetalWorkspaceLayoutTest, MoeRegionsAreAlignedAndExact) {
  TF_ASSERT_OK_AND_ASSIGN(MetalMoeWorkspaceLayout layout,
                          GetMetalMoeWorkspaceLayout(3, 5, 7));
  EXPECT_EQ(layout.order.offset % kMetalWorkspaceAlignment, 0);
  EXPECT_EQ(layout.expert_ids.offset % kMetalWorkspaceAlignment, 0);
  EXPECT_EQ(layout.x_sorted.offset % kMetalWorkspaceAlignment, 0);
  EXPECT_EQ(layout.out_sorted.offset % kMetalWorkspaceAlignment, 0);
  EXPECT_EQ(layout.order.size, 12);
  EXPECT_EQ(layout.expert_ids.size, 12);
  EXPECT_EQ(layout.x_sorted.size, 30);
  EXPECT_EQ(layout.out_sorted.size, 42);
  EXPECT_EQ(layout.total_bytes, 106);
}

TEST(MetalWorkspaceLayoutTest, RejectsOverflow) {
  auto moe =
      GetMetalMoeWorkspaceLayout(std::numeric_limits<int64_t>::max() / 2, 4, 4);
  EXPECT_EQ(moe.status().code(), absl::StatusCode::kResourceExhausted);
  auto split = GetMetalNvfp4SplitKWorkspaceLayout(
      std::numeric_limits<int64_t>::max(), 2, 2);
  EXPECT_EQ(split.status().code(), absl::StatusCode::kResourceExhausted);
}

TEST(MetalWorkspaceLayoutTest, RejectsSplitKStrideOutsideMetalInt32) {
  // Every dimension fits int32, but one partial plane's M*N stride does not
  // fit the split-K kernel's signed int ABI.
  auto workspace = GetMetalNvfp4SplitKWorkspaceLayout(2, 1 << 20, 1 << 20);
  EXPECT_EQ(workspace.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(MetalWorkspaceLayoutTest, SharedBytePlannersIncludePathSelection) {
  TF_ASSERT_OK_AND_ASSIGN(int64_t split,
                          GetMetalNvfp4WorkspaceBytes(16, 2816, 2816));
  EXPECT_EQ(split, 4 * 16 * 2816 * 2);
  TF_ASSERT_OK_AND_ASSIGN(int64_t qmv,
                          GetMetalNvfp4WorkspaceBytes(1, 2816, 2816));
  EXPECT_EQ(qmv, 0);

  TF_ASSERT_OK_AND_ASSIGN(int64_t fallback_arch,
                          GetMetalNvfp4WorkspaceBytes(20, 2048, 2048));
  EXPECT_EQ(fallback_arch, 8 * 20 * 2048 * 2);
  TF_ASSERT_OK_AND_ASSIGN(int64_t gen14_ultra,
                          GetMetalNvfp4WorkspaceBytes(20, 2048, 2048, 'd', 14));
  EXPECT_EQ(gen14_ultra, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      int64_t sorted,
      GetMetalMoeWorkspaceBytes(512, 128, 2816, 704, /*is_nvfp4=*/true));
  EXPECT_EQ(sorted, 3608576);
  TF_ASSERT_OK_AND_ASSIGN(
      int64_t decode,
      GetMetalMoeWorkspaceBytes(128, 128, 2816, 704, /*is_nvfp4=*/true));
  EXPECT_EQ(decode, 0);
  TF_ASSERT_OK_AND_ASSIGN(
      int64_t unsupported_sorted_experts,
      GetMetalMoeWorkspaceBytes(4096, 257, 2816, 704,
                                /*is_nvfp4=*/true));
  EXPECT_EQ(unsupported_sorted_experts, 0);

  // Partial-N layouts stay exact.
  TF_ASSERT_OK_AND_ASSIGN(
      int64_t partial_n,
      GetMetalMoeWorkspaceBytes(32, 2, 32, 35, /*is_nvfp4=*/true));
  EXPECT_EQ(partial_n, 4544);
}

}  // namespace
}  // namespace xla::gpu
