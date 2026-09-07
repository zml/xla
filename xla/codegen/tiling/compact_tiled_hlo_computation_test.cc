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

#include "xla/codegen/tiling/compact_tiled_hlo_computation.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/codegen/tiling/tiled_hlo_schedule.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"

namespace xla {
namespace {

class CompactTiledHloComputationTest : public HloHardwareIndependentTestBase {
 protected:
  SymbolicTileAnalysisOrError Analyze(VerifiedHloModule& module) {
    HloComputation* entry = module.entry_computation();
    const std::vector<HloInstruction*> postorder =
        entry->MakeInstructionPostOrder();
    std::vector<HloInstruction*> instructions;
    for (HloInstruction* instruction : llvm::reverse(postorder)) {
      if (instruction->opcode() != HloOpcode::kParameter) {
        instructions.push_back(instruction);
      }
    }
    HloInstruction* fusion = entry->CreateFusionInstruction(
        instructions, HloInstruction::FusionKind::kLoop);
    return SymbolicTileAnalysis::AnalyzeComputation(
        *fusion->fused_instructions_computation(), &context_);
  }

  void Compare(const TiledHloComputation& full,
               const CompactTiledHloComputation& compact) {
    using CompactInstruction = CompactTiledHloComputation::InstructionType;
    const std::vector<const TiledHloInstruction*> instructions(
        full.instructions().begin(), full.instructions().end());
    ASSERT_EQ(instructions.size(), compact.instructions().size());
    absl::flat_hash_map<const TiledHloInstruction*, int64_t> full_ids;
    absl::flat_hash_map<const CompactInstruction*, int64_t> compact_ids;
    for (int64_t i = 0; i < instructions.size(); ++i) {
      full_ids[instructions[i]] = i;
      compact_ids[compact.instructions()[i]] = i;
    }
    for (int64_t i = 0; i < instructions.size(); ++i) {
      const TiledHloInstruction* expected = instructions[i];
      const CompactInstruction* actual = compact.instructions()[i];
      EXPECT_EQ(expected->hlo(), actual->hlo());
      EXPECT_EQ(expected->tile_sizes(), actual->tile_sizes());
      EXPECT_EQ(expected->tile_strides(), actual->tile_strides());
      if (const IndexingMap* offsets = actual->tile_offsets_indexing()) {
        ASSERT_OK_AND_ASSIGN(IndexingMap expected_offsets,
                             expected->tile_offsets_indexing());
        EXPECT_EQ(expected_offsets, *offsets);
      }
      ASSERT_EQ(expected->operands().size(), actual->operands().size());
      for (int64_t j = 0; j < expected->operands().size(); ++j) {
        EXPECT_EQ(full_ids.at(expected->operand(j)),
                  compact_ids.at(actual->operand(j)));
      }
      ASSERT_TRUE(expected->is_fusion_instruction().has_value());
      EXPECT_EQ(*expected->is_fusion_instruction(),
                actual->is_fusion_instruction());
    }
    EXPECT_EQ(full.num_output_tiles(), compact.num_output_tiles());
    ASSERT_EQ(full.roots().size(), compact.roots().size());
    for (int64_t i = 0; i < full.roots().size(); ++i) {
      EXPECT_EQ(full_ids.at(full.roots()[i]),
                compact_ids.at(compact.roots()[i]));
    }
  }

  absl::Status CompareAllCandidates(absl::string_view hlo,
                                    bool require_success = true,
                                    bool require_rejection = false) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
    SymbolicTileAnalysisOrError result = Analyze(*module);
    SymbolicTileAnalysis* analysis = std::get_if<SymbolicTileAnalysis>(&result);
    if (analysis == nullptr) {
      return absl::FailedPreconditionError("Symbolic analysis failed");
    }
    Decision support = CompactTiledHloComputation::CanUse(*analysis);
    if (support.IsForbidden()) {
      return absl::FailedPreconditionError(support.Explain());
    }
    TilingEvaluationWorkspace workspace(*analysis);
    CompactTiledHloComputation compact(workspace);
    MajorToMinorTiledHloSchedule schedule;
    int64_t compact_candidates = 0;
    int64_t rejected_candidates = 0;
    ABSL_RETURN_IF_ERROR(analysis->ForEachValidFlatTiling(
        [&](absl::Span<const int64_t> parameters) -> absl::Status {
          // The existing API constructs its own workspace, keeping the full
          // reference independent from the values used by the compact path.
          auto full = analysis->ComputeTiledComputation(
              parameters, schedule, /*constraints_are_known_satisfied=*/true);
          workspace.Reset(parameters);
          auto actual = compact.Update(schedule);
          EXPECT_EQ(actual.status(), full.status());
          if (!full.ok()) {
            ++rejected_candidates;
            return absl::OkStatus();
          }
          ABSL_ASSIGN_OR_RETURN(Decision decision, std::move(actual));
          if (decision.IsAllowed()) {
            ++compact_candidates;
            Compare(*full, compact);
          }
          return absl::OkStatus();
        }));
    if (require_success) {
      EXPECT_GT(compact_candidates, 0);
    }
    if (require_rejection) {
      EXPECT_GT(rejected_candidates, 0);
    }
    return absl::OkStatus();
  }

  mlir::MLIRContext context_;
};

TEST_F(CompactTiledHloComputationTest, IrregularDiamondMatchesEveryCandidate) {
  ASSERT_OK(CompareAllCandidates(R"(
HloModule diamond
ENTRY main {
  p0 = f32[2,97] parameter(0)
  exp = f32[2,97] exponential(p0)
  log = f32[2,97] log(p0)
  ROOT subtract = f32[2,97] subtract(exp, log)
})"));
}

TEST_F(CompactTiledHloComputationTest, ScalarMatchesEveryCandidate) {
  ASSERT_OK(CompareAllCandidates(R"(
HloModule scalar
ENTRY main {
  p0 = f32[] parameter(0)
  ROOT negate = f32[] negate(p0)
})"));
}

TEST_F(CompactTiledHloComputationTest, IotaMatchesEveryCandidate) {
  ASSERT_OK(CompareAllCandidates(R"(
HloModule iota
ENTRY main {
  ROOT iota = s32[100] iota(), iota_dimension=0
})"));
}

TEST_F(CompactTiledHloComputationTest,
       ExpandingReshapeMatchesClampedCandidates) {
  ASSERT_OK(CompareAllCandidates(R"(
HloModule reshape
ENTRY main {
  p0 = f32[20] parameter(0)
  abs = f32[20] abs(p0)
  ROOT reshape = f32[4,5] reshape(abs)
})"));
}

TEST_F(CompactTiledHloComputationTest, ReductionMatchesEveryCandidate) {
  ASSERT_OK(CompareAllCandidates(R"(
HloModule reduction
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}
ENTRY main {
  p0 = f32[7,17] parameter(0)
  zero = f32[] constant(0)
  ROOT reduction = f32[7] reduce(p0, zero), dimensions={1}, to_apply=add
})"));
}

TEST_F(CompactTiledHloComputationTest,
       DuplicateLeafSizesPreserveDistinctOffsets) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule offset_collision
ENTRY main {
  p0 = f32[16] parameter(0)
  left = f32[8] slice(p0), slice={[0:8]}
  right = f32[8] slice(p0), slice={[8:16]}
  ROOT add = f32[8] add(left, right)
})"));
  SymbolicTileAnalysisOrError result = Analyze(*module);
  SymbolicTileAnalysis* analysis = std::get_if<SymbolicTileAnalysis>(&result);
  ASSERT_NE(analysis, nullptr);
  ASSERT_TRUE(CompactTiledHloComputation::CanUse(*analysis).IsAllowed());
  TilingEvaluationWorkspace workspace(*analysis);
  CompactTiledHloComputation compact(workspace);
  workspace.Reset({4});
  ASSERT_OK_AND_ASSIGN(Decision decision, compact.Update());
  EXPECT_TRUE(decision.IsAllowed());
  ASSERT_OK_AND_ASSIGN(
      TiledHloComputation full,
      analysis->ComputeTiledComputation(absl::Span<const int64_t>({4})));
  int64_t parameter_tiles = 0;
  for (const TiledHloInstruction* instruction : full.instructions()) {
    parameter_tiles += instruction->hlo() ==
                       module->entry_computation()->parameter_instruction(0);
  }
  EXPECT_EQ(parameter_tiles, 2);
  Compare(full, compact);
  ASSERT_OK(CompareAllCandidates(R"(
HloModule offset_collision_all
ENTRY main {
  p0 = f32[16] parameter(0)
  left = f32[8] slice(p0), slice={[0:8]}
  right = f32[8] slice(p0), slice={[8:16]}
  ROOT add = f32[8] add(left, right)
})"));
}

TEST_F(CompactTiledHloComputationTest, NegativeStrideErrorIsUnchanged) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule reverse
ENTRY main {
  p0 = f32[16] parameter(0)
  ROOT reverse = f32[16] reverse(p0), dimensions={0}
})"));
  SymbolicTileAnalysisOrError result = Analyze(*module);
  SymbolicTileAnalysis* analysis = std::get_if<SymbolicTileAnalysis>(&result);
  ASSERT_NE(analysis, nullptr);
  ASSERT_TRUE(CompactTiledHloComputation::CanUse(*analysis).IsAllowed());
  TilingEvaluationWorkspace workspace(*analysis);
  CompactTiledHloComputation compact(workspace);
  workspace.Reset({2});
  const absl::Status expected =
      analysis->ComputeTiledComputation(absl::Span<const int64_t>({2}))
          .status();
  EXPECT_EQ(expected.code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(compact.Update().status(), expected);
}

TEST_F(CompactTiledHloComputationTest, RegionsAndRuntimeIndexingFallBack) {
  for (absl::string_view hlo : {R"(
HloModule concat
ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ROOT concat = f32[16] concatenate(p0, p1), dimensions={0}
})",
                                R"(
HloModule dynamic
ENTRY main {
  p0 = f32[16] parameter(0)
  start = s32[] parameter(1)
  ROOT slice = f32[8] dynamic-slice(p0, start), dynamic_slice_sizes={8}
})"}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    SymbolicTileAnalysisOrError result = Analyze(*module);
    SymbolicTileAnalysis* analysis = std::get_if<SymbolicTileAnalysis>(&result);
    ASSERT_NE(analysis, nullptr);
    EXPECT_TRUE(CompactTiledHloComputation::CanUse(*analysis).IsForbidden());
  }
}

TEST_F(CompactTiledHloComputationTest, MultipleOutputsPreserveRealRootIndex) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule multiple_outputs
ENTRY main {
  p0 = f32[16] parameter(0)
  negate = f32[16] negate(p0)
  ROOT tuple = (f32[16], f32[16]) tuple(p0, negate)
})"));
  SymbolicTileAnalysisOrError result = Analyze(*module);
  SymbolicTileAnalysis* analysis = std::get_if<SymbolicTileAnalysis>(&result);
  ASSERT_NE(analysis, nullptr);
  EXPECT_TRUE(CompactTiledHloComputation::CanUse(*analysis).IsAllowed());
  EXPECT_EQ(analysis->real_root_index(), 1);
  ASSERT_OK(CompareAllCandidates(R"(
HloModule multiple_outputs_all
ENTRY main {
  p0 = f32[7,17] parameter(0)
  abs = f32[7,17] abs(p0)
  negate = f32[7,17] negate(abs)
  ROOT tuple = (f32[7,17], f32[7,17]) tuple(abs, negate)
})"));
}

TEST_F(CompactTiledHloComputationTest, MultipleOutputPaddingRejectionsMatch) {
  ASSERT_OK(CompareAllCandidates(R"(
HloModule multiple_output_padding
ENTRY main {
  p0 = f32[36] parameter(0)
  abs = f32[36] abs(p0)
  reshape = f32[3,12] reshape(abs)
  ROOT tuple = (f32[3,12], f32[36]) tuple(reshape, abs)
})",
                                 /*require_success=*/true,
                                 /*require_rejection=*/true));
}

TEST_F(CompactTiledHloComputationTest, DuplicateRootRejectionsMatch) {
  ASSERT_OK(CompareAllCandidates(R"(
HloModule repeated_roots
ENTRY main {
  p0 = f32[8,8] parameter(0)
  abs = f32[8,8] abs(p0)
  negate = f32[8,8] negate(abs)
  ROOT tuple = (f32[8,8], f32[8,8], f32[8,8]) tuple(abs, negate, abs)
})",
                                 /*require_success=*/false,
                                 /*require_rejection=*/true));
}

TEST_F(CompactTiledHloComputationTest, MultipleOutputsWithLeafOffsetsMatch) {
  ASSERT_OK(CompareAllCandidates(R"(
HloModule multiple_outputs_and_offsets
ENTRY main {
  p0 = f32[16] parameter(0)
  left = f32[8] slice(p0), slice={[0:8]}
  right = f32[8] slice(p0), slice={[8:16]}
  add = f32[8] add(left, right)
  ROOT tuple = (f32[8], f32[8]) tuple(left, add)
})"));
}

TEST_F(CompactTiledHloComputationTest, MultipleOutputsPreserveScheduleErrors) {
  class NonDefaultSchedule : public TiledHloSchedule {
   public:
    absl::StatusOr<IndexingMap> Schedule(
        const IndexingMap& offsets, IterationSpace iteration_space,
        mlir::MLIRContext* context) const override {
      return MajorToMinorTiledHloSchedule().Schedule(offsets, iteration_space,
                                                     context);
    }
  } schedule;
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule multiple_outputs_schedule
ENTRY main {
  p0 = f32[8] parameter(0)
  abs = f32[8] abs(p0)
  negate = f32[8] negate(abs)
  ROOT tuple = (f32[8], f32[8]) tuple(abs, negate)
})"));
  SymbolicTileAnalysisOrError result = Analyze(*module);
  auto* analysis = std::get_if<SymbolicTileAnalysis>(&result);
  ASSERT_NE(analysis, nullptr);
  ASSERT_TRUE(CompactTiledHloComputation::CanUse(*analysis).IsAllowed());
  TilingEvaluationWorkspace workspace(*analysis);
  CompactTiledHloComputation compact(workspace);
  workspace.Reset({4});
  auto full = analysis->ComputeTiledComputation(
      absl::Span<const int64_t>({4}), schedule,
      /*constraints_are_known_satisfied=*/true);
  EXPECT_EQ(full.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(compact.Update(schedule).status(), full.status());
}

}  // namespace
}  // namespace xla
