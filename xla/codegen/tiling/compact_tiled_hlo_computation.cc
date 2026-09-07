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

#include <cstddef>
#include <cstdint>
#include <typeinfo>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/util.h"

namespace xla {

Decision CompactTiledHloComputation::CanUse(
    const SymbolicTileAnalysis& analysis) {
  if (analysis.GetRoots().empty()) {
    return Decision::Forbid("No output roots to tile");
  }
  const IndexingMap& root = analysis.GetRealRootIndexing();
  if (!root.GetSymbolicMap().IsIdentity() || root.GetSymbolCount() != 0 ||
      !root.GetSymbolicConstraints().empty()) {
    return Decision::Forbid("Root indexing requires full output tiling");
  }
  for (const IndexingMap::Variable& dim : root.GetDimVars()) {
    if (dim.bounds.lower != 0 || dim.bounds.upper < 0) {
      return Decision::Forbid("Root bounds require full output tiling");
    }
  }
  const auto& instructions = analysis.GetSymbolicTiledHloComputation();
  if (instructions.empty() ||
      instructions.back()->hlo() !=
          analysis.GetRoot(analysis.real_root_index())) {
    return Decision::Forbid("Root instruction requires full materialization");
  }
  for (const auto& instruction : instructions) {
    if (!instruction->regions().empty() ||
        !instruction->runtime_variables().empty() ||
        instruction->indexing_map().GetRTVarsCount() != 0) {
      return Decision::Forbid(
          "Regions or runtime indexing require full tiling");
    }
    if (instruction->indexing_map().GetRangeVarsCount() != 0 &&
        !instruction->indexing_map().GetSymbolicConstraints().empty()) {
      return Decision::Forbid(
          "Range constraints require full offset validation");
    }
    // Full materialization computes offsets for iotas and leaves whose size
    // hashes collide. With zero-based range variables and no runtime symbols,
    // composing with the zero-based identity-root schedule cannot encounter
    // ComputeTileOffsetIndexing's nonzero-symbol-lower-bound error. Offsets
    // therefore need only be computed when they can affect leaf equality.
    for (const Interval& bound :
         instruction->indexing_map().GetSymbolBounds()) {
      if (bound.lower != 0) {
        return Decision::Forbid(
            "Offset validation requires full materialization");
      }
    }
  }
  return Decision::Allow();
}

CompactTiledHloComputation::CompactTiledHloComputation(
    TilingEvaluationWorkspace& workspace)
    : workspace_(workspace),
      records_(workspace.analysis().num_symbolic_tiled_hlo_instructions()),
      may_deduplicate_(records_.size(), false),
      canonical_ids_(records_.size()),
      dedup_(0, Hash{this}, Eq{this}) {
  CHECK(CanUse(workspace.analysis()).IsAllowed());
  instructions_.reserve(records_.size());
  symbolic_instructions_.reserve(records_.size());
  llvm::DenseMap<const HloInstruction*, int64_t> first_occurrence;
  first_occurrence.reserve(records_.size());
  for (const auto& symbolic :
       workspace.analysis().GetSymbolicTiledHloComputation()) {
    InstructionType& record = records_[symbolic->id()];
    record.owner_ = this;
    record.symbolic_ = symbolic.get();
    record.sizes_ = &workspace.tile_sizes(symbolic.get());
    record.strides_ = &workspace.tile_strides(symbolic.get());
    record.operand_ids_.resize(symbolic->operands().size());
    const auto [it, inserted] =
        first_occurrence.try_emplace(symbolic->hlo(), symbolic->id());
    if (!inserted) {
      may_deduplicate_[it->second] = true;
      may_deduplicate_[symbolic->id()] = true;
    }
  }
  dedup_.reserve(llvm::count(may_deduplicate_, true));
}

size_t CompactTiledHloComputation::Hash::operator()(int64_t index) const {
  const InstructionType& record = owner->records_[index];
  return absl::HashOf(
      record.hlo(), absl::Span<const int64_t>(record.tile_sizes()),
      absl::Span<const int64_t>(record.tile_strides()), record.operands());
}

bool CompactTiledHloComputation::Eq::operator()(int64_t lhs,
                                                int64_t rhs) const {
  const InstructionType& left = owner->records_[lhs];
  const InstructionType& right = owner->records_[rhs];
  if (left.hlo() != right.hlo() || left.tile_sizes() != right.tile_sizes() ||
      left.tile_strides() != right.tile_strides()) {
    return false;
  }
  if (left.operands().empty() && right.operands().empty()) {
    const IndexingMap* left_offsets = left.tile_offsets_indexing();
    const IndexingMap* right_offsets = right.tile_offsets_indexing();
    if (left_offsets == nullptr || right_offsets == nullptr) {
      return left_offsets == right_offsets;
    }
    return *left_offsets == *right_offsets;
  }
  return left.operand_ids_ == right.operand_ids_;
}

absl::StatusOr<Decision> CompactTiledHloComputation::Update() {
  MajorToMinorTiledHloSchedule schedule;
  return Update(schedule);
}

absl::StatusOr<Decision> CompactTiledHloComputation::Update(
    const TiledHloSchedule& schedule) {
  // Reset by iterators because the old keys refer to workspace values that
  // were invalidated by Reset, and clear() may release the retained capacity.
  dedup_.erase(dedup_.begin(), dedup_.end());
  instructions_.clear();
  symbolic_instructions_.clear();
  roots_.clear();
  offsets_ready_ = false;
  ABSL_RETURN_IF_ERROR(workspace_.EvaluateTileStrides());
  workspace_.EvaluateTileSizes();

  const SymbolicTileAnalysis& analysis = workspace_.analysis();
  if (analysis.GetRoots().size() > 1 ||
      typeid(schedule) != typeid(MajorToMinorTiledHloSchedule)) {
    ABSL_RETURN_IF_ERROR(analysis.ComputeTileOffsetsForCostModel(
        workspace_, schedule, offsets_));
    offsets_ready_ = true;
  }
  auto deduplicate = [&]() {
    for (const auto& symbolic : analysis.GetSymbolicTiledHloComputation()) {
      const int64_t id = symbolic->id();
      InstructionType& record = records_[id];
      for (auto [i, operand] : llvm::enumerate(symbolic->operands())) {
        record.operand_ids_[i] = canonical_ids_[operand->id()];
      }
      int64_t canonical_id = id;
      bool inserted = true;
      if (may_deduplicate_[id]) {
        const auto [it, was_inserted] = dedup_.insert(id);
        canonical_id = *it;
        inserted = was_inserted;
      }
      if (!inserted && record.operands().empty() && !offsets_ready_) {
        return false;
      }
      canonical_ids_[id] = canonical_id;
      if (inserted) {
        instructions_.push_back(&record);
        symbolic_instructions_.push_back(symbolic.get());
      }
    }
    return true;
  };
  if (!deduplicate()) {
    // The successful single-root path avoids the offset prepass. Once leaf
    // equality needs offsets, restart using the exact full-path heuristic,
    // including hash collisions and iotas, before comparing any offset maps.
    dedup_.erase(dedup_.begin(), dedup_.end());
    instructions_.clear();
    symbolic_instructions_.clear();
    ABSL_RETURN_IF_ERROR(analysis.ComputeTileOffsetsForCostModel(
        workspace_, schedule, offsets_));
    offsets_ready_ = true;
    CHECK(deduplicate());
  }

  if (offsets_ready_) {
    ABSL_ASSIGN_OR_RETURN(
        std::vector<int64_t> root_indices,
        analysis.InitializeTiledRootsForCostModel(
            workspace_, schedule, symbolic_instructions_, offsets_));
    for (int64_t index : root_indices) {
      roots_.push_back(instructions_[index]);
    }
    num_output_tiles_ = Product(offsets_.num_output_tiles_per_dim);
    return Decision::Allow();
  }

  // Identity root indexing activates each dimension exactly once, in order.
  // This is the same outer_loop_bounds product as ComputeOutputTilingInfo;
  // it intentionally uses original parameters rather than stride clamping.
  num_output_tiles_ = 1;
  for (auto [dim, tile_size] :
       llvm::zip(analysis.GetRealRootIndexing().GetDimVars(),
                 workspace_.flat_tiling())) {
    num_output_tiles_ *= CeilOfRatio(dim.bounds.upper + 1, tile_size);
  }
  roots_.push_back(instructions_.back());
  return Decision::Allow();
}

}  // namespace xla
