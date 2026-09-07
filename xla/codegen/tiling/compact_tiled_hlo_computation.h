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

#ifndef XLA_CODEGEN_TILING_COMPACT_TILED_HLO_COMPUTATION_H_
#define XLA_CODEGEN_TILING_COMPACT_TILED_HLO_COMPUTATION_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/decision.h"

namespace xla {

// A cost-model view with storage scoped to a single tiling search. Concrete
// sizes and strides are borrowed from the workspace. It is intentionally not
// an emitter representation. Offset-dependent equality and additional roots
// reuse the analysis's offset preparation and buffer-sharing validation.
class CompactTiledHloComputation {
 public:
  class InstructionType {
   public:
    const HloInstruction* hlo() const { return symbolic_->hlo(); }
    bool is_fusion_instruction() const {
      return symbolic_->is_fusion_instruction();
    }
    const llvm::SmallVector<int64_t>& tile_sizes() const { return *sizes_; }
    const llvm::SmallVector<int64_t>& tile_strides() const { return *strides_; }
    int64_t tile_size(int64_t dim) const { return tile_sizes()[dim]; }
    int64_t tile_stride(int64_t dim) const { return tile_strides()[dim]; }
    const InstructionType* operand(int64_t index) const {
      return &owner_->records_[operand_ids_[index]];
    }
    // Canonical operand IDs are sufficient for the model's emptiness checks
    // and preserve the full representation's operand identity for dedup.
    absl::Span<const int64_t> operands() const { return operand_ids_; }

    const IndexingMap* tile_offsets_indexing() const {
      return owner_->offsets_ready_ ? owner_->offsets_.Get(symbolic_) : nullptr;
    }

   private:
    friend class CompactTiledHloComputation;
    const CompactTiledHloComputation* owner_ = nullptr;
    const SymbolicTiledHloInstruction* symbolic_ = nullptr;
    // The workspace owns these vector objects at stable addresses. Candidate
    // evaluation replaces their contents; no evaluated values are cached here.
    const llvm::SmallVector<int64_t>* sizes_ = nullptr;
    const llvm::SmallVector<int64_t>* strides_ = nullptr;
    llvm::SmallVector<int64_t> operand_ids_;
  };

  // Tests indexing invariants once per analysis, independently of target
  // hardware and HLO opcode names. Unsupported analyses use the full path.
  static Decision CanUse(const SymbolicTileAnalysis& analysis);

  // Call only after CanUse returned Allow. The workspace and its analysis must
  // outlive this view. Stable ownership is necessary for the record accessors
  // and the stateful dedup hash/equality functions.
  explicit CompactTiledHloComputation(TilingEvaluationWorkspace& workspace);
  CompactTiledHloComputation(const CompactTiledHloComputation&) = delete;
  CompactTiledHloComputation& operator=(const CompactTiledHloComputation&) =
      delete;
  CompactTiledHloComputation(CompactTiledHloComputation&&) = delete;
  CompactTiledHloComputation& operator=(CompactTiledHloComputation&&) = delete;

  // After workspace.Reset(valid_flat_parameters), preserves original negative
  // stride/offset/root-validation errors. Contents are valid only
  // until the next workspace.Reset() or Update().
  absl::StatusOr<Decision> Update();
  absl::StatusOr<Decision> Update(const TiledHloSchedule& schedule);

  absl::Span<const InstructionType* const> instructions() const {
    return instructions_;
  }
  absl::Span<const InstructionType* const> roots() const { return roots_; }
  int64_t num_output_tiles() const { return num_output_tiles_; }

 private:
  struct Hash {
    const CompactTiledHloComputation* owner;
    size_t operator()(int64_t index) const;
  };
  struct Eq {
    const CompactTiledHloComputation* owner;
    bool operator()(int64_t lhs, int64_t rhs) const;
  };

  TilingEvaluationWorkspace& workspace_;
  std::vector<InstructionType> records_;
  std::vector<int64_t> canonical_ids_;
  std::vector<const InstructionType*> instructions_;
  std::vector<const SymbolicTiledHloInstruction*> symbolic_instructions_;
  llvm::SmallVector<const InstructionType*, 1> roots_;
  absl::flat_hash_set<int64_t, Hash, Eq> dedup_;
  TilingCandidateOffsets offsets_;
  bool offsets_ready_ = false;
  int64_t num_output_tiles_ = 1;
};

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_COMPACT_TILED_HLO_COMPUTATION_H_
