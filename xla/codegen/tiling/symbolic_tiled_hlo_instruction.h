/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_CODEGEN_TILING_SYMBOLIC_TILED_HLO_INSTRUCTION_H_
#define XLA_CODEGEN_TILING_SYMBOLIC_TILED_HLO_INSTRUCTION_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/symbolic_tile.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// A node in the symbolic tiled representation of an HLO computation. During
// tiling and codegen an HLO instruction may need to be emitted multiple times
// with different tiling parameters.
class SymbolicTiledHloInstruction {
 public:
  SymbolicTiledHloInstruction(
      const HloInstruction* hlo, IndexingMap indexing_map,
      std::vector<SymbolicTiledHloInstruction*> runtime_variables = {})
      : hlo_(hlo),
        indexing_map_(std::move(indexing_map)),
        runtime_variables_(std::move(runtime_variables)) {
    CHECK_EQ(indexing_map_.GetRTVars().size(), runtime_variables_.size());
  }

  virtual ~SymbolicTiledHloInstruction() = default;

  int64_t id() const { return id_; }
  void set_id(int64_t id) {
    CHECK_LT(id_, 0);
    id_ = id;
  }

  int64_t tile_parameter_bounds_id() const {
    return tile_parameter_bounds_id_;
  }
  void set_tile_parameter_bounds_id(int64_t id) {
    CHECK_LT(tile_parameter_bounds_id_, 0);
    tile_parameter_bounds_id_ = id;
  }

  const HloInstruction* hlo() const { return hlo_; }
  bool is_fusion_instruction() const { return is_fusion_instruction_; }
  void set_is_fusion_instruction(bool value) {
    is_fusion_instruction_ = value;
  }
  const IndexingMap& indexing_map() const { return indexing_map_; }
  void set_symbolic_tile(SymbolicTile symbolic_tile) {
    symbolic_tile_ = std::move(symbolic_tile);
  }
  const SymbolicTile& symbolic_tile() const { return *symbolic_tile_; }

  const SymbolicTiledHloInstruction* operand(int64_t operand_id) const {
    return operands_[operand_id];
  }
  SymbolicTiledHloInstruction* operand(int64_t operand_id) {
    return operands_[operand_id];
  }
  const std::vector<SymbolicTiledHloInstruction*>& operands() const {
    return operands_;
  }

  // Tiling of runtime variables of the indexing map of the instruction.
  const std::vector<SymbolicTiledHloInstruction*>& runtime_variables() const {
    return runtime_variables_;
  }

  // Appends an operand to the end of the operand list.
  void AppendOperand(SymbolicTiledHloInstruction* operand) {
    operands_.push_back(operand);
  }

  // List of "regions" (i.e. loop bodies or conditional branches) that are
  // part of this instruction. Regions represent control flow that will be used
  // later by the emitter. Interpretation of the contents depends on the HLO
  // opcode.
  // - dot has a single region for the entire dot loop body (including all its
  //   operands);
  // - concatenation has a region per operand.
  absl::Span<const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>>
  regions() const {
    return regions_;
  }

  void AddRegion(
      std::vector<std::unique_ptr<SymbolicTiledHloInstruction>> region) {
    regions_.push_back(std::move(region));
  }

  // Returns a string representation of the instruction. Used only for error
  // messages and debugging.
  std::string ToString(absl::string_view field_separator = "\n\t") const;

 private:
  // Dense identifier assigned by SymbolicTileAnalysis and used by concrete
  // tiling to index per-candidate data without pointer-keyed hash maps.
  int64_t id_ = -1;

  // Dense identifier of the tile-parameter upper-bound pattern. Assigned for
  // top-level instructions by SymbolicTileAnalysis.
  int64_t tile_parameter_bounds_id_ = -1;

  // Pointer to the original HLO instruction.
  const HloInstruction* hlo_;

  // Whether this HLO is emitted inside the fusion rather than loaded as an
  // external operand. Computed once while traversing the fusion adaptor.
  bool is_fusion_instruction_ = false;

  // Indexing map from the computation root to this instruction output.
  IndexingMap indexing_map_;

  // Symbolic tile derived from the indexing map. Should be computed outside of
  // this class and set before usage. Wrapped in an optional, because
  // SymbolicTile does not have a default constructor.
  std::optional<SymbolicTile> symbolic_tile_;

  // Operands of the instruction in the tiled computation graph.
  std::vector<SymbolicTiledHloInstruction*> operands_;

  // Tiling of runtime variables of `indexing_map_`.
  std::vector<SymbolicTiledHloInstruction*> runtime_variables_;

  // Regions of instructions that are part of this instruction, e.g. loop body
  // or conditional branches.
  llvm::SmallVector<std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>>
      regions_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_SYMBOLIC_TILED_HLO_INSTRUCTION_H_
