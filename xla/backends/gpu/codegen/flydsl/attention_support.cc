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

#include "xla/backends/gpu/codegen/flydsl/attention_support.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/backends/gpu/codegen/flydsl/softmax_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace xla::gpu::flydsl {
namespace {

const HloInstruction* StripViews(const HloInstruction* instruction) {
  while ((instruction->opcode() == HloOpcode::kBitcast ||
          instruction->opcode() == HloOpcode::kReshape) &&
         instruction->operand_count() == 1) {
    instruction = instruction->operand(0);
  }
  return instruction;
}

const HloInstruction* StripDataMovementViews(
    const HloInstruction* instruction) {
  while (instruction->operand_count() == 1 &&
         (instruction->opcode() == HloOpcode::kBitcast ||
          instruction->opcode() == HloOpcode::kCopy ||
          instruction->opcode() == HloOpcode::kReshape ||
          instruction->opcode() == HloOpcode::kTranspose)) {
    instruction = instruction->operand(0);
  }
  return instruction;
}

template <typename Dimensions>
bool IsSingleDimension(const Dimensions& dimensions, int64_t dimension) {
  return dimensions.size() == 1 && dimensions.Get(0) == dimension;
}

bool ContainsInstruction(const HloInstruction* root,
                         const HloInstruction* needle) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> worklist = {root};
  while (!worklist.empty()) {
    const HloInstruction* instruction = worklist.back();
    worklist.pop_back();
    if (instruction == needle) {
      return true;
    }
    if (!visited.insert(instruction).second) {
      continue;
    }
    worklist.insert(worklist.end(), instruction->operands().begin(),
                    instruction->operands().end());
  }
  return false;
}

const HloInstruction* FindUniqueParameter(const HloInstruction* root) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> worklist = {root};
  const HloInstruction* parameter = nullptr;
  while (!worklist.empty()) {
    const HloInstruction* instruction = worklist.back();
    worklist.pop_back();
    if (!visited.insert(instruction).second) {
      continue;
    }
    if (instruction->opcode() == HloOpcode::kParameter) {
      if (parameter != nullptr && parameter != instruction) {
        return nullptr;
      }
      parameter = instruction;
      continue;
    }
    worklist.insert(worklist.end(), instruction->operands().begin(),
                    instruction->operands().end());
  }
  return parameter;
}

void CollectGraph(const HloInstruction* root,
                  absl::flat_hash_set<const HloInstruction*>& visited,
                  std::vector<const HloInstruction*>& dots,
                  std::vector<const HloInstruction*>& parameters) {
  if (!visited.insert(root).second) {
    return;
  }
  if (root->opcode() == HloOpcode::kDot) {
    dots.push_back(root);
  } else if (root->opcode() == HloOpcode::kParameter) {
    parameters.push_back(root);
  }
  for (const HloInstruction* operand : root->operands()) {
    CollectGraph(operand, visited, dots, parameters);
  }
}

std::optional<double> UniformConstant(const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsScalar(instruction->shape())) {
    return instruction->literal().GetAsDouble({});
  }
  if (instruction->operand_count() != 1) {
    return std::nullopt;
  }
  switch (instruction->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kReshape:
      return UniformConstant(instruction->operand(0));
    default:
      return std::nullopt;
  }
}

std::optional<double> MatchScaledDot(const HloInstruction* instruction,
                                     const HloInstruction* dot) {
  if (instruction == dot) {
    return 1.0;
  }
  if (instruction->opcode() == HloOpcode::kMultiply &&
      instruction->operand_count() == 2) {
    for (int operand = 0; operand < 2; ++operand) {
      if (!ContainsInstruction(instruction->operand(operand), dot)) {
        continue;
      }
      if (std::optional<double> scale =
              UniformConstant(instruction->operand(1 - operand))) {
        if (std::optional<double> nested =
                MatchScaledDot(instruction->operand(operand), dot)) {
          return *scale * *nested;
        }
      }
    }
    return std::nullopt;
  }
  if (instruction->operand_count() != 1) {
    return std::nullopt;
  }
  switch (instruction->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kReshape:
      return MatchScaledDot(instruction->operand(0), dot);
    default:
      return std::nullopt;
  }
}

struct BroadcastParameterMatch {
  const HloInstruction* parameter;
  std::array<int64_t, 4> strides;
};

std::optional<std::vector<int64_t>> InitialLogicalDimensions(
    const Shape& score_shape,
    const std::array<int64_t, 4>& logical_score_shape) {
  if (score_shape.dimensions_size() == 4 &&
      score_shape.dimensions(0) == logical_score_shape[0] &&
      score_shape.dimensions(1) == logical_score_shape[1] &&
      score_shape.dimensions(2) == logical_score_shape[2] &&
      score_shape.dimensions(3) == logical_score_shape[3]) {
    return std::vector<int64_t>{0, 1, 2, 3};
  }
  if (score_shape.dimensions_size() == 3 && logical_score_shape[1] == 1 &&
      score_shape.dimensions(0) == logical_score_shape[0] &&
      score_shape.dimensions(1) == logical_score_shape[2] &&
      score_shape.dimensions(2) == logical_score_shape[3]) {
    // Layout simplification folds a unit head dimension out of attention
    // scores. Preserve the logical [B,H,Q,K] mapping with a zero H stride.
    return std::vector<int64_t>{0, 2, 3};
  }
  return std::nullopt;
}

// Carries the logical score-dimension mapping through one materialization-free
// unary operation. Reshapes are intentionally limited to inserting/removing
// unit dimensions; accepting arbitrary flattening would lose which dimension
// is the query or key coordinate.
bool MapLogicalDimensionsToOperand(
    const HloInstruction* instruction,
    std::vector<int64_t>* logical_dimensions) {
  if (instruction->operand_count() != 1 ||
      instruction->shape().dimensions_size() != logical_dimensions->size()) {
    return false;
  }
  const HloInstruction* operand = instruction->operand(0);
  switch (instruction->opcode()) {
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
      return ShapeUtil::SameDimensions(instruction->shape(), operand->shape());
    case HloOpcode::kBroadcast: {
      if (instruction->dimensions().size() !=
          operand->shape().dimensions_size()) {
        return false;
      }
      std::vector<int64_t> operand_dimensions(
          operand->shape().dimensions_size(), -1);
      for (int64_t dimension = 0;
           dimension < operand->shape().dimensions_size(); ++dimension) {
        const int64_t output_dimension = instruction->dimensions(dimension);
        if (output_dimension < 0 ||
            output_dimension >= logical_dimensions->size()) {
          return false;
        }
        operand_dimensions[dimension] =
            (*logical_dimensions)[output_dimension];
      }
      *logical_dimensions = std::move(operand_dimensions);
      return true;
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape: {
      std::vector<int64_t> output_non_unit;
      std::vector<int64_t> input_non_unit;
      for (int64_t dimension = 0;
           dimension < instruction->shape().dimensions_size(); ++dimension) {
        if (instruction->shape().dimensions(dimension) != 1) {
          output_non_unit.push_back(dimension);
        }
      }
      for (int64_t dimension = 0;
           dimension < operand->shape().dimensions_size(); ++dimension) {
        if (operand->shape().dimensions(dimension) != 1) {
          input_non_unit.push_back(dimension);
        }
      }
      if (output_non_unit.size() != input_non_unit.size()) {
        return false;
      }
      std::vector<int64_t> operand_dimensions(
          operand->shape().dimensions_size(), -1);
      for (int64_t index = 0; index < output_non_unit.size(); ++index) {
        const int64_t output_dimension = output_non_unit[index];
        const int64_t input_dimension = input_non_unit[index];
        if (instruction->shape().dimensions(output_dimension) !=
            operand->shape().dimensions(input_dimension)) {
          return false;
        }
        operand_dimensions[input_dimension] =
            (*logical_dimensions)[output_dimension];
      }
      *logical_dimensions = std::move(operand_dimensions);
      return true;
    }
    default:
      return false;
  }
}

std::optional<BroadcastParameterMatch> MatchBroadcastParameterFromDimensions(
    const HloInstruction* instruction,
    std::vector<int64_t> logical_dimensions,
    const std::array<int64_t, 4>& logical_score_shape, bool predicate) {
  while (instruction->opcode() != HloOpcode::kParameter) {
    if (!MapLogicalDimensionsToOperand(instruction, &logical_dimensions)) {
      return std::nullopt;
    }
    instruction = instruction->operand(0);
  }

  const Shape& parameter_shape = instruction->shape();
  const bool supported_type =
      predicate
          ? parameter_shape.element_type() == PRED
          : (parameter_shape.element_type() == F32 ||
             parameter_shape.element_type() == BF16 ||
             parameter_shape.element_type() == F16);
  if (!supported_type || !parameter_shape.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(parameter_shape.layout()) ||
      parameter_shape.dimensions_size() != logical_dimensions.size()) {
    return std::nullopt;
  }

  std::vector<int64_t> parameter_strides(parameter_shape.dimensions_size());
  int64_t stride = 1;
  for (int64_t dimension = parameter_shape.dimensions_size() - 1;
       dimension >= 0; --dimension) {
    parameter_strides[dimension] = stride;
    stride *= parameter_shape.dimensions(dimension);
  }
  std::array<int64_t, 4> logical_strides = {0, 0, 0, 0};
  std::array<bool, 4> seen = {false, false, false, false};
  for (int64_t dimension = 0; dimension < logical_dimensions.size();
       ++dimension) {
    const int64_t logical_dimension = logical_dimensions[dimension];
    if (logical_dimension < 0) {
      if (parameter_shape.dimensions(dimension) != 1) {
        return std::nullopt;
      }
      continue;
    }
    if (logical_dimension >= 4 || seen[logical_dimension] ||
        parameter_shape.dimensions(dimension) !=
            logical_score_shape[logical_dimension]) {
      return std::nullopt;
    }
    seen[logical_dimension] = true;
    logical_strides[logical_dimension] = parameter_strides[dimension];
  }
  return BroadcastParameterMatch{instruction, logical_strides};
}

// Maps a broadcastable bias expression back to its parameter. The score
// tensor is logically [B,H,Q,K]. We accept unary conversions/copies,
// broadcasts, and bitcasts/reshapes that only insert or remove unit
// dimensions. This is enough for XLA's canonical [1,H,Q,K] bias lowering
// without pretending arbitrary reshapes are safe.
std::optional<BroadcastParameterMatch> MatchBroadcastParameter(
    const HloInstruction* instruction, const Shape& score_shape,
    const std::array<int64_t, 4>& logical_score_shape, bool predicate) {
  if (!ShapeUtil::SameDimensions(instruction->shape(), score_shape)) {
    return std::nullopt;
  }
  std::optional<std::vector<int64_t>> logical_dimensions =
      InitialLogicalDimensions(score_shape, logical_score_shape);
  if (!logical_dimensions.has_value()) {
    return std::nullopt;
  }
  return MatchBroadcastParameterFromDimensions(
      instruction, std::move(*logical_dimensions), logical_score_shape,
      predicate);
}

bool IsCausalPredicateFromDimensions(
    const HloInstruction* instruction,
    std::vector<int64_t> logical_dimensions, int64_t sequence) {
  while (instruction->operand_count() == 1) {
    if (!MapLogicalDimensionsToOperand(instruction, &logical_dimensions)) {
      break;
    }
    instruction = instruction->operand(0);
  }
  if (instruction->opcode() != HloOpcode::kCompare ||
      instruction->operand_count() != 2 ||
      instruction->shape().dimensions_size() != logical_dimensions.size()) {
    return false;
  }
  const HloInstruction* lhs = instruction->operand(0);
  const HloInstruction* rhs = instruction->operand(1);
  if (lhs->opcode() != HloOpcode::kIota ||
      rhs->opcode() != HloOpcode::kIota ||
      !ShapeUtil::Compatible(lhs->shape(), rhs->shape())) {
    return false;
  }
  const int64_t lhs_dimension =
      Cast<const HloIotaInstruction>(lhs)->iota_dimension();
  const int64_t rhs_dimension =
      Cast<const HloIotaInstruction>(rhs)->iota_dimension();
  if (lhs_dimension < 0 || lhs_dimension >= logical_dimensions.size() ||
      rhs_dimension < 0 || rhs_dimension >= logical_dimensions.size() ||
      lhs->shape().dimensions(lhs_dimension) != sequence ||
      rhs->shape().dimensions(rhs_dimension) != sequence) {
    return false;
  }
  const int64_t lhs_logical_dimension = logical_dimensions[lhs_dimension];
  const int64_t rhs_logical_dimension = logical_dimensions[rhs_dimension];
  const bool key_le_query =
      instruction->comparison_direction() == ComparisonDirection::kLe &&
      lhs_logical_dimension == 3 && rhs_logical_dimension == 2;
  const bool query_ge_key =
      instruction->comparison_direction() == ComparisonDirection::kGe &&
      lhs_logical_dimension == 2 && rhs_logical_dimension == 3;
  return key_le_query || query_ge_key;
}

bool HasCausalPredicateFromDimensions(
    const HloInstruction* instruction,
    std::vector<int64_t> logical_dimensions, int64_t sequence) {
  if (IsCausalPredicateFromDimensions(instruction, logical_dimensions,
                                      sequence)) {
    return true;
  }
  while (instruction->operand_count() == 1 &&
         MapLogicalDimensionsToOperand(instruction, &logical_dimensions)) {
    instruction = instruction->operand(0);
  }
  if (instruction->opcode() != HloOpcode::kAnd ||
      instruction->operand_count() != 2) {
    return false;
  }
  const bool lhs_causal = IsCausalPredicateFromDimensions(
      instruction->operand(0), logical_dimensions, sequence);
  const bool rhs_causal = IsCausalPredicateFromDimensions(
      instruction->operand(1), logical_dimensions, sequence);
  // Accept exactly one top-level causal conjunct. Keeping the pure matcher
  // separate prevents nested conjunctions from silently dropping a mask.
  return lhs_causal != rhs_causal;
}

struct AttentionMaskMatch {
  bool causal;
  std::optional<BroadcastParameterMatch> external;
};

std::optional<AttentionMaskMatch> MatchAttentionMask(
    const HloInstruction* instruction, const Shape& score_shape,
    const std::array<int64_t, 4>& logical_score_shape) {
  if (instruction->opcode() != HloOpcode::kSelect ||
      instruction->operand_count() != 3 ||
      !ShapeUtil::SameDimensions(instruction->shape(), score_shape)) {
    return std::nullopt;
  }
  std::optional<double> masked_value = UniformConstant(instruction->operand(2));
  if (!masked_value.has_value() || !std::isinf(*masked_value) ||
      *masked_value >= 0.0) {
    return std::nullopt;
  }
  std::optional<std::vector<int64_t>> initial_dimensions =
      InitialLogicalDimensions(score_shape, logical_score_shape);
  if (!initial_dimensions.has_value()) {
    return std::nullopt;
  }
  const HloInstruction* predicate = instruction->operand(0);
  if (std::optional<BroadcastParameterMatch> external =
          MatchBroadcastParameterFromDimensions(
              predicate, *initial_dimensions, logical_score_shape,
              /*predicate=*/true)) {
    return AttentionMaskMatch{/*causal=*/false, std::move(external)};
  }

  const bool causal = HasCausalPredicateFromDimensions(
      predicate, *initial_dimensions, logical_score_shape[2]);
  if (!causal) {
    return std::nullopt;
  }

  // Carry the score mapping through any outer broadcast/unit views to the
  // conjunction. A bare causal predicate has no external parameter.
  std::vector<int64_t> conjunct_dimensions = *initial_dimensions;
  const HloInstruction* conjunction = predicate;
  while (conjunction->operand_count() == 1 &&
         MapLogicalDimensionsToOperand(conjunction, &conjunct_dimensions)) {
    conjunction = conjunction->operand(0);
  }
  if (conjunction->opcode() != HloOpcode::kAnd ||
      conjunction->operand_count() != 2) {
    return AttentionMaskMatch{/*causal=*/true, std::nullopt};
  }

  const bool lhs_causal = IsCausalPredicateFromDimensions(
      conjunction->operand(0), conjunct_dimensions, logical_score_shape[2]);
  const bool rhs_causal = IsCausalPredicateFromDimensions(
      conjunction->operand(1), conjunct_dimensions, logical_score_shape[2]);
  if (lhs_causal == rhs_causal) {
    return std::nullopt;
  }
  std::optional<BroadcastParameterMatch> external =
      MatchBroadcastParameterFromDimensions(
          conjunction->operand(lhs_causal ? 1 : 0), conjunct_dimensions,
          logical_score_shape, /*predicate=*/true);
  if (!external.has_value()) {
    return std::nullopt;
  }
  return AttentionMaskMatch{/*causal=*/true, std::move(external)};
}

struct AttentionScoresMatch {
  double scale;
  const HloInstruction* bias_parameter;
  std::array<int64_t, 4> bias_strides;
};

std::optional<AttentionScoresMatch> MatchAttentionScores(
    const HloInstruction* instruction, const HloInstruction* dot,
    const Shape& score_shape,
    const std::array<int64_t, 4>& logical_score_shape) {
  if (std::optional<double> scale = MatchScaledDot(instruction, dot)) {
    return AttentionScoresMatch{*scale, nullptr, {0, 0, 0, 0}};
  }
  if (instruction->opcode() != HloOpcode::kAdd ||
      instruction->operand_count() != 2) {
    return std::nullopt;
  }
  for (int dot_operand = 0; dot_operand < 2; ++dot_operand) {
    std::optional<double> scale =
        MatchScaledDot(instruction->operand(dot_operand), dot);
    if (!scale.has_value()) {
      continue;
    }
    std::optional<BroadcastParameterMatch> bias = MatchBroadcastParameter(
        instruction->operand(1 - dot_operand), score_shape,
        logical_score_shape,
        /*predicate=*/false);
    if (!bias.has_value()) {
      continue;
    }
    return AttentionScoresMatch{*scale, bias->parameter, bias->strides};
  }
  return std::nullopt;
}

bool IsQkDot(const HloInstruction& dot, int64_t batch_heads,
             int64_t query_sequence, int64_t key_value_sequence,
             int64_t head_dimension) {
  if (dot.opcode() != HloOpcode::kDot || dot.operand_count() != 2 ||
      dot.shape().dimensions_size() != 3 ||
      dot.shape().dimensions(0) != batch_heads ||
      dot.shape().dimensions(1) != query_sequence ||
      dot.shape().dimensions(2) != key_value_sequence) {
    return false;
  }
  const auto& dimensions = dot.dot_dimension_numbers();
  if (!IsSingleDimension(dimensions.lhs_batch_dimensions(), 0) ||
      !IsSingleDimension(dimensions.rhs_batch_dimensions(), 0) ||
      !IsSingleDimension(dimensions.rhs_contracting_dimensions(), 1) ||
      dimensions.lhs_contracting_dimensions().size() != 1) {
    return false;
  }
  const int64_t lhs_contracting =
      dimensions.lhs_contracting_dimensions().Get(0);
  if (lhs_contracting != 1 && lhs_contracting != 2) {
    return false;
  }
  const Shape& lhs = dot.operand(0)->shape();
  const Shape& rhs = dot.operand(1)->shape();
  if (lhs.dimensions_size() != 3 || rhs.dimensions_size() != 3 ||
      lhs.dimensions(0) != batch_heads || rhs.dimensions(0) != batch_heads ||
      lhs.dimensions(lhs_contracting) != head_dimension ||
      rhs.dimensions(1) != head_dimension) {
    return false;
  }
  const int64_t lhs_sequence_dimension = lhs_contracting == 1 ? 2 : 1;
  return lhs.dimensions(lhs_sequence_dimension) == query_sequence &&
         rhs.dimensions(2) == key_value_sequence;
}

bool IsPvDot(const HloInstruction& dot, int64_t batch_heads,
             int64_t query_sequence, int64_t key_value_sequence,
             int64_t head_dimension) {
  if (dot.opcode() != HloOpcode::kDot || dot.operand_count() != 2 ||
      dot.shape().dimensions_size() != 3 ||
      dot.shape().dimensions(0) != batch_heads ||
      dot.shape().dimensions(1) != head_dimension ||
      dot.shape().dimensions(2) != query_sequence) {
    return false;
  }
  const auto& dimensions = dot.dot_dimension_numbers();
  if (!IsSingleDimension(dimensions.lhs_batch_dimensions(), 0) ||
      !IsSingleDimension(dimensions.rhs_batch_dimensions(), 0) ||
      !IsSingleDimension(dimensions.lhs_contracting_dimensions(), 2) ||
      !IsSingleDimension(dimensions.rhs_contracting_dimensions(), 2)) {
    return false;
  }
  const Shape& lhs = dot.operand(0)->shape();
  const Shape& rhs = dot.operand(1)->shape();
  return lhs.dimensions_size() == 3 && rhs.dimensions_size() == 3 &&
         lhs.dimensions(0) == batch_heads &&
         lhs.dimensions(1) == head_dimension &&
         lhs.dimensions(2) == key_value_sequence &&
         rhs.dimensions(0) == batch_heads &&
         rhs.dimensions(1) == query_sequence &&
         rhs.dimensions(2) == key_value_sequence;
}

bool IsGroupedQkDot(const HloInstruction& dot, int64_t batch_key_value_heads,
                    int64_t sequence, int64_t grouped_sequence,
                    int64_t head_dimension) {
  if (dot.opcode() != HloOpcode::kDot || dot.operand_count() != 2 ||
      dot.shape().dimensions_size() != 3 ||
      dot.shape().dimensions(0) != batch_key_value_heads ||
      dot.shape().dimensions(1) != sequence ||
      dot.shape().dimensions(2) != grouped_sequence) {
    return false;
  }
  const auto& dimensions = dot.dot_dimension_numbers();
  if (!IsSingleDimension(dimensions.lhs_batch_dimensions(), 0) ||
      !IsSingleDimension(dimensions.rhs_batch_dimensions(), 0) ||
      !IsSingleDimension(dimensions.lhs_contracting_dimensions(), 2) ||
      !IsSingleDimension(dimensions.rhs_contracting_dimensions(), 1)) {
    return false;
  }
  const Shape& lhs = dot.operand(0)->shape();
  const Shape& rhs = dot.operand(1)->shape();
  return lhs.dimensions_size() == 3 && rhs.dimensions_size() == 3 &&
         lhs.dimensions(0) == batch_key_value_heads &&
         lhs.dimensions(1) == sequence && lhs.dimensions(2) == head_dimension &&
         rhs.dimensions(0) == batch_key_value_heads &&
         rhs.dimensions(1) == head_dimension &&
         rhs.dimensions(2) == grouped_sequence;
}

bool IsGroupedPvDot(const HloInstruction& dot, int64_t batch_key_value_heads,
                    int64_t sequence, int64_t grouped_sequence,
                    int64_t head_dimension) {
  if (dot.opcode() != HloOpcode::kDot || dot.operand_count() != 2 ||
      dot.shape().dimensions_size() != 3 ||
      dot.shape().dimensions(0) != batch_key_value_heads ||
      dot.shape().dimensions(1) != head_dimension ||
      dot.shape().dimensions(2) != grouped_sequence) {
    return false;
  }
  const auto& dimensions = dot.dot_dimension_numbers();
  if (!IsSingleDimension(dimensions.lhs_batch_dimensions(), 0) ||
      !IsSingleDimension(dimensions.rhs_batch_dimensions(), 0) ||
      !IsSingleDimension(dimensions.lhs_contracting_dimensions(), 2) ||
      !IsSingleDimension(dimensions.rhs_contracting_dimensions(), 1)) {
    return false;
  }
  const Shape& lhs = dot.operand(0)->shape();
  const Shape& rhs = dot.operand(1)->shape();
  return lhs.dimensions_size() == 3 && rhs.dimensions_size() == 3 &&
         lhs.dimensions(0) == batch_key_value_heads &&
         lhs.dimensions(1) == head_dimension && lhs.dimensions(2) == sequence &&
         rhs.dimensions(0) == batch_key_value_heads &&
         rhs.dimensions(1) == sequence && rhs.dimensions(2) == grouped_sequence;
}

std::optional<FlyAttentionDescriptor> MatchGroupedAttentionDescriptor(
    const HloInstruction* root, const Shape& output,
    PrimitiveType element_type) {
  const int64_t batch = output.dimensions(0);
  const int64_t sequence = output.dimensions(1);
  if (batch <= 0 || sequence < 64 || sequence % 64 != 0) {
    return std::nullopt;
  }

  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> dots;
  std::vector<const HloInstruction*> parameters;
  CollectGraph(root, visited, dots, parameters);
  if (dots.size() != 2 || parameters.size() != 1) {
    return std::nullopt;
  }

  for (const HloInstruction* pv_dot : dots) {
    if (pv_dot->shape().dimensions_size() != 3) {
      continue;
    }
    const int64_t batch_key_value_heads = pv_dot->shape().dimensions(0);
    const int64_t head_dimension = pv_dot->shape().dimensions(1);
    const int64_t grouped_sequence = pv_dot->shape().dimensions(2);
    if (batch_key_value_heads <= 0 || batch_key_value_heads % batch != 0 ||
        head_dimension < 64 || head_dimension % 32 != 0 ||
        grouped_sequence % sequence != 0) {
      continue;
    }
    const int64_t key_value_heads = batch_key_value_heads / batch;
    const int64_t group_size = grouped_sequence / sequence;
    const int64_t query_heads = key_value_heads * group_size;
    if (key_value_heads <= 0 || group_size <= 1 ||
        output.dimensions(2) != query_heads * head_dimension ||
        !IsGroupedPvDot(*pv_dot, batch_key_value_heads, sequence,
                        grouped_sequence, head_dimension)) {
      continue;
    }

    const HloInstruction* qk_dot = dots[0] == pv_dot ? dots[1] : dots[0];
    if (!IsGroupedQkDot(*qk_dot, batch_key_value_heads, sequence,
                        grouped_sequence, head_dimension) ||
        qk_dot->shape().element_type() != element_type ||
        pv_dot->shape().element_type() != element_type) {
      continue;
    }

    const HloInstruction* softmax_root =
        StripDataMovementViews(pv_dot->operand(1));
    const HloInstruction* softmax_input =
        GetFlyCompoundSoftmaxInputAlongDimension(*softmax_root,
                                                 /*reduction_dimension=*/2);
    if (softmax_input == nullptr ||
        !ContainsInstruction(softmax_input, qk_dot)) {
      continue;
    }
    const HloInstruction* scaled_scores = GetFlyCausalMaskScoresAlongDimensions(
        *softmax_input, sequence, /*query_dimension=*/3,
        /*key_dimension=*/2);
    const bool causal = scaled_scores != nullptr;
    if (!causal) {
      scaled_scores = softmax_input;
    }
    std::optional<double> scale = MatchScaledDot(scaled_scores, qk_dot);
    if (!scale.has_value() || *scale <= 0.0) {
      continue;
    }

    const HloInstruction* qkv = parameters.front();
    const int64_t packed_width =
        (query_heads + 2 * key_value_heads) * head_dimension;
    if (qkv->shape().element_type() != element_type ||
        qkv->shape().dimensions_size() != 2 || !qkv->shape().has_layout() ||
        qkv->shape().layout().minor_to_major(0) != 1 ||
        qkv->shape().dimensions(0) != batch * sequence ||
        qkv->shape().dimensions(1) != packed_width) {
      continue;
    }

    return FlyAttentionDescriptor{batch,
                                  sequence,
                                  sequence,
                                  query_heads,
                                  key_value_heads,
                                  head_dimension,
                                  *scale,
                                  causal,
                                  element_type,
                                  qkv,
                                  qkv,
                                  nullptr,
                                  {0, 0, 0, 0},
                                  nullptr,
                                  {0, 0, 0, 0},
                                  qk_dot,
                                  pv_dot,
                                  softmax_root};
  }
  return std::nullopt;
}

std::optional<FlyAttentionDescriptor> MatchCrossAttentionDescriptor(
    const HloInstruction* root, const Shape& output,
    PrimitiveType element_type) {
  const int64_t batch = output.dimensions(0);
  const int64_t query_sequence = output.dimensions(1);
  const int64_t heads = output.dimensions(2);
  const int64_t head_dimension = output.dimensions(3);
  if (batch <= 0 || query_sequence < 64 || query_sequence % 64 != 0 ||
      heads <= 0 || head_dimension < 64 || head_dimension % 32 != 0) {
    return std::nullopt;
  }

  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> dots;
  std::vector<const HloInstruction*> parameters;
  CollectGraph(root, visited, dots, parameters);
  if (dots.size() != 2 || parameters.size() != 2) {
    return std::nullopt;
  }

  const int64_t batch_heads = batch * heads;
  for (const HloInstruction* qk_dot : dots) {
    if (qk_dot->shape().dimensions_size() != 3 ||
        qk_dot->shape().dimensions(0) != batch_heads ||
        qk_dot->shape().dimensions(1) != query_sequence) {
      continue;
    }
    const int64_t key_value_sequence = qk_dot->shape().dimensions(2);
    if (key_value_sequence < 64 || key_value_sequence % 64 != 0 ||
        key_value_sequence == query_sequence ||
        !IsQkDot(*qk_dot, batch_heads, query_sequence, key_value_sequence,
                 head_dimension)) {
      continue;
    }
    const HloInstruction* pv_dot = dots[0] == qk_dot ? dots[1] : dots[0];
    if (!IsPvDot(*pv_dot, batch_heads, query_sequence, key_value_sequence,
                 head_dimension) ||
        qk_dot->shape().element_type() != element_type ||
        pv_dot->shape().element_type() != element_type) {
      continue;
    }

    const HloInstruction* softmax_root = StripViews(pv_dot->operand(1));
    const HloInstruction* softmax_input =
        GetFlyCompoundSoftmaxInput(*softmax_root);
    if (softmax_input == nullptr ||
        !ContainsInstruction(softmax_input, qk_dot)) {
      continue;
    }
    std::optional<double> scale = MatchScaledDot(softmax_input, qk_dot);
    if (!scale.has_value() || *scale <= 0.0) {
      continue;
    }

    const HloInstruction* query_parameter =
        FindUniqueParameter(qk_dot->operand(0));
    const HloInstruction* key_value_parameter =
        FindUniqueParameter(qk_dot->operand(1));
    if (query_parameter == nullptr || key_value_parameter == nullptr) {
      continue;
    }
    const Shape& query_shape = query_parameter->shape();
    const bool direct_query =
        query_shape.dimensions_size() == 2 &&
        query_shape.dimensions(0) == batch * query_sequence &&
        query_shape.dimensions(1) == heads * head_dimension;
    const bool split_k_query =
        query_shape.element_type() == F32 &&
        query_shape.dimensions_size() == 3 && query_shape.dimensions(0) >= 2 &&
        query_shape.dimensions(0) <= 8 &&
        query_shape.dimensions(1) == batch * query_sequence &&
        query_shape.dimensions(2) == heads * head_dimension;
    if (query_parameter == key_value_parameter ||
        FindUniqueParameter(pv_dot->operand(0)) != key_value_parameter ||
        (query_parameter->shape().element_type() != element_type &&
         query_parameter->shape().element_type() != F32) ||
        (!direct_query && !split_k_query) || !query_shape.has_layout() ||
        !LayoutUtil::IsMonotonicWithDim0Major(query_shape.layout()) ||
        key_value_parameter->shape().element_type() != element_type ||
        key_value_parameter->shape().dimensions_size() != 2 ||
        !key_value_parameter->shape().has_layout() ||
        !LayoutUtil::IsMonotonicWithDim0Major(
            key_value_parameter->shape().layout()) ||
        key_value_parameter->shape().dimensions(0) !=
            batch * key_value_sequence ||
        key_value_parameter->shape().dimensions(1) !=
            2 * heads * head_dimension) {
      continue;
    }

    return FlyAttentionDescriptor{batch,
                                  query_sequence,
                                  key_value_sequence,
                                  heads,
                                  heads,
                                  head_dimension,
                                  *scale,
                                  false,
                                  element_type,
                                  query_parameter,
                                  key_value_parameter,
                                  nullptr,
                                  {0, 0, 0, 0},
                                  nullptr,
                                  {0, 0, 0, 0},
                                  qk_dot,
                                  pv_dot,
                                  softmax_root};
  }
  return std::nullopt;
}

}  // namespace

const HloInstruction* GetFlyCausalMaskScoresAlongDimensions(
    const HloInstruction& input, int64_t sequence, int64_t query_dimension,
    int64_t key_dimension) {
  const int64_t rank = input.shape().dimensions_size();
  if (input.opcode() != HloOpcode::kSelect || input.operand_count() != 3 ||
      rank < 2 || query_dimension < 0 || query_dimension >= rank ||
      key_dimension < 0 || key_dimension >= rank ||
      query_dimension == key_dimension ||
      input.shape().dimensions(query_dimension) != sequence ||
      input.shape().dimensions(key_dimension) != sequence) {
    return nullptr;
  }
  std::optional<double> masked_value = UniformConstant(input.operand(2));
  if (!masked_value.has_value() || !std::isinf(*masked_value) ||
      *masked_value >= 0.0 ||
      !ShapeUtil::Compatible(input.shape(), input.operand(1)->shape())) {
    return nullptr;
  }

  std::vector<int64_t> logical_dimensions(rank, -1);
  logical_dimensions[query_dimension] = 2;
  logical_dimensions[key_dimension] = 3;
  return HasCausalPredicateFromDimensions(input.operand(0), logical_dimensions,
                                          sequence)
             ? input.operand(1)
             : nullptr;
}

const HloInstruction* GetFlyCausalMaskScores(const HloInstruction& input,
                                             int64_t sequence) {
  const int64_t rank = input.shape().dimensions_size();
  if (rank < 2) {
    return nullptr;
  }
  return GetFlyCausalMaskScoresAlongDimensions(input, sequence,
                                               /*query_dimension=*/rank - 2,
                                               /*key_dimension=*/rank - 1);
}

std::optional<FlyAttentionDescriptor> GetFlyAttentionDescriptor(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return std::nullopt;
  }
  const HloInstruction* root = &analysis.fusion_root(0).instruction();
  const Shape& output = root->shape();
  const PrimitiveType element_type = output.element_type();
  if ((element_type != BF16 && element_type != F16) ||
      (output.dimensions_size() != 3 && output.dimensions_size() != 4) ||
      !output.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout())) {
    return std::nullopt;
  }
  if (output.dimensions_size() == 3) {
    return MatchGroupedAttentionDescriptor(root, output, element_type);
  }
  if (std::optional<FlyAttentionDescriptor> cross_attention =
          MatchCrossAttentionDescriptor(root, output, element_type)) {
    return cross_attention;
  }
  const int64_t batch = output.dimensions(0);
  const int64_t sequence = output.dimensions(1);
  const int64_t heads = output.dimensions(2);
  const int64_t head_dimension = output.dimensions(3);
  if (batch <= 0 || sequence < 64 || sequence % 64 != 0 || heads <= 0 ||
      head_dimension < 64 || head_dimension % 32 != 0) {
    return std::nullopt;
  }

  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> dots;
  std::vector<const HloInstruction*> parameters;
  CollectGraph(root, visited, dots, parameters);
  if (dots.size() != 2 || parameters.empty() || parameters.size() > 3) {
    return std::nullopt;
  }

  const int64_t batch_heads = batch * heads;
  const HloInstruction* qk_dot = nullptr;
  const HloInstruction* pv_dot = nullptr;
  for (const HloInstruction* dot : dots) {
    if (IsQkDot(*dot, batch_heads, sequence, sequence, head_dimension)) {
      qk_dot = dot;
    }
    if (IsPvDot(*dot, batch_heads, sequence, sequence, head_dimension)) {
      pv_dot = dot;
    }
  }
  if (qk_dot == nullptr || pv_dot == nullptr || qk_dot == pv_dot ||
      qk_dot->shape().element_type() != element_type ||
      pv_dot->shape().element_type() != element_type) {
    return std::nullopt;
  }

  const HloInstruction* softmax_root = StripViews(pv_dot->operand(1));
  const HloInstruction* softmax_input =
      GetFlyCompoundSoftmaxInput(*softmax_root);
  if (softmax_input == nullptr || !ContainsInstruction(softmax_input, qk_dot)) {
    return std::nullopt;
  }
  const std::array<int64_t, 4> logical_score_shape = {
      batch, heads, sequence, sequence};
  std::optional<AttentionMaskMatch> mask = MatchAttentionMask(
      softmax_input, softmax_input->shape(), logical_score_shape);
  const bool causal = mask.has_value() && mask->causal;
  const HloInstruction* scaled_scores =
      mask.has_value() ? softmax_input->operand(1) : softmax_input;
  // XLA can remove an intermediate BF16 round trip and leave the QK scale in
  // F32. The native attention kernel accumulates and applies both the scale
  // and an optional canonical broadcast bias in F32.
  std::optional<AttentionScoresMatch> scores =
      MatchAttentionScores(scaled_scores, qk_dot, softmax_input->shape(),
                           logical_score_shape);
  if (!scores.has_value() || scores->scale <= 0.0) {
    return std::nullopt;
  }

  const HloInstruction* qkv = nullptr;
  for (const HloInstruction* parameter : parameters) {
    if (parameter->shape().element_type() == element_type &&
        parameter->shape().dimensions_size() == 2 &&
        parameter->shape().has_layout() &&
        parameter->shape().layout().minor_to_major(0) == 1 &&
        parameter->shape().dimensions(0) == batch * sequence &&
        parameter->shape().dimensions(1) == 3 * heads * head_dimension) {
      if (qkv != nullptr) {
        return std::nullopt;
      }
      qkv = parameter;
    }
  }
  const int64_t expected_parameter_count =
      1 + (scores->bias_parameter == nullptr ? 0 : 1) +
      (mask.has_value() && mask->external.has_value() ? 1 : 0);
  if (qkv == nullptr || parameters.size() != expected_parameter_count ||
      (scores->bias_parameter != nullptr &&
       scores->bias_parameter == qkv) ||
      (mask.has_value() && mask->external.has_value() &&
       (mask->external->parameter == qkv ||
        mask->external->parameter == scores->bias_parameter))) {
    return std::nullopt;
  }
  for (const HloInstruction* parameter : parameters) {
    if (parameter != qkv && parameter != scores->bias_parameter &&
        (!mask.has_value() || !mask->external.has_value() ||
         parameter != mask->external->parameter)) {
      return std::nullopt;
    }
  }

  return FlyAttentionDescriptor{
      batch,  sequence,    sequence,     heads, heads, head_dimension,
      scores->scale, causal, element_type, qkv, qkv, scores->bias_parameter,
      scores->bias_strides,
      mask.has_value() && mask->external.has_value()
          ? mask->external->parameter
          : nullptr,
      mask.has_value() && mask->external.has_value()
          ? mask->external->strides
          : std::array<int64_t, 4>{0, 0, 0, 0},
      qk_dot, pv_dot, softmax_root};
}

}  // namespace xla::gpu::flydsl
