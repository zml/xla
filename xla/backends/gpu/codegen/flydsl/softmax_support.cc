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

#include "xla/backends/gpu/codegen/flydsl/softmax_support.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::flydsl {
namespace {

bool IsScalarConstant(const HloInstruction* instruction, double value) {
  if (instruction->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(instruction->shape())) {
    return false;
  }
  return instruction->literal().GetAsDouble({}) == value;
}

bool SameNonUnitDimensions(const Shape& lhs, const Shape& rhs) {
  std::vector<int64_t> lhs_dimensions;
  std::vector<int64_t> rhs_dimensions;
  for (int64_t dimension : lhs.dimensions()) {
    if (dimension != 1) {
      lhs_dimensions.push_back(dimension);
    }
  }
  for (int64_t dimension : rhs.dimensions()) {
    if (dimension != 1) {
      rhs_dimensions.push_back(dimension);
    }
  }
  return lhs_dimensions == rhs_dimensions;
}

bool IsUnitDimensionViewOf(const HloInstruction* view,
                           const HloInstruction* source) {
  while (view != source) {
    if (view->operand_count() != 1 ||
        (view->opcode() != HloOpcode::kBitcast &&
         view->opcode() != HloOpcode::kReshape) ||
        view->shape().element_type() != source->shape().element_type() ||
        ShapeUtil::ElementsIn(view->shape()) !=
            ShapeUtil::ElementsIn(source->shape()) ||
        !SameNonUnitDimensions(view->shape(), source->shape())) {
      return false;
    }
    view = view->operand(0);
  }
  return true;
}

const HloInstruction* GetRowReduction(const HloInstruction* reduction,
                                      HloOpcode reducer_opcode,
                                      const HloInstruction* input) {
  while (reduction->opcode() == HloOpcode::kBitcast &&
         reduction->operand_count() == 1) {
    reduction = reduction->operand(0);
  }
  while (input->opcode() == HloOpcode::kBitcast &&
         input->operand_count() == 1) {
    input = input->operand(0);
  }
  const int64_t input_rank = input->shape().dimensions_size();
  const HloInstruction* reduction_input =
      reduction->operand_count() == 0 ? nullptr : reduction->operand(0);
  const int64_t reduction_input_rank =
      reduction_input == nullptr ? 0
                                 : reduction_input->shape().dimensions_size();
  if (reduction->opcode() != HloOpcode::kReduce ||
      reduction->operand_count() != 2 ||
      !IsUnitDimensionViewOf(reduction_input, input) ||
      reduction->dimensions().size() != 1 || input_rank < 2 ||
      reduction_input_rank < 2 ||
      reduction->dimensions(0) != reduction_input_rank - 1 ||
      reduction_input->shape().dimensions(reduction_input_rank - 1) !=
          input->shape().dimensions(input_rank - 1) ||
      !SameNonUnitDimensions(
          reduction->shape(),
          ShapeUtil::DeleteDimension(input_rank - 1, input->shape())) ||
      reduction->called_computations().size() != 1 ||
      reduction->called_computations().front()->root_instruction()->opcode() !=
          reducer_opcode) {
    return nullptr;
  }
  return reduction;
}

const HloInstruction* BroadcastOperand(const HloInstruction* instruction,
                                       const Shape& output_shape) {
  const int64_t output_rank = output_shape.dimensions_size();
  if (instruction->opcode() != HloOpcode::kBroadcast ||
      !ShapeUtil::Compatible(instruction->shape(), output_shape) ||
      instruction->dimensions().size() !=
          instruction->operand(0)->shape().dimensions_size()) {
    return nullptr;
  }
  std::vector<bool> mapped(output_rank, false);
  int64_t previous_output_dimension = -1;
  for (int64_t dimension = 0; dimension < instruction->dimensions().size();
       ++dimension) {
    const int64_t output_dimension = instruction->dimensions(dimension);
    if (output_dimension <= previous_output_dimension || output_dimension < 0 ||
        output_dimension >= output_rank ||
        instruction->operand(0)->shape().dimensions(dimension) !=
            output_shape.dimensions(output_dimension)) {
      return nullptr;
    }
    mapped[output_dimension] = true;
    previous_output_dimension = output_dimension;
  }
  // The final dimension is the row reduction. Any additional omitted
  // dimensions must be unit dimensions folded out by layout simplification.
  if (mapped[output_rank - 1]) {
    return nullptr;
  }
  for (int64_t dimension = 0; dimension < output_rank - 1; ++dimension) {
    if (!mapped[dimension] && output_shape.dimensions(dimension) != 1) {
      return nullptr;
    }
  }
  return instruction->operand(0);
}

// Matches `x - broadcast(reduce_max(x, -inf))` and returns x. JAX commonly
// emits this stable shift twice when an explicitly stabilized input is passed
// to jax.nn.softmax.
const HloInstruction* StableShiftInput(const HloInstruction* shifted) {
  if (shifted->opcode() != HloOpcode::kSubtract ||
      shifted->operand_count() != 2) {
    return nullptr;
  }
  const HloInstruction* input = shifted->operand(0);
  const HloInstruction* row_max =
      BroadcastOperand(shifted->operand(1), input->shape());
  const HloInstruction* reduction =
      row_max == nullptr ? nullptr
                         : GetRowReduction(row_max, HloOpcode::kMaximum, input);
  if (reduction == nullptr ||
      !IsScalarConstant(reduction->operand(1),
                        -std::numeric_limits<double>::infinity())) {
    return nullptr;
  }
  return input;
}

const HloInstruction* GetStableSoftmaxF32Input(const HloInstruction& root) {
  const PrimitiveType element_type = root.shape().element_type();
  if ((element_type != F16 && element_type != BF16 && element_type != F32) ||
      root.shape().dimensions_size() < 2) {
    return nullptr;
  }
  const HloInstruction* normalized = &root;
  if (element_type != F32) {
    if (root.opcode() != HloOpcode::kConvert || root.operand_count() != 1 ||
        root.operand(0)->shape().element_type() != F32) {
      return nullptr;
    }
    normalized = root.operand(0);
  }
  if (normalized->opcode() != HloOpcode::kDivide) {
    return nullptr;
  }
  const HloInstruction* exponential = normalized->operand(0);
  const HloInstruction* row_sum =
      BroadcastOperand(normalized->operand(1), exponential->shape());
  const HloInstruction* sum_reduction =
      row_sum == nullptr
          ? nullptr
          : GetRowReduction(row_sum, HloOpcode::kAdd, exponential);
  if (exponential->opcode() != HloOpcode::kExp || row_sum == nullptr ||
      sum_reduction == nullptr ||
      !IsScalarConstant(sum_reduction->operand(1), 0.0)) {
    return nullptr;
  }
  const HloInstruction* input = StableShiftInput(exponential->operand(0));
  if (input == nullptr) {
    return nullptr;
  }
  // Accept the second stabilization produced by `x - max(x); softmax(x)`.
  // The native kernel computes the equivalent single stable softmax directly.
  if (const HloInstruction* original = StableShiftInput(input)) {
    input = original;
  }
  return input;
}

struct ExternalRowSoftmaxInputs {
  const HloInstruction* input;
  const HloInstruction* row_offset;
  bool recompute_maximum;
};

std::optional<ExternalRowSoftmaxInputs> GetExternalRowSoftmaxF32Inputs(
    const HloInstruction& root) {
  const PrimitiveType element_type = root.shape().element_type();
  if ((element_type != F16 && element_type != BF16 && element_type != F32) ||
      root.shape().dimensions_size() < 2) {
    return std::nullopt;
  }
  const HloInstruction* normalized = &root;
  if (element_type != F32) {
    if (root.opcode() != HloOpcode::kConvert || root.operand_count() != 1 ||
        root.operand(0)->shape().element_type() != F32) {
      return std::nullopt;
    }
    normalized = root.operand(0);
  }
  if (normalized->opcode() != HloOpcode::kDivide ||
      normalized->operand_count() != 2) {
    return std::nullopt;
  }
  const HloInstruction* exponential = normalized->operand(0);
  const HloInstruction* row_sum =
      BroadcastOperand(normalized->operand(1), exponential->shape());
  const HloInstruction* sum_reduction =
      row_sum == nullptr
          ? nullptr
          : GetRowReduction(row_sum, HloOpcode::kAdd, exponential);
  if (exponential->opcode() != HloOpcode::kExp || sum_reduction == nullptr ||
      !IsScalarConstant(sum_reduction->operand(1), 0.0)) {
    return std::nullopt;
  }
  const HloInstruction* pre_shift = exponential->operand(0);
  bool recompute_maximum = false;
  if (const HloInstruction* stabilized_input = StableShiftInput(pre_shift)) {
    pre_shift = stabilized_input;
    recompute_maximum = true;
  }
  if (pre_shift->opcode() != HloOpcode::kSubtract ||
      pre_shift->operand_count() != 2) {
    return std::nullopt;
  }
  const HloInstruction* input = pre_shift->operand(0);
  const HloInstruction* row_offset =
      BroadcastOperand(pre_shift->operand(1), input->shape());
  const int64_t input_rank = input->shape().dimensions_size();
  if (row_offset == nullptr || row_offset->opcode() != HloOpcode::kParameter ||
      row_offset->shape().element_type() != F32 || input_rank < 2 ||
      !ShapeUtil::SameDimensions(
          row_offset->shape(),
          ShapeUtil::DeleteDimension(input_rank - 1, input->shape()))) {
    return std::nullopt;
  }
  return ExternalRowSoftmaxInputs{input, row_offset, recompute_maximum};
}

bool HasCompatibleSoftmaxShape(const HloInstruction& root,
                               const HloInstruction& input) {
  if (ShapeUtil::ElementsIn(root.shape()) !=
      ShapeUtil::ElementsIn(input.shape())) {
    return false;
  }
  const int64_t columns =
      root.shape().dimensions(root.shape().dimensions_size() - 1);
  if (input.shape().dimensions_size() < 2 ||
      input.shape().dimensions(input.shape().dimensions_size() - 1) !=
          columns) {
    return false;
  }
  constexpr int64_t kMaxColumns = 16 * 64 * 64;
  return columns > 0 && columns <= kMaxColumns;
}

const HloInstruction* GetReductionAlongDimension(
    const HloInstruction* reduction, HloOpcode reducer_opcode,
    const HloInstruction* input, int64_t reduction_dimension) {
  while (reduction->opcode() == HloOpcode::kBitcast &&
         reduction->operand_count() == 1) {
    reduction = reduction->operand(0);
  }
  while (input->opcode() == HloOpcode::kBitcast &&
         input->operand_count() == 1) {
    input = input->operand(0);
  }
  const int64_t input_rank = input->shape().dimensions_size();
  if (reduction_dimension < 0 || reduction_dimension >= input_rank ||
      reduction->opcode() != HloOpcode::kReduce ||
      reduction->operand_count() != 2 || reduction->operand(0) != input ||
      reduction->dimensions().size() != 1 ||
      reduction->dimensions(0) != reduction_dimension ||
      reduction->shape().dimensions_size() != input_rank - 1 ||
      !ShapeUtil::SameDimensions(
          reduction->shape(),
          ShapeUtil::DeleteDimension(reduction_dimension, input->shape())) ||
      reduction->called_computations().size() != 1 ||
      reduction->called_computations().front()->root_instruction()->opcode() !=
          reducer_opcode) {
    return nullptr;
  }
  return reduction;
}

const HloInstruction* BroadcastOperandAlongDimension(
    const HloInstruction* instruction, const Shape& output_shape,
    int64_t reduction_dimension) {
  const int64_t output_rank = output_shape.dimensions_size();
  if (reduction_dimension < 0 || reduction_dimension >= output_rank ||
      instruction->opcode() != HloOpcode::kBroadcast ||
      !ShapeUtil::Compatible(instruction->shape(), output_shape) ||
      instruction->dimensions().size() != output_rank - 1) {
    return nullptr;
  }
  int64_t operand_dimension = 0;
  for (int64_t output_dimension = 0; output_dimension < output_rank;
       ++output_dimension) {
    if (output_dimension == reduction_dimension) {
      continue;
    }
    if (instruction->dimensions(operand_dimension) != output_dimension) {
      return nullptr;
    }
    ++operand_dimension;
  }
  return instruction->operand(0);
}

const HloInstruction* StableShiftInputAlongDimension(
    const HloInstruction* shifted, int64_t reduction_dimension) {
  if (shifted->opcode() != HloOpcode::kSubtract ||
      shifted->operand_count() != 2) {
    return nullptr;
  }
  const HloInstruction* input = shifted->operand(0);
  const HloInstruction* row_max = BroadcastOperandAlongDimension(
      shifted->operand(1), input->shape(), reduction_dimension);
  const HloInstruction* reduction =
      row_max == nullptr
          ? nullptr
          : GetReductionAlongDimension(row_max, HloOpcode::kMaximum, input,
                                       reduction_dimension);
  if (reduction == nullptr ||
      !IsScalarConstant(reduction->operand(1),
                        -std::numeric_limits<double>::infinity())) {
    return nullptr;
  }
  return input;
}

const HloInstruction* GetStableSoftmaxF32InputAlongDimension(
    const HloInstruction& root, int64_t reduction_dimension) {
  const PrimitiveType element_type = root.shape().element_type();
  const int64_t rank = root.shape().dimensions_size();
  if ((element_type != F16 && element_type != BF16 && element_type != F32) ||
      rank < 2 || reduction_dimension < 0 || reduction_dimension >= rank) {
    return nullptr;
  }
  const HloInstruction* normalized = &root;
  if (element_type != F32) {
    if (root.opcode() != HloOpcode::kConvert || root.operand_count() != 1 ||
        root.operand(0)->shape().element_type() != F32) {
      return nullptr;
    }
    normalized = root.operand(0);
  }
  if (normalized->opcode() != HloOpcode::kDivide ||
      normalized->operand_count() != 2) {
    return nullptr;
  }
  const HloInstruction* exponential = normalized->operand(0);
  const HloInstruction* row_sum = BroadcastOperandAlongDimension(
      normalized->operand(1), exponential->shape(), reduction_dimension);
  const HloInstruction* sum_reduction =
      row_sum == nullptr
          ? nullptr
          : GetReductionAlongDimension(row_sum, HloOpcode::kAdd, exponential,
                                       reduction_dimension);
  if (exponential->opcode() != HloOpcode::kExp || sum_reduction == nullptr ||
      !IsScalarConstant(sum_reduction->operand(1), 0.0)) {
    return nullptr;
  }
  const HloInstruction* input = StableShiftInputAlongDimension(
      exponential->operand(0), reduction_dimension);
  if (input == nullptr) {
    return nullptr;
  }
  if (const HloInstruction* original =
          StableShiftInputAlongDimension(input, reduction_dimension)) {
    input = original;
  }
  return input;
}

bool HasCompatibleSoftmaxShapeAlongDimension(const HloInstruction& root,
                                             const HloInstruction& input,
                                             int64_t reduction_dimension) {
  const int64_t rank = root.shape().dimensions_size();
  if (rank != input.shape().dimensions_size() || reduction_dimension < 0 ||
      reduction_dimension >= rank ||
      ShapeUtil::ElementsIn(root.shape()) !=
          ShapeUtil::ElementsIn(input.shape()) ||
      root.shape().dimensions(reduction_dimension) !=
          input.shape().dimensions(reduction_dimension)) {
    return false;
  }
  constexpr int64_t kMaxColumns = 16 * 64 * 64;
  const int64_t columns = root.shape().dimensions(reduction_dimension);
  return columns > 0 && columns <= kMaxColumns;
}

std::optional<double> ScalarBroadcastValue(const HloInstruction* instruction,
                                           const Shape& output_shape) {
  if (instruction->opcode() != HloOpcode::kBroadcast ||
      instruction->dimensions().size() != 0 ||
      !ShapeUtil::Compatible(instruction->shape(), output_shape) ||
      instruction->operand_count() != 1 ||
      instruction->operand(0)->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(instruction->operand(0)->shape())) {
    return std::nullopt;
  }
  return instruction->operand(0)->literal().GetAsDouble({});
}

const HloInstruction* OtherBinaryOperand(const HloInstruction* instruction,
                                         HloOpcode opcode,
                                         const HloInstruction* operand) {
  if (instruction->opcode() != opcode || instruction->operand_count() != 2) {
    return nullptr;
  }
  if (instruction->operand(0) == operand) {
    return instruction->operand(1);
  }
  if (instruction->operand(1) == operand) {
    return instruction->operand(0);
  }
  return nullptr;
}

void CollectMultiplyLeaves(const HloInstruction* instruction,
                           std::vector<const HloInstruction*>& leaves) {
  if (instruction->opcode() == HloOpcode::kMultiply &&
      instruction->operand_count() == 2) {
    CollectMultiplyLeaves(instruction->operand(0), leaves);
    CollectMultiplyLeaves(instruction->operand(1), leaves);
    return;
  }
  leaves.push_back(instruction);
}

bool IsProductOf(const HloInstruction* instruction, const HloInstruction* lhs,
                 const HloInstruction* rhs, const HloInstruction* third) {
  std::vector<const HloInstruction*> leaves;
  CollectMultiplyLeaves(instruction, leaves);
  if (leaves.size() != 3) {
    return false;
  }
  for (const HloInstruction* expected : {lhs, rhs, third}) {
    auto it = std::find(leaves.begin(), leaves.end(), expected);
    if (it == leaves.end()) {
      return false;
    }
    leaves.erase(it);
  }
  return leaves.empty();
}

const HloInstruction* SoftmaxInterfaceValue(const HloInstruction* instruction) {
  bool peeled_conversion = false;
  while (instruction->operand_count() == 1) {
    const HloInstruction* operand = instruction->operand(0);
    if ((instruction->opcode() == HloOpcode::kBitcast ||
         instruction->opcode() == HloOpcode::kReshape) &&
        instruction->shape().element_type() ==
            operand->shape().element_type() &&
        ShapeUtil::ElementsIn(instruction->shape()) ==
            ShapeUtil::ElementsIn(operand->shape())) {
      instruction = operand;
      continue;
    }
    if (!peeled_conversion && instruction->opcode() == HloOpcode::kConvert &&
        instruction->shape().element_type() == F32 &&
        (operand->shape().element_type() == F16 ||
         operand->shape().element_type() == BF16 ||
         operand->shape().element_type() == F32) &&
        ShapeUtil::ElementsIn(instruction->shape()) ==
            ShapeUtil::ElementsIn(operand->shape())) {
      peeled_conversion = true;
      instruction = operand;
      continue;
    }
    break;
  }
  return instruction;
}

}  // namespace

std::optional<FlySoftmaxBackwardDescriptor> GetFlySoftmaxBackwardDescriptor(
    const HloInstruction& root) {
  const PrimitiveType output_type = root.shape().element_type();
  if ((output_type != F16 && output_type != BF16 && output_type != F32) ||
      root.shape().dimensions_size() < 2) {
    return std::nullopt;
  }

  const HloInstruction* derivative = &root;
  if (output_type != F32) {
    if (root.opcode() != HloOpcode::kConvert || root.operand_count() != 1 ||
        root.operand(0)->shape().element_type() != F32) {
      return std::nullopt;
    }
    derivative = root.operand(0);
  }
  if (derivative->opcode() != HloOpcode::kMultiply ||
      derivative->operand_count() != 2) {
    return std::nullopt;
  }

  // The derivative with respect to unscaled scores ends in multiplication by
  // the same scalar that produced the stable-softmax logits.
  const HloInstruction* scale_broadcast = nullptr;
  const HloInstruction* softmax_derivative = nullptr;
  for (int64_t operand = 0; operand < 2; ++operand) {
    if (ScalarBroadcastValue(derivative->operand(operand), derivative->shape())
            .has_value()) {
      scale_broadcast = derivative->operand(operand);
      softmax_derivative = derivative->operand(1 - operand);
      break;
    }
  }
  if (scale_broadcast == nullptr ||
      softmax_derivative->opcode() != HloOpcode::kMultiply) {
    return std::nullopt;
  }
  const double scale =
      *ScalarBroadcastValue(scale_broadcast, derivative->shape());
  if (!std::isfinite(scale)) {
    return std::nullopt;
  }

  const HloInstruction* exponential = nullptr;
  const HloInstruction* correction = nullptr;
  for (int64_t operand = 0; operand < 2; ++operand) {
    if (softmax_derivative->operand(operand)->opcode() == HloOpcode::kExp) {
      exponential = softmax_derivative->operand(operand);
      correction = softmax_derivative->operand(1 - operand);
      break;
    }
  }
  if (exponential == nullptr || correction->opcode() != HloOpcode::kAdd ||
      correction->operand_count() != 2) {
    return std::nullopt;
  }

  const HloInstruction* direct = nullptr;
  const HloInstruction* negative_row_term = nullptr;
  for (int64_t operand = 0; operand < 2; ++operand) {
    if (correction->operand(operand)->opcode() == HloOpcode::kDivide) {
      direct = correction->operand(operand);
      negative_row_term = correction->operand(1 - operand);
      break;
    }
  }
  if (direct == nullptr || direct->operand_count() != 2) {
    return std::nullopt;
  }
  const HloInstruction* upstream_gradient = direct->operand(0);
  const HloInstruction* sum_broadcast = direct->operand(1);
  const HloInstruction* row_sum =
      BroadcastOperand(sum_broadcast, exponential->shape());
  const HloInstruction* sum_reduction =
      row_sum == nullptr
          ? nullptr
          : GetRowReduction(row_sum, HloOpcode::kAdd, exponential);
  if (sum_reduction == nullptr ||
      !IsScalarConstant(sum_reduction->operand(1), 0.0)) {
    return std::nullopt;
  }

  const HloInstruction* negative_row =
      BroadcastOperand(negative_row_term, exponential->shape());
  if (negative_row == nullptr || negative_row->opcode() != HloOpcode::kNegate ||
      negative_row->operand_count() != 1) {
    return std::nullopt;
  }
  const HloInstruction* weighted_reduction = negative_row->operand(0);
  const HloInstruction* weighted_product =
      weighted_reduction->operand_count() == 0 ? nullptr
                                               : weighted_reduction->operand(0);
  if (weighted_product == nullptr ||
      GetRowReduction(weighted_reduction, HloOpcode::kAdd, weighted_product) ==
          nullptr ||
      !IsScalarConstant(weighted_reduction->operand(1), 0.0)) {
    return std::nullopt;
  }

  // XLA spells sum(g * p) as sum(g * exp / sum(exp)^2), then multiplies the
  // resulting correction by exp. Verify that exact shared DAG so replacing it
  // does not change the meaning of an arbitrary chain of reductions.
  std::vector<const HloInstruction*> product_leaves;
  CollectMultiplyLeaves(weighted_product, product_leaves);
  if (product_leaves.size() != 3) {
    return std::nullopt;
  }
  const HloInstruction* reciprocal_broadcast = nullptr;
  for (const HloInstruction* leaf : product_leaves) {
    if (leaf != upstream_gradient && leaf != exponential) {
      reciprocal_broadcast = leaf;
    }
  }
  if (reciprocal_broadcast == nullptr ||
      !IsProductOf(weighted_product, upstream_gradient, exponential,
                   reciprocal_broadcast)) {
    return std::nullopt;
  }
  const HloInstruction* reciprocal =
      BroadcastOperand(reciprocal_broadcast, exponential->shape());
  if (reciprocal == nullptr || reciprocal->opcode() != HloOpcode::kDivide ||
      reciprocal->operand_count() != 2 ||
      !ScalarBroadcastValue(reciprocal->operand(0), reciprocal->shape())
           .has_value() ||
      *ScalarBroadcastValue(reciprocal->operand(0), reciprocal->shape()) !=
          1.0 ||
      reciprocal->operand(1)->opcode() != HloOpcode::kMultiply ||
      reciprocal->operand(1)->operand(0) != row_sum ||
      reciprocal->operand(1)->operand(1) != row_sum) {
    return std::nullopt;
  }

  const HloInstruction* scaled_scores =
      StableShiftInput(exponential->operand(0));
  if (scaled_scores == nullptr ||
      scaled_scores->opcode() != HloOpcode::kMultiply) {
    return std::nullopt;
  }
  const HloInstruction* scores_f32 =
      OtherBinaryOperand(scaled_scores, HloOpcode::kMultiply, scale_broadcast);
  if (scores_f32 == nullptr || scores_f32->shape().element_type() != F32 ||
      upstream_gradient->shape().element_type() != F32 ||
      !ShapeUtil::SameDimensions(scores_f32->shape(), root.shape()) ||
      !ShapeUtil::SameDimensions(upstream_gradient->shape(), root.shape()) ||
      !HasCompatibleSoftmaxShape(root, *scores_f32)) {
    return std::nullopt;
  }
  const HloInstruction* scores = SoftmaxInterfaceValue(scores_f32);
  const HloInstruction* upstream = SoftmaxInterfaceValue(upstream_gradient);
  if (ShapeUtil::ElementsIn(scores->shape()) !=
          ShapeUtil::ElementsIn(root.shape()) ||
      ShapeUtil::ElementsIn(upstream->shape()) !=
          ShapeUtil::ElementsIn(root.shape()) ||
      scores->shape().dimensions_size() < 2 ||
      upstream->shape().dimensions_size() < 2 ||
      scores->shape().dimensions(scores->shape().dimensions_size() - 1) !=
          root.shape().dimensions(root.shape().dimensions_size() - 1) ||
      upstream->shape().dimensions(upstream->shape().dimensions_size() - 1) !=
          root.shape().dimensions(root.shape().dimensions_size() - 1)) {
    return std::nullopt;
  }
  return FlySoftmaxBackwardDescriptor{scores, upstream, &root, scale};
}

std::optional<FlySoftmaxBackwardDescriptor> GetFlySoftmaxBackwardDescriptor(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return std::nullopt;
  }
  return GetFlySoftmaxBackwardDescriptor(analysis.fusion_root(0).instruction());
}

namespace {

const HloInstruction* GetFlySoftmaxComputeInput(const HloInstruction& root) {
  const PrimitiveType element_type = root.shape().element_type();
  std::optional<ExternalRowSoftmaxInputs> external =
      GetExternalRowSoftmaxF32Inputs(root);
  const HloInstruction* converted =
      external.has_value() ? external->input : GetStableSoftmaxF32Input(root);
  if (converted == nullptr) {
    return nullptr;
  }
  const HloInstruction* input = converted;
  if (element_type != F32) {
    // Layout normalization can legally commute a bitcast and a widening
    // conversion: bitcast(convert(parameter)) is equivalent to
    // convert(bitcast(parameter)) for this flat row-wise kernel. Peel the
    // F32 bitcast so both normalized forms select the native emitter.
    while (converted->opcode() == HloOpcode::kBitcast &&
           converted->operand_count() == 1 &&
           converted->shape().element_type() == F32) {
      converted = converted->operand(0);
    }
    if (converted->opcode() == HloOpcode::kConvert &&
        converted->operand_count() == 1 &&
        converted->shape().element_type() == F32 &&
        converted->operand(0)->shape().element_type() == element_type) {
      input = converted->operand(0);
    } else if (converted->shape().element_type() == F32) {
      // A scaled attention score is already F32 when the softmax result is
      // narrowed to F16/BF16. The native emitter loads either interface type
      // into F32 registers, so keep that mixed-precision producer as the
      // fusion input instead of requiring a redundant widening convert.
      input = converted;
    } else {
      return nullptr;
    }
  }
  if ((input->shape().element_type() != element_type &&
       input->shape().element_type() != F32) ||
      !HasCompatibleSoftmaxShape(root, *input)) {
    return nullptr;
  }
  return input;
}

std::pair<const HloInstruction*, double> GetFlyScaledSoftmaxInterface(
    const HloInstruction& root) {
  const HloInstruction* input = GetFlySoftmaxComputeInput(root);
  if (input == nullptr) {
    return {nullptr, 1.0};
  }
  if (input->opcode() == HloOpcode::kMultiply && input->operand_count() == 2) {
    for (int64_t operand = 0; operand < 2; ++operand) {
      std::optional<double> scale =
          ScalarBroadcastValue(input->operand(operand), input->shape());
      if (scale.has_value() && std::isfinite(*scale)) {
        return {SoftmaxInterfaceValue(input->operand(1 - operand)), *scale};
      }
    }
  }
  return {input, 1.0};
}

}  // namespace

const HloInstruction* GetFlySoftmaxInput(const HloInstruction& root) {
  return GetFlyScaledSoftmaxInterface(root).first;
}

double GetFlySoftmaxInputScale(const HloInstruction& root) {
  return GetFlyScaledSoftmaxInterface(root).second;
}

const HloInstruction* GetFlySoftmaxExternalRowOffset(
    const HloInstruction& root) {
  std::optional<ExternalRowSoftmaxInputs> external =
      GetExternalRowSoftmaxF32Inputs(root);
  return external.has_value() ? external->row_offset : nullptr;
}

bool FlySoftmaxRecomputesMaximumAfterExternalRowOffset(
    const HloInstruction& root) {
  std::optional<ExternalRowSoftmaxInputs> external =
      GetExternalRowSoftmaxF32Inputs(root);
  return external.has_value() && external->recompute_maximum;
}

const HloInstruction* GetFlyCompoundSoftmaxInput(const HloInstruction& root) {
  const HloInstruction* input = GetStableSoftmaxF32Input(root);
  if (input == nullptr) {
    return nullptr;
  }
  if (input->opcode() == HloOpcode::kConvert && input->operand_count() == 1 &&
      input->operand(0)->shape().element_type() ==
          root.shape().element_type()) {
    input = input->operand(0);
  }
  return HasCompatibleSoftmaxShape(root, *input) ? input : nullptr;
}

const HloInstruction* GetFlyCompoundSoftmaxInputAlongDimension(
    const HloInstruction& root, int64_t reduction_dimension) {
  const HloInstruction* input =
      GetStableSoftmaxF32InputAlongDimension(root, reduction_dimension);
  if (input == nullptr) {
    return nullptr;
  }
  if (input->opcode() == HloOpcode::kConvert && input->operand_count() == 1 &&
      input->operand(0)->shape().element_type() ==
          root.shape().element_type()) {
    input = input->operand(0);
  }
  return HasCompatibleSoftmaxShapeAlongDimension(root, *input,
                                                 reduction_dimension)
             ? input
             : nullptr;
}

bool IsFlySoftmaxRoot(const HloInstruction& root) {
  const HloInstruction* input = GetFlySoftmaxInput(root);
  if (input == nullptr) {
    return false;
  }
  const HloInstruction* parameter = input;
  if (input->opcode() == HloOpcode::kBitcast && input->operand_count() == 1) {
    parameter = input->operand(0);
  }
  if (parameter->opcode() != HloOpcode::kParameter ||
      (parameter->shape().element_type() != root.shape().element_type() &&
       parameter->shape().element_type() != F32) ||
      ShapeUtil::ElementsIn(root.shape()) !=
          ShapeUtil::ElementsIn(parameter->shape())) {
    return false;
  }
  const HloInstruction* external_row_offset =
      GetFlySoftmaxExternalRowOffset(root);
  if (external_row_offset != nullptr &&
      (external_row_offset->parent() != root.parent() ||
       external_row_offset->parameter_number() ==
           parameter->parameter_number())) {
    return false;
  }
  const int64_t columns =
      root.shape().dimensions(root.shape().dimensions_size() - 1);
  if (parameter->shape().dimensions_size() < 2 ||
      parameter->shape().dimensions(parameter->shape().dimensions_size() - 1) !=
          columns) {
    return false;
  }
  return true;
}

bool IsFlySoftmaxFusion(const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return false;
  }
  const HloInstruction& root = analysis.fusion_root(0).instruction();
  if (std::optional<FlySoftmaxBackwardDescriptor> backward =
          GetFlySoftmaxBackwardDescriptor(root)) {
    const HloInstruction* scores = backward->scores;
    const HloInstruction* upstream = backward->upstream_gradient;
    while (scores->opcode() == HloOpcode::kBitcast &&
           scores->operand_count() == 1) {
      scores = scores->operand(0);
    }
    while (upstream->opcode() == HloOpcode::kBitcast &&
           upstream->operand_count() == 1) {
      upstream = upstream->operand(0);
    }
    return root.parent()->num_parameters() == 2 &&
           scores->opcode() == HloOpcode::kParameter &&
           upstream->opcode() == HloOpcode::kParameter &&
           scores->parameter_number() != upstream->parameter_number();
  }
  if (!IsFlySoftmaxRoot(root)) {
    return false;
  }
  return root.parent()->num_parameters() ==
         (GetFlySoftmaxExternalRowOffset(root) == nullptr ? 1 : 2);
}

}  // namespace xla::gpu::flydsl
