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

#include <limits>
#include <optional>
#include <utility>

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
  if (reduction->opcode() != HloOpcode::kReduce ||
      reduction->operand_count() != 2 || reduction->operand(0) != input ||
      reduction->dimensions().size() != 1 || input_rank < 2 ||
      reduction->dimensions(0) != input_rank - 1 ||
      reduction->shape().dimensions_size() != input_rank - 1 ||
      !ShapeUtil::SameDimensions(
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
      instruction->dimensions().size() != output_rank - 1) {
    return nullptr;
  }
  for (int64_t dimension = 0; dimension < output_rank - 1; ++dimension) {
    if (instruction->dimensions(dimension) != dimension) {
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

std::optional<ExternalRowSoftmaxInputs>
GetExternalRowSoftmaxF32Inputs(const HloInstruction& root) {
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

}  // namespace

const HloInstruction* GetFlySoftmaxInput(const HloInstruction& root) {
  const PrimitiveType element_type = root.shape().element_type();
  std::optional<ExternalRowSoftmaxInputs> external =
      GetExternalRowSoftmaxF32Inputs(root);
  const HloInstruction* converted = external.has_value()
                                        ? external->input
                                        : GetStableSoftmaxF32Input(root);
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
    if (converted->opcode() != HloOpcode::kConvert ||
        converted->operand_count() != 1 ||
        converted->shape().element_type() != F32) {
      return nullptr;
    }
    input = converted->operand(0);
  }
  if (input->shape().element_type() != element_type ||
      !HasCompatibleSoftmaxShape(root, *input)) {
    return nullptr;
  }
  return input;
}

const HloInstruction* GetFlySoftmaxExternalRowOffset(
    const HloInstruction& root) {
  std::optional<ExternalRowSoftmaxInputs>
      external = GetExternalRowSoftmaxF32Inputs(root);
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
      parameter->shape().element_type() != root.shape().element_type() ||
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
      parameter->shape().dimensions(
          parameter->shape().dimensions_size() - 1) != columns) {
    return false;
  }
  return true;
}

bool IsFlySoftmaxFusion(const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1 ||
      !IsFlySoftmaxRoot(analysis.fusion_root(0).instruction())) {
    return false;
  }
  const HloInstruction& root = analysis.fusion_root(0).instruction();
  return root.parent()->num_parameters() ==
         (GetFlySoftmaxExternalRowOffset(root) == nullptr ? 1 : 2);
}

}  // namespace xla::gpu::flydsl
