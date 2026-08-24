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

bool IsRowReduction(const HloInstruction* reduction, HloOpcode reducer_opcode,
                    const HloInstruction* input) {
  const int64_t input_rank = input->shape().dimensions_size();
  return reduction->opcode() == HloOpcode::kReduce &&
         reduction->operand_count() == 2 && reduction->operand(0) == input &&
         reduction->dimensions().size() == 1 &&
         input_rank >= 2 && reduction->dimensions(0) == input_rank - 1 &&
         reduction->shape().dimensions_size() == input_rank - 1 &&
         ShapeUtil::SameDimensions(reduction->shape(),
                                   ShapeUtil::DeleteDimension(
                                       input_rank - 1, input->shape())) &&
         reduction->called_computations().size() == 1 &&
         reduction->called_computations()
                 .front()
                 ->root_instruction()
                 ->opcode() == reducer_opcode;
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
  if (row_max == nullptr ||
      !IsRowReduction(row_max, HloOpcode::kMaximum, input) ||
      !IsScalarConstant(row_max->operand(1),
                        -std::numeric_limits<double>::infinity())) {
    return nullptr;
  }
  return input;
}

}  // namespace

bool IsFlySoftmaxRoot(const HloInstruction& root) {
  const PrimitiveType element_type = root.shape().element_type();
  if ((element_type != F16 && element_type != BF16 && element_type != F32) ||
      root.shape().dimensions_size() < 2) {
    return false;
  }
  const HloInstruction* normalized = &root;
  if (element_type != F32) {
    if (root.opcode() != HloOpcode::kConvert || root.operand_count() != 1 ||
        root.operand(0)->shape().element_type() != F32) {
      return false;
    }
    normalized = root.operand(0);
  }
  if (normalized->opcode() != HloOpcode::kDivide) {
    return false;
  }
  const HloInstruction* exponential = normalized->operand(0);
  const HloInstruction* row_sum =
      BroadcastOperand(normalized->operand(1), exponential->shape());
  if (exponential->opcode() != HloOpcode::kExp || row_sum == nullptr ||
      !IsRowReduction(row_sum, HloOpcode::kAdd, exponential) ||
      !IsScalarConstant(row_sum->operand(1), 0.0)) {
    return false;
  }
  const HloInstruction* converted =
      StableShiftInput(exponential->operand(0));
  if (converted == nullptr) {
    return false;
  }
  // Accept the second stabilization produced by `x - max(x); softmax(x)`.
  // The native kernel computes the equivalent single stable softmax directly.
  if (const HloInstruction* original = StableShiftInput(converted)) {
    converted = original;
  }
  const HloInstruction* input = converted;
  if (element_type != F32) {
    if (converted->opcode() != HloOpcode::kConvert ||
        converted->operand_count() != 1 ||
        converted->shape().element_type() != F32) {
      return false;
    }
    input = converted->operand(0);
  }
  const HloInstruction* parameter = input;
  if (input->opcode() == HloOpcode::kBitcast && input->operand_count() == 1) {
    parameter = input->operand(0);
  }
  if (parameter->opcode() != HloOpcode::kParameter ||
      parameter->shape().element_type() != element_type ||
      ShapeUtil::ElementsIn(root.shape()) !=
          ShapeUtil::ElementsIn(parameter->shape())) {
    return false;
  }
  const int64_t columns =
      root.shape().dimensions(root.shape().dimensions_size() - 1);
  if (parameter->shape().dimensions_size() < 2 ||
      parameter->shape().dimensions(
          parameter->shape().dimensions_size() - 1) != columns) {
    return false;
  }
  constexpr int64_t kMaxColumns = 16 * 64 * 64;
  return columns > 0 && columns <= kMaxColumns;
}

bool IsFlySoftmaxFusion(const HloFusionAnalysis& analysis) {
  return analysis.fusion_root_count() == 1 &&
         IsFlySoftmaxRoot(analysis.fusion_root(0).instruction());
}

}  // namespace xla::gpu::flydsl
