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

#include "xla/backends/gpu/codegen/flydsl/layer_norm_support.h"

#include <cmath>
#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace xla::gpu::flydsl {
namespace {

const HloInstruction* StripBitcasts(const HloInstruction* instruction) {
  while (instruction->opcode() == HloOpcode::kBitcast &&
         instruction->operand_count() == 1) {
    instruction = instruction->operand(0);
  }
  return instruction;
}

std::optional<double> ScalarConstant(const HloInstruction* instruction) {
  instruction = StripBitcasts(instruction);
  if (instruction->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(instruction->shape())) {
    return std::nullopt;
  }
  return instruction->literal().GetAsDouble({});
}

const HloInstruction* BroadcastOperand(const HloInstruction* instruction,
                                       const Shape& output_shape) {
  instruction = StripBitcasts(instruction);
  const int64_t rank = output_shape.dimensions_size();
  if (instruction->opcode() != HloOpcode::kBroadcast ||
      !ShapeUtil::Compatible(instruction->shape(), output_shape) ||
      instruction->dimensions().size() != rank - 1) {
    return nullptr;
  }
  for (int64_t dimension = 0; dimension < rank - 1; ++dimension) {
    if (instruction->dimensions(dimension) != dimension) return nullptr;
  }
  return StripBitcasts(instruction->operand(0));
}

std::optional<double> ScalarBroadcastConstant(const HloInstruction* instruction,
                                              const Shape& output_shape) {
  instruction = StripBitcasts(instruction);
  if (instruction->opcode() == HloOpcode::kBroadcast &&
      ShapeUtil::Compatible(instruction->shape(), output_shape) &&
      instruction->dimensions().empty()) {
    return ScalarConstant(instruction->operand(0));
  }
  if (ShapeUtil::IsScalar(output_shape)) return ScalarConstant(instruction);
  return std::nullopt;
}

const HloInstruction* MinorVectorParameter(
    const HloInstruction* instruction, const Shape& output_shape) {
  instruction = StripBitcasts(instruction);
  if (instruction->opcode() == HloOpcode::kConvert &&
      instruction->operand_count() == 1 &&
      ShapeUtil::SameDimensions(instruction->shape(), output_shape)) {
    return MinorVectorParameter(instruction->operand(0), output_shape);
  }
  const int64_t rank = output_shape.dimensions_size();
  if (instruction->opcode() != HloOpcode::kBroadcast || rank < 2 ||
      !ShapeUtil::SameDimensions(instruction->shape(), output_shape) ||
      instruction->dimensions().size() != 1 ||
      instruction->dimensions(0) != rank - 1) {
    return nullptr;
  }
  const HloInstruction* vector = StripBitcasts(instruction->operand(0));
  if (vector->opcode() == HloOpcode::kConvert &&
      vector->operand_count() == 1) {
    vector = StripBitcasts(vector->operand(0));
  }
  if (vector->opcode() != HloOpcode::kParameter ||
      vector->shape().dimensions_size() != 1 ||
      vector->shape().dimensions(0) != output_shape.dimensions(rank - 1) ||
      (vector->shape().element_type() != F16 &&
       vector->shape().element_type() != BF16 &&
       vector->shape().element_type() != F32)) {
    return nullptr;
  }
  return vector;
}

bool IsAddReduction(const HloInstruction* reduction,
                    const HloInstruction* input) {
  reduction = StripBitcasts(reduction);
  input = StripBitcasts(input);
  const int64_t rank = input->shape().dimensions_size();
  if (reduction->opcode() != HloOpcode::kReduce ||
      reduction->operand_count() != 2 ||
      StripBitcasts(reduction->operand(0)) != input || rank < 2 ||
      reduction->dimensions().size() != 1 ||
      reduction->dimensions(0) != rank - 1 ||
      reduction->called_computations().size() != 1 ||
      reduction->called_computations().front()->root_instruction()->opcode() !=
          HloOpcode::kAdd) {
    return false;
  }
  std::optional<double> init = ScalarConstant(reduction->operand(1));
  return init.has_value() && *init == 0.0;
}

const HloInstruction* ScaledRowSum(const HloInstruction* instruction,
                                   const HloInstruction* input,
                                   int64_t columns) {
  instruction = StripBitcasts(instruction);
  const HloInstruction* reduction = nullptr;
  double scale = 0.0;
  if (instruction->opcode() == HloOpcode::kMultiply &&
      instruction->operand_count() == 2) {
    for (int64_t reduction_operand = 0; reduction_operand < 2;
         ++reduction_operand) {
      const HloInstruction* candidate =
          StripBitcasts(instruction->operand(reduction_operand));
      std::optional<double> candidate_scale = ScalarBroadcastConstant(
          instruction->operand(1 - reduction_operand), instruction->shape());
      if (candidate_scale.has_value() && IsAddReduction(candidate, input)) {
        reduction = candidate;
        scale = *candidate_scale;
        break;
      }
    }
  } else if (instruction->opcode() == HloOpcode::kDivide &&
             instruction->operand_count() == 2) {
    const HloInstruction* candidate = StripBitcasts(instruction->operand(0));
    std::optional<double> divisor =
        ScalarBroadcastConstant(instruction->operand(1), instruction->shape());
    if (divisor.has_value() && *divisor != 0.0 &&
        IsAddReduction(candidate, input)) {
      reduction = candidate;
      scale = 1.0 / *divisor;
    }
  }
  if (reduction == nullptr) return nullptr;
  const double expected = 1.0 / static_cast<double>(columns);
  return std::abs(scale - expected) <= std::abs(expected) * 1e-7 ? reduction
                                                                 : nullptr;
}

bool IsReciprocalStddevCube(
    const HloInstruction* instruction,
    const FlyLayerNormDescriptor& descriptor) {
  instruction = StripBitcasts(instruction);
  const HloInstruction* variance_epsilon =
      StripBitcasts(descriptor.reciprocal_stddev->operand(0));
  return instruction->opcode() == HloOpcode::kDivide &&
         instruction->operand_count() == 2 &&
         StripBitcasts(instruction->operand(0)) ==
             StripBitcasts(descriptor.reciprocal_stddev) &&
         StripBitcasts(instruction->operand(1)) == variance_epsilon;
}

}  // namespace

std::optional<FlyLayerNormDescriptor> GetFlyLayerNormDescriptor(
    const HloInstruction& root) {
  if (root.opcode() == HloOpcode::kTuple &&
      (root.operand_count() == 3 || root.operand_count() == 4)) {
    std::optional<FlyLayerNormDescriptor> descriptor =
        GetFlyLayerNormDescriptor(*root.operand(0));
    if (!descriptor.has_value() ||
        StripBitcasts(root.operand(1)) !=
            StripBitcasts(descriptor->mean) ||
        StripBitcasts(root.operand(2)) !=
            StripBitcasts(descriptor->reciprocal_stddev)) {
      return std::nullopt;
    }
    if (root.operand_count() == 4) {
      if (!IsReciprocalStddevCube(root.operand(3), *descriptor)) {
        return std::nullopt;
      }
      descriptor->reciprocal_stddev_cube = root.operand(3);
    }
    descriptor->output_count = root.operand_count();
    return descriptor;
  }
  if (!root.shape().IsArray()) {
    return std::nullopt;
  }
  const PrimitiveType element_type = root.shape().element_type();
  const int64_t rank = root.shape().dimensions_size();
  if ((element_type != F16 && element_type != BF16 && element_type != F32) ||
      rank < 2 || !root.shape().has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(root.shape().layout())) {
    return std::nullopt;
  }
  const int64_t columns = root.shape().dimensions(rank - 1);
  constexpr int64_t kMaxColumns = 16 * 64 * 64;
  if (columns <= 0 || columns > kMaxColumns) return std::nullopt;

  const HloInstruction* normalized = &root;
  if (element_type != F32) {
    if (root.opcode() != HloOpcode::kConvert || root.operand_count() != 1 ||
        root.operand(0)->shape().element_type() != F32) {
      return std::nullopt;
    }
    normalized = StripBitcasts(root.operand(0));
  }

  const HloInstruction* beta = nullptr;
  if (normalized->opcode() == HloOpcode::kAdd &&
      normalized->operand_count() == 2) {
    for (int64_t affine_operand = 0; affine_operand < 2; ++affine_operand) {
      const HloInstruction* candidate = MinorVectorParameter(
          normalized->operand(affine_operand), normalized->shape());
      if (candidate != nullptr) {
        beta = candidate;
        normalized = StripBitcasts(normalized->operand(1 - affine_operand));
        break;
      }
    }
  }

  const HloInstruction* gamma = nullptr;
  if (normalized->opcode() == HloOpcode::kMultiply &&
      normalized->operand_count() == 2) {
    for (int64_t affine_operand = 0; affine_operand < 2; ++affine_operand) {
      const HloInstruction* candidate = MinorVectorParameter(
          normalized->operand(affine_operand), normalized->shape());
      if (candidate != nullptr) {
        gamma = candidate;
        normalized = StripBitcasts(normalized->operand(1 - affine_operand));
        break;
      }
    }
  }

  if (normalized->opcode() != HloOpcode::kMultiply ||
      normalized->operand_count() != 2) {
    return std::nullopt;
  }

  const HloInstruction* centered = nullptr;
  const HloInstruction* reciprocal_stddev = nullptr;
  for (int64_t centered_operand = 0; centered_operand < 2; ++centered_operand) {
    const HloInstruction* candidate =
        StripBitcasts(normalized->operand(centered_operand));
    const HloInstruction* candidate_scale = BroadcastOperand(
        normalized->operand(1 - centered_operand), candidate->shape());
    if (candidate->opcode() == HloOpcode::kSubtract &&
        candidate_scale != nullptr &&
        candidate_scale->opcode() == HloOpcode::kRsqrt) {
      centered = candidate;
      reciprocal_stddev = candidate_scale;
      break;
    }
  }
  if (centered == nullptr || centered->operand_count() != 2) {
    return std::nullopt;
  }

  const HloInstruction* converted = StripBitcasts(centered->operand(0));
  const HloInstruction* mean =
      BroadcastOperand(centered->operand(1), centered->shape());
  if (mean == nullptr ||
      !ShapeUtil::Compatible(centered->shape(), normalized->shape())) {
    return std::nullopt;
  }
  const HloInstruction* input = converted;
  if (element_type != F32) {
    if (converted->opcode() != HloOpcode::kConvert ||
        converted->operand_count() != 1 ||
        converted->shape().element_type() != F32) {
      return std::nullopt;
    }
    input = StripBitcasts(converted->operand(0));
  }
  if (input->opcode() != HloOpcode::kParameter ||
      input->shape().element_type() != element_type ||
      !ShapeUtil::Compatible(input->shape(), root.shape()) ||
      ScaledRowSum(mean, converted, columns) == nullptr) {
    return std::nullopt;
  }

  const HloInstruction* variance_epsilon =
      StripBitcasts(reciprocal_stddev->operand(0));
  if (variance_epsilon->opcode() != HloOpcode::kAdd ||
      variance_epsilon->operand_count() != 2) {
    return std::nullopt;
  }
  const HloInstruction* variance = nullptr;
  std::optional<double> epsilon;
  for (int64_t variance_operand = 0; variance_operand < 2; ++variance_operand) {
    const HloInstruction* candidate =
        StripBitcasts(variance_epsilon->operand(variance_operand));
    std::optional<double> candidate_epsilon =
        ScalarBroadcastConstant(variance_epsilon->operand(1 - variance_operand),
                                variance_epsilon->shape());
    if (candidate_epsilon.has_value()) {
      variance = candidate;
      epsilon = candidate_epsilon;
      break;
    }
  }
  if (variance == nullptr || !epsilon.has_value() || *epsilon < 0.0) {
    return std::nullopt;
  }
  bool uses_moments_variance = false;
  if (variance->opcode() == HloOpcode::kSubtract &&
      variance->operand_count() == 2) {
    const HloInstruction* square_mean = StripBitcasts(variance->operand(0));
    const HloInstruction* mean_square = StripBitcasts(variance->operand(1));
    if (mean_square->opcode() != HloOpcode::kMultiply ||
        mean_square->operand_count() != 2 ||
        StripBitcasts(mean_square->operand(0)) != StripBitcasts(mean) ||
        StripBitcasts(mean_square->operand(1)) != StripBitcasts(mean)) {
      return std::nullopt;
    }
    const HloInstruction* square_reduction = nullptr;
    if (square_mean->opcode() == HloOpcode::kMultiply ||
        square_mean->opcode() == HloOpcode::kDivide) {
      for (const HloInstruction* operand : square_mean->operands()) {
        const HloInstruction* candidate = StripBitcasts(operand);
        if (candidate->opcode() == HloOpcode::kReduce) {
          square_reduction = candidate;
          break;
        }
      }
    }
    if (square_reduction == nullptr ||
        square_reduction->operand_count() != 2) {
      return std::nullopt;
    }
    const HloInstruction* squared =
        StripBitcasts(square_reduction->operand(0));
    if (squared->opcode() != HloOpcode::kMultiply ||
        squared->operand_count() != 2 ||
        StripBitcasts(squared->operand(0)) != converted ||
        StripBitcasts(squared->operand(1)) != converted ||
        ScaledRowSum(square_mean, squared, columns) != square_reduction) {
      return std::nullopt;
    }
    uses_moments_variance = true;
  } else {
    const HloInstruction* square_reduction = nullptr;
    if (variance->opcode() == HloOpcode::kMultiply ||
        variance->opcode() == HloOpcode::kDivide) {
      for (const HloInstruction* operand : variance->operands()) {
        const HloInstruction* candidate = StripBitcasts(operand);
        if (candidate->opcode() == HloOpcode::kReduce) {
          square_reduction = candidate;
          break;
        }
      }
    }
    if (square_reduction == nullptr ||
        square_reduction->operand_count() != 2) {
      return std::nullopt;
    }
    const HloInstruction* squared =
        StripBitcasts(square_reduction->operand(0));
    if (squared->opcode() != HloOpcode::kMultiply ||
        squared->operand_count() != 2 ||
        StripBitcasts(squared->operand(0)) != centered ||
        StripBitcasts(squared->operand(1)) != centered ||
        ScaledRowSum(variance, squared, columns) != square_reduction) {
      return std::nullopt;
    }
  }

  return FlyLayerNormDescriptor{
      &root, input, gamma, beta, mean, reciprocal_stddev,
      /*reciprocal_stddev_cube=*/nullptr, element_type,
      ShapeUtil::ElementsIn(root.shape()) / columns, columns, *epsilon,
      /*output_count=*/1, uses_moments_variance};
}

std::optional<FlyLayerNormDescriptor> GetFlyLayerNormDescriptor(
    const HloFusionAnalysis& analysis) {
  std::optional<FlyLayerNormDescriptor> descriptor;
  if (analysis.fusion_root_count() == 1) {
    descriptor =
        GetFlyLayerNormDescriptor(analysis.fusion_root(0).instruction());
  } else if (analysis.fusion_root_count() == 3 ||
             analysis.fusion_root_count() == 4) {
    descriptor =
        GetFlyLayerNormDescriptor(analysis.fusion_root(0).instruction());
    if (descriptor.has_value() &&
        StripBitcasts(&analysis.fusion_root(1).instruction()) ==
            StripBitcasts(descriptor->mean) &&
        StripBitcasts(&analysis.fusion_root(2).instruction()) ==
            StripBitcasts(descriptor->reciprocal_stddev) &&
        (analysis.fusion_root_count() == 3 ||
         IsReciprocalStddevCube(
             &analysis.fusion_root(3).instruction(), *descriptor))) {
      descriptor->output_count = analysis.fusion_root_count();
      descriptor->reciprocal_stddev_cube =
          analysis.fusion_root_count() == 4
              ? &analysis.fusion_root(3).instruction()
              : nullptr;
    } else {
      descriptor = std::nullopt;
    }
  } else {
    return std::nullopt;
  }
  if (!descriptor.has_value()) {
    return std::nullopt;
  }
  int64_t expected_parameters = 1;
  if (descriptor->gamma != nullptr && descriptor->gamma != descriptor->input) {
    ++expected_parameters;
  }
  if (descriptor->beta != nullptr && descriptor->beta != descriptor->input &&
      descriptor->beta != descriptor->gamma) {
    ++expected_parameters;
  }
  if (descriptor->input->parent()->num_parameters() != expected_parameters) {
    return std::nullopt;
  }
  return descriptor;
}

bool IsFlyLayerNormRoot(const HloInstruction& root) {
  return GetFlyLayerNormDescriptor(root).has_value();
}

bool IsFlyLayerNormFusion(const HloFusionAnalysis& analysis) {
  return GetFlyLayerNormDescriptor(analysis).has_value();
}

}  // namespace xla::gpu::flydsl
