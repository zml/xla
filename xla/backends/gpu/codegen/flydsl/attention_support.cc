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

bool IsQkDot(const HloInstruction& dot, int64_t batch_heads,
             int64_t sequence, int64_t head_dimension) {
  if (dot.opcode() != HloOpcode::kDot || dot.operand_count() != 2 ||
      dot.shape().dimensions_size() != 3 ||
      dot.shape().dimensions(0) != batch_heads ||
      dot.shape().dimensions(1) != sequence ||
      dot.shape().dimensions(2) != sequence) {
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
  return lhs.dimensions(lhs_sequence_dimension) == sequence &&
         rhs.dimensions(2) == sequence;
}

bool IsPvDot(const HloInstruction& dot, int64_t batch_heads,
             int64_t sequence, int64_t head_dimension) {
  if (dot.opcode() != HloOpcode::kDot || dot.operand_count() != 2 ||
      dot.shape().dimensions_size() != 3 ||
      dot.shape().dimensions(0) != batch_heads ||
      dot.shape().dimensions(1) != head_dimension ||
      dot.shape().dimensions(2) != sequence) {
    return false;
  }
  const auto& dimensions = dot.dot_dimension_numbers();
  return IsSingleDimension(dimensions.lhs_batch_dimensions(), 0) &&
         IsSingleDimension(dimensions.rhs_batch_dimensions(), 0) &&
         IsSingleDimension(dimensions.lhs_contracting_dimensions(), 2) &&
         IsSingleDimension(dimensions.rhs_contracting_dimensions(), 2);
}

}  // namespace

const HloInstruction* GetFlyCausalMaskScores(const HloInstruction& input,
                                             int64_t sequence) {
  if (input.opcode() != HloOpcode::kSelect || input.operand_count() != 3 ||
      input.shape().dimensions_size() < 2 ||
      input.shape().dimensions(input.shape().dimensions_size() - 2) !=
          sequence ||
      input.shape().dimensions(input.shape().dimensions_size() - 1) !=
          sequence) {
    return nullptr;
  }
  std::optional<double> masked_value = UniformConstant(input.operand(2));
  if (!masked_value.has_value() || !std::isinf(*masked_value) ||
      *masked_value >= 0.0 ||
      !ShapeUtil::Compatible(input.shape(), input.operand(1)->shape())) {
    return nullptr;
  }

  const HloInstruction* predicate = input.operand(0);
  if (predicate->opcode() != HloOpcode::kBroadcast ||
      predicate->operand_count() != 1 ||
      !ShapeUtil::SameDimensions(predicate->shape(), input.shape())) {
    return nullptr;
  }
  const int64_t rank = input.shape().dimensions_size();
  if (predicate->dimensions().size() != 2 ||
      predicate->dimensions(0) != rank - 2 ||
      predicate->dimensions(1) != rank - 1) {
    return nullptr;
  }

  const HloInstruction* compare = predicate->operand(0);
  if (compare->opcode() != HloOpcode::kCompare ||
      compare->operand_count() != 2 ||
      compare->shape().dimensions_size() != 2 ||
      compare->shape().dimensions(0) != sequence ||
      compare->shape().dimensions(1) != sequence) {
    return nullptr;
  }
  const HloInstruction* lhs = compare->operand(0);
  const HloInstruction* rhs = compare->operand(1);
  if (lhs->opcode() != HloOpcode::kIota ||
      rhs->opcode() != HloOpcode::kIota ||
      !ShapeUtil::Compatible(lhs->shape(), rhs->shape())) {
    return nullptr;
  }

  // key <= query, or the algebraically identical query >= key.
  const int64_t lhs_dimension =
      Cast<const HloIotaInstruction>(lhs)->iota_dimension();
  const int64_t rhs_dimension =
      Cast<const HloIotaInstruction>(rhs)->iota_dimension();
  const bool key_le_query =
      compare->comparison_direction() == ComparisonDirection::kLe &&
      lhs_dimension == 1 && rhs_dimension == 0;
  const bool query_ge_key =
      compare->comparison_direction() == ComparisonDirection::kGe &&
      lhs_dimension == 0 && rhs_dimension == 1;
  return key_le_query || query_ge_key ? input.operand(1) : nullptr;
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
      output.dimensions_size() != 4 || !output.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout())) {
    return std::nullopt;
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
  if (dots.size() != 2 || parameters.size() != 1) {
    return std::nullopt;
  }

  const int64_t batch_heads = batch * heads;
  const HloInstruction* qk_dot = nullptr;
  const HloInstruction* pv_dot = nullptr;
  for (const HloInstruction* dot : dots) {
    if (IsQkDot(*dot, batch_heads, sequence, head_dimension)) {
      qk_dot = dot;
    }
    if (IsPvDot(*dot, batch_heads, sequence, head_dimension)) {
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
  if (softmax_input == nullptr ||
      !ContainsInstruction(softmax_input, qk_dot)) {
    return std::nullopt;
  }
  const HloInstruction* scaled_scores =
      GetFlyCausalMaskScores(*softmax_input, sequence);
  const bool causal = scaled_scores != nullptr;
  if (!causal) {
    scaled_scores = softmax_input;
  }
  // XLA can remove an intermediate BF16 round trip and leave the QK scale in
  // F32. The native attention kernel already accumulates and scales scores in
  // F32, so accept both forms, but only when the complete producer is a dot,
  // unary view/conversion chain, and uniform multiplication. This excludes
  // unsupported bias or arbitrary score epilogues.
  std::optional<double> scale = MatchScaledDot(scaled_scores, qk_dot);
  if (!scale.has_value() || *scale <= 0.0) {
    return std::nullopt;
  }

  const HloInstruction* qkv = parameters.front();
  if (qkv->shape().element_type() != element_type ||
      qkv->shape().dimensions_size() != 2 || !qkv->shape().has_layout() ||
      qkv->shape().layout().minor_to_major(0) != 1 ||
      qkv->shape().dimensions(0) != batch * sequence ||
      qkv->shape().dimensions(1) != 3 * heads * head_dimension) {
    return std::nullopt;
  }

  return FlyAttentionDescriptor{batch,
                                sequence,
                                heads,
                                head_dimension,
                                *scale,
                                causal,
                                element_type,
                                qkv,
                                qk_dot,
                                pv_dot,
                                softmax_root};
}

}  // namespace xla::gpu::flydsl
