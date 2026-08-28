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

  const HloInstruction* predicate = input.operand(0);
  if (predicate->opcode() != HloOpcode::kBroadcast ||
      predicate->operand_count() != 1 ||
      !ShapeUtil::SameDimensions(predicate->shape(), input.shape())) {
    return nullptr;
  }
  if (predicate->dimensions().size() != 2) {
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
  if (lhs->opcode() != HloOpcode::kIota || rhs->opcode() != HloOpcode::kIota ||
      !ShapeUtil::Compatible(lhs->shape(), rhs->shape())) {
    return nullptr;
  }

  // Broadcast dimensions map compare dimensions back to the score tensor.
  // Accept key <= query, or the algebraically identical query >= key.
  const int64_t lhs_dimension =
      Cast<const HloIotaInstruction>(lhs)->iota_dimension();
  const int64_t rhs_dimension =
      Cast<const HloIotaInstruction>(rhs)->iota_dimension();
  if (lhs_dimension < 0 || lhs_dimension >= predicate->dimensions().size() ||
      rhs_dimension < 0 || rhs_dimension >= predicate->dimensions().size()) {
    return nullptr;
  }
  const int64_t lhs_output_dimension = predicate->dimensions(lhs_dimension);
  const int64_t rhs_output_dimension = predicate->dimensions(rhs_dimension);
  const bool key_le_query =
      compare->comparison_direction() == ComparisonDirection::kLe &&
      lhs_output_dimension == key_dimension &&
      rhs_output_dimension == query_dimension;
  const bool query_ge_key =
      compare->comparison_direction() == ComparisonDirection::kGe &&
      lhs_output_dimension == query_dimension &&
      rhs_output_dimension == key_dimension;
  return key_le_query || query_ge_key ? input.operand(1) : nullptr;
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
  if (dots.size() != 2 || parameters.size() != 1) {
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

  return FlyAttentionDescriptor{
      batch,  sequence,    sequence,     heads, heads, head_dimension,
      *scale, causal,      element_type, qkv,   qkv,   qk_dot,
      pv_dot, softmax_root};
}

}  // namespace xla::gpu::flydsl
