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

#include "xla/backends/gpu/transforms/attention_rewriter_fly.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/flydsl/attention_support.h"
#include "xla/backends/gpu/codegen/flydsl/softmax_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"

namespace xla::gpu {
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

const HloInstruction* FindSourceParameter(const HloInstruction* instruction) {
  while (instruction->operand_count() == 1 &&
         (instruction->opcode() == HloOpcode::kBitcast ||
          instruction->opcode() == HloOpcode::kReshape ||
          instruction->opcode() == HloOpcode::kTranspose ||
          instruction->opcode() == HloOpcode::kConvert)) {
    instruction = instruction->operand(0);
  }
  return instruction->opcode() == HloOpcode::kParameter ? instruction : nullptr;
}

const HloInstruction* FindUniqueDot(const HloFusionInstruction& fusion) {
  const HloInstruction* dot = nullptr;
  for (const HloInstruction* instruction :
       fusion.fused_instructions_computation()->instructions()) {
    if (instruction->opcode() != HloOpcode::kDot) {
      continue;
    }
    if (dot != nullptr) {
      return nullptr;
    }
    dot = instruction;
  }
  return dot;
}

const HloInstruction* FusionOperandForParameter(
    const HloFusionInstruction& fusion, const HloInstruction* parameter) {
  if (parameter == nullptr ||
      parameter->parameter_number() >= fusion.operand_count()) {
    return nullptr;
  }
  return fusion.operand(parameter->parameter_number());
}

const HloFusionInstruction* AsFusion(const HloInstruction* instruction) {
  instruction = StripDataMovementViews(instruction);
  if (instruction->opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  return Cast<const HloFusionInstruction>(instruction);
}

const HloInstruction* MatchLastDimensionSegment(const HloInstruction* value,
                                                int64_t batch, int64_t sequence,
                                                int64_t packed_width,
                                                int64_t start, int64_t width) {
  while (value->operand_count() == 1 &&
         (value->opcode() == HloOpcode::kBitcast ||
          value->opcode() == HloOpcode::kCopy ||
          value->opcode() == HloOpcode::kReshape ||
          value->opcode() == HloOpcode::kSlice ||
          value->opcode() == HloOpcode::kTranspose)) {
    if (value->opcode() == HloOpcode::kSlice) {
      const Shape& input = value->operand(0)->shape();
      const Shape& output = value->shape();
      if (input.dimensions_size() == 3 && output.dimensions_size() == 3 &&
          input.dimensions(0) == batch && input.dimensions(1) == sequence &&
          input.dimensions(2) == packed_width &&
          output.dimensions(0) == batch && output.dimensions(1) == sequence &&
          output.dimensions(2) == width && value->slice_starts(0) == 0 &&
          value->slice_limits(0) == batch && value->slice_strides(0) == 1 &&
          value->slice_starts(1) == 0 && value->slice_limits(1) == sequence &&
          value->slice_strides(1) == 1 && value->slice_starts(2) == start &&
          value->slice_limits(2) == start + width &&
          value->slice_strides(2) == 1) {
        return StripViews(value->operand(0));
      }
    }
    value = value->operand(0);
  }
  return nullptr;
}

bool HasGroupedQueryView(const HloInstruction* value, int64_t batch,
                         int64_t sequence, int64_t key_value_heads,
                         int64_t group_size, int64_t head_dimension) {
  while (value->operand_count() == 1 &&
         (value->opcode() == HloOpcode::kBitcast ||
          value->opcode() == HloOpcode::kCopy ||
          value->opcode() == HloOpcode::kReshape ||
          value->opcode() == HloOpcode::kSlice ||
          value->opcode() == HloOpcode::kTranspose)) {
    const Shape& shape = value->shape();
    if (shape.dimensions_size() == 5 && shape.dimensions(0) == batch &&
        shape.dimensions(1) == sequence &&
        shape.dimensions(2) == key_value_heads &&
        shape.dimensions(3) == group_size &&
        shape.dimensions(4) == head_dimension) {
      return true;
    }
    value = value->operand(0);
  }
  return false;
}

bool HasGroupedKeyValuePlane(const HloInstruction* value, int64_t plane,
                             int64_t batch, int64_t sequence,
                             int64_t key_value_heads, int64_t head_dimension) {
  while (value->operand_count() == 1 &&
         (value->opcode() == HloOpcode::kBitcast ||
          value->opcode() == HloOpcode::kCopy ||
          value->opcode() == HloOpcode::kReshape ||
          value->opcode() == HloOpcode::kSlice ||
          value->opcode() == HloOpcode::kTranspose)) {
    if (value->opcode() == HloOpcode::kSlice) {
      const Shape& input = value->operand(0)->shape();
      const Shape& output = value->shape();
      if (input.dimensions_size() == 5 && output.dimensions_size() == 5 &&
          input.dimensions(0) == batch && input.dimensions(1) == sequence &&
          input.dimensions(2) == 2 && input.dimensions(3) == key_value_heads &&
          input.dimensions(4) == head_dimension &&
          output.dimensions(0) == batch && output.dimensions(1) == sequence &&
          output.dimensions(2) == 1 &&
          output.dimensions(3) == key_value_heads &&
          output.dimensions(4) == head_dimension &&
          value->slice_starts(2) == plane &&
          value->slice_limits(2) == plane + 1) {
        return true;
      }
    }
    value = value->operand(0);
  }
  return false;
}

const HloFusionInstruction* FindScoreProducerFusion(
    const HloInstruction* instruction) {
  const HloFusionInstruction* result = nullptr;
  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> worklist = {instruction};
  while (!worklist.empty()) {
    const HloInstruction* candidate = worklist.back();
    worklist.pop_back();
    if (!visited.insert(candidate).second) {
      continue;
    }
    if (candidate->opcode() == HloOpcode::kFusion) {
      const auto* fusion = Cast<const HloFusionInstruction>(candidate);
      if (FindUniqueDot(*fusion) != nullptr) {
        if (result != nullptr && result != fusion) {
          return nullptr;
        }
        result = fusion;
        continue;
      }
    }
    worklist.insert(worklist.end(), candidate->operands().begin(),
                    candidate->operands().end());
  }
  return result;
}

template <typename Dimensions>
bool IsSingleDimension(const Dimensions& dimensions, int64_t dimension) {
  return dimensions.size() == 1 && dimensions.Get(0) == dimension;
}

// Finds the source of one semantic Q/K/V plane. GEMM fusion may move the
// transpose into the contraction, and layout assignment may move the packed
// dimension, so recognize the unique full slice of a size-three dimension
// through arbitrary no-materialization views.
const HloInstruction* MatchPackedQkvPlaneSlice(const HloInstruction* value,
                                               int64_t plane,
                                               int64_t plane_count = 3) {
  while (value->operand_count() == 1 &&
         (value->opcode() == HloOpcode::kBitcast ||
          value->opcode() == HloOpcode::kReshape ||
          value->opcode() == HloOpcode::kTranspose ||
          value->opcode() == HloOpcode::kCopy)) {
    value = value->operand(0);
  }
  if (value->opcode() != HloOpcode::kSlice || value->operand_count() != 1) {
    return nullptr;
  }

  const HloInstruction* packed_view = value->operand(0);
  const Shape& packed_shape = packed_view->shape();
  if (packed_shape.dimensions_size() != value->shape().dimensions_size()) {
    return nullptr;
  }
  int64_t packed_dimension = -1;
  for (int64_t dimension = 0; dimension < packed_shape.dimensions_size();
       ++dimension) {
    if (value->slice_strides(dimension) != 1) {
      return nullptr;
    }
    const bool is_requested_plane =
        packed_shape.dimensions(dimension) == plane_count &&
        value->shape().dimensions(dimension) == 1 &&
        value->slice_starts(dimension) == plane &&
        value->slice_limits(dimension) == plane + 1;
    if (is_requested_plane) {
      if (packed_dimension != -1) {
        return nullptr;
      }
      packed_dimension = dimension;
      continue;
    }
    if (value->slice_starts(dimension) != 0 ||
        value->slice_limits(dimension) != packed_shape.dimensions(dimension) ||
        value->shape().dimensions(dimension) !=
            packed_shape.dimensions(dimension)) {
      return nullptr;
    }
  }
  if (packed_dimension == -1) {
    return nullptr;
  }

  while (packed_view->operand_count() == 1 &&
         (packed_view->opcode() == HloOpcode::kBitcast ||
          packed_view->opcode() == HloOpcode::kReshape ||
          packed_view->opcode() == HloOpcode::kCopy)) {
    packed_view = packed_view->operand(0);
  }
  return packed_view;
}

const HloInstruction* MatchProjectedQueryView(const HloInstruction* value,
                                              int64_t batch,
                                              int64_t query_sequence,
                                              int64_t heads,
                                              int64_t head_dimension) {
  const HloInstruction* source = StripDataMovementViews(value);
  if (source->shape().element_type() != value->shape().element_type() ||
      source->shape().dimensions_size() != 2 ||
      source->shape().dimensions(0) != batch * query_sequence ||
      source->shape().dimensions(1) != heads * head_dimension) {
    return nullptr;
  }
  return source;
}

const HloInstruction* MatchProjectedKeyValuePlane(const HloInstruction* value,
                                                  int64_t plane, int64_t batch,
                                                  int64_t key_value_sequence,
                                                  int64_t heads,
                                                  int64_t head_dimension) {
  const HloInstruction* source =
      MatchPackedQkvPlaneSlice(value, plane, /*plane_count=*/2);
  if (source == nullptr ||
      source->shape().element_type() != value->shape().element_type() ||
      source->shape().dimensions_size() != 2 ||
      source->shape().dimensions(0) != batch * key_value_sequence ||
      source->shape().dimensions(1) != 2 * heads * head_dimension) {
    return nullptr;
  }
  return source;
}

// Matches one of the three views of a packed QKV tensor before multi-output
// fusion has grouped the planes.
const HloInstruction* MatchPackedQkvView(const HloInstruction* view,
                                         int64_t plane, int64_t batch,
                                         int64_t sequence, int64_t heads,
                                         int64_t head_dimension) {
  const HloInstruction* source = MatchPackedQkvPlaneSlice(view, plane);
  if (source == nullptr ||
      source->shape().element_type() != view->shape().element_type() ||
      ShapeUtil::ElementsIn(source->shape()) !=
          batch * sequence * 3 * heads * head_dimension) {
    return nullptr;
  }
  return source;
}

// Multi-output fusion is free to reorder tuple roots. Recover the semantic
// Q/K/V plane from the slice inside the tuple fusion instead of assigning a
// meaning to get-tuple-element indices. Layout assignment can also move the
// packed dimension (for example [B,S,3,H,D] can become [1,B,S,3,D]), so find
// the unique sliced size-three dimension structurally.
const HloInstruction* MatchPackedQkvTupleView(const HloInstruction* view,
                                              int64_t plane, int64_t batch,
                                              int64_t sequence, int64_t heads,
                                              int64_t head_dimension) {
  const HloInstruction* gte = StripViews(view);
  if (gte->opcode() != HloOpcode::kGetTupleElement ||
      gte->operand_count() != 1 ||
      gte->operand(0)->opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  const auto* tuple_fusion = Cast<const HloFusionInstruction>(gte->operand(0));
  const HloInstruction* tuple_root =
      tuple_fusion->fused_instructions_computation()->root_instruction();
  if (tuple_fusion->operand_count() != 1 ||
      tuple_root->opcode() != HloOpcode::kTuple || gte->tuple_index() < 0 ||
      gte->tuple_index() >= tuple_root->operand_count()) {
    return nullptr;
  }

  const HloInstruction* packed_view =
      MatchPackedQkvPlaneSlice(tuple_root->operand(gte->tuple_index()), plane);
  if (packed_view == nullptr) {
    return nullptr;
  }
  if (packed_view->opcode() != HloOpcode::kParameter) {
    return nullptr;
  }
  const HloInstruction* source =
      FusionOperandForParameter(*tuple_fusion, packed_view);
  if (source == nullptr ||
      source->shape().element_type() != view->shape().element_type() ||
      ShapeUtil::ElementsIn(source->shape()) !=
          batch * sequence * 3 * heads * head_dimension) {
    return nullptr;
  }
  return source;
}

struct AttentionMatch {
  HloFusionInstruction* pv = nullptr;
  HloInstruction* output = nullptr;
  const HloInstruction* qkv = nullptr;
  const HloInstruction* key_value = nullptr;
  int64_t batch = 0;
  int64_t sequence = 0;
  int64_t key_value_sequence = 0;
  int64_t heads = 0;
  int64_t key_value_heads = 0;
  int64_t head_dimension = 0;
};

std::optional<AttentionMatch> MatchAttention(HloInstruction* instruction) {
  const PrimitiveType element_type = instruction->shape().element_type();
  if ((element_type != BF16 && element_type != F16) ||
      instruction->shape().dimensions_size() != 4 ||
      !instruction->shape().has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(instruction->shape().layout())) {
    return std::nullopt;
  }
  const HloFusionInstruction* pv_fusion = AsFusion(instruction);
  if (pv_fusion == nullptr) {
    return std::nullopt;
  }
  auto* pv = const_cast<HloFusionInstruction*>(pv_fusion);
  const HloInstruction* pv_dot = FindUniqueDot(*pv);
  if (pv_dot == nullptr || pv_dot->operand_count() != 2 ||
      pv_dot->shape().element_type() != element_type ||
      pv_dot->shape().dimensions_size() != 3) {
    return std::nullopt;
  }
  const auto& pv_dimensions = pv_dot->dot_dimension_numbers();
  if (!IsSingleDimension(pv_dimensions.lhs_batch_dimensions(), 0) ||
      !IsSingleDimension(pv_dimensions.rhs_batch_dimensions(), 0) ||
      !IsSingleDimension(pv_dimensions.lhs_contracting_dimensions(), 2) ||
      !IsSingleDimension(pv_dimensions.rhs_contracting_dimensions(), 2)) {
    return std::nullopt;
  }
  const HloInstruction* pv_lhs = FindSourceParameter(pv_dot->operand(0));
  const HloInstruction* pv_rhs = FindSourceParameter(pv_dot->operand(1));
  const HloInstruction* v_view = FusionOperandForParameter(*pv, pv_lhs);
  const HloInstruction* probability_view =
      FusionOperandForParameter(*pv, pv_rhs);
  if (v_view == nullptr || probability_view == nullptr) {
    return std::nullopt;
  }

  // Standalone softmax fusion is the later pipeline form used by the original
  // prototype. In the production pass order the softmax is still raw HLO, so
  // recognize its root directly and follow its input back to the QK fusion.
  const HloInstruction* softmax_input = nullptr;
  if (const HloFusionInstruction* softmax = AsFusion(probability_view)) {
    if (softmax->operand_count() != 1 || softmax->user_count() != 1 ||
        !flydsl::IsFlySoftmaxRoot(
            *softmax->fused_instructions_computation()->root_instruction())) {
      return std::nullopt;
    }
    softmax_input = softmax->operand(0);
  } else {
    const HloInstruction* softmax_root = StripViews(probability_view);
    softmax_input = flydsl::GetFlyCompoundSoftmaxInput(*softmax_root);
    if (softmax_input == nullptr || probability_view->user_count() != 1) {
      return std::nullopt;
    }
  }
  const int64_t sequence = instruction->shape().dimensions(1);
  const HloInstruction* score_producer = softmax_input;
  if (const HloInstruction* causal_scores =
          flydsl::GetFlyCausalMaskScores(*softmax_input, sequence)) {
    score_producer = causal_scores;
  }
  const HloFusionInstruction* qk = FindScoreProducerFusion(score_producer);
  if (qk == nullptr || qk->operand_count() != 2 || qk->user_count() != 1) {
    return std::nullopt;
  }
  const HloInstruction* qk_dot = FindUniqueDot(*qk);
  if (qk_dot == nullptr || qk_dot->operand_count() != 2 ||
      qk_dot->shape().element_type() != element_type ||
      qk_dot->shape().dimensions_size() != 3) {
    return std::nullopt;
  }
  const auto& qk_dimensions = qk_dot->dot_dimension_numbers();
  const bool has_canonical_q_layout =
      IsSingleDimension(qk_dimensions.lhs_contracting_dimensions(), 2);
  const bool has_layout_folded_q =
      IsSingleDimension(qk_dimensions.lhs_contracting_dimensions(), 1) &&
      qk_dot->operand(0)->shape().dimensions_size() == 3 &&
      qk_dot->operand(0)->shape().dimensions(2) ==
          qk_dot->shape().dimensions(1);
  if (!IsSingleDimension(qk_dimensions.lhs_batch_dimensions(), 0) ||
      !IsSingleDimension(qk_dimensions.rhs_batch_dimensions(), 0) ||
      (!has_canonical_q_layout && !has_layout_folded_q) ||
      !IsSingleDimension(qk_dimensions.rhs_contracting_dimensions(), 1)) {
    return std::nullopt;
  }
  const HloInstruction* q_view =
      FusionOperandForParameter(*qk, FindSourceParameter(qk_dot->operand(0)));
  const HloInstruction* k_view =
      FusionOperandForParameter(*qk, FindSourceParameter(qk_dot->operand(1)));
  if (q_view == nullptr || k_view == nullptr) {
    return std::nullopt;
  }

  const Shape& output = instruction->shape();
  const int64_t batch = output.dimensions(0);
  const int64_t heads = output.dimensions(2);
  const int64_t head_dimension = output.dimensions(3);
  const int64_t key_value_sequence = qk_dot->shape().dimensions(2);
  if (batch <= 0 || sequence < 64 || sequence % 64 != 0 || heads <= 0 ||
      key_value_sequence < 64 || key_value_sequence % 64 != 0 ||
      head_dimension < 64 || head_dimension % 32 != 0 ||
      qk_dot->shape().dimensions(0) != batch * heads ||
      qk_dot->shape().dimensions(1) != sequence ||
      pv_dot->shape().dimensions(0) != batch * heads ||
      pv_dot->shape().dimensions(1) != head_dimension ||
      pv_dot->shape().dimensions(2) != sequence ||
      qk_dot->operand(1)->shape().dimensions_size() != 3 ||
      pv_dot->operand(0)->shape().dimensions_size() != 3 ||
      pv_dot->operand(1)->shape().dimensions_size() != 3 ||
      qk_dot->operand(1)->shape().dimensions(2) != key_value_sequence ||
      pv_dot->operand(0)->shape().dimensions(2) != key_value_sequence ||
      pv_dot->operand(1)->shape().dimensions(2) != key_value_sequence) {
    return std::nullopt;
  }

  // Accept both the later tuple-fusion representation and the earlier raw
  // packed-slice representation used in the real GPU compiler pipeline.
  if (key_value_sequence == sequence) {
    const HloInstruction* tuple_qkv = MatchPackedQkvTupleView(
        q_view, 0, batch, sequence, heads, head_dimension);
    if (tuple_qkv != nullptr &&
        MatchPackedQkvTupleView(k_view, 1, batch, sequence, heads,
                                head_dimension) == tuple_qkv &&
        MatchPackedQkvTupleView(v_view, 2, batch, sequence, heads,
                                head_dimension) == tuple_qkv) {
      return AttentionMatch{pv,    instruction,   tuple_qkv, tuple_qkv,
                            batch, sequence,      sequence,  heads,
                            heads, head_dimension};
    }

    const HloInstruction* qkv =
        MatchPackedQkvView(q_view, 0, batch, sequence, heads, head_dimension);
    if (qkv != nullptr &&
        MatchPackedQkvView(k_view, 1, batch, sequence, heads, head_dimension) ==
            qkv &&
        MatchPackedQkvView(v_view, 2, batch, sequence, heads, head_dimension) ==
            qkv) {
      return AttentionMatch{pv,    instruction,   qkv,      qkv,
                            batch, sequence,      sequence, heads,
                            heads, head_dimension};
    }
  }

  const HloInstruction* query =
      MatchProjectedQueryView(q_view, batch, sequence, heads, head_dimension);
  const HloInstruction* key_value = MatchProjectedKeyValuePlane(
      k_view, /*plane=*/0, batch, key_value_sequence, heads, head_dimension);
  if (query == nullptr || key_value == nullptr || query == key_value ||
      MatchProjectedKeyValuePlane(v_view, /*plane=*/1, batch,
                                  key_value_sequence, heads,
                                  head_dimension) != key_value) {
    return std::nullopt;
  }
  return AttentionMatch{
      pv,       instruction,        query, key_value, batch,
      sequence, key_value_sequence, heads, heads,     head_dimension};
}

std::optional<AttentionMatch> MatchGroupedQueryAttention(
    HloInstruction* instruction) {
  const Shape& output = instruction->shape();
  const PrimitiveType element_type = output.element_type();
  if ((element_type != BF16 && element_type != F16) ||
      output.dimensions_size() != 3 || !output.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout())) {
    return std::nullopt;
  }
  const HloFusionInstruction* pv_fusion = AsFusion(instruction);
  if (pv_fusion == nullptr) {
    return std::nullopt;
  }
  auto* pv = const_cast<HloFusionInstruction*>(pv_fusion);
  const HloInstruction* pv_dot = FindUniqueDot(*pv);
  if (pv_dot == nullptr || pv_dot->operand_count() != 2 ||
      pv_dot->shape().element_type() != element_type ||
      pv_dot->shape().dimensions_size() != 3) {
    return std::nullopt;
  }
  const auto& pv_dimensions = pv_dot->dot_dimension_numbers();
  if (!IsSingleDimension(pv_dimensions.lhs_batch_dimensions(), 0) ||
      !IsSingleDimension(pv_dimensions.rhs_batch_dimensions(), 0) ||
      !IsSingleDimension(pv_dimensions.lhs_contracting_dimensions(), 2) ||
      !IsSingleDimension(pv_dimensions.rhs_contracting_dimensions(), 1)) {
    return std::nullopt;
  }

  const int64_t batch = output.dimensions(0);
  const int64_t sequence = output.dimensions(1);
  const int64_t batch_key_value_heads = pv_dot->shape().dimensions(0);
  const int64_t head_dimension = pv_dot->shape().dimensions(1);
  const int64_t grouped_sequence = pv_dot->shape().dimensions(2);
  if (batch <= 0 || sequence < 64 || sequence % 64 != 0 ||
      batch_key_value_heads % batch != 0 || grouped_sequence % sequence != 0 ||
      head_dimension < 64 || head_dimension % 32 != 0) {
    return std::nullopt;
  }
  const int64_t key_value_heads = batch_key_value_heads / batch;
  const int64_t group_size = grouped_sequence / sequence;
  const int64_t query_heads = key_value_heads * group_size;
  if (key_value_heads <= 0 || group_size <= 1 ||
      output.dimensions(2) != query_heads * head_dimension) {
    return std::nullopt;
  }

  const HloInstruction* pv_lhs = FindSourceParameter(pv_dot->operand(0));
  const HloInstruction* pv_rhs = FindSourceParameter(pv_dot->operand(1));
  const HloInstruction* v_view = FusionOperandForParameter(*pv, pv_lhs);
  const HloInstruction* probability_view =
      FusionOperandForParameter(*pv, pv_rhs);
  if (v_view == nullptr || probability_view == nullptr) {
    return std::nullopt;
  }
  const HloInstruction* softmax_root = StripDataMovementViews(probability_view);
  const HloInstruction* softmax_input =
      flydsl::GetFlyCompoundSoftmaxInputAlongDimension(
          *softmax_root, /*reduction_dimension=*/2);
  if (softmax_input == nullptr || probability_view->user_count() != 1) {
    return std::nullopt;
  }
  const HloFusionInstruction* qk = FindScoreProducerFusion(softmax_input);
  if (qk == nullptr || qk->operand_count() != 2 || qk->user_count() != 1) {
    return std::nullopt;
  }
  const HloInstruction* qk_dot = FindUniqueDot(*qk);
  if (qk_dot == nullptr || qk_dot->operand_count() != 2 ||
      qk_dot->shape().element_type() != element_type ||
      qk_dot->shape().dimensions_size() != 3 ||
      qk_dot->shape().dimensions(0) != batch_key_value_heads ||
      qk_dot->shape().dimensions(1) != sequence ||
      qk_dot->shape().dimensions(2) != grouped_sequence) {
    return std::nullopt;
  }
  const auto& qk_dimensions = qk_dot->dot_dimension_numbers();
  if (!IsSingleDimension(qk_dimensions.lhs_batch_dimensions(), 0) ||
      !IsSingleDimension(qk_dimensions.rhs_batch_dimensions(), 0) ||
      !IsSingleDimension(qk_dimensions.lhs_contracting_dimensions(), 2) ||
      !IsSingleDimension(qk_dimensions.rhs_contracting_dimensions(), 1)) {
    return std::nullopt;
  }
  const HloInstruction* k_view =
      FusionOperandForParameter(*qk, FindSourceParameter(qk_dot->operand(0)));
  const HloInstruction* q_view =
      FusionOperandForParameter(*qk, FindSourceParameter(qk_dot->operand(1)));
  if (q_view == nullptr || k_view == nullptr) {
    return std::nullopt;
  }

  const int64_t query_width = query_heads * head_dimension;
  const int64_t key_value_width = 2 * key_value_heads * head_dimension;
  const int64_t packed_width = query_width + key_value_width;
  const HloInstruction* qkv = MatchLastDimensionSegment(
      q_view, batch, sequence, packed_width, /*start=*/0, query_width);
  if (qkv == nullptr ||
      MatchLastDimensionSegment(k_view, batch, sequence, packed_width,
                                query_width, key_value_width) != qkv ||
      MatchLastDimensionSegment(v_view, batch, sequence, packed_width,
                                query_width, key_value_width) != qkv ||
      !HasGroupedQueryView(q_view, batch, sequence, key_value_heads, group_size,
                           head_dimension) ||
      !HasGroupedKeyValuePlane(k_view, /*plane=*/0, batch, sequence,
                               key_value_heads, head_dimension) ||
      !HasGroupedKeyValuePlane(v_view, /*plane=*/1, batch, sequence,
                               key_value_heads, head_dimension) ||
      qkv->shape().element_type() != element_type ||
      qkv->shape().dimensions_size() != 2 ||
      qkv->shape().dimensions(0) != batch * sequence ||
      qkv->shape().dimensions(1) != packed_width) {
    return std::nullopt;
  }
  return AttentionMatch{pv,
                        instruction,
                        qkv,
                        qkv,
                        batch,
                        sequence,
                        sequence,
                        query_heads,
                        key_value_heads,
                        head_dimension};
}

// Clones a graph while recursively inlining nested fusion instructions. The
// selected packed QKV producer, or separate projected Q and packed KV
// producers, are deliberately left as the only external parameters. This
// removes the materialized Q/K/V transpose along with QK, softmax and PV.
class AttentionFusionCloner {
 public:
  AttentionFusionCloner(HloComputation::Builder* builder,
                        const std::vector<const HloInstruction*>& inputs)
      : builder_(builder) {
    for (int64_t index = 0; index < inputs.size(); ++index) {
      old_to_new_[inputs[index]] =
          builder_->AddInstruction(HloInstruction::CreateParameter(
              index, inputs[index]->shape(), "attention_input"));
    }
  }

  absl::StatusOr<HloInstruction*> Clone(const HloInstruction* instruction) {
    return Clone(instruction, /*parameter_bindings=*/nullptr);
  }

 private:
  using ParameterBindings =
      absl::flat_hash_map<const HloInstruction*, const HloInstruction*>;

  absl::StatusOr<HloInstruction*> Clone(
      const HloInstruction* instruction,
      const ParameterBindings* parameter_bindings) {
    if (auto existing = old_to_new_.find(instruction);
        existing != old_to_new_.end()) {
      return existing->second;
    }
    if (instruction->opcode() == HloOpcode::kParameter) {
      if (parameter_bindings == nullptr) {
        return absl::InvalidArgumentError(
            "Fly attention encountered an unexpected external parameter.");
      }
      auto binding = parameter_bindings->find(instruction);
      if (binding == parameter_bindings->end()) {
        return absl::InvalidArgumentError(
            "Fly attention could not resolve an inlined parameter.");
      }
      return Clone(binding->second, /*parameter_bindings=*/nullptr);
    }
    if (instruction->opcode() == HloOpcode::kFusion) {
      const auto* fusion = Cast<const HloFusionInstruction>(instruction);
      ParameterBindings bindings;
      for (const HloInstruction* parameter :
           fusion->fused_instructions_computation()->parameter_instructions()) {
        bindings[parameter] = fusion->operand(parameter->parameter_number());
      }
      TF_ASSIGN_OR_RETURN(
          HloInstruction * result,
          Clone(fusion->fused_instructions_computation()->root_instruction(),
                &bindings));
      old_to_new_[instruction] = result;
      return result;
    }

    std::vector<HloInstruction*> operands;
    operands.reserve(instruction->operand_count());
    for (const HloInstruction* operand : instruction->operands()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * cloned,
                          Clone(operand, parameter_bindings));
      operands.push_back(cloned);
    }
    HloInstruction* cloned = builder_->AddInstruction(
        instruction->CloneWithNewOperands(instruction->shape(), operands));
    old_to_new_[instruction] = cloned;
    return cloned;
  }

  HloComputation::Builder* builder_;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> old_to_new_;
};

absl::StatusOr<HloFusionInstruction*> MakeAttentionFusion(
    const AttentionMatch& match) {
  std::vector<const HloInstruction*> inputs = {match.qkv};
  if (match.key_value != match.qkv) {
    inputs.push_back(match.key_value);
  }
  HloComputation::Builder builder("fly_attention_computation");
  AttentionFusionCloner cloner(&builder, inputs);
  TF_ASSIGN_OR_RETURN(HloInstruction * root, cloner.Clone(match.output));

  HloModule* module = match.output->GetModule();
  HloComputation* parent = match.output->parent();
  HloComputation* fused_computation =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(root),
                                                /*is_entry=*/false);
  std::vector<HloInstruction*> fusion_operands;
  fusion_operands.reserve(inputs.size());
  for (const HloInstruction* input : inputs) {
    fusion_operands.push_back(const_cast<HloInstruction*>(input));
  }
  HloInstruction* fusion = parent->AddInstruction(
      HloInstruction::CreateFusion(match.output->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   fusion_operands, fused_computation),
      /*new_name=*/"fly_attention");
  fusion->set_metadata(match.output->metadata());

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig* fusion_config =
      gpu_config.mutable_fusion_backend_config();
  fusion_config->set_kind(kFlyFusionKind);
  BlockLevelFusionConfig* block =
      fusion_config->mutable_block_level_fusion_config();
  // The gfx942 kernel assigns 32 query rows to each Wave64. Prefer the
  // reference 128-row tile when it divides the sequence, otherwise use the
  // universally valid 64-row tile admitted by MatchAttention.
  const int64_t block_m = match.sequence % 128 == 0 ? 128 : 64;
  Tile* output_tile = block->add_output_tiles();
  output_tile->add_sizes(1);
  output_tile->add_sizes(block_m);
  output_tile->add_sizes(1);
  output_tile->add_sizes(match.head_dimension);
  block->set_num_warps(block_m / 32);
  block->set_num_ctas(1);
  block->set_num_stages(1);
  block->set_waves_per_eu(2);
  TF_RETURN_IF_ERROR(fusion->set_backend_config(std::move(gpu_config)));

  if (match.output->IsRoot()) {
    parent->set_root_instruction(fusion);
  } else {
    TF_RETURN_IF_ERROR(match.output->ReplaceAllUsesWith(fusion));
  }
  TF_RETURN_IF_ERROR(parent->RemoveInstructionAndUnusedOperands(match.output));
  return Cast<HloFusionInstruction>(fusion);
}

}  // namespace

absl::StatusOr<bool> AttentionRewriterFly::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<AttentionMatch> matches;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      std::optional<AttentionMatch> match = MatchAttention(instruction);
      if (!match.has_value()) {
        match = MatchGroupedQueryAttention(instruction);
      }
      if (match.has_value()) {
        matches.push_back(*match);
      }
    }
  }
  for (const AttentionMatch& match : matches) {
    TF_ASSIGN_OR_RETURN(HloFusionInstruction * fusion,
                        MakeAttentionFusion(match));
    (void)fusion;
  }
  return !matches.empty();
}

}  // namespace xla::gpu
