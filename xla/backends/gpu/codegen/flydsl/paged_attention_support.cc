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

#include "xla/backends/gpu/codegen/flydsl/paged_attention_support.h"

#include <cmath>
#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace xla::gpu::flydsl {
namespace {

std::optional<FlyPagedAttentionDescriptor> GetDescriptorForTarget(
    const HloInstruction& call, absl::string_view target,
    const Shape& logical_output) {
  if (call.opcode() != HloOpcode::kCustomCall ||
      call.custom_call_target() != target || call.operand_count() != 6) {
    return std::nullopt;
  }

  const HloInstruction* query = call.operand(0);
  const HloInstruction* key_cache = call.operand(1);
  const HloInstruction* value_cache = call.operand(2);
  const HloInstruction* used_k = call.operand(3);
  const HloInstruction* block_table = call.operand(4);
  const HloInstruction* scale = call.operand(5);
  const PrimitiveType element_type = logical_output.element_type();
  if ((element_type != BF16 && element_type != F16) ||
      logical_output.dimensions_size() != 3 || !logical_output.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(logical_output.layout()) ||
      !ShapeUtil::Compatible(logical_output, query->shape()) ||
      !query->shape().has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(query->shape().layout()) ||
      key_cache->shape().element_type() != element_type ||
      value_cache->shape().element_type() != element_type ||
      key_cache->shape().dimensions_size() != 4 ||
      !ShapeUtil::Compatible(key_cache->shape(), value_cache->shape()) ||
      !key_cache->shape().has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(key_cache->shape().layout()) ||
      !value_cache->shape().has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(value_cache->shape().layout()) ||
      used_k->shape().element_type() != S32 ||
      used_k->shape().dimensions_size() != 1 ||
      block_table->shape().element_type() != S32 ||
      block_table->shape().dimensions_size() != 2 ||
      !block_table->shape().has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(block_table->shape().layout()) ||
      scale->opcode() != HloOpcode::kConstant ||
      scale->shape().element_type() != F32 ||
      !ShapeUtil::IsScalar(scale->shape())) {
    return std::nullopt;
  }

  const int64_t sequences = logical_output.dimensions(0);
  const int64_t query_heads = logical_output.dimensions(1);
  const int64_t head_dimension = logical_output.dimensions(2);
  const int64_t cache_blocks = key_cache->shape().dimensions(0);
  const int64_t page_size = key_cache->shape().dimensions(1);
  const int64_t kv_heads = key_cache->shape().dimensions(2);
  const int64_t gqa_group =
      kv_heads > 0 && query_heads % kv_heads == 0 ? query_heads / kv_heads : 0;
  const int64_t cache_head_dimension = key_cache->shape().dimensions(3);
  const int64_t pages_per_sequence = block_table->shape().dimensions(1);
  const int64_t max_context = pages_per_sequence * page_size;
  const std::optional<double> softmax_scale = scale->literal().GetAsDouble({});
  if (sequences <= 0 || query_heads <= 0 || kv_heads <= 0 || gqa_group <= 0 ||
      gqa_group > 16 || head_dimension != 128 ||
      cache_head_dimension != head_dimension || cache_blocks <= 0 ||
      (page_size != 16 && page_size != 32 && page_size != 64) ||
      pages_per_sequence <= 0 || max_context > 262144 ||
      used_k->shape().dimensions(0) != sequences ||
      block_table->shape().dimensions(0) != sequences ||
      !softmax_scale.has_value() || !std::isfinite(*softmax_scale) ||
      *softmax_scale <= 0.0) {
    return std::nullopt;
  }

  return FlyPagedAttentionDescriptor{
      sequences,      query_heads,    kv_heads,     gqa_group,
      head_dimension, cache_blocks,   page_size,    pages_per_sequence,
      max_context,    *softmax_scale, element_type, query,
      key_cache,      value_cache,    used_k,       block_table,
      &call};
}

}  // namespace

std::optional<FlyPagedAttentionDescriptor> GetFlyPagedAttentionDescriptor(
    const HloInstruction& call) {
  return GetDescriptorForTarget(call, kFlyPagedAttentionDecodeCallTarget,
                                call.shape());
}

std::optional<FlyPagedAttentionDescriptor> GetFlyPagedAttentionDescriptor(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return std::nullopt;
  }
  return GetFlyPagedAttentionDescriptor(analysis.fusion_root(0).instruction());
}

std::optional<FlyPagedAttentionSegmentedProducerDescriptor>
GetFlyPagedAttentionSegmentedProducerDescriptor(const HloInstruction& call) {
  if (call.opcode() != HloOpcode::kCustomCall ||
      call.custom_call_target() !=
          kFlyPagedAttentionSegmentedProducerCallTarget ||
      !call.shape().IsTuple() || call.shape().tuple_shapes_size() != 3 ||
      call.operand_count() != 6) {
    return std::nullopt;
  }
  const Shape& partial_output = call.shape().tuple_shapes(0);
  const Shape& partial_maximum = call.shape().tuple_shapes(1);
  const Shape& partial_sum = call.shape().tuple_shapes(2);
  if (partial_output.element_type() != F32 ||
      partial_output.dimensions_size() != 4 ||
      !partial_output.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(partial_output.layout()) ||
      partial_maximum.element_type() != F32 ||
      partial_maximum.dimensions_size() != 3 ||
      !partial_maximum.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(partial_maximum.layout()) ||
      partial_sum.element_type() != F32 || partial_sum.dimensions_size() != 3 ||
      !partial_sum.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(partial_sum.layout()) ||
      !ShapeUtil::Compatible(partial_maximum, partial_sum)) {
    return std::nullopt;
  }
  const Shape& query_shape = call.operand(0)->shape();
  std::optional<FlyPagedAttentionDescriptor> attention =
      GetDescriptorForTarget(call,
                             kFlyPagedAttentionSegmentedProducerCallTarget,
                             query_shape);
  if (!attention.has_value()) {
    return std::nullopt;
  }
  const int64_t num_segments = partial_output.dimensions(2);
  if (partial_output.dimensions(0) != attention->sequences ||
      partial_output.dimensions(1) != attention->query_heads ||
      partial_output.dimensions(3) != attention->head_dimension ||
      partial_maximum.dimensions(0) != attention->sequences ||
      partial_maximum.dimensions(1) != attention->query_heads ||
      partial_maximum.dimensions(2) != num_segments ||
      num_segments <= 1 || num_segments > 256) {
    return std::nullopt;
  }
  constexpr int64_t kTokenAlignment = 16;
  const int64_t segment_tokens =
      ((attention->max_context + num_segments - 1) / num_segments +
       kTokenAlignment - 1) /
      kTokenAlignment * kTokenAlignment;
  return FlyPagedAttentionSegmentedProducerDescriptor{
      *attention, num_segments, segment_tokens};
}

std::optional<FlyPagedAttentionSegmentedProducerDescriptor>
GetFlyPagedAttentionSegmentedProducerDescriptor(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return std::nullopt;
  }
  return GetFlyPagedAttentionSegmentedProducerDescriptor(
      analysis.fusion_root(0).instruction());
}

std::optional<FlyPagedAttentionSegmentedReducerDescriptor>
GetFlyPagedAttentionSegmentedReducerDescriptor(const HloInstruction& call) {
  if (call.opcode() != HloOpcode::kCustomCall ||
      call.custom_call_target() !=
          kFlyPagedAttentionSegmentedReducerCallTarget ||
      call.operand_count() != 3) {
    return std::nullopt;
  }
  const HloInstruction* partial_output = call.operand(0);
  const HloInstruction* partial_maximum = call.operand(1);
  const HloInstruction* partial_sum = call.operand(2);
  const Shape& partial_output_shape = partial_output->shape();
  const Shape& partial_maximum_shape = partial_maximum->shape();
  const Shape& partial_sum_shape = partial_sum->shape();
  const Shape& output_shape = call.shape();
  const PrimitiveType element_type = output_shape.element_type();
  if ((element_type != BF16 && element_type != F16) ||
      output_shape.dimensions_size() != 3 || !output_shape.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout()) ||
      partial_output_shape.element_type() != F32 ||
      partial_output_shape.dimensions_size() != 4 ||
      !partial_output_shape.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(partial_output_shape.layout()) ||
      partial_maximum_shape.element_type() != F32 ||
      partial_maximum_shape.dimensions_size() != 3 ||
      !partial_maximum_shape.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(partial_maximum_shape.layout()) ||
      partial_sum_shape.element_type() != F32 ||
      partial_sum_shape.dimensions_size() != 3 ||
      !partial_sum_shape.has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(partial_sum_shape.layout()) ||
      !ShapeUtil::Compatible(partial_maximum_shape, partial_sum_shape)) {
    return std::nullopt;
  }
  const int64_t sequences = output_shape.dimensions(0);
  const int64_t query_heads = output_shape.dimensions(1);
  const int64_t head_dimension = output_shape.dimensions(2);
  const int64_t num_segments = partial_output_shape.dimensions(2);
  if (sequences <= 0 || query_heads <= 0 || head_dimension != 128 ||
      partial_output_shape.dimensions(0) != sequences ||
      partial_output_shape.dimensions(1) != query_heads ||
      partial_output_shape.dimensions(3) != head_dimension ||
      partial_maximum_shape.dimensions(0) != sequences ||
      partial_maximum_shape.dimensions(1) != query_heads ||
      partial_maximum_shape.dimensions(2) != num_segments ||
      num_segments <= 1 || num_segments > 256) {
    return std::nullopt;
  }
  return FlyPagedAttentionSegmentedReducerDescriptor{
      sequences, query_heads, num_segments, head_dimension, element_type,
      partial_output, partial_maximum, partial_sum, &call};
}

std::optional<FlyPagedAttentionSegmentedReducerDescriptor>
GetFlyPagedAttentionSegmentedReducerDescriptor(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return std::nullopt;
  }
  return GetFlyPagedAttentionSegmentedReducerDescriptor(
      analysis.fusion_root(0).instruction());
}

}  // namespace xla::gpu::flydsl
