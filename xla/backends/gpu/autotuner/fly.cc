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

#include "xla/backends/gpu/autotuner/fly.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/codegen/flydsl/attention_support.h"
#include "xla/backends/gpu/codegen/flydsl/fusion_support.h"
#include "xla/backends/gpu/codegen/flydsl/layer_norm_support.h"
#include "xla/backends/gpu/codegen/flydsl/paged_attention_support.h"
#include "xla/backends/gpu/codegen/flydsl/scan_support.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_elementwise.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_reduction.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_softmax.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_transpose.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

const HloInstruction* FindContraction(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  const HloInstruction* contraction = hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kDot);
  if (contraction != nullptr) {
    return contraction;
  }
  return hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
}

HloInstruction* FindContraction(HloInstruction& instr) {
  return const_cast<HloInstruction*>(
      FindContraction(static_cast<const HloInstruction&>(instr)));
}

bool IsFnuzFp8(PrimitiveType type) {
  return type == F8E4M3FNUZ || type == F8E5M2FNUZ;
}

bool IsSupportedScaleType(PrimitiveType type) {
  return type == BF16 || type == F16 || type == F32 || type == F8E8M0FNU ||
         type == F8E4M3FN || type == F8E5M2 || type == S8;
}

bool IsUniformScale(const HloInstruction& scale) {
  return ShapeUtil::ElementsIn(scale.shape()) == 1;
}

bool IsSupportedScale(const HloInstruction& scale,
                      const HloInstruction& operand,
                      int64_t contracting_dimension) {
  if (scale.opcode() != HloOpcode::kParameter ||
      !IsSupportedScaleType(scale.shape().element_type()) ||
      scale.shape().dimensions_size() != operand.shape().dimensions_size()) {
    return false;
  }
  for (int64_t dimension = 0; dimension < operand.shape().dimensions_size();
       ++dimension) {
    const int64_t operand_size = operand.shape().dimensions(dimension);
    const int64_t scale_size = scale.shape().dimensions(dimension);
    if (scale_size <= 0 || operand_size % scale_size != 0) {
      return false;
    }
    if (dimension != contracting_dimension && scale_size != 1 &&
        scale_size != operand_size) {
      // The emitter currently needs either one scale for the complete
      // noncontracting dimension or one scale per output row/column.
      return false;
    }
  }
  return true;
}

std::optional<int64_t> ScaledDotKBlockSize(const HloInstruction& dot) {
  if (dot.opcode() != HloOpcode::kScaledDot || dot.operand_count() != 4) {
    return std::nullopt;
  }
  const DotDimensionNumbers& dims = dot.dot_dimension_numbers();
  if (dims.lhs_contracting_dimensions_size() != 1 ||
      dims.rhs_contracting_dimensions_size() != 1) {
    return std::nullopt;
  }
  const int64_t lhs_contracting = dims.lhs_contracting_dimensions(0);
  const int64_t rhs_contracting = dims.rhs_contracting_dimensions(0);
  if (!IsSupportedScale(*dot.operand(2), *dot.operand(0), lhs_contracting) ||
      !IsSupportedScale(*dot.operand(3), *dot.operand(1), rhs_contracting)) {
    return std::nullopt;
  }
  const int64_t k = dot.operand(0)->shape().dimensions(lhs_contracting);
  const int64_t lhs_scale_k =
      dot.operand(2)->shape().dimensions(lhs_contracting);
  const int64_t rhs_scale_k =
      dot.operand(3)->shape().dimensions(rhs_contracting);
  const int64_t lhs_block = k / lhs_scale_k;
  const int64_t rhs_block = k / rhs_scale_k;
  if (lhs_scale_k != 1 && rhs_scale_k != 1 && lhs_block != rhs_block) {
    return std::nullopt;
  }
  return lhs_scale_k == 1 ? rhs_block : lhs_block;
}

bool IsHalfPrecisionFloat(PrimitiveType type) {
  return type == F16 || type == BF16;
}

bool IsSupportedContractionConversion(PrimitiveType source,
                                      PrimitiveType destination) {
  return (source == F32 && destination == BF16) ||
         (source == S4 && (destination == S8 || destination == BF16)) ||
         (source == S8 && destination == BF16);
}

// XLA represents the Q operand of a batched attention score contraction as a
// view of [B, K, M], not as a materialized [B, M, K] buffer.  Keep this first
// input-transpose grammar deliberately narrow until the emitter supports
// composing arbitrary permutations with every other input view.
bool IsSupportedBatchInnerTranspose(const HloInstruction& transpose) {
  return transpose.opcode() == HloOpcode::kTranspose &&
         transpose.operand_count() == 1 &&
         transpose.shape().dimensions_size() == 3 &&
         transpose.operand(0)->shape().dimensions_size() == 3 &&
         transpose.shape().element_type() ==
             transpose.operand(0)->shape().element_type() &&
         transpose.dimensions() == std::vector<int64_t>({0, 2, 1});
}

struct ContractingScaleInput {
  const HloInstruction* data;
  const HloInstruction* data_parameter;
  const HloInstruction* scale;
  const HloInstruction* scale_view;
  std::vector<int64_t> broadcast_dimensions;
  bool collapsed_subchannel_scale = false;
};

const HloInstruction* FindContractionInputParameter(
    const HloInstruction& input);

// Matches the BF16/F16 rounding boundary used by RMSNorm's learned weight:
//
//   convert<T>(convert<f32>(data) * convert<f32>(broadcast(scale)))
//
// Keeping this producer inside a decoder projection avoids materializing the
// scaled activation while preserving the multiply-before-rounding semantics.
std::optional<ContractingScaleInput> MatchContractingScaleInput(
    const HloInstruction& input) {
  if (!IsHalfPrecisionFloat(input.shape().element_type())) {
    return std::nullopt;
  }
  // After layout assignment and BF16 normalization, the same subchannel
  // producer is represented as:
  //
  //   bitcast(convert<T>(multiply(convert<f32>(transpose(convert(s4))),
  //                               convert<f32>(broadcast(transpose(scale))))))
  //
  // The transposes only expose the physical layout selected for the dot.  The
  // Fly emitter recovers the original parameter coordinates from these views.
  if (input.opcode() == HloOpcode::kBitcast && input.operand_count() == 1 &&
      input.operand(0)->opcode() == HloOpcode::kConvert &&
      input.operand(0)->operand_count() == 1 &&
      input.operand(0)->shape().element_type() == input.shape().element_type() &&
      input.operand(0)->operand(0)->opcode() == HloOpcode::kMultiply &&
      input.operand(0)->operand(0)->shape().element_type() == F32 &&
      ShapeUtil::ElementsIn(input.shape()) ==
          ShapeUtil::ElementsIn(input.operand(0)->shape())) {
    const HloInstruction* multiply = input.operand(0)->operand(0);
    for (int64_t data_operand = 0; data_operand < 2; ++data_operand) {
      const HloInstruction* data_f32 = multiply->operand(data_operand);
      const HloInstruction* broadcast_f32 =
          multiply->operand(1 - data_operand);
      if (data_f32->opcode() != HloOpcode::kConvert ||
          data_f32->operand_count() != 1 ||
          data_f32->shape().element_type() != F32 ||
          broadcast_f32->opcode() != HloOpcode::kConvert ||
          broadcast_f32->operand_count() != 1 ||
          broadcast_f32->shape().element_type() != F32 ||
          !ShapeUtil::SameDimensions(data_f32->shape(), multiply->shape()) ||
          !ShapeUtil::SameDimensions(broadcast_f32->shape(),
                                     multiply->shape())) {
        continue;
      }

      const HloInstruction* data = data_f32->operand(0);
      const HloInstruction* broadcast = broadcast_f32->operand(0);
      if (data->opcode() != HloOpcode::kTranspose ||
          data->operand_count() != 1 ||
          data->shape().element_type() != input.shape().element_type() ||
          !ShapeUtil::SameDimensions(data->shape(), multiply->shape()) ||
          broadcast->opcode() != HloOpcode::kBroadcast ||
          broadcast->operand_count() != 1 ||
          broadcast->shape().element_type() != input.shape().element_type() ||
          !ShapeUtil::SameDimensions(broadcast->shape(), multiply->shape()) ||
          broadcast->dimensions().empty()) {
        continue;
      }

      const HloInstruction* data_parameter = data->operand(0);
      bool has_s4_conversion = false;
      while (data_parameter->operand_count() == 1 &&
             (data_parameter->opcode() == HloOpcode::kConvert ||
              data_parameter->opcode() == HloOpcode::kBitcast)) {
        const HloInstruction* operand = data_parameter->operand(0);
        if (data_parameter->opcode() == HloOpcode::kBitcast &&
            (data_parameter->shape().element_type() !=
                 operand->shape().element_type() ||
             ShapeUtil::ElementsIn(data_parameter->shape()) !=
                 ShapeUtil::ElementsIn(operand->shape()))) {
          break;
        }
        has_s4_conversion |= operand->shape().element_type() == S4;
        data_parameter = operand;
      }
      if (data_parameter->opcode() != HloOpcode::kParameter ||
          data_parameter->shape().element_type() != S4 ||
          !has_s4_conversion ||
          ShapeUtil::ElementsIn(data_parameter->shape()) !=
              ShapeUtil::ElementsIn(input.shape())) {
        continue;
      }

      const HloInstruction* scale_view = broadcast->operand(0);
      if (scale_view->opcode() != HloOpcode::kTranspose ||
          scale_view->operand_count() != 1 ||
          scale_view->shape().element_type() != input.shape().element_type()) {
        continue;
      }
      const HloInstruction* scale_parameter = scale_view->operand(0);
      while (scale_parameter->operand_count() == 1 &&
             scale_parameter->opcode() == HloOpcode::kBitcast &&
             scale_parameter->shape().element_type() ==
                 scale_parameter->operand(0)->shape().element_type() &&
             ShapeUtil::ElementsIn(scale_parameter->shape()) ==
                 ShapeUtil::ElementsIn(scale_parameter->operand(0)->shape())) {
        scale_parameter = scale_parameter->operand(0);
      }
      if (scale_parameter->opcode() != HloOpcode::kParameter ||
          scale_parameter->shape().element_type() !=
              input.shape().element_type() ||
          scale_view->shape().dimensions_size() !=
              broadcast->dimensions().size() ||
          !std::equal(scale_view->shape().dimensions().begin(),
                      scale_view->shape().dimensions().end(),
                      broadcast->dimensions().begin(),
                      [&](int64_t scale_size, int64_t input_dimension) {
                        return scale_size == multiply->shape().dimensions(
                                                 input_dimension);
                      })) {
        continue;
      }
      return ContractingScaleInput{
          data,
          data_parameter,
          scale_parameter,
          scale_view,
          std::vector<int64_t>(broadcast->dimensions().begin(),
                               broadcast->dimensions().end()),
          /*collapsed_subchannel_scale=*/true};
    }
  }
  // Triton's subchannel dequantization grammar applies a scale before a
  // bitcast collapses [batch, group, block, row] into [batch, k, row]:
  //
  //   bitcast(multiply(convert(bitcast(convert(s4))),
  //                    broadcast(bitcast(scale))))
  //
  // Keep the physical views explicit in the match. The emitter uses them to
  // recover the scale coordinate for each final contraction coordinate.
  if (input.opcode() == HloOpcode::kBitcast && input.operand_count() == 1 &&
      input.operand(0)->opcode() == HloOpcode::kMultiply &&
      input.operand(0)->operand_count() == 2 &&
      ShapeUtil::ElementsIn(input.shape()) ==
          ShapeUtil::ElementsIn(input.operand(0)->shape())) {
    const HloInstruction* multiply = input.operand(0);
    for (int64_t data_operand = 0; data_operand < 2; ++data_operand) {
      const HloInstruction* data = multiply->operand(data_operand);
      const HloInstruction* broadcast = multiply->operand(1 - data_operand);
      if (data->shape().element_type() != input.shape().element_type() ||
          broadcast->opcode() != HloOpcode::kBroadcast ||
          broadcast->operand_count() != 1 ||
          broadcast->shape().element_type() != input.shape().element_type() ||
          !ShapeUtil::SameDimensions(data->shape(), multiply->shape()) ||
          !ShapeUtil::SameDimensions(broadcast->shape(), multiply->shape()) ||
          broadcast->dimensions().empty()) {
        continue;
      }

      const HloInstruction* data_parameter = data;
      bool has_s4_conversion = false;
      while (data_parameter->operand_count() == 1 &&
             (data_parameter->opcode() == HloOpcode::kConvert ||
              data_parameter->opcode() == HloOpcode::kBitcast)) {
        const HloInstruction* operand = data_parameter->operand(0);
        if (data_parameter->opcode() == HloOpcode::kBitcast &&
            (data_parameter->shape().element_type() !=
                 operand->shape().element_type() ||
             ShapeUtil::ElementsIn(data_parameter->shape()) !=
                 ShapeUtil::ElementsIn(operand->shape()))) {
          break;
        }
        has_s4_conversion |= operand->shape().element_type() == S4;
        data_parameter = operand;
      }
      if (data_parameter->opcode() != HloOpcode::kParameter ||
          data_parameter->shape().element_type() != S4 ||
          !has_s4_conversion ||
          ShapeUtil::ElementsIn(data_parameter->shape()) !=
              ShapeUtil::ElementsIn(input.shape())) {
        continue;
      }

      const HloInstruction* scale_view = broadcast->operand(0);
      const HloInstruction* scale_parameter = scale_view;
      while (scale_parameter->operand_count() == 1 &&
             scale_parameter->opcode() == HloOpcode::kBitcast &&
             scale_parameter->shape().element_type() ==
                 scale_parameter->operand(0)->shape().element_type() &&
             ShapeUtil::ElementsIn(scale_parameter->shape()) ==
                 ShapeUtil::ElementsIn(scale_parameter->operand(0)->shape())) {
        scale_parameter = scale_parameter->operand(0);
      }
      if (scale_parameter->opcode() != HloOpcode::kParameter ||
          scale_parameter->shape().element_type() !=
              input.shape().element_type() ||
          scale_view->shape().dimensions_size() !=
              broadcast->dimensions().size() ||
          !std::equal(scale_view->shape().dimensions().begin(),
                      scale_view->shape().dimensions().end(),
                      broadcast->dimensions().begin(),
                      [&](int64_t scale_size, int64_t input_dimension) {
                        return scale_size == multiply->shape().dimensions(
                                                 input_dimension);
                      })) {
        continue;
      }
      return ContractingScaleInput{
          data,
          data_parameter,
          scale_parameter,
          scale_view,
          std::vector<int64_t>(broadcast->dimensions().begin(),
                               broadcast->dimensions().end()),
          /*collapsed_subchannel_scale=*/true};
    }
  }
  const bool direct_half_multiply =
      input.opcode() == HloOpcode::kMultiply && input.operand_count() == 2;
  const bool rounded_f32_multiply =
      input.opcode() == HloOpcode::kConvert && input.operand_count() == 1 &&
      input.operand(0)->shape().element_type() == F32 &&
      ShapeUtil::SameDimensions(input.shape(), input.operand(0)->shape()) &&
      input.operand(0)->opcode() == HloOpcode::kMultiply &&
      input.operand(0)->operand_count() == 2;
  if (!direct_half_multiply && !rounded_f32_multiply) {
    return std::nullopt;
  }
  const HloInstruction* multiply =
      direct_half_multiply ? &input : input.operand(0);
  for (int64_t data_operand = 0; data_operand < 2; ++data_operand) {
    const HloInstruction* data = multiply->operand(data_operand);
    const HloInstruction* broadcast = multiply->operand(1 - data_operand);
    if (rounded_f32_multiply) {
      if (data->opcode() != HloOpcode::kConvert ||
          data->operand_count() != 1 || data->shape().element_type() != F32 ||
          !ShapeUtil::SameDimensions(data->shape(), input.shape()) ||
          broadcast->opcode() != HloOpcode::kConvert ||
          broadcast->operand_count() != 1 ||
          broadcast->shape().element_type() != F32) {
        continue;
      }
      data = data->operand(0);
      broadcast = broadcast->operand(0);
    }
    if (broadcast->opcode() != HloOpcode::kBroadcast ||
        broadcast->operand_count() != 1 ||
        broadcast->shape().element_type() != input.shape().element_type() ||
        !ShapeUtil::SameDimensions(broadcast->shape(), input.shape()) ||
        broadcast->dimensions().empty()) {
      continue;
    }
    const HloInstruction* data_parameter =
        FindContractionInputParameter(*data);
    if (data->shape().element_type() != input.shape().element_type() ||
        !ShapeUtil::SameDimensions(data->shape(), input.shape()) ||
        data_parameter == nullptr ||
        (data_parameter->shape().element_type() !=
             input.shape().element_type() &&
         data_parameter->shape().element_type() != S4)) {
      continue;
    }
    const HloInstruction* scale = broadcast->operand(0);
    if (scale->opcode() != HloOpcode::kParameter ||
        scale->shape().dimensions_size() != broadcast->dimensions().size() ||
        scale->shape().element_type() != input.shape().element_type() ||
        !std::equal(scale->shape().dimensions().begin(),
                    scale->shape().dimensions().end(),
                    broadcast->dimensions().begin(),
                    [&](int64_t scale_size, int64_t input_dimension) {
                      return scale_size ==
                             input.shape().dimensions(input_dimension);
                    })) {
      continue;
    }
    return ContractingScaleInput{
        data, data_parameter, scale, scale,
        std::vector<int64_t>(broadcast->dimensions().begin(),
                             broadcast->dimensions().end())};
  }
  return std::nullopt;
}

const HloInstruction* FindContractionInputParameter(
    const HloInstruction& input) {
  const PrimitiveType contraction_type = input.shape().element_type();
  if (!IsHalfPrecisionFloat(contraction_type) && !IsFnuzFp8(contraction_type) &&
      contraction_type != F32 && contraction_type != S8) {
    return nullptr;
  }
  const std::optional<ContractingScaleInput> contracting_scale =
      MatchContractingScaleInput(input);
  if (contracting_scale.has_value() &&
      contracting_scale->collapsed_subchannel_scale) {
    return contracting_scale->data_parameter;
  }
  const HloInstruction* value =
      contracting_scale.has_value() ? contracting_scale->data : &input;
  bool has_conversion = false;
  bool has_input_view = false;
  bool physical_mapping_fixed = false;
  bool physical_mapping_from_bitcast = false;
  bool has_static_slice = false;
  bool has_dynamic_slice = false;
  bool has_batch_inner_transpose = false;
  while (value->opcode() == HloOpcode::kBitcast ||
         value->opcode() == HloOpcode::kConvert ||
         value->opcode() == HloOpcode::kSlice ||
         value->opcode() == HloOpcode::kDynamicSlice ||
         value->opcode() == HloOpcode::kTranspose) {
    const bool is_dynamic_slice = value->opcode() == HloOpcode::kDynamicSlice;
    if ((!is_dynamic_slice && value->operand_count() != 1) ||
        (is_dynamic_slice &&
         value->operand_count() != value->shape().dimensions_size() + 1)) {
      return nullptr;
    }
    const HloInstruction* operand = value->operand(0);
    if (value->opcode() == HloOpcode::kTranspose) {
      if (physical_mapping_fixed || !IsSupportedBatchInnerTranspose(*value)) {
        return nullptr;
      }
      has_input_view = true;
      physical_mapping_fixed = true;
      has_batch_inner_transpose = true;
    } else if (value->opcode() == HloOpcode::kBitcast) {
      if (value->shape().element_type() != operand->shape().element_type() ||
          ShapeUtil::ElementsIn(value->shape()) !=
              ShapeUtil::ElementsIn(operand->shape())) {
        return nullptr;
      }
      has_input_view = true;
      if (!physical_mapping_fixed) {
        physical_mapping_from_bitcast = true;
      }
      physical_mapping_fixed = true;
    } else if (value->opcode() == HloOpcode::kConvert) {
      if (!IsSupportedContractionConversion(operand->shape().element_type(),
                                            value->shape().element_type()) ||
          !ShapeUtil::SameDimensions(value->shape(), operand->shape())) {
        return nullptr;
      }
      has_conversion = true;
    } else if (value->opcode() == HloOpcode::kSlice) {
      if (value->shape().dimensions_size() !=
              operand->shape().dimensions_size() ||
          !std::all_of(value->slice_strides().begin(),
                       value->slice_strides().end(),
                       [](int64_t stride) { return stride == 1; })) {
        return nullptr;
      }
      if (physical_mapping_fixed &&
          (!physical_mapping_from_bitcast || has_static_slice ||
           !IsContiguousSlice(*value) ||
           ShapeUtil::ElementsIn(value->shape()) !=
               ShapeUtil::ElementsIn(input.shape()))) {
        return nullptr;
      }
      has_input_view = true;
      has_static_slice = true;
    } else {
      if (physical_mapping_fixed || has_dynamic_slice ||
          value->shape().dimensions_size() !=
              operand->shape().dimensions_size()) {
        return nullptr;
      }
      for (int64_t dimension = 0; dimension < value->shape().dimensions_size();
           ++dimension) {
        const HloInstruction* start = value->operand(dimension + 1);
        if (!ShapeUtil::IsScalar(start->shape()) ||
            (start->shape().element_type() != S32 &&
             start->shape().element_type() != S64) ||
            (start->opcode() != HloOpcode::kParameter &&
             start->opcode() != HloOpcode::kConstant)) {
          return nullptr;
        }
      }
      has_input_view = true;
      has_dynamic_slice = true;
    }
    value = operand;
  }
  if (value->opcode() != HloOpcode::kParameter) {
    return nullptr;
  }
  const PrimitiveType parameter_type = value->shape().element_type();
  if (parameter_type == S4) {
    // Zero-offset batch-inner transposes preserve an element-addressable S4
    // view and are needed by channel-scaled batched dequantization. Slices and
    // dynamic offsets still require a separate nibble-aware representation.
    return contraction_type == BF16 && has_conversion &&
                   (!has_input_view ||
                    (has_batch_inner_transpose && !has_static_slice &&
                     !has_dynamic_slice))
               ? value
               : nullptr;
  }
  if (parameter_type == F32) {
    return ((contraction_type == BF16 && has_conversion) ||
            (contraction_type == F32 && !has_conversion))
               ? value
               : nullptr;
  }
  return parameter_type == contraction_type && !has_conversion ? value
                                                               : nullptr;
}

bool IsDimensionContiguous(const HloInstruction& input, int64_t dimension) {
  if (!input.shape().has_layout()) {
    return false;
  }
  // Singleton dimensions do not contribute to the physical stride. XLA's
  // token-one canonicalization commonly presents a contiguous [K] buffer as
  // [K,1] with the unit dimension listed first in minor-to-major order.
  for (int64_t physical_dimension :
       input.shape().layout().minor_to_major()) {
    if (physical_dimension == dimension) {
      return true;
    }
    if (input.shape().dimensions(physical_dimension) != 1) {
      return false;
    }
  }
  return false;
}

bool IsContractingDimensionContiguous(const HloInstruction& input,
                                      int64_t contracting_dimension) {
  if (IsSupportedBatchInnerTranspose(input)) {
    const HloInstruction* operand = input.operand(0);
    return operand->shape().has_layout() &&
           operand->shape().layout().minor_to_major(0) ==
               input.dimensions(contracting_dimension);
  }
  return IsDimensionContiguous(input, contracting_dimension);
}

bool IsS4DequantizedInput(const HloInstruction& input) {
  const HloInstruction* parameter = FindContractionInputParameter(input);
  return parameter != nullptr && parameter->shape().element_type() == S4;
}

struct ConcatInputInfo {
  int64_t dimension;
  std::vector<int64_t> fragment_sizes;
};

std::optional<ConcatInputInfo> FindSupportedConcatInput(
    const HloInstruction& input) {
  const HloInstruction* concat = &input;
  std::optional<std::vector<int64_t>> output_to_concat_dimensions;
  if (input.opcode() == HloOpcode::kTranspose && input.operand_count() == 1 &&
      input.shape().dimensions_size() == 2 &&
      input.dimensions() == std::vector<int64_t>({1, 0}) &&
      input.operand(0)->opcode() == HloOpcode::kConcatenate) {
    concat = input.operand(0);
    output_to_concat_dimensions = std::vector<int64_t>(
        input.dimensions().begin(), input.dimensions().end());
  }
  if (concat->opcode() != HloOpcode::kConcatenate ||
      concat->operand_count() < 2 || input.shape().element_type() != BF16 ||
      concat->shape().element_type() != BF16) {
    return std::nullopt;
  }
  const int64_t concat_dimension = concat->concatenate_dimension();
  int64_t dimension = concat_dimension;
  if (output_to_concat_dimensions.has_value()) {
    auto it = std::find(output_to_concat_dimensions->begin(),
                        output_to_concat_dimensions->end(), concat_dimension);
    if (it == output_to_concat_dimensions->end()) {
      return std::nullopt;
    }
    dimension = std::distance(output_to_concat_dimensions->begin(), it);
  }
  PrimitiveType parameter_type = PRIMITIVE_TYPE_INVALID;
  std::vector<int64_t> fragment_sizes;
  fragment_sizes.reserve(concat->operand_count());
  for (const HloInstruction* operand : concat->operands()) {
    const HloInstruction* parameter = FindContractionInputParameter(*operand);
    if (parameter == nullptr ||
        (parameter_type != PRIMITIVE_TYPE_INVALID &&
         parameter->shape().element_type() != parameter_type)) {
      return std::nullopt;
    }
    parameter_type = parameter->shape().element_type();
    fragment_sizes.push_back(operand->shape().dimensions(concat_dimension));
  }
  return ConcatInputInfo{dimension, std::move(fragment_sizes)};
}

bool IsSupportedContractionInput(const HloInstruction& input) {
  return FindContractionInputParameter(input) != nullptr ||
         FindSupportedConcatInput(input).has_value();
}

bool IsGlobalSplitKContraction(const HloInstruction& dot) {
  if (dot.shape().dimensions_size() != 3) {
    return false;
  }
  const DotDimensionNumbers& dims = dot.dot_dimension_numbers();
  if (dims.lhs_batch_dimensions_size() != 1 ||
      dims.rhs_batch_dimensions_size() != 1 ||
      dims.lhs_contracting_dimensions_size() != 1 ||
      dims.rhs_contracting_dimensions_size() != 1 ||
      dims.lhs_batch_dimensions(0) != 1 ||
      dims.lhs_contracting_dimensions(0) != 2 ||
      dot.shape().element_type() != F32) {
    return false;
  }
  // SplitkRewriter inserts the partition dimension immediately before each
  // operand's contracting dimension. For an RHS stored as [N,K], that gives
  // [N,S,K]; for the common [K,N] model layout it gives [S,K,N]. Accept both
  // physical forms and let the emitter map batch/K/N from the dot dimensions.
  const int64_t rhs_batch = dims.rhs_batch_dimensions(0);
  const int64_t rhs_contracting = dims.rhs_contracting_dimensions(0);
  return rhs_batch != rhs_contracting &&
         ((rhs_batch == 1 && rhs_contracting == 2) ||
          (rhs_batch == 0 && rhs_contracting == 1));
}

bool IsBatchedContraction(const HloInstruction& dot) {
  const int64_t output_rank = dot.shape().dimensions_size();
  if (output_rank < 3) {
    return false;
  }
  const DotDimensionNumbers& dims = dot.dot_dimension_numbers();
  const int64_t batch_rank = output_rank - 2;
  if (dims.lhs_batch_dimensions_size() != batch_rank ||
      dims.rhs_batch_dimensions_size() != batch_rank ||
      dims.lhs_contracting_dimensions_size() != 1 ||
      dims.rhs_contracting_dimensions_size() != 1) {
    return false;
  }
  const int64_t lhs_contracting = dims.lhs_contracting_dimensions(0);
  const int64_t rhs_contracting = dims.rhs_contracting_dimensions(0);
  if (absl::c_linear_search(dims.lhs_batch_dimensions(), lhs_contracting) ||
      absl::c_linear_search(dims.rhs_batch_dimensions(), rhs_contracting)) {
    return false;
  }
  for (int64_t batch = 0; batch < batch_rank; ++batch) {
    const int64_t lhs_dimension = dims.lhs_batch_dimensions(batch);
    const int64_t rhs_dimension = dims.rhs_batch_dimensions(batch);
    if (lhs_dimension < 0 ||
        lhs_dimension >= dot.operand(0)->shape().dimensions_size() ||
        rhs_dimension < 0 ||
        rhs_dimension >= dot.operand(1)->shape().dimensions_size() ||
        dot.shape().dimensions(batch) !=
            dot.operand(0)->shape().dimensions(lhs_dimension) ||
        dot.shape().dimensions(batch) !=
            dot.operand(1)->shape().dimensions(rhs_dimension)) {
      return false;
    }
  }
  return true;
}

bool IsSupportedContraction(const HloInstruction& dot) {
  const int64_t rank = dot.shape().dimensions_size();
  const bool is_scaled_dot = dot.opcode() == HloOpcode::kScaledDot;
  const int64_t expected_operands = is_scaled_dot ? 4 : 2;
  if (dot.operand_count() != expected_operands) {
    return false;
  }
  const PrimitiveType lhs_type = dot.operand(0)->shape().element_type();
  const PrimitiveType rhs_type = dot.operand(1)->shape().element_type();
  const bool is_bf16 = lhs_type == BF16 && rhs_type == BF16;
  const bool is_f16 = lhs_type == F16 && rhs_type == F16;
  const bool is_f32 = lhs_type == F32 && rhs_type == F32;
  const bool is_fp8 = IsFnuzFp8(lhs_type) && IsFnuzFp8(rhs_type);
  const bool is_int8 = lhs_type == S8 && rhs_type == S8;
  const bool is_int4 = IsS4DequantizedInput(*dot.operand(0)) ||
                       IsS4DequantizedInput(*dot.operand(1));
  // Fly's MFMA kernels implement the storage-type contraction directly. Dot
  // algorithms that decompose F32 inputs into several BF16 contractions must
  // first go through DotAlgorithmRewriter in the Fly fission backend;
  // accepting them here would let autotuning select a numerically different
  // single F32 contraction. On ROCm, Triton intentionally leaves TF32x3
  // undecomposed and lowers it to the native full-precision F32 MFMA. Offer
  // that same, more-accurate implementation directly in Fly. Likewise, the
  // current half-precision kernels use F32 accumulators, so they do not
  // implement the explicit F16/BF16 accumulator algorithms.
  switch (dot.precision_config().algorithm()) {
    case PrecisionConfig::ALG_UNSET:
      break;
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
      if (!is_f16) return false;
      break;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      if (!is_bf16) return false;
      break;
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      // Triton's regular-dot lowering also accepts F16/BF16 storage for this
      // algorithm.  That is numerically valid: products of two half-precision
      // values are exactly representable in F32, and Fly's half-precision MFMA
      // paths already accumulate in F32 before the requested output rounding.
      if (!is_f32 && !is_f16 && !is_bf16) return false;
      break;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      if (!is_f32) return false;
      break;
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
      if (!is_fp8) return false;
      break;
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
    default:
      return false;
  }
  if (rank < 2 ||
      dot.operand(0)->shape().dimensions_size() != rank ||
      dot.operand(1)->shape().dimensions_size() != rank ||
      (!is_bf16 && !is_f16 && !is_f32 && !is_fp8 && !is_int8) ||
      (is_int4 &&
       (is_scaled_dot || dot.shape().dimensions(rank - 2) == 1 ||
        dot.shape().dimensions(rank - 1) == 1)) ||
      !IsSupportedContractionInput(*dot.operand(0)) ||
      !IsSupportedContractionInput(*dot.operand(1)) ||
      (dot.shape().element_type() != F32 &&
       !(is_bf16 && dot.shape().element_type() == BF16) &&
       !(is_f16 && dot.shape().element_type() == F16) &&
       !(is_fp8 && dot.shape().element_type() == BF16) &&
       !(is_int8 && dot.shape().element_type() == S32))) {
    return false;
  }
  const DotDimensionNumbers& dims = dot.dot_dimension_numbers();
  if (dims.lhs_contracting_dimensions_size() != 1 ||
      dims.rhs_contracting_dimensions_size() != 1) {
    return false;
  }
  const std::optional<ContractingScaleInput> lhs_contracting_scale =
      MatchContractingScaleInput(*dot.operand(0));
  const std::optional<ContractingScaleInput> rhs_contracting_scale =
      MatchContractingScaleInput(*dot.operand(1));
  bool supported_lhs_scale = !lhs_contracting_scale.has_value();
  if (lhs_contracting_scale.has_value()) {
    const std::vector<int64_t>& scale_dimensions =
        lhs_contracting_scale->broadcast_dimensions;
    const int64_t lhs_contracting = dims.lhs_contracting_dimensions(0);
    int64_t lhs_noncontracting = -1;
    for (int64_t dimension = 0;
         dimension < dot.operand(0)->shape().dimensions_size(); ++dimension) {
      if (dimension != lhs_contracting &&
          !absl::c_linear_search(dims.lhs_batch_dimensions(), dimension)) {
        lhs_noncontracting = dimension;
        break;
      }
    }
    const bool contracting_scale =
        scale_dimensions == std::vector<int64_t>{lhs_contracting};
    const bool learned_half_scale =
        rank == 2 && is_bf16 &&
        (contracting_scale ||
         scale_dimensions == std::vector<int64_t>{lhs_noncontracting});
    const bool s4_channel_scale =
        IsS4DequantizedInput(*dot.operand(0)) &&
        !absl::c_linear_search(scale_dimensions, lhs_contracting) &&
        absl::c_linear_search(scale_dimensions, lhs_noncontracting);
    supported_lhs_scale =
        (lhs_contracting_scale->collapsed_subchannel_scale &&
         IsS4DequantizedInput(*dot.operand(0))) ||
        learned_half_scale ||
        (contracting_scale && dot.shape().dimensions(rank - 2) <= 8 &&
         dot.shape().dimensions(rank - 1) > 1) ||
        s4_channel_scale;
  }
  if (rhs_contracting_scale.has_value() || !supported_lhs_scale) {
    return false;
  }
  if (is_scaled_dot) {
    const int64_t m = dot.shape().dimensions(rank - 2);
    const int64_t n = dot.shape().dimensions(rank - 1);
    const bool uniform_scale =
        IsUniformScale(*dot.operand(2)) && IsUniformScale(*dot.operand(3));
    if (is_f32 || is_int8 || (rank >= 3 && !IsBatchedContraction(dot)) ||
        ((m == 1 || n == 1) && !uniform_scale) ||
        !ScaledDotKBlockSize(dot).has_value()) {
      return false;
    }
    if (!uniform_scale) {
      const int64_t scale_block = *ScaledDotKBlockSize(dot);
      const int64_t atom_k = is_fp8 ? 32 : 16;
      if (scale_block < atom_k || scale_block % atom_k != 0 ||
          scale_block > 256) {
        return false;
      }
    }
  }
  const std::optional<ConcatInputInfo> lhs_concat =
      FindSupportedConcatInput(*dot.operand(0));
  const std::optional<ConcatInputInfo> rhs_concat =
      FindSupportedConcatInput(*dot.operand(1));
  if ((lhs_concat.has_value() || rhs_concat.has_value()) &&
      (is_scaled_dot || rank != 2 || dot.shape().dimensions(0) == 1 ||
       dot.shape().dimensions(1) == 1 ||
       (lhs_concat.has_value() &&
        lhs_concat->dimension != 1 - dims.lhs_contracting_dimensions(0)) ||
       (rhs_concat.has_value() &&
        rhs_concat->dimension != 1 - dims.rhs_contracting_dimensions(0)))) {
    return false;
  }
  if (rank == 2) {
    return dims.lhs_batch_dimensions().empty() &&
           dims.rhs_batch_dimensions().empty() &&
           (dims.lhs_contracting_dimensions(0) == 0 ||
            dims.lhs_contracting_dimensions(0) == 1) &&
           (dims.rhs_contracting_dimensions(0) == 0 ||
            dims.rhs_contracting_dimensions(0) == 1);
  }
  // SplitkRewriter uses the middle dimension as a batch of K partitions.
  // Ordinary batched GEMM instead keeps batch as the major dimension and may
  // store the RHS as either [B,K,N] or [B,N,K].
  return IsGlobalSplitKContraction(dot) || IsBatchedContraction(dot);
}

bool ContainsInstruction(const HloInstruction& root,
                         const HloInstruction& target) {
  if (&root == &target) {
    return true;
  }
  return std::any_of(root.operands().begin(), root.operands().end(),
                     [&](const HloInstruction* operand) {
                       return ContainsInstruction(*operand, target);
                     });
}

bool IsNarrowingConvert(const HloInstruction& value) {
  return value.opcode() == HloOpcode::kConvert && value.operand_count() == 1 &&
         value.operand(0)->shape().element_type() == F32 &&
         (value.shape().element_type() == BF16 ||
          value.shape().element_type() == F16) &&
         ShapeUtil::SameDimensions(value.shape(), value.operand(0)->shape());
}

bool IsWideningConvertOf(const HloInstruction& value,
                         const HloInstruction& operand) {
  return value.opcode() == HloOpcode::kConvert && value.operand_count() == 1 &&
         value.operand(0) == &operand && value.shape().element_type() == F32 &&
         IsHalfPrecisionFloat(operand.shape().element_type()) &&
         ShapeUtil::SameDimensions(value.shape(), operand.shape());
}

bool IsSupportedBroadcastInput(const HloInstruction& broadcast_input,
                               const HloInstruction& operation) {
  const HloInstruction* broadcast = &broadcast_input;
  if (broadcast_input.opcode() == HloOpcode::kConvert) {
    if (broadcast_input.operand_count() != 1 ||
        broadcast_input.shape().element_type() != F32 ||
        !ShapeUtil::SameDimensions(broadcast_input.shape(),
                                   broadcast_input.operand(0)->shape()) ||
        !IsHalfPrecisionFloat(
            broadcast_input.operand(0)->shape().element_type())) {
      return false;
    }
    broadcast = broadcast_input.operand(0);
  }
  if (broadcast->opcode() != HloOpcode::kBroadcast ||
      broadcast->operand_count() != 1 || broadcast->dimensions().size() > 1 ||
      !ShapeUtil::SameDimensions(broadcast->shape(), operation.shape()) ||
      broadcast_input.shape().element_type() !=
          operation.shape().element_type()) {
    return false;
  }
  const HloInstruction* input = broadcast->operand(0);
  const int64_t input_rank = input->shape().dimensions_size();
  const int64_t output_rank = operation.shape().dimensions_size();
  const bool scalar = input_rank == 0 && broadcast->dimensions().empty();
  const bool vector =
      input_rank == 1 && broadcast->dimensions().size() == 1 &&
      output_rank >= 2 &&
      (broadcast->dimensions(0) == output_rank - 2 ||
       broadcast->dimensions(0) == output_rank - 1 ||
       (output_rank == 3 && broadcast->dimensions(0) == 0)) &&
      input->shape().dimensions(0) ==
          operation.shape().dimensions(broadcast->dimensions(0));
  const bool supported_value =
      input->opcode() == HloOpcode::kParameter ||
      (scalar && input->opcode() == HloOpcode::kConstant &&
       input->literal().GetAsDouble({}).has_value());
  return supported_value && (scalar || vector) &&
         input->shape().element_type() == broadcast->shape().element_type();
}

bool IsSupportedBinaryEpilogue(HloOpcode opcode) {
  return opcode == HloOpcode::kAdd || opcode == HloOpcode::kMultiply ||
         opcode == HloOpcode::kSubtract || opcode == HloOpcode::kDivide ||
         opcode == HloOpcode::kMaximum || opcode == HloOpcode::kMinimum;
}

// Matches the layout transform between the attention-value contraction and
// the output projection:
//
//   [B*H, M, N] -> bitcast [B, H, M, N]
//                -> transpose [B, N, H, M]
//
// The final M dimension is physically contiguous. The Fly emitter writes each
// lane's four-row MFMA fragment directly with a packed store.
std::optional<int64_t> OutputTransposeBatchInner(const HloInstruction& root,
                                                 const HloInstruction& dot) {
  if (dot.shape().element_type() != BF16 ||
      dot.shape().dimensions_size() != 3 ||
      root.opcode() != HloOpcode::kTranspose || root.operand_count() != 1 ||
      root.shape().element_type() != BF16 ||
      root.shape().dimensions_size() != 4 ||
      root.dimensions() != std::vector<int64_t>({0, 3, 1, 2})) {
    return std::nullopt;
  }
  const HloInstruction* bitcast = root.operand(0);
  if (bitcast->opcode() != HloOpcode::kBitcast ||
      bitcast->operand_count() != 1 || bitcast->operand(0) != &dot ||
      bitcast->shape().dimensions_size() != 4) {
    return std::nullopt;
  }
  const int64_t batch_outer = bitcast->shape().dimensions(0);
  const int64_t batch_inner = bitcast->shape().dimensions(1);
  if (batch_outer <= 0 || batch_inner <= 0 ||
      batch_outer * batch_inner != dot.shape().dimensions(0) ||
      bitcast->shape().dimensions(2) != dot.shape().dimensions(1) ||
      bitcast->shape().dimensions(3) != dot.shape().dimensions(2)) {
    return std::nullopt;
  }
  return batch_inner;
}

bool IsSupportedEpilogue(const HloInstruction& instr,
                         const HloInstruction& dot) {
  const HloInstruction* root =
      instr.fused_instructions_computation()->root_instruction();
  if (root == &dot) {
    return true;
  }
  if (OutputTransposeBatchInner(*root, dot).has_value()) {
    return true;
  }
  const int64_t output_rank = dot.shape().dimensions_size();
  if ((output_rank != 2 && output_rank != 3) ||
      (output_rank == 3 && !IsBatchedContraction(dot))) {
    return false;
  }

  // XLA commonly narrows an FP32 accumulator directly or after the
  // elementwise epilogue. Strip that final output conversion first.
  const HloInstruction* value = root;
  if (root->opcode() == HloOpcode::kConvert) {
    if (!IsNarrowingConvert(*root)) {
      return false;
    }
    value = root->operand(0);
  }
  if (value == &dot) {
    return true;
  }

  // Preserve shared contraction operands as a one-step accumulator epilogue.
  // The GEMM emitter evaluates the dot once and feeds that accumulator to both
  // sides of the elementwise operation.
  if (IsSupportedBinaryEpilogue(value->opcode()) &&
      value->operand_count() == 2 && value->operand(0) == &dot &&
      value->operand(1) == &dot &&
      ShapeUtil::SameDimensions(value->shape(), dot.shape()) &&
      value->shape().element_type() == dot.shape().element_type()) {
    return true;
  }

  // Accept a bounded chain of scalar operations, row/column/batch broadcasts,
  // min/max activations, and negate. A small fixed grammar keeps autotuner
  // eligibility predictable while covering alpha/beta and bias/ReLU patterns.
  // A narrowing conversion on the contraction side of any step is retained as
  // an explicit HLO rounding boundary.
  if (root->shape().element_type() != BF16 &&
      root->shape().element_type() != F16) {
    return false;
  }

  constexpr int64_t kMaxEpilogueSteps = 3;
  for (int64_t step = 0; step < kMaxEpilogueSteps; ++step) {
    if (value == &dot ||
        (IsNarrowingConvert(*value) && value->operand(0) == &dot) ||
        IsWideningConvertOf(*value, dot)) {
      return true;
    }
    if (!ShapeUtil::SameDimensions(value->shape(), dot.shape())) {
      return false;
    }

    const HloInstruction* contraction_value = nullptr;
    if (value->opcode() == HloOpcode::kNegate && value->operand_count() == 1 &&
        ContainsInstruction(*value->operand(0), dot) &&
        value->operand(0)->shape().element_type() ==
            value->shape().element_type()) {
      contraction_value = value->operand(0);
    } else if (IsSupportedBinaryEpilogue(value->opcode()) &&
               value->operand_count() == 2) {
      const bool lhs_contains = ContainsInstruction(*value->operand(0), dot);
      const bool rhs_contains = ContainsInstruction(*value->operand(1), dot);
      if (lhs_contains == rhs_contains) {
        return false;
      }
      const int64_t contraction_operand = lhs_contains ? 0 : 1;
      contraction_value = value->operand(contraction_operand);
      if (!IsSupportedBroadcastInput(*value->operand(1 - contraction_operand),
                                     *value) ||
          contraction_value->shape().element_type() !=
              value->shape().element_type()) {
        return false;
      }
    } else {
      return false;
    }

    if (IsNarrowingConvert(*contraction_value)) {
      contraction_value = contraction_value->operand(0);
    }
    value = contraction_value;
  }
  return value == &dot ||
         (IsNarrowingConvert(*value) && value->operand(0) == &dot) ||
         IsWideningConvertOf(*value, dot);
}

std::unique_ptr<BackendConfig> MakeConfig(
    int64_t block_m, int64_t block_n, int64_t block_k, int64_t num_warps,
    FlyGemmConfig::MfmaAtom mfma_atom, bool prefetch_rhs = false,
    bool stage_output = false, int32_t waves_per_eu = 0,
    bool schedule_instructions = false, bool stage_rhs = false,
    bool async_lhs = false, bool preload_lds_fragments = false,
    bool single_buffer_lds = false, bool direct_to_vgpr = false,
    bool rolling_refill = false, bool local_split_k = false,
    int32_t gemv_outputs_per_wave = 0, int32_t gemv_k_vector_width = 0,
    bool gemv_split_k = false) {
  auto config = std::make_unique<BackendConfig>();
  FlyGemmConfig* key = config->mutable_fly();
  key->set_block_m(block_m);
  key->set_block_n(block_n);
  key->set_block_k(block_k);
  key->set_num_warps(num_warps);
  key->set_mfma_atom(mfma_atom);
  key->set_prefetch_rhs(prefetch_rhs);
  key->set_stage_output(stage_output);
  key->set_waves_per_eu(waves_per_eu);
  key->set_schedule_instructions(schedule_instructions);
  key->set_stage_rhs(stage_rhs);
  key->set_async_lhs(async_lhs);
  key->set_preload_lds_fragments(preload_lds_fragments);
  key->set_single_buffer_lds(single_buffer_lds);
  key->set_direct_to_vgpr(direct_to_vgpr);
  key->set_rolling_refill(rolling_refill);
  key->set_local_split_k(local_split_k);
  key->set_gemv_outputs_per_wave(gemv_outputs_per_wave);
  key->set_gemv_k_vector_width(gemv_k_vector_width);
  key->set_gemv_split_k(gemv_split_k);
  return config;
}

std::unique_ptr<BackendConfig> MakeGemvConfig(
    int64_t block_m, int64_t block_n, int64_t num_warps,
    int32_t outputs_per_wave = 0, int32_t k_vector_width = 0,
    FlyGemmConfig::MfmaAtom mfma_atom = FlyGemmConfig::FLY_MFMA_16X16X16,
    bool split_k = false, int64_t block_k = 32) {
  return MakeConfig(block_m, block_n, block_k, num_warps, mfma_atom,
                    /*prefetch_rhs=*/false, /*stage_output=*/false,
                    /*waves_per_eu=*/0, /*schedule_instructions=*/false,
                    /*stage_rhs=*/false, /*async_lhs=*/false,
                    /*preload_lds_fragments=*/false,
                    /*single_buffer_lds=*/false, /*direct_to_vgpr=*/false,
                    /*rolling_refill=*/false, /*local_split_k=*/false,
                    outputs_per_wave, k_vector_width, split_k);
}

}  // namespace

bool FlyBackend::IsSupported(const HloInstruction& instr) {
  if (!debug_options().xla_gpu_enable_flydsl_gemm()) {
    return false;
  }
  if (!target_config().device_description.gpu_compute_capability().IsRocm()) {
    return false;
  }
  const HloInstruction* dot = FindContraction(instr);
  return dot != nullptr && IsSupportedContraction(*dot) &&
         IsSupportedEpilogue(instr, *dot);
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
FlyBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }

  const HloInstruction* dot = FindContraction(instr);
  const PrimitiveType output_element_type =
      instr.fused_instructions_computation()
          ->root_instruction()
          ->shape()
          .element_type();
  const bool is_f16 = dot->operand(0)->shape().element_type() == F16 &&
                      dot->operand(1)->shape().element_type() == F16;
  const bool is_f32 = dot->operand(0)->shape().element_type() == F32 &&
                      dot->operand(1)->shape().element_type() == F32;
  const bool is_fp8 = IsFnuzFp8(dot->operand(0)->shape().element_type());
  const bool is_int8 = dot->operand(0)->shape().element_type() == S8 &&
                       dot->operand(1)->shape().element_type() == S8;
  const bool is_int4 = IsS4DequantizedInput(*dot->operand(0)) ||
                       IsS4DequantizedInput(*dot->operand(1));
  const int64_t output_rank = dot->shape().dimensions_size();
  const bool rank_three = output_rank == 3;
  const bool global_split_k = IsGlobalSplitKContraction(*dot);
  const int64_t m = dot->shape().dimensions(output_rank - 2);
  const int64_t n = dot->shape().dimensions(output_rank - 1);
  const int64_t k = dot->operand(0)->shape().dimensions(
      dot->dot_dimension_numbers().lhs_contracting_dimensions(0));
  const int64_t lhs_contracting_dimension =
      dot->dot_dimension_numbers().lhs_contracting_dimensions(0);
  const int64_t rhs_contracting_dimension =
      dot->dot_dimension_numbers().rhs_contracting_dimensions(0);
  int64_t rhs_noncontracting_dimension = -1;
  for (int64_t dimension = 0;
       dimension < dot->operand(1)->shape().dimensions_size(); ++dimension) {
    if (dimension == rhs_contracting_dimension ||
        absl::c_linear_search(
            dot->dot_dimension_numbers().rhs_batch_dimensions(), dimension)) {
      continue;
    }
    rhs_noncontracting_dimension = dimension;
    break;
  }
  const bool is_scaled_dot = dot->opcode() == HloOpcode::kScaledDot;
  const bool is_block_scaled_dot =
      is_scaled_dot &&
      (!IsUniformScale(*dot->operand(2)) || !IsUniformScale(*dot->operand(3)));
  const bool has_output_transpose =
      OutputTransposeBatchInner(
          *instr.fused_instructions_computation()->root_instruction(), *dot)
          .has_value();
  const bool lhs_k_contiguous = IsContractingDimensionContiguous(
      *dot->operand(0), lhs_contracting_dimension);
  const bool rhs_k_contiguous = IsContractingDimensionContiguous(
      *dot->operand(1), rhs_contracting_dimension);
  const bool rhs_column_contiguous =
      rhs_noncontracting_dimension >= 0 &&
      IsDimensionContiguous(*dot->operand(1), rhs_noncontracting_dimension);
  const bool has_lhs_input_scale =
      MatchContractingScaleInput(*dot->operand(0)).has_value();
  const std::optional<ConcatInputInfo> lhs_concat =
      FindSupportedConcatInput(*dot->operand(0));
  const std::optional<ConcatInputInfo> rhs_concat =
      FindSupportedConcatInput(*dot->operand(1));
  const int64_t contraction_atom_k = (is_fp8 || is_int8) ? 32 : 16;
  const bool supports_masked_k_tail = !is_fp8 && !is_int8 && !is_int4 &&
                                      !is_block_scaled_dot &&
                                      !global_split_k;
  if (k % contraction_atom_k != 0 && !supports_masked_k_tail) {
    return configs;
  }

  constexpr std::array<int64_t, 5> kBlockSizes = {16, 32, 64, 128, 256};
  const bool masked_k_tail = k % contraction_atom_k != 0;
  auto supports_gemm_k_tile = [&](int64_t block_k) {
    if (!masked_k_tail) {
      return block_k <= k && k % block_k == 0;
    }
    // Admit every complete power-of-two stage and the first stage larger than
    // K. The latter avoids forcing very short contractions into K16 without
    // also offering successively larger mostly-zero stages.
    return block_k == 16 || block_k <= k || block_k / 2 < k;
  };
  if (is_int8) {
    // Follow FlyDSL's CDNA3 preshuffle GEMM geometry: a 16x16x32 signed-byte
    // MFMA, four-wave macro-tiles, and K stages made from whole K32 atoms.
    // XLA owns ordinary (not preshuffled) buffers, so start with the generic
    // LHS-LDS/RHS-register transport that accepts either logical RHS layout.
    // Dedicated two-operand LDS candidates can be layered on after this broad
    // correctness path is established.
    if (m == 1 || n == 1 || k % 32 != 0) {
      return configs;
    }
    constexpr std::array<int64_t, 5> kInt8BlockSizes = {16, 32, 64, 128, 256};
    for (int64_t block_m : kInt8BlockSizes) {
      if (block_m > m || m % block_m != 0) {
        continue;
      }
      for (int64_t block_n : kInt8BlockSizes) {
        if (block_n > n || n % block_n != 0) {
          continue;
        }
        const int64_t wave_tiles = (block_m / 16) * (block_n / 16);
        for (int64_t block_k : {32, 64, 128, 256}) {
          if (block_k > k || k % block_k != 0 ||
              2 * block_m * block_k > 64 * 1024) {
            continue;
          }
          for (int64_t num_warps : {1, 2, 4, 8}) {
            if (num_warps > wave_tiles || wave_tiles % num_warps != 0) {
              continue;
            }
            configs.push_back(MakeConfig(
                block_m, block_n, block_k, num_warps,
                FlyGemmConfig::FLY_MFMA_16X16X32_I8));
            // FlyDSL's preshuffle GEMM keeps A in a two-stage LDS pipeline and
            // B in a two-stage VGPR pipeline.  This is the corresponding XLA
            // transport: `prefetch_rhs` carries the next complete B fragment
            // bank while the current bank feeds the MFMAs.  Keep the ordinary
            // streaming variant because N-contiguous XLA weights do not have
            // FlyDSL's K-contiguous/preshuffled B representation.
            const bool flydsl_register_rhs_pipeline =
                num_warps == 4 && block_m >= 64 && block_n >= 64 &&
                block_k >= 64 && block_k <= 128 && k >= 2 * block_k;
            if (flydsl_register_rhs_pipeline) {
              configs.push_back(MakeConfig(
                  block_m, block_n, block_k, num_warps,
                  FlyGemmConfig::FLY_MFMA_16X16X32_I8,
                  /*prefetch_rhs=*/true));
            }
            const int64_t two_operand_lds_bytes =
                2 * block_k * (block_m + block_n);
            const bool flydsl_two_operand_pipeline =
                block_m >= 64 && block_n >= 64 && block_k >= 64 &&
                num_warps >= 4 && two_operand_lds_bytes <= 64 * 1024;
            const bool native_preloaded_shape =
                num_warps == 4 &&
                ((block_k == 64 && block_m == block_n &&
                  (block_m == 64 || block_m == 128)) ||
                 (block_k == 128 && block_m == 128 && block_n == 128));
            if (flydsl_two_operand_pipeline) {
              configs.push_back(MakeConfig(
                  block_m, block_n, block_k, num_warps,
                  FlyGemmConfig::FLY_MFMA_16X16X32_I8,
                  /*prefetch_rhs=*/false, /*stage_output=*/false,
                  /*waves_per_eu=*/0, /*schedule_instructions=*/false,
                  /*stage_rhs=*/true));
              if (native_preloaded_shape) {
                configs.push_back(MakeConfig(
                    block_m, block_n, block_k, num_warps,
                    FlyGemmConfig::FLY_MFMA_16X16X32_I8,
                    /*prefetch_rhs=*/false, /*stage_output=*/false,
                    /*waves_per_eu=*/0, /*schedule_instructions=*/false,
                    /*stage_rhs=*/true, /*async_lhs=*/false,
                    /*preload_lds_fragments=*/true));
              }
            }
            if (k >= 1024) {
              configs.push_back(MakeConfig(
                  block_m, block_n, block_k, num_warps,
                  FlyGemmConfig::FLY_MFMA_16X16X32_I8,
                  /*prefetch_rhs=*/false, /*stage_output=*/false,
                  /*waves_per_eu=*/0, /*schedule_instructions=*/true));
              if (flydsl_register_rhs_pipeline) {
                configs.push_back(MakeConfig(
                    block_m, block_n, block_k, num_warps,
                    FlyGemmConfig::FLY_MFMA_16X16X32_I8,
                    /*prefetch_rhs=*/true, /*stage_output=*/false,
                    /*waves_per_eu=*/0, /*schedule_instructions=*/true));
              }
              if (flydsl_two_operand_pipeline) {
                configs.push_back(MakeConfig(
                    block_m, block_n, block_k, num_warps,
                    FlyGemmConfig::FLY_MFMA_16X16X32_I8,
                    /*prefetch_rhs=*/false, /*stage_output=*/false,
                    /*waves_per_eu=*/0, /*schedule_instructions=*/true,
                    /*stage_rhs=*/true));
                if (native_preloaded_shape) {
                  configs.push_back(MakeConfig(
                      block_m, block_n, block_k, num_warps,
                      FlyGemmConfig::FLY_MFMA_16X16X32_I8,
                      /*prefetch_rhs=*/false, /*stage_output=*/false,
                      /*waves_per_eu=*/0, /*schedule_instructions=*/true,
                      /*stage_rhs=*/true, /*async_lhs=*/false,
                      /*preload_lds_fragments=*/true));
                }
              }
            }
          }
        }
      }
    }
    return configs;
  }
  if (m == 1) {
    if (is_f32) {
      return configs;
    }
    constexpr std::array<int64_t, 9> kGemvOutputBlockSizes = {
        1, 2, 4, 8, 16, 32, 64, 128, 256};
    for (int64_t output_block : kGemvOutputBlockSizes) {
      if (output_block > n && output_block / 2 >= n) {
        continue;
      }
      for (int64_t num_warps : {1, 2, 4, 8}) {
        const int64_t wave_tiles = std::max<int64_t>(1, output_block / 16);
        if (!rhs_k_contiguous) {
          if (num_warps <= wave_tiles && wave_tiles % num_warps == 0) {
            configs.push_back(MakeGemvConfig(
                /*block_m=*/16, output_block, num_warps,
                /*outputs_per_wave=*/0, /*k_vector_width=*/0,
                is_fp8 ? FlyGemmConfig::FLY_MFMA_16X16X32_FP8
                       : FlyGemmConfig::FLY_MFMA_16X16X16));
          }
          if (output_block >= 64 && output_block % 64 == 0 && num_warps > 1) {
            configs.push_back(MakeGemvConfig(
                /*block_m=*/16, output_block, num_warps,
                /*outputs_per_wave=*/0, /*k_vector_width=*/0,
                is_fp8 ? FlyGemmConfig::FLY_MFMA_16X16X32_FP8
                       : FlyGemmConfig::FLY_MFMA_16X16X16,
                /*split_k=*/true));
          }
          continue;
        }
        if (num_warps > wave_tiles || wave_tiles % num_warps != 0) {
          continue;
        }
        for (int32_t outputs_per_wave : {1, 2, 4, 8}) {
          if (outputs_per_wave > output_block) {
            continue;
          }
          for (int32_t k_vector_width : {1, 2, 4}) {
            if (k_vector_width == 4 && !is_fp8) {
              continue;
            }
            if (k_vector_width > 1 && k % (64 * k_vector_width) != 0) {
              continue;
            }
            configs.push_back(MakeGemvConfig(
                /*block_m=*/16, output_block, num_warps, outputs_per_wave,
                k_vector_width,
                is_fp8 ? FlyGemmConfig::FLY_MFMA_16X16X32_FP8
                       : FlyGemmConfig::FLY_MFMA_16X16X16));
          }
        }
      }
    }
    if (!is_fp8 && !is_int4 && !is_block_scaled_dot && !global_split_k &&
        lhs_k_contiguous && n % 16 == 0 && k >= 512 && k % 256 == 0 &&
        dot->operand(0)->shape().element_type() ==
            dot->operand(1)->shape().element_type()) {
      // Match Triton's successful row-vector tile while using Fly's cheaper
      // local split: two or four waves cooperate on one 16x16 MFMA atom and
      // reduce their K partitions through LDS.
      for (int64_t num_warps : {2, 4}) {
        for (int32_t waves_per_eu : {0, 2, 4}) {
          configs.insert(
              configs.begin(),
              MakeConfig(
                  /*block_m=*/16, /*block_n=*/16, /*block_k=*/256, num_warps,
                  FlyGemmConfig::FLY_MFMA_16X16X16,
                  /*prefetch_rhs=*/false, /*stage_output=*/false, waves_per_eu,
                  /*schedule_instructions=*/false,
                  /*stage_rhs=*/true, /*async_lhs=*/false,
                  /*preload_lds_fragments=*/true,
                  /*single_buffer_lds=*/false, /*direct_to_vgpr=*/false,
                  /*rolling_refill=*/false, /*local_split_k=*/true));
        }
      }
    }
    return configs;
  }
  if (n == 1) {
    if (is_f32) {
      return configs;
    }
    for (int64_t output_block : kBlockSizes) {
      if (output_block > m && output_block / 2 >= m) {
        continue;
      }
      for (int64_t num_warps : {1, 2, 4, 8}) {
        for (int32_t outputs_per_wave : {1, 2, 4, 8}) {
          if (output_block % outputs_per_wave != 0 ||
              num_warps * outputs_per_wave > output_block) {
            continue;
          }
          for (int32_t k_vector_width : {1, 2, 4}) {
            if (k_vector_width > 1 &&
                (!lhs_k_contiguous || !rhs_k_contiguous)) {
              continue;
            }
            if (k_vector_width > 1 && k % (64 * k_vector_width) != 0) {
              continue;
            }
            configs.push_back(
                MakeGemvConfig(output_block, /*block_n=*/16, num_warps,
                               outputs_per_wave, k_vector_width,
                               is_fp8 ? FlyGemmConfig::FLY_MFMA_16X16X32_FP8
                                      : FlyGemmConfig::FLY_MFMA_16X16X16));
          }
        }
      }
    }
    return configs;
  }

  if (is_block_scaled_dot) {
    // gfx942 has native FNUZ/BF16/F16 MFMA but no CDNA4 scaled-MFMA atom.
    // Compute exactly one software scale interval into a temporary FP32
    // accumulator, apply its row/column scales, and then add it to the running
    // accumulator. Keeping block_k equal to the scale interval is required for
    // correctness; the generic LDS/register pipeline is used until a dedicated
    // modulo schedule is tuned for the extra scale loads and VALU work.
    const int64_t block_k = *ScaledDotKBlockSize(*dot);
    for (int64_t block_m : kBlockSizes) {
      if (block_m > m || m % block_m != 0) {
        continue;
      }
      for (int64_t block_n : kBlockSizes) {
        if (block_n > n || n % block_n != 0) {
          continue;
        }
        const int64_t mfma16_tiles = (block_m / 16) * (block_n / 16);
        const int64_t mfma32_tiles = (block_m / 32) * (block_n / 32);
        for (int64_t num_warps : {1, 2, 4, 8}) {
          auto add_variants = [&](FlyGemmConfig::MfmaAtom atom,
                                  bool dequantize_block_scales = false) {
            for (bool schedule : {false, true}) {
              std::unique_ptr<BackendConfig> config =
                  MakeConfig(block_m, block_n, block_k, num_warps, atom,
                             /*prefetch_rhs=*/false, /*stage_output=*/false,
                             /*waves_per_eu=*/0, schedule);
              config->mutable_fly()->set_dequantize_block_scales(
                  dequantize_block_scales);
              configs.push_back(std::move(config));
            }
          };
          if (num_warps <= mfma16_tiles && mfma16_tiles % num_warps == 0) {
            add_variants(is_fp8 ? FlyGemmConfig::FLY_MFMA_16X16X32_FP8
                                : FlyGemmConfig::FLY_MFMA_16X16X16);
            if (is_fp8) {
              add_variants(FlyGemmConfig::FLY_MFMA_16X16X16,
                           /*dequantize_block_scales=*/true);
            }
          }
          if (block_m % 32 == 0 && block_n % 32 == 0 &&
              num_warps <= mfma32_tiles && mfma32_tiles % num_warps == 0) {
            add_variants(is_fp8 ? FlyGemmConfig::FLY_MFMA_32X32X16_FP8
                                : FlyGemmConfig::FLY_MFMA_32X32X8);
            if (is_fp8) {
              add_variants(FlyGemmConfig::FLY_MFMA_32X32X8,
                           /*dequantize_block_scales=*/true);
            }
          }
        }
      }
    }
    return configs;
  }

  if (is_fp8) {
    // Retain the type-generic LHS-LDS/RHS-register pipeline as the broad
    // fallback. Homogeneous FP8 also gets the FlyDSL-style two-operand LDS
    // pipeline below.
    for (int64_t block_m : kBlockSizes) {
      if (block_m > m || m % block_m != 0) {
        continue;
      }
      for (int64_t block_n : kBlockSizes) {
        if (block_n > n || n % block_n != 0) {
          continue;
        }
        for (int64_t block_k : {32, 64, 128, 256}) {
          if (block_k > k || k % block_k != 0 ||
              2 * block_m * block_k > 64 * 1024) {
            continue;
          }
          const int64_t mfma16_tiles = (block_m / 16) * (block_n / 16);
          const int64_t mfma32_tiles = (block_m / 32) * (block_n / 32);
          for (int64_t num_warps : {1, 2, 4, 8}) {
            if (num_warps <= mfma16_tiles && mfma16_tiles % num_warps == 0) {
              configs.push_back(MakeConfig(block_m, block_n, block_k, num_warps,
                                           FlyGemmConfig::FLY_MFMA_16X16X32_FP8,
                                           /*prefetch_rhs=*/false,
                                           /*stage_output=*/false,
                                           /*waves_per_eu=*/0,
                                           /*schedule_instructions=*/false));
              if (k >= 1024) {
                configs.push_back(
                    MakeConfig(block_m, block_n, block_k, num_warps,
                               FlyGemmConfig::FLY_MFMA_16X16X32_FP8,
                               /*prefetch_rhs=*/false, /*stage_output=*/false,
                               /*waves_per_eu=*/0,
                               /*schedule_instructions=*/true));
              }
            }
            if (block_m % 32 == 0 && block_n % 32 == 0 &&
                num_warps <= mfma32_tiles && mfma32_tiles % num_warps == 0) {
              configs.push_back(MakeConfig(block_m, block_n, block_k, num_warps,
                                           FlyGemmConfig::FLY_MFMA_32X32X16_FP8,
                                           /*prefetch_rhs=*/false,
                                           /*stage_output=*/false,
                                           /*waves_per_eu=*/0,
                                           /*schedule_instructions=*/false));
              if (k >= 1024) {
                configs.push_back(
                    MakeConfig(block_m, block_n, block_k, num_warps,
                               FlyGemmConfig::FLY_MFMA_32X32X16_FP8,
                               /*prefetch_rhs=*/false, /*stage_output=*/false,
                               /*waves_per_eu=*/0,
                               /*schedule_instructions=*/true));
              }
            }
          }
        }
      }
    }
    // FlyDSL's CDNA4 FP8 GEMM overlaps 16-byte-per-lane global-to-LDS copies
    // with MFMA work through XOR-swizzled ping-pong tiles. gfx942 can move only
    // one dword per lane directly to LDS, so use that native operation with a
    // smaller 128x128 tile and K-contiguous inputs. Two A+B stages occupy 32
    // KiB at K64 or the full 64 KiB at K128.
    if (rhs_k_contiguous &&
        dot->operand(0)->shape().element_type() ==
            dot->operand(1)->shape().element_type() &&
        m % 128 == 0 && n % 128 == 0 && k % 128 == 0) {
      for (int64_t block_k : {64, 128}) {
        for (int32_t waves_per_eu : {0, 2, 4}) {
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/128, /*block_n=*/128, block_k,
                /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X32_FP8,
                /*prefetch_rhs=*/false, /*stage_output=*/false, waves_per_eu,
                schedule, /*stage_rhs=*/true, /*async_lhs=*/false,
                /*preload_lds_fragments=*/true,
                /*single_buffer_lds=*/false));
          }
        }
      }

      // Match the large-shape gfx942 hipBLASLt/Tensile FP8 algorithm: four
      // waves cover a 256x224 macro-tile, keep the 256-wide A operand in
      // VGPRs, and ping-pong the 224x128 B operand through the 64 KiB LDS.
      // At 4096^3 this produces exactly 16 * ceil(4096 / 224) = 304
      // workgroups, one per MI300X CU.
      if (m % 256 == 0 && n >= 224 && n % 16 == 0) {
        configs.push_back(MakeConfig(
            /*block_m=*/256, /*block_n=*/224, /*block_k=*/128,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X32_FP8,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/false,
            /*direct_to_vgpr=*/true));
      }
    }
    return configs;
  }

  if (is_int4) {
    // Dequantize packed S4 values into the ordinary BF16 MFMA fragments. The
    // generic candidates provide broad tile coverage; the preloaded-fragment
    // candidates below use the packed dword loads shared by the specialized
    // BF16 schedules to overlap dequantization with MFMA work.
    for (int64_t block_m : kBlockSizes) {
      if (block_m > m && block_m / 2 >= m && block_m != 16) {
        continue;
      }
      for (int64_t block_n : kBlockSizes) {
        if (block_n > n && block_n / 2 >= n && block_n != 16) {
          continue;
        }
        for (int64_t block_k : {32, 64, 128, 256}) {
          if (block_k > k || k % block_k != 0 ||
              2 * block_m * block_k * sizeof(uint16_t) > 64 * 1024) {
            continue;
          }
          const int64_t mfma16_tiles = (block_m / 16) * (block_n / 16);
          const int64_t mfma32_tiles = (block_m / 32) * (block_n / 32);
          for (int64_t num_warps : {1, 2, 4, 8}) {
            auto add_candidate = [&](FlyGemmConfig::MfmaAtom atom) {
              for (int32_t waves_per_eu : {0, 2, 4}) {
                configs.push_back(
                    MakeConfig(block_m, block_n, block_k, num_warps, atom,
                               /*prefetch_rhs=*/false, /*stage_output=*/false,
                               waves_per_eu, /*schedule_instructions=*/false));
                if (k >= 1024) {
                  configs.push_back(
                      MakeConfig(block_m, block_n, block_k, num_warps, atom,
                                 /*prefetch_rhs=*/false, /*stage_output=*/false,
                                 waves_per_eu, /*schedule_instructions=*/true));
                }
              }
            };
            if (num_warps <= mfma16_tiles && mfma16_tiles % num_warps == 0) {
              add_candidate(FlyGemmConfig::FLY_MFMA_16X16X16);
              const int64_t block_threads = num_warps * 64;
              const int64_t staged_lds_bytes =
                  2 * (block_m + block_n) * block_k * sizeof(uint16_t);
              const bool supports_staged_output_tile =
                  (m >= block_m && n >= block_n) ||
                  (m >= 2 && m <= 8 && block_m == 16 &&
                   n % block_n == 0);
              if ((block_m * block_k / 2) % block_threads == 0 &&
                  (block_n * block_k / 2) % block_threads == 0 &&
                  staged_lds_bytes <= 64 * 1024 &&
                  supports_staged_output_tile) {
                for (int32_t waves_per_eu : {0, 2, 4}) {
                  for (bool schedule : {false, true}) {
                    configs.push_back(MakeConfig(
                        block_m, block_n, block_k, num_warps,
                        FlyGemmConfig::FLY_MFMA_16X16X16,
                        /*prefetch_rhs=*/false, /*stage_output=*/false,
                        waves_per_eu, schedule, /*stage_rhs=*/true));
                  }
                }
              }
            }
            if (block_m % 32 == 0 && block_n % 32 == 0 &&
                num_warps <= mfma32_tiles && mfma32_tiles % num_warps == 0) {
              add_candidate(FlyGemmConfig::FLY_MFMA_32X32X8);
            }
          }
        }
      }
    }

    // Follow FlyDSL's small-grid B_TO_LDS schedule: keep one A+B tile in LDS,
    // preload its MFMA fragments, and refill the tile while those fragments
    // are consumed. This avoids placing the next tile's complete S4 unpack and
    // LDS write sequence on the critical path before the current tile's MFMAs.
    if (m % 128 == 0 && n % 64 == 0 && k % 128 == 0) {
      for (bool rolling_refill : {false, true}) {
        for (bool stage_output : {false, true}) {
          if (stage_output && output_element_type != BF16) {
            continue;
          }
          for (int32_t waves_per_eu : {0, 2, 4}) {
            for (bool schedule : {false, true}) {
              configs.push_back(MakeConfig(
                  /*block_m=*/128, /*block_n=*/64, /*block_k=*/128,
                  /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
                  /*prefetch_rhs=*/false, stage_output, waves_per_eu, schedule,
                  /*stage_rhs=*/true, /*async_lhs=*/false,
                  /*preload_lds_fragments=*/true,
                  /*single_buffer_lds=*/true, /*direct_to_vgpr=*/false,
                  rolling_refill));
            }
          }
        }
      }
    }

    // Reuse FlyDSL's/Tensile's MI300X wide-tile pipeline when the packed S4
    // operand is the RHS. The BF16 LHS is loaded DirectToVgpr while the
    // 224x64 S4 RHS tile is dequantized into the padded, double-buffered LDS
    // layout. At 4096^3 this launches 16 * ceil(4096 / 224) = 304 workgroups,
    // exactly one per MI300X CU, instead of underfilling the device with the
    // generic 256x256 winner's 256 workgroups.
    if (rhs_k_contiguous && m % 256 == 0 && n >= 224 && n % 16 == 0 &&
        k % 64 == 0 && !IsS4DequantizedInput(*dot->operand(0))) {
      configs.push_back(MakeConfig(
          /*block_m=*/256, /*block_n=*/224, /*block_k=*/64,
          /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
          /*prefetch_rhs=*/false, /*stage_output=*/false,
          /*waves_per_eu=*/0, /*schedule_instructions=*/true,
          /*stage_rhs=*/true, /*async_lhs=*/false,
          /*preload_lds_fragments=*/true,
          /*single_buffer_lds=*/false,
          /*direct_to_vgpr=*/true));
    }

    // Prefer the transposed wide tile when the S4 RHS itself can remain
    // DirectToVgpr. This stages only the BF16 LHS and dequantizes each packed
    // RHS fragment immediately before its MFMA batch, avoiding the BF16 LDS
    // refill traffic of the 256x224 orientation. The 4K grid is again exactly
    // ceil(4096 / 224) * 16 = 304 workgroups.
    if (rhs_k_contiguous && m >= 224 && m % 16 == 0 && n % 256 == 0 &&
        k % 64 == 0 && IsS4DequantizedInput(*dot->operand(1)) &&
        !IsS4DequantizedInput(*dot->operand(0))) {
      configs.push_back(MakeConfig(
          /*block_m=*/224, /*block_n=*/256, /*block_k=*/64,
          /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
          /*prefetch_rhs=*/false, /*stage_output=*/false,
          /*waves_per_eu=*/0, /*schedule_instructions=*/true,
          /*stage_rhs=*/true, /*async_lhs=*/false,
          /*preload_lds_fragments=*/true,
          /*single_buffer_lds=*/false,
          /*direct_to_vgpr=*/true));
    }

    // N-contiguous inputs benefit from FlyDSL's square MFMA32 double buffer:
    // stage packed rows while preloaded LDS fragments feed the current tile.
    if (!rhs_k_contiguous && !has_lhs_input_scale && m % 128 == 0 &&
        n % 128 == 0 && k % 64 == 0) {
      for (bool stage_output : {false, true}) {
        if (stage_output && output_element_type != BF16) {
          continue;
        }
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/128, /*block_k=*/64,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_32X32X8,
            /*prefetch_rhs=*/false, stage_output,
            /*waves_per_eu=*/0, /*schedule_instructions=*/false,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/false));
      }
    }
    return configs;
  }

  auto add_gemm_configs = [&](int64_t block_m, int64_t block_n, int64_t block_k,
                              int64_t num_warps,
                              FlyGemmConfig::MfmaAtom mfma_atom) {
    auto add_variants = [&](bool prefetch_rhs, bool stage_output) {
      configs.push_back(MakeConfig(block_m, block_n, block_k, num_warps,
                                   mfma_atom, prefetch_rhs, stage_output,
                                   /*waves_per_eu=*/0,
                                   /*schedule_instructions=*/false));
      const bool tune_schedule =
          k >= 1024 && block_m * block_n >= 16 * 1024 && !stage_output;
      if (!tune_schedule) {
        return;
      }
      // The AMDGPU instruction scheduler can interleave the next tile's VMEM
      // loads with LDS reads and MFMAs when explicit scheduling groups are
      // present. Keep the unscheduled kernel as a baseline because the best
      // schedule depends on tile shape and register pressure.
      configs.push_back(MakeConfig(block_m, block_n, block_k, num_warps,
                                   mfma_atom, prefetch_rhs, stage_output,
                                   /*waves_per_eu=*/0,
                                   /*schedule_instructions=*/true));
      // FlyDSL kernels tune occupancy as well as instruction order. The common
      // MLIR kernel compiler applies this as amdgpu-waves-per-eu.
      configs.push_back(MakeConfig(block_m, block_n, block_k, num_warps,
                                   mfma_atom, prefetch_rhs, stage_output,
                                   /*waves_per_eu=*/2,
                                   /*schedule_instructions=*/true));
      // Four waves/EU improved the best unscheduled high-occupancy tile by
      // about 7% on MI300X, while one wave/EU consistently regressed.
      configs.push_back(MakeConfig(block_m, block_n, block_k, num_warps,
                                   mfma_atom, prefetch_rhs, stage_output,
                                   /*waves_per_eu=*/4,
                                   /*schedule_instructions=*/false));
    };
    add_variants(/*prefetch_rhs=*/false, /*stage_output=*/false);
    // FlyDSL GEMMs transpose accumulator fragments through LDS so the global
    // epilogue can use packed stores. Keep it as an autotune dimension because
    // reserving the whole output tile in LDS can reduce occupancy.
    constexpr int64_t kMinStagedOutputElements = 8 * 1024;
    constexpr int64_t kMaxStagedOutputElements = 32 * 1024;
    const bool can_stage_output =
        output_element_type == BF16 && m % block_m == 0 && n % block_n == 0 &&
        (has_output_transpose ||
         block_m * block_n >= kMinStagedOutputElements) &&
        block_m * block_n <= kMaxStagedOutputElements;
    if (can_stage_output) {
      add_variants(/*prefetch_rhs=*/false, /*stage_output=*/true);
    }
    // Register-pipelining the RHS can overlap VMEM latency with MFMA, but its
    // extra live vectors are not universally profitable. Preserve both
    // variants so autotuning makes that tradeoff for the actual shape.
    if (k >= 1024 && block_k <= 64) {
      add_variants(/*prefetch_rhs=*/true, /*stage_output=*/false);
      if (can_stage_output) {
        add_variants(/*prefetch_rhs=*/true, /*stage_output=*/true);
      }
    }
  };

  auto add_generic_gemm_configs = [&] {
    if (!debug_options().xla_gpu_exhaustive_tiling_search()) {
      // Follow Triton's online-autotuning policy: use a compact set of
      // architecture-tuned output/K tiles by default and reserve the full
      // Cartesian product for --xla_gpu_exhaustive_tiling_search.  Compiling
      // the product is especially expensive for Fly because schedule,
      // occupancy, output-staging, and MFMA-atom variants are real machine
      // schedules rather than late metadata changes.
      //
      // The first group mirrors the ROCm/MI300 Triton hint set after rounding
      // its N8 tiles up to Fly's minimum N16 MFMA atom.  The final group keeps
      // the compact tiles selected by the audited transformer-training graph;
      // those cover shallow projection gradients without reopening the full
      // product.
      constexpr std::array<std::array<int64_t, 4>, 40> kDefaultTiles = {{
          {32, 32, 256, 4},
          {64, 32, 32, 4},
          {32, 64, 64, 4},
          {128, 128, 64, 4},
          {16, 16, 256, 1},
          {16, 128, 32, 4},
          {256, 256, 32, 8},
          {128, 256, 64, 8},
          {128, 256, 32, 4},
          {256, 128, 64, 8},
          {32, 16, 16, 2},
          {64, 32, 16, 2},
          {128, 32, 16, 4},
          {128, 64, 128, 8},
          {128, 128, 32, 4},
          {256, 128, 32, 4},
          {128, 32, 32, 4},
          {64, 32, 128, 2},
          {32, 32, 32, 2},
          {32, 16, 128, 2},
          {64, 16, 128, 2},
          {32, 16, 256, 2},
          {128, 16, 128, 8},
          {64, 16, 16, 2},
          {128, 16, 16, 2},
          {256, 16, 32, 2},
          {16, 16, 16, 1},
          {16, 16, 32, 1},
          {16, 16, 64, 1},
          {16, 16, 128, 1},
          {32, 16, 32, 2},
          {32, 16, 64, 2},
          {64, 32, 64, 4},
          {32, 64, 16, 4},
          {16, 128, 64, 4},
          {16, 128, 128, 4},
          {32, 32, 64, 4},
          {32, 32, 128, 4},
          {64, 64, 64, 4},
          {64, 128, 64, 4},
      }};

      std::vector<std::array<int64_t, 5>> added;
      auto add_default_tile = [&](int64_t block_m, int64_t block_n,
                                  int64_t block_k, int64_t num_warps,
                                  FlyGemmConfig::MfmaAtom atom) {
        if ((block_m > m && block_m / 2 >= m && block_m != 16) ||
            (block_n > n && block_n / 2 >= n && block_n != 16) ||
            !supports_gemm_k_tile(block_k) ||
            2 * block_m * block_k * sizeof(uint16_t) > 64 * 1024) {
          return;
        }
        const bool atom_32 = atom == FlyGemmConfig::FLY_MFMA_32X32X8;
        const int64_t atom_size = atom_32 ? 32 : 16;
        if (block_m % atom_size != 0 || block_n % atom_size != 0) {
          return;
        }
        const int64_t wave_tiles =
            (block_m / atom_size) * (block_n / atom_size);
        if (num_warps > wave_tiles || wave_tiles % num_warps != 0) {
          return;
        }
        std::array<int64_t, 5> key = {
            block_m, block_n, block_k, num_warps,
            static_cast<int64_t>(atom)};
        if (absl::c_linear_search(added, key)) {
          return;
        }
        added.push_back(key);
        add_gemm_configs(block_m, block_n, block_k, num_warps, atom);
      };

      for (const auto& tile : kDefaultTiles) {
        add_default_tile(tile[0], tile[1], tile[2], tile[3],
                         FlyGemmConfig::FLY_MFMA_16X16X16);
        if (tile[0] % 32 == 0 && tile[1] % 32 == 0) {
          add_default_tile(tile[0], tile[1], tile[2], tile[3],
                           FlyGemmConfig::FLY_MFMA_32X32X8);
        }
      }
      return;
    }

    for (int64_t block_m : kBlockSizes) {
      // The generic emitter predicates its final M/N tile. Include the first
      // power-of-two tile larger than an odd dimension so small tails do not
      // force an extra workgroup, while avoiding successively larger tiles
      // whose wasted MFMA work cannot be competitive.
      if (block_m > m && block_m / 2 >= m && block_m != 16) {
        continue;
      }
      for (int64_t block_n : kBlockSizes) {
        if (block_n > n && block_n / 2 >= n) {
          continue;
        }
        const int64_t wave_tiles = (block_m / 16) * (block_n / 16);
        for (int64_t block_k : {16, 32, 64, 128, 256}) {
          constexpr int64_t kMaxLdsBytes = 64 * 1024;
          const int64_t lhs_lds_bytes =
              2 * block_m * block_k * sizeof(uint16_t);
          if (!supports_gemm_k_tile(block_k) || lhs_lds_bytes > kMaxLdsBytes) {
            continue;
          }
          for (int64_t num_warps : {1, 2, 4, 8}) {
            if (num_warps <= wave_tiles && wave_tiles % num_warps == 0) {
              add_gemm_configs(block_m, block_n, block_k, num_warps,
                               FlyGemmConfig::FLY_MFMA_16X16X16);
            }
            const int64_t mfma32_tiles = (block_m / 32) * (block_n / 32);
            if (block_m % 32 == 0 && block_n % 32 == 0 &&
                num_warps <= mfma32_tiles && mfma32_tiles % num_warps == 0) {
              add_gemm_configs(block_m, block_n, block_k, num_warps,
                               FlyGemmConfig::FLY_MFMA_32X32X8);
            }
          }
        }
      }
    }
  };

  auto add_staged_rhs_configs = [&](int64_t block_m, int64_t block_n,
                                    int64_t block_k,
                                    std::vector<int64_t> num_warps_values) {
    constexpr int64_t kMaxLdsBytes = 64 * 1024;
    const int64_t staged_lds_bytes =
        2 * (block_m + block_n) * block_k * sizeof(uint16_t);
    if ((!rhs_k_contiguous && !global_split_k) || block_m > m || block_n > n ||
        !supports_gemm_k_tile(block_k) || staged_lds_bytes > kMaxLdsBytes) {
      return;
    }
    auto add_config = [&](int64_t num_warps, bool stage_output,
                          int32_t waves_per_eu, bool schedule,
                          int32_t workgroup_mapping_n) {
      if (stage_output && output_element_type != BF16) {
        return;
      }
      std::unique_ptr<BackendConfig> config = MakeConfig(
          block_m, block_n, block_k, num_warps,
          FlyGemmConfig::FLY_MFMA_16X16X16,
          /*prefetch_rhs=*/false, stage_output, waves_per_eu,
          /*schedule_instructions=*/schedule,
          /*stage_rhs=*/true);
      config->mutable_fly()->set_workgroup_mapping_n(workgroup_mapping_n);
      configs.push_back(std::move(config));
    };

    if (!debug_options().xla_gpu_exhaustive_tiling_search()) {
      // Keep the staged-RHS schedules which won the audited MI300X workloads
      // without compiling their full Cartesian product. Occupancy caps only
      // paid off with the matching instruction schedule, while CShuffle's
      // four-wave cap won without it. Exhaustive mode below remains available
      // for discovering new combinations.
      struct PipelineVariant {
        bool stage_output;
        int32_t waves_per_eu;
        bool schedule;
      };
      constexpr std::array<PipelineVariant, 8> kDefaultPipelineVariants = {{
          {/*stage_output=*/false, /*waves_per_eu=*/0, /*schedule=*/false},
          {/*stage_output=*/false, /*waves_per_eu=*/0, /*schedule=*/true},
          {/*stage_output=*/false, /*waves_per_eu=*/2, /*schedule=*/true},
          {/*stage_output=*/false, /*waves_per_eu=*/4, /*schedule=*/false},
          {/*stage_output=*/true, /*waves_per_eu=*/0, /*schedule=*/false},
          {/*stage_output=*/true, /*waves_per_eu=*/0, /*schedule=*/true},
          {/*stage_output=*/true, /*waves_per_eu=*/2, /*schedule=*/true},
          {/*stage_output=*/true, /*waves_per_eu=*/4, /*schedule=*/false},
      }};
      const bool tune_occupancy =
          k < 1024 && block_m <= 32 && block_n <= 32;
      const bool tune_workgroup_mapping =
          k < 1024 && block_m == 32 && block_n == 16 && block_k == 128;
      for (int64_t num_warps : num_warps_values) {
        for (const PipelineVariant& variant : kDefaultPipelineVariants) {
          if (variant.waves_per_eu != 0 && !tune_occupancy) {
            continue;
          }
          add_config(num_warps, variant.stage_output, variant.waves_per_eu,
                     variant.schedule, /*workgroup_mapping_n=*/0);
        }
        if (tune_workgroup_mapping && num_warps == 2 &&
            output_element_type == BF16) {
          // Mapping 4/8/16 covers the locality search while retaining the
          // transformer winner at mapping 16. Mapping zero was added above.
          for (int32_t mapping : {4, 8, 16}) {
            add_config(num_warps, /*stage_output=*/true,
                       /*waves_per_eu=*/0, /*schedule=*/true, mapping);
          }
          add_config(num_warps, /*stage_output=*/true,
                     /*waves_per_eu=*/4, /*schedule=*/false,
                     /*workgroup_mapping_n=*/16);
        }
      }
      return;
    }
    for (int64_t num_warps : num_warps_values) {
      for (bool stage_output : {false, true}) {
        if (stage_output && output_element_type != BF16) {
          continue;
        }
        const std::vector<int32_t> waves_per_eu_values =
            k < 1024 && block_m <= 32 && block_n <= 32
                ? std::vector<int32_t>{0, 2, 4}
                : std::vector<int32_t>{0};
        const std::vector<int32_t> workgroup_mapping_values =
            k < 1024 && block_m == 32 && block_n == 16 && block_k == 128 &&
                    num_warps == 2
                ? std::vector<int32_t>{0, 4, 8, 16}
                : std::vector<int32_t>{0};
        for (int32_t workgroup_mapping_n : workgroup_mapping_values) {
          for (int32_t waves_per_eu : waves_per_eu_values) {
            for (bool schedule : {false, true}) {
              add_config(num_warps, stage_output, waves_per_eu, schedule,
                         workgroup_mapping_n);
            }
          }
        }
      }
    }
  };

  auto add_preloaded_rhs_configs = [&](int64_t block_m, int64_t block_n,
                                       int64_t block_k,
                                       std::vector<int64_t> num_warps_values) {
    constexpr int64_t kMaxLdsBytes = 64 * 1024;
    const int64_t staged_lds_bytes =
        2 * (block_m + block_n) * block_k * sizeof(uint16_t);
    if ((!rhs_k_contiguous && !global_split_k) || block_m > m || block_n > n ||
        !supports_gemm_k_tile(block_k) || staged_lds_bytes > kMaxLdsBytes) {
      return;
    }
    const bool can_stage_output =
        output_element_type == BF16 &&
        block_m * block_n * sizeof(uint16_t) <= kMaxLdsBytes;
    for (int64_t num_warps : num_warps_values) {
      const int64_t wave_tiles = (block_m / 16) * (block_n / 16);
      const int64_t block_threads = num_warps * 64;
      if (num_warps > wave_tiles || wave_tiles % num_warps != 0 ||
          (block_m * block_k / 2) % block_threads != 0 ||
          (block_n * block_k / 2) % block_threads != 0) {
        continue;
      }
      for (bool stage_output : {false, true}) {
        if (stage_output && !can_stage_output) {
          continue;
        }
        configs.push_back(MakeConfig(block_m, block_n, block_k, num_warps,
                                     FlyGemmConfig::FLY_MFMA_16X16X16,
                                     /*prefetch_rhs=*/false, stage_output,
                                     /*waves_per_eu=*/0,
                                     /*schedule_instructions=*/false,
                                     /*stage_rhs=*/true, /*async_lhs=*/false,
                                     /*preload_lds_fragments=*/true));
      }
    }
  };

  if (is_f32) {
    // CDNA3 provides native full-precision FP32 MFMA atoms. Keep the broad
    // LHS-LDS/RHS-register schedule for layout coverage, and offer the same
    // ping-pong A+B LDS pipeline used by the half-precision kernels whenever
    // the RHS contracting dimension is physically contiguous. Four FP32
    // elements fill one 16-byte VMEM transaction.
    constexpr std::array<int64_t, 4> kF32BlockSizes = {16, 32, 64, 128};
    constexpr std::array<int64_t, 5> kF32KBlockSizes = {4, 8, 16, 32, 64};
    const PrecisionConfig& precision = dot->precision_config();
    const bool require_xf32 =
        precision.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32;
    const bool default_operand_precision = absl::c_all_of(
        precision.operand_precision(),
        [](int value) { return value == PrecisionConfig::DEFAULT; });
    const bool allow_xf32 =
        default_operand_precision &&
        (precision.algorithm() == PrecisionConfig::ALG_UNSET ||
         precision.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32);
    for (int64_t block_m : kF32BlockSizes) {
      if (block_m > m && block_m / 2 >= m && block_m != 16) {
        continue;
      }
      for (int64_t block_n : kF32BlockSizes) {
        if (block_n > n && block_n / 2 >= n && block_n != 16) {
          continue;
        }
        const int64_t mfma16_tiles = (block_m / 16) * (block_n / 16);
        const int64_t mfma32_tiles = (block_m / 32) * (block_n / 32);
        for (int64_t block_k : kF32KBlockSizes) {
          // The generic kernel ping-pongs two block_m x block_k FP32 LHS
          // stages and moves four FP32 values per 16-byte VMEM operation.
          const int64_t lhs_lds_bytes = 2 * block_m * block_k * sizeof(float);
          if ((block_k > k && block_k / 2 >= k && block_k != 16) ||
              lhs_lds_bytes > 64 * 1024) {
            continue;
          }
          for (int64_t num_warps : {1, 2, 4, 8}) {
            const int64_t block_threads = num_warps * 64;
            if ((block_m * block_k / 4) % block_threads != 0) {
              continue;
            }
            if (num_warps <= mfma16_tiles && mfma16_tiles % num_warps == 0) {
              if (!require_xf32) {
                configs.push_back(
                    MakeConfig(block_m, block_n, block_k, num_warps,
                               FlyGemmConfig::FLY_MFMA_16X16X4_F32));
                const int64_t rhs_lds_bytes =
                    2 * block_n * block_k * sizeof(float);
                if (rhs_k_contiguous && block_k >= 32 && block_k % 32 == 0 &&
                    (block_n * block_k / 4) % block_threads == 0 &&
                    lhs_lds_bytes + rhs_lds_bytes <= 64 * 1024) {
                  for (int32_t waves_per_eu : {0, 2, 4}) {
                    for (bool schedule : {false, true}) {
                      configs.push_back(MakeConfig(
                          block_m, block_n, block_k, num_warps,
                          FlyGemmConfig::FLY_MFMA_16X16X4_F32,
                          /*prefetch_rhs=*/false, /*stage_output=*/false,
                          waves_per_eu, schedule, /*stage_rhs=*/true));
                    }
                  }
                }
              }
            }
            if (block_m % 32 == 0 && block_n % 32 == 0 &&
                num_warps <= mfma32_tiles && mfma32_tiles % num_warps == 0) {
              if (!require_xf32) {
                configs.push_back(
                    MakeConfig(block_m, block_n, block_k, num_warps,
                               FlyGemmConfig::FLY_MFMA_32X32X2_F32));
              }
              if (!allow_xf32) {
                continue;
              }
              configs.push_back(
                  MakeConfig(block_m, block_n, block_k, num_warps,
                             FlyGemmConfig::FLY_MFMA_32X32X4_XF32));
              const int64_t rhs_lds_bytes =
                  2 * block_n * block_k * sizeof(float);
              if (rhs_k_contiguous && block_k >= 32 && block_k % 32 == 0 &&
                  (block_n * block_k / 4) % block_threads == 0 &&
                  lhs_lds_bytes + rhs_lds_bytes <= 64 * 1024) {
                for (int32_t waves_per_eu : {0, 2, 4}) {
                  for (bool schedule : {false, true}) {
                    configs.push_back(MakeConfig(
                        block_m, block_n, block_k, num_warps,
                        FlyGemmConfig::FLY_MFMA_32X32X4_XF32,
                        /*prefetch_rhs=*/false, /*stage_output=*/false,
                        waves_per_eu, schedule, /*stage_rhs=*/true));
                  }
                }
              }
            }
          }
        }
      }
    }

    if (rhs_k_contiguous && k >= 64 && k % 32 == 0) {
      auto add_single_buffer_f32 = [&](int64_t block_m, int64_t block_n,
                                       int64_t block_k, int64_t num_warps,
                                       FlyGemmConfig::MfmaAtom atom) {
        if (block_m > m || block_n > n || m % block_m != 0 ||
            n % block_n != 0 || k % block_k != 0) {
          return;
        }
        for (int32_t waves_per_eu : {0, 2, 4}) {
          configs.push_back(MakeConfig(
              block_m, block_n, block_k, num_warps, atom,
              /*prefetch_rhs=*/false, /*stage_output=*/false, waves_per_eu,
              /*schedule_instructions=*/false, /*stage_rhs=*/true,
              /*async_lhs=*/false, /*preload_lds_fragments=*/true,
              /*single_buffer_lds=*/true));
          if (atom == FlyGemmConfig::FLY_MFMA_32X32X4_XF32) {
            configs.push_back(MakeConfig(
                block_m, block_n, block_k, num_warps, atom,
                /*prefetch_rhs=*/false, /*stage_output=*/false, waves_per_eu,
                /*schedule_instructions=*/true, /*stage_rhs=*/true,
                /*async_lhs=*/false, /*preload_lds_fragments=*/true,
                /*single_buffer_lds=*/true));
          }
        }
      };

      // Mirror Triton's two high-throughput FP32 modulo-pipeline tiles. One
      // physical A+B LDS stage uses 48 KiB for 256x128xK32 and the full 64
      // KiB for 128x128xK64, while the next tile is held in VGPRs under the
      // current tile's MFMAs. ROCm Triton maps TF32x3 to the native full-F32
      // atom and selects the same 256x128xK32 geometry, so make the atom part
      // of the measured Fly search rather than limiting this pipeline to
      // XF32.
      if (!require_xf32) {
        add_single_buffer_f32(/*block_m=*/256, /*block_n=*/128,
                              /*block_k=*/32, /*num_warps=*/8,
                              FlyGemmConfig::FLY_MFMA_32X32X2_F32);
        add_single_buffer_f32(/*block_m=*/128, /*block_n=*/256,
                              /*block_k=*/32, /*num_warps=*/8,
                              FlyGemmConfig::FLY_MFMA_32X32X2_F32);
        add_single_buffer_f32(/*block_m=*/128, /*block_n=*/128,
                              /*block_k=*/32, /*num_warps=*/4,
                              FlyGemmConfig::FLY_MFMA_32X32X2_F32);
        add_single_buffer_f32(/*block_m=*/128, /*block_n=*/128,
                              /*block_k=*/32, /*num_warps=*/8,
                              FlyGemmConfig::FLY_MFMA_32X32X2_F32);
        add_single_buffer_f32(/*block_m=*/128, /*block_n=*/128,
                              /*block_k=*/64, /*num_warps=*/4,
                              FlyGemmConfig::FLY_MFMA_32X32X2_F32);
      }
      if (allow_xf32) {
        add_single_buffer_f32(/*block_m=*/256, /*block_n=*/128,
                              /*block_k=*/32, /*num_warps=*/8,
                              FlyGemmConfig::FLY_MFMA_32X32X4_XF32);
        add_single_buffer_f32(/*block_m=*/128, /*block_n=*/128,
                              /*block_k=*/64, /*num_warps=*/4,
                              FlyGemmConfig::FLY_MFMA_32X32X4_XF32);
      }
    }
    return configs;
  }

  if (m >= 2 && m <= 8 && n > 1 && !is_f32 && !is_fp8 && !is_int4 &&
      !is_scaled_dot && !has_output_transpose && lhs_k_contiguous &&
      rhs_column_contiguous) {
    // Decoder projections are matrices mathematically, but padding M=2..8 to
    // the minimum 16-row MFMA atom performs up to 8x unnecessary work. Offer
    // a GEMV-family kernel that assigns output columns to lanes, computes only
    // the live rows, and optionally partitions a long K reduction across all
    // waves in the workgroup.
    for (int64_t block_n : {64, 128, 256}) {
      if (block_n > n && block_n / 2 >= n) {
        continue;
      }
      for (int64_t num_warps : {1, 2, 4}) {
        if (num_warps * 64 <= block_n) {
          configs.push_back(MakeGemvConfig(
              /*block_m=*/m, block_n, num_warps,
              /*outputs_per_wave=*/m, /*k_vector_width=*/0));
        }
      }
      if (k >= 512) {
        for (int64_t num_warps : {2, 4, 8}) {
          configs.push_back(MakeGemvConfig(
              /*block_m=*/m, block_n, num_warps,
              /*outputs_per_wave=*/m, /*k_vector_width=*/0,
              FlyGemmConfig::FLY_MFMA_16X16X16,
              /*split_k=*/true));
        }
      }
    }
    // Also let the MFMA path compete with explicit A+B LDS staging. It still
    // computes the padded 16-row atom, but unlike the generic direct-load
    // path it reuses every K stage across the output-column waves. These
    // geometries mirror Triton's successful small-M tiles on gfx942.
    for (const auto& [block_n, block_k, num_warps] :
         std::array<std::array<int64_t, 3>, 2>{{
             // Fused decoder projections can source the lhs through an f32
             // producer. Keep every row-stage copy divisible by the complete
             // workgroup even in that four-byte input case.
             {64, 64, 4},
             {128, 64, 4},
         }}) {
      if ((block_n > n && block_n / 2 >= n) || n % block_n != 0) {
        continue;
      }
      if (has_lhs_input_scale && block_n > 64) {
        continue;
      }
      if (has_lhs_input_scale && k % block_k != 0) {
        continue;
      }
      configs.insert(
          configs.begin(),
          MakeConfig(/*block_m=*/16, block_n, block_k, num_warps,
                     FlyGemmConfig::FLY_MFMA_16X16X16,
                     /*prefetch_rhs=*/false, /*stage_output=*/false,
                     /*waves_per_eu=*/0, /*schedule_instructions=*/false,
                     /*stage_rhs=*/true));
    }
    if (k >= 128 && k % 128 == 0) {
      // Increase K reuse without exceeding gfx942's LDS limit. A double
      // buffered M16/N64/K128 tile occupies 40 KiB; the corresponding N128
      // tile would require 72 KiB and needs a dedicated live-row refill
      // schedule before it can safely mirror Triton's wider tile.
      for (bool schedule : {false, true}) {
        configs.insert(
            configs.begin(),
            MakeConfig(/*block_m=*/16, /*block_n=*/64, /*block_k=*/128,
                       /*num_warps=*/4,
                       FlyGemmConfig::FLY_MFMA_16X16X16,
                       /*prefetch_rhs=*/false, /*stage_output=*/false,
                       /*waves_per_eu=*/0, schedule,
                       /*stage_rhs=*/true));
      }
      // FlyDSL's dedicated small-M B_TO_LDS kernel preloads the complete
      // current A/B fragment bank before issuing the next tile's vector loads.
      // Keep this as a separate candidate: it changes the machine schedule,
      // not just the macro-tile geometry.
      configs.insert(
          configs.begin(),
          MakeConfig(/*block_m=*/16, /*block_n=*/64, /*block_k=*/128,
                     /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
                     /*prefetch_rhs=*/false, /*stage_output=*/false,
                     /*waves_per_eu=*/0, /*schedule_instructions=*/true,
                     /*stage_rhs=*/true, /*async_lhs=*/false,
                     /*preload_lds_fragments=*/true));
    }
  }
  if (has_lhs_input_scale) {
    const bool homogeneous_bf16 =
        dot->operand(0)->shape().element_type() == BF16 &&
        dot->operand(1)->shape().element_type() == BF16;
    if (dot->shape().dimensions_size() == 2 && m >= 16 && n % 64 == 0 &&
        homogeneous_bf16 && rhs_column_contiguous &&
        output_element_type == BF16) {
      if (lhs_k_contiguous && !rhs_k_contiguous && m % 32 == 0 &&
          k >= 256 && k % 128 == 0) {
        // Transformer forward projections apply a learned RMSNorm scale to
        // an MxK activation before multiplying it by a row-major KxN weight.
        // Keep that multiply and its BF16 rounding boundary in the LHS load
        // feeding the same short-K MFMA32 pipeline used after Fly fission.
        configs.insert(
            configs.begin(),
            MakeConfig(
                /*block_m=*/32, /*block_n=*/64, /*block_k=*/128,
                /*num_warps=*/2, FlyGemmConfig::FLY_MFMA_32X32X8,
                /*prefetch_rhs=*/false, /*stage_output=*/false,
                /*waves_per_eu=*/0, /*schedule_instructions=*/false,
                /*stage_rhs=*/true, /*async_lhs=*/false,
                /*preload_lds_fragments=*/true,
                /*single_buffer_lds=*/true, /*direct_to_vgpr=*/false,
                /*rolling_refill=*/true));

        // The learned-scale projection is shallow enough that a compact tile
        // wins by exposing four times as many CTAs as the M32/N64
        // repository-style refill. M32/N32/K64 uses four MFMA16 waves and a
        // 16 KiB ping-pong A+B allocation, so four workgroups can reside on a
        // gfx942 CU. Keep instruction scheduling and occupancy measured: the
        // same tile wins both the N768 QKV and N1024 MLP projections, while
        // the larger K128/K256 alternatives only make autotuning noisier.
        for (int32_t waves_per_eu : {0, 2, 4}) {
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/32, /*block_n=*/32, /*block_k=*/64,
                /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
                /*prefetch_rhs=*/false, /*stage_output=*/false,
                waves_per_eu, schedule, /*stage_rhs=*/true));
          }
        }
      } else if (!lhs_k_contiguous && k % 32 == 0 && n % 128 == 0) {
        // The corresponding weight-gradient dot contracts the leading
        // dimension of the same scaled activation. The generic register-RHS
        // path already maps that physical transpose and applies the learned
        // per-output-channel scale while loading each LHS vector.
        configs.insert(
            configs.begin(),
            MakeConfig(/*block_m=*/16, /*block_n=*/128, /*block_k=*/32,
                       /*num_warps=*/4,
                       FlyGemmConfig::FLY_MFMA_16X16X16));
      }
    }
    if (dot->shape().dimensions_size() == 2 && m == 4 && n % 128 == 0 &&
        k % 128 == 0 && rhs_column_contiguous &&
        homogeneous_bf16) {
      // Keep the learned per-K scale inside the MFMA4 decoder projection.
      // Returning before this candidate forces Fly fission to materialize a
      // 4xK temporary, while Triton folds the same multiply/rounding producer
      // into its GEMM. The xTile GEMV emitter accepts this exact producer and
      // applies the scale as the live LHS tile is loaded.
      configs.insert(
          configs.begin(),
          MakeGemvConfig(/*block_m=*/4, /*block_n=*/128, /*num_warps=*/8,
                         /*outputs_per_wave=*/1, /*k_vector_width=*/1,
                         FlyGemmConfig::FLY_MFMA_4X4X4_BF16,
                         /*split_k=*/false, /*block_k=*/128));
    }
    // Do not expose configurations outside the learned-scale load paths
    // audited above. In particular, local-split and direct-to-VGPR schedules
    // intentionally bypass producer arithmetic while moving split buffers.
    return configs;
  }

  if (is_f16) {
    add_generic_gemm_configs();
    // The ordinary two-operand LDS paths use the same 16-bit fragment layout
    // for FP16 and BF16. Include them now; the fixed instruction-by-instruction
    // Tensile schedules below remain BF16-only until their FP16 ISA is audited.
    add_staged_rhs_configs(/*block_m=*/64, /*block_n=*/32, /*block_k=*/128,
                           /*num_warps_values=*/{2, 4});
    add_staged_rhs_configs(/*block_m=*/128, /*block_n=*/128, /*block_k=*/64,
                           /*num_warps_values=*/{4, 8, 16});
    add_preloaded_rhs_configs(/*block_m=*/64, /*block_n=*/64, /*block_k=*/64,
                              /*num_warps_values=*/{2, 4});
    add_preloaded_rhs_configs(/*block_m=*/128, /*block_n=*/128,
                              /*block_k=*/64,
                              /*num_warps_values=*/{4, 8, 16});
    // The asynchronous-LHS pipeline also operates on dword-sized direct LDS
    // copies and has the same fragment topology for F16 and BF16. Keep the
    // generic F16 epilogue while overlapping the next LHS tile with the
    // current MFMA batch.
    if (rhs_k_contiguous && m % 128 == 0 && n % 128 == 0 && k % 64 == 0 &&
        k >= 1024) {
      for (int64_t num_warps : {4, 8, 16}) {
        for (bool schedule : {false, true}) {
          configs.push_back(MakeConfig(
              /*block_m=*/128, /*block_n=*/128, /*block_k=*/64, num_warps,
              FlyGemmConfig::FLY_MFMA_16X16X16,
              /*prefetch_rhs=*/true, /*stage_output=*/false,
              /*waves_per_eu=*/0, /*schedule_instructions=*/schedule,
              /*stage_rhs=*/false, /*async_lhs=*/true));
        }
      }
    }
    // Reuse the gfx942 wide DirectToVgpr schedule that mirrors the best
    // BF16 hipBLASLt/Tensile solution. F16 has the same 16x16x16 MFMA
    // fragment topology and the same two-byte LDS footprint; only the packed
    // output conversion differs in the emitter.
    if (rhs_k_contiguous && m >= 256 && n >= 224 && k % 64 == 0) {
      // The ROCm 7.14 heuristic library exposes the same kernel body with
      // WGM=4, 6, and 8.  Their relative order depends on the matrix grid and
      // current XCC/cache state, so make workgroup mapping part of XLA's
      // measured search instead of baking in one Tensile heuristic result.
      for (int32_t workgroup_mapping_n : {4, 6, 8}) {
        for (bool single_buffer_lds : {true, false}) {
          std::unique_ptr<BackendConfig> config = MakeConfig(
              /*block_m=*/256, /*block_n=*/224, /*block_k=*/64,
              /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
              /*prefetch_rhs=*/false, /*stage_output=*/false,
              /*waves_per_eu=*/0, /*schedule_instructions=*/true,
              /*stage_rhs=*/true, /*async_lhs=*/false,
              /*preload_lds_fragments=*/true, single_buffer_lds,
              /*direct_to_vgpr=*/true);
          config->mutable_fly()->set_workgroup_mapping_n(workgroup_mapping_n);
          configs.push_back(std::move(config));
        }
      }
    }
    if (rhs_k_contiguous && m >= 224 && n >= 256 && k % 64 == 0) {
      configs.push_back(MakeConfig(
          /*block_m=*/224, /*block_n=*/256, /*block_k=*/64,
          /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
          /*prefetch_rhs=*/false, /*stage_output=*/false,
          /*waves_per_eu=*/0, /*schedule_instructions=*/true,
          /*stage_rhs=*/true, /*async_lhs=*/false,
          /*preload_lds_fragments=*/true,
          /*single_buffer_lds=*/false,
          /*direct_to_vgpr=*/true));
    }
    return configs;
  }

  if (global_split_k) {
    // A rank-three BF16 contraction with an FP32 output is structurally
    // indistinguishable from XLA split-K when the LHS stores its batch between
    // M and K. Decoder attention and MLP projections use exactly that layout
    // for small M. Keep the ordinary predicated MFMA tile in the search: the
    // emitter derives all batch/contracting dimensions from the dot and the
    // minimal M16 tile safely masks the unused rows.
    // A real batched dot with its LHS batch dimension between M and K is
    // structurally identical to an XLA global-split-K partial.  In
    // particular, JAX emits [M,B,K] x [B,K,N] for some einsums.  Do not leave
    // short-K instances without a configuration merely because the split-K
    // tuned pipelines below start at K512: the generic emitter already uses
    // the dot dimension numbers for all three operand indices and is valid
    // for both interpretations.
    if (m < 16 || k < 512) {
      add_generic_gemm_configs();
    }
    if (m == 4 && n % 64 == 0 && k >= 128 && rhs_column_contiguous && !is_f32 &&
        !is_fp8 && !is_int4 && !is_block_scaled_dot &&
        !has_lhs_input_scale &&
        dot->operand(0)->shape().element_type() == BF16 &&
        dot->operand(1)->shape().element_type() == BF16) {
      // Decoder MLP-down dots put their real batch dimension between M and K,
      // which is structurally identical to an XLA global split-K partial.
      // Triton's selected kernel still treats it as a batched M4/N64/K128
      // projection: both waves stage B and one live wave executes MFMA4.
      configs.insert(
          configs.begin(),
          MakeGemvConfig(/*block_m=*/4, /*block_n=*/64, /*num_warps=*/2,
                         /*outputs_per_wave=*/1, /*k_vector_width=*/1,
                         FlyGemmConfig::FLY_MFMA_4X4X4_BF16,
                         /*split_k=*/false, /*block_k=*/128));
    }
    // Global split-K partials are FP32 and XLA owns the final reduction. Only
    // offer the A+B LDS pipelines whose global accesses are explicitly
    // batch-aware; generic rank-2 tensor-indexing candidates remain excluded.
    if (k >= 512 && m >= 16) {
      add_staged_rhs_configs(/*block_m=*/64, /*block_n=*/32,
                             /*block_k=*/128,
                             /*num_warps_values=*/{2, 4});
      add_staged_rhs_configs(/*block_m=*/128, /*block_n=*/128,
                             /*block_k=*/64,
                             /*num_warps_values=*/{4, 8, 16});
      // Match hipBLASLt's winning MI300X geometry for M=128 inference GEMMs.
      // N=96 cuts the grid from 344 to 230 workgroups at split-K=2. The final
      // partial tile is predicated because common model widths (for example
      // 11008) are only guaranteed to be multiples of 16.
      if (rhs_k_contiguous && m % 128 == 0 && n >= 96 && n % 16 == 0 &&
          k % 64 == 0) {
        for (int64_t num_warps : {4, 8}) {
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/128, /*block_n=*/96, /*block_k=*/64, num_warps,
                FlyGemmConfig::FLY_MFMA_16X16X16,
                /*prefetch_rhs=*/false, /*stage_output=*/false,
                /*waves_per_eu=*/0, schedule, /*stage_rhs=*/true));
          }
        }
      }
      // hipBLASLt's selected gfx942 kernel for this family is effectively a
      // 128x96x128 four-wave tile backed by one ~57 KiB A+B allocation. Match
      // both its grid and its synchronization granularity; bounded buffer
      // loads zero-fill the final partial N tile.
      if (rhs_k_contiguous && m % 128 == 0 && n >= 96 && n % 16 == 0 &&
          k % 128 == 0) {
        for (bool rolling_refill : {false, true}) {
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/128, /*block_n=*/96, /*block_k=*/128,
                /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
                /*prefetch_rhs=*/false, /*stage_output=*/false,
                /*waves_per_eu=*/0, schedule, /*stage_rhs=*/true,
                /*async_lhs=*/false, /*preload_lds_fragments=*/true,
                /*single_buffer_lds=*/true, /*direct_to_vgpr=*/false,
                rolling_refill));
          }
        }
      }
      for (int64_t num_warps : {4, 8, 16}) {
        for (int32_t waves_per_eu : {2, 4}) {
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/128, /*block_n=*/128, /*block_k=*/64, num_warps,
                FlyGemmConfig::FLY_MFMA_16X16X16,
                /*prefetch_rhs=*/false, /*stage_output=*/false, waves_per_eu,
                schedule, /*stage_rhs=*/true));
          }
        }
      }
      // With M=128 and two split-K batches, these asymmetric tiles expose 344
      // workgroups on common inference GEMMs instead of the 172 produced by
      // the native 128x128 FlyDSL tile. Keep both orientations because the
      // LHS/RHS reuse tradeoff depends on N and the memory layout.
      add_staged_rhs_configs(/*block_m=*/64, /*block_n=*/128,
                             /*block_k=*/64,
                             /*num_warps_values=*/{4, 8});
      add_staged_rhs_configs(/*block_m=*/128, /*block_n=*/64,
                             /*block_k=*/64,
                             /*num_warps_values=*/{4, 8});
      // Preserve the 172-workgroup grid while doubling RHS reuse. K32 keeps
      // the two-stage A+B allocation at 40 KiB.
      add_staged_rhs_configs(/*block_m=*/64, /*block_n=*/256,
                             /*block_k=*/32,
                             /*num_warps_values=*/{4, 8});
      add_preloaded_rhs_configs(/*block_m=*/128, /*block_n=*/128,
                                /*block_k=*/64,
                                /*num_warps_values=*/{4, 8, 16});
      // This is FlyDSL hgemm_splitk.py's default gfx942 algorithm:
      // B_TO_LDS=false, asynchronous A global-to-LDS copies, and the next B
      // tile retained in VGPRs. The rank-3 address helpers account for XLA's
      // split batch, so this is the preferred faithful split-K pipeline.
      if (rhs_k_contiguous && m % 128 == 0 && n % 128 == 0 && k % 64 == 0) {
        for (int64_t num_warps : {4, 8, 16}) {
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/128, /*block_n=*/128, /*block_k=*/64, num_warps,
                FlyGemmConfig::FLY_MFMA_16X16X16,
                /*prefetch_rhs=*/true, /*stage_output=*/false,
                /*waves_per_eu=*/0, /*schedule_instructions=*/schedule,
                /*stage_rhs=*/false, /*async_lhs=*/true));
          }
        }
      }
      add_preloaded_rhs_configs(/*block_m=*/64, /*block_n=*/64,
                                /*block_k=*/64,
                                /*num_warps_values=*/{2, 4});
      // XLA's current split-K rewriter produces a row-major [S,K,N] RHS.
      // Use the dedicated row-major MFMA32 single-buffer schedule for that
      // layout: its K128 tile halves the MFMA issue count relative to the
      // generic MFMA16 staged candidates while retaining eight-wave
      // occupancy for the 128x64 output tile.
      if (!rhs_k_contiguous && m % 128 == 0 && n % 64 == 0 && k % 128 == 0) {
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/64, /*block_k=*/128,
            /*num_warps=*/8, FlyGemmConfig::FLY_MFMA_32X32X8,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true));
      }
      if (rhs_k_contiguous && m % 128 == 0 && n % 64 == 0 && k % 128 == 0) {
        for (bool rolling_refill : {false, true}) {
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/128, /*block_n=*/64, /*block_k=*/128,
                /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
                /*prefetch_rhs=*/false, /*stage_output=*/false,
                /*waves_per_eu=*/0, /*schedule_instructions=*/schedule,
                /*stage_rhs=*/true, /*async_lhs=*/false,
                /*preload_lds_fragments=*/true,
                /*single_buffer_lds=*/true,
                /*direct_to_vgpr=*/false, rolling_refill));
          }
        }
        // Compose XLA's grid-level split with Fly's two-way wave-local split.
        // The workgroup reduces both local K partitions into one FP32 partial;
        // XLA's existing reduction then combines the global split batches.
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/64, /*block_k=*/128,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true,
            /*direct_to_vgpr=*/false,
            /*rolling_refill=*/false,
            /*local_split_k=*/true));
      }
      if (rhs_k_contiguous && m % 128 == 0 && n % 256 == 0 && k % 32 == 0) {
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/256, /*block_k=*/32,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_32X32X8,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true));
      }
      if (rhs_k_contiguous && m % 128 == 0 && n % 128 == 0 && k % 64 == 0) {
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/128, /*block_k=*/64,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_32X32X8,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true));
      }
    }
    return configs;
  }

  // Wide-N tiles increase RHS reuse without increasing the LHS LDS footprint.
  // Keep these candidates explicit: admitting 512 into the Cartesian search
  // also creates 256x512 and 512x512 kernels whose register allocation is
  // prohibitively expensive during online autotuning.
  if (m % 64 == 0 && n % 512 == 0 && k % 32 == 0 && k >= 1024) {
    configs.push_back(MakeConfig(
        /*block_m=*/64, /*block_n=*/512, /*block_k=*/32, /*num_warps=*/4,
        FlyGemmConfig::FLY_MFMA_16X16X16));
  }
  if (rank_three && !rhs_k_contiguous && m >= 64 && n >= 256 && k >= 1024 &&
      supports_gemm_k_tile(/*block_k=*/32)) {
    for (int32_t workgroup_mapping_n : {1, 2, 4, 8}) {
      for (bool schedule : {false, true}) {
        std::unique_ptr<BackendConfig> config = MakeConfig(
            /*block_m=*/64, /*block_n=*/256, /*block_k=*/32,
            /*num_warps=*/8, FlyGemmConfig::FLY_MFMA_16X16X16,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, schedule);
        config->mutable_fly()->set_workgroup_mapping_n(workgroup_mapping_n);
        configs.push_back(std::move(config));
      }
    }
  }
  if (rank_three && output_element_type == BF16 && !rhs_k_contiguous &&
      m >= 128 && n >= 128 && k >= 1024 &&
      supports_gemm_k_tile(/*block_k=*/32)) {
    // Match Triton's winning ragged-batch geometry while retaining explicit
    // control of the row-major B transpose and the one-stage LDS refill. K32
    // halves the A+B footprint relative to the established K64 square path;
    // autotune both direct and CShuffle stores because the latter reuses the
    // same allocation without growing beyond 32 KiB.
    for (bool stage_output : {false, true}) {
      for (bool schedule : {false, true}) {
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/128, /*block_k=*/32,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_32X32X8,
            /*prefetch_rhs=*/false, stage_output,
            /*waves_per_eu=*/0, schedule,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/false));
      }
    }
  }
  if (m % 128 == 0 && n % 512 == 0 && k % 32 == 0 && k >= 1024) {
    configs.push_back(MakeConfig(
        /*block_m=*/128, /*block_n=*/512, /*block_k=*/32, /*num_warps=*/8,
        FlyGemmConfig::FLY_MFMA_16X16X16,
        /*prefetch_rhs=*/true, /*stage_output=*/false,
        /*waves_per_eu=*/0, /*schedule_instructions=*/true));
  }
  // The native 64x64x64 FlyDSL A+B pipeline is also valid for short
  // contractions. It avoids the generic path's repeated global RHS loads and
  // is especially useful when launch-scale 256/512 GEMMs need more work per
  // workgroup. The larger-K search below adds the same family with its other
  // long-pipeline candidates.
  if (k >= 128 && k < 1024) {
    add_preloaded_rhs_configs(/*block_m=*/64, /*block_n=*/64,
                              /*block_k=*/64,
                              /*num_warps_values=*/{2, 4});
    // Short square GEMMs need enough workgroups to fill MI300X. Explore the
    // small geometries selected by Triton while retaining Fly's explicit A+B
    // LDS staging. K128 permits two resident workgroups; K256 trades that
    // occupancy for fewer barriers.
    if (k >= 256) {
      add_staged_rhs_configs(/*block_m=*/16, /*block_n=*/16,
                             /*block_k=*/128,
                             /*num_warps_values=*/{1});
      add_staged_rhs_configs(/*block_m=*/32, /*block_n=*/16,
                             /*block_k=*/128,
                             /*num_warps_values=*/{1, 2});
      add_staged_rhs_configs(/*block_m=*/32, /*block_n=*/32,
                             /*block_k=*/128,
                             /*num_warps_values=*/{4});
      add_staged_rhs_configs(/*block_m=*/32, /*block_n=*/16,
                             /*block_k=*/256,
                             /*num_warps_values=*/{1, 2});
    }
  }
  if (rank_three && !lhs_k_contiguous && !rhs_k_contiguous && k >= 32 &&
      k <= 128 && m % 64 == 0 && n % 64 == 0) {
    // Batched attention scores are Q^T K with both source tensors physically
    // stored as [B,K,S].  Stage contiguous S vectors from both operands and
    // transpose them into the native FlyDSL [S,K] LDS fragments.  Include the
    // one-workgroup-per-head geometry chosen by Triton and a 64x64 alternative
    // that exposes more of MI300X's 304 CUs.
    for (int64_t block_m : {64, 128}) {
      for (int64_t block_n : {64, 128}) {
        if (block_m > m || block_n > n) {
          continue;
        }
        for (int64_t block_k : {32, 64}) {
          if (!supports_gemm_k_tile(block_k)) {
            continue;
          }
          const int64_t lds_bytes =
              2 * (block_m + block_n) * block_k * sizeof(uint16_t);
          if (lds_bytes > 64 * 1024) {
            continue;
          }
          const int64_t wave_tiles = (block_m / 16) * (block_n / 16);
          for (int64_t num_warps : {4, 8}) {
            if (num_warps > wave_tiles || wave_tiles % num_warps != 0) {
              continue;
            }
            for (int32_t waves_per_eu : {0, 2, 4}) {
              for (bool schedule : {false, true}) {
                configs.push_back(
                    MakeConfig(block_m, block_n, block_k, num_warps,
                               FlyGemmConfig::FLY_MFMA_16X16X16,
                               /*prefetch_rhs=*/false, /*stage_output=*/false,
                               waves_per_eu, schedule, /*stage_rhs=*/true));
              }
            }
          }
        }
      }
    }
  }
  // Row-major RHS needs a K/N register transpose before LDS. For shallow M,
  // keep one MFMA32 output atom per wave: the 32x64 tile exposes four times as
  // many workgroups as the repository's 128x64 geometry while preserving the
  // same four coalesced A vectors per thread. This is important for projection
  // GEMMs whose concatenated N dimension is large but M cannot fill gfx942.
  if (k >= 256 && !rhs_k_contiguous && m % 32 == 0 && n % 64 == 0 &&
      k % 128 == 0) {
    configs.push_back(MakeConfig(
        /*block_m=*/32, /*block_n=*/64, /*block_k=*/128,
        /*num_warps=*/2, FlyGemmConfig::FLY_MFMA_32X32X8,
        /*prefetch_rhs=*/false, /*stage_output=*/false,
        /*waves_per_eu=*/0, /*schedule_instructions=*/false,
        /*stage_rhs=*/true, /*async_lhs=*/false,
        /*preload_lds_fragments=*/true,
        /*single_buffer_lds=*/true, /*direct_to_vgpr=*/false,
        /*rolling_refill=*/true));
  }
  // Pair adjacent K rows, then overlap the single-buffer refill with the
  // second half of the MFMA tile. Four K128 stages are already sufficient to
  // amortize this pipeline in batched K512 GEMMs.
  if (k >= 256 && !rhs_k_contiguous && m % 128 == 0 && n % 64 == 0 &&
      k % 128 == 0) {
    configs.push_back(MakeConfig(
        /*block_m=*/128, /*block_n=*/64, /*block_k=*/128,
        /*num_warps=*/8, FlyGemmConfig::FLY_MFMA_32X32X8,
        /*prefetch_rhs=*/false, /*stage_output=*/false,
        /*waves_per_eu=*/0, /*schedule_instructions=*/false,
        /*stage_rhs=*/true, /*async_lhs=*/false,
        /*preload_lds_fragments=*/true,
        /*single_buffer_lds=*/true, /*direct_to_vgpr=*/false,
        /*rolling_refill=*/true));
  }
  // Apply the remaining gfx942 A+B LDS pipelines selected for long row-major
  // RHS inputs. Both geometries fit within the MI300X 64 KiB LDS allocation.
  if (k >= 1024) {
    // Triton's winning shallow-output projection tile stages a K256 panel for
    // two waves that share the same N16 RHS fragment. The generic Fly kernel
    // streams that fragment independently in both waves; offer the equivalent
    // A+B LDS pipeline so autotuning can measure reuse against the extra
    // synchronization. Two stages occupy 48 KiB for BF16/F16.
    add_staged_rhs_configs(/*block_m=*/32, /*block_n=*/16, /*block_k=*/256,
                           /*num_warps_values=*/{2});
    if (rhs_k_contiguous && m % 32 == 0 && n % 16 == 0 && k % 256 == 0) {
      for (bool schedule : {false, true}) {
        configs.push_back(MakeConfig(
            /*block_m=*/32, /*block_n=*/16, /*block_k=*/256,
            /*num_warps=*/2, FlyGemmConfig::FLY_MFMA_16X16X16,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, schedule, /*stage_rhs=*/true,
            /*async_lhs=*/false, /*preload_lds_fragments=*/false,
            /*single_buffer_lds=*/true));
      }
    }
    if (output_element_type == BF16 && !rhs_k_contiguous && m % 128 == 0 &&
        n % 128 == 0 && k % 64 == 0) {
      // Keep one 32 KiB A/B tile in LDS while the next tile is prefetched in
      // VGPRs. Autotune direct stores against the CShuffle epilogue: the
      // latter restores coalescing from the native MFMA32 accumulator layout
      // without increasing the 32 KiB allocation, while the former avoids its
      // extra LDS round trip and workgroup barriers.
      for (bool schedule : {false, true}) {
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/128, /*block_k=*/64,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_32X32X8,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, schedule,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/false, /*direct_to_vgpr=*/false,
            /*rolling_refill=*/false));
      }
      configs.push_back(MakeConfig(
          /*block_m=*/128, /*block_n=*/128, /*block_k=*/64,
          /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_32X32X8,
          /*prefetch_rhs=*/false, /*stage_output=*/true,
          /*waves_per_eu=*/0, /*schedule_instructions=*/false,
          /*stage_rhs=*/true, /*async_lhs=*/false,
          /*preload_lds_fragments=*/true,
          /*single_buffer_lds=*/false, /*direct_to_vgpr=*/false,
          /*rolling_refill=*/false));
    }
    if (output_element_type == BF16 && !rhs_k_contiguous && m % 128 == 0 &&
        n % 256 == 0 && k % 32 == 0) {
      // Match Triton's winning 128x256x32 macro-tile while retaining Fly's
      // explicit one-stage A+B refill. The emitter source-swaps MFMA32 so each
      // lane writes four adjacent BF16 columns directly, avoiding both the
      // scalar native-layout epilogue and a 64 KiB CShuffle allocation.
      configs.push_back(MakeConfig(
          /*block_m=*/128, /*block_n=*/256, /*block_k=*/32,
          /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_32X32X8,
          /*prefetch_rhs=*/false, /*stage_output=*/false,
          /*waves_per_eu=*/0, /*schedule_instructions=*/true,
          /*stage_rhs=*/true, /*async_lhs=*/false,
          /*preload_lds_fragments=*/true,
          /*single_buffer_lds=*/false, /*direct_to_vgpr=*/false,
          /*rolling_refill=*/false));
    }
    add_staged_rhs_configs(/*block_m=*/64, /*block_n=*/32, /*block_k=*/128,
                           /*num_warps_values=*/{2, 4});
    add_staged_rhs_configs(/*block_m=*/128, /*block_n=*/128, /*block_k=*/64,
                           /*num_warps_values=*/{4, 8, 16});
    if (rhs_k_contiguous && m % 128 == 0 && n % 128 == 0 && k % 64 == 0) {
      for (int64_t num_warps : {4, 8, 16}) {
        for (bool stage_output : {false, true}) {
          if (stage_output && output_element_type != BF16) {
            continue;
          }
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/128, /*block_n=*/128, /*block_k=*/64, num_warps,
                FlyGemmConfig::FLY_MFMA_16X16X16,
                /*prefetch_rhs=*/true, stage_output,
                /*waves_per_eu=*/0, /*schedule_instructions=*/schedule,
                /*stage_rhs=*/false, /*async_lhs=*/true));
          }
        }
      }
    }
    // FlyDSL's newer splitk_hgemm preloads the current LDS fragments, issues
    // the next tile's direct copies, then consumes the preloaded fragments.
    // Keep the native square geometry. A 128x256x32 variant matching Triton's
    // winning tile was 26% slower than this kernel on 4096^3.
    add_preloaded_rhs_configs(/*block_m=*/128, /*block_n=*/128,
                              /*block_k=*/64,
                              /*num_warps_values=*/{4, 8, 16});
    // The native 128x128 tile underfills MI300X on moderately sized outputs.
    // A balanced 64x64 tile preserves the same full-fragment, two-stage
    // pipeline while exposing four times as many independent workgroups.
    add_preloaded_rhs_configs(/*block_m=*/64, /*block_n=*/64,
                              /*block_k=*/64,
                              /*num_warps_values=*/{2, 4});
    // hipBLASLt's small-grid BF16 solutions use a single 48 KiB A+B tile at
    // K128. This keeps four waves busy with more work per synchronization than
    // the double-buffered 64x64x64 kernel.
    if (rhs_k_contiguous && m % 128 == 0 && n % 64 == 0 && k % 128 == 0) {
      for (bool rolling_refill : {false, true}) {
        for (bool stage_output : {false, true}) {
          if (stage_output && output_element_type != BF16) {
            continue;
          }
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/128, /*block_n=*/64, /*block_k=*/128,
                /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
                /*prefetch_rhs=*/false, stage_output,
                /*waves_per_eu=*/0, /*schedule_instructions=*/schedule,
                /*stage_rhs=*/true, /*async_lhs=*/false,
                /*preload_lds_fragments=*/true,
                /*single_buffer_lds=*/true,
                /*direct_to_vgpr=*/false, rolling_refill));
          }
        }
      }
      if (output_element_type == BF16) {
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/64, /*block_k=*/128,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true,
            /*direct_to_vgpr=*/false,
            /*rolling_refill=*/false,
            /*local_split_k=*/true));
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/64, /*block_k=*/128,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/false,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true,
            /*direct_to_vgpr=*/false,
            /*rolling_refill=*/false,
            /*local_split_k=*/true));
      }
    }
    // A square MFMA32 tile can reuse its 32 KiB A+B allocation for FlyDSL's
    // 32 KiB CShuffle epilogue while retaining two workgroups per CU.
    if (rhs_k_contiguous && m % 128 == 0 && n % 128 == 0 && k % 64 == 0) {
      for (bool stage_output : {false, true}) {
        if (stage_output && output_element_type != BF16) {
          continue;
        }
        configs.push_back(MakeConfig(
            /*block_m=*/128, /*block_n=*/128, /*block_k=*/64,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_32X32X8,
            /*prefetch_rhs=*/false, stage_output,
            /*waves_per_eu=*/0,
            /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true));
      }
    }
    // Triton uses one 24 KiB A+B LDS tile for this geometry. Test the same
    // footprint with FlyDSL's explicit full-fragment preload and scheduler;
    // the next global tile is held briefly in VGPRs across the first barrier.
    if (rhs_k_contiguous && m % 128 == 0 && n % 256 == 0 && k % 32 == 0) {
      configs.push_back(MakeConfig(
          /*block_m=*/128, /*block_n=*/256, /*block_k=*/32,
          /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_32X32X8,
          /*prefetch_rhs=*/false, /*stage_output=*/false,
          /*waves_per_eu=*/0, /*schedule_instructions=*/true,
          /*stage_rhs=*/true, /*async_lhs=*/false,
          /*preload_lds_fragments=*/true,
          /*single_buffer_lds=*/true));
    }
    // Follow FlyDSL's native hGEMM search and Triton's winning geometry for
    // tall, wide inference GEMMs. Keeping one A+B K64 tile in LDS costs 48 KiB;
    // the next tile is held in VGPRs while the current fragments execute.
    // ShiftPtr handles non-divisible output edges, while grouped workgroup
    // orders expose cache-reuse choices to autotuning.
    if (rhs_k_contiguous && m >= 256 && n >= 128 && k % 64 == 0) {
      for (int32_t workgroup_mapping_n : {0, 1, 2, 4, 8}) {
        std::unique_ptr<BackendConfig> config = MakeConfig(
            /*block_m=*/256, /*block_n=*/128, /*block_k=*/64,
            /*num_warps=*/8, FlyGemmConfig::FLY_MFMA_32X32X8,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true);
        config->mutable_fly()->set_workgroup_mapping_n(workgroup_mapping_n);
        configs.push_back(std::move(config));
      }
    }
    // Match hipBLASLt's best 4096^3 BF16 solution: four 64x224 wave tiles and
    // a 64-wide K software pipeline.  The single-buffer implementation uses
    // unpredicated tensor transfers, so only offer it for complete N tiles.
    // The direct-to-VGPR double-buffer implementation uses bounded buffer
    // loads and supports the partial N tile needed by 4096x4096.
    if (rhs_k_contiguous && m >= 256 && n >= 224 && k % 64 == 0) {
      if (n % 224 == 0) {
        configs.push_back(MakeConfig(
            /*block_m=*/256, /*block_n=*/224, /*block_k=*/64,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true));
        configs.push_back(MakeConfig(
            /*block_m=*/256, /*block_n=*/224, /*block_k=*/64,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true,
            /*direct_to_vgpr=*/true));
      }
      configs.push_back(MakeConfig(
          /*block_m=*/256, /*block_n=*/224, /*block_k=*/64,
          /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
          /*prefetch_rhs=*/false, /*stage_output=*/false,
          /*waves_per_eu=*/0, /*schedule_instructions=*/true,
          /*stage_rhs=*/true, /*async_lhs=*/false,
          /*preload_lds_fragments=*/true,
          /*single_buffer_lds=*/false,
          /*direct_to_vgpr=*/true));
    }
    // The row-major counterpart of the Tensile tile. Swapping the MFMA
    // operands makes each wave compute a native 64x224 transposed view while
    // exposing four contiguous output columns per lane.
    if (rhs_k_contiguous && m >= 224 && n >= 256 && k % 64 == 0) {
      // The single-buffer path uses unpredicated transfers, whereas the
      // DirectToVgprB double-buffer path has bounded loads for the final
      // partial M tile.
      if (m % 224 == 0) {
        configs.push_back(MakeConfig(
            /*block_m=*/224, /*block_n=*/256, /*block_k=*/64,
            /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
            /*prefetch_rhs=*/false, /*stage_output=*/false,
            /*waves_per_eu=*/0, /*schedule_instructions=*/true,
            /*stage_rhs=*/true, /*async_lhs=*/false,
            /*preload_lds_fragments=*/true,
            /*single_buffer_lds=*/true,
            /*direct_to_vgpr=*/true));
      }
      configs.push_back(MakeConfig(
          /*block_m=*/224, /*block_n=*/256, /*block_k=*/64,
          /*num_warps=*/4, FlyGemmConfig::FLY_MFMA_16X16X16,
          /*prefetch_rhs=*/false, /*stage_output=*/false,
          /*waves_per_eu=*/0, /*schedule_instructions=*/true,
          /*stage_rhs=*/true, /*async_lhs=*/false,
          /*preload_lds_fragments=*/true,
          /*single_buffer_lds=*/false,
          /*direct_to_vgpr=*/true));
    }
  }

  add_generic_gemm_configs();
  if (dot->shape().dimensions_size() == 2 && m == 4 && n % 128 == 0 &&
      k % 128 == 0 && rhs_column_contiguous &&
      !is_f32 && !is_fp8 && !is_int4 && !is_block_scaled_dot &&
      dot->operand(0)->shape().element_type() == BF16 &&
      dot->operand(1)->shape().element_type() == BF16) {
    // Triton's successful decoder projection maps one Wave64 to a native
    // 4x64 output atom. Eight waves cooperatively transpose a K128xN128 RHS
    // tile into LDS; the two live N waves use CBSZ/ABID MFMA4 while the
    // padded-M waves provide the copy bandwidth and latency hiding.
    configs.insert(
        configs.begin(),
        MakeGemvConfig(/*block_m=*/4, /*block_n=*/128, /*num_warps=*/8,
                       /*outputs_per_wave=*/1, /*k_vector_width=*/1,
                       FlyGemmConfig::FLY_MFMA_4X4X4_BF16,
                       /*split_k=*/false, /*block_k=*/128));
  }
  if (has_output_transpose && rhs_k_contiguous && m % 64 == 0 && n % 64 == 0 &&
      k % 32 == 0) {
    // The context-layout epilogue stores four contiguous M values directly
    // from each MFMA lane. Match Triton's winning M64-oriented short-K tile,
    // retain its transposed M32xN64 counterpart, and expose occupancy and
    // instruction scheduling explicitly. Also stage RHS as a measured option:
    // the four-wave tile reuses each B fragment across multiple M atoms.
    for (auto [block_m, block_n] :
         {std::pair<int64_t, int64_t>{64, 32}, {32, 64}}) {
      for (int32_t waves_per_eu : {0, 2, 4}) {
        for (bool schedule : {false, true}) {
          configs.push_back(
              MakeConfig(block_m, block_n, /*block_k=*/32, /*num_warps=*/4,
                         FlyGemmConfig::FLY_MFMA_16X16X16,
                         /*prefetch_rhs=*/false, /*stage_output=*/true,
                         waves_per_eu, schedule));
          configs.push_back(
              MakeConfig(block_m, block_n, /*block_k=*/32, /*num_warps=*/4,
                         FlyGemmConfig::FLY_MFMA_16X16X16,
                         /*prefetch_rhs=*/false, /*stage_output=*/true,
                         waves_per_eu, schedule, /*stage_rhs=*/true));
        }
      }
    }
  }
  if (lhs_concat.has_value() || rhs_concat.has_value()) {
    configs.erase(
        std::remove_if(
            configs.begin(), configs.end(),
            [&](const std::unique_ptr<BackendConfig>& config) {
              const FlyGemmConfig& fly = config->fly();
              return (lhs_concat.has_value() &&
                      !absl::c_all_of(
                          lhs_concat->fragment_sizes,
                          [&](int64_t size) {
                            return size % fly.block_m() == 0;
                          })) ||
                     (rhs_concat.has_value() &&
                      !absl::c_all_of(
                          rhs_concat->fragment_sizes,
                          [&](int64_t size) {
                            return size % fly.block_n() == 0;
                          }));
            }),
        configs.end());
  }
  if (has_output_transpose) {
    configs.erase(
        std::remove_if(configs.begin(), configs.end(),
                       [](const std::unique_ptr<BackendConfig>& config) {
                         return !config->fly().stage_output();
                       }),
        configs.end());
  }
  if (m >= 2 && m <= 8) {
    // Generic configuration families were designed around complete MFMA
    // macro-tiles. Once the emitter admits a masked M tail, discard shapes
    // that cannot assign at least one output atom per compute wave. For an
    // LDS-staged tail, also retain only the dedicated M16 family and require
    // both cooperative copies to divide evenly across the workgroup.
    const int64_t copy_elements = is_f32 ? 4 : (is_fp8 ? 8 : 2);
    configs.erase(
        std::remove_if(
            configs.begin(), configs.end(),
            [&](const std::unique_ptr<BackendConfig>& config) {
              const FlyGemmConfig& fly = config->fly();
              if (fly.gemv_outputs_per_wave() != 0) {
                return false;
              }
              const bool atom_32 =
                  fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 ||
                  fly.mfma_atom() ==
                      FlyGemmConfig::FLY_MFMA_32X32X16_FP8 ||
                  fly.mfma_atom() ==
                      FlyGemmConfig::FLY_MFMA_32X32X2_F32 ||
                  fly.mfma_atom() ==
                      FlyGemmConfig::FLY_MFMA_32X32X4_XF32;
              const int64_t atom_m = atom_32 ? 32 : 16;
              const int64_t output_waves =
                  fly.num_warps() / (fly.local_split_k() ? 2 : 1);
              if ((fly.block_m() / atom_m) * (fly.block_n() / atom_m) <
                  output_waves) {
                return true;
              }
              if (!fly.stage_rhs()) {
                return false;
              }
              const int64_t threads = fly.num_warps() * 64;
              return fly.block_m() != 16 ||
                     (fly.block_m() * fly.block_k() / copy_elements) %
                             threads !=
                         0 ||
                     (fly.block_n() * fly.block_k() / copy_elements) %
                             threads !=
                         0;
            }),
        configs.end());
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> FlyBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                   GetSupportedConfigs(instr));
  if (configs.empty()) {
    return absl::InvalidArgumentError(
        "FlyBackend has no supported configuration for this instruction.");
  }
  return std::move(configs.back());
}

absl::Status FlyBackend::ApplyConfig(HloInstruction& instr,
                                     const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "FlyBackend does not support this instruction.");
  }
  if (!config.has_fly()) {
    return absl::InvalidArgumentError("Expected FlyGemmConfig for FlyBackend.");
  }
  const FlyGemmConfig& fly_config = config.fly();

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig* fusion_config =
      gpu_config.mutable_fusion_backend_config();
  HloInstruction* dot = FindContraction(instr);
  const int64_t output_rank = dot->shape().dimensions_size();
  const bool is_gemv = dot->shape().dimensions(output_rank - 2) == 1 ||
                       dot->shape().dimensions(output_rank - 1) == 1;
  const bool gemv_emitter_compatible = output_rank <= 3;
  const bool is_small_m_gemv =
      dot->shape().dimensions(output_rank - 2) >= 2 &&
      dot->shape().dimensions(output_rank - 2) <= 8 &&
      dot->shape().dimensions(output_rank - 1) > 1 &&
      fly_config.block_m() ==
          dot->shape().dimensions(output_rank - 2) &&
      fly_config.gemv_outputs_per_wave() != 0;
  // Scalar/vector GEMV configs use the dedicated Fly GEMV emitter.  The
  // cooperative MFMA candidate is structurally a tiled GEMM with a masked
  // M-tail and must therefore use the xTile GEMM emitter.
  fusion_config->set_kind(gemv_emitter_compatible &&
                                  (is_gemv || is_small_m_gemv) &&
                                  !fly_config.local_split_k()
                              ? kFlyGemvFusionKind
                              : kFlyGemmFusionKind);
  *fusion_config->mutable_fly_gemm_config() = fly_config;

  BlockLevelFusionConfig* block_config =
      fusion_config->mutable_block_level_fusion_config();
  block_config->Clear();
  Tile* output_tile = block_config->add_output_tiles();
  for (int64_t batch_dimension = 0; batch_dimension < output_rank - 2;
       ++batch_dimension) {
    output_tile->add_sizes(1);
  }
  output_tile->add_sizes(fly_config.block_m());
  output_tile->add_sizes(fly_config.block_n());
  block_config->set_num_warps(fly_config.num_warps());
  block_config->set_num_ctas(1);
  block_config->set_num_stages(1);
  block_config->set_waves_per_eu(fly_config.waves_per_eu());
  fusion_config->clear_triton_gemm_config();

  Tile contraction_tile;
  contraction_tile.add_sizes(fly_config.block_k());
  RETURN_IF_ERROR(dot->set_backend_config(contraction_tile));
  return instr.set_backend_config(gpu_config);
}

bool FlyFusionBackend::IsSupported(const HloInstruction& instr) {
  if (!debug_options().xla_gpu_enable_flydsl_fusion() ||
      !target_config().device_description.gpu_compute_capability().IsRocm() ||
      instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  const auto* fusion = Cast<const HloFusionInstruction>(&instr);
  // A parameter-free scalar fusion is compile-time constant materialization.
  // Leave it out of autotuning; FlyAutotuneCleanup evaluates it after backend
  // selection so it cannot become a standalone GPU launch.
  if (fusion->operand_count() == 0 && ShapeUtil::IsScalar(fusion->shape()) &&
      !debug_options().xla_gpu_flydsl_replace_triton()) {
    return false;
  }
  bool is_generic_block_fusion = false;
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    auto gpu_config = instr.backend_config<GpuBackendConfig>();
    if (!gpu_config.ok()) {
      return false;
    }
    absl::string_view kind = gpu_config->fusion_backend_config().kind();
    if (kind != kTritonFusionKind && kind != kFlyFusionKind &&
        !(debug_options().xla_gpu_flydsl_replace_triton() &&
          kind == kTritonNestedGemmFusionKind)) {
      return false;
    }
    is_generic_block_fusion = true;
  }
  if (!is_generic_block_fusion &&
      flydsl::ShouldKeepLargeIndexedDagOnNativeEmitter(instr)) {
    return false;
  }
  HloFusionAnalysis analysis =
      HloFusionAnalysis::Create(instr, target_config().device_description);
  if (flydsl::GetFlyScanDescriptor(analysis).has_value()) {
    return true;
  }
  if (flydsl::GetFlyPagedAttentionDescriptor(analysis).has_value() ||
      flydsl::GetFlyPagedAttentionSegmentedProducerDescriptor(analysis)
          .has_value() ||
      flydsl::GetFlyPagedAttentionSegmentedReducerDescriptor(analysis)
          .has_value()) {
    return true;
  }
  if (flydsl::GetFlyAttentionDescriptor(analysis).has_value()) {
    return true;
  }
  if (flydsl::IsFlySoftmaxFusion(analysis)) {
    return true;
  }
  if (flydsl::IsFlyLayerNormFusion(analysis)) {
    return true;
  }
  if (flydsl::IsFlyXTileTransposeFusion(analysis)) {
    return true;
  }
  if (flydsl::ContainsUnsupportedCustomCall(analysis)) {
    return false;
  }
  if (flydsl::IsFlyXTileElementwiseFusion(analysis) ||
      flydsl::IsFlyXTileRowReductionFusion(analysis)) {
    return true;
  }
  // Strict replacement is a native-Fly contract.  Do not attach __fly to an
  // otherwise valid XLA loop/reduction merely because the legacy generic
  // emitter can compile it: doing so hides coverage gaps and can also hand a
  // multi-output graph to a generic reduction emitter whose hero assumptions
  // do not apply.  Leave such fusions on their ordinary XLA backend until a
  // native Fly route owns their complete graph.
  if (debug_options().xla_gpu_flydsl_replace_triton()) {
    return false;
  }
  // Generic block-level fusions have already been marked custom, so their HLO
  // fusion analysis reports the selected backend instead of the underlying
  // loop, reduction, or transpose emitter kind. Fly dispatches those
  // structures directly in GetFusionEmitter and can retune them without
  // undoing the fusion.
  if (is_generic_block_fusion) {
    return true;
  }
  return analysis.emitter_fusion_kind() ==
             HloFusionAnalysis::EmitterFusionKind::kLoop ||
         analysis.emitter_fusion_kind() ==
             HloFusionAnalysis::EmitterFusionKind::kReduction ||
         analysis.emitter_fusion_kind() ==
             HloFusionAnalysis::EmitterFusionKind::kTranspose;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
FlyFusionBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }

  HloFusionAnalysis analysis =
      HloFusionAnalysis::Create(instr, target_config().device_description);
  if (std::optional<flydsl::FlyScanDescriptor> scan =
          flydsl::GetFlyScanDescriptor(analysis)) {
    // One Wave64 owns each contiguous row. Measure CTA packing and occupancy;
    // both vary with the number of striped 64-element prefix tiles per row.
    constexpr std::array<std::pair<int64_t, int64_t>, 8> kScanConfigs = {{
        {8, 0}, {4, 0}, {16, 0}, {2, 0},
        {8, 2}, {4, 2}, {1, 0},  {8, 4},
    }};
    for (auto [num_warps, waves_per_eu] : kScanConfigs) {
      auto config = std::make_unique<BackendConfig>();
      BlockLevelFusionConfig* block = config->mutable_block_level();
      Tile* tile = block->add_output_tiles();
      for (int64_t dimension = 0;
           dimension < scan->input->shape().dimensions_size(); ++dimension) {
        tile->add_sizes(1);
      }
      block->set_num_warps(num_warps);
      block->set_num_ctas(1);
      block->set_num_stages(1);
      block->set_waves_per_eu(waves_per_eu);
      configs.push_back(std::move(config));
    }
    return configs;
  }
  if (std::optional<flydsl::FlyPagedAttentionDescriptor> attention =
          flydsl::GetFlyPagedAttentionDescriptor(analysis)) {
    // The two-phase specialization has a fixed four-wave MFMA16 layout.  The
    // only profitable code-generation knob is the occupancy request.
    for (int64_t waves_per_eu : {0, 1, 2, 4}) {
      auto config = std::make_unique<BackendConfig>();
      BlockLevelFusionConfig* block = config->mutable_block_level();
      Tile* tile = block->add_output_tiles();
      tile->add_sizes(1);
      tile->add_sizes(1);
      tile->add_sizes(attention->head_dimension);
      block->set_num_warps(4);
      block->set_num_ctas(1);
      block->set_num_stages(1);
      block->set_waves_per_eu(waves_per_eu);
      configs.push_back(std::move(config));
    }
    return configs;
  }
  if (std::optional<flydsl::FlyPagedAttentionSegmentedProducerDescriptor>
          producer =
              flydsl::GetFlyPagedAttentionSegmentedProducerDescriptor(
                  analysis)) {
    const bool cooperative = producer->attention.max_context >= 65536 &&
                             producer->attention.gqa_group == 4;
    for (int64_t num_stages : cooperative
                                  ? std::initializer_list<int64_t>{1, 2, 3}
                                  : std::initializer_list<int64_t>{1}) {
      for (int64_t waves_per_eu : {0, 1, 2, 4}) {
        auto config = std::make_unique<BackendConfig>();
        BlockLevelFusionConfig* block = config->mutable_block_level();
        if (producer->fused_reducer) {
          Tile* final_tile = block->add_output_tiles();
          final_tile->add_sizes(1);
          final_tile->add_sizes(1);
          final_tile->add_sizes(producer->attention.head_dimension);
        }
        Tile* output_tile = block->add_output_tiles();
        output_tile->add_sizes(1);
        output_tile->add_sizes(1);
        output_tile->add_sizes(1);
        output_tile->add_sizes(producer->attention.head_dimension);
        for (int64_t state = 0; state < 2; ++state) {
          Tile* state_tile = block->add_output_tiles();
          state_tile->add_sizes(1);
          state_tile->add_sizes(1);
          state_tile->add_sizes(1);
        }
        if (producer->fused_reducer) {
          Tile* ticket_tile = block->add_output_tiles();
          ticket_tile->add_sizes(1);
          ticket_tile->add_sizes(1);
          ticket_tile->add_sizes(1);
        }
        block->set_num_warps((producer->attention.max_context >= 65536 ||
                              producer->segment_tokens <= 64)
                                 ? 2
                                 : 4);
        block->set_num_ctas(1);
        block->set_num_stages(num_stages);
        block->set_waves_per_eu(waves_per_eu);
        configs.push_back(std::move(config));
      }
    }
    return configs;
  }
  if (std::optional<flydsl::FlyPagedAttentionSegmentedReducerDescriptor>
          reducer =
              flydsl::GetFlyPagedAttentionSegmentedReducerDescriptor(
                  analysis)) {
    for (int64_t waves_per_eu : {0, 1, 2, 4}) {
      auto config = std::make_unique<BackendConfig>();
      BlockLevelFusionConfig* block = config->mutable_block_level();
      Tile* tile = block->add_output_tiles();
      tile->add_sizes(1);
      tile->add_sizes(1);
      tile->add_sizes(reducer->head_dimension);
      block->set_num_warps(1);
      block->set_num_ctas(1);
      block->set_num_stages(1);
      block->set_waves_per_eu(waves_per_eu);
      configs.push_back(std::move(config));
    }
    return configs;
  }
  if (std::optional<flydsl::FlyAttentionDescriptor> attention =
          flydsl::GetFlyAttentionDescriptor(analysis)) {
    // The reference FlyDSL gfx942 kernel assigns 32 query rows to every
    // Wave64. Tune the CTA's query-row tile and occupancy, while retaining
    // its fixed 64-key streaming tile and MFMA32 register mapping.
    for (int64_t block_m : {32, 64, 128, 256}) {
      if (block_m > attention->sequence || attention->sequence % block_m != 0) {
        continue;
      }
      for (int64_t waves_per_eu : {2, 4}) {
        auto config = std::make_unique<BackendConfig>();
        BlockLevelFusionConfig* block = config->mutable_block_level();
        Tile* tile = block->add_output_tiles();
        tile->add_sizes(1);
        tile->add_sizes(block_m);
        tile->add_sizes(1);
        tile->add_sizes(attention->head_dimension);
        block->set_num_warps(block_m / 32);
        block->set_num_ctas(1);
        block->set_num_stages(1);
        block->set_waves_per_eu(waves_per_eu);
        configs.push_back(std::move(config));
      }
    }
    return configs;
  }
  if (flydsl::IsFlySoftmaxFusion(analysis) ||
      flydsl::IsFlyLayerNormFusion(analysis)) {
    const Shape& shape = analysis.first_result_shape();
    const int64_t rank = shape.dimensions_size();
    const int64_t columns = shape.dimensions(rank - 1);
    const bool independent_rows = columns <= 256;
    for (int64_t num_warps : {1, 2, 4, 8, 16}) {
      const int64_t threads = num_warps * 64;
      const int64_t threads_per_row = independent_rows ? 64 : threads;
      const int64_t values_per_thread =
          (columns + threads_per_row - 1) / threads_per_row;
      const int64_t rounded_columns = ((columns + 63) / 64) * 64;
      if ((!independent_rows && threads > rounded_columns) ||
          values_per_thread > 64 ||
          (independent_rows && shape.dimensions(rank - 2) < num_warps)) {
        continue;
      }
      auto config = std::make_unique<BackendConfig>();
      BlockLevelFusionConfig* block = config->mutable_block_level();
      Tile* tile = block->add_output_tiles();
      for (int64_t dimension = 0; dimension < rank - 1; ++dimension) {
        tile->add_sizes(independent_rows && dimension == rank - 2 ? num_warps
                                                                  : 1);
      }
      tile->add_sizes(columns);
      block->set_num_warps(num_warps);
      block->set_num_ctas(1);
      block->set_num_stages(1);
      configs.push_back(std::move(config));
    }
    return configs;
  }
  if (flydsl::IsFlyXTileTransposeFusion(analysis)) {
    std::optional<std::pair<int64_t, int64_t>> matrix_shape =
        flydsl::GetFlyXTileTransposeMatrixShape(analysis);
    TF_RET_CHECK(matrix_shape.has_value());
    auto [rows, columns] = *matrix_shape;
    const int64_t element_bits =
        primitive_util::BitWidth(analysis.first_result_shape().element_type());
    const int64_t vector_width = 128 / element_bits;
    const int64_t maximum_tile_elements = 65536 * 8 / element_bits;
    for (int64_t tile_rows :
         {32, 64, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416}) {
      for (int64_t tile_columns : {32, 64, 128, 256}) {
        if (tile_rows > rows || tile_columns > columns ||
            (tile_rows + (element_bits == 32 ? 2 : 0)) * tile_columns >
                maximum_tile_elements) {
          continue;
        }
        for (int64_t vectors_per_thread : {2, 4}) {
          const int64_t threads = tile_rows * tile_columns /
                                  (vectors_per_thread * vector_width);
          if (threads < 64 || threads > 1024 || threads % 64 != 0) {
            continue;
          }
          auto config = std::make_unique<BackendConfig>();
          BlockLevelFusionConfig* block = config->mutable_block_level();
          Tile* tile = block->add_output_tiles();
          tile->add_sizes(tile_rows);
          tile->add_sizes(tile_columns);
          block->set_num_warps(threads / 64);
          block->set_num_ctas(1);
          block->set_num_stages(1);
          configs.push_back(std::move(config));
        }
      }
    }
    return configs;
  }
  if (flydsl::IsFlyXTileRowReductionFusion(analysis)) {
    // FlyDSL's native reduction kernels issue one 64-lane wave per row and
    // use 128-bit buffer copies on the fast path. Also keep 64-bit copies in
    // the search because the smaller register vectors can win at short row
    // sizes or under register pressure from fused input expressions.
    auto add_configs = [&](int64_t vector_size_bits, int64_t output_partitions,
                           absl::Span<const int64_t> warp_counts) {
      if (!flydsl::IsFlyXTileRowReductionConfigSupported(
              analysis, vector_size_bits, output_partitions)) {
        return false;
      }
      for (int64_t num_warps : warp_counts) {
        auto config = std::make_unique<BackendConfig>();
        BlockLevelFusionConfig* block = config->mutable_block_level();
        Tile* tile = block->add_output_tiles();
        tile->add_sizes(output_partitions);
        block->set_num_warps(num_warps);
        block->set_num_ctas(1);
        block->set_num_stages(1);
        block->set_vector_size_bits(vector_size_bits);
        configs.push_back(std::move(config));
      }
      return true;
    };
    constexpr std::array<int64_t, 5> kFullRowWarps = {1, 2, 4, 8, 16};
    for (int64_t vector_size_bits : {64, 128}) {
      add_configs(vector_size_bits, /*output_partitions=*/1, kFullRowWarps);
    }
    // RMSNorm-like fusions with only a few hundred rows underfill MI300X when
    // one wave owns a whole row. Match the block emitter's successful 128-wide
    // output geometry by redundantly reducing each row in eight partitions.
    // The autotuner retains the non-redundant alternatives above for larger
    // batches and bandwidth-bound reductions.
    constexpr std::array<int64_t, 3> kPartitionedWarps = {1, 2, 4};
    for (int64_t output_partitions : {2, 4}) {
      for (int64_t vector_size_bits : {64, 128}) {
        add_configs(vector_size_bits, output_partitions, kPartitionedWarps);
      }
    }
    for (int64_t vector_size_bits : {32, 64}) {
      add_configs(vector_size_bits, /*output_partitions=*/8, kPartitionedWarps);
    }
    return configs;
  }
  const auto& heroes = analysis.fusion_heroes();
  const bool is_native_elementwise =
      flydsl::IsFlyXTileElementwiseFusion(analysis);
  const bool has_tiled_transpose_hero =
      std::any_of(heroes.begin(), heroes.end(), [](const auto& hero) {
        return GetDescriptionForTiledTransposeEmitter(hero.instruction())
            .has_value();
      });
  if (has_tiled_transpose_hero) {
    auto config = std::make_unique<BackendConfig>();
    BlockLevelFusionConfig* block = config->mutable_block_level();
    Tile* tile = block->add_output_tiles();
    for (int64_t dimension = 0;
         dimension < analysis.first_result_shape().dimensions_size();
         ++dimension) {
      tile->add_sizes(1);
    }
    block->set_num_warps(4);
    block->set_num_ctas(1);
    block->set_num_stages(1);
    configs.push_back(std::move(config));
    return configs;
  }
  const HloInstruction* reduction_hero = nullptr;
  for (const HloInstructionAdaptor& hero : heroes) {
    if (hero.opcode() == HloOpcode::kReduce) {
      reduction_hero = &hero.instruction();
      break;
    }
  }
  if (reduction_hero != nullptr && !is_native_elementwise) {
    ReductionDimensions dimensions =
        GetReductionKindAndContiguousComponents(*reduction_hero);
    auto add_reduction_config = [&](int64_t elements_per_thread,
                                    int64_t num_warps) {
      auto config = std::make_unique<BackendConfig>();
      BlockLevelFusionConfig* block = config->mutable_block_level();
      Tile* tile = block->add_output_tiles();
      tile->add_sizes(elements_per_thread);
      block->set_num_warps(num_warps);
      block->set_num_ctas(1);
      block->set_num_stages(1);
      configs.push_back(std::move(config));
    };
    if (dimensions.is_row_reduction) {
      // Start with XLA's native geometry, then cover one to sixteen AMD waves
      // per row while preserving a single-block reduction.
      add_reduction_config(/*elements_per_thread=*/16, /*num_warps=*/4);
      add_reduction_config(/*elements_per_thread=*/8, /*num_warps=*/8);
      add_reduction_config(/*elements_per_thread=*/4, /*num_warps=*/16);
      add_reduction_config(/*elements_per_thread=*/32, /*num_warps=*/2);
      add_reduction_config(/*elements_per_thread=*/64, /*num_warps=*/1);
    } else {
      add_reduction_config(/*elements_per_thread=*/1, /*num_warps=*/4);
    }
    return configs;
  }

  const bool is_native_overwrite_scatter =
      flydsl::IsFlyXTileOverwriteRowScatterFusion(analysis);
  const bool is_native_atomic_overwrite_scatter =
      flydsl::IsFlyXTileAtomicOverwriteRowScatterFusion(analysis);
  const int64_t elements =
      flydsl::GetFlyXTileElementwiseElementCount(analysis);
  const int64_t native_unroll =
      is_native_overwrite_scatter ? 1 : ComputeLoopFusionConfig(analysis);
  const bool is_native_indexed = flydsl::IsFlyXTileIndexedFusion(analysis);
  const bool has_type_changing_bitcast =
      is_native_elementwise &&
      absl::c_any_of(
          instr.fused_instructions_computation()->instructions(),
          [](const HloInstruction* instruction) {
            return (instruction->opcode() == HloOpcode::kBitcast ||
                    instruction->opcode() == HloOpcode::kBitcastConvert) &&
                   instruction->operand_count() == 1 &&
                   instruction->shape().element_type() !=
                       instruction->operand(0)->shape().element_type();
          });
  const bool has_shape_changing_physical_view =
      is_native_elementwise &&
      absl::c_any_of(
          instr.fused_instructions_computation()->instructions(),
          [](const HloInstruction* instruction) {
            switch (instruction->opcode()) {
              case HloOpcode::kBitcast:
              case HloOpcode::kBitcastConvert:
              case HloOpcode::kReshape:
              case HloOpcode::kTranspose:
                return instruction->operand_count() == 1 &&
                       !ShapeUtil::Equal(instruction->shape(),
                                         instruction->operand(0)->shape());
              default:
                return false;
            }
          });
  const bool has_native_leading_reduction =
      is_native_elementwise &&
      absl::c_any_of(instr.fused_instructions_computation()->instructions(),
                     [](const HloInstruction* instruction) {
                       return instruction->opcode() == HloOpcode::kReduce &&
                              instruction->dimensions().size() == 1 &&
                              instruction->dimensions(0) == 0;
                     });
  const bool has_native_cooperative_strided_reduction =
      is_native_elementwise &&
      absl::c_any_of(
          instr.fused_instructions_computation()->instructions(),
          [](const HloInstruction* instruction) {
            if (instruction->opcode() != HloOpcode::kReduce ||
                instruction->dimensions().size() != 1 ||
                instruction->operand_count() != 2 ||
                instruction->shape().element_type() != F32 ||
                instruction->operand(0)->shape().element_type() != F32) {
              return false;
            }
            const Shape& input = instruction->operand(0)->shape();
            const int64_t reduction_dimension = instruction->dimensions(0);
            if (reduction_dimension <= 0 ||
                reduction_dimension >= input.dimensions_size() - 1 ||
                input.dimensions(reduction_dimension) % 4 != 0) {
              return false;
            }
            int64_t inner_elements = 1;
            for (int64_t dimension = reduction_dimension + 1;
                 dimension < input.dimensions_size(); ++dimension) {
              inner_elements *= input.dimensions(dimension);
            }
            return inner_elements % 256 == 0;
          });
  const bool has_native_arbitrary_broadcast =
      is_native_indexed &&
      absl::c_any_of(
          instr.fused_instructions_computation()->instructions(),
          [](const HloInstruction* instruction) {
            if (instruction->opcode() != HloOpcode::kBroadcast ||
                instruction->operand_count() != 1 ||
                ShapeUtil::IsScalar(instruction->operand(0)->shape())) {
              return false;
            }
            const int64_t input_rank =
                instruction->operand(0)->shape().dimensions_size();
            const int64_t output_rank = instruction->shape().dimensions_size();
            bool leading = instruction->dimensions().size() == input_rank;
            bool trailing = leading;
            for (int64_t dimension = 0; dimension < input_rank; ++dimension) {
              leading &= instruction->dimensions(dimension) == dimension;
              trailing &= instruction->dimensions(dimension) ==
                          output_rank - input_rank + dimension;
            }
            return !leading && !trailing;
          });
  const bool has_native_nonleading_concatenate =
      is_native_indexed &&
      absl::c_any_of(instr.fused_instructions_computation()->instructions(),
                     [](const HloInstruction* instruction) {
                       return instruction->opcode() ==
                                  HloOpcode::kConcatenate &&
                              instruction->dimensions().size() == 1 &&
                              instruction->dimensions(0) != 0;
                     });
  const bool is_native_slice =
      is_native_indexed && (hlo_query::GetFirstInstructionWithOpcode(
                                *instr.fused_instructions_computation(),
                                HloOpcode::kSlice) != nullptr ||
                            hlo_query::GetFirstInstructionWithOpcode(
                                *instr.fused_instructions_computation(),
                                HloOpcode::kDynamicSlice) != nullptr);
  const bool has_native_minor_strided_slice =
      is_native_indexed &&
      absl::c_any_of(
          instr.fused_instructions_computation()->instructions(),
          [](const HloInstruction* instruction) {
            return instruction->opcode() == HloOpcode::kSlice &&
                   !instruction->slice_strides().empty() &&
                   instruction->slice_strides().back() > 1;
          });
  const bool is_native_dynamic_update =
      is_native_indexed && hlo_query::GetFirstInstructionWithOpcode(
                               *instr.fused_instructions_computation(),
                               HloOpcode::kDynamicUpdateSlice) != nullptr;
  const bool is_native_gather =
      is_native_indexed && hlo_query::GetFirstInstructionWithOpcode(
                               *instr.fused_instructions_computation(),
                               HloOpcode::kGather) != nullptr;
  const bool is_native_reverse =
      is_native_indexed &&
      hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kReverse) !=
          nullptr;
  const bool is_native_pad =
      is_native_indexed &&
      hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kPad) != nullptr;
  const bool is_native_reduce_window =
      is_native_indexed &&
      hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(),
          HloOpcode::kReduceWindow) != nullptr;
  const bool is_native_data_moving_transpose =
      is_native_indexed &&
      hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kTranspose) !=
          nullptr;
  auto storage_bit_width = [](PrimitiveType type) {
    // XLA's device ABI stores PRED as one byte even though its logical value
    // has one bit.
    return type == PRED ? int64_t{8} : primitive_util::BitWidth(type);
  };
  int64_t max_external_element_bits =
      storage_bit_width(analysis.first_result_shape().element_type());
  const std::vector<int64_t> scalar_index_parameter_numbers =
      is_native_elementwise
          ? flydsl::GetFlyXTileScalarIndexParameterNumbers(analysis)
          : std::vector<int64_t>{};
  for (int64_t root_index = 1; root_index < analysis.fusion_root_count();
       ++root_index) {
    max_external_element_bits = std::max<int64_t>(
        max_external_element_bits,
        storage_bit_width(
            analysis.fusion_root(root_index).shape().element_type()));
  }
  for (auto [operand_index, operand] : llvm::enumerate(instr.operands())) {
    if (ShapeUtil::IsScalar(operand->shape()) &&
        primitive_util::IsIntegralType(operand->shape().element_type())) {
      continue;
    }
    if (std::binary_search(scalar_index_parameter_numbers.begin(),
                           scalar_index_parameter_numbers.end(),
                           operand_index)) {
      continue;
    }
    max_external_element_bits =
        std::max<int64_t>(max_external_element_bits,
                          storage_bit_width(operand->shape().element_type()));
  }
  const int64_t max_elementwise_configs = std::max<int64_t>(
      1, debug_options().xla_gpu_fusion_autotune_top_k_configs());

  auto add_config = [&](int64_t unroll, int64_t num_warps,
                        int64_t vector_size_bits, int64_t waves_per_eu = 0) {
    if (unroll <= 0 || unroll > elements) {
      return;
    }
    if (has_native_cooperative_strided_reduction && num_warps == 4 &&
        vector_size_bits == 128 && unroll != 1) {
      return;
    }
    const bool scalarized_wide_s4_output =
        analysis.first_result_shape().element_type() == S4 &&
        vector_size_bits == 16;
    if (is_native_elementwise && !has_type_changing_bitcast &&
        vector_size_bits > 0 &&
        vector_size_bits /
                storage_bit_width(
                    analysis.first_result_shape().element_type()) *
                max_external_element_bits >
            128 &&
        !scalarized_wide_s4_output) {
      return;
    }
    for (const auto& existing : configs) {
      if (existing->block_level().output_tiles(0).sizes(0) == unroll &&
          existing->block_level().num_warps() == num_warps &&
          existing->block_level().vector_size_bits() == vector_size_bits &&
          existing->block_level().waves_per_eu() == waves_per_eu) {
        return;
      }
    }
    auto config = std::make_unique<BackendConfig>();
    BlockLevelFusionConfig* block = config->mutable_block_level();
    Tile* tile = block->add_output_tiles();
    tile->add_sizes(unroll);
    block->set_num_warps(num_warps);
    block->set_num_ctas(1);
    block->set_num_stages(1);
    block->set_vector_size_bits(vector_size_bits);
    block->set_waves_per_eu(waves_per_eu);
    configs.push_back(std::move(config));
  };

  if (is_native_indexed) {
    const int64_t scalar_bits =
        primitive_util::BitWidth(analysis.first_result_shape().element_type());
    if (has_native_arbitrary_broadcast) {
      // A wide vector amortizes the row-major dimension-map calculation. The
      // emitter carries middle-dimension broadcast values across striped
      // output vectors, so rank the measured gfx942 reuse schedules before
      // the generic indexed-fusion search consumes the top-k budget.
      constexpr std::array<std::array<int64_t, 4>, 8> kBroadcastConfigs = {{
          {8, 8, 64, 0},
          {8, 8, 64, 2},
          {8, 8, 64, 4},
          {1, 8, 128, 0},
          {1, 8, 128, 2},
          {1, 8, 128, 4},
          {4, 4, 128, 0},
          {4, 4, 128, 4},
      }};
      for (const auto& [unroll, num_warps, vector_size_bits, waves_per_eu] :
           kBroadcastConfigs) {
        add_config(unroll, num_warps, vector_size_bits, waves_per_eu);
        if (configs.size() >= max_elementwise_configs) {
          return configs;
        }
      }
    }
    if (has_native_nonleading_concatenate) {
      // A middle/minor concatenate repeats each operand segment for every
      // outer coordinate. Rank the measured gfx942 128-bit schedules that
      // amortize the segment remapping and keep full-vector copies on the
      // ordinary path; only vectors crossing a segment edge scalarize.
      constexpr std::array<std::array<int64_t, 3>, 8> kConcatConfigs = {{
          {4, 4, 128},
          {4, 2, 128},
          {2, 4, 128},
          {4, 8, 128},
          {2, 2, 128},
          {8, 4, 128},
          {4, 4, 64},
          {8, 4, 64},
      }};
      for (const auto& [unroll, num_warps, vector_size_bits] : kConcatConfigs) {
        add_config(unroll, num_warps, vector_size_bits);
        if (configs.size() >= max_elementwise_configs) {
          return configs;
        }
      }
    }
    if (is_native_reduce_window) {
      // Sliding windows reuse overlapping input spans in registers. Put the
      // 128-bit geometries that minimize global transactions ahead of the
      // generic indexed-fusion order; otherwise XLA's default top-eight cap
      // is exhausted entirely by 64-bit concatenate-oriented candidates.
      constexpr std::array<std::array<int64_t, 3>, 8> kPreferredConfigs = {{
          {2, 8, 128},
          {2, 4, 128},
          {4, 4, 128},
          {1, 8, 128},
          {4, 8, 128},
          {1, 4, 128},
          {4, 8, 64},
          {8, 4, 64},
      }};
      for (const auto& [unroll, num_warps, vector_size_bits] :
           kPreferredConfigs) {
        add_config(unroll, num_warps, vector_size_bits);
        if (configs.size() >= max_elementwise_configs) {
          return configs;
        }
      }
    }
    if (is_native_data_moving_transpose) {
      // A data-moving transpose either preserves a contiguous physical-minor
      // input dimension or gathers lanes from a short packed span. Interleave
      // 64- and 128-bit copies and small/large CTAs in the ordinary top-eight
      // budget so neither rank-3 case is forced into one geometry.
      constexpr std::array<std::array<int64_t, 3>, 8> kTransposeConfigs = {{
          {1, 2, 64},
          {1, 2, 128},
          {1, 4, 64},
          {1, 4, 128},
          {2, 2, 64},
          {2, 4, 64},
          {1, 8, 64},
          {1, 8, 128},
      }};
      for (const auto& [unroll, num_warps, vector_size_bits] :
           kTransposeConfigs) {
        add_config(unroll, num_warps, vector_size_bits);
        if (configs.size() >= max_elementwise_configs) {
          return configs;
        }
      }
    }
    if (is_native_slice) {
      if (has_native_minor_strided_slice) {
        // Minor-stride slices gather several logical lanes from one physical
        // span.  Prefer narrower outputs and tune occupancy explicitly: the
        // emitter can service a 64-bit BF16 output with one 128-bit packed
        // load, while 128-bit output vectors require scalar gathers.
        for (int64_t waves_per_eu : {0, 1, 2, 4}) {
          add_config(/*unroll=*/1, /*num_warps=*/8,
                     /*vector_size_bits=*/64, waves_per_eu);
        }
        add_config(/*unroll=*/2, /*num_warps=*/8,
                   /*vector_size_bits=*/64);
        add_config(/*unroll=*/1, /*num_warps=*/4,
                   /*vector_size_bits=*/64);
        add_config(/*unroll=*/1, /*num_warps=*/8,
                   /*vector_size_bits=*/32);
        add_config(/*unroll=*/2, /*num_warps=*/8,
                   /*vector_size_bits=*/32);
        if (configs.size() >= max_elementwise_configs) {
          return configs;
        }
      }
      // Contiguous slices have no boundary scalarization in the main vector
      // loop. Start with the measured eight-wave/128-bit winner, including
      // occupancy requests, then keep both copy widths and nearby unrolls in
      // the ordinary top-eight search. The 64-bit alternatives also cover a
      // slice combined with another indexed operation such as concatenate.
      for (int64_t waves_per_eu : {0, 1, 2, 4}) {
        add_config(/*unroll=*/1, /*num_warps=*/8,
                   /*vector_size_bits=*/128, waves_per_eu);
      }
      add_config(/*unroll=*/2, /*num_warps=*/8,
                 /*vector_size_bits=*/64);
      add_config(/*unroll=*/1, /*num_warps=*/8,
                 /*vector_size_bits=*/64);
      add_config(/*unroll=*/2, /*num_warps=*/8,
                 /*vector_size_bits=*/128);
      add_config(/*unroll=*/4, /*num_warps=*/8,
                 /*vector_size_bits=*/128);
      if (configs.size() >= max_elementwise_configs) {
        return configs;
      }
      for (int64_t num_warps : {8, 4, 2, 1}) {
        for (int64_t unroll : {4, 2, 8, 1}) {
          for (int64_t vector_bits : {128, 64}) {
            add_config(unroll, num_warps, vector_bits);
            if (configs.size() >= max_elementwise_configs) {
              return configs;
            }
          }
        }
      }
    }
    if (is_native_dynamic_update) {
      // Dynamic update retains full-width loads and stores except for the two
      // transactions that intersect each row's update boundaries. Rank the
      // measured MI300X 128-bit families first and retain the best scalar-copy
      // geometry as protection for boundary-heavy small updates.
      constexpr std::array<std::array<int64_t, 3>, 8> kPreferredConfigs = {{
          {2, 4, 128},
          {1, 8, 128},
          {1, 1, 64},
          {1, 4, 128},
          {1, 1, 128},
          {4, 8, 128},
          {2, 2, 128},
          {4, 4, 128},
      }};
      for (const auto& [unroll, num_warps, vector_size_bits] :
           kPreferredConfigs) {
        add_config(unroll, num_warps, vector_size_bits);
        if (configs.size() >= max_elementwise_configs) {
          return configs;
        }
      }
    }
    if (is_native_reverse || is_native_pad) {
      // A flat reversal or edge pad retains contiguous vector loads but adds
      // indexing/control flow around them. Rank the stable
      // 128-bit/single-vector and 64-bit/two-vector families across all Wave64
      // workgroup sizes.
      for (int64_t num_warps : {8, 4, 2, 1}) {
        add_config(/*unroll=*/1, num_warps, /*vector_size_bits=*/128);
        add_config(/*unroll=*/2, num_warps, /*vector_size_bits=*/64);
        if (configs.size() >= max_elementwise_configs) {
          return configs;
        }
      }
    }
    if (is_native_gather) {
      // Gather indices are loaded as scalars and do not constrain the
      // transaction width of the contiguous row payload. Interleave 64- and
      // 128-bit row copies inside the ordinary top-eight budget.
      constexpr std::array<std::array<int64_t, 3>, 8> kGatherConfigs = {{
          {8, 8, 128},
          {8, 8, 64},
          {4, 8, 128},
          {4, 8, 64},
          {4, 4, 128},
          {4, 4, 64},
          {2, 4, 128},
          {2, 4, 64},
      }};
      for (const auto& [unroll, num_warps, vector_size_bits] :
           kGatherConfigs) {
        add_config(unroll, num_warps, vector_size_bits);
        if (configs.size() >= max_elementwise_configs) {
          return configs;
        }
      }
    }
    if (is_native_overwrite_scatter) {
      // Row scatter traverses the update payload, not the much larger aliased
      // output. Unique indices retain wide suffix stores; colliding indices
      // issue one scalar atomic exchange per lane, so rank scalar transactions
      // with enough per-thread unroll to amortize address calculation.
      const std::array<std::array<int64_t, 3>, 8> kScatterConfigs =
          is_native_atomic_overwrite_scatter
              ? std::array<std::array<int64_t, 3>, 8>{{
                    {8, 8, 32},
                    {4, 8, 32},
                    {2, 8, 32},
                    {1, 8, 32},
                    {8, 4, 32},
                    {4, 4, 32},
                    {4, 2, 32},
                    {4, 1, 32},
                }}
              : std::array<std::array<int64_t, 3>, 8>{{
                    {1, 8, 128},
                    {2, 8, 128},
                    {4, 8, 128},
                    {1, 4, 128},
                    {2, 4, 128},
                    {4, 4, 128},
                    {1, 8, 64},
                    {2, 8, 64},
                }};
      for (const auto& [unroll, num_warps, vector_size_bits] :
           kScatterConfigs) {
        add_config(unroll, num_warps, vector_size_bits);
        if (configs.size() >= max_elementwise_configs) {
          return configs;
        }
      }
    }
    // Full vectors remain legal everywhere except the single transaction that
    // straddles each concatenation boundary; the emitter scalarizes that rare
    // path. Rank vector copies first and keep a scalar candidate as a robust
    // alternative for very short fragments.
    for (int64_t vector_bits :
         std::array<int64_t, 3>{64, 128, scalar_bits}) {
      for (int64_t num_warps : {8, 4, 2, 1}) {
        for (int64_t unroll : {4, 2, 8, 1}) {
          add_config(unroll, num_warps, vector_bits);
          if (configs.size() >= max_elementwise_configs) {
            return configs;
          }
        }
      }
    }
    return configs;
  }

  if (is_native_elementwise) {
    if (has_native_cooperative_strided_reduction) {
      // Split a non-minor reduction across four Wave64s while each lane loads
      // four contiguous output columns. This is the native Fly equivalent of
      // Triton's cooperative [1, 256] reduction tile on gfx942.
      add_config(/*unroll=*/1, /*num_warps=*/4,
                 /*vector_size_bits=*/128);
      if (configs.size() >= max_elementwise_configs) {
        return configs;
      }
    }
    if (has_native_leading_reduction) {
      // Leading reductions run one independent accumulator per output lane.
      // A single Wave64 block with a 64-bit output vector reaches the MI300X
      // HBM ceiling for the common split/leading dimensions. Tune occupancy
      // explicitly; larger CTAs only reduce the number of schedulable blocks.
      for (int64_t waves_per_eu : {0, 2, 4}) {
        add_config(/*unroll=*/1, /*num_warps=*/1,
                   /*vector_size_bits=*/64, waves_per_eu);
      }
      add_config(/*unroll=*/1, /*num_warps=*/1,
                 /*vector_size_bits=*/128);
      if (configs.size() >= max_elementwise_configs) {
        return configs;
      }
    }
    // Rank the native MI300X configurations before applying XLA's top-k
    // fusion-autotuning limit. Two-wave, single-vector blocks are the stable
    // steady-state winner across F16, BF16, F32, aligned, and ragged HBM
    // graphs. Interleave dtype-specific alternatives early enough that an
    // explicit wider search still measures their throughput/register-pressure
    // tradeoffs.
    constexpr std::array<std::array<int64_t, 3>, 8> kAlignedPreferredConfigs = {
        {
            {1, 2, 64},
            {1, 2, 128},
            {2, 4, 64},
            {1, 4, 64},
            {2, 4, 128},
            {2, 2, 64},
            {4, 4, 64},
            {1, 4, 128},
        }};
    constexpr std::array<std::array<int64_t, 3>, 8> kTailPreferredConfigs = {{
        {1, 2, 64},
        {1, 2, 128},
        {2, 2, 64},
        {2, 4, 64},
        {2, 4, 128},
        {1, 4, 64},
        {4, 4, 64},
        {1, 4, 128},
    }};
    constexpr std::array<std::array<int64_t, 3>, 8> kF16PreferredConfigs = {{
        {1, 2, 64},
        {1, 2, 128},
        {2, 4, 64},
        {1, 4, 64},
        {2, 2, 64},
        {2, 4, 128},
        {4, 4, 64},
        {1, 4, 128},
    }};
    constexpr std::array<std::array<int64_t, 3>, 8> kF32PreferredConfigs = {{
        {1, 2, 64},
        {1, 4, 64},
        {1, 2, 128},
        {2, 2, 64},
        {2, 4, 64},
        {2, 4, 128},
        {4, 4, 64},
        {1, 4, 128},
    }};
    // Shape-changing physical views remain flat, coalesced memory operations,
    // but the extra indexing context changes the best workgroup geometry on
    // gfx942.  Keep the measured 128-bit winners inside XLA's ordinary
    // top-eight budget instead of requiring an expanded diagnostic search.
    constexpr std::array<std::array<int64_t, 3>, 8>
        kPhysicalViewPreferredConfigs = {{{1, 2, 64},
                                          {1, 8, 128},
                                          {2, 1, 128},
                                          {2, 4, 128},
                                          {1, 2, 128},
                                          {1, 4, 128},
                                          {2, 2, 64},
                                          {2, 2, 128}}};
    const PrimitiveType element_type =
        analysis.first_result_shape().element_type();
    const int64_t element_bits = storage_bit_width(element_type);
    const bool has_scalar_tail = elements % (64 / element_bits) != 0;
    // Mixed-width conversions use the output lane count for every external
    // buffer. When a narrow result reads a wider input, 64- and 128-bit output
    // vectors can therefore exceed the copy atom's 128-bit transaction limit.
    // Seed the search with the widest legal narrow vector instead of dropping
    // every native Fly candidate (for example, S8 output fed by F64 input).
    const int64_t maximum_safe_vector_bits =
        128 * element_bits / max_external_element_bits;
    if (!has_type_changing_bitcast && maximum_safe_vector_bits < 64) {
      constexpr std::array<std::array<int64_t, 2>, 8>
          kNarrowPreferredConfigs = {{{1, 2},
                                      {1, 4},
                                      {2, 2},
                                      {2, 4},
                                      {4, 2},
                                      {4, 4},
                                      {1, 1},
                                      {1, 8}}};
      for (int64_t vector_size_bits : {32, 16}) {
        if ((vector_size_bits <= maximum_safe_vector_bits ||
             (element_type == S4 && vector_size_bits == 16)) &&
            vector_size_bits % element_bits == 0) {
          for (const auto& [unroll, num_warps] :
               kNarrowPreferredConfigs) {
            add_config(unroll, num_warps, vector_size_bits);
            if (configs.size() >= max_elementwise_configs) {
              return configs;
            }
          }
          break;
        }
      }
    }
    const auto& preferred_configs =
        has_shape_changing_physical_view
            ? kPhysicalViewPreferredConfigs
        : element_type == F16   ? kF16PreferredConfigs
        : element_type == F32   ? kF32PreferredConfigs
        : has_scalar_tail       ? kTailPreferredConfigs
                                : kAlignedPreferredConfigs;
    for (const auto& [unroll, num_warps, vector_size_bits] :
         preferred_configs) {
      add_config(unroll, num_warps, vector_size_bits);
      if (configs.size() >= max_elementwise_configs) {
        return configs;
      }
    }
    for (int64_t vector_size_bits : {64, 128}) {
      add_config(native_unroll, 4, vector_size_bits);
      if (configs.size() >= max_elementwise_configs) {
        return configs;
      }
      for (int64_t num_warps : {1, 2, 4, 8}) {
        for (int64_t unroll : {1, 2, 4, 8}) {
          add_config(unroll, num_warps, vector_size_bits);
          if (configs.size() >= max_elementwise_configs) {
            return configs;
          }
        }
      }
    }
    return configs;
  }

  add_config(native_unroll, 4, /*vector_size_bits=*/0);
  for (int64_t num_warps : {1, 2, 4, 8}) {
    for (int64_t unroll : {1, 2, 4, 8}) {
      add_config(unroll, num_warps, /*vector_size_bits=*/0);
    }
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
FlyFusionBackend::GetDefaultConfig(const HloInstruction& instr) {
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                   GetSupportedConfigs(instr));
  if (configs.empty()) {
    return absl::InvalidArgumentError(
        "FlyFusionBackend has no supported configuration for this "
        "instruction.");
  }
  return std::move(configs.front());
}

absl::Status FlyFusionBackend::ApplyConfig(HloInstruction& instr,
                                           const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "FlyFusionBackend does not support this instruction.");
  }
  if (!config.has_block_level() ||
      config.block_level().output_tiles().empty()) {
    return absl::InvalidArgumentError(
        "Expected a non-empty Fly fusion tile configuration.");
  }
  for (const Tile& output_tile : config.block_level().output_tiles()) {
    if (output_tile.sizes().empty() ||
        absl::c_any_of(output_tile.sizes(),
                       [](int64_t size) { return size <= 0; })) {
      return absl::InvalidArgumentError(
          "Expected positive Fly fusion tile sizes.");
    }
  }

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig* fusion_config =
      gpu_config.mutable_fusion_backend_config();
  fusion_config->set_kind(kFlyFusionKind);
  *fusion_config->mutable_block_level_fusion_config() = config.block_level();
  gpu_config.clear_native_emitter_backend_config();
  RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  instr.set_fusion_kind(HloInstruction::FusionKind::kCustom);
  return absl::OkStatus();
}

}  // namespace xla::gpu
