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
#include "xla/tsl/platform/status_macros.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_softmax.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_transpose.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
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
  return type == BF16 || type == F16 || type == F32 ||
         type == F8E8M0FNU || type == F8E4M3FN || type == F8E5M2 ||
         type == S8;
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
  for (int64_t dimension = 0;
       dimension < operand.shape().dimensions_size(); ++dimension) {
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

const HloInstruction* FindContractionInputParameter(
    const HloInstruction& input) {
  const PrimitiveType contraction_type = input.shape().element_type();
  if (!IsHalfPrecisionFloat(contraction_type) &&
      !IsFnuzFp8(contraction_type) && contraction_type != F32) {
    return nullptr;
  }
  const HloInstruction* value = &input;
  bool has_conversion = false;
  bool has_input_view = false;
  bool physical_mapping_fixed = false;
  bool has_dynamic_slice = false;
  while (value->opcode() == HloOpcode::kBitcast ||
         value->opcode() == HloOpcode::kConvert ||
         value->opcode() == HloOpcode::kSlice ||
         value->opcode() == HloOpcode::kDynamicSlice) {
    const bool is_dynamic_slice = value->opcode() == HloOpcode::kDynamicSlice;
    if ((!is_dynamic_slice && value->operand_count() != 1) ||
        (is_dynamic_slice &&
         value->operand_count() != value->shape().dimensions_size() + 1)) {
      return nullptr;
    }
    const HloInstruction* operand = value->operand(0);
    if (value->opcode() == HloOpcode::kBitcast) {
      if (value->shape().element_type() != operand->shape().element_type() ||
          ShapeUtil::ElementsIn(value->shape()) !=
              ShapeUtil::ElementsIn(operand->shape())) {
        return nullptr;
      }
      has_input_view = true;
      physical_mapping_fixed = true;
    } else if (value->opcode() == HloOpcode::kConvert) {
      if (!IsSupportedContractionConversion(operand->shape().element_type(),
                                            value->shape().element_type()) ||
          !ShapeUtil::SameDimensions(value->shape(), operand->shape())) {
        return nullptr;
      }
      has_conversion = true;
    } else if (value->opcode() == HloOpcode::kSlice) {
      if (physical_mapping_fixed ||
          value->shape().dimensions_size() !=
              operand->shape().dimensions_size() ||
          !std::all_of(value->slice_strides().begin(),
                       value->slice_strides().end(),
                       [](int64_t stride) { return stride == 1; })) {
        return nullptr;
      }
      has_input_view = true;
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
    // Sub-byte views need nibble-aware physical offsets. Admit only direct
    // dequantization until those views are represented explicitly.
    return contraction_type == BF16 && has_conversion && !has_input_view
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

bool IsS4DequantizedInput(const HloInstruction& input) {
  const HloInstruction* parameter = FindContractionInputParameter(input);
  return parameter != nullptr && parameter->shape().element_type() == S4;
}

struct ConcatInputInfo {
  int64_t dimension;
  int64_t fragment_size;
};

std::optional<ConcatInputInfo> FindSupportedConcatInput(
    const HloInstruction& input) {
  if (input.opcode() != HloOpcode::kConcatenate || input.operand_count() < 2 ||
      input.shape().element_type() != BF16) {
    return std::nullopt;
  }
  const int64_t dimension = input.concatenate_dimension();
  const Shape& first_shape = input.operand(0)->shape();
  PrimitiveType parameter_type = PRIMITIVE_TYPE_INVALID;
  for (const HloInstruction* operand : input.operands()) {
    if (!ShapeUtil::Equal(operand->shape(), first_shape)) {
      return std::nullopt;
    }
    const HloInstruction* parameter = nullptr;
    if (operand->opcode() == HloOpcode::kParameter &&
        operand->shape().element_type() == BF16) {
      parameter = operand;
    } else if (operand->opcode() == HloOpcode::kConvert &&
               operand->operand_count() == 1 &&
               operand->shape().element_type() == BF16 &&
               operand->operand(0)->opcode() == HloOpcode::kParameter &&
               operand->operand(0)->shape().element_type() == F32 &&
               ShapeUtil::SameDimensions(operand->shape(),
                                         operand->operand(0)->shape())) {
      parameter = operand->operand(0);
    }
    if (parameter == nullptr ||
        (parameter_type != PRIMITIVE_TYPE_INVALID &&
         parameter->shape().element_type() != parameter_type)) {
      return std::nullopt;
    }
    parameter_type = parameter->shape().element_type();
  }
  return ConcatInputInfo{dimension, first_shape.dimensions(dimension)};
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
  return dims.lhs_batch_dimensions_size() == 1 &&
         dims.rhs_batch_dimensions_size() == 1 &&
         dims.lhs_batch_dimensions(0) == 1 &&
         dims.rhs_batch_dimensions(0) == 1 &&
         dims.lhs_contracting_dimensions_size() == 1 &&
         dims.rhs_contracting_dimensions_size() == 1 &&
         dims.lhs_contracting_dimensions(0) == 2 &&
         dims.rhs_contracting_dimensions(0) == 2 &&
         dot.shape().element_type() == F32;
}

bool IsBatchedContraction(const HloInstruction& dot) {
  if (dot.shape().dimensions_size() != 3) {
    return false;
  }
  const DotDimensionNumbers& dims = dot.dot_dimension_numbers();
  return dims.lhs_batch_dimensions_size() == 1 &&
         dims.rhs_batch_dimensions_size() == 1 &&
         dims.lhs_batch_dimensions(0) == 0 &&
         dims.rhs_batch_dimensions(0) == 0 &&
         dims.lhs_contracting_dimensions_size() == 1 &&
         dims.rhs_contracting_dimensions_size() == 1 &&
         dims.lhs_contracting_dimensions(0) == 2 &&
         (dims.rhs_contracting_dimensions(0) == 1 ||
          dims.rhs_contracting_dimensions(0) == 2);
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
  const bool is_int4 = IsS4DequantizedInput(*dot.operand(0)) ||
                       IsS4DequantizedInput(*dot.operand(1));
  if ((rank != 2 && rank != 3) ||
      dot.operand(0)->shape().dimensions_size() != rank ||
      dot.operand(1)->shape().dimensions_size() != rank ||
      (!is_bf16 && !is_f16 && !is_f32 && !is_fp8) ||
      (is_int4 &&
       (rank != 2 || is_scaled_dot || dot.shape().dimensions(0) == 1 ||
        dot.shape().dimensions(1) == 1)) ||
      !IsSupportedContractionInput(*dot.operand(0)) ||
      !IsSupportedContractionInput(*dot.operand(1)) ||
      (dot.shape().element_type() != F32 &&
       !(is_bf16 && dot.shape().element_type() == BF16) &&
       !(is_f16 && dot.shape().element_type() == F16) &&
       !(is_fp8 && dot.shape().element_type() == BF16))) {
    return false;
  }
  const DotDimensionNumbers& dims = dot.dot_dimension_numbers();
  if (dims.lhs_contracting_dimensions_size() != 1 ||
      dims.rhs_contracting_dimensions_size() != 1) {
    return false;
  }
  if (is_scaled_dot) {
    const int64_t m = dot.shape().dimensions(rank - 2);
    const int64_t n = dot.shape().dimensions(rank - 1);
    const bool uniform_scale = IsUniformScale(*dot.operand(2)) &&
                               IsUniformScale(*dot.operand(3));
    if (is_f32 || (rank == 3 && !IsBatchedContraction(dot)) ||
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
       (lhs_concat.has_value() && lhs_concat->dimension != 0) ||
       (rhs_concat.has_value() &&
        rhs_concat->dimension != 1 - dims.rhs_contracting_dimensions(0)))) {
    return false;
  }
  if (rank == 2) {
    return dims.lhs_batch_dimensions().empty() &&
           dims.rhs_batch_dimensions().empty() &&
           dims.lhs_contracting_dimensions(0) == 1 &&
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

bool IsSupportedBroadcastInput(const HloInstruction& broadcast,
                               const HloInstruction& operation) {
  if (broadcast.opcode() != HloOpcode::kBroadcast ||
      broadcast.operand_count() != 1 || broadcast.dimensions().size() > 1 ||
      !ShapeUtil::SameDimensions(broadcast.shape(), operation.shape())) {
    return false;
  }
  const HloInstruction* input = broadcast.operand(0);
  const int64_t input_rank = input->shape().dimensions_size();
  const int64_t output_rank = operation.shape().dimensions_size();
  const bool scalar = input_rank == 0 && broadcast.dimensions().empty();
  const bool vector =
      input_rank == 1 && broadcast.dimensions().size() == 1 &&
      output_rank >= 2 &&
      (broadcast.dimensions(0) == output_rank - 2 ||
       broadcast.dimensions(0) == output_rank - 1 ||
       (output_rank == 3 && broadcast.dimensions(0) == 0)) &&
      input->shape().dimensions(0) ==
          operation.shape().dimensions(broadcast.dimensions(0));
  const bool supported_value =
      input->opcode() == HloOpcode::kParameter ||
      (scalar && input->opcode() == HloOpcode::kConstant &&
       input->literal().GetAsDouble({}).has_value());
  return supported_value && (scalar || vector) &&
         input->shape().element_type() == operation.shape().element_type();
}

bool IsSupportedBinaryEpilogue(HloOpcode opcode) {
  return opcode == HloOpcode::kAdd || opcode == HloOpcode::kMultiply ||
         opcode == HloOpcode::kSubtract || opcode == HloOpcode::kDivide ||
         opcode == HloOpcode::kMaximum || opcode == HloOpcode::kMinimum;
}

bool IsSupportedEpilogue(const HloInstruction& instr,
                         const HloInstruction& dot) {
  const HloInstruction* root =
      instr.fused_instructions_computation()->root_instruction();
  if (root == &dot) {
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
        (IsNarrowingConvert(*value) && value->operand(0) == &dot)) {
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
         (IsNarrowingConvert(*value) && value->operand(0) == &dot);
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
    FlyGemmConfig::MfmaAtom mfma_atom =
        FlyGemmConfig::FLY_MFMA_16X16X16,
    bool split_k = false) {
  return MakeConfig(block_m, block_n, /*block_k=*/32, num_warps,
                    mfma_atom,
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
  const bool is_f16 =
      dot->operand(0)->shape().element_type() == F16 &&
      dot->operand(1)->shape().element_type() == F16;
  const bool is_f32 =
      dot->operand(0)->shape().element_type() == F32 &&
      dot->operand(1)->shape().element_type() == F32;
  const bool is_fp8 = IsFnuzFp8(dot->operand(0)->shape().element_type());
  const bool is_int4 = IsS4DequantizedInput(*dot->operand(0)) ||
                       IsS4DequantizedInput(*dot->operand(1));
  const bool rank_three = dot->shape().dimensions_size() == 3;
  const bool global_split_k = IsGlobalSplitKContraction(*dot);
  const int64_t m = dot->shape().dimensions(rank_three ? 1 : 0);
  const int64_t n = dot->shape().dimensions(rank_three ? 2 : 1);
  const int64_t k = dot->operand(0)->shape().dimensions(
      dot->dot_dimension_numbers().lhs_contracting_dimensions(0));
  const int64_t lhs_contracting_dimension =
      dot->dot_dimension_numbers().lhs_contracting_dimensions(0);
  const int64_t rhs_contracting_dimension =
      dot->dot_dimension_numbers().rhs_contracting_dimensions(0);
  const bool is_scaled_dot = dot->opcode() == HloOpcode::kScaledDot;
  const bool is_block_scaled_dot =
      is_scaled_dot && (!IsUniformScale(*dot->operand(2)) ||
                        !IsUniformScale(*dot->operand(3)));
  const bool lhs_k_contiguous =
      dot->operand(0)->shape().has_layout() &&
      dot->operand(0)->shape().layout().minor_to_major(0) ==
          lhs_contracting_dimension;
  const bool rhs_k_contiguous =
      dot->operand(1)->shape().has_layout() &&
      dot->operand(1)->shape().layout().minor_to_major(0) ==
          rhs_contracting_dimension;
  const std::optional<ConcatInputInfo> lhs_concat =
      FindSupportedConcatInput(*dot->operand(0));
  const std::optional<ConcatInputInfo> rhs_concat =
      FindSupportedConcatInput(*dot->operand(1));
  const bool supports_masked_k_tail =
      !is_fp8 && !is_int4 && !is_block_scaled_dot && !global_split_k;
  if (k % (is_fp8 ? 32 : 16) != 0 && !supports_masked_k_tail) {
    return configs;
  }

  constexpr std::array<int64_t, 5> kBlockSizes = {16, 32, 64, 128, 256};
  const bool masked_k_tail = k % 16 != 0;
  auto supports_gemm_k_tile = [&](int64_t block_k) {
    if (!masked_k_tail) {
      return block_k <= k && k % block_k == 0;
    }
    // Admit every complete power-of-two stage and the first stage larger than
    // K. The latter avoids forcing very short contractions into K16 without
    // also offering successively larger mostly-zero stages.
    return block_k == 16 || block_k <= k || block_k / 2 < k;
  };
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
          if (output_block >= 64 && output_block % 64 == 0 &&
              num_warps > 1) {
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
            if (k_vector_width > 1 &&
                k % (64 * k_vector_width) != 0) {
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
        lhs_k_contiguous && n % 16 == 0 && k >= 512 &&
        k % 256 == 0 &&
        dot->operand(0)->shape().element_type() ==
            dot->operand(1)->shape().element_type()) {
      // Match Triton's successful row-vector tile while using Fly's cheaper
      // local split: two or four waves cooperate on one 16x16 MFMA atom and
      // reduce their K partitions through LDS.
      for (int64_t num_warps : {2, 4}) {
        for (int32_t waves_per_eu : {0, 2, 4}) {
          configs.insert(configs.begin(), MakeConfig(
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
            if (k_vector_width > 1 &&
                k % (64 * k_vector_width) != 0) {
              continue;
            }
            configs.push_back(MakeGemvConfig(
                output_block, /*block_n=*/16, num_warps, outputs_per_wave,
                k_vector_width,
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
              std::unique_ptr<BackendConfig> config = MakeConfig(
                  block_m, block_n, block_k, num_warps, atom,
                  /*prefetch_rhs=*/false, /*stage_output=*/false,
                  /*waves_per_eu=*/0, schedule);
              config->mutable_fly()->set_dequantize_block_scales(
                  dequantize_block_scales);
              configs.push_back(std::move(config));
            }
          };
          if (num_warps <= mfma16_tiles &&
              mfma16_tiles % num_warps == 0) {
            add_variants(is_fp8
                             ? FlyGemmConfig::FLY_MFMA_16X16X32_FP8
                             : FlyGemmConfig::FLY_MFMA_16X16X16);
            if (is_fp8) {
              add_variants(FlyGemmConfig::FLY_MFMA_16X16X16,
                           /*dequantize_block_scales=*/true);
            }
          }
          if (block_m % 32 == 0 && block_n % 32 == 0 &&
              num_warps <= mfma32_tiles &&
              mfma32_tiles % num_warps == 0) {
            add_variants(is_fp8
                             ? FlyGemmConfig::FLY_MFMA_32X32X16_FP8
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
      if (block_m > m || m % block_m != 0) {
        continue;
      }
      for (int64_t block_n : kBlockSizes) {
        if (block_n > n || n % block_n != 0) {
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
              if ((block_m * block_k / 2) % block_threads == 0 &&
                  (block_n * block_k / 2) % block_threads == 0 &&
                  staged_lds_bytes <= 64 * 1024) {
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
    if (!rhs_k_contiguous && m % 128 == 0 && n % 128 == 0 && k % 64 == 0) {
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
        output_element_type == BF16 &&
        m % block_m == 0 && n % block_n == 0 &&
        block_m * block_n >= kMinStagedOutputElements &&
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
    for (int64_t block_m : kBlockSizes) {
      // The generic emitter predicates its final M/N tile. Include the first
      // power-of-two tile larger than an odd dimension so small tails do not
      // force an extra workgroup, while avoiding successively larger tiles
      // whose wasted MFMA work cannot be competitive.
      if (block_m > m && block_m / 2 >= m) {
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
          if (!supports_gemm_k_tile(block_k) ||
              lhs_lds_bytes > kMaxLdsBytes) {
            continue;
          }
          for (int64_t num_warps : {1, 2, 4, 8}) {
            if (num_warps <= wave_tiles && wave_tiles % num_warps == 0) {
              add_gemm_configs(block_m, block_n, block_k, num_warps,
                               FlyGemmConfig::FLY_MFMA_16X16X16);
            }
            const int64_t mfma32_tiles =
                (block_m / 32) * (block_n / 32);
            if (block_m % 32 == 0 && block_n % 32 == 0 &&
                num_warps <= mfma32_tiles &&
                mfma32_tiles % num_warps == 0) {
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
    if (!rhs_k_contiguous || block_m > m || block_n > n ||
        !supports_gemm_k_tile(block_k) ||
        staged_lds_bytes > kMaxLdsBytes) {
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
              std::unique_ptr<BackendConfig> config =
                  MakeConfig(block_m, block_n, block_k, num_warps,
                             FlyGemmConfig::FLY_MFMA_16X16X16,
                             /*prefetch_rhs=*/false, stage_output, waves_per_eu,
                             /*schedule_instructions=*/schedule,
                             /*stage_rhs=*/true);
              config->mutable_fly()->set_workgroup_mapping_n(
                  workgroup_mapping_n);
              configs.push_back(std::move(config));
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
    if (!rhs_k_contiguous || block_m > m || block_n > n ||
        !supports_gemm_k_tile(block_k) ||
        staged_lds_bytes > kMaxLdsBytes) {
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
    const bool default_operand_precision =
        absl::c_all_of(precision.operand_precision(), [](int value) {
          return value == PrecisionConfig::DEFAULT;
        });
    const bool allow_xf32 =
        default_operand_precision &&
        (precision.algorithm() == PrecisionConfig::ALG_UNSET ||
         precision.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32);
    for (int64_t block_m : kF32BlockSizes) {
      if (block_m > m && block_m / 2 >= m) {
        continue;
      }
      for (int64_t block_n : kF32BlockSizes) {
        if (block_n > n && block_n / 2 >= n) {
          continue;
        }
        const int64_t mfma16_tiles = (block_m / 16) * (block_n / 16);
        const int64_t mfma32_tiles = (block_m / 32) * (block_n / 32);
        for (int64_t block_k : kF32KBlockSizes) {
          // The generic kernel ping-pongs two block_m x block_k FP32 LHS
          // stages and moves four FP32 values per 16-byte VMEM operation.
          const int64_t lhs_lds_bytes =
              2 * block_m * block_k * sizeof(float);
          if ((block_k > k && block_k / 2 >= k) ||
              lhs_lds_bytes > 64 * 1024) {
            continue;
          }
          for (int64_t num_warps : {1, 2, 4, 8}) {
            const int64_t block_threads = num_warps * 64;
            if ((block_m * block_k / 4) % block_threads != 0) {
              continue;
            }
            if (num_warps <= mfma16_tiles &&
                mfma16_tiles % num_warps == 0) {
              configs.push_back(MakeConfig(
                  block_m, block_n, block_k, num_warps,
                  FlyGemmConfig::FLY_MFMA_16X16X4_F32));
              const int64_t rhs_lds_bytes =
                  2 * block_n * block_k * sizeof(float);
              if (rhs_k_contiguous && block_k >= 32 &&
                  block_k % 32 == 0 &&
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
            if (block_m % 32 == 0 && block_n % 32 == 0 &&
                num_warps <= mfma32_tiles &&
                mfma32_tiles % num_warps == 0) {
              configs.push_back(MakeConfig(
                  block_m, block_n, block_k, num_warps,
                  FlyGemmConfig::FLY_MFMA_32X32X2_F32));
              if (!allow_xf32) {
                continue;
              }
              configs.push_back(MakeConfig(
                  block_m, block_n, block_k, num_warps,
                  FlyGemmConfig::FLY_MFMA_32X32X4_XF32));
              const int64_t rhs_lds_bytes =
                  2 * block_n * block_k * sizeof(float);
              if (rhs_k_contiguous && block_k >= 32 &&
                  block_k % 32 == 0 &&
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

    if (allow_xf32 && rhs_k_contiguous && k >= 64 && k % 32 == 0) {
      auto add_single_buffer_xf32 = [&](int64_t block_m, int64_t block_n,
                                        int64_t block_k,
                                        int64_t num_warps) {
        if (block_m > m || block_n > n || m % block_m != 0 ||
            n % block_n != 0 || k % block_k != 0) {
          return;
        }
        for (int32_t waves_per_eu : {0, 2, 4}) {
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                block_m, block_n, block_k, num_warps,
                FlyGemmConfig::FLY_MFMA_32X32X4_XF32,
                /*prefetch_rhs=*/false, /*stage_output=*/false, waves_per_eu,
                schedule, /*stage_rhs=*/true, /*async_lhs=*/false,
                /*preload_lds_fragments=*/true,
                /*single_buffer_lds=*/true));
          }
        }
      };

      // Mirror Triton's two high-throughput FP32 modulo-pipeline tiles. One
      // physical A+B LDS stage uses 48 KiB for 256x128xK32 and the full 64
      // KiB for 128x128xK64, while the next tile is held in VGPRs under the
      // current tile's XF32 MFMAs.
      add_single_buffer_xf32(/*block_m=*/256, /*block_n=*/128,
                             /*block_k=*/32, /*num_warps=*/8);
      add_single_buffer_xf32(/*block_m=*/128, /*block_n=*/128,
                             /*block_k=*/64, /*num_warps=*/4);
    }
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
          config->mutable_fly()->set_workgroup_mapping_n(
              workgroup_mapping_n);
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
    // Global split-K partials are FP32 and XLA owns the final reduction. Only
    // offer the A+B LDS pipelines whose global accesses are explicitly
    // batch-aware; generic rank-2 tensor-indexing candidates remain excluded.
    if (k >= 512) {
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
  if (rank_three && !rhs_k_contiguous && m >= 64 && n >= 256 &&
      k >= 1024 && supports_gemm_k_tile(/*block_k=*/32)) {
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
  // Row-major RHS needs a K/N register transpose before LDS. Pair adjacent K
  // rows, then overlap the single-buffer refill with the second half of the
  // MFMA tile. Four K128 stages are already sufficient to amortize this
  // pipeline in batched K512 GEMMs.
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
    if (output_element_type == BF16 && !rhs_k_contiguous &&
        m % 128 == 0 && n % 128 == 0 && k % 64 == 0) {
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
    if (output_element_type == BF16 && !rhs_k_contiguous &&
        m % 128 == 0 && n % 256 == 0 && k % 32 == 0) {
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
  if (lhs_concat.has_value() || rhs_concat.has_value()) {
    configs.erase(
        std::remove_if(
            configs.begin(), configs.end(),
            [&](const std::unique_ptr<BackendConfig>& config) {
              const FlyGemmConfig& fly = config->fly();
              return (lhs_concat.has_value() &&
                      lhs_concat->fragment_size % fly.block_m() != 0) ||
                     (rhs_concat.has_value() &&
                      rhs_concat->fragment_size % fly.block_n() != 0);
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
  // Scalar/vector GEMV configs use the dedicated Fly GEMV emitter.  The
  // cooperative MFMA candidate is structurally a tiled GEMM with a masked
  // M-tail and must therefore use the xTile GEMM emitter.
  fusion_config->set_kind(is_gemv && !fly_config.local_split_k()
                              ? kFlyGemvFusionKind
                              : kFlyGemmFusionKind);
  *fusion_config->mutable_fly_gemm_config() = fly_config;

  BlockLevelFusionConfig* block_config =
      fusion_config->mutable_block_level_fusion_config();
  block_config->Clear();
  Tile* output_tile = block_config->add_output_tiles();
  if (output_rank == 3) {
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
  bool is_generic_triton_fusion = false;
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    auto gpu_config = instr.backend_config<GpuBackendConfig>();
    if (!gpu_config.ok() ||
        gpu_config->fusion_backend_config().kind() != kTritonFusionKind) {
      return false;
    }
    is_generic_triton_fusion = true;
  }
  HloFusionAnalysis analysis =
      HloFusionAnalysis::Create(instr, target_config().device_description);
  if (flydsl::IsFlySoftmaxFusion(analysis)) {
    return true;
  }
  if (flydsl::IsFlyXTileTransposeFusion(analysis)) {
    return true;
  }
  // Generic __triton fusions have already been marked custom, so their HLO
  // fusion analysis reports kTriton instead of the underlying loop, reduction,
  // or transpose emitter kind. Fly dispatches those structures directly in
  // GetFusionEmitter and can autotune them without undoing the fusion.
  if (is_generic_triton_fusion) {
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
  if (flydsl::IsFlySoftmaxFusion(analysis)) {
    const Shape& shape = analysis.first_result_shape();
    const int64_t columns = shape.dimensions(1);
    for (int64_t num_warps : {1, 2, 4, 8, 16}) {
      const int64_t threads = num_warps * 64;
      const int64_t values_per_thread = (columns + threads - 1) / threads;
      const int64_t rounded_columns = ((columns + 63) / 64) * 64;
      if (threads > rounded_columns || values_per_thread > 64) {
        continue;
      }
      auto config = std::make_unique<BackendConfig>();
      BlockLevelFusionConfig* block = config->mutable_block_level();
      Tile* tile = block->add_output_tiles();
      tile->add_sizes(1);
      tile->add_sizes(columns);
      block->set_num_warps(num_warps);
      block->set_num_ctas(1);
      block->set_num_stages(1);
      configs.push_back(std::move(config));
    }
    return configs;
  }
  if (flydsl::IsFlyXTileTransposeFusion(analysis)) {
    const Shape& input_shape =
        analysis.fusion_root(0).instruction().operand(0)->shape();
    for (auto [tile_size, num_warps] :
         {std::pair<int64_t, int64_t>{32, 1}, {64, 4}, {128, 16}}) {
      if (input_shape.dimensions(0) % tile_size != 0 ||
          input_shape.dimensions(1) % tile_size != 0) {
        continue;
      }
      auto config = std::make_unique<BackendConfig>();
      BlockLevelFusionConfig* block = config->mutable_block_level();
      Tile* tile = block->add_output_tiles();
      tile->add_sizes(tile_size);
      tile->add_sizes(tile_size);
      block->set_num_warps(num_warps);
      block->set_num_ctas(1);
      block->set_num_stages(1);
      configs.push_back(std::move(config));
    }
    return configs;
  }
  const auto& heroes = analysis.fusion_heroes();
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
  if (reduction_hero != nullptr) {
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

  const int64_t elements = ShapeUtil::ElementsIn(analysis.first_result_shape());
  const int64_t native_unroll = ComputeLoopFusionConfig(analysis);

  auto add_config = [&](int64_t unroll, int64_t num_warps) {
    if (unroll <= 0 || unroll > elements) {
      return;
    }
    for (const auto& existing : configs) {
      if (existing->block_level().output_tiles(0).sizes(0) == unroll &&
          existing->block_level().num_warps() == num_warps) {
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
    configs.push_back(std::move(config));
  };

  add_config(native_unroll, 4);
  for (int64_t num_warps : {1, 2, 4, 8}) {
    for (int64_t unroll : {1, 2, 4, 8}) {
      add_config(unroll, num_warps);
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
      config.block_level().output_tiles_size() != 1 ||
      config.block_level().output_tiles(0).sizes().empty()) {
    return absl::InvalidArgumentError(
        "Expected a non-empty Fly fusion tile configuration.");
  }
  for (int64_t size : config.block_level().output_tiles(0).sizes()) {
    if (size <= 0) {
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
