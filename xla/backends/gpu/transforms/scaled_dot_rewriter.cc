/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/metal_custom_calls.h"
#include "xla/stream_executor/device_description.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// Returns the type to use for the scaled dot operation operands.
// If both types are smaller than 16 bits, use BF16.
// If both types are the same, use the same type.
// Otherwise, use the bigger type.
PrimitiveType GetTargetType(PrimitiveType type_one, PrimitiveType type_two) {
  constexpr int kMinBitWidth = 16;
  if (primitive_util::BitWidth(type_one) < kMinBitWidth &&
      primitive_util::BitWidth(type_two) < kMinBitWidth) {
    return PrimitiveType::BF16;
  }
  if (type_one == type_two) {
    return type_one;
  }
  if (primitive_util::BitWidth(type_one) ==
          primitive_util::BitWidth(type_two) &&
      type_one != type_two) {
    return PrimitiveType::F32;
  }
  return (primitive_util::BitWidth(type_one) >
          primitive_util::BitWidth(type_two))
             ? type_one
             : type_two;
}

HloInstruction* Convert(HloInstruction* instr, PrimitiveType target_type) {
  if (instr->shape().element_type() == target_type) {
    return instr;
  }
  HloComputation* computation = instr->parent();
  Shape shape = ShapeUtil::ChangeElementType(instr->shape(), target_type);
  return computation->AddInstruction(
      HloInstruction::CreateConvert(shape, instr));
}

// Returns a pair of instructions that upscale both operands to the same type.
std::pair<HloInstruction*, HloInstruction*> UpscaleBoth(
    HloInstruction* first, HloInstruction* second) {
  PrimitiveType target_type = GetTargetType(first->shape().element_type(),
                                            second->shape().element_type());
  first = Convert(first, target_type);
  second = Convert(second, target_type);
  return std::make_pair(first, second);
}

absl::Status CheckOperandAndScaleShapes(absl::string_view side,
                                        const HloInstruction* operand,
                                        const HloInstruction* scale) {
  if (operand->shape().dimensions().size() !=
      scale->shape().dimensions().size()) {
    return InvalidArgument(
        "%s: operand and scale must have the same rank: %d vs %d", side,
        operand->shape().dimensions().size(),
        scale->shape().dimensions().size());
  }

  for (int i = 0; i < operand->shape().dimensions().size(); ++i) {
    if (operand->shape().dimensions(i) % scale->shape().dimensions(i)) {
      return InvalidArgument(
          "%s: operand and scale dimensions must match or scale dimension must "
          "be divider of operand dimension: %d vs %d at index %d",
          side, operand->shape().dimensions(i), scale->shape().dimensions(i),
          i);
    }
  }
  return absl::OkStatus();
}

HloInstruction* BroadcastAndReshape(HloInstruction* scale,
                                    const Shape& operand_shape,
                                    HloComputation* computation) {
  Shape scale_shape = scale->shape();
  std::vector<int64_t> broadcast_dims;
  std::vector<int64_t> shape_dims;

  for (int shape_index = 0, i = 0; i < operand_shape.dimensions().size();
       ++shape_index, ++i) {
    broadcast_dims.push_back(shape_index);
    shape_dims.push_back(scale_shape.dimensions(i));
    if (operand_shape.dimensions(i) != scale_shape.dimensions(i)) {
      ++shape_index;
      shape_dims.push_back(operand_shape.dimensions(i) /
                           scale_shape.dimensions(i));
    }
  }
  Shape new_scales_shape(scale_shape.element_type(), shape_dims);
  LayoutUtil::SetToDefaultLayout(&new_scales_shape);
  HloInstruction* new_scales = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_scales_shape, scale, broadcast_dims));
  Shape reshaped_scales_shape(scale_shape.element_type(),
                              operand_shape.dimensions());
  LayoutUtil::SetToDefaultLayout(&reshaped_scales_shape);
  return computation->AddInstruction(
      HloInstruction::CreateReshape(reshaped_scales_shape, new_scales));
}

// Dequantizes the dot operation operand at the given index.
// The scale operand is expected to be broadcastable to the operand shape.
absl::StatusOr<HloInstruction*> Dequantize(HloInstruction* dot,
                                           int operand_index, int scale_index,
                                           absl::string_view side) {
  HloComputation* computation = dot->parent();
  HloInstruction* operand = dot->mutable_operand(operand_index);
  HloInstruction* scale = dot->mutable_operand(scale_index);
  if (scale->shape().dimensions().empty()) {
    // If the scale is a scalar, we don't need to do anything.
    return operand;
  }
  std::tie(operand, scale) = UpscaleBoth(operand, scale);
  RETURN_IF_ERROR(CheckOperandAndScaleShapes(side, operand, scale));
  HloInstruction* broadcasted_scale =
      BroadcastAndReshape(scale, operand->shape(), computation);
  HloInstruction* dequantized =
      computation->AddInstruction(HloInstruction::CreateBinary(
          operand->shape(), HloOpcode::kMultiply, operand, broadcasted_scale));
  return dequantized;
}

// === Weight-only fused scaled matmul: shared layout predicates + platform emit ===

// True if `s` is the identity (all-ones) scale of an unquantized activation:
// rank-0, or a bf16 all-ones dense constant with every dim == 1 (ZML weight-only
// form). Used by backends that fuse only when the act side has no real scale.
bool IsIdentityScale(const HloInstruction* s) {
  if (s->shape().element_type() != BF16) return false;
  for (int64_t d : s->shape().dimensions()) {
    if (d != 1) return false;
  }
  if (s->shape().dimensions().empty()) return true;  // rank-0 pass-through
  return s->opcode() == HloOpcode::kConstant && s->literal().IsAllFloat(1.0);
}

// MX group-32: f8e4m3fn/f4e2m1 weight [N,K] + f8e8m0 scale [N,K/32], K minor.
bool IsMxGroup32Weight(const HloInstruction* w, const HloInstruction* s,
                       int64_t c) {
  const PrimitiveType wt = w->shape().element_type();
  if (wt != F8E4M3FN && wt != F4E2M1FN) return false;
  if (s->shape().element_type() != F8E8M0FNU) return false;
  if (w->shape().dimensions().size() != 2 ||
      s->shape().dimensions().size() != 2) {
    return false;
  }
  if (c != 1) return false;  // K minor
  const int64_t n = w->shape().dimensions(0), k = w->shape().dimensions(1);
  if (s->shape().dimensions(0) != n) return false;
  const int64_t sc = s->shape().dimensions(1);
  return sc != 0 && k == sc * 32;
}

// 128x128 block scales: f8e4m3fn weight [N,K] + bf16 scale [N/128,K/128]
// (vLLM weight_scale_inv). N and K multiples of 128.
bool IsBlock128Bf16Weight(const HloInstruction* w, const HloInstruction* s,
                          int64_t c) {
  if (w->shape().element_type() != F8E4M3FN) return false;
  if (s->shape().element_type() != BF16) return false;
  if (w->shape().dimensions().size() != 2 ||
      s->shape().dimensions().size() != 2) {
    return false;
  }
  if (c != 1) return false;  // K minor
  const int64_t n = w->shape().dimensions(0), k = w->shape().dimensions(1);
  if (n % 128 != 0 || k % 128 != 0) return false;
  return s->shape().dimensions(0) == n / 128 &&
         s->shape().dimensions(1) == k / 128;
}

// Per-channel: f8e4m3fn weight [N,K] + bf16 scale [N,1], K minor.
bool IsPerChannelWeight(const HloInstruction* w, const HloInstruction* s,
                        int64_t c) {
  if (w->shape().element_type() != F8E4M3FN) return false;
  if (s->shape().element_type() != BF16) return false;
  if (w->shape().dimensions().size() != 2 ||
      s->shape().dimensions().size() != 2) {
    return false;
  }
  if (c != 1) return false;
  return s->shape().dimensions(0) == w->shape().dimensions(0) &&
         s->shape().dimensions(1) == 1;
}

// NVFP4: f4e2m1 weight [N,K] + e4m3 group-16 scale [N,K/16], K minor.
// Caller typically pre-scales x by 1/global; kernel sees raw e4m3 scales.
bool IsNvfp4Weight(const HloInstruction* w, const HloInstruction* s,
                   int64_t c) {
  if (w->shape().element_type() != F4E2M1FN) return false;
  if (s->shape().element_type() != F8E4M3FN) return false;
  if (w->shape().dimensions().size() != 2 ||
      s->shape().dimensions().size() != 2) {
    return false;
  }
  if (c != 1) return false;  // K minor
  const int64_t n = w->shape().dimensions(0), k = w->shape().dimensions(1);
  if (s->shape().dimensions(0) != n) return false;
  const int64_t sk = s->shape().dimensions(1);
  return sk != 0 && k == sk * 16;
}

// Weight-only layout common to fused backends: bf16[M,K] x quant[N,K] -> bf16[M,N],
// K minor on both, identity lhs scale, one of the supported weight/scale layouts.
bool IsWeightOnlyFusableScaledDot(const HloScaledDotInstruction* dot) {
  const DotDimensionNumbers& dn = dot->dot_dimension_numbers();
  if (dn.lhs_batch_dimensions_size() != 0 ||
      dn.rhs_batch_dimensions_size() != 0 ||
      dn.lhs_contracting_dimensions_size() != 1 ||
      dn.rhs_contracting_dimensions_size() != 1) {
    return false;
  }
  if (dot->shape().element_type() != BF16) return false;
  const HloInstruction* x = dot->operand(0);
  const HloInstruction* w = dot->operand(1);
  const HloInstruction* xs = dot->operand(2);
  const HloInstruction* ws = dot->operand(3);
  if (x->shape().dimensions().size() != 2 ||
      x->shape().element_type() != BF16 ||
      dn.lhs_contracting_dimensions(0) != 1 || !IsIdentityScale(xs)) {
    return false;
  }
  const int64_t rhs_c = dn.rhs_contracting_dimensions(0);
  return IsMxGroup32Weight(w, ws, rhs_c) || IsBlock128Bf16Weight(w, ws, rhs_c) ||
         IsPerChannelWeight(w, ws, rhs_c) || IsNvfp4Weight(w, ws, rhs_c);
}

// Metal: emit zml$scaled_matmul {x, w, w_scale}; ThunkEmitter dispatches scheme.
absl::StatusOr<HloInstruction*> TryEmitMetalScaledMatmul(
    HloComputation* comp, HloScaledDotInstruction* dot) {
  if (!IsWeightOnlyFusableScaledDot(dot)) return nullptr;
  return comp->AddInstruction(HloInstruction::CreateCustomCall(
      dot->shape(),
      {dot->mutable_operand(0), dot->mutable_operand(1),
       dot->mutable_operand(3)},
      std::string(kMetalScaledMatmulCallTarget)));
}

// Backend switch: try a fused custom call for this platform. Returns nullptr to
// fall through to generic dequant+Dot. CUDA/ROCm typically keep kScaledDot for
// Triton (this pass is skipped when the Triton flag is on); fill arms as needed.
absl::StatusOr<HloInstruction*> TryFusedScaledMatmul(
    HloComputation* comp, HloScaledDotInstruction* dot,
    const se::GpuComputeCapability& gpu_version) {
  if (gpu_version.IsMetal()) {
    return TryEmitMetalScaledMatmul(comp, dot);
  }
  // CUDA / ROCm / other: no fused custom-call arm in this pass yet.
  return nullptr;
}
}  // namespace

absl::StatusOr<bool> ScaledDotRewriter::RewriteComputation(
    HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kScaledDot) {
      continue;
    }
    changed = true;
    HloScaledDotInstruction* dot = Cast<HloScaledDotInstruction>(instruction);
    ASSIGN_OR_RETURN(HloInstruction * fused,
                     TryFusedScaledMatmul(computation, dot, gpu_version_));
    if (fused != nullptr) {
      RETURN_IF_ERROR(dot->ReplaceAllUsesWith(fused));
      RETURN_IF_ERROR(computation->RemoveInstruction(dot));
      continue;  // fused; skip generic dequant lowering
    }
    ASSIGN_OR_RETURN(HloInstruction * lhs, Dequantize(dot, 0, 2, "LHS"));
    ASSIGN_OR_RETURN(HloInstruction * rhs, Dequantize(dot, 1, 3, "RHS"));

    std::tie(lhs, rhs) = UpscaleBoth(lhs, rhs);

    Shape dot_shape = dot->shape();
    dot_shape.set_element_type(GetTargetType(lhs->shape().element_type(),
                                             dot->shape().element_type()));

    RETURN_IF_ERROR(dot->ReplaceAllUsesWith(
        Convert(computation->AddInstruction(HloInstruction::CreateDot(
                    dot_shape, lhs, rhs, dot->dot_dimension_numbers(),
                    dot->precision_config())),
                dot->shape().element_type())));
    RETURN_IF_ERROR(computation->RemoveInstruction(dot));
  }
  return changed;
}

absl::StatusOr<bool> ScaledDotRewriter::RunImpl(
    HloModule* module, const absl::flat_hash_set<absl::string_view>&) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    ASSIGN_OR_RETURN(bool result, RewriteComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
