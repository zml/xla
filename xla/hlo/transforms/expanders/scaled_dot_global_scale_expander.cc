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

#include "xla/hlo/transforms/expanders/scaled_dot_global_scale_expander.h"

#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

bool ScaledDotGlobalScaleExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kScaledDot &&
         instruction->operand_count() >
             HloScaledDotInstruction::kOperands;
}

absl::StatusOr<HloInstruction*> ScaledDotGlobalScaleExpander::ExpandInstruction(
    HloInstruction* instruction) {
  HloComputation* computation = instruction->parent();
  auto* dot = Cast<HloScaledDotInstruction>(instruction);

  // The four-operand op, with the trailing NVFP4 global scales dropped. Built
  // directly rather than via CloneWithNewOperands, which deliberately CHECKs
  // that a clone preserves the arity -- narrowing is legal exactly here, where
  // the dropped operands are replaced by the epilogue below, and nowhere else.
  HloInstruction* narrowed =
      computation->AddInstruction(HloInstruction::CreateScaledDot(
          dot->shape(), dot->mutable_operand(0), dot->mutable_operand(1),
          dot->mutable_operand(2), dot->mutable_operand(3),
          dot->dot_dimension_numbers(), dot->precision_config()));
  dot->SetupDerivedInstruction(narrowed);

  // Weight global scale as an explicit epilogue. The compressed-tensors
  // convention stores it as (448*6)/amax(W), a large reciprocal, so this is a
  // divide. Done in F32: the dot result is ~weight_global_scale larger than the
  // true value, and rounding the scalar itself to a narrow type would impose a
  // uniform relative error on the whole tensor.
  const Shape f32_shape =
      ShapeUtil::ChangeElementType(dot->shape(), PrimitiveType::F32);
  HloInstruction* wgs =
      dot->mutable_operand(HloScaledDotInstruction::kOperandsWithGlobals - 1);
  if (wgs->shape().element_type() != PrimitiveType::F32) {
    wgs = computation->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::ChangeElementType(wgs->shape(), PrimitiveType::F32), wgs));
  }
  HloInstruction* wgs_bcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(f32_shape, wgs, {}));
  HloInstruction* dot_f32 =
      dot->shape().element_type() == PrimitiveType::F32
          ? narrowed
          : computation->AddInstruction(
                HloInstruction::CreateConvert(f32_shape, narrowed));
  HloInstruction* scaled =
      computation->AddInstruction(HloInstruction::CreateBinary(
          f32_shape, HloOpcode::kDivide, dot_f32, wgs_bcast));

  if (dot->shape().element_type() == PrimitiveType::F32) {
    return scaled;
  }
  return computation->AddInstruction(
      HloInstruction::CreateConvert(dot->shape(), scaled));
}

}  // namespace xla
