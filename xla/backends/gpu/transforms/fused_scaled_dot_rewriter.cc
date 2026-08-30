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

#include "xla/backends/gpu/transforms/fused_scaled_dot_rewriter.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> FusedScaledDotRewriter::RewriteComputation(
    HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kScaledDot) {
      continue;
    }
    HloScaledDotInstruction* dot = Cast<HloScaledDotInstruction>(instruction);
    HloInstruction* fused = nullptr;
    for (const FusedScaledDotArm& arm : arms_) {
      ABSL_ASSIGN_OR_RETURN(fused, arm(computation, dot));
      if (fused != nullptr) break;
    }
    if (fused == nullptr) {
      continue;  // No arm claims it; ScaledDotRewriter expands it behind us.
    }
    ABSL_RETURN_IF_ERROR(dot->ReplaceAllUsesWith(fused));
    ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(dot));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> FusedScaledDotRewriter::RunImpl(
    HloModule* module, const absl::flat_hash_set<absl::string_view>&) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    ABSL_ASSIGN_OR_RETURN(bool result, RewriteComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
