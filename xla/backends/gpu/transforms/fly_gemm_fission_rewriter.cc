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

#include "xla/backends/gpu/transforms/fly_gemm_fission_rewriter.h"

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

absl::StatusOr<bool> FlyGemmFissionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kDot) {
        continue;
      }
      HloInstruction* fusion = computation->AddInstruction(
          HloInstruction::CreateFusion(
              instruction->shape(), HloInstruction::FusionKind::kCustom,
              instruction),
          /*new_name=*/"fly_fission_gemm");
      fusion->set_metadata(instruction->metadata());
      ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
      gpu_config.mutable_fusion_backend_config()->set_kind(kFlyGemmFusionKind);
      RETURN_IF_ERROR(fusion->set_backend_config(std::move(gpu_config)));
      RETURN_IF_ERROR(computation->ReplaceInstruction(instruction, fusion));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla::gpu
