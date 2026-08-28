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

#include "xla/backends/gpu/autotuner/autotuner_main_util.h"

#include <utility>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/status_macros.h"

namespace xla::gpu {

absl::Status ClearPreexistingBlockLevelConfigs(HloModule& module) {
  for (HloComputation* computation : module.computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kFusion) {
        continue;
      }
      ASSIGN_OR_RETURN(GpuBackendConfig config,
                       instruction->backend_config<GpuBackendConfig>());
      FusionBackendConfig* fusion_config =
          config.mutable_fusion_backend_config();
      if (fusion_config->kind() != kTritonFusionKind ||
          !fusion_config->has_block_level_fusion_config()) {
        continue;
      }
      fusion_config->clear_block_level_fusion_config();
      RETURN_IF_ERROR(instruction->set_backend_config(std::move(config)));
    }
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
