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

#include "xla/backends/gpu/transforms/flydsl_replacement_verifier.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/codegen/flydsl/fusion_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla::gpu {
namespace {

bool IsTritonFusionKind(absl::string_view kind) {
  return kind == kTritonFusionKind || kind == kTritonGemmFusionKind ||
         kind == kTritonNestedGemmFusionKind ||
         kind == kTritonCollectiveFusionKind;
}

bool IsFlyFusionKind(absl::string_view kind) {
  return kind == kFlyFusionKind || kind == kFlyGemmFusionKind ||
         kind == kFlyGemvFusionKind || kind == kFlyCollectiveFusionKind;
}

bool IsTritonCustomCall(const HloInstruction& instruction) {
  if (instruction.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const absl::string_view target = instruction.custom_call_target();
  return target == "__triton" || target == "__gpu$xla.gpu.triton";
}

}  // namespace

absl::StatusOr<bool> FlyDslReplacementVerifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& /*execution_threads*/) {
  if (!module->config().debug_options().xla_gpu_flydsl_replace_triton()) {
    return false;
  }

  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (IsTritonCustomCall(*instruction)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "FlyDSL replacement mode found Triton custom call '",
            instruction->custom_call_target(), "' on instruction ",
            instruction->name()));
      }
      if (instruction->opcode() != HloOpcode::kFusion) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(const GpuBackendConfig gpu_config,
                          instruction->backend_config<GpuBackendConfig>());
      const absl::string_view kind =
          gpu_config.fusion_backend_config().kind();
      if (IsTritonFusionKind(kind)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "FlyDSL replacement mode found Triton fusion kind '", kind,
            "' on instruction ", instruction->name()));
      }
      if (IsFlyFusionKind(kind) &&
          flydsl::ContainsUnsupportedCustomCall(*instruction)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "FlyDSL replacement mode found an unsupported custom call inside "
            "Fly fusion ",
            instruction->name()));
      }
      if (kind == kFlyFusionKind) {
        HloFusionAnalysis analysis =
            HloFusionAnalysis::Create(*instruction, device_description_);
        const flydsl::FlyFusionRoute route =
            flydsl::ClassifyFlyFusion(analysis);
        if (!flydsl::IsNativeFlyFusionRoute(route)) {
          return absl::FailedPreconditionError(absl::StrCat(
              "FlyDSL replacement mode found non-native Fly route '",
              flydsl::FlyFusionRouteName(route), "' on instruction ",
              instruction->name(), ":\n",
              Cast<const HloFusionInstruction>(instruction)
                  ->fused_instructions_computation()
                  ->ToString()));
        }
      }
    }
  }
  return false;
}

}  // namespace xla::gpu
