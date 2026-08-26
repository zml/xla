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

#include "xla/backends/gpu/autotuner/fission_backend.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace gpu {

namespace {

std::string FlyTileKey(const FlyGemmConfig& config) {
  return absl::StrCat(config.block_m(), ":", config.block_n(), ":",
                      config.block_k(), ":", config.num_warps());
}

std::string FlyConfigKey(const BackendConfig& config) {
  if (!config.has_fly()) {
    return config.SerializeAsString();
  }
  const FlyGemmConfig& fly = config.fly();
  return absl::StrCat(
      fly.block_m(), ":", fly.block_n(), ":", fly.block_k(), ":",
      fly.num_warps(), ":", static_cast<int>(fly.mfma_atom()), ":",
      fly.prefetch_rhs(), ":", fly.stage_output(), ":", fly.waves_per_eu(), ":",
      fly.schedule_instructions(), ":", fly.stage_rhs(), ":", fly.async_lhs(),
      ":", fly.preload_lds_fragments(), ":", fly.single_buffer_lds(), ":",
      fly.direct_to_vgpr(), ":", fly.rolling_refill(), ":", fly.local_split_k(),
      ":", fly.workgroup_mapping_n(), ":", fly.gemv_outputs_per_wave(), ":",
      fly.gemv_k_vector_width(), ":", fly.gemv_split_k(), ":",
      fly.dequantize_block_scales());
}

bool IsFlySpecificPipeline(const FlyGemmConfig& config) {
  return config.stage_rhs() || config.async_lhs() ||
         config.preload_lds_fragments() || config.single_buffer_lds() ||
         config.direct_to_vgpr() || config.rolling_refill() ||
         config.local_split_k() || config.workgroup_mapping_n() != 0 ||
         config.gemv_outputs_per_wave() != 0 ||
         config.gemv_k_vector_width() != 0 || config.gemv_split_k() ||
         config.dequantize_block_scales();
}

const absl::flat_hash_set<std::string>& Mi300DefaultFlyTileKeys() {
  // These are macro-tile priors, not Triton configurations. They cover the
  // useful MI300 workgroup geometries found by XLA's existing hardware search
  // while leaving Fly's MFMA atom and software pipeline choices independent.
  // Specialized Fly pipelines bypass this list entirely.
  static const auto* keys = new absl::flat_hash_set<std::string>({
      "16:16:256:4",  "16:128:32:4",  "32:8:16:2",    "32:16:128:2",
      "32:16:128:4",  "32:16:256:2",  "32:32:32:2",   "32:32:256:4",
      "32:64:64:4",   "64:8:128:2",   "64:32:16:2",   "64:32:32:2",
      "64:32:32:4",   "64:32:128:2",  "128:8:16:2",   "128:16:128:8",
      "128:32:16:4",  "128:32:32:4",  "128:64:128:8", "128:128:32:4",
      "128:128:64:4", "128:256:32:4", "128:256:64:8", "256:8:16:2",
      "256:8:32:2",   "256:128:32:4", "256:128:64:8", "256:256:32:8",
  });
  return *keys;
}

// Replaces the fusion instruction with the instructions from the fissioned
// computation.
absl::Status InlineFissionedComputation(HloInstruction* fusion_instr,
                                        HloComputation* fissioned_computation) {
  if (fusion_instr->opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError("Not a fusion instruction.");
  }
  HloModule* original_module = fusion_instr->GetModule();
  HloCloneContext clone_context(original_module);
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      cloned_instructions;
  HloComputation* parent_computation = fusion_instr->parent();

  for (HloInstruction* instruction_to_clone :
       fissioned_computation->MakeInstructionPostOrder()) {
    if (instruction_to_clone->opcode() == HloOpcode::kParameter) {
      cloned_instructions[instruction_to_clone] = fusion_instr->mutable_operand(
          instruction_to_clone->parameter_number());
      continue;
    }

    std::vector<HloInstruction*> new_operands;
    for (const HloInstruction* operand : instruction_to_clone->operands()) {
      new_operands.push_back(cloned_instructions.at(operand));
    }
    HloInstruction* new_instruction = parent_computation->AddInstruction(
        instruction_to_clone->CloneWithNewOperands(
            instruction_to_clone->shape(), new_operands, &clone_context));
    cloned_instructions[instruction_to_clone] = new_instruction;
  }
  HloInstruction* new_root =
      cloned_instructions.at(fissioned_computation->root_instruction());
  ASSIGN_OR_RETURN(bool replaced,
                   parent_computation->ReplaceInstruction(
                       fusion_instr, new_root, /*preserve_sharding=*/false,
                       /*relay_control_dependency=*/true));
  TF_RET_CHECK(replaced) << "Failed to inline fissioned computation for "
                         << fusion_instr->name();
  return absl::OkStatus();
}

}  // namespace

std::vector<std::unique_ptr<BackendConfig>> OptimizeFlyFissionConfigSet(
    std::vector<std::unique_ptr<BackendConfig>> configs,
    bool restrict_to_mi300_default_tiles) {
  const size_t original_size = configs.size();
  absl::flat_hash_set<std::string> seen;
  std::vector<std::unique_ptr<BackendConfig>> unique_configs;
  unique_configs.reserve(configs.size());
  for (std::unique_ptr<BackendConfig>& config : configs) {
    if (seen.insert(FlyConfigKey(*config)).second) {
      unique_configs.push_back(std::move(config));
    }
  }
  if (!restrict_to_mi300_default_tiles || unique_configs.empty()) {
    VLOG(2) << "Deduplicated Fly fission configs from " << original_size
            << " to " << unique_configs.size();
    return unique_configs;
  }

  // Output-transpose fusions can make staging mandatory. Preserve hinted
  // staged-output choices when no generic direct-output choice exists.
  bool has_generic_config = false;
  bool all_generic_configs_stage_output = true;
  for (const std::unique_ptr<BackendConfig>& config : unique_configs) {
    if (!config->has_fly() || IsFlySpecificPipeline(config->fly())) {
      continue;
    }
    has_generic_config = true;
    all_generic_configs_stage_output &= config->fly().stage_output();
  }
  all_generic_configs_stage_output &= has_generic_config;

  // FlyBackend's default is the last generated configuration. Keep a copy so
  // an unusual shape that misses every macro-tile prior remains supported.
  auto fallback = std::make_unique<BackendConfig>(*unique_configs.back());
  std::vector<std::unique_ptr<BackendConfig>> optimized_configs;
  optimized_configs.reserve(unique_configs.size());
  for (std::unique_ptr<BackendConfig>& config : unique_configs) {
    if (!config->has_fly()) {
      optimized_configs.push_back(std::move(config));
      continue;
    }
    const FlyGemmConfig& fly = config->fly();
    if (IsFlySpecificPipeline(fly)) {
      optimized_configs.push_back(std::move(config));
      continue;
    }
    const bool hinted_tile =
        Mi300DefaultFlyTileKeys().contains(FlyTileKey(fly));
    const bool useful_generic_variant =
        fly.waves_per_eu() == 0 &&
        (!fly.stage_output() || all_generic_configs_stage_output) &&
        !(fly.prefetch_rhs() && fly.schedule_instructions());
    if (hinted_tile && useful_generic_variant) {
      optimized_configs.push_back(std::move(config));
    }
  }
  if (optimized_configs.empty()) {
    optimized_configs.push_back(std::move(fallback));
  }
  VLOG(1) << "Restricted MI300 Fly fission configs from " << original_size
          << " to " << optimized_configs.size();
  return optimized_configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
FissionBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    VLOG(3) << "Instruction not supported by " << name() << ": "
            << instr.ToString();
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                   GetFissionedAndRewrittenModule(instr));
  absl::StatusOr<std::vector<HloInstruction*>> supported_instrs =
      FindSupportedInstructions(hlo_module.get());
  if (supported_instrs.status().code() == absl::StatusCode::kNotFound) {
    VLOG(3) << "No supported instructions found by " << name() << ": "
            << instr.ToString();
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  RETURN_IF_ERROR(supported_instrs.status());
  ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      codegen_backend_->GetSupportedConfigs(*(*supported_instrs)[0]));
  if (codegen_backend_->backend() != autotuner::Backend::FLY) {
    return configs;
  }
  const auto& compute_capability =
      target_config().device_description.gpu_compute_capability();
  const bool use_mi300_default_tiles =
      !debug_options().xla_gpu_exhaustive_tiling_search() &&
      compute_capability.IsRocm() &&
      compute_capability.rocm_compute_capability()->gfx9_mi300();
  return OptimizeFlyFissionConfigSet(std::move(configs),
                                     use_mi300_default_tiles);
}

absl::StatusOr<std::unique_ptr<BackendConfig>> FissionBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError("Not a fusion instruction.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                   GetFissionedAndRewrittenModule(instr));
  ASSIGN_OR_RETURN(std::vector<HloInstruction*> supported_instrs,
                   FindSupportedInstructions(hlo_module.get()));
  return codegen_backend_->GetDefaultConfig(*supported_instrs[0]);
}

absl::Status FissionBackend::RunPriorityFusion(HloModule* module) {
  HloCostAnalysis::Options priority_fusion_options;
  priority_fusion_options.count_multiple_input_accesses = true;
  PriorityFusion priority_fusion(
      /*thread_pool=*/nullptr, target_config().device_description, alias_info_,
      priority_fusion_options, mlir_context_);
  return priority_fusion.Run(module).status();
}

absl::StatusOr<std::unique_ptr<HloModule>> FissionBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      codegen_backend_->RunHloPasses(std::move(hlo_module), options));

  RETURN_IF_ERROR(RunPriorityFusion(module.get()));
  return module;
}

absl::Status FissionBackend::ApplyConfig(HloInstruction& instr,
                                         const BackendConfig& config) {
  HloModule* module = instr.GetModule();
  ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                   GetFissionedAndRewrittenModule(instr));
  ASSIGN_OR_RETURN(std::vector<HloInstruction*> supported_instrs,
                   FindSupportedInstructions(hlo_module.get()));

  for (size_t i = 0; i < supported_instrs.size(); ++i) {
    HloInstruction* supported_instr = supported_instrs[i];
    if (i > 0) {
      if (supported_instr->opcode() != supported_instrs[0]->opcode()) {
        return absl::InternalError(absl::StrCat(
            "FissionBackend expected isomorphic supported instructions, but "
            "found different opcodes: ",
            HloOpcodeString(supported_instrs[0]->opcode()), " vs ",
            HloOpcodeString(supported_instr->opcode())));
      }
      if (!ShapeUtil::Compatible(supported_instr->shape(),
                                 supported_instrs[0]->shape())) {
        return absl::InternalError(
            "FissionBackend expected isomorphic supported instructions with "
            "compatible shapes, but found incompatible shapes.");
      }
    }
    RETURN_IF_ERROR(codegen_backend_->ApplyConfig(*supported_instr, config));
  }

  // Given that the autotuner runs post fusion, we have to run priority fusion
  // again to fuse the epilogue and prologues.
  RETURN_IF_ERROR(RunPriorityFusion(hlo_module.get()));

  RETURN_IF_ERROR(
      InlineFissionedComputation(&instr, hlo_module->entry_computation()));
  return module->RemoveUnusedComputations();
}

bool FissionBackend::IsSupported(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kFusion;
}

absl::StatusOr<std::unique_ptr<HloModule>>
FissionBackend::GetFissionedAndRewrittenModule(
    const HloInstruction& fusion_instr) {
  const auto* fusion = Cast<HloFusionInstruction>(&fusion_instr);
  std::unique_ptr<HloModule> hlo_module =
      ExtractComputationIntoNewModule(*fusion->called_computation());
  // ExtractComputationIntoNewModule creates a new HloModule with a default
  // HloModuleConfig, whose DebugOptions are initialized to
  // DefaultDebugOptionsIgnoringFlags() — not the user's values. Propagate the
  // user-defined debug options before running any passes so that the rewriter
  // pipeline (e.g. GemmRewriter reads xla_gpu_gemm_rewrite_size_threshold) and
  // any subsequent PriorityFusion run observe the correct flag values.
  DebugOptions options = debug_options();
  AdjustDebugOptionsForAutotuning(options);
  hlo_module->mutable_config().set_debug_options(options);
  RETURN_IF_ERROR(rewriter_pipeline_->Run(hlo_module.get()).status());
  return hlo_module;
}

absl::StatusOr<std::vector<HloInstruction*>>
FissionBackend::FindSupportedInstructions(const HloModule* module) {
  std::vector<HloInstruction*> supported_instructions;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (codegen_backend_->IsSupported(*instruction)) {
        supported_instructions.push_back(instruction);
      }
    }
  }
  if (supported_instructions.empty()) {
    return absl::NotFoundError("No supported instructions found.");
  }
  return supported_instructions;
}

}  // namespace gpu

}  // namespace xla
