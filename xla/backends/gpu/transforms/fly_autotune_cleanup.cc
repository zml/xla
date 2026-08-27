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

#include "xla/backends/gpu/transforms/fly_autotune_cleanup.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

bool HasFlyFusionKind(const HloInstruction& instruction,
                      absl::string_view expected_kind) {
  if (instruction.opcode() != HloOpcode::kFusion) {
    return false;
  }
  absl::StatusOr<GpuBackendConfig> config =
      instruction.backend_config<GpuBackendConfig>();
  return config.ok() &&
         config->fusion_backend_config().kind() == expected_kind;
}

bool IsFlyGemm(const HloInstruction& instruction) {
  return HasFlyFusionKind(instruction, kFlyGemmFusionKind) ||
         HasFlyFusionKind(instruction, kFlyGemvFusionKind);
}

bool IsParameter(const HloInstruction* instruction, int64_t number) {
  return instruction->opcode() == HloOpcode::kParameter &&
         instruction->parameter_number() == number;
}

bool IsExactContractingScalePrologue(const HloInstruction& instruction) {
  if (instruction.opcode() != HloOpcode::kFusion ||
      instruction.operand_count() != 2) {
    return false;
  }
  const HloInstruction* root = instruction.fused_expression_root();
  if (root->opcode() != HloOpcode::kConvert ||
      root->operand(0)->opcode() != HloOpcode::kMultiply) {
    return false;
  }
  const HloInstruction* multiply = root->operand(0);
  for (int64_t data_index : {0, 1}) {
    const HloInstruction* data = multiply->operand(data_index);
    const HloInstruction* scale = multiply->operand(1 - data_index);
    if (data->opcode() != HloOpcode::kConvert ||
        !IsParameter(data->operand(0), 0) ||
        scale->opcode() != HloOpcode::kConvert ||
        scale->operand(0)->opcode() != HloOpcode::kBroadcast) {
      continue;
    }
    const HloInstruction* broadcast = scale->operand(0);
    if (IsParameter(broadcast->operand(0), 1) &&
        broadcast->dimensions().size() == 1 &&
        broadcast->dimensions(0) == 1) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<bool> FoldScalePrologue(HloInstruction* gemm) {
  if (!IsFlyGemm(*gemm)) {
    return false;
  }
  for (HloInstruction* operand : gemm->mutable_operands()) {
    if (!IsExactContractingScalePrologue(*operand) ||
        operand->user_count() != 1) {
      continue;
    }
    auto* prologue = Cast<HloFusionInstruction>(operand);
    HloInstruction* view = prologue->mutable_operand(0);
    if (view->opcode() != HloOpcode::kBitcast || view->user_count() != 1) {
      continue;
    }
    HloInstruction* producer = view->mutable_operand(0);
    if (!HasFlyFusionKind(*producer, kFlyFusionKind) ||
        producer->user_count() != 1) {
      continue;
    }
    auto* producer_fusion = Cast<HloFusionInstruction>(producer);
    ASSIGN_OR_RETURN(GpuBackendConfig producer_config,
                     producer_fusion->backend_config<GpuBackendConfig>());

    prologue->FuseInstruction(view);
    TF_RET_CHECK(view->user_count() == 0);
    RETURN_IF_ERROR(view->parent()->RemoveInstruction(view));
    TF_RET_CHECK(absl::c_linear_search(prologue->operands(), producer));
    prologue->MergeFusionInstruction(producer_fusion);
    RETURN_IF_ERROR(prologue->set_backend_config(std::move(producer_config)));
    if (producer->user_count() == 0 && !producer->IsRoot()) {
      RETURN_IF_ERROR(producer->parent()->RemoveInstruction(producer));
    }
    return true;
  }
  return false;
}

absl::StatusOr<bool> FoldScalarMaterialization(HloInstruction* producer) {
  if (producer->opcode() != HloOpcode::kFusion ||
      producer->operand_count() != 0 ||
      !ShapeUtil::IsScalar(producer->shape()) ||
      producer->user_count() != 1) {
    return false;
  }
  HloInstruction* consumer = producer->users().front();
  if (!HasFlyFusionKind(*consumer, kFlyFusionKind)) {
    return false;
  }
  std::vector<const Literal*> no_arguments;
  ASSIGN_OR_RETURN(
      Literal literal,
      HloEvaluator().Evaluate(*producer->fused_instructions_computation(),
                              no_arguments));
  HloComputation* parent = producer->parent();
  HloInstruction* constant = parent->AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  constant->set_metadata(producer->metadata());
  RETURN_IF_ERROR(parent->ReplaceInstruction(producer, constant));
  return true;
}

}  // namespace

absl::StatusOr<bool> FlyAutotuneCleanup::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> scalar_materializations;
    std::vector<HloInstruction*> gemms;
    for (HloInstruction* instruction : computation->instructions()) {
      if (IsFlyGemm(*instruction)) {
        gemms.push_back(instruction);
      } else if (instruction->opcode() == HloOpcode::kFusion &&
                 instruction->operand_count() == 0 &&
                 ShapeUtil::IsScalar(instruction->shape())) {
        scalar_materializations.push_back(instruction);
      }
    }
    for (HloInstruction* scalar : scalar_materializations) {
      ASSIGN_OR_RETURN(bool folded, FoldScalarMaterialization(scalar));
      changed |= folded;
    }
    for (HloInstruction* gemm : gemms) {
      ASSIGN_OR_RETURN(bool folded, FoldScalePrologue(gemm));
      changed |= folded;
    }
  }
  return changed;
}

}  // namespace xla::gpu
