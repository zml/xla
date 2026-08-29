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

#include "xla/backends/gpu/transforms/softmax_rewriter_fly.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/codegen/flydsl/layer_norm_support.h"
#include "xla/backends/gpu/codegen/flydsl/softmax_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/status_macros.h"

namespace xla::gpu {
namespace {

int64_t DefaultNumWarps(int64_t columns) {
  if (columns >= 64 && columns <= 256) {
    return 4;
  }
  const int64_t rounded_columns = ((columns + 63) / 64) * 64;
  // Four waves is the native MI300X fast path. Reduce occupancy for short
  // rows, or increase it only when a wave would otherwise own over 64 values.
  for (int64_t num_warps : std::array<int64_t, 5>{4, 2, 1, 8, 16}) {
    const int64_t threads = num_warps * 64;
    const int64_t values_per_thread = (columns + threads - 1) / threads;
    if (threads <= rounded_columns && values_per_thread <= 64) {
      return num_warps;
    }
  }
  return 16;
}

bool HasEscapingSoftmaxIntermediate(const HloInstruction* root,
                                    const HloInstruction* input) {
  const HloInstruction* row_offset =
      flydsl::GetFlySoftmaxExternalRowOffset(*root);
  absl::flat_hash_set<const HloInstruction*> body;
  std::vector<const HloInstruction*> worklist = {root};
  while (!worklist.empty()) {
    const HloInstruction* instruction = worklist.back();
    worklist.pop_back();
    if (instruction == input || instruction == row_offset ||
        !body.insert(instruction).second) {
      continue;
    }
    worklist.insert(worklist.end(), instruction->operands().begin(),
                    instruction->operands().end());
  }
  for (const HloInstruction* instruction : body) {
    if (instruction == root) {
      continue;
    }
    for (const HloInstruction* user : instruction->users()) {
      if (user->parent() == root->parent() && !body.contains(user)) {
        return true;
      }
    }
  }
  return false;
}

absl::Status ConfigureFlyNormalizationFusion(HloInstruction* fusion,
                                             const Shape& tiled_shape) {
  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig* fusion_config =
      gpu_config.mutable_fusion_backend_config();
  fusion_config->set_kind(kFlyFusionKind);
  BlockLevelFusionConfig* block_config =
      fusion_config->mutable_block_level_fusion_config();
  const int64_t rank = tiled_shape.dimensions_size();
  const int64_t columns = tiled_shape.dimensions(rank - 1);
  const int64_t num_warps = DefaultNumWarps(columns);
  Tile* output_tile = block_config->add_output_tiles();
  for (int64_t dimension = 0; dimension < rank - 1; ++dimension) {
    output_tile->add_sizes(columns <= 256 && dimension == rank - 2 ? num_warps
                                                                   : 1);
  }
  output_tile->add_sizes(columns);
  block_config->set_num_warps(num_warps);
  block_config->set_num_ctas(1);
  block_config->set_num_stages(1);
  return fusion->set_backend_config(std::move(gpu_config));
}

absl::StatusOr<HloFusionInstruction*> MakeFlySoftmaxBackwardFusion(
    HloInstruction* root) {
  std::optional<flydsl::FlySoftmaxBackwardDescriptor> backward =
      flydsl::GetFlySoftmaxBackwardDescriptor(*root);
  TF_RET_CHECK(backward.has_value());
  HloComputation::Builder builder("fly_softmax_backward_computation");
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> old_to_new;
  std::vector<HloInstruction*> parameters;
  int64_t parameter_number = 0;

  std::function<void(HloInstruction*)> clone =
      [&](HloInstruction* instruction) {
        if (old_to_new.contains(instruction)) {
          return;
        }
        if (instruction == backward->scores ||
            instruction == backward->upstream_gradient) {
          old_to_new[instruction] =
              builder.AddInstruction(HloInstruction::CreateParameter(
                  parameter_number, instruction->shape(),
                  absl::StrCat("parameter_", parameter_number)));
          parameters.push_back(instruction);
          ++parameter_number;
          return;
        }
        std::vector<HloInstruction*> operands;
        operands.reserve(instruction->operand_count());
        for (HloInstruction* operand : instruction->mutable_operands()) {
          clone(operand);
          operands.push_back(old_to_new.at(operand));
        }
        old_to_new[instruction] = builder.AddInstruction(
            instruction->CloneWithNewOperands(instruction->shape(), operands));
      };
  clone(root);

  HloModule* module = root->GetModule();
  HloComputation* parent = root->parent();
  HloComputation* fused_computation =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                /*is_entry=*/false);
  HloInstruction* fusion = parent->AddInstruction(
      HloInstruction::CreateFusion(root->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   parameters, fused_computation),
      /*new_name=*/"fly_softmax_backward");
  fusion->set_metadata(root->metadata());
  RETURN_IF_ERROR(ConfigureFlyNormalizationFusion(fusion, root->shape()));

  if (root->IsRoot()) {
    parent->set_root_instruction(fusion);
  } else {
    RETURN_IF_ERROR(root->ReplaceAllUsesWith(fusion));
  }
  RETURN_IF_ERROR(parent->RemoveInstructionAndUnusedOperands(root));
  return Cast<HloFusionInstruction>(fusion);
}

absl::StatusOr<HloFusionInstruction*> MakeFlyNormalizationFusion(
    HloInstruction* root) {
  const HloInstruction* softmax_input = flydsl::GetFlySoftmaxInput(*root);
  const HloInstruction* softmax_row_offset =
      flydsl::GetFlySoftmaxExternalRowOffset(*root);
  const bool is_softmax = softmax_input != nullptr;
  std::optional<flydsl::FlyLayerNormDescriptor> layer_norm =
      is_softmax ? std::nullopt : flydsl::GetFlyLayerNormDescriptor(*root);
  TF_RET_CHECK(is_softmax || layer_norm.has_value());
  HloComputation::Builder builder(is_softmax ? "fly_softmax_computation"
                                             : "fly_layer_norm_computation");
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> old_to_new;
  std::vector<HloInstruction*> parameters;
  int64_t parameter_number = 0;

  std::function<void(HloInstruction*)> clone =
      [&](HloInstruction* instruction) {
        if (old_to_new.contains(instruction)) {
          return;
        }
        // Keep an arbitrary softmax producer outside the normalization
        // fusion. In a transformer this is normally the QK contraction; the
        // standalone matcher used to require it to be an HLO parameter and
        // consequently missed real graphs. The normalization body sees the
        // producer (and an optional externally-computed row offset) as fusion
        // parameters, just like Triton's softmax diamond rewriter.
        const bool is_softmax_boundary =
            is_softmax &&
            (instruction == softmax_input || instruction == softmax_row_offset);
        if (instruction->opcode() == HloOpcode::kParameter ||
            is_softmax_boundary) {
          old_to_new[instruction] =
              builder.AddInstruction(HloInstruction::CreateParameter(
                  parameter_number, instruction->shape(),
                  absl::StrCat("parameter_", parameter_number)));
          parameters.push_back(instruction);
          ++parameter_number;
          return;
        }

        std::vector<HloInstruction*> operands;
        operands.reserve(instruction->operand_count());
        for (HloInstruction* operand : instruction->mutable_operands()) {
          clone(operand);
          operands.push_back(old_to_new.at(operand));
        }
        old_to_new[instruction] = builder.AddInstruction(
            instruction->CloneWithNewOperands(instruction->shape(), operands));
      };
  clone(root);

  HloModule* module = root->GetModule();
  HloComputation* parent = root->parent();
  HloComputation* fused_computation =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                /*is_entry=*/false);
  HloInstruction* fusion = parent->AddInstruction(
      HloInstruction::CreateFusion(root->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   parameters, fused_computation),
      /*new_name=*/is_softmax ? "fly_softmax" : "fly_layer_norm");
  fusion->set_metadata(root->metadata());

  const Shape& tiled_shape =
      is_softmax ? root->shape() : layer_norm->output->shape();
  RETURN_IF_ERROR(ConfigureFlyNormalizationFusion(fusion, tiled_shape));

  if (root->IsRoot()) {
    parent->set_root_instruction(fusion);
  } else {
    RETURN_IF_ERROR(root->ReplaceAllUsesWith(fusion));
  }
  RETURN_IF_ERROR(parent->RemoveInstructionAndUnusedOperands(root));
  return Cast<HloFusionInstruction>(fusion);
}

}  // namespace

absl::StatusOr<bool> SoftmaxRewriterFly::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Rewrite the derivative first. It consumes the forward exponentials and
  // row sum in XLA's autodiff graph. Once those external uses disappear, a
  // second scan can safely collapse the now-unshared forward softmax too.
  std::vector<HloInstruction*> backward_roots;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!computation->caller_instructions(HloOpcode::kCustomCall).empty()) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (!flydsl::GetFlySoftmaxBackwardDescriptor(*instruction).has_value()) {
        continue;
      }
      backward_roots.erase(
          std::remove_if(backward_roots.begin(), backward_roots.end(),
                         [&](const HloInstruction* candidate) {
                           return instruction->IsUserOf(candidate);
                         }),
          backward_roots.end());
      backward_roots.push_back(instruction);
    }
  }
  for (HloInstruction* root : backward_roots) {
    ASSIGN_OR_RETURN(HloFusionInstruction * fusion,
                     MakeFlySoftmaxBackwardFusion(root));
    (void)fusion;
  }

  std::vector<HloInstruction*> normalization_roots;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!computation->caller_instructions(HloOpcode::kCustomCall).empty()) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      const HloInstruction* softmax_input =
          flydsl::GetFlySoftmaxInput(*instruction);
      if (softmax_input != nullptr &&
          !HasEscapingSoftmaxIntermediate(instruction, softmax_input)) {
        // A narrowed softmax has two individually valid roots: the inner F32
        // divide and the outer F16/BF16 conversion. Prefer the outer root so
        // the conversion remains in the native kernel and do not later visit
        // a candidate whose body was removed by the first rewrite.
        normalization_roots.erase(
            std::remove_if(normalization_roots.begin(),
                           normalization_roots.end(),
                           [&](const HloInstruction* candidate) {
                             return instruction->IsUserOf(candidate);
                           }),
            normalization_roots.end());
        normalization_roots.push_back(instruction);
        continue;
      }
      std::optional<flydsl::FlyLayerNormDescriptor> layer_norm =
          flydsl::GetFlyLayerNormDescriptor(*instruction);
      if (!layer_norm.has_value()) {
        continue;
      }
      if (layer_norm->output_count > 1) {
        normalization_roots.erase(
            std::remove(normalization_roots.begin(), normalization_roots.end(),
                        layer_norm->output),
            normalization_roots.end());
      }
      normalization_roots.push_back(instruction);
    }
  }

  for (HloInstruction* root : normalization_roots) {
    ASSIGN_OR_RETURN(HloFusionInstruction * fusion,
                     MakeFlyNormalizationFusion(root));
    (void)fusion;
  }
  return !backward_roots.empty() || !normalization_roots.empty();
}

}  // namespace xla::gpu
