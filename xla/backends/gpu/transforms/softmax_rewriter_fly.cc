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

absl::StatusOr<HloFusionInstruction*> MakeFlyNormalizationFusion(
    HloInstruction* root) {
  const bool is_softmax = flydsl::IsFlySoftmaxRoot(*root);
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
        if (instruction->opcode() == HloOpcode::kParameter) {
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

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig* fusion_config =
      gpu_config.mutable_fusion_backend_config();
  fusion_config->set_kind(kFlyFusionKind);
  BlockLevelFusionConfig* block_config =
      fusion_config->mutable_block_level_fusion_config();
  const Shape& tiled_shape =
      is_softmax ? root->shape() : layer_norm->output->shape();
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
  RETURN_IF_ERROR(fusion->set_backend_config(std::move(gpu_config)));

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
  std::vector<HloInstruction*> normalization_roots;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!computation->caller_instructions(HloOpcode::kCustomCall).empty()) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (flydsl::IsFlySoftmaxRoot(*instruction)) {
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
            std::remove(normalization_roots.begin(),
                        normalization_roots.end(), layer_norm->output),
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
  return !normalization_roots.empty();
}

}  // namespace xla::gpu
