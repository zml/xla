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

#include "xla/backends/gpu/transforms/fp8_block_gemv_arm.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/codegen/triton/fp8_block_gemv.h"
#include "xla/backends/gpu/transforms/fused_scaled_dot_rewriter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

absl::StatusOr<HloInstruction*> TryEmitFp8BlockGemvFusion(
    HloComputation* comp, HloScaledDotInstruction* dot) {
  std::optional<Fp8BlockGemvConfig> config = Fp8BlockGemvConfigFor(*dot);
  if (!config.has_value()) return nullptr;
  VLOG(1) << "fused scaled dot claimed " << dot->name() << ": "
          << dot->shape().ToString() << " tile " << dot->shape().dimensions(0)
          << "x" << config->block_n << " block_k " << config->block_k << " "
          << config->num_warps << " warps, " << config->num_stages
          << " stages";

  HloComputation::Builder builder(
      absl::StrCat(kFp8BlockGemvComputationPrefix, dot->name()));
  std::vector<HloInstruction*> operands;
  std::vector<HloInstruction*> parameters;
  operands.reserve(dot->operand_count());
  parameters.reserve(dot->operand_count());
  // The scales are read as f32, so the convert stays outside the fusion.
  const bool w8a8 = dot->operand(0)->shape().element_type() == F8E4M3FN;
  for (int64_t i = 0; i < dot->operand_count(); ++i) {
    HloInstruction* operand = dot->mutable_operand(i);
    if (w8a8 && i >= 2 && operand->shape().element_type() != F32) {
      operand = comp->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(operand->shape(), F32), operand));
    }
    operands.push_back(operand);
    parameters.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i, operand->shape(), absl::StrCat("p", i))));
  }
  builder.AddInstruction(dot->CloneWithNewOperands(dot->shape(), parameters));
  HloComputation* body = comp->parent()->AddComputationAndUnifyNamesAndIds(
      builder.Build(), /*is_entry=*/false);

  xla::xtile::Tile contracting_tile;
  contracting_tile.add_sizes(config->block_k);
  ABSL_RETURN_IF_ERROR(body->root_instruction()->set_backend_config(
      contracting_tile));

  HloInstruction* fusion = comp->AddInstruction(HloInstruction::CreateFusion(
      dot->shape(), HloInstruction::FusionKind::kCustom, operands, body));

  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kTritonNestedGemmFusionKind));
  xla::xtile::BlockLevelFusionConfig& block_config =
      *backend_config.mutable_block_level_fusion_config();
  xla::xtile::Tile& output_tile = *block_config.add_output_tiles();
  output_tile.add_sizes(config->block_m);
  output_tile.add_sizes(config->block_n);
  block_config.set_num_warps(config->num_warps);
  block_config.set_num_stages(config->num_stages);
  block_config.set_num_ctas(1);
  ABSL_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  if (!MatchFp8BlockGemv(*Cast<HloFusionInstruction>(fusion)).has_value()) {
    VLOG(1) << "fused scaled dot unclaimed after building " << dot->name()
            << ": the emitter would decline it";
    ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(fusion));
    return nullptr;
  }
  return fusion;
}

}  // namespace

FusedScaledDotArm Fp8BlockGemvArm() {
  return [](HloComputation* comp, HloScaledDotInstruction* dot) {
    return TryEmitFp8BlockGemvFusion(comp, dot);
  };
}

}  // namespace gpu
}  // namespace xla
