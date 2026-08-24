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

#include "xla/backends/gpu/transforms/fused_scaled_dot_arms_cuda.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/transforms/fused_scaled_dot_rewriter.h"
#include "xla/backends/gpu/codegen/triton/fp8_block_gemv.h"
#include "xla/backends/gpu/codegen/triton/nvfp4_decode_dot.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

absl::StatusOr<HloInstruction*> TryEmitFp8BlockGemvFusion(
    HloComputation* comp, HloScaledDotInstruction* dot,
    const se::GpuComputeCapability& gpu_version) {
  std::optional<Fp8BlockGemvConfig> config =
      Fp8BlockGemvConfigFor(*dot, gpu_version);
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
  for (int64_t i = 0; i < dot->operand_count(); ++i) {
    HloInstruction* operand = dot->mutable_operand(i);
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
  static const bool prefer_tile_ir = [] {
    const char* env = std::getenv("ZML_FP8_BLOCK_GEMV_TILEIR");
    return env != nullptr && absl::string_view(env) == "1";
  }();
  backend_config.set_kind(std::string(
      prefer_tile_ir ? kTileIrFusionKind : kTritonNestedGemmFusionKind));
  xla::xtile::BlockLevelFusionConfig& block_config =
      *backend_config.mutable_block_level_fusion_config();
  xla::xtile::Tile& output_tile = *block_config.add_output_tiles();
  output_tile.add_sizes(dot->shape().dimensions(0));
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

absl::StatusOr<HloInstruction*> TryEmitNvfp4DecodeDotFusion(
    HloComputation* comp, HloScaledDotInstruction* dot,
    const se::GpuComputeCapability& gpu_version) {
  std::optional<Nvfp4DecodeDotConfig> config =
      Nvfp4DecodeDotConfigFor(*dot, gpu_version);
  if (!config.has_value()) {
    VLOG(1) << "nvfp4 arm declined " << dot->name() << ": "
            << dot->ToString();
    return nullptr;
  }

  HloComputation::Builder builder(
      absl::StrCat(kNvfp4DecodeDotComputationPrefix, dot->name()));
  std::vector<HloInstruction*> operands;
  std::vector<HloInstruction*> parameters;
  operands.reserve(dot->operand_count());
  parameters.reserve(dot->operand_count());

  if (!config->swap) {
    for (int64_t i = 0; i < dot->operand_count(); ++i) {
      HloInstruction* operand = dot->mutable_operand(i);
      operands.push_back(operand);
      parameters.push_back(builder.AddInstruction(
          HloInstruction::CreateParameter(i, operand->shape(),
                                          absl::StrCat("p", i))));
    }
    builder.AddInstruction(dot->CloneWithNewOperands(dot->shape(), parameters));
  } else {
  ABSL_ASSIGN_OR_RETURN(DotOperandDims lhs_dims,
                   DotOperandDims::FromDotOperand(dot, 0));
  ABSL_ASSIGN_OR_RETURN(DotOperandDims rhs_dims,
                   DotOperandDims::FromDotOperand(dot, 1));
  const size_t num_batch_dims = lhs_dims.Rank(DotOperandDims::kBatch);
  const size_t num_lhs_noncontracting =
      lhs_dims.Rank(DotOperandDims::kNonContracting);
  const size_t num_rhs_noncontracting =
      rhs_dims.Rank(DotOperandDims::kNonContracting);
  std::vector<int64_t> permutation;
  permutation.reserve(dot->shape().dimensions().size());
  auto fill = [&](int64_t count, int64_t start) {
    while (count--) permutation.push_back(start++);
  };
  fill(num_batch_dims, 0);
  fill(num_rhs_noncontracting, num_batch_dims + num_lhs_noncontracting);
  fill(num_lhs_noncontracting, num_batch_dims);
  const Shape swapped_shape =
      ShapeUtil::ReorderLogicalDimensions(dot->shape(), permutation);
  ABSL_ASSIGN_OR_RETURN(DotDimensionNumbers swapped_dnums,
                   DotOperandDims::CreateDotDimensionNumbers(rhs_dims,
                                                             lhs_dims));

  constexpr std::array<int, 4> kSwappedOrder = {1, 0, 3, 2};
  for (int i = 0; i < static_cast<int>(kSwappedOrder.size()); ++i) {
    HloInstruction* operand = dot->mutable_operand(kSwappedOrder[i]);
    operands.push_back(operand);
    parameters.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i, operand->shape(), absl::StrCat("p", i))));
  }
  HloInstruction* swapped = builder.AddInstruction(
      HloInstruction::CreateScaledDot(swapped_shape, parameters[0],
                                      parameters[1], parameters[2],
                                      parameters[3], swapped_dnums,
                                      dot->precision_config()));
  builder.AddInstruction(
      HloInstruction::CreateBitcast(dot->shape(), swapped));
  }
  HloComputation* body = comp->parent()->AddComputationAndUnifyNamesAndIds(
      builder.Build(), /*is_entry=*/false);

  xla::xtile::Tile contracting_tile;
  contracting_tile.add_sizes(config->block_k);
  ABSL_RETURN_IF_ERROR(
      hlo_query::GetFirstInstructionWithOpcode(*body, HloOpcode::kScaledDot)
          ->set_backend_config(contracting_tile));

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
  for (int64_t d = 0; d + 2 < dot->shape().dimensions().size(); ++d) {
    output_tile.add_sizes(1);
  }
  output_tile.add_sizes(config->batch_tile);
  output_tile.add_sizes(config->weight_tile);
  block_config.set_num_warps(config->num_warps);
  block_config.set_num_stages(config->num_stages);
  block_config.set_num_ctas(1);
  block_config.set_is_tma_allowed(true);
  block_config.set_is_warp_specialization_allowed(true);
  ABSL_RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));

  return fusion;
}

}  // namespace

FusedScaledDotArm Fp8BlockGemvArm(
    const se::GpuComputeCapability& gpu_version) {
  return [gpu_version](HloComputation* comp, HloScaledDotInstruction* dot) {
    return TryEmitFp8BlockGemvFusion(comp, dot, gpu_version);
  };
}

FusedScaledDotArm Nvfp4DecodeDotArm(
    const se::GpuComputeCapability& gpu_version) {
  return [gpu_version](HloComputation* comp, HloScaledDotInstruction* dot) {
    return TryEmitNvfp4DecodeDotFusion(comp, dot, gpu_version);
  };
}

}  // namespace gpu
}  // namespace xla
