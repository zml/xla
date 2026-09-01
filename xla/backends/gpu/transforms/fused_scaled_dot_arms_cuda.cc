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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/transforms/fused_scaled_dot_rewriter.h"
#include "xla/backends/gpu/transforms/splitk_rewriter.h"
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
  // The CUTLASS rung reads f32 scales straight from the buffers, so the convert stays outside the fusion.
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

bool ScaledDotOperandsDivide(const HloScaledDotInstruction& dot,
                             int64_t split_k) {
  absl::StatusOr<std::array<DotOperandDims, 4>> dims =
      DotOperandDims::FromScaledDot(&dot);
  if (!dims.ok()) return false;
  for (int i = 0; i < 4; ++i) {
    absl::Span<const int64_t> contracting =
        (*dims)[i].Indices(DotOperandDims::kContracting);
    if (contracting.size() != 1) return false;
    if (dot.operand(i)->shape().dimensions(contracting[0]) % split_k != 0) {
      return false;
    }
  }
  return true;
}

// Some tile must survive the split, not the widest one.
bool SplitLeavesAConfig(int64_t k, int64_t split_k) {
  return k % split_k == 0 && HasNvfp4BlockK(k / split_k);
}

// Split K only where narrowing N cannot fill the machine: tiles are counted against the
// autotuner's floor tile, not this arm's seed. The batch axis keeps its real tile.
int64_t ChooseNvfp4SplitK(const HloScaledDotInstruction& dot,
                          const Nvfp4DecodeDotConfig& config,
                          const se::DeviceDescription& device,
                          const Nvfp4DecodeLimits& limits) {
  if (limits.max_split_k <= 1) return 1;

  std::optional<Nvfp4DecodeDotSpec> spec =
      MatchNvfp4DecodeDot(dot, device.gpu_compute_capability());
  if (!spec.has_value()) return 1;

  const int64_t weight_tile = std::max<int64_t>(1, limits.min_weight_tile);
  const int64_t batch_tile = std::max<int64_t>(1, config.batch_tile);
  const int64_t tiles = CeilOfRatio(spec->weight_rows, weight_tile) *
                        CeilOfRatio(spec->batch, batch_tile);
  const int64_t cores = std::max<int64_t>(1, device.core_count());
  if (tiles >= cores) return 1;

  int64_t split_k = 1;
  while (split_k * 2 <= limits.max_split_k && tiles * split_k * 2 <= cores &&
         SplitLeavesAConfig(spec->k, split_k * 2)) {
    split_k *= 2;
  }
  return split_k;
}

absl::StatusOr<HloInstruction*> TryEmitNvfp4DecodeDotFusion(
    HloComputation* comp, HloScaledDotInstruction* dot,
    const se::DeviceDescription& device_description) {
  const se::GpuComputeCapability& gpu_version =
      device_description.gpu_compute_capability();
  std::optional<Nvfp4DecodeDotConfig> config =
      Nvfp4DecodeDotConfigFor(*dot, gpu_version);
  if (!config.has_value()) {
    VLOG(1) << "nvfp4 arm declined " << dot->name() << ": "
            << dot->ToString();
    return nullptr;
  }

  // Decide fully before mutating: the split is the one mutation, the re-match a contract check.
  HloInstruction* replacement = nullptr;
  const int64_t split_k =
      dot->dot_dimension_numbers().lhs_batch_dimensions().empty()
          ? ChooseNvfp4SplitK(*dot, *config, device_description,
                              Nvfp4DecodeLimitsFor(gpu_version))
          : 1;
  if (split_k > 1 && ScaledDotOperandsDivide(*dot, split_k)) {
    ABSL_ASSIGN_OR_RETURN(SplitScaledDot split,
                          SplitScaledDotContraction(dot, split_k));
    auto* split_dot = Cast<HloScaledDotInstruction>(split.dot);
    std::optional<Nvfp4DecodeDotConfig> split_config =
        Nvfp4DecodeDotConfigFor(*split_dot, gpu_version);
    TF_RET_CHECK(split_config.has_value())
        << "nvfp4 arm split " << dot->name() << " by " << split_k
        << " and then declined the result";
    replacement = split.root;
    dot = split_dot;
    config = split_config;
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

  if (replacement == nullptr) return fusion;
  ABSL_RETURN_IF_ERROR(dot->parent()->ReplaceInstruction(dot, fusion));
  return replacement;
}

}  // namespace

FusedScaledDotArm Fp8BlockGemvArm(
    const se::GpuComputeCapability& gpu_version) {
  return [gpu_version](HloComputation* comp, HloScaledDotInstruction* dot) {
    return TryEmitFp8BlockGemvFusion(comp, dot, gpu_version);
  };
}

FusedScaledDotArm Nvfp4DecodeDotArm(
    const se::DeviceDescription& device_description) {
  return [device_description](HloComputation* comp,
                              HloScaledDotInstruction* dot) {
    return TryEmitNvfp4DecodeDotFusion(comp, dot, device_description);
  };
}

}  // namespace gpu
}  // namespace xla
