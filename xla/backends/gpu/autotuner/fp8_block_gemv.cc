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

#include "xla/backends/gpu/autotuner/fp8_block_gemv.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/gpu/codegen/triton/fp8_block_gemv.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

constexpr int64_t kScaleBlock = 128;

std::unique_ptr<BackendConfig> Pack(int64_t batch, int64_t block_n,
                                    int64_t block_k, int num_warps,
                                    int num_stages) {
  auto config = std::make_unique<BackendConfig>();
  TileIrFusionConfig& fp8 = *config->mutable_fp8_block_gemv();
  xla::xtile::BlockLevelFusionConfig& block = *fp8.mutable_block_level_fusion_config();
  xla::xtile::Tile& tile = *block.add_output_tiles();
  tile.add_sizes(batch);
  tile.add_sizes(block_n);
  block.set_num_warps(num_warps);
  block.set_num_stages(num_stages);
  block.set_num_ctas(1);
  fp8.set_contracting_tile_size(block_k);
  return config;
}

}  // namespace

bool Fp8BlockGemvBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) return false;
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) return false;
  if (gpu_config->fusion_backend_config().kind() !=
      kTritonNestedGemmFusionKind) {
    return false;
  }
  return MatchFp8BlockGemv(*Cast<HloFusionInstruction>(&instr)).has_value();
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
Fp8BlockGemvBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) return configs;
  std::optional<Fp8BlockGemvSpec> spec =
      MatchFp8BlockGemv(*Cast<HloFusionInstruction>(&instr));
  const int64_t batch = spec->batch;
  const int64_t n = spec->n;
  const int64_t k = spec->k;

  const bool single_row = batch == 1;
  const int64_t min_block_k = single_row ? kScaleBlock : 2 * kScaleBlock;
  const int max_warps = single_row ? 16 : 8;

  for (int64_t block_n = 4; block_n <= kScaleBlock; block_n *= 2) {
    if (n % block_n != 0 || kScaleBlock % block_n != 0) continue;
    for (int64_t block_k = min_block_k; block_k <= 2048; block_k *= 2) {
      if (k % block_k != 0) continue;
      for (int num_warps = 2; num_warps <= max_warps; num_warps *= 2) {
        for (int num_stages = 2; num_stages <= 6; ++num_stages) {
          configs.push_back(
              Pack(batch, block_n, block_k, num_warps, num_stages));
        }
      }
    }
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
Fp8BlockGemvBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "Fp8BlockGemvBackend does not support this instruction.");
  }
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  const HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
  int64_t block_k = 0;
  if (dot != nullptr) {
    if (absl::StatusOr<xla::xtile::Tile> tile = dot->backend_config<xla::xtile::Tile>();
        tile.ok() && tile->sizes_size() > 0) {
      block_k = tile->sizes(tile->sizes_size() - 1);
    }
  }
  auto config = std::make_unique<BackendConfig>();
  TileIrFusionConfig& fp8 = *config->mutable_fp8_block_gemv();
  *fp8.mutable_block_level_fusion_config() =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  fp8.set_contracting_tile_size(block_k);
  return config;
}

absl::Status Fp8BlockGemvBackend::ApplyConfig(HloInstruction& instr,
                                              const BackendConfig& config) {
  if (!config.has_fp8_block_gemv()) {
    return absl::InvalidArgumentError(
        "Expected an fp8_block_gemv config for Fp8BlockGemvBackend.");
  }
  const TileIrFusionConfig& fp8 = config.fp8_block_gemv();
  const int64_t block_k = fp8.contracting_tile_size();
  if (block_k <= 0 || block_k % kScaleBlock != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("contracting tile ", block_k,
                     " must be a positive multiple of ", kScaleBlock));
  }
  if (instr.opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError("expected a fusion");
  }
  HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
  if (dot == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("no scaled dot in fusion ", instr.name()));
  }
  xla::xtile::Tile dot_tile;
  dot_tile.add_sizes(block_k);
  ABSL_RETURN_IF_ERROR(dot->set_backend_config(dot_tile));

  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kTritonNestedGemmFusionKind));
  backend_config.clear_triton_gemm_config();
  *backend_config.mutable_block_level_fusion_config() =
      fp8.block_level_fusion_config();
  ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  instr.set_fusion_kind(HloInstruction::FusionKind::kCustom);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
