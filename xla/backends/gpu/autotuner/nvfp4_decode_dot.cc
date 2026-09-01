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

#include "xla/backends/gpu/autotuner/nvfp4_decode_dot.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/gpu/codegen/triton/nvfp4_decode_dot.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/backends/gpu/transforms/convert_triton_gemm_config.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

constexpr int64_t kScaleBlock = 16;

TileIrFusionConfig* SlotFor(Nvfp4DecodeDotBackend::Rung rung,
                            BackendConfig& config) {
  return rung == Nvfp4DecodeDotBackend::Rung::kTileIr
             ? config.mutable_nvfp4_decode_dot_tile_ir()
             : config.mutable_nvfp4_decode_dot();
}

const TileIrFusionConfig* ConstSlotFor(Nvfp4DecodeDotBackend::Rung rung,
                                       const BackendConfig& config) {
  if (rung == Nvfp4DecodeDotBackend::Rung::kTileIr) {
    return config.has_nvfp4_decode_dot_tile_ir()
               ? &config.nvfp4_decode_dot_tile_ir()
               : nullptr;
  }
  return config.has_nvfp4_decode_dot() ? &config.nvfp4_decode_dot() : nullptr;
}

std::unique_ptr<BackendConfig> Pack(Nvfp4DecodeDotBackend::Rung rung,
                                    int64_t block_m, int64_t block_n,
                                    int64_t block_k, int num_warps,
                                    int num_stages, bool tma_allowed) {
  auto config = std::make_unique<BackendConfig>();
  TileIrFusionConfig& nvfp4 = *SlotFor(rung, *config);
  xla::xtile::BlockLevelFusionConfig& block =
      *nvfp4.mutable_block_level_fusion_config();
  xla::xtile::Tile& tile = *block.add_output_tiles();
  tile.add_sizes(block_m);
  tile.add_sizes(block_n);
  block.set_num_warps(num_warps);
  block.set_num_stages(num_stages);
  block.set_num_ctas(1);
  const bool triton = rung != Nvfp4DecodeDotBackend::Rung::kTileIr;
  block.set_is_tma_allowed(triton && tma_allowed);
  block.set_is_warp_specialization_allowed(triton);
  nvfp4.set_contracting_tile_size(block_k);
  return config;
}

}  // namespace

bool Nvfp4DecodeDotBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) return false;
  absl::StatusOr<GpuBackendConfig> gpu_config =
      instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) return false;
  const absl::string_view kind = gpu_config->fusion_backend_config().kind();
  if (kind != kTritonNestedGemmFusionKind && kind != kTileIrFusionKind) {
    VLOG(1) << "nvfp4 backend declined " << instr.name() << ": kind is "
            << kind;
    return false;
  }
  if (rung_ == Rung::kTileIr &&
      (!debug_options().xla_gpu_experimental_scaled_dot_with_tile_ir() ||
       !Nvfp4DecodeLimitsFor(
            target_config().device_description.gpu_compute_capability())
            .offer_tile_ir_rung)) {
    return false;
  }
  if (rung_ == Rung::kTileIr && instr.shape().dimensions().size() > 2) {
    // The lowering to cuda_tile rejects a batch dimension, so a dot the arm
    // split cannot be served here. Decline it at bid time: offering it instead
    // spends a tile search and an emit on something that can only compile-fail,
    // and the failure is swallowed at VLOG(2). Worse, a lone surviving config
    // is returned without ever being compiled, so a fusion that reached this
    // rung by itself would hard-error at emission rather than fall back.
    VLOG(1) << "nvfp4 backend declined " << instr.name()
            << ": the Tile IR rung cannot serve a rank-"
            << instr.shape().dimensions().size() << " (split) dot";
    return false;
  }
  if (!MatchNvfp4DecodeDotFusion(*Cast<HloFusionInstruction>(&instr))
           .has_value()) {
    VLOG(1) << "nvfp4 backend declined " << instr.name()
            << ": not one of ours (computation "
            << (instr.fused_instructions_computation() != nullptr
                    ? instr.fused_instructions_computation()->name()
                    : "<null>")
            << ")";
    return false;
  }
  return true;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
Nvfp4DecodeDotBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) return configs;
  std::optional<Nvfp4DecodeDotSpec> spec =
      MatchNvfp4DecodeDotFusion(*Cast<HloFusionInstruction>(&instr));

  const Nvfp4DecodeLimits& limits = Nvfp4DecodeLimitsFor(
      target_config().device_description.gpu_compute_capability());
  const int64_t weight_rows = spec->weight_rows;
  const int64_t k = spec->k;

  const bool tile_ir = rung_ == Rung::kTileIr;
  const int64_t min_weight_tile =
      tile_ir ? limits.tile_ir_min_weight_tile : limits.min_weight_tile;
  const int64_t min_batch_tile =
      tile_ir ? limits.tile_ir_min_batch_tile : limits.min_batch_tile;
  // tileiras runs NVFP4 on sm_103 at block_n 16 only, so the Tile IR rung
  // tiles the batch exactly instead of searching wider.
  int64_t max_batch_tile = 256;
  if (tile_ir) {
    max_batch_tile = min_batch_tile;
    while (max_batch_tile < spec->batch) max_batch_tile *= 2;
    if (max_batch_tile > limits.tile_ir_max_batch_tile) {
      VLOG(1) << "nvfp4 backend (tile_ir) declined " << instr.name()
              << ": batch " << spec->batch << " needs a " << max_batch_tile
              << "-wide tile, tileiras runs " << limits.tile_ir_max_batch_tile;
      return configs;
    }
  }
  const bool weight_on_lhs = spec->weight_on_lhs;
  auto pack = [&](int64_t weight_tile, int64_t batch_tile, int64_t block_k,
                  int num_warps, int num_stages) {
    const int64_t block_m = weight_on_lhs ? weight_tile : batch_tile;
    const int64_t block_n = weight_on_lhs ? batch_tile : weight_tile;
    return Pack(rung_, block_m, block_n, block_k, num_warps, num_stages,
                /*tma_allowed=*/block_m >= 128);
  };
  for (int64_t weight_tile = min_weight_tile; weight_tile <= 256;
       weight_tile *= 2) {
    if (weight_tile > weight_rows) break;
    for (int64_t batch_tile = tile_ir ? max_batch_tile : min_batch_tile;
         batch_tile <= max_batch_tile; batch_tile *= 2) {
      for (int64_t block_k = 128; block_k <= 512; block_k *= 2) {
        if (k % block_k != 0 || block_k % kScaleBlock != 0) continue;
        if (tile_ir) {
          configs.push_back(pack(weight_tile, batch_tile, block_k,
                                 /*num_warps=*/4, /*num_stages=*/1));
          continue;
        }
        for (int num_warps = 4; num_warps <= 8; num_warps *= 2) {
          const int64_t elements_per_thread =
              (weight_tile * batch_tile) / (num_warps * 32);
          if (elements_per_thread > 64) continue;
          for (int num_stages = 1; num_stages <= 8; ++num_stages) {
            configs.push_back(pack(weight_tile, batch_tile, block_k, num_warps,
                                   num_stages));
          }
        }
      }
    }
  }
  VLOG(1) << "nvfp4 backend (" << (tile_ir ? "tile_ir" : "triton")
          << ") offering " << configs.size() << " configs for "
          << instr.name() << " (batch=" << spec->batch
          << " weight_rows=" << weight_rows << " k=" << k
          << (weight_on_lhs ? ", weight on lhs)" : ", weight on rhs)");
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
Nvfp4DecodeDotBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "Nvfp4DecodeDotBackend does not support this instruction.");
  }
  std::optional<Nvfp4DecodeDotSpec> spec =
      MatchNvfp4DecodeDotFusion(*Cast<HloFusionInstruction>(&instr));
  const Nvfp4DecodeLimits& limits = Nvfp4DecodeLimitsFor(
      target_config().device_description.gpu_compute_capability());
  const Nvfp4DecodeDotConfig seed = Nvfp4DecodeDotSeed(*spec, limits);
  const HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
  if (dot == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("no scaled dot in fusion ", instr.name()));
  }
  int64_t block_k = 0;
  if (absl::StatusOr<xla::xtile::Tile> tile = dot->backend_config<xla::xtile::Tile>();
      tile.ok() && tile->sizes_size() > 0) {
    block_k = tile->sizes(tile->sizes_size() - 1);
  }
  if (block_k <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("no contracting tile on the dot in ", instr.name()));
  }
  const int64_t block_m =
      spec->weight_on_lhs ? seed.weight_tile : seed.batch_tile;
  const int64_t block_n =
      spec->weight_on_lhs ? seed.batch_tile : seed.weight_tile;
  return Pack(rung_, block_m, block_n, block_k, seed.num_warps,
              seed.num_stages, /*tma_allowed=*/block_m >= 128);
}

absl::Status Nvfp4DecodeDotBackend::ApplyConfig(HloInstruction& instr,
                                                const BackendConfig& config) {
  const TileIrFusionConfig* slot = ConstSlotFor(rung_, config);
  if (slot == nullptr) {
    return absl::InvalidArgumentError(
        "Expected an nvfp4_decode_dot config for Nvfp4DecodeDotBackend.");
  }
  const TileIrFusionConfig& nvfp4 = *slot;
  const int64_t block_k = nvfp4.contracting_tile_size();
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
  backend_config.set_kind(std::string(rung_ == Rung::kTileIr
                                          ? kTileIrFusionKind
                                          : kTritonNestedGemmFusionKind));
  backend_config.clear_triton_gemm_config();

  const xla::xtile::BlockLevelFusionConfig& knobs = nvfp4.block_level_fusion_config();
  const int64_t block_m = knobs.output_tiles(0).sizes(0);
  const int64_t block_n = knobs.output_tiles(0).sizes(1);
  TritonGemmConfig gemm_config(
      static_cast<int>(block_m), static_cast<int>(block_n),
      static_cast<int>(block_k), knobs.num_stages(), knobs.num_warps(),
      knobs.num_ctas(), knobs.is_tma_allowed(),
      knobs.is_warp_specialization_allowed());
  ABSL_ASSIGN_OR_RETURN(xla::xtile::BlockLevelParameters params,
                   FindBlockLevelParameters(dot, gemm_config, mlir_context_,
                                            target_config().device_description));
  *backend_config.mutable_block_level_fusion_config() =
      params.ToBlockLevelFusionConfig();
  ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  instr.set_fusion_kind(HloInstruction::FusionKind::kCustom);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
