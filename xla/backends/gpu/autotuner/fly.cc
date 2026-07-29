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

#include "xla/backends/gpu/autotuner/fly.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

const HloInstruction* FindDot(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  return hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kDot);
}

bool IsSupportedDot(const HloInstruction& dot) {
  if (dot.operand_count() != 2 ||
      dot.operand(0)->shape().dimensions_size() != 2 ||
      dot.operand(1)->shape().dimensions_size() != 2 ||
      dot.shape().dimensions_size() != 2 ||
      dot.operand(0)->shape().element_type() != BF16 ||
      dot.operand(1)->shape().element_type() != BF16 ||
      (dot.shape().element_type() != BF16 &&
       dot.shape().element_type() != F32)) {
    return false;
  }
  const DotDimensionNumbers& dims = dot.dot_dimension_numbers();
  return dims.lhs_batch_dimensions().empty() &&
         dims.rhs_batch_dimensions().empty() &&
         dims.lhs_contracting_dimensions_size() == 1 &&
         dims.rhs_contracting_dimensions_size() == 1 &&
         dims.lhs_contracting_dimensions(0) == 1 &&
         dims.rhs_contracting_dimensions(0) == 0;
}

std::unique_ptr<BackendConfig> MakeConfig(int64_t block_m, int64_t block_n,
                                          int64_t block_k,
                                          int64_t num_warps,
                                          FlyGemmConfig::MfmaAtom mfma_atom,
                                          bool prefetch_rhs = false,
                                          bool stage_output = false,
                                          int32_t waves_per_eu = 0,
                                          bool schedule_instructions = false,
                                          bool stage_rhs = false,
                                          bool async_lhs = false) {
  auto config = std::make_unique<BackendConfig>();
  FlyGemmConfig* key = config->mutable_fly();
  key->set_block_m(block_m);
  key->set_block_n(block_n);
  key->set_block_k(block_k);
  key->set_num_warps(num_warps);
  key->set_mfma_atom(mfma_atom);
  key->set_prefetch_rhs(prefetch_rhs);
  key->set_stage_output(stage_output);
  key->set_waves_per_eu(waves_per_eu);
  key->set_schedule_instructions(schedule_instructions);
  key->set_stage_rhs(stage_rhs);
  key->set_async_lhs(async_lhs);
  return config;
}

}  // namespace

bool FlyBackend::IsSupported(const HloInstruction& instr) {
  if (!debug_options().xla_gpu_enable_flydsl_gemm()) {
    return false;
  }
  if (!target_config()
           .device_description.gpu_compute_capability()
           .IsRocm()) {
    return false;
  }
  const HloInstruction* dot = FindDot(instr);
  return dot != nullptr &&
         instr.fused_instructions_computation()->root_instruction() == dot &&
         IsSupportedDot(*dot);
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
FlyBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }

  const HloInstruction* dot = FindDot(instr);
  const int64_t m = dot->shape().dimensions(0);
  const int64_t n = dot->shape().dimensions(1);
  const int64_t k = dot->operand(0)->shape().dimensions(1);
  const bool rhs_k_contiguous =
      dot->operand(1)->shape().has_layout() &&
      dot->operand(1)->shape().layout().minor_to_major(0) == 0;
  if ((m != 1 && m % 16 != 0) || (n != 1 && n % 16 != 0) ||
      k % 16 != 0) {
    return configs;
  }

  constexpr std::array<int64_t, 5> kBlockSizes = {16, 32, 64, 128, 256};
  if (m == 1) {
    for (int64_t output_block : kBlockSizes) {
      if (output_block > n || n % output_block != 0) {
        continue;
      }
      for (int64_t block_k : {32, 64, 128, 256}) {
        if (block_k > k || k % block_k != 0) {
          continue;
        }
        for (int64_t num_warps : {1, 2, 4, 8}) {
          const int64_t wave_tiles = output_block / 16;
          if (num_warps > wave_tiles || wave_tiles % num_warps != 0) {
            continue;
          }
          configs.push_back(MakeConfig(
              /*block_m=*/16, output_block, block_k, num_warps,
              FlyGemmConfig::FLY_MFMA_16X16X16));
        }
      }
    }
    return configs;
  }
  if (n == 1) {
    for (int64_t output_block : kBlockSizes) {
      if (output_block > m || m % output_block != 0) {
        continue;
      }
      for (int64_t num_warps : {1, 2, 4, 8}) {
        configs.push_back(MakeConfig(
            output_block, /*block_n=*/16, /*block_k=*/32, num_warps,
            FlyGemmConfig::FLY_MFMA_16X16X16));
      }
    }
    return configs;
  }

  auto add_gemm_configs = [&](int64_t block_m, int64_t block_n,
                              int64_t block_k, int64_t num_warps,
                              FlyGemmConfig::MfmaAtom mfma_atom) {
    auto add_variants = [&](bool prefetch_rhs, bool stage_output) {
      configs.push_back(MakeConfig(
          block_m, block_n, block_k, num_warps, mfma_atom, prefetch_rhs,
          stage_output, /*waves_per_eu=*/0,
          /*schedule_instructions=*/false));
      const bool tune_schedule =
          k >= 1024 && block_m * block_n >= 16 * 1024 && !stage_output;
      if (!tune_schedule) {
        return;
      }
      // The AMDGPU instruction scheduler can interleave the next tile's VMEM
      // loads with LDS reads and MFMAs when explicit scheduling groups are
      // present. Keep the unscheduled kernel as a baseline because the best
      // schedule depends on tile shape and register pressure.
      configs.push_back(MakeConfig(
          block_m, block_n, block_k, num_warps, mfma_atom, prefetch_rhs,
          stage_output, /*waves_per_eu=*/0,
          /*schedule_instructions=*/true));
      // FlyDSL kernels tune occupancy as well as instruction order. The common
      // MLIR kernel compiler applies this as amdgpu-waves-per-eu.
      configs.push_back(MakeConfig(
          block_m, block_n, block_k, num_warps, mfma_atom, prefetch_rhs,
          stage_output, /*waves_per_eu=*/2,
          /*schedule_instructions=*/true));
      // Four waves/EU improved the best unscheduled high-occupancy tile by
      // about 7% on MI300X, while one wave/EU consistently regressed.
      configs.push_back(MakeConfig(
          block_m, block_n, block_k, num_warps, mfma_atom, prefetch_rhs,
          stage_output, /*waves_per_eu=*/4,
          /*schedule_instructions=*/false));
    };
    add_variants(/*prefetch_rhs=*/false, /*stage_output=*/false);
    // FlyDSL GEMMs transpose accumulator fragments through LDS so the global
    // epilogue can use packed stores. Keep it as an autotune dimension because
    // reserving the whole output tile in LDS can reduce occupancy.
    constexpr int64_t kMinStagedOutputElements = 8 * 1024;
    constexpr int64_t kMaxStagedOutputElements = 32 * 1024;
    const bool can_stage_output =
        dot->shape().element_type() == BF16 &&
        block_m * block_n >= kMinStagedOutputElements &&
        block_m * block_n <= kMaxStagedOutputElements;
    if (can_stage_output) {
      add_variants(/*prefetch_rhs=*/false, /*stage_output=*/true);
    }
    // Register-pipelining the RHS can overlap VMEM latency with MFMA, but its
    // extra live vectors are not universally profitable. Preserve both
    // variants so autotuning makes that tradeoff for the actual shape.
    if (k >= 1024 && block_k <= 64) {
      add_variants(/*prefetch_rhs=*/true, /*stage_output=*/false);
      if (can_stage_output) {
        add_variants(/*prefetch_rhs=*/true, /*stage_output=*/true);
      }
    }
  };

  auto add_staged_rhs_configs =
      [&](int64_t block_m, int64_t block_n, int64_t block_k,
          std::vector<int64_t> num_warps_values) {
        constexpr int64_t kMaxLdsBytes = 64 * 1024;
        const int64_t staged_lds_bytes =
            2 * (block_m + block_n) * block_k * sizeof(uint16_t);
        if (!rhs_k_contiguous || block_m > m || m % block_m != 0 ||
            block_n > n || n % block_n != 0 || block_k > k ||
            k % block_k != 0 || staged_lds_bytes > kMaxLdsBytes) {
          return;
        }
        for (int64_t num_warps : num_warps_values) {
          for (bool stage_output : {false, true}) {
            for (bool schedule : {false, true}) {
              configs.push_back(MakeConfig(
                  block_m, block_n, block_k, num_warps,
                  FlyGemmConfig::FLY_MFMA_16X16X16,
                  /*prefetch_rhs=*/false, stage_output,
                  /*waves_per_eu=*/0, /*schedule_instructions=*/schedule,
                  /*stage_rhs=*/true));
            }
          }
        }
      };

  // Wide-N tiles increase RHS reuse without increasing the LHS LDS footprint.
  // Keep these candidates explicit: admitting 512 into the Cartesian search
  // also creates 256x512 and 512x512 kernels whose register allocation is
  // prohibitively expensive during online autotuning.
  if (m % 64 == 0 && n % 512 == 0 && k % 32 == 0 && k >= 1024) {
    configs.push_back(MakeConfig(
        /*block_m=*/64, /*block_n=*/512, /*block_k=*/32, /*num_warps=*/4,
        FlyGemmConfig::FLY_MFMA_16X16X16));
  }
  if (m % 128 == 0 && n % 512 == 0 && k % 32 == 0 && k >= 1024) {
    configs.push_back(MakeConfig(
        /*block_m=*/128, /*block_n=*/512, /*block_k=*/32, /*num_warps=*/8,
        FlyGemmConfig::FLY_MFMA_16X16X16,
        /*prefetch_rhs=*/true, /*stage_output=*/false,
        /*waves_per_eu=*/0, /*schedule_instructions=*/true));
  }
  // Apply FlyDSL's gfx942 A+B LDS pipeline to its native square tile and the
  // small tile selected by Triton. Both geometries fit a two-stage pipeline
  // in the MI300X 64 KiB LDS allocation.
  if (k >= 1024) {
    add_staged_rhs_configs(/*block_m=*/64, /*block_n=*/32, /*block_k=*/128,
                           /*num_warps_values=*/{2, 4});
    add_staged_rhs_configs(/*block_m=*/128, /*block_n=*/128, /*block_k=*/64,
                           /*num_warps_values=*/{4, 8, 16});
    if (rhs_k_contiguous && m % 128 == 0 && n % 128 == 0 && k % 64 == 0) {
      for (int64_t num_warps : {4, 8, 16}) {
        for (bool stage_output : {false, true}) {
          for (bool schedule : {false, true}) {
            configs.push_back(MakeConfig(
                /*block_m=*/128, /*block_n=*/128, /*block_k=*/64, num_warps,
                FlyGemmConfig::FLY_MFMA_16X16X16,
                /*prefetch_rhs=*/true, stage_output,
                /*waves_per_eu=*/0, /*schedule_instructions=*/schedule,
                /*stage_rhs=*/false, /*async_lhs=*/true));
          }
        }
      }
    }
  }

  for (int64_t block_m : kBlockSizes) {
    if ((m == 1 && block_m != 16) ||
        (m != 1 && (block_m > m || m % block_m != 0))) {
      continue;
    }
    for (int64_t block_n : kBlockSizes) {
      if ((n == 1 && block_n != 16) ||
          (n != 1 && (block_n > n || n % block_n != 0))) {
        continue;
      }
      const int64_t wave_tiles = (block_m / 16) * (block_n / 16);
      for (int64_t block_k : {32, 64, 128, 256}) {
        // The emitter double-buffers the lhs tile in LDS as BF16.
        constexpr int64_t kMaxLdsBytes = 64 * 1024;
        const int64_t lhs_lds_bytes =
            2 * block_m * block_k * sizeof(uint16_t);
        if (block_k > k || k % block_k != 0 ||
            lhs_lds_bytes > kMaxLdsBytes) {
          continue;
        }
        for (int64_t num_warps : {1, 2, 4, 8}) {
          if (num_warps <= wave_tiles && wave_tiles % num_warps == 0) {
            add_gemm_configs(block_m, block_n, block_k, num_warps,
                             FlyGemmConfig::FLY_MFMA_16X16X16);
          }
          const int64_t mfma32_tiles =
              (block_m / 32) * (block_n / 32);
          if (block_m % 32 == 0 && block_n % 32 == 0 &&
              num_warps <= mfma32_tiles &&
              mfma32_tiles % num_warps == 0) {
            add_gemm_configs(block_m, block_n, block_k, num_warps,
                             FlyGemmConfig::FLY_MFMA_32X32X8);
          }
        }
      }
    }
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> FlyBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                   GetSupportedConfigs(instr));
  if (configs.empty()) {
    return absl::InvalidArgumentError(
        "FlyBackend has no supported configuration for this instruction.");
  }
  return std::move(configs.back());
}

absl::Status FlyBackend::ApplyConfig(HloInstruction& instr,
                                     const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "FlyBackend does not support this instruction.");
  }
  if (!config.has_fly()) {
    return absl::InvalidArgumentError(
        "Expected FlyGemmConfig for FlyBackend.");
  }
  const FlyGemmConfig& fly_config = config.fly();

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig* fusion_config =
      gpu_config.mutable_fusion_backend_config();
  HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kDot);
  const bool is_gemv =
      dot->shape().dimensions(0) == 1 || dot->shape().dimensions(1) == 1;
  fusion_config->set_kind(is_gemv ? kFlyGemvFusionKind
                                  : kFlyGemmFusionKind);
  *fusion_config->mutable_fly_gemm_config() = fly_config;

  BlockLevelFusionConfig* block_config =
      fusion_config->mutable_block_level_fusion_config();
  block_config->Clear();
  Tile* output_tile = block_config->add_output_tiles();
  output_tile->add_sizes(fly_config.block_m());
  output_tile->add_sizes(fly_config.block_n());
  block_config->set_num_warps(fly_config.num_warps());
  block_config->set_num_ctas(1);
  block_config->set_num_stages(1);
  block_config->set_waves_per_eu(fly_config.waves_per_eu());
  fusion_config->clear_triton_gemm_config();

  Tile contraction_tile;
  contraction_tile.add_sizes(fly_config.block_k());
  RETURN_IF_ERROR(dot->set_backend_config(contraction_tile));
  return instr.set_backend_config(gpu_config);
}

}  // namespace xla::gpu
