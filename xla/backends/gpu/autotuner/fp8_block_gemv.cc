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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/gpu/codegen/kernels/fp8_block_gemm_cutlass.h"
#include "xla/backends/gpu/codegen/triton/fp8_block_gemv.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/backends/gpu/codegen/kernels/fp8_block_gemv_kernel.h"
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

TileIrFusionConfig* SlotFor(Fp8BlockGemvBackend::Rung rung,
                            BackendConfig& config) {
  switch (rung) {
    case Fp8BlockGemvBackend::Rung::kTriton:
      return config.mutable_fp8_block_gemv();
    case Fp8BlockGemvBackend::Rung::kTileIr:
      return config.mutable_fp8_block_gemv_tile_ir();
    case Fp8BlockGemvBackend::Rung::kCuda:
      return config.mutable_fp8_block_gemv_cuda();
    case Fp8BlockGemvBackend::Rung::kCutlass:
      // The CUTLASS rung's config is a table index, not a tile, so it has no
      // slot here and takes the PackCutlass path instead.
      LOG(FATAL) << "the CUTLASS rung has no tile slot";
  }
}

const TileIrFusionConfig* ConstSlotFor(Fp8BlockGemvBackend::Rung rung,
                                       const BackendConfig& config) {
  switch (rung) {
    case Fp8BlockGemvBackend::Rung::kTriton:
      return config.has_fp8_block_gemv() ? &config.fp8_block_gemv() : nullptr;
    case Fp8BlockGemvBackend::Rung::kTileIr:
      return config.has_fp8_block_gemv_tile_ir()
                 ? &config.fp8_block_gemv_tile_ir()
                 : nullptr;
    case Fp8BlockGemvBackend::Rung::kCuda:
      return config.has_fp8_block_gemv_cuda() ? &config.fp8_block_gemv_cuda()
                                              : nullptr;
    case Fp8BlockGemvBackend::Rung::kCutlass:
      return nullptr;
  }
}

std::unique_ptr<BackendConfig> Pack(Fp8BlockGemvBackend::Rung rung,
                                    int64_t block_m, int64_t block_n,
                                    int64_t block_k, int num_warps,
                                    int num_stages, bool tma = false,
                                    bool warp_specialization = false) {
  auto config = std::make_unique<BackendConfig>();
  TileIrFusionConfig& fp8 = *SlotFor(rung, *config);
  xla::xtile::BlockLevelFusionConfig& block = *fp8.mutable_block_level_fusion_config();
  xla::xtile::Tile& tile = *block.add_output_tiles();
  tile.add_sizes(block_m);
  tile.add_sizes(block_n);
  block.set_num_warps(num_warps);
  block.set_num_stages(num_stages);
  block.set_num_ctas(1);
  block.set_is_tma_allowed(tma);
  block.set_is_warp_specialization_allowed(warp_specialization);
  fp8.set_contracting_tile_size(block_k);
  return config;
}

std::unique_ptr<BackendConfig> PackCutlass(int config_index) {
  auto config = std::make_unique<BackendConfig>();
  config->mutable_fp8_block_gemm_cutlass()->set_config_index(config_index);
  return config;
}

}  // namespace

int Fp8BlockGemvBackend::CutlassCcMajor() const {
  const se::CudaComputeCapability* cc =
      target_config().device_description.gpu_compute_capability()
          .cuda_compute_capability();
  return cc == nullptr ? 0 : cc->major;
}

bool Fp8BlockGemvBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) return false;
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) return false;
  absl::string_view kind = gpu_config->fusion_backend_config().kind();
  if (kind != kTritonNestedGemmFusionKind && kind != kTileIrFusionKind &&
      kind != kFp8BlockGemvCudaFusionKind &&
      kind != kFp8BlockGemmCutlassFusionKind) {
    return false;
  }
  if (rung_ == Rung::kTileIr &&
      !debug_options().xla_gpu_experimental_scaled_dot_with_tile_ir()) {
    return false;
  }
  std::optional<Fp8BlockGemvSpec> spec =
      MatchFp8BlockGemv(*Cast<HloFusionInstruction>(&instr));
  if (!spec.has_value()) return false;
  if (rung_ == Rung::kCutlass) {
    // The vendored collective is SM100-family and reads a real activation
    // scale; a W8A16 fusion has no such buffer.
    if (!HasCutlassBlockGemm(
            target_config().device_description.gpu_compute_capability())) {
      return false;
    }
    if (!spec->w8a8) return false;
  }
  return true;
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

  if (rung_ == Rung::kCutlass) {
    // The table is fixed at build time, so the search is over indices; CanRun
    // is the kernel's own legality test (which Blackwell family the config was
    // built for, and the scale grids).
    for (int i = 0; i < kernel::Fp8BlockGemmCutlassNumConfigs(); ++i) {
      if (!kernel::Fp8BlockGemmCutlassCanRun(i, CutlassCcMajor(), batch, n, k)) {
        continue;
      }
      configs.push_back(PackCutlass(i));
    }
    return configs;
  }

  const bool single_row = batch == 1;
  const bool is_prefill = batch > 16;
  const int64_t min_block_k = single_row ? kScaleBlock : 2 * kScaleBlock;
  const int max_warps = single_row ? 16 : 8;
  const bool tile_ir = rung_ == Rung::kTileIr;
  // The Tile IR lowering has no fp8 dot and no wide-row tiling here.
  if (tile_ir && (is_prefill || spec->w8a8)) return configs;

  if (rung_ == Rung::kCuda) {
    // The hand-written kernel reads a bf16 activation row.
    if (!single_row || k % 16 != 0 || spec->w8a8) return configs;
    for (int num_warps : {2, 4, 8, 16}) {
      for (int rows_per_warp : {2, 4}) {
        const int64_t rows_per_block = num_warps * rows_per_warp;
        if (kScaleBlock % rows_per_block != 0) continue;
        if (n % rows_per_block != 0) continue;
        for (int unroll : {4, 8}) {
          if (!kernel::IsSupportedFp8BlockGemvKernelConfig(
                  {num_warps, rows_per_warp, unroll})) {
            continue;
          }
          configs.push_back(Pack(rung_, batch, rows_per_block, kScaleBlock,
                                 num_warps, unroll));  // batch == 1 here
        }
      }
    }
    return configs;
  }

  llvm::SmallVector<int64_t, 5> block_ms;
  if (single_row) {
    block_ms.push_back(1);
  } else {
    for (int64_t candidate : {16, 32, 64, 128, 256}) {
      if (candidate <= batch && batch % candidate == 0 &&
          (is_prefill || candidate <= 128)) {
        block_ms.push_back(candidate);
      }
    }
    // No fallback tile when nothing divides the batch: xtile.extract does not
    // mask, so a tile the batch does not divide reads past the activation, and
    // this rung has no reference check to catch it. A batch like that is the
    // CUTLASS rung's -- it tiles an arbitrary M -- and the arm only claims one
    // where that rung exists.
    if (block_ms.empty()) return configs;
  }
  if (is_prefill) {
    // Rows below 64 only when nothing wider divides the batch.
    llvm::SmallVector<int64_t, 5> wide;
    for (int64_t block_m : block_ms) {
      if (block_m >= 64) wide.push_back(block_m);
    }
    if (!wide.empty()) block_ms = wide;
  }

  // Operand tiles per pipeline stage must fit shared memory: the activation
  // tile in its own width and the fp8 weight tile.
  const int64_t smem =
      target_config().device_description.shared_memory_per_block_optin();
  const int64_t act_bytes = spec->w8a8 ? 1 : 2;
  auto fits = [&](int64_t block_m, int64_t block_n, int64_t block_k,
                  int num_stages) {
    const int64_t per_stage =
        block_m * block_k * act_bytes + block_n * block_k;
    return num_stages * per_stage <= smem;
  };

  // Prefill tiles are shared-memory bound at bf16 activations: one scale
  // block deep leaves room for four stages where two blocks leave two.
  const int64_t max_block_k = is_prefill ? 512 : 2048;
  const int64_t min_prefill_block_k = kScaleBlock;
  for (int64_t block_m : block_ms) {
    for (int64_t block_n = is_prefill ? 64 : 4; block_n <= kScaleBlock;
         block_n *= 2) {
      if (n % block_n != 0 || kScaleBlock % block_n != 0) continue;
      for (int64_t block_k = is_prefill ? min_prefill_block_k : min_block_k;
           block_k <= max_block_k; block_k *= 2) {
        if (k % block_k != 0) continue;
        if (tile_ir) {
          configs.push_back(Pack(rung_, block_m, block_n, block_k,
                                 /*num_warps=*/4, /*num_stages=*/1));
          continue;
        }
        for (int num_warps = is_prefill ? 4 : 2; num_warps <= max_warps;
             num_warps *= 2) {
          for (int num_stages = 2; num_stages <= 6; ++num_stages) {
            if (!fits(block_m, block_n, block_k, num_stages)) break;
            configs.push_back(Pack(rung_, block_m, block_n, block_k, num_warps,
                                   num_stages));
            // TMA on tiles that reach the 128-row MMA, head to head with the
            // plain load, as the generic rung offers it.
            if (is_prefill && block_m >= 128) {
              configs.push_back(Pack(rung_, block_m, block_n, block_k,
                                     num_warps, num_stages, /*tma=*/true));
            }
          }
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
  if (rung_ == Rung::kCutlass) {
    std::optional<Fp8BlockGemvSpec> spec =
        MatchFp8BlockGemv(*Cast<HloFusionInstruction>(&instr));
    for (int i = 0; i < kernel::Fp8BlockGemmCutlassNumConfigs(); ++i) {
      if (kernel::Fp8BlockGemmCutlassCanRun(i, CutlassCcMajor(), spec->batch,
                                            spec->n, spec->k)) {
        return PackCutlass(i);
      }
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "no CUTLASS block gemm config runs m=", spec->batch, " n=", spec->n,
        " k=", spec->k));
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
    if (block_k <= 0) {
      if (std::optional<Fp8BlockGemvConfig> seed = Fp8BlockGemvConfigFor(
              *Cast<HloScaledDotInstruction>(dot),
              target_config().device_description.gpu_compute_capability());
          seed.has_value()) {
        block_k = seed->block_k;
      }
    }
  }
  if (block_k <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "no contracting tile for ", instr.name(),
        " and the emitter would not choose one"));
  }
  auto config = std::make_unique<BackendConfig>();
  TileIrFusionConfig& fp8 = *SlotFor(rung_, *config);
  *fp8.mutable_block_level_fusion_config() =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  fp8.set_contracting_tile_size(block_k);

  if (rung_ == Rung::kCuda) {
    xla::xtile::BlockLevelFusionConfig& block = *fp8.mutable_block_level_fusion_config();
    const int64_t rows = block.output_tiles_size() > 0 &&
                                 block.output_tiles(0).sizes_size() > 1
                             ? block.output_tiles(0).sizes(1)
                             : 0;
    const int warps = block.num_warps();
    const bool ok =
        warps > 0 && rows > 0 && rows % warps == 0 &&
        kernel::IsSupportedFp8BlockGemvKernelConfig(
            {warps, static_cast<int>(rows / warps), block.num_stages()});
    if (!ok) {
      block.clear_output_tiles();
      xla::xtile::Tile& tile = *block.add_output_tiles();
      tile.add_sizes(1);
      tile.add_sizes(16);
      block.set_num_warps(8);
      block.set_num_stages(8);
    }
  }
  return config;
}

absl::Status Fp8BlockGemvBackend::ApplyConfig(HloInstruction& instr,
                                              const BackendConfig& config) {
  if (rung_ == Rung::kCutlass) {
    if (!config.has_fp8_block_gemm_cutlass()) {
      return absl::InvalidArgumentError(
          "Expected an fp8_block_gemm_cutlass config for the CUTLASS rung.");
    }
    if (instr.opcode() != HloOpcode::kFusion) {
      return absl::InvalidArgumentError("expected a fusion");
    }
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instr.backend_config<GpuBackendConfig>());
    FusionBackendConfig& backend_config =
        *gpu_config.mutable_fusion_backend_config();
    backend_config.set_kind(std::string(kFp8BlockGemmCutlassFusionKind));
    backend_config.clear_triton_gemm_config();
    backend_config.clear_block_level_fusion_config();
    backend_config.mutable_fp8_block_gemm_cutlass_config()->set_config_index(
        config.fp8_block_gemm_cutlass().config_index());
    ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
    instr.set_fusion_kind(HloInstruction::FusionKind::kCustom);
    return absl::OkStatus();
  }

  const bool tile_ir = rung_ == Rung::kTileIr;
  const TileIrFusionConfig* slot = ConstSlotFor(rung_, config);
  if (slot == nullptr) {
    return absl::InvalidArgumentError(
        "Expected an fp8_block_gemv config for Fp8BlockGemvBackend.");
  }
  const TileIrFusionConfig& fp8 = *slot;
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
  backend_config.set_kind(std::string(
      rung_ == Rung::kCuda
          ? kFp8BlockGemvCudaFusionKind
          : (tile_ir ? kTileIrFusionKind : kTritonNestedGemmFusionKind)));
  backend_config.clear_triton_gemm_config();
  *backend_config.mutable_block_level_fusion_config() =
      fp8.block_level_fusion_config();
  ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  instr.set_fusion_kind(HloInstruction::FusionKind::kCustom);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
