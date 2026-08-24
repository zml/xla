/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/autotuner/triton.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/triton/cost_model_config_optimization.h"
#include "xla/backends/gpu/autotuner/triton/dot_search_space.h"
#include "xla/backends/gpu/autotuner/triton/triton_configs.h"
#include "xla/backends/gpu/transforms/convert_triton_gemm_config.h"
#include "xla/backends/gpu/transforms/fusion_wrapper.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "triton/Version.h"

namespace xla {
namespace gpu {

namespace {
std::vector<TritonGemmConfig> GetDefaultTritonConfigs(
    se::GpuComputeCapability compute_capability) {
  if (compute_capability.IsRocm()) {
    const auto* rocm_cc = compute_capability.rocm_compute_capability();
    if (rocm_cc->gfx9_mi300()) {
      return GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI300);
    }
    if (rocm_cc->gfx9_mi350()) {
      return GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI350);
    }
    return GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultRocm);
  }

  CHECK(compute_capability.IsCuda());
  auto* cuda_compute_capability = compute_capability.cuda_compute_capability();
  std::vector<TritonGemmConfig> configs;

  if (cuda_compute_capability->IsBlackwell()) {
    // SM 10.0 (datacenter: B200, B100)
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwell);
  } else if (cuda_compute_capability->IsAtLeastBlackwell()) {
    // SM 11.0+ / 12.0+ (consumer: RTX 5090, etc.)
    configs =
        GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwellConsumer);
  } else if (cuda_compute_capability->IsHopper()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kHopper);
  } else if (cuda_compute_capability->IsAmpere()) {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kAmpere);
  } else {
    configs = GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultCuda);
  }

  return configs;
}

bool IsWarpSpecializationAvailable(
    se::GpuComputeCapability compute_capability) {
  return compute_capability.IsCuda() &&
         compute_capability.cuda_compute_capability()->IsAtLeastBlackwell();
}

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<BackendConfig>> overridden_configs,
      GetOverriddenConfigs(&instr));
  if (!overridden_configs.empty()) {
    return overridden_configs;
  }

  const HloInstruction* dot_instr = hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kDot);
  if (dot_instr != nullptr) {
    return GetSupportedConfigsForDot(dot_instr);
  }
  const HloInstruction* scaled_dot_instr =
      hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
  if (scaled_dot_instr != nullptr) {
    return GetSupportedConfigsForScaledDot(scaled_dot_instr);
  }
  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigsForDot(const HloInstruction* instr) {
  const HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
  TritonDotFusionSearchSpace search_space(target_config().device_description,
                                          dot);
  bool autotune_warp_specialization =
      debug_options()
          .xla_gpu_experimental_enable_triton_warp_specialization() &&
      IsWarpSpecializationAvailable(
          target_config().device_description.gpu_compute_capability());

  std::vector<std::unique_ptr<BackendConfig>> configs;
  VLOG(1) << "Generating configs from search space: "
          << search_space.ToString();
  // We don't need to consider small_dot here. The new search space will
  // already generate a unique config for small problems.
  std::vector<TritonGemmConfig> gemm_configs = search_space.GenerateConfigs(
      /*autotune_warp_specialization=*/autotune_warp_specialization);

  if (!debug_options().xla_gpu_exhaustive_tiling_search()) {
    VLOG(1) << "Restricting configs to the default set.";
    std::vector<TritonGemmConfig> all_configs = gemm_configs;
    gemm_configs = search_space.OptimizeConfigSet(
        gemm_configs, /*hints=*/GetDefaultTritonConfigs(
            target_config().device_description.gpu_compute_capability()));

    if (!debug_options()
             .xla_gpu_experimental_cost_model_gemm_tiling_options()
             .empty()) {
      ASSIGN_OR_RETURN(gemm_configs, OptimizeConfigsWithCostModel(
                                         dot, all_configs, gemm_configs,
                                         target_config().device_description,
                                         debug_options(), mlir_context_));
    }
  }
  configs.reserve(gemm_configs.size());
  for (const auto& gemm_config : gemm_configs) {
    auto config = std::make_unique<BackendConfig>();
    *config->mutable_triton() = gemm_config.ToProto();
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetSupportedConfigsForScaledDot(const HloInstruction* instr) {
  // The ROCm Triton backend does not support mixed FP4/FP8 scaled-dot inputs.
  const auto& gpu_cc =
      target_config().device_description.gpu_compute_capability();
  if (gpu_cc.IsRocm()) {
    PrimitiveType lhs_type = instr->operand(0)->shape().element_type();
    PrimitiveType rhs_type = instr->operand(1)->shape().element_type();
    auto is_fp4 = [](PrimitiveType t) { return t == F4E2M1FN; };
    auto is_fp8 = [](PrimitiveType t) { return t == F8E4M3FN || t == F8E5M2; };
    if ((is_fp4(lhs_type) && is_fp8(rhs_type)) ||
        (is_fp8(lhs_type) && is_fp4(rhs_type))) {
      return std::vector<std::unique_ptr<BackendConfig>>();
    }
  }

  std::vector<std::unique_ptr<BackendConfig>> configs;

  const bool exhaustive_search =
      debug_options().xla_gpu_exhaustive_tiling_search();

  const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
  const Shape& lhs_shape = instr->operand(0)->shape();
  int64_t m = 1;
  for (int64_t d = 0; d < lhs_shape.dimensions().size(); ++d) {
    if (!absl::c_linear_search(dnums.lhs_contracting_dimensions(), d) &&
        !absl::c_linear_search(dnums.lhs_batch_dimensions(), d)) {
      m *= lhs_shape.dimensions(d);
    }
  }
  const bool is_thin_m = m < 128;
  // sm_100 / sm_103: the architectures where a thin-M scaled dot only reaches
  // the block-scaled tensor core by padding its tile out to 128 rows.
  const se::CudaComputeCapability* cuda_cc = gpu_cc.cuda_compute_capability();
  const bool is_tcgen05 =
      cuda_cc != nullptr &&
      cuda_cc->major == se::CudaComputeCapability::kBlackwell;

  // Thin-M decode: search block_m from mma.sync m16. The ceiling stays at 128
  // even though M is smaller, because on sm_100/sm_103 that is the difference
  // between a block-scaled tensor-core dot and no block-scaled dot at all.
  // Triton's tcgen05 pattern (ScaledBlockedToMMAv5 in AccelerateMatmul.cpp)
  // declines any tile with retShapePerCTA[0] < 128, and the warp-level pattern
  // that would otherwise catch a thin tile is sm_120-only -- it emits
  // mma.sync mxf4nvf4 and builds an SM120-specific scale layout. So a decode
  // dot capped at block_m=64 matches neither and falls back to
  // DecomposeScaledBlocked, which upcasts NVFP4 to bf16 and reads four times
  // the weight bytes. Padding a one-row tile out to 128 rows costs almost
  // nothing at bs1, where the weights dominate and are read once either way.
  // Offer both and let the autotuner settle it: on sm_120 the thin tiles still
  // win, because there the warp-level path handles them natively.
  const int min_block_m = is_thin_m ? 16 : 128;
  const int max_block_m = is_thin_m ? 128 : 256;

  for (int block_m = min_block_m; block_m <= max_block_m; block_m *= 2) {
    for (int block_n = 16; block_n <= 256; block_n *= 2) {
      // 512 is the measured sweet spot for NVFP4 decode: on the widest
      // projection the autotuner sees k128=68us, k256=56us, k512=43us, and
      // k1024 back up at 36-48us, so the ceiling stays where the curve turns.
      for (int block_k = 128; block_k <= 512; block_k *= 2) {
        // Decode projections are bandwidth-starved rather than compute-bound:
        // with M small the tile grid is one row deep, so a narrow-N dot such as
        // [6656,19968] launches ~104 blocks against 152 SMs and cannot fill the
        // machine, let alone keep enough loads in flight to reach HBM speed.
        // Pipeline depth is the lever that adds outstanding loads without
        // changing the tiling, so search deeper than the 4 stages that were
        // enough when the ceiling was a 5090-class memory system. Configs that
        // do not fit in shared memory fail to compile and are simply dropped.
        // The same argument bounds the search from below on the tcgen05
        // architectures, and there it is not a preference but a trap. A
        // pipeline shallower than three stages leaves nothing to overlap the
        // loads with, so it cannot win a bandwidth-bound decode dot -- and on
        // every projection in Muse-Glimmer-30B-NVFP4 it does not: the best
        // stages<=2 config trails the overall best by 12.4%, 13.6%, 4.5% and
        // 22.1%. Those margins sit inside the autotuner's run-to-run noise, so
        // offering the shallow configs does not give the autotuner a choice, it
        // gives it a coin flip. A batch-size=1 server lost that flip on two of
        // the four projections -- qkv and out came out at num_stages=1 and ran
        // 23.0 and 13.6 us against the 14.2 and 9.8 us the same dots get at
        // batch-size 16, which is how bs1 ended up SLOWER than bs16 end to end.
        // Left alone on sm_120: there thin M goes down Triton's warp-level
        // path, not the padded-to-128-rows tcgen05 one, and none of this was
        // measured against it.
        const int min_stages = (is_thin_m && is_tcgen05) ? 3 : 1;
        const int max_stages = is_thin_m ? 8 : 1;
        const int max_warps = is_thin_m ? 8 : 4;
        for (int num_stages = min_stages; num_stages <= max_stages;
             ++num_stages) {
          for (int num_warps = 4; num_warps <= max_warps; num_warps *= 2) {
            // TODO(b/436988479): fine tune the search space.
            // Registers held per thread. Depends on num_warps -- the old form
            // hardcoded 4 and so mis-scored every 8-warp config.
            const int elements_per_thread =
                (block_m * block_n) / (num_warps * 32);
            // The block_k>=256 clause is dropped for scaled dots. It exists
            // to avoid register pressure, but an NVFP4 dot spills 1.1-1.3 KB
            // whatever tile it picks -- dequantising fp4 with an e4m3 scale per
            // 16 elements needs the registers -- so pruning on predicted spill
            // only removes configs, it does not avoid any. A long contraction
            // like [6656,19968] wants the deeper block_k precisely because it
            // halves the loop trips, and measuring the whole space instead of
            // this heuristic is worth +30% end to end on GB300.
            if (!exhaustive_search && elements_per_thread > 64) {
              VLOG(3) << "Ignoring spill over config: block_m=" << block_m
                      << " block_n=" << block_n << " block_k=" << block_k
                      << " num_warps=" << num_warps;
              continue;
            }
            // A TMA variant is offered alongside the plain one, but only for
            // the tiles that reach the tcgen05 path (block_m >= 128). There it
            // is worth having -- on the widest decode contraction,
            // [6656,19968] at M=16, TMA loads run 37.3us against 40.3us -- and
            // restricting it to those tiles keeps the candidate count from
            // doubling. Below 128 rows Triton is on the warp-level path, whose
            // loads are already narrow enough that TMA has nothing to add.
            // TODO(raph): Triton itself is not healthy on every NVFP4 shape on
            // sm_103. On a 512x2048x2048 scaled dot, 12 of 30 configs fail to
            // legalize ttng.tc_gen5_mma_scaled, and the first one that does
            // launch throws CUDA_ERROR_ILLEGAL_INSTRUCTION -- which poisons the
            // context, so the remaining Triton candidates AND every Tile IR
            // candidate die after it with "Failed to load in-memory CUBIN" and
            // the process cores. A default all-backends compile of that shape
            // aborts. The real model's shapes (e.g. qkv [512,6656]x[8704,6656])
            // are fine, which is why this has not bitten in anger, but it is a
            // live sticky-fault path with no guard in front of it. Find which
            // configs legalize and exclude the rest, the way the scaled-dot
            // tile floors do for Tile IR.
            //
            // Two knobs measured and left alone on GB300: num_ctas=2 (the
            // 2-CTA tcgen05 MMA cuDNN's own kernels use) runs 113us against
            // 17us on the decode qkv dot and crashes some candidates outright,
            // and is_warp_specialization_allowed changes nothing at all --
            // Triton emits the same kernel either way.
            for (bool tma : {false, true}) {
              if (tma && block_m < 128) {
                continue;
              }
              auto config = std::make_unique<BackendConfig>();
              *config->mutable_triton() =
                  TritonGemmConfig(block_m, block_n,
                                   /*block_k=*/block_k, num_stages, num_warps,
                                   /*num_ctas=*/1,
                                   /*is_tma_allowed=*/tma)
                      .ToProto();
              configs.push_back(std::move(config));
            }
          }
        }
      }
    }
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TritonBackend::GetOverriddenConfigs(const HloInstruction* instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  const std::string& override_file =
      debug_options().xla_gpu_gemm_autotuner_override_file();
  if (!override_file.empty()) {
    std::string file_content;
    RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), override_file,
                                          &file_content));
    TritonGemmConfigsProto gemm_configs;
    if (!tsl::protobuf::TextFormat::ParseFromString(file_content,
                                                    &gemm_configs)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Could not parse override file: ", override_file));
    }
    configs.reserve(gemm_configs.config_size());
    for (const auto& gemm_config : gemm_configs.config()) {
      auto config = std::make_unique<BackendConfig>();
      *config->mutable_triton() = gemm_config;
      configs.push_back(std::move(config));
    }
  }
  if (!debug_options().xla_gpu_override_gemm_autotuner().empty()) {
    AutotuneResult::TritonGemmKey gemm_config;
    CHECK(tsl::protobuf::TextFormat::ParseFromString(
        debug_options().xla_gpu_override_gemm_autotuner(), &gemm_config));
    auto config = std::make_unique<BackendConfig>();
    *config->mutable_triton() = gemm_config;
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> TritonBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                   GetSupportedConfigs(instr));

  if (configs.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("TritonBackend has no supported configs for '",
                     instr.name(), "' instruction"));
  }
  return std::move(configs[0]);
}

absl::Status TritonBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "TritonBackend does not support this instruction.");
  }
  if (!config.has_triton()) {
    return absl::InvalidArgumentError(
        "Expected TritonGemmKey config for TritonBackend.");
  }
  const AutotuneResult::TritonGemmKey& triton_config_proto = config.triton();

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();

  backend_config.set_kind(kTritonGemmFusionKind);
  *backend_config.mutable_triton_gemm_config() = triton_config_proto;
  RETURN_IF_ERROR(instr.set_backend_config(gpu_config));

  // FromProto has validation checks, that's why we call it here.
  RETURN_IF_ERROR(TritonGemmConfig::FromProto(triton_config_proto).status());
  if (triton_config_proto.split_k() > 1) {
    return absl::InvalidArgumentError(
        "TritonBackend no longer supports split-k (split_k > 1).");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<HloModule>> TritonBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  auto gpu_device_info = target_config().device_description;
  for (PrimitiveType type :
       {BF16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ}) {
    GpuFloatSupport float_support(gpu_device_info.gpu_compute_capability(),
                                  type);
    FloatNormalization float_normalization(&float_support);
    RETURN_IF_ERROR(float_normalization.Run(hlo_module.get()).status());
  }

  HloCostAnalysis::Options priority_fusion_options;
  priority_fusion_options.count_multiple_input_accesses = true;
  PriorityFusion priority_fusion(
      /*thread_pool=*/nullptr, gpu_device_info, alias_info_,
      priority_fusion_options, mlir_context_);
  RETURN_IF_ERROR(priority_fusion.Run(hlo_module.get()).status());

  // If the priority fusion pass above skipped some instructions, turn them
  // into fusions.
  FusionWrapper fusion_wrapper(gpu_device_info);
  RETURN_IF_ERROR(fusion_wrapper.Run(hlo_module.get()).status());
  ConvertTritonGemmConfig convert_triton_gemm_config(gpu_device_info,
                                                     mlir_context_);
  RETURN_IF_ERROR(convert_triton_gemm_config.Run(hlo_module.get()).status());
  return hlo_module;
}

bool TritonBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  const FusionBackendConfig& backend_config =
      gpu_config->fusion_backend_config();

  if (hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kScaledDot) !=
          nullptr &&
      !instr.GetModule()
           ->config()
           .debug_options()
           .xla_gpu_experimental_scaled_dot_with_triton()) {
    return false;
  }

  // TODO: b/487920266 - sometimes we create fusions that can't be tiled.
  // Bail out here if that's the case.
  if (IsGemmFusionAutotuneKind(backend_config.kind())) {
    auto fusion = Cast<HloFusionInstruction>(&instr);
    std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
        HloFusionAdaptor::ForInstruction(fusion);
    if (instr.GetModule()
            ->config()
            .debug_options()
            .xla_gpu_experimental_enable_tiling_propagation()) {
      auto ts =
          experimental::TilingSpace::Create(*fusion_adaptor, mlir_context_);
      if (!ts.ok()) {
        VLOG(1) << "Failed to create tiling space: " << ts.status().message();
        return false;
      }
      auto tiled_computation_or = experimental::TiledHloComputation::Tile(
          *fusion_adaptor, std::move(ts.value()));
      if (!tiled_computation_or.ok()) {
        VLOG(1) << "Fusion is not tileable with experimental tiling: "
                << tiled_computation_or.status().message();
        return false;
      }
      // We don't have concrete tile sizes here and don't validate Triton
      // constraints here.
      return true;
    }

    auto device_info = target_config().device_description;
    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeFusion(
            *fusion_adaptor, mlir_context_,
            TritonEmitterConstraints::GetBuilder(device_info));
    if (const auto* fusion_decision =
            std::get_if<FusionDecision>(&analysis_or_error)) {
      VLOG(1) << "Fusion not tileable: " << fusion_decision->Explain();
      return false;
    }
    return true;
  }
  return backend_config.kind() == kCuDnnFusionKind ||
         backend_config.kind() == kCustomFusionKind;
}

std::string TritonBackend::version() const { return TRITON_VERSION; }

}  // namespace gpu
}  // namespace xla
