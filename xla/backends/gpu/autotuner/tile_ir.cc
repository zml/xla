#include "xla/backends/gpu/autotuner/tile_ir.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/MathExtras.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/transforms/convert_triton_gemm_config.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// NVFP4 scale block is 16 elements along K.
constexpr int64_t kScaleBlockSize = 16;

using OutputTile = std::array<int64_t, 2>;

// What tileiras can build, per architecture. This is deliberately NOT one
// shared space: every bound below is an observed property of one tileiras
// version on one target, and the two targets measured so far disagree about
// most of them. A bound proven on GB300 is not evidence about anything else.
struct TileIrLimits {
  absl::Span<const OutputTile> output_tiles;
  absl::Span<const int64_t> contracting_tiles;
  OutputTile default_tile;
  int64_t default_contracting_tile;
  // Floors the search clamps candidate tiles up to.
  int64_t min_block_m;
  int64_t min_contracting_tile;
  // Decline the fusion outright below this many output rows / contracting
  // elements, rather than spend tileiras time on a tile that cannot win. Zero
  // means never decline on that axis.
  int64_t decline_below_rows;
  int64_t decline_below_contracting;
  int64_t max_operand_tile_bytes;
};

// sm_120 (RTX 50-series). This is the space the backend shipped with and was
// tuned on, restored verbatim. None of the GB300 bounds below have been
// re-measured here, and the one datapoint that exists points the other way:
// Tile IR was ahead of Triton on NVFP4 decode on a 5090, which needs exactly
// the thin-M tiles that sm_103 has to give up.
//
// TODO(raph): re-measure this space on sm_120 with tileiras 13.3.36. Three
// specific questions, all cheap now that XLA_TILE_IR_FORCE_TILE exists:
//   1. Does the block_k 96/128 misaligned-address fault reproduce here? If it
//      does, this list needs the same >=160 safety floor sm_103 has. If it
//      does not, the bug is sm_103 codegen and should be reported as such.
//   2. Does tileiras emit UTCOMMA below block_m 128 here? On sm_103 it does
//      not, which is the whole reason thin M is hopeless there.
//   3. Do block_k 512 and the 112 KB operand budget help here too? They were
//      worth 25.2 vs 37.6 us on GB300 and are currently left off sm_120
//      because "untested" beats "probably fine" on a path where a bad
//      candidate takes the whole compilation down.
constexpr OutputTile kOutputTilesSm120[] = {
    {16, 32},  {16, 64},   {16, 128},  {16, 256},
    {32, 128}, {64, 128},  {128, 128}, {128, 256}, {256, 128},
};
constexpr int64_t kContractingTilesSm120[] = {128, 256, 384};

// sm_103 (GB300). Measured with tileiras 13.3.36 on an NVFP4 512x2048x2048
// scaled dot, re-verified 2026-08-24 by forcing each tile in its own process:
//
//   block_k  96 / 128    THE ONLY CRASH. Kernel faults with
//                        CUDA_ERROR_MISALIGNED_ADDRESS, and the fault is
//                        sticky -- afterwards even cuMemHostRegister fails and
//                        the process cores, so it takes the whole compilation
//                        down rather than losing its own measurement. 32, 48
//                        and 64 are safe, and so is everything >=160, so the
//                        precise safety floor is 160; the 256 below is a
//                        performance choice stacked on top of it.
//   block_m  16 / 32 / 64  Correct, not a crash -- XLA's own output clustering
//                        puts them with the cuBLASLt reference, and 16/32/64
//                        are byte-identical to each other. They are simply
//                        ~1500x slow (20-25 ms against cuBLASLt's 15 us),
//                        because tileiras emits ZERO UTCOMMA below 128 rows:
//                        it cannot reach the tcgen05 block-scaled MMA at all
//                        and falls back to a software HADD2/F2FP loop.
//   block_m  512, block_n 512, 256x256   tileiras runs past the 2-minute
//                        budget and is killed. Compile time, not correctness.
//
// The contracting tile is the strongest lever there is: on the Muse-Glimmer
// qkv dot, 256 -> 384 -> 512 runs 39.3 -> 30.1 -> 25.0 us, and 640 turns over
// again (45.7 us). The 112 KB operand budget exists to admit exactly the 512
// combinations, which a 64 KB cap silently dropped.
//
// TODO(raph): the block_k 96/128 fault is a tileiras codegen bug worth
// reporting upstream, and the SASS says precisely what it is. The two faulting
// widths are exactly the ones that emit the 8-byte `LDGSTS.E.64`; safe narrow
// tiles emit the 4-byte `LDGSTS.E` and safe wide tiles emit none at all. It is
// an alignment bug in the wide cp.async staging path -- NOT the software-
// upcast fallback an earlier version of this comment claimed. That diagnosis
// was wrong in three ways: block_k 64 also emits LDGSTS, 96/128 do emit
// UTCOMMA.BLOCK16, and F2FP.F16.E2M1 shows up at every width including 512.
// Re-test on the next tileiras and delete the floor if it is fixed.
constexpr OutputTile kOutputTilesSm103[] = {
    {128, 128}, {128, 256}, {256, 128},
};
constexpr int64_t kContractingTilesSm103[] = {256, 384, 512};

// Two different floors, kept apart on purpose so a tileiras fix can retire one
// without touching the other. SAFE is the smallest contracting tile that does
// not hit the misaligned-address codegen (96 and 128 fault; 32/48/64 and
// >=160 do not). PERF is where the measurements actually want to start.
constexpr int64_t kSm103SafeContractingTile = 160;
constexpr int64_t kSm103PerfContractingTile = 256;
static_assert(kSm103PerfContractingTile >= kSm103SafeContractingTile,
              "The performance floor must not reopen the faulting block_k "
              "band; see the LDGSTS.E.64 note above.");
static_assert(kSm103SafeContractingTile % kScaleBlockSize == 0,
              "A contracting tile must be a whole number of scale blocks.");

bool IsBlackwellUltra(const stream_executor::CudaComputeCapability& cc) {
  return cc.major == stream_executor::CudaComputeCapability::kBlackwell &&
         cc.minor == 3;
}

const TileIrLimits& LimitsFor(const stream_executor::CudaComputeCapability& cc) {
  static const TileIrLimits kSm103 = {
      /*output_tiles=*/kOutputTilesSm103,
      /*contracting_tiles=*/kContractingTilesSm103,
      /*default_tile=*/{128, 128},
      /*default_contracting_tile=*/256,
      /*min_block_m=*/128,
      /*min_contracting_tile=*/kSm103PerfContractingTile,
      /*decline_below_rows=*/128,
      /*decline_below_contracting=*/kSm103PerfContractingTile,
      /*max_operand_tile_bytes=*/112 * 1024,
  };
  // Everything that is not sm_103 keeps the original space. MMA m is 16
  // (mma.sync.m16n8k64), so 16 is the floor on both axes, and nothing is
  // declined outright -- that is what shipped, and it is what the sm_120
  // numbers were taken against.
  static const TileIrLimits kDefault = {
      /*output_tiles=*/kOutputTilesSm120,
      /*contracting_tiles=*/kContractingTilesSm120,
      /*default_tile=*/{32, 128},
      /*default_contracting_tile=*/256,
      /*min_block_m=*/16,
      /*min_contracting_tile=*/kScaleBlockSize,
      /*decline_below_rows=*/0,
      /*decline_below_contracting=*/0,
      /*max_operand_tile_bytes=*/64 * 1024,
  };
  return IsBlackwellUltra(cc) ? kSm103 : kDefault;
}

// Only reachable behind GetGemmBounds, which returns nullopt unless
// IsSupported has already established a CUDA target.
const TileIrLimits& LimitsForOrDie(
    const stream_executor::CudaComputeCapability* cc) {
  CHECK(cc != nullptr);
  return LimitsFor(*cc);
}

int64_t ClampBlockM(const TileIrLimits& limits, int64_t block_m,
                    int64_t max_m) {
  return std::max(limits.min_block_m, std::min(block_m, max_m));
}

int64_t OperandTileBytes(int64_t block_m, int64_t block_n, int64_t block_k) {
  return (block_m * block_k + block_k * block_n) * 9 / 16;
}

int64_t ClampContractingTile(const TileIrLimits& limits, int64_t block_k,
                             int64_t contracting_size) {
  if (contracting_size <= 0) {
    return block_k;
  }
  // The floor wins even when it exceeds the whole contraction. That is only
  // reachable on a target whose floor is above kScaleBlockSize, i.e. sm_103 --
  // and there IsBigEnough has already declined any dot with a contraction
  // below the same floor, so the case cannot arise through the normal path.
  // Keep the two numbers equal in TileIrLimits or this becomes reachable.
  return std::max(limits.min_contracting_tile,
                  std::min(block_k, contracting_size));
}

std::vector<int64_t> ContractingTilesForOutputTile(const TileIrLimits& limits,
                                                   int64_t block_m,
                                                   int64_t block_n,
                                                   int64_t contracting_size) {
  std::vector<int64_t> tiles;
  for (int64_t candidate : limits.contracting_tiles) {
    int64_t block_k = ClampContractingTile(limits, candidate, contracting_size);
    if (OperandTileBytes(block_m, block_n, block_k) >
        limits.max_operand_tile_bytes) {
      continue;
    }
    tiles.push_back(block_k);
  }
  return tiles;
}

int64_t ContractingSize(const HloInstruction& dot) {
  const DotDimensionNumbers& dnums = dot.dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions().empty()) {
    return 0;
  }
  const Shape& lhs_shape = dot.operand(0)->shape();
  int64_t dim = dnums.lhs_contracting_dimensions(0);
  if (!lhs_shape.IsArray() ||
      dim >= static_cast<int64_t>(lhs_shape.dimensions().size())) {
    return 0;
  }
  return static_cast<int64_t>(
      llvm::PowerOf2Ceil(static_cast<uint64_t>(lhs_shape.dimensions(dim))));
}

constexpr int kNumWarps = 4;
constexpr int kNumStages = 1;
constexpr int kNumCtas = 1;

std::unique_ptr<BackendConfig> Pack(
    const BlockLevelFusionConfig& block_level_config, int64_t block_k) {
  auto config = std::make_unique<BackendConfig>();
  TileIrFusionConfig& tile_ir_config = *config->mutable_tile_ir();
  *tile_ir_config.mutable_block_level_fusion_config() = block_level_config;
  tile_ir_config.set_contracting_tile_size(block_k);
  return config;
}

const HloInstruction* GetScaledDot(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return nullptr;
  }
  return hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
}

// CanLower before offering configs: unprofiled stamp has no emit fallback.
bool CanLower(const HloInstruction& fusion, const HloInstruction& dot) {
  // mmaf_scaled is same-type only; bf16xfp4 has no cuda_tile spelling.
  if (dot.operand(0)->shape().element_type() !=
      dot.operand(1)->shape().element_type()) {
    return false;
  }
  // One mmaf_scaled per kernel; multi-dot fusions decline.
  return absl::c_count_if(fusion.fused_instructions_computation()->instructions(),
                          HloPredicateIsOp<HloOpcode::kScaledDot>) == 1;
}

// A dot smaller than the target's smallest viable tile has no candidate worth
// building, and tileiras is slow enough that offering one anyway costs the
// whole compilation real time. Both thresholds are zero on targets that have
// not been measured, so this is a no-op there rather than a guess.
bool IsBigEnough(const TileIrLimits& limits, const HloInstruction& fusion,
                 const HloInstruction& dot) {
  const Shape& root_shape =
      fusion.fused_instructions_computation()->root_instruction()->shape();
  if (!root_shape.IsArray() || root_shape.dimensions().size() < 2) {
    return false;
  }
  int64_t rows = root_shape.dimensions(root_shape.dimensions().size() - 2);
  return rows >= limits.decline_below_rows &&
         ContractingSize(dot) >= limits.decline_below_contracting;
}

// XLA_TILE_IR_FORCE_TILE=m,n,k pins the search to a single tile, so a faulting
// kernel can be bisected without running the whole candidate list.
std::optional<std::tuple<int64_t, int64_t, int64_t>> ForcedTile() {
  static const std::optional<std::tuple<int64_t, int64_t, int64_t>> forced = [] {
    std::optional<std::tuple<int64_t, int64_t, int64_t>> result;
    const char* env = std::getenv("XLA_TILE_IR_FORCE_TILE");
    if (env == nullptr) {
      return result;
    }
    std::vector<absl::string_view> parts = absl::StrSplit(env, ',');
    int64_t m, n, k;
    if (parts.size() != 3 || !absl::SimpleAtoi(parts[0], &m) ||
        !absl::SimpleAtoi(parts[1], &n) || !absl::SimpleAtoi(parts[2], &k)) {
      LOG(ERROR) << "XLA_TILE_IR_FORCE_TILE must be m,n,k; got " << env;
      return result;
    }
    result = std::make_tuple(m, n, k);
    return result;
  }();
  return forced;
}

}  // namespace

bool TileIrBackend::IsSupported(const HloInstruction& instr) {
  if (!debug_options().xla_gpu_experimental_scaled_dot_with_tile_ir()) {
    return false;
  }
  const stream_executor::CudaComputeCapability* cc =
      target_config().device_description.gpu_compute_capability()
          .cuda_compute_capability();
  if (cc == nullptr || cc->major < 10) {
    return false;
  }
  const HloInstruction* dot = GetScaledDot(instr);
  return dot != nullptr && CanLower(instr, *dot) &&
         IsBigEnough(LimitsFor(*cc), instr, *dot);
}

std::optional<TileIrBackend::GemmBounds> TileIrBackend::GetGemmBounds(
    const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::nullopt;
  }
  const HloInstruction* dot = GetScaledDot(instr);
  const Shape& root_shape =
      instr.fused_instructions_computation()->root_instruction()->shape();
  if (!root_shape.IsArray() || root_shape.dimensions().size() < 2) {
    return std::nullopt;
  }
  int64_t rank = root_shape.dimensions().size();
  return GemmBounds{
      /*dot=*/dot,
      /*max_m=*/
      static_cast<int64_t>(llvm::PowerOf2Ceil(
          static_cast<uint64_t>(root_shape.dimensions(rank - 2)))),
      /*max_n=*/
      static_cast<int64_t>(llvm::PowerOf2Ceil(
          static_cast<uint64_t>(root_shape.dimensions(rank - 1)))),
      /*contracting_size=*/ContractingSize(*dot),
  };
}

absl::StatusOr<BlockLevelFusionConfig> TileIrBackend::BlockLevelConfigForTile(
    const HloInstruction& dot, int64_t block_m, int64_t block_n,
    int64_t block_k) {
  TritonGemmConfig gemm_config(
      /*block_m=*/static_cast<int>(block_m),
      /*block_n=*/static_cast<int>(block_n),
      /*block_k=*/static_cast<int>(block_k),
      /*num_stages=*/kNumStages, /*num_warps=*/kNumWarps,
      /*num_ctas=*/kNumCtas);
  ASSIGN_OR_RETURN(
      BlockLevelParameters params,
      FindBlockLevelParameters(&dot, gemm_config, mlir_context_,
                               target_config().device_description));
  return params.ToBlockLevelFusionConfig();
}

absl::StatusOr<std::unique_ptr<BackendConfig>> TileIrBackend::ConfigForTile(
    const HloInstruction& dot, int64_t block_m, int64_t block_n,
    int64_t block_k) {
  ASSIGN_OR_RETURN(BlockLevelFusionConfig block_level_config,
                   BlockLevelConfigForTile(dot, block_m, block_n, block_k));
  return Pack(block_level_config, block_k);
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TileIrBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  std::optional<GemmBounds> bounds = GetGemmBounds(instr);
  if (!bounds.has_value()) {
    return configs;
  }

  if (std::optional<std::tuple<int64_t, int64_t, int64_t>> forced = ForcedTile();
      forced.has_value()) {
    auto [block_m, block_n, block_k] = *forced;
    ASSIGN_OR_RETURN(std::unique_ptr<BackendConfig> config,
                     ConfigForTile(*bounds->dot, block_m, block_n, block_k));
    configs.push_back(std::move(config));
    return configs;
  }

  const TileIrLimits& limits = LimitsForOrDie(
      target_config().device_description.gpu_compute_capability()
          .cuda_compute_capability());
  absl::flat_hash_set<std::tuple<int64_t, int64_t, int64_t>> seen;
  for (const OutputTile& tile : limits.output_tiles) {
    int64_t block_m = ClampBlockM(limits, tile[0], bounds->max_m);
    int64_t block_n = std::min(tile[1], bounds->max_n);
    for (int64_t block_k : ContractingTilesForOutputTile(
             limits, block_m, block_n, bounds->contracting_size)) {
      if (!seen.insert({block_m, block_n, block_k}).second) {
        continue;
      }
      absl::StatusOr<std::unique_ptr<BackendConfig>> config =
          ConfigForTile(*bounds->dot, block_m, block_n, block_k);
      if (!config.ok()) {
        VLOG(2) << "Tile IR: dropping tile " << block_m << "x" << block_n << "x"
                << block_k << " for " << instr.name() << ": "
                << config.status().message();
        continue;
      }
      configs.push_back(*std::move(config));
    }
  }
  VLOG(1) << "Tile IR: offering " << configs.size() << " config(s) for "
          << instr.name();
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> TileIrBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (std::optional<GemmBounds> bounds = GetGemmBounds(instr);
      bounds.has_value()) {
    const TileIrLimits& limits = LimitsForOrDie(
        target_config().device_description.gpu_compute_capability()
            .cuda_compute_capability());
    int64_t block_m =
        ClampBlockM(limits, limits.default_tile[0], bounds->max_m);
    int64_t block_n = std::min(limits.default_tile[1], bounds->max_n);
    int64_t block_k = ClampContractingTile(
        limits, limits.default_contracting_tile, bounds->contracting_size);
    absl::StatusOr<std::unique_ptr<BackendConfig>> config =
        ConfigForTile(*bounds->dot, block_m, block_n, block_k);
    if (config.ok()) {
      return std::move(config).value();
    }
    VLOG(2) << "Tile IR: default tile " << block_m << "x" << block_n << "x"
            << block_k << " rejected for " << instr.name() << ": "
            << config.status().message();
  }

  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                   GetSupportedConfigs(instr));
  if (configs.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TileIrBackend has no supported configs for '", instr.name(), "'"));
  }
  return std::move(configs[configs.size() / 2]);
}

absl::Status TileIrBackend::ApplyConfig(HloInstruction& instr,
                                        const BackendConfig& config) {
  if (!config.has_tile_ir()) {
    return absl::InvalidArgumentError(
        "Expected TileIrFusionConfig for TileIrBackend.");
  }
  const TileIrFusionConfig& tile_ir_config = config.tile_ir();
  int64_t block_k = tile_ir_config.contracting_tile_size();
  if (block_k <= 0 || block_k % kScaleBlockSize != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TileIrBackend: contracting tile size ", block_k,
        " must be a positive multiple of ", kScaleBlockSize, "."));
  }
  if (instr.opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TileIrBackend: expected a fusion, got ", instr.ToString()));
  }
  HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *instr.fused_instructions_computation(), HloOpcode::kScaledDot);
  if (dot == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TileIrBackend: no scaled dot in fusion ", instr.name()));
  }

  Tile dot_tile;
  dot_tile.add_sizes(block_k);
  RETURN_IF_ERROR(dot->set_backend_config(dot_tile));

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(kTileIrFusionKind);
  backend_config.clear_triton_gemm_config();
  *backend_config.mutable_block_level_fusion_config() =
      tile_ir_config.block_level_fusion_config();
  RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  instr.set_fusion_kind(HloInstruction::FusionKind::kCustom);
  return absl::OkStatus();
}

std::string TileIrBackend::version() const {
  return "cuda_tile_ir_13.3";
}

}  // namespace gpu
}  // namespace xla
