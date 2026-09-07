/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_INDEXING_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_INDEXING_PERFORMANCE_MODEL_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Contains informations about block level parameters and run time of a fusion.
struct TiledRunTimeData {
  EstimateRunTimeData runtime_data;
  xla::xtile::BlockLevelParameters block_level_parameters;

  std::string ToString() const {
    return absl::StrCat("block_config: {", block_level_parameters.ToString(),
                        "}, ", runtime_data.ToString());
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TiledRunTimeData& data) {
    sink.Append(data.ToString());
  }
};

using TiledRunTimeDataOrError = std::variant<TiledRunTimeData, FusionDecision>;

using TopKTiledRunTimeDataOrError =
    std::variant<absl::InlinedVector<TiledRunTimeData, 4>, FusionDecision>;

// Optional diagnostics for replaying a search. Collection is disabled in normal
// compilation unless verbose logging is enabled.
struct TilingSearchStats {
  int64_t symbolic_instructions = 0;
  int64_t enumerated = 0;
  int64_t constraint_rejections = 0;
  int64_t memory_rejections = 0;
  int64_t materialized_candidates = 0;
  int64_t materialized_instructions = 0;
  int64_t accepted_candidates = 0;
  int64_t compact_candidates = 0;
  int64_t compact_fallbacks = 0;
  int64_t expression_preparations = 0;
  int64_t prepared_evaluations = 0;
  std::string compact_rejection_reason;
  absl::Duration analysis_time = absl::ZeroDuration();
  absl::Duration constraint_time = absl::ZeroDuration();
  absl::Duration memory_bound_time = absl::ZeroDuration();
  absl::Duration tile_sizes_time = absl::ZeroDuration();
  absl::Duration materialization_time = absl::ZeroDuration();
  absl::Duration compact_time = absl::ZeroDuration();
  absl::Duration expression_preparation_time = absl::ZeroDuration();
  absl::Duration estimation_time = absl::ZeroDuration();
  absl::Duration total_time = absl::ZeroDuration();
};

// Defaults select the materialized reference for controlled replay. The overload
// without options enables the optimized search path.
struct TilingSearchOptions {
  bool use_workspace = false;
  // Independently disable sharing with the bound to measure storage reuse.
  bool share_tile_sizes = true;
  // Reuse immutable operand/root metadata and candidate scratch in the bound.
  // This path always shares evaluated sizes, independently of share_tile_sizes.
  bool use_memory_bound_workspace = false;
  // Requires use_memory_bound_workspace. Evaluate only root/external sizes
  // before pruning, then fill remaining sizes for surviving candidates.
  bool lazy_tile_sizes = false;
  bool use_compact = false;
  bool use_prepared_expressions = false;
  int64_t expression_preparation_threshold = 32;
  // Compare the compact estimate with independently materialized instructions.
  bool verify_compact = false;
  // Also verify workspace/prepared candidates which use full materialization.
  bool verify_evaluation = false;
  bool enable_memory_bound = true;
  // Evaluate even pruned candidates and check the bound against the full model.
  // Also compare the metadata bound exactly with the original bound.
  bool verify_memory_bound = false;
  TilingSearchStats* stats = nullptr;
  // Called for every fully evaluated candidate, including infinite estimates
  // (nullopt). Disable pruning to observe the complete search.
  std::function<void(absl::Span<const int64_t>,
                     const std::optional<TiledRunTimeData>&)>
      observe_candidate;
};

// Implementation of Cost Model that uses indexing analysis to estimate amount
// of compute and memory access time.
class GpuPerformanceModelWithIndexingAnalysis : public GpuPerformanceModelBase {
 public:
  explicit GpuPerformanceModelWithIndexingAnalysis(
      const se::DeviceDescription* device_info,
      HloFusionAnalysisCache* fusion_analysis_cache,
      HloCostAnalysis::ShapeSizeFunction shape_size,
      mlir::MLIRContext* mlir_context, bool use_experimental_tiling,
      bool enable_same_shape_multi_output_fusion)
      : hlo_op_profile_(&HloOpProfiles::Singleton().GetProfile(*device_info)),
        device_info_(device_info),
        fusion_analysis_cache_(fusion_analysis_cache),
        shape_size_(shape_size),
        cost_analysis_(
            GpuHloCostAnalysis::Options{shape_size_,
                                        /*per_second_rates=*/{},
                                        /*min_latencies_seconds=*/{},
                                        /*count_multiple_input_accesses=*/true},
            *device_info_),
        mlir_context_(mlir_context),
        use_experimental_tiling_(use_experimental_tiling),
        enable_same_shape_multi_output_fusion_(
            enable_same_shape_multi_output_fusion) {}

  // Returns the number of warps for the given tiled HLO computation.
  static int64_t EstimateNumWarps(
      const TiledHloComputation& tiled_hlo_computation);

  // Returns the number of warps for the given tiled HLO computation.
  static int64_t EstimateNumWarps(
      const experimental::TiledHloComputation& tiled_hlo_computation);

  absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForTiledHloComputation(
      const HloFusionAdaptor& fusion_adaptor,
      const TiledHloComputation& tiled_hlo_computation,
      const xla::xtile::BlockLevelParameters& block_level_parameters);

  // Estimate the run time of the fusion with the given launch dimensions and
  // output tile sizes.
  //
  // The model uses SymbolicTileAnalysis to build a TiledHloComputation with the
  // given tile sizes. This way it can better estimate the amount of memory
  // access and computation.
  absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForTiledFusion(
      const HloFusionAdaptor& fusion_adaptor,
      const xla::xtile::BlockLevelParameters& block_level_parameters);

  // Estimate the run time of an Hlo instruction assuming it is emitted by
  // Triton.
  absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForTriton(
      const HloInstruction* instr,
      const xla::xtile::BlockLevelParameters* block_level_parameters = nullptr);

  // Estimates the best tile sizes for the given fusion. Iterates over all the
  // good tile sizes provided by SymbolicTileAnalysis, estimates the run time
  // for each of them.
  //
  // Returns status if there is an error that we can't recover from.
  // Returns FusionDecision if the fusion can't be tiled or there are no valid
  // block level parameters.
  // Otherwise returns block level parameters that give the best execution time.
  absl::StatusOr<TiledRunTimeDataOrError> TryFindBestTilingForFusion(
      const HloFusionAdaptor& fusion_adaptor);

  // Returns top_k (possibly fewer if not enough valid tilings are found) block
  // level parameters for the given fusion.
  absl::StatusOr<TopKTiledRunTimeDataOrError> TryFindTopKBestTilingsForFusion(
      const HloFusionAdaptor& fusion_adaptor, int top_k);

  absl::StatusOr<TopKTiledRunTimeDataOrError> TryFindTopKBestTilingsForFusion(
      const HloFusionAdaptor& fusion_adaptor, int top_k,
      const TilingSearchOptions& options);

  // Returns an estimate how many FLOPs will be used to produce one element of
  // the output.
  int64_t FlopsPerElement(const HloInstruction* instr);

 private:
  const HloOpProfiles::HloOpProfile* hlo_op_profile_;
  const se::DeviceDescription* device_info_;
  HloFusionAnalysisCache* fusion_analysis_cache_;
  HloCostAnalysis::ShapeSizeFunction shape_size_;
  GpuHloCostAnalysis cost_analysis_;
  mlir::MLIRContext* mlir_context_;
  bool use_experimental_tiling_;
  bool enable_same_shape_multi_output_fusion_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_INDEXING_PERFORMANCE_MODEL_H_
