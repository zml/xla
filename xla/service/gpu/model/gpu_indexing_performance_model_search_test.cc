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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "tsl/platform/status_matchers.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using Candidates = absl::InlinedVector<TiledRunTimeData, 4>;

void ExpectEqualCandidate(const TiledRunTimeData& expected,
                          const TiledRunTimeData& actual) {
  const auto& e = expected.runtime_data;
  const auto& a = actual.runtime_data;
  EXPECT_EQ(std::tie(e.flops, e.bytes_read, e.bytes_written, e.read_time,
                     e.write_time, e.compute_time, e.exec_time, e.l2_bytes_read,
                     e.shared_memory_per_block_bytes, e.registers_per_thread,
                     e.compute_utilization, e.memory_utilization),
            std::tie(a.flops, a.bytes_read, a.bytes_written, a.read_time,
                     a.write_time, a.compute_time, a.exec_time, a.l2_bytes_read,
                     a.shared_memory_per_block_bytes, a.registers_per_thread,
                     a.compute_utilization, a.memory_utilization));
  const auto& e_params = expected.block_level_parameters;
  const auto& a_params = actual.block_level_parameters;
  EXPECT_EQ(
      std::tie(e_params.output_tile_sizes, e_params.num_warps,
               e_params.num_ctas, e_params.num_stages,
               e_params.global_scratch_memory_size, e_params.is_tma_allowed,
               e_params.is_warp_specialization_allowed,
               e_params.num_tiles_per_pid, e_params.waves_per_eu),
      std::tie(a_params.output_tile_sizes, a_params.num_warps,
               a_params.num_ctas, a_params.num_stages,
               a_params.global_scratch_memory_size, a_params.is_tma_allowed,
               a_params.is_warp_specialization_allowed,
               a_params.num_tiles_per_pid, a_params.waves_per_eu));
}

class TilingSearchTest : public HloHardwareIndependentTestBase,
                         public ::testing::WithParamInterface<std::string> {
 public:
  TilingSearchTest() {
    RegisterSymbolicExprStorage(&context_);
    if (GetParam() == "a100") {
      device_ = TestGpuDeviceInfo::A100SXMDeviceInfo();
    } else if (GetParam() == "mi350") {
      device_ = TestGpuDeviceInfo::AMDMI350DeviceInfo();
    }
  }

 protected:
  absl::StatusOr<Candidates> Search(const HloModule& module, int top_k,
                                    const TilingSearchOptions& options) {
    auto fusion = HloFusionAdaptor::ForInstruction(
        module.entry_computation()->root_instruction());
    HloFusionAnalysisCache cache(device_);
    GpuPerformanceModelWithIndexingAnalysis model(
        &device_, &cache, HloCostAnalysis::DefaultShapeSize, &context_,
        /*use_experimental_tiling=*/false,
        /*enable_same_shape_multi_output_fusion=*/false);
    ABSL_ASSIGN_OR_RETURN(auto result, model.TryFindTopKBestTilingsForFusion(
                                           *fusion, top_k, options));
    if (const auto* decision = std::get_if<FusionDecision>(&result)) {
      return absl::InternalError(decision->Explain());
    }
    return std::get<Candidates>(std::move(result));
  }

  void ComparePrunedAndExhaustive(absl::string_view hlo, int top_k,
                                  bool expect_pruning = false) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    TilingSearchOptions exhaustive;
    exhaustive.enable_memory_bound = false;
    exhaustive.verify_memory_bound = true;
    using Observation =
        std::pair<std::vector<int64_t>, std::optional<TiledRunTimeData>>;
    std::vector<Observation> observations;
    exhaustive.observe_candidate =
        [&](absl::Span<const int64_t> tiling,
            const std::optional<TiledRunTimeData>& result) {
          observations.emplace_back(
              std::vector<int64_t>(tiling.begin(), tiling.end()), result);
        };
    ASSERT_OK_AND_ASSIGN(Candidates expected,
                         Search(*module, top_k, exhaustive));
    ASSERT_FALSE(expected.empty());

    for (int mode = 0; mode < 10; ++mode) {
      SCOPED_TRACE(mode);
      TilingSearchStats stats;
      TilingSearchOptions pruned;
      pruned.use_workspace = mode > 0;
      pruned.share_tile_sizes = mode != 1;
      pruned.use_prepared_expressions =
          mode == 3 || mode == 4 || mode == 6 || mode == 9;
      pruned.expression_preparation_threshold = mode == 4 ? 32 : 1;
      pruned.use_compact = mode == 5 || mode == 6 || mode == 9;
      // Compare invariant bound metadata with eager sizes, then partial sizes
      // with both materialized and prepared/compact candidate evaluation.
      pruned.use_memory_bound_workspace = mode >= 7;
      pruned.lazy_tile_sizes = mode >= 8;
      pruned.verify_compact = true;
      pruned.verify_evaluation = true;
      pruned.stats = &stats;
      ASSERT_OK_AND_ASSIGN(Candidates actual, Search(*module, top_k, pruned));
      ASSERT_EQ(expected.size(), actual.size());
      for (int64_t i = 0; i < expected.size(); ++i) {
        SCOPED_TRACE(i);
        ExpectEqualCandidate(expected[i], actual[i]);
      }
      if (expect_pruning) EXPECT_GT(stats.memory_rejections, 0);

      // Compare every finite/rejected candidate, not only the winners.
      int64_t observed = 0;
      pruned.enable_memory_bound = false;
      pruned.verify_memory_bound = true;
      pruned.observe_candidate =
          [&](absl::Span<const int64_t> tiling,
              const std::optional<TiledRunTimeData>& result) {
            ASSERT_LT(observed, observations.size());
            const Observation& expected = observations[observed++];
            EXPECT_EQ(expected.first,
                      std::vector<int64_t>(tiling.begin(), tiling.end()));
            ASSERT_EQ(expected.second.has_value(), result.has_value());
            if (result.has_value())
              ExpectEqualCandidate(*expected.second, *result);
          };
      ASSERT_OK(Search(*module, top_k, pruned).status());
      EXPECT_EQ(observed, observations.size());
    }
  }

  mlir::MLIRContext context_;
  se::DeviceDescription device_ = TestGpuDeviceInfo::RTXA6000DeviceInfo();
};

INSTANTIATE_TEST_SUITE_P(DeviceFixtures, TilingSearchTest,
                         ::testing::Values("a6000", "a100", "mi350"));

TEST_P(TilingSearchTest, PreservesEnumerationOrderForEqualScores) {
  constexpr absl::string_view kHlo = R"(
HloModule iota_ties
fused {
  ROOT iota = s32[1048576]{0} iota(), iota_dimension=0
}
ENTRY main {
  ROOT fusion = s32[1048576]{0} fusion(), kind=kCustom, calls=fused
})";
  for (int top_k : {1, 4}) {
    SCOPED_TRACE(top_k);
    ComparePrunedAndExhaustive(kHlo, top_k, /*expect_pruning=*/true);
  }
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  TilingSearchOptions exhaustive;
  exhaustive.enable_memory_bound = false;
  ASSERT_OK_AND_ASSIGN(Candidates results, Search(*module, 4, exhaustive));
  ASSERT_EQ(results.size(), 4);
  for (int64_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[0].runtime_data.exec_time,
              results[i].runtime_data.exec_time);
    EXPECT_LT(results[i - 1].block_level_parameters.output_tile_sizes[0][0],
              results[i].block_level_parameters.output_tile_sizes[0][0]);
  }
  TilingSearchStats compact_stats;
  exhaustive.use_compact = true;
  exhaustive.verify_compact = true;
  exhaustive.stats = &compact_stats;
  ASSERT_OK(Search(*module, 4, exhaustive).status());
  EXPECT_GT(compact_stats.compact_candidates, 0);
  EXPECT_EQ(compact_stats.materialized_candidates, 0);
}

TEST_P(TilingSearchTest, LargeWriteBoundMatchesFullModelRounding) {
  // An 8 GB symbolic output requires no data allocation. The old float
  // WriteTime bound rounds one quarter-nanosecond above the full double model
  // for the A6000 fixture, despite both assuming fully coalesced writes.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule large_iota
fused {
  ROOT iota = s32[2000000000]{0} iota(), iota_dimension=0
}
ENTRY main {
  ROOT fusion = s32[2000000000]{0} fusion(), kind=kCustom, calls=fused
})"));
  const absl::Duration expected =
      absl::Seconds(1.0 * int64_t{8000000000} / device_.memory_bandwidth());
  if (GetParam() == "a6000") {
    EXPECT_GT(GpuPerformanceModelBase::WriteTime(device_, 8000000000),
              expected);
  }
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    TilingSearchOptions options;
    options.verify_memory_bound = true;
    options.use_memory_bound_workspace = mode > 0;
    options.lazy_tile_sizes = mode >= 2;
    options.use_compact = mode == 3;
    options.use_prepared_expressions = mode == 3;
    options.expression_preparation_threshold = 1;
    ASSERT_OK_AND_ASSIGN(Candidates results, Search(*module, 4, options));
    ASSERT_EQ(results.size(), 4);
    EXPECT_EQ(results.front().runtime_data.exec_time, expected);
    EXPECT_EQ(results.front().runtime_data.bytes_read, 0);
    EXPECT_EQ(results.front().runtime_data.compute_time, absl::ZeroDuration());
  }
}

TEST_P(TilingSearchTest, IrregularReductionAndBroadcast) {
  constexpr absl::string_view kHlo = R"(
HloModule reduction
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
fused {
  p = f32[17,129]{1,0} parameter(0)
  zero = f32[] constant(0)
  sum = f32[17]{0} reduce(p, zero), dimensions={1}, to_apply=add
  broadcast = f32[17,129]{1,0} broadcast(sum), dimensions={0}
  ROOT normalized = f32[17,129]{1,0} subtract(p, broadcast)
}
ENTRY main {
  p = f32[17,129]{1,0} parameter(0)
  ROOT fusion = f32[17,129]{1,0} fusion(p), kind=kCustom, calls=fused
})";
  for (int top_k : {1, 4}) ComparePrunedAndExhaustive(kHlo, top_k);
}

TEST_P(TilingSearchTest, OffsetDistinctReadsOfSameOperand) {
  constexpr absl::string_view kHlo = R"(
HloModule sliced_reads
fused {
  p = f32[1025]{0} parameter(0)
  first = f32[1024]{0} slice(p), slice={[0:1024]}
  second = f32[1024]{0} slice(p), slice={[1:1025]}
  ROOT sum = f32[1024]{0} add(first, second)
}
ENTRY main {
  p = f32[1025]{0} parameter(0)
  ROOT fusion = f32[1024]{0} fusion(p), kind=kCustom, calls=fused
})";
  for (int top_k : {1, 4}) ComparePrunedAndExhaustive(kHlo, top_k);
}

TEST_P(TilingSearchTest, DistinctTileSizesOfSameOperand) {
  // The slice and reduction access the same external HLO through different
  // tile sizes. The bound must preserve the per-operand maximum calculation.
  constexpr absl::string_view kHlo = R"(
HloModule differently_sized_reads
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
fused {
  p = f32[17,65]{1,0} parameter(0)
  slice = f32[17,64]{1,0} slice(p), slice={[0:17], [0:64]}
  zero = f32[] constant(0)
  sum = f32[17]{0} reduce(p, zero), dimensions={1}, to_apply=add
  broadcast = f32[17,64]{1,0} broadcast(sum), dimensions={0}
  ROOT normalized = f32[17,64]{1,0} subtract(slice, broadcast)
}
ENTRY main {
  p = f32[17,65]{1,0} parameter(0)
  ROOT fusion = f32[17,64]{1,0} fusion(p), kind=kCustom, calls=fused
})";
  for (int top_k : {1, 4}) ComparePrunedAndExhaustive(kHlo, top_k);
}

TEST_P(TilingSearchTest, MultiOutputReductionAndBitcast) {
  constexpr absl::string_view kHlo = R"(
HloModule multioutput
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}
fused {
  p = f32[64]{0} parameter(0)
  absolute = f32[64]{0} abs(p)
  bitcast = f32[4,4,4]{2,1,0} bitcast(absolute)
  zero = f32[] constant(0)
  reduce = f32[4,4]{1,0} reduce(bitcast, zero), dimensions={1}, to_apply=add
  ROOT tuple = (f32[4,4]{1,0}, f32[64]{0}) tuple(reduce, absolute)
}
ENTRY main {
  p = f32[64]{0} parameter(0)
  ROOT fusion = (f32[4,4]{1,0}, f32[64]{0}) fusion(p), kind=kCustom, calls=fused
})";
  for (int top_k : {1, 4}) ComparePrunedAndExhaustive(kHlo, top_k);
}

TEST_P(TilingSearchTest, StrideErrorPrecedesRegisterRejection) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule reversed
fused {
  p = f32[16] parameter(0)
  ROOT reverse = f32[16] reverse(p), dimensions={0}
}
ENTRY main {
  p = f32[16] parameter(0)
  ROOT fusion = f32[16] fusion(p), kind=kCustom, calls=fused
})"));
  // Make even the smallest tile exceed the register limit, to verify that
  // rejection cannot hide the original materialization's negative-stride error.
  device_.set_registers_per_block_limit(1);
  TilingSearchOptions reference;
  reference.enable_memory_bound = false;
  const auto expected = Search(*module, 0, reference);
  EXPECT_EQ(expected.status().code(), absl::StatusCode::kUnimplemented);
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    TilingSearchOptions candidate = reference;
    candidate.use_compact = true;
    candidate.verify_memory_bound = true;
    candidate.use_memory_bound_workspace = mode > 0;
    candidate.lazy_tile_sizes = mode >= 2;
    candidate.use_prepared_expressions = mode == 3;
    candidate.expression_preparation_threshold = 1;
    EXPECT_EQ(Search(*module, 0, candidate).status(), expected.status());
  }
}

TEST_P(TilingSearchTest, RegisterCheckDistinguishesInternalTiles) {
  constexpr absl::string_view kHlo = R"(
HloModule internal_tiles
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}
fused {
  p = f32[] parameter(0)
  broadcast = f32[65536] broadcast(p), dimensions={}
  exp = f32[65536] exponential(broadcast)
  zero = f32[] constant(0)
  ROOT sum = f32[] reduce(exp, zero), dimensions={0}, to_apply=add
}
ENTRY main {
  p = f32[] parameter(0)
  ROOT fusion = f32[] fusion(p), kind=kCustom, calls=fused
})";
  ComparePrunedAndExhaustive(kHlo, 4);
}

TEST_P(TilingSearchTest, ZeroTopKStillObservesCandidates) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule scalar
fused {
  p = f32[] parameter(0)
  ROOT negate = f32[] negate(p)
}
ENTRY main {
  p = f32[] parameter(0)
  ROOT fusion = f32[] fusion(p), kind=kCustom, calls=fused
})"));
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    int observed = 0;
    TilingSearchOptions options;
    options.verify_memory_bound = true;
    options.use_memory_bound_workspace = mode > 0;
    options.lazy_tile_sizes = mode >= 2;
    options.use_compact = mode == 3;
    options.use_prepared_expressions = mode == 3;
    options.expression_preparation_threshold = 1;
    options.observe_candidate =
        [&](absl::Span<const int64_t> tiling,
            const std::optional<TiledRunTimeData>& result) {
          ++observed;
          EXPECT_TRUE(tiling.empty());
          EXPECT_TRUE(result.has_value());
        };
    ASSERT_OK_AND_ASSIGN(Candidates results, Search(*module, 0, options));
    EXPECT_TRUE(results.empty());
    EXPECT_EQ(observed, 1);
    EXPECT_EQ(Search(*module, -1, options).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace xla::gpu
