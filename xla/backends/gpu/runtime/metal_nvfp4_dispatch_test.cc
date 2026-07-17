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

// Host unit tests for shipped NVFP4 dense dispatch = MLX policy helpers
// (metal_nvfp4_dispatch.h), not parallel reimplementations.

#include "xla/backends/gpu/runtime/metal_nvfp4_dispatch.h"

#include <gtest/gtest.h>

namespace xla {
namespace gpu {
namespace {

TEST(MetalNvfp4DispatchTest, VectorLimitMatchesMlxTable) {
  // gen>=15, non-d
  EXPECT_EQ(GetNvfp4QmvBatchLimit(2048, 2048), 18);
  EXPECT_EQ(GetNvfp4QmvBatchLimit(2816, 2816), 12);
  EXPECT_EQ(GetNvfp4QmvBatchLimit(4096, 4096), 12);
  EXPECT_EQ(GetNvfp4QmvBatchLimit(8192, 8192), 10);

  EXPECT_EQ(GetNvfp4QmvBatchLimit(2048, 2048, 'd', 15), 32);
  EXPECT_EQ(GetNvfp4QmvBatchLimit(4096, 4096, 'd', 15), 18);
  EXPECT_EQ(GetNvfp4QmvBatchLimit(8192, 8192, 'd', 15), 12);

  EXPECT_EQ(GetNvfp4QmvBatchLimit(2048, 2048, ' ', 13), 14);
  EXPECT_EQ(GetNvfp4QmvBatchLimit(4096, 4096, ' ', 14), 10);
  EXPECT_EQ(GetNvfp4QmvBatchLimit(8192, 8192, ' ', 13), 6);
}

TEST(MetalNvfp4DispatchTest, WideVecsPerTgMatchesMlxFormula) {
  EXPECT_EQ(Nvfp4QmvWideVecsPerTg(2), 2);
  EXPECT_EQ(Nvfp4QmvWideVecsPerTg(3), 3);
  EXPECT_EQ(Nvfp4QmvWideVecsPerTg(4), 4);
  EXPECT_EQ(Nvfp4QmvWideVecsPerTg(5), 5);
  EXPECT_EQ(Nvfp4QmvWideVecsPerTg(6), 3);
  EXPECT_EQ(Nvfp4QmvWideVecsPerTg(9), 5);
  EXPECT_EQ(Nvfp4QmvWideVecsPerTg(10), 5);
}

TEST(MetalNvfp4DispatchTest, SplitKMatchesMlxHeuristicTiles) {
  // MLX: bm=bn=32, sk=max(1,512/tgs), cap K/gs, K%(sk*gs)==0 (+ body BK).
  // M=16, N=2816: m_tiles=1, n_tiles=88, tgs=88, sk=max(1,512/88)=5
  // → min(5,176)=5; 2816%80!=0 → 4; 2816%64==0; 704%32==0 → 4
  EXPECT_EQ(ComputeNvfp4QmmSplitK(16, 2816, 2816), 4);
  EXPECT_EQ(ComputeNvfp4QmmSplitK(12, 2816, 2816), 4);

  // Large M: tgs high → sk=1
  EXPECT_EQ(ComputeNvfp4QmmSplitK(256, 2816, 2816), 1);
  EXPECT_EQ(ComputeNvfp4QmmSplitK(1, 32, 16), 1);

  // The Steel body predicates K_rem, so the MLX-valid 176-value partitions
  // must not be rejected merely because 176 is not a multiple of BK=32.
  EXPECT_EQ(ComputeNvfp4QmmSplitK(16, 2816, 704), 4);
}

TEST(MetalNvfp4DispatchTest, PathSelectionIsPureMlx) {
  // K=N=2816 → vector_limit=12 (no wide floor, no mid-M hacks).
  constexpr int64_t K = 2816;
  constexpr int64_t N = 2816;
  EXPECT_EQ(GetNvfp4QmvBatchLimit(static_cast<int>(K), static_cast<int>(N)),
            12);

  EXPECT_EQ(SelectNvfp4DensePath(1, K, N), Nvfp4DensePath::kQmv);
  EXPECT_EQ(SelectNvfp4DensePath(2, K, N), Nvfp4DensePath::kQmvWide);
  EXPECT_EQ(SelectNvfp4DensePath(8, K, N), Nvfp4DensePath::kQmvWide);
  EXPECT_EQ(SelectNvfp4DensePath(11, K, N), Nvfp4DensePath::kQmvWide);
  // M >= 12 → qmm path; split_k=4 for M=16 → split-K
  EXPECT_EQ(SelectNvfp4DensePath(12, K, N), Nvfp4DensePath::kQmmSplitK);
  EXPECT_EQ(SelectNvfp4DensePath(16, K, N), Nvfp4DensePath::kQmmSplitK);
  EXPECT_EQ(SelectNvfp4DensePath(256, K, N), Nvfp4DensePath::kQmm);

  EXPECT_STREQ(Nvfp4QmvWideKernelName(Nvfp4QmvWideVecsPerTg(2)),
               "nvfp4_qmv_wide_2");
  EXPECT_STREQ(Nvfp4QmvWideKernelName(Nvfp4QmvWideVecsPerTg(5)),
               "nvfp4_qmv_wide_5");
}

TEST(MetalNvfp4DispatchTest, TargetArchitectureChangesWideQmvBoundary) {
  // K=N=2048, M=20 is deliberately on opposite sides of MLX's tables:
  // fallback gen>=15 non-Ultra has limit 18, while gen-14 Ultra has limit 32.
  // The former therefore enters split-K Steel and the latter stays on wide QMV.
  EXPECT_EQ(SelectNvfp4DensePath(20, 2048, 2048), Nvfp4DensePath::kQmmSplitK);
  EXPECT_EQ(SelectNvfp4DensePath(20, 2048, 2048, 'd', 14),
            Nvfp4DensePath::kQmvWide);
  EXPECT_EQ(GetNvfp4QmvBatchLimit(2048, 2048), 18);
  EXPECT_EQ(GetNvfp4QmvBatchLimit(2048, 2048, 'd', 14), 32);
  EXPECT_EQ(ComputeNvfp4QmmSplitK(20, 2048, 2048), 8);
}

TEST(MetalNvfp4DispatchTest, AlignNAndKernelNames) {
  EXPECT_TRUE(Nvfp4QmmAlignN(2816));
  EXPECT_FALSE(Nvfp4QmmAlignN(2800));
  EXPECT_STREQ(Nvfp4QmmKernelName(2816), "nvfp4_qmm_t_alN");
  EXPECT_STREQ(Nvfp4QmmKernelName(2800), "nvfp4_qmm_t");
  EXPECT_STREQ(Nvfp4QmmSplitkKernelName(2816), "nvfp4_qmm_t_splitk_alN");
  EXPECT_STREQ(Nvfp4QmmSplitkKernelName(2800), "nvfp4_qmm_t_splitk");
}

TEST(MetalNvfp4DispatchTest, LargeShapeLimit) {
  // limit=10 for big K/N; M=10 → split or qmm via MLX formula only.
  EXPECT_EQ(GetNvfp4QmvBatchLimit(8192, 8192), 10);
  EXPECT_EQ(SelectNvfp4DensePath(9, 8192, 8192), Nvfp4DensePath::kQmvWide);
  EXPECT_NE(SelectNvfp4DensePath(10, 8192, 8192), Nvfp4DensePath::kQmvWide);
  EXPECT_GT(ComputeNvfp4QmmSplitK(10, 8192, 8192), 1);
  EXPECT_EQ(SelectNvfp4DensePath(10, 8192, 8192), Nvfp4DensePath::kQmmSplitK);
}

TEST(MetalNvfp4DispatchTest, MoeGatherQmmMatchesMlxReuseGate) {
  // MLX gather_qmm_rhs reuse gate: B=R>=16 and B/E>=4.
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(15, 1, 32, 32));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(511, 128, 2816, 1408));
  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(512, 128, 2816, 1408));
  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(512, 128, 704, 2816));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(128, 128, 2816, 1408));

  // The counting sort has 256 buckets. Larger expert counts must fall back to
  // gather_qmv instead of failing when R crosses the reuse threshold.
  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(1024, 256, 32, 32));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(1028, 257, 32, 32));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(16, 0, 32, 32));

  // Partial 32-wide Steel tiles and odd row widths are covered by safe/scalar
  // tails and stay on the MLX gather-QMM path. Only group-16 K is required.
  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(16, 1, 48, 32));
  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(16, 1, 32, 36));
  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(16, 1, 32, 35));
  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(16, 1, 16, 1));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(16, 1, 24, 32));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(16, 1, 0, 32));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(16, 1, 32, 0));
}

TEST(MetalNvfp4DispatchTest, MoeSortedPathKeepsPerSchemeGates) {
  constexpr int64_t E = 128, K = 2816, N = 704;
  // NVFP4 follows the MLX reuse gate; for E=128 that is R >= 512 (prefill).
  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(128, E, K, N, /*is_nvfp4=*/true));
  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(511, E, K, N, /*is_nvfp4=*/true));
  EXPECT_TRUE(ShouldUseMetalMoeSortedPath(512, E, K, N, /*is_nvfp4=*/true));
  // E > 256 exceeds the moe_argsort bucket capacity at any R.
  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(4096, 257, K, N, /*is_nvfp4=*/true));

  // bf16/fp8 keep their measured large-route cutoff.
  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(1023, 8, 32, 32, /*is_nvfp4=*/false));
  EXPECT_TRUE(ShouldUseMetalMoeSortedPath(1024, 8, 32, 32, /*is_nvfp4=*/false));
  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(2048, 257, 32, 32,
                                           /*is_nvfp4=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
