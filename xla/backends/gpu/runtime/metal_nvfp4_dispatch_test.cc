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

#include "xla/backends/gpu/runtime/metal_nvfp4_dispatch.h"

#include <gtest/gtest.h>

namespace xla {
namespace gpu {
namespace {

TEST(MetalNvfp4DispatchTest, VectorLimitMatchesMlxTable) {
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
  EXPECT_EQ(ComputeNvfp4QmmSplitK(16, 2816, 2816), 4);
  EXPECT_EQ(ComputeNvfp4QmmSplitK(12, 2816, 2816), 4);

  EXPECT_EQ(ComputeNvfp4QmmSplitK(256, 2816, 2816), 1);
  EXPECT_EQ(ComputeNvfp4QmmSplitK(1, 32, 16), 1);

  EXPECT_EQ(ComputeNvfp4QmmSplitK(16, 2816, 704), 4);
}

TEST(MetalNvfp4DispatchTest, PathSelectionIsPureMlx) {
  constexpr int64_t K = 2816;
  constexpr int64_t N = 2816;
  EXPECT_EQ(GetNvfp4QmvBatchLimit(static_cast<int>(K), static_cast<int>(N)),
            12);

  EXPECT_EQ(SelectNvfp4DensePath(1, K, N), Nvfp4DensePath::kQmv);
  EXPECT_EQ(SelectNvfp4DensePath(2, K, N), Nvfp4DensePath::kQmvWide);
  EXPECT_EQ(SelectNvfp4DensePath(8, K, N), Nvfp4DensePath::kQmvWide);
  EXPECT_EQ(SelectNvfp4DensePath(11, K, N), Nvfp4DensePath::kQmvWide);
  EXPECT_EQ(SelectNvfp4DensePath(12, K, N), Nvfp4DensePath::kQmmSplitK);
  EXPECT_EQ(SelectNvfp4DensePath(16, K, N), Nvfp4DensePath::kQmmSplitK);
  EXPECT_EQ(SelectNvfp4DensePath(256, K, N), Nvfp4DensePath::kQmm);

  EXPECT_STREQ(Nvfp4QmvWideKernelName(Nvfp4QmvWideVecsPerTg(2)),
               "nvfp4_qmv_wide_2");
  EXPECT_STREQ(Nvfp4QmvWideKernelName(Nvfp4QmvWideVecsPerTg(5)),
               "nvfp4_qmv_wide_5");
}

TEST(MetalNvfp4DispatchTest, TargetArchitectureChangesWideQmvBoundary) {
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

TEST(MetalNvfp4DispatchTest, QmmTileFollowsTheMeasuredOptimum) {
  EXPECT_EQ(SelectNvfp4QmmTile(16, 21504).bm, 16);
  EXPECT_EQ(SelectNvfp4QmmTile(256, 21504).bm, 64);
  EXPECT_STREQ(SelectNvfp4QmmTile(256, 21504).name, "nvfp4_qmm_t_bm64_alN");
  EXPECT_EQ(SelectNvfp4QmmTile(256, 5376).bm, 64);

  EXPECT_EQ(SelectNvfp4QmmTile(48, 21504).bm, 16);
  EXPECT_EQ(SelectNvfp4QmmTile(96, 21504).bm, 32);
  EXPECT_EQ(SelectNvfp4QmmTile(32, 21504).bm, 32);

  EXPECT_EQ(SelectNvfp4QmmTile(64, 5376).bm, 32);

  EXPECT_STREQ(SelectNvfp4QmmTile(256, 2800).name, "nvfp4_qmm_t_bm64");

  EXPECT_EQ(kNvfp4SplitkBM, 16);
  EXPECT_EQ(kNvfp4QmmBM, 16);
}

TEST(MetalNvfp4DispatchTest, TileChoiceDoesNotDisturbPathOrWorkspace) {
  EXPECT_EQ(SelectNvfp4DensePath(256, 5376, 21504), Nvfp4DensePath::kQmm);
  EXPECT_EQ(ComputeNvfp4QmmSplitK(256, 21504, 5376), 1);
  EXPECT_EQ(SelectNvfp4DensePath(1, 2048, 2048), Nvfp4DensePath::kQmv);
}

TEST(MetalNvfp4DispatchTest, LargeShapeLimit) {
  EXPECT_EQ(GetNvfp4QmvBatchLimit(8192, 8192), 10);
  EXPECT_EQ(SelectNvfp4DensePath(9, 8192, 8192), Nvfp4DensePath::kQmvWide);
  EXPECT_NE(SelectNvfp4DensePath(10, 8192, 8192), Nvfp4DensePath::kQmvWide);
  EXPECT_GT(ComputeNvfp4QmmSplitK(10, 8192, 8192), 1);
  EXPECT_EQ(SelectNvfp4DensePath(10, 8192, 8192), Nvfp4DensePath::kQmmSplitK);
}

TEST(MetalNvfp4DispatchTest, MoeGatherQmmMatchesMlxReuseGate) {
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(15, 1, 32, 32));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(511, 128, 2816, 1408));
  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(512, 128, 2816, 1408));
  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(512, 128, 704, 2816));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(128, 128, 2816, 1408));

  EXPECT_TRUE(ShouldUseNvfp4MoeGatherQmm(1024, 256, 32, 32));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(1028, 257, 32, 32));
  EXPECT_FALSE(ShouldUseNvfp4MoeGatherQmm(16, 0, 32, 32));

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
  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(128, E, K, N, /*is_nvfp4=*/true));
  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(511, E, K, N, /*is_nvfp4=*/true));
  EXPECT_TRUE(ShouldUseMetalMoeSortedPath(512, E, K, N, /*is_nvfp4=*/true));
  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(4096, 257, K, N, /*is_nvfp4=*/true));

  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(1023, 8, 32, 32, /*is_nvfp4=*/false));
  EXPECT_TRUE(ShouldUseMetalMoeSortedPath(1024, 8, 32, 32, /*is_nvfp4=*/false));
  EXPECT_FALSE(ShouldUseMetalMoeSortedPath(2048, 257, 32, 32,
                                           /*is_nvfp4=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
