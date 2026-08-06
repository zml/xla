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

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"

namespace xla::gpu {
namespace {

class FlyGemmDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {};

TEST_F(FlyGemmDeviceTest, Bf16Mma64x64x64) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_gemm

fly_gemm {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = bf16[64,64]{1,0} parameter(1)
  ROOT dot = bf16[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = bf16[64,64]{1,0} parameter(1)
  ROOT fusion = bf16[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8", "stage_output":true,
          "waves_per_eu":2, "schedule_instructions":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1",
          "waves_per_eu":2}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.2, /*arel=*/0.02}));
}

TEST_F(FlyGemmDeviceTest, Bf16Mma256x256x256MultipleBlocks) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_gemm_multiple_blocks

fly_gemm {
  lhs = bf16[256,256]{1,0} parameter(0)
  rhs = bf16[256,256]{1,0} parameter(1)
  ROOT dot = bf16[256,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[256,256]{1,0} parameter(0)
  rhs = bf16[256,256]{1,0} parameter(1)
  ROOT fusion = bf16[256,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16Mfma32x32x8) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_mfma32

fly_gemm {
  lhs = bf16[128,128]{1,0} parameter(0)
  rhs = bf16[128,256]{1,0} parameter(1)
  ROOT dot = bf16[128,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[128,128]{1,0} parameter(0)
  rhs = bf16[128,256]{1,0} parameter(1)
  ROOT fusion = bf16[128,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"128",
          "block_k":"32", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_32X32X8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","128"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16SingleBufferLdsMfma32) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_single_buffer_lds

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,512]{0,1} parameter(1)
  ROOT dot = bf16[256,512]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,512]{0,1} parameter(1)
  ROOT fusion = bf16[256,512]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"256",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8", "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","256"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16Pipelined128x64x128RollingRefill) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_pipelined_128x64x128

fly_gemm {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  ROOT dot = bf16[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  ROOT fusion = bf16[256,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"64",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true,
          "rolling_refill":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16Pipelined128x64x128LocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_pipelined_128x64x128_local_split_k

fly_gemm {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  ROOT dot = bf16[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  ROOT fusion = bf16[256,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"64",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true,
          "local_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16LocalSplitKWithTransposedRhsDotDimension) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_local_split_k_transposed_rhs

fly_gemm {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[128,384]{1,0} parameter(1)
  ROOT dot = bf16[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[128,384]{1,0} parameter(1)
  ROOT fusion = bf16[256,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"64",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true,
          "local_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16Preloaded64x64Tile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_preloaded_64x64

fly_gemm {
  lhs = bf16[128,256]{1,0} parameter(0)
  rhs = bf16[256,128]{0,1} parameter(1)
  ROOT dot = bf16[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[128,256]{1,0} parameter(0)
  rhs = bf16[256,128]{0,1} parameter(1)
  ROOT fusion = bf16[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4", "stage_rhs":true,
          "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"2", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16WideN128x512Tile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_wide_n

fly_gemm {
  lhs = bf16[128,64]{1,0} parameter(0)
  rhs = bf16[64,512]{1,0} parameter(1)
  ROOT dot = bf16[128,512]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[128,64]{1,0} parameter(0)
  rhs = bf16[64,512]{1,0} parameter(1)
  ROOT fusion = bf16[128,512]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"512",
          "block_k":"32", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_16X16X16", "prefetch_rhs":true,
          "schedule_instructions":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","512"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16ColumnMajorRhsVectorLoads) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_column_major_rhs

fly_gemm {
  lhs = bf16[128,128]{1,0} parameter(0)
  rhs = bf16[128,256]{0,1} parameter(1)
  ROOT dot = bf16[128,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[128,128]{1,0} parameter(0)
  rhs = bf16[128,256]{0,1} parameter(1)
  ROOT fusion = bf16[128,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"128",
          "block_k":"32", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","128"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16RhsRegisterPipeline) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_rhs_register_pipeline

fly_gemm {
  lhs = bf16[64,1024]{1,0} parameter(0)
  rhs = bf16[1024,256]{1,0} parameter(1)
  ROOT dot = bf16[64,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[64,1024]{1,0} parameter(0)
  rhs = bf16[1024,256]{1,0} parameter(1)
  ROOT fusion = bf16[64,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"256",
          "block_k":"64", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_16X16X16", "prefetch_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","256"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16FlyDslStagedRhsPipeline) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_staged_rhs_pipeline

fly_gemm {
  lhs = bf16[128,1024]{1,0} parameter(0)
  rhs = bf16[1024,128]{0,1} parameter(1)
  ROOT dot = bf16[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[128,1024]{1,0} parameter(0)
  rhs = bf16[1024,128]{0,1} parameter(1)
  ROOT fusion = bf16[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"128",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_output":true,
          "schedule_instructions":true, "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","128"]}],
          "num_stages":"2", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16FlyDslStagedRhsSixteenWaves) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_staged_rhs_sixteen_waves

fly_gemm {
  lhs = bf16[128,1024]{1,0} parameter(0)
  rhs = bf16[1024,128]{0,1} parameter(1)
  ROOT dot = bf16[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[128,1024]{1,0} parameter(0)
  rhs = bf16[1024,128]{0,1} parameter(1)
  ROOT fusion = bf16[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"128",
          "block_k":"64", "num_warps":"16",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","128"]}],
          "num_stages":"2", "num_warps":"16", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16FlyDslStagedRhsSmallTile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_staged_rhs_small_tile

fly_gemm {
  lhs = bf16[64,1024]{1,0} parameter(0)
  rhs = bf16[1024,32]{0,1} parameter(1)
  ROOT dot = bf16[64,32]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[64,1024]{1,0} parameter(0)
  rhs = bf16[1024,32]{0,1} parameter(1)
  ROOT fusion = bf16[64,32]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"32",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "schedule_instructions":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","32"]}],
          "num_stages":"2", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16FlyDslRegisterRhsPipeline) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_register_rhs_pipeline

fly_gemm {
  lhs = bf16[128,1024]{1,0} parameter(0)
  rhs = bf16[1024,128]{0,1} parameter(1)
  ROOT dot = bf16[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[128,1024]{1,0} parameter(0)
  rhs = bf16[1024,128]{0,1} parameter(1)
  ROOT fusion = bf16[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"128",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "prefetch_rhs":true,
          "stage_output":true, "schedule_instructions":true,
          "async_lhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","128"]}],
          "num_stages":"2", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16DirectToVgprRhsOddTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_direct_to_vgpr_rhs

fly_gemm {
  lhs = bf16[224,192]{1,0} parameter(0)
  rhs_row_major = bf16[256,192]{1,0} parameter(1)
  rhs = bf16[192,256]{0,1} bitcast(rhs_row_major)
  ROOT dot = bf16[224,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[224,192]{1,0} parameter(0)
  rhs = bf16[256,192]{1,0} parameter(1)
  ROOT fusion = bf16[224,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"224", "block_n":"256",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true,
          "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["224","256"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.01}));
}

TEST_F(FlyGemmDeviceTest, Bf16DirectToVgprLhsOddTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_direct_to_vgpr_lhs

fly_gemm {
  lhs = bf16[256,192]{1,0} parameter(0)
  rhs_row_major = bf16[224,192]{1,0} parameter(1)
  rhs = bf16[192,224]{0,1} bitcast(rhs_row_major)
  ROOT dot = bf16[256,224]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[256,192]{1,0} parameter(0)
  rhs = bf16[224,192]{1,0} parameter(1)
  ROOT fusion = bf16[256,224]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true,
          "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.01}));
}

TEST_F(FlyGemmDeviceTest, Bf16VectorMatrix1x256x256) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_vector_matrix

fly_gemv {
  lhs = bf16[1,256]{1,0} parameter(0)
  rhs = bf16[256,256]{1,0} parameter(1)
  ROOT dot = bf16[1,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1,256]{1,0} parameter(0)
  rhs = bf16[256,256]{1,0} parameter(1)
  ROOT fusion = bf16[1,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16MatrixVector256x256x1) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_matrix_vector

fly_gemv {
  lhs = bf16[256,256]{1,0} parameter(0)
  rhs = bf16[256,1]{1,0} parameter(1)
  ROOT dot = bf16[256,1]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[256,256]{1,0} parameter(0)
  rhs = bf16[256,1]{1,0} parameter(1)
  ROOT fusion = bf16[256,1]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

}  // namespace
}  // namespace xla::gpu
