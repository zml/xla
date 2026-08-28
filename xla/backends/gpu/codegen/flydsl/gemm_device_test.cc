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

TEST_F(FlyGemmDeviceTest, F16MfmaGemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_mfma_gemm

fly_gemm {
  lhs = f16[64,256]{1,0} parameter(0)
  rhs = f16[256,64]{0,1} parameter(1)
  ROOT dot = f16[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = f16[64,256]{1,0} parameter(0)
  rhs = f16[256,64]{0,1} parameter(1)
  ROOT fusion = f16[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, S8MfmaStagesBothOperands) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_s8_mfma_gemm

fly_gemm {
  lhs = s8[64,256]{1,0} parameter(0)
  rhs = s8[256,64]{1,0} parameter(1)
  ROOT dot = s32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = s8[64,256]{1,0} parameter(0)
  rhs = s8[256,64]{1,0} parameter(1)
  ROOT fusion = s32[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_I8", "stage_rhs":true,
          "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/0.0, /*arel=*/0.0}));
}

TEST_F(FlyGemmDeviceTest, S8MfmaFlyDslRegisterRhsPipeline) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_s8_mfma_register_rhs_pipeline

fly_gemm {
  lhs = s8[256,1024]{1,0} parameter(0)
  rhs = s8[1024,256]{0,1} parameter(1)
  ROOT dot = s32[256,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = s8[256,1024]{1,0} parameter(0)
  rhs = s8[1024,256]{0,1} parameter(1)
  ROOT fusion = s32[256,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"128",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_I8", "prefetch_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","128"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/0.0, /*arel=*/0.0}));
}

TEST_F(FlyGemmDeviceTest, F32Mfma32Gemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_mfma32_gemm

fly_gemm {
  lhs = f32[64,128]{1,0} parameter(0)
  rhs = f32[128,64]{0,1} parameter(1)
  ROOT dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = f32[64,128]{1,0} parameter(0)
  rhs = f32[128,64]{0,1} parameter(1)
  ROOT fusion = f32[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"64", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_32X32X2_F32"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(FlyGemmDeviceTest, F32Mfma16StagesBothOperands) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_mfma16_stages_both_operands

fly_gemm {
  lhs = f32[64,128]{1,0} parameter(0)
  rhs = f32[128,64]{0,1} parameter(1)
  ROOT dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f32[64,128]{1,0} parameter(0)
  rhs = f32[128,64]{0,1} parameter(1)
  ROOT fusion = f32[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"32",
          "block_k":"32", "num_warps":"2",
          "mfma_atom":"FLY_MFMA_16X16X4_F32", "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","32"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(FlyGemmDeviceTest, F32Mfma32StagesBothOperands) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_mfma32_stages_both_operands

fly_gemm {
  lhs = f32[64,128]{1,0} parameter(0)
  rhs = f32[128,64]{0,1} parameter(1)
  ROOT dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f32[64,128]{1,0} parameter(0)
  rhs = f32[128,64]{0,1} parameter(1)
  ROOT fusion = f32[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"32",
          "block_k":"32", "num_warps":"2",
          "mfma_atom":"FLY_MFMA_32X32X2_F32", "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","32"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(FlyGemmDeviceTest, F32Xf32StagesBothOperands) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_xf32_stages_both_operands

fly_gemm {
  lhs = f32[64,128]{1,0} parameter(0)
  rhs = f32[128,64]{0,1} parameter(1)
  ROOT dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f32[64,128]{1,0} parameter(0)
  rhs = f32[128,64]{0,1} parameter(1)
  ROOT fusion = f32[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"32",
          "block_k":"32", "num_warps":"2",
          "mfma_atom":"FLY_MFMA_32X32X4_XF32", "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","32"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(FlyGemmDeviceTest, F32Xf32SingleBufferLargeTileScheduled) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_xf32_single_buffer_large_tile

fly_gemm {
  lhs = f32[256,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f32[256,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT fusion = f32[256,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"128",
          "block_k":"32", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_32X32X4_XF32", "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true,
          "schedule_instructions":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","128"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(FlyGemmDeviceTest, Bf16OddOutputTailsWithColumnBias) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_odd_output_tails

fly_gemm {
  lhs = bf16[79,48]{1,0} parameter(0)
  rhs = bf16[48,111]{0,1} parameter(1)
  bias = f32[111]{0} parameter(2)
  dot = f32[79,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["16"]}
  broadcast = f32[79,111]{1,0} broadcast(bias), dimensions={1}
  add = f32[79,111]{1,0} add(dot, broadcast)
  ROOT convert = bf16[79,111]{1,0} convert(add)
}

ENTRY main {
  lhs = bf16[79,48]{1,0} parameter(0)
  rhs = bf16[48,111]{0,1} parameter(1)
  bias = f32[111]{0} parameter(2)
  ROOT fusion = bf16[79,111]{1,0} fusion(lhs, rhs, bias),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"16", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16OddKAndOutputTailsWithColumnBias) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_odd_k_and_output_tails

fly_gemm {
  lhs = bf16[79,47]{1,0} parameter(0)
  rhs = bf16[47,111]{0,1} parameter(1)
  bias = f32[111]{0} parameter(2)
  dot = f32[79,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
  broadcast = f32[79,111]{1,0} broadcast(bias), dimensions={1}
  add = f32[79,111]{1,0} add(dot, broadcast)
  ROOT convert = bf16[79,111]{1,0} convert(add)
}

ENTRY main {
  lhs = bf16[79,47]{1,0} parameter(0)
  rhs = bf16[47,111]{0,1} parameter(1)
  bias = f32[111]{0} parameter(2)
  ROOT fusion = bf16[79,111]{1,0} fusion(lhs, rhs, bias),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16OddKWithNonContiguousRhsAndOutputTails) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_odd_k_non_contiguous_rhs

fly_gemm {
  lhs = f16[33,23]{1,0} parameter(0)
  rhs = f16[23,37]{1,0} parameter(1)
  ROOT dot = f16[33,37]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f16[33,23]{1,0} parameter(0)
  rhs = f16[23,37]{1,0} parameter(1)
  ROOT fusion = f16[33,37]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16StagedRhsPadsFinalKStage) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_staged_rhs_long_final_k_stage

fly_gemm {
  lhs = bf16[79,1023]{1,0} parameter(0)
  rhs = bf16[1023,111]{0,1} parameter(1)
  ROOT dot = bf16[79,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[79,1023]{1,0} parameter(0)
  rhs = bf16[1023,111]{0,1} parameter(1)
  ROOT fusion = bf16[79,111]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16DirectToVgprShiftsBothOutputEdges) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_direct_to_vgpr_both_output_edges

fly_gemm {
  lhs = bf16[257,128]{1,0} parameter(0)
  rhs = bf16[128,241]{0,1} parameter(1)
  ROOT dot = bf16[257,241]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[257,128]{1,0} parameter(0)
  rhs = bf16[128,241]{0,1} parameter(1)
  ROOT fusion = bf16[257,241]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16StagedRhsShiftsBothOutputEdges) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_staged_rhs_both_output_edges

fly_gemm {
  lhs = bf16[79,128]{1,0} parameter(0)
  rhs = bf16[128,111]{0,1} parameter(1)
  ROOT dot = bf16[79,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[79,128]{1,0} parameter(0)
  rhs = bf16[128,111]{0,1} parameter(1)
  ROOT fusion = bf16[79,111]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16StagedOutputShiftsBothOutputEdges) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_staged_output_both_output_edges

fly_gemm {
  lhs = bf16[79,128]{1,0} parameter(0)
  rhs = bf16[128,111]{0,1} parameter(1)
  ROOT dot = bf16[79,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[79,128]{1,0} parameter(0)
  rhs = bf16[128,111]{0,1} parameter(1)
  ROOT fusion = bf16[79,111]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_output":true,
          "stage_rhs":true, "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16AttentionValueOutputTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_attention_value_output_transpose

fly_gemm {
  lhs = bf16[32,64,128]{2,1,0} parameter(0)
  rhs = bf16[32,128,128]{2,1,0} parameter(1)
  dot = bf16[32,64,128]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2},
      backend_config={"sizes":["32"]}
  bitcast = bf16[2,16,64,128]{3,2,1,0} bitcast(dot)
  ROOT transpose = bf16[2,128,16,64]{3,2,1,0} transpose(bitcast),
      dimensions={0,3,1,2}
}

ENTRY main {
  lhs = bf16[32,64,128]{2,1,0} parameter(0)
  rhs = bf16[32,128,128]{2,1,0} parameter(1)
  ROOT fusion = bf16[2,128,16,64]{3,2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_output":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","32","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16AttentionScoreInputTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_attention_score_input_transpose

fly_gemm {
  q = bf16[32,64,128]{2,1,0} parameter(0)
  q_transposed = bf16[32,128,64]{2,1,0} transpose(q),
      dimensions={0,2,1}
  k = bf16[32,64,128]{2,1,0} parameter(1)
  dot = bf16[32,128,128]{2,1,0} dot(q_transposed, k),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1},
      backend_config={"sizes":["32"]}
  converted = f32[32,128,128]{2,1,0} convert(dot)
  scale = bf16[] constant(0.125)
  broadcast = bf16[32,128,128]{2,1,0} broadcast(scale), dimensions={}
  widened_scale = f32[32,128,128]{2,1,0} convert(broadcast)
  scaled = f32[32,128,128]{2,1,0} multiply(converted, widened_scale)
  ROOT narrowed = bf16[32,128,128]{2,1,0} convert(scaled)
}

ENTRY main {
  q = bf16[32,64,128]{2,1,0} parameter(0)
  k = bf16[32,64,128]{2,1,0} parameter(1)
  ROOT fusion = bf16[32,128,128]{2,1,0} fusion(q, k),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"128",
          "block_k":"32", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","128","128"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16PackedAttentionScoreSlices) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_packed_attention_score_slices

fly_gemm {
  packed = bf16[3,4,32,1,64]{4,3,2,1,0} parameter(0)
  q_slice = bf16[1,4,32,1,64]{4,3,2,1,0} slice(packed),
      slice={[0:1], [0:4], [0:32], [0:1], [0:64]}
  q = bf16[4,32,64]{2,1,0} bitcast(q_slice)
  k_slice = bf16[1,4,32,1,64]{4,3,2,1,0} slice(packed),
      slice={[1:2], [0:4], [0:32], [0:1], [0:64]}
  k = bf16[4,32,64]{2,1,0} bitcast(k_slice)
  dot = bf16[4,64,64]{2,1,0} dot(q, k),
      lhs_batch_dims={0}, lhs_contracting_dims={1},
      rhs_batch_dims={0}, rhs_contracting_dims={1},
      backend_config={"sizes":["32"]}
  converted = f32[4,64,64]{2,1,0} convert(dot)
  scale = bf16[] constant(0.125)
  broadcast = bf16[4,64,64]{2,1,0} broadcast(scale), dimensions={}
  widened_scale = f32[4,64,64]{2,1,0} convert(broadcast)
  scaled = f32[4,64,64]{2,1,0} multiply(converted, widened_scale)
  ROOT narrowed = bf16[4,64,64]{2,1,0} convert(scaled)
}

ENTRY main {
  packed = bf16[3,4,32,1,64]{4,3,2,1,0} parameter(0)
  ROOT fusion = bf16[4,64,64]{2,1,0} fusion(packed), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16Mfma32SingleBufferShiftsBothOutputEdges) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_mfma32_single_buffer_both_output_edges

fly_gemm {
  lhs = bf16[257,128]{1,0} parameter(0)
  rhs = bf16[128,129]{0,1} parameter(1)
  ROOT dot = bf16[257,129]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[257,128]{1,0} parameter(0)
  rhs = bf16[128,129]{0,1} parameter(1)
  ROOT fusion = bf16[257,129]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"128",
          "block_k":"64", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_32X32X8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","128"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16DirectToVgprLhsShiftedNBoundary) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_direct_to_vgpr_lhs_shifted_n_boundary

fly_gemm {
  lhs = f16[256,256]{1,0} parameter(0)
  rhs_row_major = f16[240,256]{1,0} parameter(1)
  rhs = f16[256,240]{0,1} bitcast(rhs_row_major)
  ROOT dot = f16[256,240]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = f16[256,256]{1,0} parameter(0)
  rhs = f16[240,256]{1,0} parameter(1)
  ROOT fusion = f16[256,240]{1,0} fusion(lhs, rhs),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16DirectToVgprRhsShiftedMBoundary) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_direct_to_vgpr_rhs_shifted_m_boundary

fly_gemm {
  lhs = f16[240,256]{1,0} parameter(0)
  rhs = f16[256,256]{0,1} parameter(1)
  ROOT dot = f16[240,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = f16[240,256]{1,0} parameter(0)
  rhs = f16[256,256]{0,1} parameter(1)
  ROOT fusion = f16[240,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"224", "block_n":"256",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["224","256"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16F32DirectAccumulatorStore) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_f32_direct_accumulator_store

fly_gemm {
  lhs = bf16[256,256]{1,0} parameter(0)
  rhs = bf16[256,224]{0,1} parameter(1)
  ROOT dot = f32[256,224]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[256,256]{1,0} parameter(0)
  rhs = bf16[256,224]{0,1} parameter(1)
  ROOT fusion = f32[256,224]{1,0} fusion(lhs, rhs),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, F16VectorMatrixKContiguousRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_vector_matrix_k_contiguous_rhs

fly_gemv {
  lhs = f16[1,256]{1,0} parameter(0)
  rhs = f16[256,192]{0,1} parameter(1)
  ROOT dot = f16[1,192]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f16[1,256]{1,0} parameter(0)
  rhs = f16[256,192]{0,1} parameter(1)
  ROOT fusion = f16[1,192]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{
          "block_m":"16", "block_n":"64", "block_k":"32",
          "num_warps":"4", "mfma_atom":"FLY_MFMA_16X16X16",
          "gemv_outputs_per_wave":4, "gemv_k_vector_width":2},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16MatrixVector) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_matrix_vector

fly_gemv {
  lhs = f16[128,256]{1,0} parameter(0)
  rhs = f16[1,256]{1,0} parameter(1)
  ROOT dot = f16[128,1]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f16[128,256]{1,0} parameter(0)
  rhs = f16[1,256]{1,0} parameter(1)
  ROOT fusion = f16[128,1]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","16"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, ScalarAddAfterNarrowingStagedOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_scalar_add_after_narrowing_staged_output

fly_gemm {
  lhs = bf16[64,256]{1,0} parameter(0)
  rhs = bf16[256,64]{0,1} parameter(1)
  bias = bf16[] parameter(2)
  dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  narrow = bf16[64,64]{1,0} convert(dot)
  broadcast = bf16[64,64]{1,0} broadcast(bias), dimensions={}
  ROOT add = bf16[64,64]{1,0} add(narrow, broadcast)
}

ENTRY main {
  lhs = bf16[64,256]{1,0} parameter(0)
  rhs = bf16[256,64]{0,1} parameter(1)
  bias = bf16[] parameter(2)
  ROOT fusion = bf16[64,64]{1,0} fusion(lhs, rhs, bias),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_output":true,
          "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, ScalarMultiplyBeforeNarrowingLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_scalar_multiply_before_narrowing_local_split

fly_gemm {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  alpha = f32[] parameter(2)
  dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  broadcast = f32[256,128]{1,0} broadcast(alpha), dimensions={}
  multiply = f32[256,128]{1,0} multiply(dot, broadcast)
  ROOT convert = bf16[256,128]{1,0} convert(multiply)
}

ENTRY main {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  alpha = f32[] parameter(2)
  ROOT fusion = bf16[256,128]{1,0} fusion(lhs, rhs, alpha),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, ScalarAddBeforeNarrowingDirectToVgpr) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_scalar_add_before_narrowing_direct_to_vgpr

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,224]{0,1} parameter(1)
  bias = f32[] parameter(2)
  dot = f32[256,224]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
  broadcast = f32[256,224]{1,0} broadcast(bias), dimensions={}
  add = f32[256,224]{1,0} add(dot, broadcast)
  ROOT convert = bf16[256,224]{1,0} convert(add)
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,224]{0,1} parameter(1)
  bias = f32[] parameter(2)
  ROOT fusion = bf16[256,224]{1,0} fusion(lhs, rhs, bias),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, RowScaleAfterNarrowingStagedOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_row_scale_after_narrowing_staged_output

fly_gemm {
  lhs = bf16[64,256]{1,0} parameter(0)
  rhs = bf16[256,64]{0,1} parameter(1)
  scale = bf16[64]{0} parameter(2)
  dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  narrow = bf16[64,64]{1,0} convert(dot)
  broadcast = bf16[64,64]{1,0} broadcast(scale), dimensions={0}
  ROOT multiply = bf16[64,64]{1,0} multiply(narrow, broadcast)
}

ENTRY main {
  lhs = bf16[64,256]{1,0} parameter(0)
  rhs = bf16[256,64]{0,1} parameter(1)
  scale = bf16[64]{0} parameter(2)
  ROOT fusion = bf16[64,64]{1,0} fusion(lhs, rhs, scale),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_output":true,
          "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, ColumnBiasBeforeNarrowingLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_column_bias_before_narrowing_local_split

fly_gemm {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  bias = f32[128]{0} parameter(2)
  dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  broadcast = f32[256,128]{1,0} broadcast(bias), dimensions={1}
  add = f32[256,128]{1,0} add(dot, broadcast)
  ROOT convert = bf16[256,128]{1,0} convert(add)
}

ENTRY main {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  bias = f32[128]{0} parameter(2)
  ROOT fusion = bf16[256,128]{1,0} fusion(lhs, rhs, bias),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, ColumnScaleBeforeNarrowingDirectToVgpr) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_column_scale_before_narrowing_direct_to_vgpr

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,224]{0,1} parameter(1)
  scale = f32[224]{0} parameter(2)
  dot = f32[256,224]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
  broadcast = f32[256,224]{1,0} broadcast(scale), dimensions={1}
  multiply = f32[256,224]{1,0} multiply(dot, broadcast)
  ROOT convert = bf16[256,224]{1,0} convert(multiply)
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,224]{0,1} parameter(1)
  scale = f32[224]{0} parameter(2)
  ROOT fusion = bf16[256,224]{1,0} fusion(lhs, rhs, scale),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, ColumnBiasDirectToVgprOutputTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_column_bias_direct_to_vgpr_output_tail

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,240]{0,1} parameter(1)
  bias = f32[240]{0} parameter(2)
  dot = f32[256,240]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
  broadcast = f32[256,240]{1,0} broadcast(bias), dimensions={1}
  add = f32[256,240]{1,0} add(dot, broadcast)
  ROOT convert = bf16[256,240]{1,0} convert(add)
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,240]{0,1} parameter(1)
  bias = f32[240]{0} parameter(2)
  ROOT fusion = bf16[256,240]{1,0} fusion(lhs, rhs, bias),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, EpilogueChainLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_epilogue_chain_local_split

fly_gemm {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  scale = f32[128]{0} parameter(2)
  bias = f32[] parameter(3)
  dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  scale_broadcast = f32[256,128]{1,0} broadcast(scale), dimensions={1}
  multiply = f32[256,128]{1,0} multiply(dot, scale_broadcast)
  bias_broadcast = f32[256,128]{1,0} broadcast(bias), dimensions={}
  add = f32[256,128]{1,0} add(multiply, bias_broadcast)
  negate = f32[256,128]{1,0} negate(add)
  ROOT convert = bf16[256,128]{1,0} convert(negate)
}

ENTRY main {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  scale = f32[128]{0} parameter(2)
  bias = f32[] parameter(3)
  ROOT fusion = bf16[256,128]{1,0} fusion(lhs, rhs, scale, bias),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, ColumnBiasReluLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_column_bias_relu_local_split_k

fly_gemm {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  bias = f32[128]{0} parameter(2)
  zero = f32[] constant(0)
  dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  bias_broadcast = f32[256,128]{1,0} broadcast(bias), dimensions={1}
  add = f32[256,128]{1,0} add(dot, bias_broadcast)
  zero_broadcast = f32[256,128]{1,0} broadcast(zero), dimensions={}
  maximum = f32[256,128]{1,0} maximum(add, zero_broadcast)
  ROOT convert = bf16[256,128]{1,0} convert(maximum)
}

ENTRY main {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  bias = f32[128]{0} parameter(2)
  ROOT fusion = bf16[256,128]{1,0} fusion(lhs, rhs, bias),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, ScalarReverseSubtractDivideMinimumStagedOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_scalar_reverse_subtract_divide_minimum_staged_output

fly_gemm {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = bf16[64,64]{0,1} parameter(1)
  ten = f32[] constant(10)
  two = f32[] constant(2)
  upper = f32[] constant(100)
  dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
  ten_broadcast = f32[64,64]{1,0} broadcast(ten), dimensions={}
  subtract = f32[64,64]{1,0} subtract(ten_broadcast, dot)
  two_broadcast = f32[64,64]{1,0} broadcast(two), dimensions={}
  divide = f32[64,64]{1,0} divide(subtract, two_broadcast)
  upper_broadcast = f32[64,64]{1,0} broadcast(upper), dimensions={}
  minimum = f32[64,64]{1,0} minimum(divide, upper_broadcast)
  ROOT convert = bf16[64,64]{1,0} convert(minimum)
}

ENTRY main {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = bf16[64,64]{0,1} parameter(1)
  ROOT fusion = bf16[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8", "stage_output":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16EpilogueChainStagedOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_epilogue_chain_staged_output

fly_gemm {
  lhs = bf16[64,256]{1,0} parameter(0)
  rhs = bf16[256,64]{0,1} parameter(1)
  row_bias = bf16[64]{0} parameter(2)
  column_scale = bf16[64]{0} parameter(3)
  dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  narrow = bf16[64,64]{1,0} convert(dot)
  row_broadcast = bf16[64,64]{1,0} broadcast(row_bias), dimensions={0}
  add = bf16[64,64]{1,0} add(narrow, row_broadcast)
  column_broadcast = bf16[64,64]{1,0} broadcast(column_scale), dimensions={1}
  multiply = bf16[64,64]{1,0} multiply(add, column_broadcast)
  ROOT negate = bf16[64,64]{1,0} negate(multiply)
}

ENTRY main {
  lhs = bf16[64,256]{1,0} parameter(0)
  rhs = bf16[256,64]{0,1} parameter(1)
  row_bias = bf16[64]{0} parameter(2)
  column_scale = bf16[64]{0} parameter(3)
  ROOT fusion = bf16[64,64]{1,0} fusion(lhs, rhs, row_bias, column_scale),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_output":true,
          "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, EpilogueChainDirectToVgprOutputTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_epilogue_chain_direct_to_vgpr_output_tail

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,240]{0,1} parameter(1)
  scale = f32[240]{0} parameter(2)
  bias = f32[] parameter(3)
  dot = f32[256,240]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
  scale_broadcast = f32[256,240]{1,0} broadcast(scale), dimensions={1}
  multiply = f32[256,240]{1,0} multiply(dot, scale_broadcast)
  bias_broadcast = f32[256,240]{1,0} broadcast(bias), dimensions={}
  add = f32[256,240]{1,0} add(multiply, bias_broadcast)
  negate = f32[256,240]{1,0} negate(add)
  ROOT convert = bf16[256,240]{1,0} convert(negate)
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,240]{0,1} parameter(1)
  scale = f32[240]{0} parameter(2)
  bias = f32[] parameter(3)
  ROOT fusion = bf16[256,240]{1,0} fusion(lhs, rhs, scale, bias),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32InputsConvertedToBf16StagedOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_inputs_converted_to_bf16_staged_output

fly_gemm {
  lhs_f32 = f32[64,64]{1,0} parameter(0)
  rhs_f32 = f32[64,64]{0,1} parameter(1)
  lhs = bf16[64,64]{1,0} convert(lhs_f32)
  rhs = bf16[64,64]{0,1} convert(rhs_f32)
  dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
  ROOT convert = bf16[64,64]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[64,64]{1,0} parameter(0)
  rhs = f32[64,64]{0,1} parameter(1)
  ROOT fusion = bf16[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8", "stage_output":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32InputsConvertedToBf16LocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_inputs_converted_to_bf16_local_split

fly_gemm {
  lhs_f32 = f32[256,384]{1,0} parameter(0)
  rhs_f32 = f32[384,128]{0,1} parameter(1)
  lhs = bf16[256,384]{1,0} convert(lhs_f32)
  rhs = bf16[384,128]{0,1} convert(rhs_f32)
  dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  ROOT convert = bf16[256,128]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[256,384]{1,0} parameter(0)
  rhs = f32[384,128]{0,1} parameter(1)
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32InputsConvertedToBf16DirectToVgpr) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_inputs_converted_to_bf16_direct_to_vgpr

fly_gemm {
  lhs_f32 = f32[256,128]{1,0} parameter(0)
  rhs_f32 = f32[128,224]{0,1} parameter(1)
  lhs = bf16[256,128]{1,0} convert(lhs_f32)
  rhs = bf16[128,224]{0,1} convert(rhs_f32)
  dot = f32[256,224]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
  ROOT convert = bf16[256,224]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[256,128]{1,0} parameter(0)
  rhs = f32[128,224]{0,1} parameter(1)
  ROOT fusion = bf16[256,224]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32InputsBitcastAndConvertedToBf16) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_inputs_bitcast_and_converted_to_bf16

fly_gemm {
  lhs_f32_physical = f32[32,128]{1,0} parameter(0)
  rhs_f32_physical = f32[32,128]{1,0} parameter(1)
  lhs_f32 = f32[64,64]{1,0} bitcast(lhs_f32_physical)
  lhs = bf16[64,64]{1,0} convert(lhs_f32)
  rhs_bf16_physical = bf16[32,128]{1,0} convert(rhs_f32_physical)
  rhs = bf16[64,64]{0,1} bitcast(rhs_bf16_physical)
  dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
  ROOT convert = bf16[64,64]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[32,128]{1,0} parameter(0)
  rhs = f32[32,128]{1,0} parameter(1)
  ROOT fusion = bf16[64,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8", "stage_output":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32InputsSlicedAndConvertedToBf16LocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_inputs_sliced_and_converted_to_bf16_local_split_k

fly_gemm {
  lhs_f32_physical = f32[320,512]{1,0} parameter(0)
  rhs_f32_physical = f32[512,192]{0,1} parameter(1)
  lhs_f32 = f32[256,384]{1,0} slice(lhs_f32_physical),
      slice={[32:288], [64:448]}
  rhs_f32 = f32[384,128]{0,1} slice(rhs_f32_physical),
      slice={[64:448], [32:160]}
  lhs = bf16[256,384]{1,0} convert(lhs_f32)
  rhs = bf16[384,128]{0,1} convert(rhs_f32)
  dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  ROOT convert = bf16[256,128]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[320,512]{1,0} parameter(0)
  rhs = f32[512,192]{0,1} parameter(1)
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16DynamicSliceInputs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_dynamic_slice_inputs

fly_gemm {
  lhs_physical = bf16[96,80]{1,0} parameter(0)
  rhs_physical = bf16[80,96]{0,1} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  lhs = bf16[64,64]{1,0} dynamic-slice(
      lhs_physical, lhs_start_m, lhs_start_k),
      dynamic_slice_sizes={64,64}
  rhs = bf16[64,64]{0,1} dynamic-slice(
      rhs_physical, rhs_start_k, rhs_start_n),
      dynamic_slice_sizes={64,64}
  ROOT dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[96,80]{1,0} parameter(0)
  rhs = bf16[80,96]{0,1} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  ROOT fusion = f32[64,64]{1,0} fusion(
      lhs, rhs, lhs_start_m, lhs_start_k, rhs_start_k, rhs_start_n),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"16",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16DynamicSliceMatrixVectorInputs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_dynamic_slice_matrix_vector_inputs

fly_gemv {
  lhs_physical = bf16[96,80]{1,0} parameter(0)
  rhs_physical = bf16[80,3]{0,1} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  lhs = bf16[64,64]{1,0} dynamic-slice(
      lhs_physical, lhs_start_m, lhs_start_k),
      dynamic_slice_sizes={64,64}
  rhs = bf16[64,1]{0,1} dynamic-slice(
      rhs_physical, rhs_start_k, rhs_start_n),
      dynamic_slice_sizes={64,1}
  ROOT dot = f32[64,1]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[96,80]{1,0} parameter(0)
  rhs = bf16[80,3]{0,1} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  ROOT fusion = f32[64,1]{1,0} fusion(
      lhs, rhs, lhs_start_m, lhs_start_k, rhs_start_k, rhs_start_n),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"16",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8Mfma16) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_mfma16

fly_gemm {
  lhs = f8e4m3fnuz[64,64]{1,0} parameter(0)
  rhs = f8e4m3fnuz[64,64]{1,0} parameter(1)
  ROOT dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[64,64]{1,0} parameter(0)
  rhs = f8e4m3fnuz[64,64]{1,0} parameter(1)
  ROOT fusion = f32[64,64]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8Mfma16StagedDoubleBuffer) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_mfma16_staged_double_buffer

fly_gemm {
  lhs = f8e4m3fnuz[128,256]{1,0} parameter(0)
  rhs = f8e4m3fnuz[256,128]{1,0} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[128,256]{1,0} parameter(0)
  rhs = f8e4m3fnuz[256,128]{1,0} parameter(1)
  ROOT fusion = f32[128,128]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"128",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8",
          "stage_rhs":true, "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","128"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8Mfma16DirectLdsDoubleBuffer) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_mfma16_direct_lds_double_buffer

fly_gemm {
  lhs = f8e4m3fnuz[128,256]{1,0} parameter(0)
  rhs = f8e4m3fnuz[256,128]{0,1} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[128,256]{1,0} parameter(0)
  rhs = f8e4m3fnuz[256,128]{0,1} parameter(1)
  ROOT fusion = f32[128,128]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"128",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8",
          "stage_rhs":true, "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","128"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8DirectToVgprLhsSteadyState) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_direct_to_vgpr_lhs_steady_state

fly_gemm {
  lhs = f8e4m3fnuz[256,512]{1,0} parameter(0)
  rhs_row_major = f8e4m3fnuz[224,512]{1,0} parameter(1)
  rhs = f8e4m3fnuz[512,224]{0,1} bitcast(rhs_row_major)
  ROOT dot = f32[256,224]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[256,512]{1,0} parameter(0)
  rhs = f8e4m3fnuz[224,512]{1,0} parameter(1)
  ROOT fusion = f32[256,224]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8DirectToVgprShiftedNBoundary) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_direct_to_vgpr_shifted_n_boundary

fly_gemm {
  lhs = f8e4m3fnuz[256,512]{1,0} parameter(0)
  rhs_row_major = f8e4m3fnuz[240,512]{1,0} parameter(1)
  rhs = f8e4m3fnuz[512,240]{0,1} bitcast(rhs_row_major)
  ROOT dot = f32[256,240]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[256,512]{1,0} parameter(0)
  rhs = f8e4m3fnuz[240,512]{1,0} parameter(1)
  ROOT fusion = f32[256,240]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8ScaledF32DirectToVgprLhsSteadyState) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_scaled_f32_direct_to_vgpr_lhs_steady_state

fly_gemm {
  lhs = f8e4m3fnuz[256,512]{1,0} parameter(0)
  rhs_row_major = f8e4m3fnuz[224,512]{1,0} parameter(1)
  rhs = f8e4m3fnuz[512,224]{0,1} bitcast(rhs_row_major)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = f32[256,224]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[256,512]{1,0} parameter(0)
  rhs = f8e4m3fnuz[224,512]{1,0} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = f32[256,224]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, MixedFnuzFp8Mfma32) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_mixed_fnuz_fp8_mfma32

fly_gemm {
  lhs = f8e4m3fnuz[64,64]{1,0} parameter(0)
  rhs = f8e5m2fnuz[64,64]{0,1} parameter(1)
  ROOT dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[64,64]{1,0} parameter(0)
  rhs = f8e5m2fnuz[64,64]{0,1} parameter(1)
  ROOT fusion = f32[64,64]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_32X32X16_FP8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, MixedFnuzFp8ScaledMfma32) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_mixed_fnuz_fp8_scaled_mfma32

fly_gemm {
  lhs = f8e4m3fnuz[64,64]{1,0} parameter(0)
  rhs = f8e5m2fnuz[64,64]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = f32[64,64]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[64,64]{1,0} parameter(0)
  rhs = f8e5m2fnuz[64,64]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = f32[64,64]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_32X32X16_FP8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, MixedFnuzFp8ScaledMfma16Bf16Output) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_mixed_fnuz_fp8_scaled_mfma16_bf16_output

fly_gemm {
  lhs = f8e4m3fnuz[64,64]{1,0} parameter(0)
  rhs = f8e5m2fnuz[64,64]{1,0} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = bf16[64,64]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[64,64]{1,0} parameter(0)
  rhs = f8e5m2fnuz[64,64]{1,0} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = bf16[64,64]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, Bf16BlockScaledMfma16) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_block_scaled_mfma16

fly_gemm {
  lhs = bf16[64,128]{1,0} parameter(0)
  rhs = bf16[128,64]{0,1} parameter(1)
  lhs_scale = f32[64,4]{1,0} parameter(2)
  rhs_scale = f32[4,64]{1,0} parameter(3)
  ROOT dot = f32[64,64]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[64,128]{1,0} parameter(0)
  rhs = bf16[128,64]{0,1} parameter(1)
  lhs_scale = f32[64,4]{1,0} parameter(2)
  rhs_scale = f32[4,64]{1,0} parameter(3)
  ROOT fusion = f32[64,64]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.05}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8BlockScaledMfma16) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_block_scaled_mfma16

fly_gemm {
  lhs = f8e4m3fnuz[64,128]{1,0} parameter(0)
  rhs = f8e5m2fnuz[128,64]{0,1} parameter(1)
  lhs_scale = bf16[64,4]{1,0} parameter(2)
  rhs_scale = bf16[4,64]{1,0} parameter(3)
  ROOT dot = f32[64,64]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[64,128]{1,0} parameter(0)
  rhs = f8e5m2fnuz[128,64]{0,1} parameter(1)
  lhs_scale = bf16[64,4]{1,0} parameter(2)
  rhs_scale = bf16[4,64]{1,0} parameter(3)
  ROOT fusion = f32[64,64]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8BlockScaledDequantizedBf16Mfma16) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_block_scaled_dequantized_bf16_mfma16

fly_gemm {
  lhs = f8e4m3fnuz[64,128]{1,0} parameter(0)
  rhs = f8e5m2fnuz[128,64]{0,1} parameter(1)
  lhs_scale = f8e8m0fnu[64,4]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[4,64]{1,0} parameter(3)
  ROOT dot = f32[64,64]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[64,128]{1,0} parameter(0)
  rhs = f8e5m2fnuz[128,64]{0,1} parameter(1)
  lhs_scale = f8e8m0fnu[64,4]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[4,64]{1,0} parameter(3)
  ROOT fusion = f32[64,64]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "dequantize_block_scales":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8E8m0BlockScaledMfma32) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_e8m0_block_scaled_mfma32

fly_gemm {
  lhs = f8e4m3fnuz[64,128]{1,0} parameter(0)
  rhs = f8e4m3fnuz[64,128]{1,0} parameter(1)
  lhs_scale = f8e8m0fnu[64,4]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[64,4]{1,0} parameter(3)
  ROOT dot = f32[64,64]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[64,128]{1,0} parameter(0)
  rhs = f8e4m3fnuz[64,128]{1,0} parameter(1)
  lhs_scale = f8e8m0fnu[64,4]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[64,4]{1,0} parameter(3)
  ROOT fusion = f32[64,64]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_32X32X16_FP8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/4.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8VectorMatrixSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_vector_matrix_split_k

fly_gemv {
  lhs = f8e4m3fnuz[1,256]{1,0} parameter(0)
  rhs = f8e5m2fnuz[256,128]{1,0} parameter(1)
  ROOT dot = f32[1,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[1,256]{1,0} parameter(0)
  rhs = f8e5m2fnuz[256,128]{1,0} parameter(1)
  ROOT fusion = f32[1,128]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, Bf16TransposedSingletonLhsVectorMatrix) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transposed_singleton_lhs_vector_matrix

fly_gemv {
  lhs = bf16[257,1]{1,0} parameter(0)
  rhs = bf16[257,193]{1,0} parameter(1)
  ROOT dot = bf16[1,193]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[257,1]{1,0} parameter(0)
  rhs = bf16[257,193]{1,0} parameter(1)
  ROOT fusion = bf16[1,193]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"16", "num_warps":"4",
          "gemv_outputs_per_wave":"4"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16BatchedTransposedSingletonLhsVectorMatrix) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_transposed_singleton_lhs_vector_matrix

fly_gemv {
  lhs = bf16[3,257,1]{2,1,0} parameter(0)
  rhs = bf16[3,257,193]{2,1,0} parameter(1)
  ROOT dot = f32[3,1,193]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[3,257,1]{2,1,0} parameter(0)
  rhs = bf16[3,257,193]{2,1,0} parameter(1)
  ROOT fusion = f32[3,1,193]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"16", "num_warps":"4",
          "gemv_outputs_per_wave":"4"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16TransposedSingletonLhsMfmaLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transposed_singleton_lhs_mfma_local_split_k

fly_gemm {
  lhs = bf16[512,1]{1,0} parameter(0)
  rhs = bf16[512,128]{1,0} parameter(1)
  ROOT dot = bf16[1,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      backend_config={"sizes":["256"]}
}

ENTRY main {
  lhs = bf16[512,1]{1,0} parameter(0)
  rhs = bf16[512,128]{1,0} parameter(1)
  ROOT fusion = bf16[1,128]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"16",
          "block_k":"256", "num_warps":"2",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "preload_lds_fragments":true, "local_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, MixedFnuzFp8VectorMatrixKContiguousRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_mixed_fnuz_fp8_vector_matrix_k_contiguous_rhs

fly_gemv {
  lhs = f8e4m3fnuz[1,256]{1,0} parameter(0)
  rhs = f8e5m2fnuz[256,128]{0,1} parameter(1)
  ROOT dot = f32[1,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[1,256]{1,0} parameter(0)
  rhs = f8e5m2fnuz[256,128]{0,1} parameter(1)
  ROOT fusion = f32[1,128]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8",
          "gemv_outputs_per_wave":4, "gemv_k_vector_width":4},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8MatrixVector) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_matrix_vector

fly_gemv {
  lhs = f8e4m3fnuz[128,256]{1,0} parameter(0)
  rhs = f8e4m3fnuz[256,1]{1,0} parameter(1)
  ROOT dot = f32[128,1]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[128,256]{1,0} parameter(0)
  rhs = f8e4m3fnuz[256,1]{1,0} parameter(1)
  ROOT fusion = f32[128,1]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"16",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8BatchedGemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_batched_gemm

fly_gemm {
  lhs = f8e4m3fnuz[3,64,64]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,64,64]{2,1,0} parameter(1)
  ROOT dot = f32[3,64,64]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f8e4m3fnuz[3,64,64]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,64,64]{2,1,0} parameter(1)
  ROOT fusion = f32[3,64,64]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, UniformScaledBf16BatchedGemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_uniform_scaled_bf16_batched_gemm

fly_gemm {
  lhs = bf16[3,64,64]{2,1,0} parameter(0)
  rhs = bf16[3,64,64]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT dot = bf16[3,64,64]{2,1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[3,64,64]{2,1,0} parameter(0)
  rhs = bf16[3,64,64]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT fusion = bf16[3,64,64]{2,1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, UniformScaledF16BatchedGemmKContiguousRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_uniform_scaled_f16_batched_gemm_k_contiguous_rhs

fly_gemm {
  lhs = f16[3,64,64]{2,1,0} parameter(0)
  rhs = f16[3,64,64]{2,1,0} parameter(1)
  lhs_scale = f16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = f16[1,1,1]{2,1,0} parameter(3)
  ROOT dot = f16[3,64,64]{2,1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f16[3,64,64]{2,1,0} parameter(0)
  rhs = f16[3,64,64]{2,1,0} parameter(1)
  lhs_scale = f16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = f16[1,1,1]{2,1,0} parameter(3)
  ROOT fusion = f16[3,64,64]{2,1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, UniformScaledFnuzFp8BatchedGemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_uniform_scaled_fnuz_fp8_batched_gemm

fly_gemm {
  lhs = f8e4m3fnuz[3,64,64]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,64,64]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT dot = f32[3,64,64]{2,1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[3,64,64]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,64,64]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT fusion = f32[3,64,64]{2,1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, FnuzFp8BatchedVectorMatrixSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fnuz_fp8_batched_vector_matrix_split_k

fly_gemv {
  lhs = f8e4m3fnuz[3,1,256]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,256,128]{2,1,0} parameter(1)
  ROOT dot = f32[3,1,128]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f8e4m3fnuz[3,1,256]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,256,128]{2,1,0} parameter(1)
  ROOT fusion = f32[3,1,128]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, UniformScaledFnuzFp8VectorMatrixSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_uniform_scaled_fnuz_fp8_vector_matrix_split_k

fly_gemv {
  lhs = f8e4m3fnuz[1,256]{1,0} parameter(0)
  rhs = f8e5m2fnuz[256,128]{1,0} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = f32[1,128]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[1,256]{1,0} parameter(0)
  rhs = f8e5m2fnuz[256,128]{1,0} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = f32[1,128]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, UniformScaledFnuzFp8VectorMatrixKContiguousRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_uniform_scaled_fnuz_fp8_vector_matrix_k_contiguous_rhs

fly_gemv {
  lhs = f8e4m3fnuz[1,256]{1,0} parameter(0)
  rhs = f8e5m2fnuz[256,128]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = f32[1,128]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f8e4m3fnuz[1,256]{1,0} parameter(0)
  rhs = f8e5m2fnuz[256,128]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = f32[1,128]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8",
          "gemv_outputs_per_wave":4, "gemv_k_vector_width":4},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, UniformScaledBf16MatrixVector) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_uniform_scaled_bf16_matrix_vector

fly_gemv {
  lhs = bf16[128,256]{1,0} parameter(0)
  rhs = bf16[256,1]{1,0} parameter(1)
  lhs_scale = f32[1,1]{1,0} parameter(2)
  rhs_scale = f32[1,1]{1,0} parameter(3)
  ROOT dot = bf16[128,1]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[128,256]{1,0} parameter(0)
  rhs = bf16[256,1]{1,0} parameter(1)
  lhs_scale = f32[1,1]{1,0} parameter(2)
  rhs_scale = f32[1,1]{1,0} parameter(3)
  ROOT fusion = bf16[128,1]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"64", "block_n":"16",
          "block_k":"32", "num_warps":"8",
          "gemv_outputs_per_wave":"8", "gemv_k_vector_width":"4"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","16"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, UniformScaledF16BatchedMatrixVectorTransposedRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_uniform_scaled_f16_batched_matrix_vector_transposed_rhs

fly_gemv {
  lhs = f16[2,128,256]{2,1,0} parameter(0)
  rhs = f16[2,1,256]{2,1,0} parameter(1)
  lhs_scale = f16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = f16[1,1,1]{2,1,0} parameter(3)
  ROOT dot = f16[2,128,1]{2,1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = f16[2,128,256]{2,1,0} parameter(0)
  rhs = f16[2,1,256]{2,1,0} parameter(1)
  lhs_scale = f16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = f16[1,1,1]{2,1,0} parameter(3)
  ROOT fusion = f16[2,128,1]{2,1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"64", "block_n":"16",
          "block_k":"32", "num_warps":"8",
          "gemv_outputs_per_wave":"8", "gemv_k_vector_width":"4"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","64","16"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, S4DequantizedRhsMfma16) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_s4_dequantized_rhs_mfma16

fly_gemm {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs.s4 = s4[64,64]{1,0:E(4)} parameter(1)
  rhs.s8 = s8[64,64]{1,0} convert(rhs.s4)
  rhs.bf16 = bf16[64,64]{1,0} convert(rhs.s8)
  ROOT dot = f32[64,64]{1,0} dot(lhs, rhs.bf16),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = s4[64,64]{1,0:E(4)} parameter(1)
  ROOT fusion = f32[64,64]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, S4DequantizedKContiguousRhsMfma16) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_s4_dequantized_k_contiguous_rhs_mfma16

fly_gemm {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs.s4 = s4[64,64]{0,1:E(4)} parameter(1)
  rhs.s8 = s8[64,64]{0,1} convert(rhs.s4)
  rhs.bf16 = bf16[64,64]{0,1} convert(rhs.s8)
  ROOT dot = f32[64,64]{1,0} dot(lhs, rhs.bf16),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = s4[64,64]{0,1:E(4)} parameter(1)
  ROOT fusion = f32[64,64]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"32",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","32"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, S4DequantizedKContiguousRhsWideTile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_s4_dequantized_k_contiguous_rhs_wide_tile

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs.s4 = s4[128,240]{0,1:E(4)} parameter(1)
  rhs.s8 = s8[128,240]{0,1} convert(rhs.s4)
  rhs.bf16 = bf16[128,240]{0,1} convert(rhs.s8)
  ROOT dot = f32[256,240]{1,0} dot(lhs, rhs.bf16),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = s4[128,240]{0,1:E(4)} parameter(1)
  ROOT fusion = f32[256,240]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "schedule_instructions":true, "preload_lds_fragments":true,
          "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, S4DequantizedKContiguousRhsDirectToVgpr) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_s4_dequantized_k_contiguous_rhs_direct_to_vgpr

fly_gemm {
  lhs = bf16[240,128]{1,0} parameter(0)
  rhs.s4 = s4[128,256]{0,1:E(4)} parameter(1)
  rhs.s8 = s8[128,256]{0,1} convert(rhs.s4)
  rhs.bf16 = bf16[128,256]{0,1} convert(rhs.s8)
  ROOT dot = f32[240,256]{1,0} dot(lhs, rhs.bf16),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[240,128]{1,0} parameter(0)
  rhs = s4[128,256]{0,1:E(4)} parameter(1)
  ROOT fusion = f32[240,256]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"224", "block_n":"256",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "schedule_instructions":true, "preload_lds_fragments":true,
          "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["224","256"]}],
          "num_stages":"2", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, S4DequantizedLhsMfma32) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_s4_dequantized_lhs_mfma32

fly_gemm {
  lhs.s4 = s4[64,64]{1,0:E(4)} parameter(0)
  lhs.bf16 = bf16[64,64]{1,0} convert(lhs.s4)
  rhs = bf16[64,64]{0,1} parameter(1)
  ROOT dot = f32[64,64]{1,0} dot(lhs.bf16, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = s4[64,64]{1,0:E(4)} parameter(0)
  rhs = bf16[64,64]{0,1} parameter(1)
  ROOT fusion = f32[64,64]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"1",
          "mfma_atom":"FLY_MFMA_32X32X8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32InputsConcatenatedAndConvertedLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_inputs_concatenated_and_converted_local_split_k

fly_gemm {
  lhs0_f32 = f32[128,384]{1,0} parameter(0)
  lhs1_f32 = f32[128,384]{1,0} parameter(1)
  rhs0_f32 = f32[384,64]{0,1} parameter(2)
  rhs1_f32 = f32[384,64]{0,1} parameter(3)
  lhs0 = bf16[128,384]{1,0} convert(lhs0_f32)
  lhs1 = bf16[128,384]{1,0} convert(lhs1_f32)
  rhs0 = bf16[384,64]{0,1} convert(rhs0_f32)
  rhs1 = bf16[384,64]{0,1} convert(rhs1_f32)
  lhs = bf16[256,384]{1,0} concatenate(lhs0, lhs1), dimensions={0}
  rhs = bf16[384,128]{0,1} concatenate(rhs0, rhs1), dimensions={1}
  dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  ROOT convert = bf16[256,128]{1,0} convert(dot)
}

ENTRY main {
  lhs0 = f32[128,384]{1,0} parameter(0)
  lhs1 = f32[128,384]{1,0} parameter(1)
  rhs0 = f32[384,64]{0,1} parameter(2)
  rhs1 = f32[384,64]{0,1} parameter(3)
  ROOT fusion = bf16[256,128]{1,0} fusion(lhs0, lhs1, rhs0, rhs1),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16InputsConcatenatedLocalSplitKAsyncCopies) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_inputs_concatenated_local_split_k_async_copies

fly_gemm {
  lhs0 = bf16[128,384]{1,0} parameter(0)
  lhs1 = bf16[128,384]{1,0} parameter(1)
  rhs0 = bf16[384,64]{0,1} parameter(2)
  rhs1 = bf16[384,64]{0,1} parameter(3)
  lhs = bf16[256,384]{1,0} concatenate(lhs0, lhs1), dimensions={0}
  rhs = bf16[384,128]{0,1} concatenate(rhs0, rhs1), dimensions={1}
  dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  ROOT convert = bf16[256,128]{1,0} convert(dot)
}

ENTRY main {
  lhs0 = bf16[128,384]{1,0} parameter(0)
  lhs1 = bf16[128,384]{1,0} parameter(1)
  rhs0 = bf16[384,64]{0,1} parameter(2)
  rhs1 = bf16[384,64]{0,1} parameter(3)
  ROOT fusion = bf16[256,128]{1,0} fusion(lhs0, lhs1, rhs0, rhs1),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16TransposedUnevenConcatRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transposed_uneven_concat_rhs

fly_gemm {
  lhs = bf16[64,128]{1,0} parameter(0)
  rhs0 = bf16[128,128]{1,0} parameter(1)
  rhs1 = bf16[128,64]{1,0} parameter(2)
  concatenated = bf16[128,192]{1,0}
      concatenate(rhs0, rhs1), dimensions={1}
  rhs = bf16[192,128]{0,1}
      transpose(concatenated), dimensions={1,0}
  ROOT dot = bf16[64,192]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[64,128]{1,0} parameter(0)
  rhs0 = bf16[128,128]{1,0} parameter(1)
  rhs1 = bf16[128,64]{1,0} parameter(2)
  ROOT fusion = bf16[64,192]{1,0} fusion(lhs, rhs0, rhs1),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16TransposedUnevenConcatRhsStaged) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transposed_uneven_concat_rhs_staged

fly_gemm {
  lhs = bf16[64,128]{1,0} parameter(0)
  rhs0 = bf16[128,128]{1,0} parameter(1)
  rhs1 = bf16[128,64]{1,0} parameter(2)
  concatenated = bf16[128,192]{1,0}
      concatenate(rhs0, rhs1), dimensions={1}
  rhs = bf16[192,128]{0,1}
      transpose(concatenated), dimensions={1,0}
  ROOT dot = bf16[64,192]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[64,128]{1,0} parameter(0)
  rhs0 = bf16[128,128]{1,0} parameter(1)
  rhs1 = bf16[128,64]{1,0} parameter(2)
  ROOT fusion = bf16[64,192]{1,0} fusion(lhs, rhs0, rhs1),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"64",
          "block_k":"128", "num_warps":"2",
          "mfma_atom":"FLY_MFMA_32X32X8", "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true,
          "rolling_refill":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","64"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32DotWithNarrowingBf16Epilogue) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_dot_bf16_epilogue

fly_gemm {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = bf16[64,64]{1,0} parameter(1)
  dot = f32[64,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
  ROOT convert = bf16[64,64]{1,0} convert(dot)
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
          "mfma_atom":"FLY_MFMA_32X32X8", "stage_output":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32DotNarrowingLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_dot_bf16_epilogue_local_split

fly_gemm {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  dot = f32[256,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  ROOT convert = bf16[256,128]{1,0} convert(dot)
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32DotNarrowingDirectToVgpr) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_dot_bf16_epilogue_direct_to_vgpr

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,224]{0,1} parameter(1)
  dot = f32[256,224]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
  ROOT convert = bf16[256,224]{1,0} convert(dot)
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,224]{0,1} parameter(1)
  ROOT fusion = bf16[256,224]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32DotNarrowingShortTileWgmTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_dot_bf16_epilogue_short_tile_wgm_tail

fly_gemm {
  lhs = bf16[96,256]{1,0} parameter(0)
  rhs = bf16[256,80]{0,1} parameter(1)
  dot = f32[96,80]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  ROOT convert = bf16[96,80]{1,0} convert(dot)
}

ENTRY main {
  lhs = bf16[96,256]{1,0} parameter(0)
  rhs = bf16[256,80]{0,1} parameter(1)
  ROOT fusion = bf16[96,80]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"32", "block_n":"16",
          "block_k":"128", "num_warps":"2",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "workgroup_mapping_n":4},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["32","16"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16ScaledMma64x64x64) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_scaled_gemm

fly_gemm {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = bf16[64,64]{1,0} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = bf16[64,64]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = bf16[64,64]{1,0} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = bf16[64,64]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16ScaledMmaWithF32Output) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_scaled_gemm_f32_output

fly_gemm {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = bf16[64,64]{1,0} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = f32[64,64]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[64,64]{1,0} parameter(0)
  rhs = bf16[64,64]{1,0} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = f32[64,64]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16ScaledLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_scaled_local_split_k

fly_gemm {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = bf16[256,128]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[256,384]{1,0} parameter(0)
  rhs = bf16[384,128]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = bf16[256,128]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale),
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
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16ScaledDirectToVgprA) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_scaled_direct_to_vgpr_a

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,224]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = bf16[256,224]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,224]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = bf16[256,224]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16ScaledDirectToVgprB) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_scaled_direct_to_vgpr_b

fly_gemm {
  lhs = bf16[224,128]{1,0} parameter(0)
  rhs = bf16[128,256]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = bf16[224,256]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[224,128]{1,0} parameter(0)
  rhs = bf16[128,256]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = bf16[224,256]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"224", "block_n":"256",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["224","256"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

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

TEST_F(FlyGemmDeviceTest, Bf16Mfma32RowMajorRhsSquarePipeline) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_mfma32_row_major_rhs_square_pipeline

fly_gemm {
  lhs = bf16[256,1024]{1,0} parameter(0)
  rhs = bf16[1024,256]{1,0} parameter(1)
  ROOT dot = bf16[256,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[256,1024]{1,0} parameter(0)
  rhs = bf16[1024,256]{1,0} parameter(1)
  ROOT fusion = bf16[256,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"128",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8", "stage_output":true,
          "stage_rhs":true, "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","128"]}],
          "num_stages":"2", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16Mfma32RowMajorRhsSquareDirectOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_mfma32_row_major_rhs_square_direct_output

fly_gemm {
  lhs = bf16[128,128]{1,0} parameter(0)
  rhs = bf16[128,128]{1,0} parameter(1)
  ROOT dot = bf16[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[128,128]{1,0} parameter(0)
  rhs = bf16[128,128]{1,0} parameter(1)
  ROOT fusion = bf16[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"128",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","128"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16Mfma32RowMajorRhsWidePackedOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_mfma32_row_major_rhs_wide_packed_output

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,512]{1,0} parameter(1)
  ROOT dot = bf16[256,512]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  one = bf16[] constant(1)
  lhs = bf16[256,128]{1,0} broadcast(one), dimensions={}
  rhs = bf16[128,512]{1,0} broadcast(one), dimensions={}
  ROOT fusion = bf16[256,512]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"256",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","256"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.1, /*arel=*/0.01}));
}

TEST_F(FlyGemmDeviceTest, Bf16Mfma32RowMajorRhsWidePackedOutputNonuniform) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_mfma32_row_major_rhs_wide_packed_output_nonuniform

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,512]{1,0} parameter(1)
  ROOT dot = bf16[256,512]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,512]{1,0} parameter(1)
  ROOT fusion = bf16[256,512]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"256",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","256"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16Mfma32RowMajorRhsWideScalarEpilogue) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_mfma32_row_major_rhs_wide_scalar_epilogue

fly_gemm {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,512]{1,0} parameter(1)
  dot = bf16[256,512]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
  scale = bf16[] constant(0.125)
  scale_broadcast = bf16[256,512]{1,0} broadcast(scale), dimensions={}
  ROOT scaled = bf16[256,512]{1,0} multiply(dot, scale_broadcast)
}

ENTRY main {
  lhs = bf16[256,128]{1,0} parameter(0)
  rhs = bf16[128,512]{1,0} parameter(1)
  ROOT fusion = bf16[256,512]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"256",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","256"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16BatchedMfma32RowMajorRhsK32MaskedTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_mfma32_row_major_rhs_k32_masked_tail

fly_gemm {
  lhs = bf16[2,129,95]{2,1,0} parameter(0)
  rhs = bf16[2,95,131]{2,1,0} parameter(1)
  ROOT dot = bf16[2,129,131]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[2,129,95]{2,1,0} parameter(0)
  rhs = bf16[2,95,131]{2,1,0} parameter(1)
  ROOT fusion = bf16[2,129,131]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"128",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_32X32X8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","128","128"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1.0, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16NonStagedWorkgroupMappingAndWaveGrid) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_non_staged_workgroup_mapping_and_wave_grid

fly_gemm {
  lhs = bf16[128,32]{1,0} parameter(0)
  rhs = bf16[32,640]{1,0} parameter(1)
  ROOT dot = bf16[128,640]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[128,32]{1,0} parameter(0)
  rhs = bf16[32,640]{1,0} parameter(1)
  ROOT fusion = bf16[128,640]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"320",
          "block_k":"32", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "workgroup_mapping_n":4},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","320"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16BatchedGemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_gemm

fly_gemm {
  lhs = bf16[3,128,256]{2,1,0} parameter(0)
  rhs = bf16[3,256,192]{2,1,0} parameter(1)
  ROOT dot = bf16[3,128,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[3,128,256]{2,1,0} parameter(0)
  rhs = bf16[3,256,192]{2,1,0} parameter(1)
  ROOT fusion = bf16[3,128,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16BatchedGemmStagedOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_gemm_staged_output

fly_gemm {
  lhs = bf16[2,64,128]{2,1,0} parameter(0)
  rhs = bf16[2,256,128]{2,1,0} parameter(1)
  ROOT dot = bf16[2,64,256]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[2,64,128]{2,1,0} parameter(0)
  rhs = bf16[2,256,128]{2,1,0} parameter(1)
  ROOT fusion = bf16[2,64,256]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"256",
          "block_k":"32", "num_warps":"8", "stage_output":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","64","256"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, F16BatchedGemmTransposedRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_batched_gemm_transposed_rhs

fly_gemm {
  lhs = f16[3,128,256]{2,1,0} parameter(0)
  rhs = f16[3,192,256]{2,1,0} parameter(1)
  ROOT dot = f16[3,128,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = f16[3,128,256]{2,1,0} parameter(0)
  rhs = f16[3,192,256]{2,1,0} parameter(1)
  ROOT fusion = f16[3,128,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16BatchedBroadcastEpilogueChain) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_batched_broadcast_epilogue_chain

fly_gemm {
  lhs = f16[3,128,256]{2,1,0} parameter(0)
  rhs = f16[3,256,192]{2,1,0} parameter(1)
  column_bias = f16[192]{0} parameter(2)
  batch_scale = f16[3]{0} parameter(3)
  row_bias = f16[128]{0} parameter(4)
  dot = f16[3,128,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["64"]}
  column_broadcast = f16[3,128,192]{2,1,0} broadcast(column_bias),
      dimensions={2}
  add_column = f16[3,128,192]{2,1,0} add(dot, column_broadcast)
  batch_broadcast = f16[3,128,192]{2,1,0} broadcast(batch_scale),
      dimensions={0}
  multiply_batch = f16[3,128,192]{2,1,0} multiply(add_column,
      batch_broadcast)
  row_broadcast = f16[3,128,192]{2,1,0} broadcast(row_bias), dimensions={1}
  ROOT add_row = f16[3,128,192]{2,1,0} add(multiply_batch, row_broadcast)
}

ENTRY main {
  lhs = f16[3,128,256]{2,1,0} parameter(0)
  rhs = f16[3,256,192]{2,1,0} parameter(1)
  column_bias = f16[192]{0} parameter(2)
  batch_scale = f16[3]{0} parameter(3)
  row_bias = f16[128]{0} parameter(4)
  ROOT fusion = f16[3,128,192]{2,1,0} fusion(lhs, rhs, column_bias,
      batch_scale, row_bias), kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.75, /*arel=*/0.05}));
}

TEST_F(FlyGemmDeviceTest, F16BatchedDirectToVgprShortKShiftedNBoundary) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_batched_direct_to_vgpr_shifted_n_boundary

fly_gemm {
  lhs = f16[2,256,128]{2,1,0} parameter(0)
  rhs = f16[2,128,240]{1,2,0} parameter(1)
  ROOT dot = f16[2,256,240]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = f16[2,256,128]{2,1,0} parameter(0)
  rhs = f16[2,128,240]{1,2,0} parameter(1)
  ROOT fusion = f16[2,256,240]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16BatchedDirectToVgprShiftedNBoundary) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_batched_direct_to_vgpr_shifted_n_boundary

fly_gemm {
  lhs = f16[2,256,256]{2,1,0} parameter(0)
  rhs = f16[2,256,240]{1,2,0} parameter(1)
  ROOT dot = f16[2,256,240]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = f16[2,256,256]{2,1,0} parameter(0)
  rhs = f16[2,256,240]{1,2,0} parameter(1)
  ROOT fusion = f16[2,256,240]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"256", "block_n":"224",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","256","224"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
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

TEST_F(FlyGemmDeviceTest, Bf16RowMajorRhsMfma32RollingRefill) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_row_major_rhs_mfma32_rolling_refill

fly_gemm {
  lhs = bf16[128,384]{1,0} parameter(0)
  rhs = bf16[384,64]{1,0} parameter(1)
  ROOT dot = bf16[128,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[128,384]{1,0} parameter(0)
  rhs = bf16[384,64]{1,0} parameter(1)
  ROOT fusion = bf16[128,64]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"64",
          "block_k":"128", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_32X32X8", "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true,
          "rolling_refill":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","64"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16RowMajorRhsMfma32RollingRefillBiasRelu) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_row_major_rhs_mfma32_rolling_refill_bias_relu

fly_gemm {
  lhs = bf16[128,384]{1,0} parameter(0)
  rhs = bf16[384,64]{1,0} parameter(1)
  bias = f32[64]{0} parameter(2)
  zero = f32[] constant(0)
  dot = f32[128,64]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
  bias_broadcast = f32[128,64]{1,0} broadcast(bias), dimensions={1}
  add = f32[128,64]{1,0} add(dot, bias_broadcast)
  zero_broadcast = f32[128,64]{1,0} broadcast(zero), dimensions={}
  maximum = f32[128,64]{1,0} maximum(add, zero_broadcast)
  ROOT convert = bf16[128,64]{1,0} convert(maximum)
}

ENTRY main {
  lhs = bf16[128,384]{1,0} parameter(0)
  rhs = bf16[384,64]{1,0} parameter(1)
  bias = f32[64]{0} parameter(2)
  ROOT fusion = bf16[128,64]{1,0} fusion(lhs, rhs, bias),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"64",
          "block_k":"128", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_32X32X8", "waves_per_eu":2,
          "stage_rhs":true, "preload_lds_fragments":true,
          "single_buffer_lds":true, "rolling_refill":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["128","64"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1",
          "waves_per_eu":2}}}
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

TEST_F(FlyGemmDeviceTest, Bf16CurrentGlobalSplitKOperandLayout) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_current_global_split_k_layout

fly_gemm {
  lhs = bf16[64,2,512]{2,1,0} parameter(0)
  rhs = bf16[2,512,64]{2,1,0} parameter(1)
  ROOT dot = f32[2,64,64]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[64,2,512]{2,1,0} parameter(0)
  rhs = bf16[2,512,64]{2,1,0} parameter(1)
  ROOT fusion = f32[2,64,64]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"32",
          "block_k":"128", "num_warps":"2",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","64","32"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16CurrentGlobalSplitKMfma32SingleBuffer) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_current_global_split_k_mfma32

fly_gemm {
  lhs = bf16[128,2,128]{2,1,0} parameter(0)
  rhs = bf16[2,128,64]{2,1,0} parameter(1)
  ROOT dot = f32[2,128,64]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[128,2,128]{2,1,0} parameter(0)
  rhs = bf16[2,128,64]{2,1,0} parameter(1)
  ROOT fusion = f32[2,128,64]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"128", "block_n":"64",
          "block_k":"128", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_32X32X8",
          "schedule_instructions":true, "stage_rhs":true,
          "preload_lds_fragments":true, "single_buffer_lds":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","128","64"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
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
          "preload_lds_fragments":true, "direct_to_vgpr":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["224","256"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.01}));
}

TEST_F(FlyGemmDeviceTest, Bf16DirectToVgprLhsSteadyState) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_direct_to_vgpr_lhs_steady_state

fly_gemm {
  lhs = bf16[256,256]{1,0} parameter(0)
  rhs_row_major = bf16[224,256]{1,0} parameter(1)
  rhs = bf16[256,224]{0,1} bitcast(rhs_row_major)
  ROOT dot = bf16[256,224]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[256,256]{1,0} parameter(0)
  rhs = bf16[224,256]{1,0} parameter(1)
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

TEST_F(FlyGemmDeviceTest, Bf16VectorMatrixTwoWaveMfmaLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_vector_matrix_two_wave_mfma_local_split_k

fly_gemm {
  lhs = bf16[1,512]{1,0} parameter(0)
  rhs = bf16[512,32]{1,0} parameter(1)
  dot = f32[1,32]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["256"]}
  ROOT convert = bf16[1,32]{1,0} convert(dot)
}

ENTRY main {
  lhs = bf16[1,512]{1,0} parameter(0)
  rhs = bf16[512,32]{1,0} parameter(1)
  ROOT fusion = bf16[1,32]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"16",
          "block_k":"256", "num_warps":"2",
          "mfma_atom":"FLY_MFMA_16X16X16", "waves_per_eu":2,
          "stage_rhs":true, "preload_lds_fragments":true,
          "local_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1",
          "waves_per_eu":2}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16VectorMatrixFourWaveMfmaLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_vector_matrix_four_wave_mfma_local_split_k

fly_gemm {
  lhs = bf16[1,512]{1,0} parameter(0)
  rhs = bf16[512,32]{1,0} parameter(1)
  dot = f32[1,32]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["256"]}
  ROOT convert = bf16[1,32]{1,0} convert(dot)
}

ENTRY main {
  lhs = bf16[1,512]{1,0} parameter(0)
  rhs = bf16[512,32]{1,0} parameter(1)
  ROOT fusion = bf16[1,32]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"16",
          "block_k":"256", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "waves_per_eu":2,
          "stage_rhs":true, "preload_lds_fragments":true,
          "local_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1",
          "waves_per_eu":2}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest,
       F16VectorMatrixFourWaveMfmaLocalSplitKKContiguousRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_vector_matrix_four_wave_mfma_local_split_k_k_contiguous_rhs

fly_gemm {
  lhs = f16[1,512]{1,0} parameter(0)
  rhs = f16[512,32]{0,1} parameter(1)
  ROOT dot = f16[1,32]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["256"]}
}

ENTRY main {
  lhs = f16[1,512]{1,0} parameter(0)
  rhs = f16[512,32]{0,1} parameter(1)
  ROOT fusion = f16[1,32]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"16",
          "block_k":"256", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "waves_per_eu":2,
          "stage_rhs":true, "preload_lds_fragments":true,
          "local_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1",
          "waves_per_eu":2}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16VectorMatrixFourWaveMfmaLocalSplitKEpilogue) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_vector_matrix_four_wave_mfma_local_split_k_epilogue

fly_gemm {
  lhs = bf16[1,512]{1,0} parameter(0)
  rhs = bf16[512,32]{1,0} parameter(1)
  bias = bf16[32]{0} parameter(2)
  zero = bf16[] constant(0)
  dot = f32[1,32]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["256"]}
  narrowed = bf16[1,32]{1,0} convert(dot)
  bias_broadcast = bf16[1,32]{1,0} broadcast(bias), dimensions={1}
  added = bf16[1,32]{1,0} add(narrowed, bias_broadcast)
  zero_broadcast = bf16[1,32]{1,0} broadcast(zero), dimensions={}
  ROOT maximum = bf16[1,32]{1,0} maximum(added, zero_broadcast)
}

ENTRY main {
  lhs = bf16[1,512]{1,0} parameter(0)
  rhs = bf16[512,32]{1,0} parameter(1)
  bias = bf16[32]{0} parameter(2)
  ROOT fusion = bf16[1,32]{1,0} fusion(lhs, rhs, bias),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"16",
          "block_k":"256", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "waves_per_eu":2,
          "stage_rhs":true, "preload_lds_fragments":true,
          "local_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","16"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1",
          "waves_per_eu":2}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest,
       Bf16BatchedVectorMatrixFourWaveMfmaLocalSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_vector_matrix_four_wave_mfma_local_split_k

fly_gemm {
  lhs = bf16[2,1,512]{2,1,0} parameter(0)
  rhs = bf16[2,512,32]{2,1,0} parameter(1)
  dot = f32[2,1,32]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["256"]}
  ROOT convert = bf16[2,1,32]{2,1,0} convert(dot)
}

ENTRY main {
  lhs = bf16[2,1,512]{2,1,0} parameter(0)
  rhs = bf16[2,512,32]{2,1,0} parameter(1)
  ROOT fusion = bf16[2,1,32]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"16",
          "block_k":"256", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "waves_per_eu":2,
          "stage_rhs":true, "preload_lds_fragments":true,
          "local_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","16"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1",
          "waves_per_eu":2}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16VectorMatrixOddKAndOutputTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_vector_matrix_odd_k_output_tail

fly_gemv {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{0,1} parameter(1)
  ROOT dot = bf16[1,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{0,1} parameter(1)
  ROOT fusion = bf16[1,111]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "gemv_outputs_per_wave":"4", "gemv_k_vector_width":"1"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16SplitVectorMatrixOddKAndOutputTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_split_vector_matrix_odd_k_output_tail

fly_gemv {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{1,0} parameter(1)
  ROOT dot = bf16[1,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{1,0} parameter(1)
  ROOT fusion = bf16[1,111]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"128",
          "block_k":"32", "num_warps":"8", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","128"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16MatrixVectorOddKAndOutputTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_matrix_vector_odd_k_output_tail

fly_gemv {
  lhs = f16[79,127]{1,0} parameter(0)
  rhs = f16[127,1]{1,0} parameter(1)
  ROOT dot = f16[79,1]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f16[79,127]{1,0} parameter(0)
  rhs = f16[127,1]{1,0} parameter(1)
  ROOT fusion = f16[79,1]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"64", "block_n":"16",
          "block_k":"32", "num_warps":"8",
          "gemv_outputs_per_wave":"8", "gemv_k_vector_width":"1"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","16"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16VectorMatrixColumnAndScalarEpilogue) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_vector_matrix_column_and_scalar_epilogue

fly_gemv {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{1,0} parameter(1)
  bias = bf16[111]{0} parameter(2)
  zero = bf16[] constant(0)
  dot = f32[1,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  narrowed = bf16[1,111]{1,0} convert(dot)
  bias_broadcast = bf16[1,111]{1,0} broadcast(bias), dimensions={1}
  add = bf16[1,111]{1,0} add(narrowed, bias_broadcast)
  zero_broadcast = bf16[1,111]{1,0} broadcast(zero), dimensions={}
  ROOT maximum = bf16[1,111]{1,0} maximum(add, zero_broadcast)
}

ENTRY main {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{1,0} parameter(1)
  bias = bf16[111]{0} parameter(2)
  ROOT fusion = bf16[1,111]{1,0} fusion(lhs, rhs, bias), kind=kCustom,
      calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"128",
          "block_k":"32", "num_warps":"8", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","128"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16MatrixVectorRowAndOrderedScalarEpilogue) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_matrix_vector_row_and_ordered_scalar_epilogue

fly_gemv {
  lhs = f16[79,127]{1,0} parameter(0)
  rhs = f16[127,1]{1,0} parameter(1)
  scale = f16[79]{0} parameter(2)
  ceiling = f16[] parameter(3)
  dot = f32[79,1]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  narrowed = f16[79,1]{1,0} convert(dot)
  scale_broadcast = f16[79,1]{1,0} broadcast(scale), dimensions={0}
  multiply = f16[79,1]{1,0} multiply(narrowed, scale_broadcast)
  ceiling_broadcast = f16[79,1]{1,0} broadcast(ceiling), dimensions={}
  ROOT subtract = f16[79,1]{1,0} subtract(ceiling_broadcast, multiply)
}

ENTRY main {
  lhs = f16[79,127]{1,0} parameter(0)
  rhs = f16[127,1]{1,0} parameter(1)
  scale = f16[79]{0} parameter(2)
  ceiling = f16[] parameter(3)
  ROOT fusion = f16[79,1]{1,0} fusion(lhs, rhs, scale, ceiling),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"64", "block_n":"16",
          "block_k":"32", "num_warps":"8",
          "gemv_outputs_per_wave":"8", "gemv_k_vector_width":"1"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["64","16"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.05}));
}

TEST_F(FlyGemmDeviceTest, UniformScaledFnuzFp8BatchedGemvEpilogue) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_uniform_scaled_fnuz_fp8_batched_gemv_epilogue

fly_gemv {
  lhs = f8e4m3fnuz[3,1,256]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,256,128]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  batch_scale = bf16[3]{0} parameter(4)
  column_bias = bf16[128]{0} parameter(5)
  dot = f32[3,1,128]{2,1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["32"]}
  narrowed = bf16[3,1,128]{2,1,0} convert(dot)
  batch_broadcast = bf16[3,1,128]{2,1,0} broadcast(batch_scale),
      dimensions={0}
  multiply = bf16[3,1,128]{2,1,0} multiply(narrowed, batch_broadcast)
  column_broadcast = bf16[3,1,128]{2,1,0} broadcast(column_bias),
      dimensions={2}
  ROOT add = bf16[3,1,128]{2,1,0} add(multiply, column_broadcast)
}

ENTRY main {
  lhs = f8e4m3fnuz[3,1,256]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,256,128]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  batch_scale = bf16[3]{0} parameter(4)
  column_bias = bf16[128]{0} parameter(5)
  ROOT fusion = bf16[3,1,128]{2,1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale, batch_scale, column_bias),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X32_FP8", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2.0, /*arel=*/0.1}));
}

TEST_F(FlyGemmDeviceTest, Bf16BatchedVectorMatrix) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_vector_matrix

fly_gemv {
  lhs = bf16[3,1,256]{2,1,0} parameter(0)
  rhs = bf16[3,256,192]{2,1,0} parameter(1)
  ROOT dot = bf16[3,1,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[3,1,256]{2,1,0} parameter(0)
  rhs = bf16[3,256,192]{2,1,0} parameter(1)
  ROOT fusion = bf16[3,1,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16BatchedVectorMatrixKContiguousRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_vector_matrix_k_contiguous_rhs

fly_gemv {
  lhs = bf16[3,1,256]{2,1,0} parameter(0)
  rhs = bf16[3,256,192]{1,2,0} parameter(1)
  ROOT dot = bf16[3,1,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[3,1,256]{2,1,0} parameter(0)
  rhs = bf16[3,256,192]{1,2,0} parameter(1)
  ROOT fusion = bf16[3,1,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{
          "block_m":"16", "block_n":"64", "block_k":"32",
          "num_warps":"4", "mfma_atom":"FLY_MFMA_16X16X16",
          "gemv_outputs_per_wave":4, "gemv_k_vector_width":2},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16SingleBatchVectorMatrixKContiguousRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_single_batch_vector_matrix_k_contiguous_rhs

fly_gemv {
  lhs = bf16[1,1,256]{2,1,0} parameter(0)
  rhs = bf16[1,256,64]{1,2,0} parameter(1)
  ROOT dot = bf16[1,1,64]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[1,1,256]{2,1,0} parameter(0)
  rhs = bf16[1,256,64]{1,2,0} parameter(1)
  ROOT fusion = bf16[1,1,64]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{
          "block_m":"16", "block_n":"2", "block_k":"32",
          "num_warps":"1", "mfma_atom":"FLY_MFMA_16X16X16",
          "gemv_outputs_per_wave":2, "gemv_k_vector_width":1},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","2"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16BatchedVectorMatrixTransposedRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_vector_matrix_transposed_rhs

fly_gemv {
  lhs = bf16[2,1,256]{2,1,0} parameter(0)
  rhs = bf16[2,192,256]{1,2,0} parameter(1)
  ROOT dot = bf16[2,1,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  lhs = bf16[2,1,256]{2,1,0} parameter(0)
  rhs = bf16[2,192,256]{1,2,0} parameter(1)
  ROOT fusion = bf16[2,1,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","64"]}],
          "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16BatchedMatrixVectorTransposedRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_matrix_vector_transposed_rhs

fly_gemv {
  lhs = bf16[2,128,256]{2,1,0} parameter(0)
  rhs = bf16[2,1,256]{2,1,0} parameter(1)
  ROOT dot = bf16[2,128,1]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  lhs = bf16[2,128,256]{2,1,0} parameter(0)
  rhs = bf16[2,1,256]{2,1,0} parameter(1)
  ROOT fusion = bf16[2,128,1]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","16"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, F16BatchedVectorMatrixKContiguousRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_batched_vector_matrix_k_contiguous_rhs

fly_gemv {
  lhs = f16[3,1,256]{2,1,0} parameter(0)
  rhs = f16[3,256,192]{1,2,0} parameter(1)
  ROOT dot = f16[3,1,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f16[3,1,256]{2,1,0} parameter(0)
  rhs = f16[3,256,192]{1,2,0} parameter(1)
  ROOT fusion = f16[3,1,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{
          "block_m":"16", "block_n":"64", "block_k":"32",
          "num_warps":"4", "mfma_atom":"FLY_MFMA_16X16X16",
          "gemv_outputs_per_wave":4, "gemv_k_vector_width":2},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16BatchedVectorMatrixSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_batched_vector_matrix_split_k

fly_gemv {
  lhs = f16[2,1,256]{2,1,0} parameter(0)
  rhs = f16[2,256,128]{2,1,0} parameter(1)
  ROOT dot = f16[2,1,128]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f16[2,1,256]{2,1,0} parameter(0)
  rhs = f16[2,256,128]{2,1,0} parameter(1)
  ROOT fusion = f16[2,1,128]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F16BatchedMatrixVectorTransposedRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_batched_matrix_vector_transposed_rhs

fly_gemv {
  lhs = f16[2,128,256]{2,1,0} parameter(0)
  rhs = f16[2,1,256]{2,1,0} parameter(1)
  ROOT dot = f16[2,128,1]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  lhs = f16[2,128,256]{2,1,0} parameter(0)
  rhs = f16[2,1,256]{2,1,0} parameter(1)
  ROOT fusion = f16[2,128,1]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"64", "block_n":"16",
          "block_k":"32", "num_warps":"8",
          "gemv_outputs_per_wave":"8", "gemv_k_vector_width":"4"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","64","16"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, F32InputsConvertedToBf16VectorMatrix) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_inputs_converted_to_bf16_vector_matrix

fly_gemv {
  lhs_f32 = f32[1,256]{1,0} parameter(0)
  rhs_f32 = f32[256,256]{1,0} parameter(1)
  lhs = bf16[1,256]{1,0} convert(lhs_f32)
  rhs = bf16[256,256]{1,0} convert(rhs_f32)
  ROOT dot = bf16[1,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f32[1,256]{1,0} parameter(0)
  rhs = f32[256,256]{1,0} parameter(1)
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

TEST_F(FlyGemmDeviceTest, F32InputsConvertedToBf16MatrixVector) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_inputs_converted_to_bf16_matrix_vector

fly_gemv {
  lhs_f32 = f32[256,256]{1,0} parameter(0)
  rhs_f32 = f32[256,1]{1,0} parameter(1)
  lhs = bf16[256,256]{1,0} convert(lhs_f32)
  rhs = bf16[256,1]{1,0} convert(rhs_f32)
  ROOT dot = bf16[256,1]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f32[256,256]{1,0} parameter(0)
  rhs = f32[256,1]{1,0} parameter(1)
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

TEST_F(FlyGemmDeviceTest, Bf16PredicatedSmallMGemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_predicated_small_m_gemm

fly_gemm {
  lhs = bf16[4,256]{1,0} parameter(0)
  rhs = bf16[256,192]{1,0} parameter(1)
  ROOT dot = bf16[4,192]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[4,256]{1,0} parameter(0)
  rhs = bf16[256,192]{1,0} parameter(1)
  ROOT fusion = bf16[4,192]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16SmallMRank3DecoderContraction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_small_m_rank3_decoder_contraction

fly_gemm {
  lhs = bf16[4,3,128]{2,1,0} parameter(0)
  rhs = bf16[3,128,192]{2,1,0} parameter(1)
  ROOT dot = f32[3,4,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[4,3,128]{2,1,0} parameter(0)
  rhs = bf16[3,128,192]{2,1,0} parameter(1)
  ROOT fusion = f32[3,4,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16ShortKMiddleBatchDimension) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_short_k_middle_batch

fly_gemm {
  lhs = bf16[256,4,128]{2,1,0} parameter(0)
  rhs = bf16[4,128,384]{2,1,0} parameter(1)
  ROOT dot = f32[4,256,384]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[256,4,128]{2,1,0} parameter(0)
  rhs = bf16[4,128,384]{2,1,0} parameter(1)
  ROOT fusion = f32[4,256,384]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"64", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","64","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.03}));
}

TEST_F(FlyGemmDeviceTest, Bf16SmallMVectorizedSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_small_m_vectorized_split_k

fly_gemv {
  lhs = bf16[4,512]{1,0} parameter(0)
  rhs = bf16[512,192]{1,0} parameter(1)
  ROOT dot = bf16[4,192]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[4,512]{1,0} parameter(0)
  rhs = bf16[512,192]{1,0} parameter(1)
  ROOT fusion = bf16[4,192]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"4", "block_n":"64",
          "block_k":"32", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "gemv_outputs_per_wave":"4", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["4","64"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16SmallMRoundedContractingScaleSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_small_m_rounded_contracting_scale_split_k

fly_gemv {
  data = bf16[4,512]{1,0} parameter(0)
  scale = bf16[512]{0} parameter(1)
  matrix = bf16[512,192]{1,0} parameter(2)
  data_f32 = f32[4,512]{1,0} convert(data)
  scale_broadcast = bf16[4,512]{1,0} broadcast(scale), dimensions={1}
  scale_f32 = f32[4,512]{1,0} convert(scale_broadcast)
  product = f32[4,512]{1,0} multiply(data_f32, scale_f32)
  rounded = bf16[4,512]{1,0} convert(product)
  ROOT dot = bf16[4,192]{1,0} dot(rounded, matrix),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  data = bf16[4,512]{1,0} parameter(0)
  scale = bf16[512]{0} parameter(1)
  matrix = bf16[512,192]{1,0} parameter(2)
  ROOT fusion = bf16[4,192]{1,0} fusion(data, scale, matrix),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"4", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "gemv_outputs_per_wave":"4", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["4","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16SmallMBatchedVectorizedSplitK) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_small_m_batched_vectorized_split_k

fly_gemv {
  lhs = bf16[4,3,512]{2,1,0} parameter(0)
  rhs = bf16[3,512,192]{2,1,0} parameter(1)
  ROOT dot = f32[3,4,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  lhs = bf16[4,3,512]{2,1,0} parameter(0)
  rhs = bf16[3,512,192]{2,1,0} parameter(1)
  ROOT fusion = f32[3,4,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"4", "block_n":"64",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16",
          "gemv_outputs_per_wave":"4", "gemv_split_k":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","4","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16SmallMStagedMfmaTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_small_m_staged_mfma_tail

fly_gemm {
  lhs = bf16[4,256]{1,0} parameter(0)
  rhs = bf16[256,192]{1,0} parameter(1)
  ROOT dot = bf16[4,192]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["64"]}
}

ENTRY main {
  lhs = bf16[4,256]{1,0} parameter(0)
  rhs = bf16[256,192]{1,0} parameter(1)
  ROOT fusion = bf16[4,192]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"64", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16SmallMRoundedScaleStagedMfmaTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_small_m_rounded_scale_staged_mfma_tail

fly_gemm {
  data = bf16[4,4096]{1,0} parameter(0)
  scale = bf16[4096]{0} parameter(1)
  matrix = bf16[4096,192]{1,0} parameter(2)
  data_f32 = f32[4,4096]{1,0} convert(data)
  scale_broadcast = bf16[4,4096]{1,0} broadcast(scale), dimensions={1}
  scale_f32 = f32[4,4096]{1,0} convert(scale_broadcast)
  product = f32[4,4096]{1,0} multiply(data_f32, scale_f32)
  rounded = bf16[4,4096]{1,0} convert(product)
  ROOT dot = bf16[4,192]{1,0} dot(rounded, matrix),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  data = bf16[4,4096]{1,0} parameter(0)
  scale = bf16[4096]{0} parameter(1)
  matrix = bf16[4096,192]{1,0} parameter(2)
  ROOT fusion = bf16[4,192]{1,0} fusion(data, scale, matrix),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"64",
          "block_k":"128", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16", "stage_rhs":true,
          "schedule_instructions":true, "preload_lds_fragments":true},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","64"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16PipelinedMfma4DecoderProjection) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_pipelined_mfma4_decoder_projection

fly_gemv {
  lhs = bf16[4,512]{1,0} parameter(0)
  rhs = bf16[512,256]{1,0} parameter(1)
  ROOT dot = bf16[4,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[4,512]{1,0} parameter(0)
  rhs = bf16[512,256]{1,0} parameter(1)
  ROOT fusion = bf16[4,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"4", "block_n":"128",
          "block_k":"128", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_4X4X4_BF16",
          "gemv_outputs_per_wave":"1", "gemv_k_vector_width":"1"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["4","128"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16PipelinedMfma4RoundedContractingScale) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_pipelined_mfma4_rounded_contracting_scale

fly_gemv {
  data = bf16[4,512]{1,0} parameter(0)
  scale = bf16[512]{0} parameter(1)
  matrix = bf16[512,256]{1,0} parameter(2)
  data_f32 = f32[4,512]{1,0} convert(data)
  scale_broadcast = bf16[4,512]{1,0} broadcast(scale), dimensions={1}
  scale_f32 = f32[4,512]{1,0} convert(scale_broadcast)
  product = f32[4,512]{1,0} multiply(data_f32, scale_f32)
  rounded = bf16[4,512]{1,0} convert(product)
  ROOT dot = bf16[4,256]{1,0} dot(rounded, matrix),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  data = bf16[4,512]{1,0} parameter(0)
  scale = bf16[512]{0} parameter(1)
  matrix = bf16[512,256]{1,0} parameter(2)
  ROOT fusion = bf16[4,256]{1,0} fusion(data, scale, matrix),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"4", "block_n":"128",
          "block_k":"128", "num_warps":"8",
          "mfma_atom":"FLY_MFMA_4X4X4_BF16",
          "gemv_outputs_per_wave":"1", "gemv_k_vector_width":"1"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["4","128"]}],
          "num_stages":"1", "num_warps":"8", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

TEST_F(FlyGemmDeviceTest, Bf16BatchedPipelinedMfma4KTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_batched_pipelined_mfma4_k_tail

fly_gemv {
  lhs = bf16[4,3,224]{2,1,0} parameter(0)
  rhs = bf16[3,224,128]{2,1,0} parameter(1)
  ROOT dot = f32[3,4,128]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  lhs = bf16[4,3,224]{2,1,0} parameter(0)
  rhs = bf16[3,224,128]{2,1,0} parameter(1)
  ROOT fusion = f32[3,4,128]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=fly_gemv,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemv",
        "fly_gemm_config":{"block_m":"4", "block_n":"64",
          "block_k":"128", "num_warps":"2",
          "mfma_atom":"FLY_MFMA_4X4X4_BF16",
          "gemv_outputs_per_wave":"1", "gemv_k_vector_width":"1"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1","4","64"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

}  // namespace
}  // namespace xla::gpu
