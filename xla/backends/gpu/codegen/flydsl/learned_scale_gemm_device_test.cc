/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

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

class FlyLearnedScaleGemmDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {};

TEST_F(FlyLearnedScaleGemmDeviceTest, ForwardProjectionContractingScale) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_learned_scale_forward_gemm

fly_gemm {
  data = bf16[128,256]{1,0} parameter(0)
  data_f32 = f32[128,256]{1,0} convert(data)
  scale = bf16[256]{0} parameter(1)
  scale_broadcast = bf16[128,256]{1,0} broadcast(scale), dimensions={1}
  scale_f32 = f32[128,256]{1,0} convert(scale_broadcast)
  scaled_f32 = f32[128,256]{1,0} multiply(data_f32, scale_f32)
  scaled = bf16[128,256]{1,0} convert(scaled_f32)
  weight = bf16[256,768]{1,0} parameter(2)
  ROOT dot = bf16[128,768]{1,0} dot(scaled, weight),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      backend_config={"sizes":["128"]}
}

ENTRY main {
  data = bf16[128,256]{1,0} parameter(0)
  scale = bf16[256]{0} parameter(1)
  weight = bf16[256,768]{1,0} parameter(2)
  ROOT fusion = bf16[128,768]{1,0} fusion(data, scale, weight),
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

TEST_F(FlyLearnedScaleGemmDeviceTest, WeightGradientNoncontractingScale) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_learned_scale_weight_gradient_gemm

fly_gemm {
  data = bf16[128,256]{1,0} parameter(0)
  data_f32 = f32[128,256]{1,0} convert(data)
  scale = bf16[256]{0} parameter(1)
  scale_broadcast = bf16[128,256]{1,0} broadcast(scale), dimensions={1}
  scale_f32 = f32[128,256]{1,0} convert(scale_broadcast)
  scaled_f32 = f32[128,256]{1,0} multiply(data_f32, scale_f32)
  scaled = bf16[128,256]{1,0} convert(scaled_f32)
  gradient = bf16[128,256]{1,0} parameter(2)
  ROOT dot = bf16[256,256]{1,0} dot(scaled, gradient),
      lhs_contracting_dims={0}, rhs_contracting_dims={0},
      backend_config={"sizes":["32"]}
}

ENTRY main {
  data = bf16[128,256]{1,0} parameter(0)
  scale = bf16[256]{0} parameter(1)
  gradient = bf16[128,256]{1,0} parameter(2)
  ROOT fusion = bf16[256,256]{1,0} fusion(data, scale, gradient),
      kind=kCustom, calls=fly_gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__fly_gemm",
        "fly_gemm_config":{"block_m":"16", "block_n":"128",
          "block_k":"32", "num_warps":"4",
          "mfma_atom":"FLY_MFMA_16X16X16"},
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["16","128"]}],
          "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.04}));
}

}  // namespace
}  // namespace xla::gpu
