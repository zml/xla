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
#include "xla/backends/gpu/transforms/gemm_rewriter.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla::gpu {
namespace {

class MusaGemmRewriterTest : public HloHardwareIndependentTestBase {};

stream_executor::GpuComputeCapability MusaS80Capability() {
  return stream_executor::GpuComputeCapability(
      stream_executor::MusaComputeCapability("mp_21", 2, 1,
                                             /*hardware_warp_size=*/128,
                                             /*logical_subgroup_size=*/32));
}

TEST_F(MusaGemmRewriterTest, WidensLowPrecisionGemmToF32) {
  constexpr absl::string_view hlo = R"(
HloModule MusaGemms

ENTRY main {
  lhs_f16 = f16[8,16]{1,0} parameter(0)
  rhs_f16 = f16[16,4]{1,0} parameter(1)
  gemm_f16 = f16[8,4]{1,0} dot(lhs_f16, rhs_f16),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  lhs_f32 = f32[7,9]{1,0} parameter(2)
  rhs_f32 = f32[9,5]{1,0} parameter(3)
  gemm_f32 = f32[7,5]{1,0} dot(lhs_f32, rhs_f32),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  lhs_bf16 = bf16[8,16]{1,0} parameter(4)
  rhs_bf16 = bf16[16,4]{1,0} parameter(5)
  gemm_bf16 = bf16[8,4]{1,0} dot(lhs_bf16, rhs_bf16),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  lhs_f64 = f64[5,7]{1,0} parameter(6)
  rhs_f64 = f64[7,2]{1,0} parameter(7)
  gemm_f64 = f64[5,2]{1,0} dot(lhs_f64, rhs_f64),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT result = (f16[8,4]{1,0}, f32[7,5]{1,0}, bf16[8,4]{1,0}, f64[5,2]{1,0}) tuple(gemm_f16, gemm_f32, gemm_bf16, gemm_f64)
}
)";

  RunAndFilecheckHloRewrite(
      hlo,
      GemmRewriter(MusaS80Capability(),
                   stream_executor::SemanticVersion{4, 0, 1},
                   GemmRewriterOptions{GemmRewriterOptions::DType::kNonFp8Only,
                                       GemmRewriterOptions::BiasMode::kNoBias}),
      R"(
// CHECK-DAG: [[F16_LHS:%[^ ]+]] = f32[8,16]{1,0} convert({{.*}})
// CHECK-DAG: [[F16_RHS:%[^ ]+]] = f32[16,4]{1,0} convert({{.*}})
// CHECK-DAG: [[F16_GEMM:%[^ ]+]] = f32[8,4]{1,0} custom-call([[F16_LHS]], [[F16_RHS]]), custom_call_target="__mublas$gemm"
// CHECK-DAG: f16[8,4]{1,0} convert([[F16_GEMM]])
// CHECK-DAG: f32[7,5]{1,0} custom-call({{.*}}), custom_call_target="__mublas$gemm"
// CHECK-DAG: [[BF16_LHS:%[^ ]+]] = f32[8,16]{1,0} convert({{.*}})
// CHECK-DAG: [[BF16_RHS:%[^ ]+]] = f32[16,4]{1,0} convert({{.*}})
// CHECK-DAG: [[BF16_GEMM:%[^ ]+]] = f32[8,4]{1,0} custom-call([[BF16_LHS]], [[BF16_RHS]]), custom_call_target="__mublas$gemm"
// CHECK-DAG: bf16[8,4]{1,0} convert([[BF16_GEMM]])
// CHECK-DAG: f64[5,2]{1,0} custom-call({{.*}}), custom_call_target="__mublas$gemm"
// CHECK-NOT: __cublas
)");
}

TEST_F(MusaGemmRewriterTest, RewritesBatchAndKeepsF16ToF32Output) {
  constexpr absl::string_view hlo = R"(
HloModule UnsupportedMusaGemms

ENTRY main {
  batch_lhs = f32[2,8,16]{2,1,0} parameter(0)
  batch_rhs = f32[2,16,4]{2,1,0} parameter(1)
  batch = f32[2,8,4]{2,1,0} dot(batch_lhs, batch_rhs),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  mixed_lhs = f16[8,16]{1,0} parameter(2)
  mixed_rhs = f16[16,4]{1,0} parameter(3)
  mixed = f32[8,4]{1,0} dot(mixed_lhs, mixed_rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT result = (f32[2,8,4]{2,1,0}, f32[8,4]{1,0}) tuple(batch, mixed)
}
)";

  RunAndFilecheckHloRewrite(
      hlo,
      GemmRewriter(MusaS80Capability(),
                   stream_executor::SemanticVersion{4, 0, 1},
                   GemmRewriterOptions{GemmRewriterOptions::DType::kNonFp8Only,
                                       GemmRewriterOptions::BiasMode::kNoBias}),
      R"(
// CHECK-DAG: f32[2,8,4]{2,1,0} custom-call({{.*}}), custom_call_target="__mublas$gemm"
// CHECK-DAG: [[MIXED_LHS:%[^ ]+]] = f32[8,16]{1,0} convert({{.*}})
// CHECK-DAG: [[MIXED_RHS:%[^ ]+]] = f32[16,4]{1,0} convert({{.*}})
// CHECK-DAG: [[MIXED_GEMM:%[^ ]+]] = f32[8,4]{1,0} custom-call([[MIXED_LHS]], [[MIXED_RHS]]), custom_call_target="__mublas$gemm"
// CHECK-NOT: __cublas
// CHECK-NOT: convert([[MIXED_GEMM]])
)");
}

}  // namespace
}  // namespace xla::gpu
