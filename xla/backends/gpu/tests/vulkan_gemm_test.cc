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
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla::gpu {
namespace {

using VulkanGemmTest = HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(VulkanGemmTest, KxNWithDimensionTails) {
  constexpr absl::string_view kHlo = R"(
    HloModule vulkan_bf16_gemm_kxn

    ENTRY main {
      lhs = bf16[17,21]{1,0} parameter(0)
      rhs = bf16[21,19]{1,0} parameter(1)
      ROOT result = bf16[17,19]{1,0} dot(lhs, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.03, /*arel=*/0.03}));
}

TEST_F(VulkanGemmTest, NxKWithDimensionTails) {
  constexpr absl::string_view kHlo = R"(
    HloModule vulkan_bf16_gemm_nxk

    ENTRY main {
      lhs = bf16[17,21]{1,0} parameter(0)
      rhs = bf16[19,21]{1,0} parameter(1)
      ROOT result = bf16[17,19]{1,0} dot(lhs, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })";

  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.03, /*arel=*/0.03}));
}

}  // namespace
}  // namespace xla::gpu
