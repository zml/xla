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

#include "xla/backends/gpu/transforms/flydsl_replacement_verifier.h"

#include <array>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class FlyDslReplacementVerifierTest
    : public HloHardwareIndependentTestBase {};

std::string FusionModule(absl::string_view kind) {
  return absl::StrCat(R"(
HloModule replacement_verifier

fused_computation {
  parameter = f32[8] parameter(0)
  ROOT negate = f32[8] negate(parameter)
}

ENTRY main {
  parameter = f32[8] parameter(0)
  ROOT fusion = f32[8] fusion(parameter), kind=kCustom,
      calls=fused_computation,
      backend_config={"fusion_backend_config":{"kind":")",
                      kind, R"("}}
}
)");
}

TEST_F(FlyDslReplacementVerifierTest, DisabledModeAllowsTriton) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(
                              FusionModule("__triton_gemm")));
  EXPECT_THAT(FlyDslReplacementVerifier().Run(module.get()),
              IsOkAndHolds(false));
}

TEST_F(FlyDslReplacementVerifierTest, EnabledModeAllowsEveryFlyKind) {
  constexpr std::array<absl::string_view, 4> kKinds = {
      "__fly", "__fly_gemm", "__fly_gemv", "__fly_collective"};
  for (absl::string_view kind : kKinds) {
    SCOPED_TRACE(kind);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(FusionModule(kind)));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_flydsl_replace_triton(true);
    EXPECT_THAT(FlyDslReplacementVerifier().Run(module.get()),
                IsOkAndHolds(false));
  }
}

TEST_F(FlyDslReplacementVerifierTest, RejectsEveryTritonFusionKind) {
  constexpr std::array<absl::string_view, 4> kKinds = {
      "__triton", "__triton_gemm", "__triton_nested_gemm_fusion",
      "__triton_collective"};
  for (absl::string_view kind : kKinds) {
    SCOPED_TRACE(kind);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(FusionModule(kind)));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_flydsl_replace_triton(true);
    EXPECT_THAT(FlyDslReplacementVerifier().Run(module.get()),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr(kind)));
  }
}

TEST_F(FlyDslReplacementVerifierTest, RejectsTritonCustomCalls) {
  constexpr std::array<absl::string_view, 2> kTargets = {
      "__triton", "__gpu$xla.gpu.triton"};
  for (absl::string_view target : kTargets) {
    SCOPED_TRACE(target);
    const std::string hlo = absl::StrCat(R"(
HloModule replacement_verifier_custom_call

ENTRY main {
  parameter = f32[8] parameter(0)
  ROOT custom_call = f32[8] custom-call(parameter), custom_call_target=")",
                                         target, R"("
}
)");
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(hlo));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_flydsl_replace_triton(true);
    EXPECT_THAT(FlyDslReplacementVerifier().Run(module.get()),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr(target)));
  }
}

TEST_F(FlyDslReplacementVerifierTest,
       RejectsOpaqueCustomCallInsideFlyFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule replacement_verifier_opaque_fly_call

body {
  p0 = f32[8]{0} parameter(0)
  ROOT unsupported_call = f32[8]{0} custom-call(p0),
      custom_call_target="__opaque$unsupported"
}

ENTRY main {
  p0 = f32[8]{0} parameter(0)
  ROOT fusion = f32[8]{0} fusion(p0), kind=kCustom, calls=body,
      backend_config={"fusion_backend_config":{"kind":"__fly"}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_flydsl_replace_triton(true);
  EXPECT_THAT(FlyDslReplacementVerifier().Run(module.get()),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("unsupported custom call")));
}

}  // namespace
}  // namespace xla::gpu
