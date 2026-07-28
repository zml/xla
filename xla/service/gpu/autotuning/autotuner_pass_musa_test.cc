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

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/autotuning/autotuner_pass.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

constexpr char kMublasCustomCallHlo[] = R"hlo(
HloModule mublas_filter

ENTRY main {
  lhs = f32[8,16]{1,0} parameter(0)
  rhs = f32[16,4]{1,0} parameter(1)
  ROOT gemm = f32[8,4]{1,0} custom-call(lhs, rhs),
    custom_call_target="__mublas$gemm",
    backend_config={"gemm_backend_config":{
      "alpha_real":1,
      "beta":0,
      "dot_dimension_numbers":{
        "lhs_contracting_dimensions":["1"],
        "rhs_contracting_dimensions":["0"]
      }
    }}
}
)hlo";

constexpr char kWrappedConvertFusionHlo[] = R"hlo(
HloModule wrapped_convert_filter

wrapped_convert_computation {
  parameter = f32[8,4]{1,0} parameter(0)
  ROOT convert = f16[8,4]{1,0} convert(parameter)
}

ENTRY main {
  parameter = f32[8,4]{1,0} parameter(0)
  ROOT wrapped_convert = f16[8,4]{1,0} fusion(parameter), kind=kLoop,
    calls=wrapped_convert_computation
}
)hlo";

class MusaAutotunerFilterTest : public HloHardwareIndependentTestBase {
 protected:
  stream_executor::GpuComputeCapability capability_{
      stream_executor::MusaComputeCapability("mp_21", 2, 1,
                                             /*hardware_warp_size=*/128,
                                             /*logical_subgroup_size=*/32)};
};

TEST_F(MusaAutotunerFilterTest, RecognizesUnconfiguredMublasGemm) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMublasCustomCallHlo));
  DebugOptions debug_options;
  debug_options.set_xla_gpu_autotune_level(4);
  InstructionFilterFn filter =
      GetShouldAutotuneInstructionFn(debug_options, capability_);
  EXPECT_TRUE(filter(*module->entry_computation()->root_instruction()));
}

TEST_F(MusaAutotunerFilterTest, SkipsConfiguredMublasGemm) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMublasCustomCallHlo));
  HloInstruction* gemm = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                       gemm->backend_config<GpuBackendConfig>());
  config.mutable_gemm_backend_config()->set_selected_algorithm(0);
  ASSERT_OK(gemm->set_backend_config(config));

  DebugOptions debug_options;
  debug_options.set_xla_gpu_autotune_level(4);
  InstructionFilterFn filter =
      GetShouldAutotuneInstructionFn(debug_options, capability_);
  EXPECT_FALSE(filter(*gemm));
}

TEST_F(MusaAutotunerFilterTest, BinaryLibraryDisableFlagSkipsMublas) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMublasCustomCallHlo));
  DebugOptions debug_options;
  debug_options.set_xla_gpu_autotune_level(4);
  debug_options.set_xla_gpu_experimental_disable_binary_libraries(true);
  InstructionFilterFn filter =
      GetShouldAutotuneInstructionFn(debug_options, capability_);
  EXPECT_FALSE(filter(*module->entry_computation()->root_instruction()));
}

TEST_F(MusaAutotunerFilterTest, SkipsGenericFusionWithoutMusaBackend) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kWrappedConvertFusionHlo));
  DebugOptions debug_options;
  debug_options.set_xla_gpu_autotune_level(4);
  debug_options.set_xla_gpu_experimental_enable_fusion_autotuner(true);
  InstructionFilterFn filter =
      GetShouldAutotuneInstructionFn(debug_options, capability_);
  EXPECT_FALSE(filter(*module->entry_computation()->root_instruction()));
}

}  // namespace
}  // namespace xla::gpu
