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

#include "xla/backends/gpu/autotuner/fly.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOk;
using ::testing::IsEmpty;

constexpr char kScaledDotHlo[] = R"(
HloModule fly_scaled_dot

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = bf16[1024,1024]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kUnsupportedScaleHlo[] = R"(
HloModule fly_scaled_dot_nonuniform_scale

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs_scale = f32[1024,8]{1,0} parameter(2)
  rhs_scale = f32[8,1024]{1,0} parameter(3)
  ROOT dot = bf16[1024,1024]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs_scale = f32[1024,8]{1,0} parameter(2)
  rhs_scale = f32[8,1024]{1,0} parameter(3)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kNarrowingEpilogueHlo[] = R"(
HloModule fly_narrowing_epilogue

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kNarrowingShortKEpilogueHlo[] = R"(
HloModule fly_narrowing_short_k_epilogue

gemm {
  lhs = bf16[512,512]{1,0} parameter(0)
  rhs = bf16[512,512]{0,1} parameter(1)
  dot = f32[512,512]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[512,512]{1,0} convert(dot)
}

ENTRY main {
  lhs = bf16[512,512]{1,0} parameter(0)
  rhs = bf16[512,512]{0,1} parameter(1)
  ROOT fusion = bf16[512,512]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kScalarEpilogueHlo[] = R"(
HloModule fly_scalar_epilogue

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  alpha = f32[] parameter(2)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  broadcast = f32[1024,1024]{1,0} broadcast(alpha), dimensions={}
  multiply = f32[1024,1024]{1,0} multiply(dot, broadcast)
  ROOT convert = bf16[1024,1024]{1,0} convert(multiply)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  alpha = f32[] parameter(2)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs, alpha), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kVectorEpilogueHlo[] = R"(
HloModule fly_vector_epilogue

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  bias = f32[1024]{0} parameter(2)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  broadcast = f32[1024,1024]{1,0} broadcast(bias), dimensions={1}
  add = f32[1024,1024]{1,0} add(dot, broadcast)
  ROOT convert = bf16[1024,1024]{1,0} convert(add)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  bias = f32[1024]{0} parameter(2)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs, bias), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kEpilogueChainHlo[] = R"(
HloModule fly_epilogue_chain

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  scale = f32[1024]{0} parameter(2)
  bias = f32[] parameter(3)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  scale_broadcast = f32[1024,1024]{1,0} broadcast(scale), dimensions={1}
  multiply = f32[1024,1024]{1,0} multiply(dot, scale_broadcast)
  bias_broadcast = f32[1024,1024]{1,0} broadcast(bias), dimensions={}
  add = f32[1024,1024]{1,0} add(multiply, bias_broadcast)
  negate = f32[1024,1024]{1,0} negate(add)
  ROOT convert = bf16[1024,1024]{1,0} convert(negate)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  scale = f32[1024]{0} parameter(2)
  bias = f32[] parameter(3)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs, scale, bias),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kBiasReluEpilogueHlo[] = R"(
HloModule fly_bias_relu_epilogue

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{1,0} parameter(1)
  bias = f32[1024]{0} parameter(2)
  zero = f32[] constant(0)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias_broadcast = f32[1024,1024]{1,0} broadcast(bias), dimensions={1}
  add = f32[1024,1024]{1,0} add(dot, bias_broadcast)
  zero_broadcast = f32[1024,1024]{1,0} broadcast(zero), dimensions={}
  maximum = f32[1024,1024]{1,0} maximum(add, zero_broadcast)
  ROOT convert = bf16[1024,1024]{1,0} convert(maximum)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{1,0} parameter(1)
  bias = f32[1024]{0} parameter(2)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs, bias),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedInputsHlo[] = R"(
HloModule fly_converted_inputs

gemm {
  lhs_f32 = f32[1024,1024]{1,0} parameter(0)
  rhs_f32 = f32[1024,1024]{0,1} parameter(1)
  lhs = bf16[1024,1024]{1,0} convert(lhs_f32)
  rhs = bf16[1024,1024]{0,1} convert(rhs_f32)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[1024,1024]{1,0} parameter(0)
  rhs = f32[1024,1024]{0,1} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedGemvInputsHlo[] = R"(
HloModule fly_converted_gemv_inputs

gemv {
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
  ROOT fusion = bf16[256,1]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedBitcastInputsHlo[] = R"(
HloModule fly_converted_bitcast_inputs

gemm {
  lhs_f32_physical = f32[512,2048]{1,0} parameter(0)
  rhs_f32_physical = f32[512,2048]{1,0} parameter(1)
  lhs_f32 = f32[1024,1024]{1,0} bitcast(lhs_f32_physical)
  lhs = bf16[1024,1024]{1,0} convert(lhs_f32)
  rhs_bf16_physical = bf16[512,2048]{1,0} convert(rhs_f32_physical)
  rhs = bf16[1024,1024]{0,1} bitcast(rhs_bf16_physical)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[512,2048]{1,0} parameter(0)
  rhs = f32[512,2048]{1,0} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedSliceInputsHlo[] = R"(
HloModule fly_converted_slice_inputs

gemm {
  lhs_f32_physical = f32[1056,1088]{1,0} parameter(0)
  rhs_f32_physical = f32[1088,1056]{0,1} parameter(1)
  lhs_f32 = f32[1024,1024]{1,0} slice(lhs_f32_physical),
      slice={[16:1040], [32:1056]}
  rhs_f32 = f32[1024,1024]{0,1} slice(rhs_f32_physical),
      slice={[32:1056], [16:1040]}
  lhs = bf16[1024,1024]{1,0} convert(lhs_f32)
  rhs = bf16[1024,1024]{0,1} convert(rhs_f32)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[1056,1088]{1,0} parameter(0)
  rhs = f32[1088,1056]{0,1} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kStridedSliceInputHlo[] = R"(
HloModule fly_strided_slice_input

gemm {
  lhs_physical = bf16[1024,2048]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs = bf16[1024,1024]{1,0} slice(lhs_physical),
      slice={[0:1024], [0:2048:2]}
  ROOT dot = bf16[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,2048]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kDynamicSliceInputsHlo[] = R"(
HloModule fly_dynamic_slice_inputs

gemm {
  lhs_physical = bf16[1056,1088]{1,0} parameter(0)
  rhs_physical = bf16[1088,1056]{0,1} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  lhs = bf16[1024,1024]{1,0} dynamic-slice(
      lhs_physical, lhs_start_m, lhs_start_k),
      dynamic_slice_sizes={1024,1024}
  rhs = bf16[1024,1024]{0,1} dynamic-slice(
      rhs_physical, rhs_start_k, rhs_start_n),
      dynamic_slice_sizes={1024,1024}
  ROOT dot = bf16[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1056,1088]{1,0} parameter(0)
  rhs = bf16[1088,1056]{0,1} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(
      lhs, rhs, lhs_start_m, lhs_start_k, rhs_start_k, rhs_start_n),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedConcatInputsHlo[] = R"(
HloModule fly_converted_concat_inputs

gemm {
  lhs0_f32 = f32[512,1024]{1,0} parameter(0)
  lhs1_f32 = f32[512,1024]{1,0} parameter(1)
  rhs0_f32 = f32[1024,512]{0,1} parameter(2)
  rhs1_f32 = f32[1024,512]{0,1} parameter(3)
  lhs0 = bf16[512,1024]{1,0} convert(lhs0_f32)
  lhs1 = bf16[512,1024]{1,0} convert(lhs1_f32)
  rhs0 = bf16[1024,512]{0,1} convert(rhs0_f32)
  rhs1 = bf16[1024,512]{0,1} convert(rhs1_f32)
  lhs = bf16[1024,1024]{1,0} concatenate(lhs0, lhs1), dimensions={0}
  rhs = bf16[1024,1024]{0,1} concatenate(rhs0, rhs1), dimensions={1}
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs0 = f32[512,1024]{1,0} parameter(0)
  lhs1 = f32[512,1024]{1,0} parameter(1)
  rhs0 = f32[1024,512]{0,1} parameter(2)
  rhs1 = f32[1024,512]{0,1} parameter(3)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs0, lhs1, rhs0, rhs1),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kContractingConcatInputHlo[] = R"(
HloModule fly_contracting_concat_input

gemm {
  lhs0 = bf16[1024,512]{1,0} parameter(0)
  lhs1 = bf16[1024,512]{1,0} parameter(1)
  rhs = bf16[1024,1024]{0,1} parameter(2)
  lhs = bf16[1024,1024]{1,0} concatenate(lhs0, lhs1), dimensions={1}
  ROOT dot = bf16[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs0 = bf16[1024,512]{1,0} parameter(0)
  lhs1 = bf16[1024,512]{1,0} parameter(1)
  rhs = bf16[1024,1024]{0,1} parameter(2)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs0, lhs1, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kBatchedGemmHlo[] = R"(
HloModule fly_batched_gemm

gemm {
  lhs = bf16[4,256,128]{2,1,0} parameter(0)
  rhs = bf16[4,192,128]{2,1,0} parameter(1)
  ROOT dot = bf16[4,256,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  lhs = bf16[4,256,128]{2,1,0} parameter(0)
  rhs = bf16[4,192,128]{2,1,0} parameter(1)
  ROOT fusion = bf16[4,256,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kBatchedGemvHlo[] = R"(
HloModule fly_batched_gemv

gemv {
  lhs = bf16[4,1,256]{2,1,0} parameter(0)
  rhs = bf16[4,256,192]{2,1,0} parameter(1)
  ROOT dot = bf16[4,1,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[4,1,256]{2,1,0} parameter(0)
  rhs = bf16[4,256,192]{2,1,0} parameter(1)
  ROOT fusion = bf16[4,1,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

class FlyBackendTest : public HloHardwareIndependentTestBase {
 protected:
  FlyBackendTest()
      : platform_(PlatformUtil::GetDefaultPlatform().value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        target_config_(stream_executor_),
        compiler_(Compiler::GetForPlatform(platform_->id()).value()),
        backend_(&debug_options_, compiler_.get(), &target_config_) {
    debug_options_.set_xla_gpu_enable_flydsl_gemm(true);
  }

  DebugOptions debug_options_;
  se::Platform* platform_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  std::unique_ptr<Compiler> compiler_;
  FlyBackend backend_;
};

TEST_F(FlyBackendTest, SupportsUniformBf16ScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kScaledDotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().local_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, RejectsNonuniformScale) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kUnsupportedScaleHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsEmpty());
}

TEST_F(FlyBackendTest, SupportsNarrowingBf16Epilogue) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kNarrowingEpilogueHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().stage_output(); }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().local_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, TunesShortKWorkgroupMapping) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kNarrowingShortKEpilogueHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().workgroup_mapping_n() == 4;
      }));
}

TEST_F(FlyBackendTest, SupportsScalarEpilogue) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kScalarEpilogueHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsVectorEpilogue) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kVectorEpilogueHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsThreeStepEpilogueChain) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kEpilogueChainHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().local_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsBiasReluEpilogue) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kBiasReluEpilogueHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 128 && fly.block_n() == 64 &&
               fly.block_k() == 128 && fly.num_warps() == 8 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               fly.single_buffer_lds() && fly.rolling_refill();
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsF32ToBf16ContractionInputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kConvertedInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().local_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsF32ToBf16GemvInputs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kConvertedGemvInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, SupportsBitcastsAroundF32ToBf16Inputs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kConvertedBitcastInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsStaticUnitStrideInputSlices) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kConvertedSliceInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, RejectsStridedInputSlices) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kStridedSliceInputHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsEmpty());
}

TEST_F(FlyBackendTest, SupportsDynamicSliceInputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kDynamicSliceInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsTileAlignedConvertedConcatInputs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kConvertedConcatInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::all_of(configs.begin(), configs.end(), [](const auto& config) {
        return 512 % config->fly().block_m() == 0 &&
               512 % config->fly().block_n() == 0;
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, RejectsContractingDimensionConcatInput) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kContractingConcatInputHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsEmpty());
}

TEST_F(FlyBackendTest, SupportsBatchedBf16Gemm) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kBatchedGemmHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
  const BlockLevelFusionConfig& block =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  ASSERT_EQ(block.output_tiles_size(), 1);
  ASSERT_EQ(block.output_tiles(0).sizes_size(), 3);
  EXPECT_EQ(block.output_tiles(0).sizes(0), 1);
}

TEST_F(FlyBackendTest, SupportsBatchedBf16Gemv) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kBatchedGemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
  const BlockLevelFusionConfig& block =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  ASSERT_EQ(block.output_tiles_size(), 1);
  ASSERT_EQ(block.output_tiles(0).sizes_size(), 3);
  EXPECT_EQ(block.output_tiles(0).sizes(0), 1);
}

}  // namespace
}  // namespace xla::gpu
