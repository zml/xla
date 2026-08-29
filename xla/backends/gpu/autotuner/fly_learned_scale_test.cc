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

#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "gtest/gtest.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/fly.h"
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

class FlyLearnedScaleBackendTest : public HloHardwareIndependentTestBase {
 protected:
  FlyLearnedScaleBackendTest()
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

TEST_F(FlyLearnedScaleBackendTest, OffersForwardProjectionPipeline) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_learned_scale_forward_policy

gemm {
  data = bf16[128,256]{1,0} parameter(0)
  data_f32 = f32[128,256]{1,0} convert(data)
  scale = bf16[256]{0} parameter(1)
  scale_broadcast = bf16[128,256]{1,0} broadcast(scale), dimensions={1}
  scale_f32 = f32[128,256]{1,0} convert(scale_broadcast)
  scaled_f32 = f32[128,256]{1,0} multiply(data_f32, scale_f32)
  scaled = bf16[128,256]{1,0} convert(scaled_f32)
  weight = bf16[256,768]{1,0} parameter(2)
  ROOT dot = bf16[128,768]{1,0} dot(scaled, weight),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  data = bf16[128,256]{1,0} parameter(0)
  scale = bf16[256]{0} parameter(1)
  weight = bf16[256,768]{1,0} parameter(2)
  ROOT fusion = bf16[128,768]{1,0} fusion(data, scale, weight),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__fly_gemm"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));
  ASSERT_EQ(configs.size(), 7);
  const FlyGemmConfig& config = configs.front()->fly();
  EXPECT_EQ(config.block_m(), 32);
  EXPECT_EQ(config.block_n(), 64);
  EXPECT_EQ(config.block_k(), 128);
  EXPECT_EQ(config.mfma_atom(), FlyGemmConfig::FLY_MFMA_32X32X8);
  EXPECT_TRUE(config.stage_rhs());
  EXPECT_TRUE(config.preload_lds_fragments());
  EXPECT_TRUE(config.single_buffer_lds());
  EXPECT_TRUE(config.rolling_refill());

  int compact_configs = 0;
  bool has_scheduled_compact_config = false;
  for (const std::unique_ptr<BackendConfig>& candidate : configs) {
    const FlyGemmConfig& fly = candidate->fly();
    if (fly.block_m() == 32 && fly.block_n() == 32 &&
        fly.block_k() == 64 && fly.num_warps() == 4 && fly.stage_rhs() &&
        !fly.preload_lds_fragments() && !fly.single_buffer_lds()) {
      ++compact_configs;
      has_scheduled_compact_config |= fly.schedule_instructions();
    }
  }
  EXPECT_EQ(compact_configs, 6);
  EXPECT_TRUE(has_scheduled_compact_config);
}

TEST_F(FlyLearnedScaleBackendTest, OffersWeightGradientPipeline) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_learned_scale_weight_gradient_policy

gemm {
  data = bf16[128,256]{1,0} parameter(0)
  data_f32 = f32[128,256]{1,0} convert(data)
  scale = bf16[256]{0} parameter(1)
  scale_broadcast = bf16[128,256]{1,0} broadcast(scale), dimensions={1}
  scale_f32 = f32[128,256]{1,0} convert(scale_broadcast)
  scaled_f32 = f32[128,256]{1,0} multiply(data_f32, scale_f32)
  scaled = bf16[128,256]{1,0} convert(scaled_f32)
  gradient = bf16[128,768]{1,0} parameter(2)
  ROOT dot = bf16[256,768]{1,0} dot(scaled, gradient),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY main {
  data = bf16[128,256]{1,0} parameter(0)
  scale = bf16[256]{0} parameter(1)
  gradient = bf16[128,768]{1,0} parameter(2)
  ROOT fusion = bf16[256,768]{1,0} fusion(data, scale, gradient),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__fly_gemm"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));
  ASSERT_EQ(configs.size(), 1);
  const FlyGemmConfig& config = configs.front()->fly();
  EXPECT_EQ(config.block_m(), 16);
  EXPECT_EQ(config.block_n(), 128);
  EXPECT_EQ(config.block_k(), 32);
  EXPECT_EQ(config.mfma_atom(), FlyGemmConfig::FLY_MFMA_16X16X16);
  EXPECT_FALSE(config.stage_rhs());
}

}  // namespace
}  // namespace xla::gpu
