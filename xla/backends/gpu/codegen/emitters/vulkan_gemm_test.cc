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

#include "xla/backends/gpu/codegen/emitters/vulkan_gemm.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

se::DeviceDescription VulkanDeviceInfo() {
  se::DeviceDescription device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  device.set_vulkan_compute_capability(
      /*api_version_major=*/1, /*api_version_minor=*/4,
      /*shader_bfloat16=*/true, /*storage_buffer_16bit_access=*/true,
      /*subgroup_size=*/32, /*subgroup_basic=*/true,
      /*subgroup_shuffle=*/true);
  return device;
}

class VulkanGemmEmitterTest : public HloHardwareIndependentTestBase {
 protected:
  void ExpectLowersToVulkanSpirv(absl::string_view hlo) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    se::DeviceDescription device = VulkanDeviceInfo();
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device);
    std::optional<VulkanGemmConfig> config = MatchVulkanGemm(analysis);
    ASSERT_TRUE(config.has_value());

    std::unique_ptr<mlir::MLIRContext> context = CreateMlirContext();
    VulkanGemmEmitter emitter(*config);
    ASSERT_OK_AND_ASSIGN(
        mlir::OwningOpRef<mlir::ModuleOp> emitted_module,
        emitter.CreateMLIRModule(*context, *fusion, "vulkan_gemm",
                                 /*buffer_assignment=*/nullptr));
    ASSERT_OK_AND_ASSIGN(
        LlvmKernelSource llvm_source,
        CompileMlirToLlvm(device, *module, "vulkan_gemm",
                          /*unroll_factor=*/0, *context,
                          MlirKernelSource(std::move(emitted_module))));
    ASSERT_OK_AND_ASSIGN(
        std::string spirv,
        spirv::CompileToVulkanSPIRV(llvm_source.module(),
                                    device.gpu_compute_capability(),
                                    module->config().debug_options()));

    ASSERT_GE(spirv.size(), sizeof(uint32_t));
    uint32_t magic = 0;
    std::memcpy(&magic, spirv.data(), sizeof(magic));
    EXPECT_EQ(magic, 0x07230203);
  }
};

TEST_F(VulkanGemmEmitterTest, MatchesNormalRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule gemm

gemm {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[21,19]{1,0} parameter(1)
  ROOT dot = bf16[17,19]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[21,19]{1,0} parameter(1)
  ROOT fusion = bf16[17,19]{1,0} fusion(lhs, rhs), kind=kLoop, calls=gemm
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  se::DeviceDescription device = VulkanDeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device);
  std::optional<VulkanGemmConfig> config = MatchVulkanGemm(analysis);

  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->m, 17);
  EXPECT_EQ(config->n, 19);
  EXPECT_EQ(config->k, 21);
  EXPECT_EQ(config->rhs_layout, VulkanGemmRhsLayout::kKxN);
}

TEST_F(VulkanGemmEmitterTest, MatchesTransposedRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule gemm

gemm {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[19,21]{1,0} parameter(1)
  ROOT dot = bf16[17,19]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[19,21]{1,0} parameter(1)
  ROOT fusion = bf16[17,19]{1,0} fusion(lhs, rhs), kind=kLoop, calls=gemm
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  se::DeviceDescription device = VulkanDeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device);
  std::optional<VulkanGemmConfig> config = MatchVulkanGemm(analysis);

  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->m, 17);
  EXPECT_EQ(config->n, 19);
  EXPECT_EQ(config->k, 21);
  EXPECT_EQ(config->rhs_layout, VulkanGemmRhsLayout::kNxK);
}

TEST_F(VulkanGemmEmitterTest, RejectsUnsupportedElementType) {
  constexpr absl::string_view kHlo = R"(
HloModule gemm

gemm {
  lhs = f32[17,21]{1,0} parameter(0)
  rhs = f32[19,21]{1,0} parameter(1)
  ROOT dot = f32[17,19]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f32[17,21]{1,0} parameter(0)
  rhs = f32[19,21]{1,0} parameter(1)
  ROOT fusion = f32[17,19]{1,0} fusion(lhs, rhs), kind=kLoop, calls=gemm
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  se::DeviceDescription device = VulkanDeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device);
  EXPECT_EQ(MatchVulkanGemm(analysis), std::nullopt);
}

TEST_F(VulkanGemmEmitterTest, RejectsUnsupportedLayout) {
  constexpr absl::string_view kHlo = R"(
HloModule gemm

gemm {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[19,21]{0,1} parameter(1)
  ROOT dot = bf16[17,19]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[19,21]{0,1} parameter(1)
  ROOT fusion = bf16[17,19]{1,0} fusion(lhs, rhs), kind=kLoop, calls=gemm
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  se::DeviceDescription device = VulkanDeviceInfo();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device);
  EXPECT_EQ(MatchVulkanGemm(analysis), std::nullopt);
}

TEST_F(VulkanGemmEmitterTest, LowersNormalRhsToVulkanSpirv) {
  ExpectLowersToVulkanSpirv(R"(
HloModule gemm

gemm {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[21,19]{1,0} parameter(1)
  ROOT dot = bf16[17,19]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[21,19]{1,0} parameter(1)
  ROOT fusion = bf16[17,19]{1,0} fusion(lhs, rhs), kind=kLoop, calls=gemm
})");
}

TEST_F(VulkanGemmEmitterTest, LowersTransposedRhsToVulkanSpirv) {
  ExpectLowersToVulkanSpirv(R"(
HloModule gemm

gemm {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[19,21]{1,0} parameter(1)
  ROOT dot = bf16[17,19]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[17,21]{1,0} parameter(0)
  rhs = bf16[19,21]{1,0} parameter(1)
  ROOT fusion = bf16[17,19]{1,0} fusion(lhs, rhs), kind=kLoop, calls=gemm
})");
}

}  // namespace
}  // namespace xla::gpu
