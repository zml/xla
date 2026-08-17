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

#include "xla/backends/gpu/codegen/emitters/vulkan_flash_attention.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

void ExpectSelectTensorElementOrZeroTypes(bool use_bf16) {
  mlir::MLIRContext context(MlirKernelEmitter::GetDialectRegistry());
  context.loadAllAvailableDialects();
  mlir::OpBuilder builder(&context);
  mlir::Type element_type =
      use_bf16 ? builder.getBF16Type() : builder.getF32Type();
  mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  mlir::FunctionType function_type = builder.getFunctionType(
      {mlir::RankedTensorType::get({1}, element_type)}, {});
  mlir::func::FuncOp function =
      mlir::func::FuncOp::create(builder, location, "test", function_type);
  module->push_back(function);

  mlir::ImplicitLocOpBuilder implicit_builder(location, function);
  implicit_builder.setInsertionPointToStart(function.addEntryBlock());
  mlir::Value condition = mlir::arith::ConstantIntOp::create(
      implicit_builder, implicit_builder.getI1Type(), true);
  mlir::Value index = mlir::arith::ConstantIndexOp::create(implicit_builder, 0);
  mlir::Value result = internal::SelectTensorElementOrZero(
      implicit_builder, condition, function.getArgument(0),
      mlir::ValueRange{index});
  mlir::func::ReturnOp::create(implicit_builder);

  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(result.getType(), element_type);

  mlir::scf::IfOp if_op = result.getDefiningOp<mlir::scf::IfOp>();
  ASSERT_TRUE(if_op);
  ASSERT_EQ(if_op->getNumResults(), 1);
  EXPECT_EQ(if_op.getResult(0).getType(), element_type);

  for (mlir::Region* region :
       {&if_op.getThenRegion(), &if_op.getElseRegion()}) {
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(region->front().getTerminator());
    ASSERT_EQ(yield.getNumOperands(), 1);
    EXPECT_EQ(yield.getOperand(0).getType(), element_type);
  }
}

TEST(VulkanFlashAttentionHelperTest, SelectTensorElementOrZeroTypes) {
  ExpectSelectTensorElementOrZeroTypes(/*use_bf16=*/true);
  ExpectSelectTensorElementOrZeroTypes(/*use_bf16=*/false);
}

class VulkanFlashAttentionEmitterTest : public HloHardwareIndependentTestBase {
};

TEST_F(VulkanFlashAttentionEmitterTest, LowersPrefillToVulkanSpirv) {
  constexpr absl::string_view kHlo = R"(
HloModule flash_attention

vulkan_flash_attention {
  q = bf16[2,2,16]{2,1,0} parameter(0)
  k = bf16[1,3,16]{2,1,0} parameter(1)
  v = bf16[1,3,16]{2,1,0} parameter(2)
  token_index = s32[] parameter(3)
  num_tokens = u32[] parameter(4)
  ROOT attention = bf16[2,2,16]{2,1,0} custom-call(
      q, k, v, token_index, num_tokens), custom_call_target="zml$flash_attn"
}

ENTRY main {
  q = bf16[2,2,16]{2,1,0} parameter(0)
  k = bf16[1,3,16]{2,1,0} parameter(1)
  v = bf16[1,3,16]{2,1,0} parameter(2)
  token_index = s32[] parameter(3)
  num_tokens = u32[] parameter(4)
  ROOT attention = bf16[2,2,16]{2,1,0} fusion(
      q, k, v, token_index, num_tokens), kind=kCustom,
      calls=vulkan_flash_attention
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());

  se::DeviceDescription device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  device.set_vulkan_compute_capability(
      /*api_version_major=*/1, /*api_version_minor=*/4,
      /*shader_bfloat16=*/true, /*storage_buffer_16bit_access=*/true,
      /*subgroup_size=*/32, /*subgroup_basic=*/true,
      /*subgroup_shuffle=*/true);

  std::unique_ptr<mlir::MLIRContext> context = CreateMlirContext();
  VulkanFlashAttentionEmitter emitter(*fusion, device);
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> emitted_module,
      emitter.CreateMLIRModule(*context, *fusion, "vulkan_flash_attention",
                               /*buffer_assignment=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      CompileMlirToLlvm(device, *module, "vulkan_flash_attention",
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

}  // namespace
}  // namespace xla::gpu
