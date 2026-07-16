/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu::spirv {
namespace {

TEST(SpirvBackendTest, TestSPIRVExtensions) {
  auto extensions = SPIRVExtensionsEnumToString(common_spirv_extensions);
  auto extensions_set =
      std::set<std::string>(extensions.begin(), extensions.end());

  EXPECT_NE(extensions_set.find("SPV_EXT_optnone"), extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_uniform_group_instructions"),
            extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_linkonce_odr"), extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_cooperative_matrix"),
            extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_EXT_shader_atomic_float_add"),
            extensions_set.end());
  EXPECT_EQ(extensions_set.find("SPV_NV_cooperative_matrix"),
            extensions_set.end());
}

TEST(SpirvBackendTest, EmitsVulkanComputeShader) {
  llvm::LLVMContext context;
  llvm::Module module("vulkan_add", context);
  llvm::PointerType* buffer = llvm::PointerType::get(context, 11);
  llvm::Function* kernel = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                              {buffer, buffer}, false),
      llvm::GlobalValue::ExternalLinkage, "add_one", module);
  kernel->addFnAttr("hlsl.shader", "compute");
  kernel->addFnAttr("hlsl.numthreads", "1,1,1");
  kernel->getArg(0)->addAttr(llvm::Attribute::ReadOnly);

  llvm::IRBuilder<> builder(
      llvm::BasicBlock::Create(context, "entry", kernel));
  llvm::Value* value = builder.CreateLoad(builder.getInt32Ty(),
                                           kernel->getArg(0));
  builder.CreateStore(builder.CreateAdd(value, builder.getInt32(1)),
                      kernel->getArg(1));
  builder.CreateRetVoid();

  auto spirv = CompileToVulkanSPIRV(
      &module,
      stream_executor::GpuComputeCapability(
          stream_executor::VulkanComputeCapability(1, 2)),
      DebugOptions());
  ASSERT_TRUE(spirv.ok()) << spirv.status();
  ASSERT_GE(spirv->size(), sizeof(uint32_t));
  EXPECT_EQ(static_cast<uint8_t>((*spirv)[0]), 0x03);
  EXPECT_EQ(static_cast<uint8_t>((*spirv)[1]), 0x02);
  EXPECT_EQ(static_cast<uint8_t>((*spirv)[2]), 0x23);
  EXPECT_EQ(static_cast<uint8_t>((*spirv)[3]), 0x07);

  auto word = [&](size_t index) {
    size_t offset = index * sizeof(uint32_t);
    return static_cast<uint32_t>(static_cast<uint8_t>((*spirv)[offset])) |
           (static_cast<uint32_t>(
                static_cast<uint8_t>((*spirv)[offset + 1]))
            << 8) |
           (static_cast<uint32_t>(
                static_cast<uint8_t>((*spirv)[offset + 2]))
            << 16) |
           (static_cast<uint32_t>(
                static_cast<uint8_t>((*spirv)[offset + 3]))
            << 24);
  };
  bool has_gl_compute_entry_point = false;
  std::set<uint32_t> bindings;
  std::vector<std::vector<uint32_t>> decorations;
  int descriptor_set_zero_decorations = 0;
  for (size_t index = 5; index < spirv->size() / sizeof(uint32_t);) {
    uint32_t instruction = word(index);
    uint32_t word_count = instruction >> 16;
    uint32_t opcode = instruction & 0xffff;
    ASSERT_GT(word_count, 0);
    ASSERT_LE(index + word_count, spirv->size() / sizeof(uint32_t));
    // OpEntryPoint with ExecutionModel GLCompute.
    if (opcode == 15 && word_count >= 3 && word(index + 1) == 5) {
      has_gl_compute_entry_point = true;
    }
    // OpDecorate with Decoration Binding or DescriptorSet.
    if (opcode == 71 && word_count >= 4) {
      decorations.push_back(
          {word(index + 1), word(index + 2), word(index + 3)});
      if (word(index + 2) == 33) bindings.insert(word(index + 3));
      if (word(index + 2) == 34 && word(index + 3) == 0) {
        ++descriptor_set_zero_decorations;
      }
    }
    index += word_count;
  }
  EXPECT_TRUE(has_gl_compute_entry_point);
  EXPECT_EQ(bindings, (std::set<uint32_t>{0, 1}))
      << testing::PrintToString(decorations);
  EXPECT_EQ(descriptor_set_zero_decorations, 2)
      << testing::PrintToString(decorations);
}

}  // namespace
}  // namespace xla::gpu::spirv
