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

#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"
#include "xla/xla.pb.h"

namespace xla::gpu::spirv {
namespace {

std::vector<uint32_t> DecodeSpirvWords(std::string_view binary) {
  std::vector<uint32_t> words(binary.size() / sizeof(uint32_t));
  std::memcpy(words.data(), binary.data(), binary.size());
  return words;
}

bool ContainsSpirvOpcode(const std::vector<uint32_t>& words,
                         uint16_t opcode) {
  constexpr int kHeaderWordCount = 5;
  for (size_t i = kHeaderWordCount; i < words.size();) {
    uint16_t instruction_opcode = words[i] & 0xffff;
    uint16_t instruction_word_count = words[i] >> 16;
    if (instruction_opcode == opcode) {
      return true;
    }
    if (instruction_word_count == 0) {
      return false;
    }
    i += instruction_word_count;
  }
  return false;
}

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

TEST(SpirvBackendTest, UAddWithOverflowDoesNotLowerToIAddCarry) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("uaddo", context);
  llvm::IRBuilder<> builder(context);

  llvm::Type* i64_type = builder.getInt64Ty();
  llvm::Type* ptr_type = llvm::PointerType::get(context, /*AddressSpace=*/1);
  llvm::FunctionType* kernel_type =
      llvm::FunctionType::get(builder.getVoidTy(),
                              {ptr_type, ptr_type, i64_type, i64_type},
                              /*isVarArg=*/false);
  llvm::Function* kernel = llvm::Function::Create(
      kernel_type, llvm::GlobalValue::ExternalLinkage, "uaddo_func",
      module.get());
  kernel->setCallingConv(llvm::CallingConv::SPIR_FUNC);

  llvm::Function::arg_iterator args = kernel->arg_begin();
  llvm::Argument* sum_out = &*args++;
  llvm::Argument* overflow_out = &*args++;
  llvm::Argument* lhs = &*args++;
  llvm::Argument* rhs = &*args++;

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", kernel);
  builder.SetInsertPoint(entry);
  llvm::Value* result =
      builder.CreateBinaryIntrinsic(llvm::Intrinsic::uadd_with_overflow, lhs,
                                    rhs);
  llvm::Value* sum = builder.CreateExtractValue(result, {0}, "sum");
  llvm::Value* overflow = builder.CreateExtractValue(result, {1}, "overflow");
  builder.CreateStore(sum, sum_out);
  builder.CreateStore(builder.CreateZExt(overflow, i64_type), overflow_out);
  builder.CreateRetVoid();

  std::string verifier_errors;
  llvm::raw_string_ostream verifier_errors_stream(verifier_errors);
  EXPECT_FALSE(llvm::verifyModule(*module, &verifier_errors_stream))
      << verifier_errors_stream.str();

  absl::StatusOr<std::string> spirv =
      CompileToSPIRV(module.get(),
                     stream_executor::GpuComputeCapability(
                         stream_executor::OneAPIComputeCapability::BMG()),
                     DebugOptions());
  ASSERT_TRUE(spirv.ok()) << spirv.status();
  ASSERT_EQ(spirv->size() % sizeof(uint32_t), 0);

  std::vector<uint32_t> words = DecodeSpirvWords(*spirv);
  ASSERT_GE(words.size(), 5);
  EXPECT_EQ(words[0], 0x07230203);
  EXPECT_TRUE(ContainsSpirvOpcode(words, 128));   // OpIAdd
  EXPECT_FALSE(ContainsSpirvOpcode(words, 149));  // OpIAddCarry
  EXPECT_TRUE(ContainsSpirvOpcode(words, 176));   // OpULessThan
}

}  // namespace
}  // namespace xla::gpu::spirv
