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

#include "xla/service/gpu/metal_msl_emitter.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

std::unique_ptr<llvm::Module> ParseModule(absl::string_view source,
                                          llvm::LLVMContext& context) {
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(std::string(source), diagnostic, context);
  if (module == nullptr) {
    std::string message;
    llvm::raw_string_ostream os(message);
    diagnostic.print("metal_msl_emitter_test", os);
    ADD_FAILURE() << os.str();
  }
  return module;
}

TEST(MetalMslEmitterTest, EmitsBoundsCheckedElementwiseKernel) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @fusion(ptr %arg0, ptr %arg1, ptr %arg2) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = mul i32 %ctaid, %ntid
  %idx = add i32 %block, %tid
  %in_bounds = icmp ult i32 %idx, 1024
  br i1 %in_bounds, label %then, label %exit

then:
  %arg0_gep = getelementptr inbounds float, ptr %arg0, i32 %idx
  %arg1_gep = getelementptr inbounds float, ptr %arg1, i32 %idx
  %arg2_gep = getelementptr inbounds float, ptr %arg2, i32 %idx
  %lhs = load float, ptr %arg0_gep, align 4
  %rhs = load float, ptr %arg1_gep, align 4
  %sum = fadd float %lhs, %rhs
  store float %sum, ptr %arg2_gep, align 4
  br label %exit

exit:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @fusion, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("kernel void fusion("), std::string::npos);
  EXPECT_NE(msl.find("device float* arg0 [[buffer(0)]]"), std::string::npos);
  EXPECT_NE(msl.find("v0 = metal_tid.x;"), std::string::npos);
  EXPECT_NE(msl.find("v1 = metal_bid.x;"), std::string::npos);
  EXPECT_NE(msl.find("v2 = metal_ntid.x;"), std::string::npos);
  EXPECT_NE(msl.find("if (v5) {"), std::string::npos);
  EXPECT_NE(msl.find("arg2[v4] = v8;"), std::string::npos);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
