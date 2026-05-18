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

TEST(MetalMslEmitterTest, EmitsArrayElementGepWithoutDoubleScaling) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @iota(ptr %arg0) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %value = sitofp i32 %tid to float
  %gep = getelementptr inbounds [8 x float], ptr %arg0, i32 0, i32 %tid
  store float %value, ptr %gep, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @iota, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("arg0[v0] = v1;"), std::string::npos) << msl;
  EXPECT_EQ(msl.find("arg0[(v0 * 8)]"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsVectorLoadFromScalarBuffer) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @dot(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %offset = mul i32 %tid, 2
  %gep = getelementptr inbounds [4 x float], ptr %arg0, i32 0, i32 %offset
  %vector = load <2 x float>, ptr %gep, align 4
  %element = extractelement <2 x float> %vector, i64 1
  %out = getelementptr inbounds [4 x float], ptr %arg1, i32 0, i32 %tid
  store float %element, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @dot, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("device float* arg0 [[buffer(0)]]"), std::string::npos)
      << msl;
  EXPECT_NE(msl.find("*reinterpret_cast<device float2*>(&arg0[v1])"),
            std::string::npos)
      << msl;
}

TEST(MetalMslEmitterTest, EmitsEmptyLibraryForModulesWithoutKernels) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

@constant = private unnamed_addr constant [4 x i8] c"test"
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("#include <metal_stdlib>"), std::string::npos);
  EXPECT_EQ(msl.find("kernel void"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsValidFloatConstantLiteral) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @constant(ptr %arg0) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %gep = getelementptr inbounds [4 x float], ptr %arg0, i32 0, i32 %tid
  store float 0.000000e+00, ptr %gep, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @constant, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("float(0)"), std::string::npos) << msl;
  EXPECT_EQ(msl.find("0f"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsValidInfinityConstantLiteral) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @constant(ptr %arg0) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %gep = getelementptr inbounds [4 x float], ptr %arg0, i32 0, i32 %tid
  store float 0xFFF0000000000000, ptr %gep, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @constant, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("float(-INFINITY)"), std::string::npos) << msl;
  EXPECT_EQ(msl.find("-inf"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsInlineReducerAndShuffleDown) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @reduce(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %gep = getelementptr inbounds [2 x float], ptr %arg0, i32 0, i32 %tid
  %value = load float, ptr %gep, align 4
  %sum = call float @scalar_add(float 0.000000e+00, float %value)
  %peer = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %sum, i32 1, i32 31)
  %total = call float @scalar_add(float %sum, float %peer)
  %is_first = icmp eq i32 %tid, 0
  br i1 %is_first, label %then, label %exit

then:
  store float %total, ptr %arg1, align 4
  br label %exit

exit:
  ret void
}

define internal float @scalar_add(float %lhs, float %rhs) {
entry:
  %sum = fadd float %lhs, %rhs
  ret float %sum
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare float @llvm.nvvm.shfl.sync.down.f32(i32, float, i32, i32)

!nvvm.annotations = !{!0}
!0 = !{ptr @reduce, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("simd_shuffle_down("), std::string::npos) << msl;
  EXPECT_NE(msl.find("float(0) +"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsLibdeviceAndMinMaxCalls) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @calls(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %limited = call i32 @llvm.smin.i32(i32 %tid, i32 4)
  %in = getelementptr inbounds [8 x float], ptr %arg0, i32 0, i32 %limited
  %value = load float, ptr %in, align 4
  %sine = call float @__nv_sinf(float %value)
  %out = getelementptr inbounds [8 x float], ptr %arg1, i32 0, i32 %tid
  store float %sine, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.smin.i32(i32, i32)
declare float @__nv_sinf(float)

!nvvm.annotations = !{!0}
!0 = !{ptr @calls, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("min(v0, 4)"), std::string::npos) << msl;
  EXPECT_NE(msl.find("sin(v2)"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsNegateAndFma) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @fma_kernel(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %in = getelementptr inbounds [8 x float], ptr %arg0, i32 0, i32 %tid
  %value = load float, ptr %in, align 4
  %negated = fneg float %value
  %fused = call float @__nv_fmaf(float %negated, float 2.000000e+00, float 1.000000e+00)
  %out = getelementptr inbounds [8 x float], ptr %arg1, i32 0, i32 %tid
  store float %fused, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare float @__nv_fmaf(float, float, float)

!nvvm.annotations = !{!0}
!0 = !{ptr @fma_kernel, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("(-v1)"), std::string::npos) << msl;
  EXPECT_NE(msl.find("fma(v2, float(2), float(1))"), std::string::npos)
      << msl;
}

TEST(MetalMslEmitterTest, EmitsIfElseDiamondWithPhi) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @pad(ptr %arg0, ptr %arg1, ptr %arg2) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ge_lower = icmp sge i32 %tid, 1
  %le_upper = icmp sle i32 %tid, 4
  %in_bounds = and i1 %ge_lower, %le_upper
  br i1 %in_bounds, label %then, label %else

then:
  %input_index = add i32 %tid, -1
  %input_gep = getelementptr inbounds [4 x float], ptr %arg0, i32 0, i32 %input_index
  %input_value = load float, ptr %input_gep, align 4
  br label %merge

else:
  %pad_gep = getelementptr inbounds [1 x float], ptr %arg1, i32 0, i32 0
  %pad_value = load float, ptr %pad_gep, align 4
  br label %merge

merge:
  %value = phi float [ %input_value, %then ], [ %pad_value, %else ]
  br label %exit

exit:
  %out_gep = getelementptr inbounds [7 x float], ptr %arg2, i32 0, i32 %tid
  store float %value, ptr %out_gep, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @pad, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("} else {"), std::string::npos) << msl;
  EXPECT_NE(msl.find("v4 = (v0 + -1);"), std::string::npos) << msl;
  EXPECT_NE(msl.find("arg2[v0] = v7;"), std::string::npos) << msl;
}

}  // namespace
}  // namespace gpu
}  // namespace xla
