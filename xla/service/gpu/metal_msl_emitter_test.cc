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

TEST(MetalMslEmitterTest, EmitsWideByteVectorCopy) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @copy(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %offset = mul i32 %tid, 8
  %in = getelementptr inbounds [64 x i8], ptr %arg0, i32 0, i32 %offset
  %vector = load <8 x i8>, ptr %in, align 1
  %out = getelementptr inbounds [64 x i8], ptr %arg1, i32 0, i32 %offset
  store <8 x i8> %vector, ptr %out, align 1
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @copy, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("struct xla_metal_vec8_char"), std::string::npos) << msl;
  EXPECT_NE(msl.find("*reinterpret_cast<device xla_metal_vec8_char*>(&arg0"
                     "[v1])"),
            std::string::npos)
      << msl;
  EXPECT_NE(msl.find("*reinterpret_cast<device xla_metal_vec8_char*>(&arg1"
                     "[v1])"),
            std::string::npos)
      << msl;
}

TEST(MetalMslEmitterTest, EmitsComplexFloatStructAsFloat2) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @complex_add(ptr %arg0, ptr %arg1, ptr %arg2) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lhs_gep = getelementptr inbounds [2 x { float, float }], ptr %arg0, i32 0, i32 %tid
  %lhs = load { float, float }, ptr %lhs_gep, align 4
  %rhs_gep = getelementptr inbounds [2 x { float, float }], ptr %arg1, i32 0, i32 %tid
  %rhs = load { float, float }, ptr %rhs_gep, align 4
  %lhs_real = extractvalue { float, float } %lhs, 0
  %rhs_real = extractvalue { float, float } %rhs, 0
  %real = fadd float %lhs_real, %rhs_real
  %lhs_imag = extractvalue { float, float } %lhs, 1
  %rhs_imag = extractvalue { float, float } %rhs, 1
  %imag = fadd float %lhs_imag, %rhs_imag
  %with_real = insertvalue { float, float } poison, float %real, 0
  %complex = insertvalue { float, float } %with_real, float %imag, 1
  %out = getelementptr inbounds [2 x { float, float }], ptr %arg2, i32 0, i32 %tid
  store { float, float } %complex, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @complex_add, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("device float2* arg0 [[buffer(0)]]"), std::string::npos)
      << msl;
  EXPECT_NE(msl.find("device float2* arg1 [[buffer(1)]]"), std::string::npos)
      << msl;
  EXPECT_NE(msl.find("device float2* arg2 [[buffer(2)]]"), std::string::npos)
      << msl;
  EXPECT_NE(msl.find(".x"), std::string::npos) << msl;
  EXPECT_NE(msl.find(".y"), std::string::npos) << msl;
  EXPECT_NE(msl.find("float2("), std::string::npos) << msl;
  EXPECT_NE(msl.find("arg2[v0] ="), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsArgumentBufferForLargeArityKernels) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @many_args(ptr %arg0, ptr %arg1, ptr %arg2, ptr %arg3, ptr %arg4, ptr %arg5, ptr %arg6, ptr %arg7, ptr %arg8, ptr %arg9, ptr %arg10, ptr %arg11, ptr %arg12, ptr %arg13, ptr %arg14, ptr %arg15, ptr %arg16, ptr %arg17, ptr %arg18, ptr %arg19, ptr %arg20, ptr %arg21, ptr %arg22, ptr %arg23, ptr %arg24, ptr %arg25, ptr %arg26, ptr %arg27, ptr %arg28, ptr %arg29, ptr %arg30, ptr %arg31) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %in = getelementptr inbounds [4 x float], ptr %arg31, i32 0, i32 %tid
  %value = load float, ptr %in, align 4
  %out = getelementptr inbounds [4 x float], ptr %arg0, i32 0, i32 %tid
  store float %value, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @many_args, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("struct many_args_args"), std::string::npos) << msl;
  EXPECT_NE(msl.find("device float* arg31 [[id(31)]]"), std::string::npos)
      << msl;
  EXPECT_NE(msl.find("kernel void many_args(device many_args_args& args "
                     "[[buffer(0)]]"),
            std::string::npos)
      << msl;
  EXPECT_NE(msl.find("device float* arg31 = args.arg31;"), std::string::npos)
      << msl;
  EXPECT_EQ(msl.find("[[buffer(31)]]"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsInlineComplexAggregateValue) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @complex_wrap(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %in = getelementptr inbounds [2 x { float, float }], ptr %arg0, i32 0, i32 %tid
  %value = load { float, float }, ptr %in, align 4
  %real = extractvalue { float, float } %value, 0
  %imag = extractvalue { float, float } %value, 1
  %wrapped = call { { float, float } } @wrap_complex(float %real, float %imag)
  %complex = extractvalue { { float, float } } %wrapped, 0
  %out = getelementptr inbounds [2 x { float, float }], ptr %arg1, i32 0, i32 %tid
  store { float, float } %complex, ptr %out, align 4
  ret void
}

define internal { { float, float } } @wrap_complex(float %real, float %imag) {
entry:
  %with_real = insertvalue { float, float } poison, float %real, 0
  %complex = insertvalue { float, float } %with_real, float %imag, 1
  %wrapped = insertvalue { { float, float } } poison, { float, float } %complex, 0
  ret { { float, float } } %wrapped
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @complex_wrap, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("float2("), std::string::npos) << msl;
  EXPECT_NE(msl.find("arg1[v0] ="), std::string::npos) << msl;
  EXPECT_EQ(msl.find("wrap_complex("), std::string::npos) << msl;
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

TEST(MetalMslEmitterTest, EmitsBfloat16AsUshortPayload) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @bf16_convert(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %in = getelementptr inbounds [4 x float], ptr %arg0, i32 0, i32 %tid
  %value = load float, ptr %in, align 4
  %bf16 = fptrunc float %value to bfloat
  %out = getelementptr inbounds [4 x bfloat], ptr %arg1, i32 0, i32 %tid
  store bfloat %bf16, ptr %out, align 2
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @bf16_convert, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("inline ushort xla_metal_f32_to_bf16"),
            std::string::npos)
      << msl;
  EXPECT_NE(msl.find("device ushort* arg1 [[buffer(1)]]"), std::string::npos)
      << msl;
  EXPECT_NE(msl.find("v2 = xla_metal_f32_to_bf16(v1);"), std::string::npos)
      << msl;
  EXPECT_NE(msl.find("arg1[v0] = v2;"), std::string::npos) << msl;
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

TEST(MetalMslEmitterTest, EmitsInlineFunctionArgumentLoad) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @caller(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %value = call float @read_value(ptr %arg0, i32 %tid)
  %out = getelementptr inbounds [4 x float], ptr %arg1, i32 0, i32 %tid
  store float %value, ptr %out, align 4
  ret void
}

define internal float @read_value(ptr %base, i32 %index) {
entry:
  %in = getelementptr inbounds [4 x float], ptr %base, i32 0, i32 %index
  %value = load float, ptr %in, align 4, !invariant.load !1
  ret float %value
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @caller, !"kernel", i32 1}
!1 = !{}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("arg0[v0]"), std::string::npos) << msl;
  EXPECT_NE(msl.find("arg1[v0] ="), std::string::npos) << msl;
  EXPECT_EQ(msl.find("read_value("), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsNestedInlineHelperCall) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @caller(ptr %arg0, ptr %arg1, ptr %arg2) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %value = call float @select_value(ptr %arg0, ptr %arg1, i32 %tid)
  %out = getelementptr inbounds [4 x float], ptr %arg2, i32 0, i32 %tid
  store float %value, ptr %out, align 4
  ret void
}

define internal float @select_value(ptr %lhs, ptr %rhs, i32 %index) {
entry:
  %flag = call i8 @compare_index(ptr %lhs, ptr %rhs, i32 %index)
  %lhs_gep = getelementptr inbounds [4 x float], ptr %lhs, i32 0, i32 %index
  %lhs_value = load float, ptr %lhs_gep, align 4, !invariant.load !1
  %rhs_gep = getelementptr inbounds [4 x float], ptr %rhs, i32 0, i32 %index
  %rhs_value = load float, ptr %rhs_gep, align 4, !invariant.load !1
  %condition = trunc i8 %flag to i1
  %selected = select i1 %condition, float %lhs_value, float %rhs_value
  ret float %selected
}

define internal i8 @compare_index(ptr %lhs, ptr %rhs, i32 %index) {
entry:
  %lhs_gep = getelementptr inbounds [4 x float], ptr %lhs, i32 0, i32 %index
  %lhs_value = load float, ptr %lhs_gep, align 4, !invariant.load !1
  %rhs_gep = getelementptr inbounds [4 x float], ptr %rhs, i32 0, i32 %index
  %rhs_value = load float, ptr %rhs_gep, align 4, !invariant.load !1
  %greater = fcmp ogt float %lhs_value, %rhs_value
  %result = zext i1 %greater to i8
  ret i8 %result
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @caller, !"kernel", i32 1}
!1 = !{}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("arg0[v0]"), std::string::npos) << msl;
  EXPECT_NE(msl.find("arg1[v0]"), std::string::npos) << msl;
  EXPECT_NE(msl.find("?"), std::string::npos) << msl;
  EXPECT_EQ(msl.find("select_value("), std::string::npos) << msl;
  EXPECT_EQ(msl.find("compare_index("), std::string::npos) << msl;
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

TEST(MetalMslEmitterTest, EmitsLogicalRightShiftWithUnsignedType) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @shift(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %in = getelementptr inbounds [4 x i32], ptr %arg0, i32 0, i32 %tid
  %value = load i32, ptr %in, align 4
  %shifted = lshr i32 %value, 1
  %out = getelementptr inbounds [4 x i32], ptr %arg1, i32 0, i32 %tid
  store i32 %shifted, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @shift, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("as_type<int>(static_cast<uint>(v1) >> 1)"),
            std::string::npos)
      << msl;
}

TEST(MetalMslEmitterTest, EmitsMoreMathCallsAndUnorderedCompare) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @math(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %in = getelementptr inbounds [8 x float], ptr %arg0, i32 0, i32 %tid
  %value = load float, ptr %in, align 4
  %floored = call float @__nv_floorf(float %value)
  %rounded = call float @__nv_rintf(float %floored)
  %powered = call float @__nv_powf(float %rounded, float 2.000000e+00)
  %signed = call float @__nv_copysignf(float %powered, float %value)
  %logged = call float @__nv_log1pf(float %signed)
  %unordered = fcmp uno float %value, %value
  %selected = select i1 %unordered, float 0.000000e+00, float %logged
  %out = getelementptr inbounds [8 x float], ptr %arg1, i32 0, i32 %tid
  store float %selected, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare float @__nv_floorf(float)
declare float @__nv_rintf(float)
declare float @__nv_powf(float, float)
declare float @__nv_copysignf(float, float)
declare float @__nv_log1pf(float)

!nvvm.annotations = !{!0}
!0 = !{ptr @math, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("floor(v1)"), std::string::npos) << msl;
  EXPECT_NE(msl.find("rint(v2)"), std::string::npos) << msl;
  EXPECT_NE(msl.find("pow(v3, float(2))"), std::string::npos) << msl;
  EXPECT_NE(msl.find("copysign(v4, v1)"), std::string::npos) << msl;
  EXPECT_NE(msl.find("log((1.0f + v5))"), std::string::npos) << msl;
  EXPECT_NE(msl.find("(isnan(v1) || isnan(v1))"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsIntegerBitCountIntrinsics) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @bit_count(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %in = getelementptr inbounds [8 x i32], ptr %arg0, i32 0, i32 %tid
  %value = load i32, ptr %in, align 4
  %population = call i32 @llvm.ctpop.i32(i32 %value)
  %leading = call i32 @llvm.ctlz.i32(i32 %value, i1 false)
  %sum = add i32 %population, %leading
  %out = getelementptr inbounds [8 x i32], ptr %arg1, i32 0, i32 %tid
  store i32 %sum, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.ctpop.i32(i32)
declare i32 @llvm.ctlz.i32(i32, i1 immarg)

!nvvm.annotations = !{!0}
!0 = !{ptr @bit_count, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("popcount(static_cast<uint>("), std::string::npos)
      << msl;
  EXPECT_NE(msl.find("clz(static_cast<uint>("), std::string::npos) << msl;
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

TEST(MetalMslEmitterTest, EmitsNestedGuardChain) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @scatter(ptr %arg0, ptr %arg1, ptr %arg2) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %has_update = icmp ult i32 %tid, 2
  br i1 %has_update, label %load_index, label %exit

load_index:
  %index_gep = getelementptr inbounds [2 x i32], ptr %arg1, i32 0, i32 %tid
  %index = load i32, ptr %index_gep, align 4
  %index_in_bounds = icmp ule i32 %index, 7
  br i1 %index_in_bounds, label %check_update, label %join_index

check_update:
  %update_in_bounds = icmp sle i32 %tid, 1
  br i1 %update_in_bounds, label %store_update, label %join_update

store_update:
  %update_gep = getelementptr inbounds [2 x float], ptr %arg2, i32 0, i32 %tid
  %update = load float, ptr %update_gep, align 4
  %out_gep = getelementptr inbounds [8 x float], ptr %arg0, i32 0, i32 %index
  store atomic float %update, ptr %out_gep unordered, align 4
  br label %join_update

join_update:
  br label %join_index

join_index:
  br label %exit

exit:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @scatter, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("if (v1) {"), std::string::npos) << msl;
  EXPECT_NE(msl.find("if (v3) {"), std::string::npos) << msl;
  EXPECT_NE(msl.find("if (v4) {"), std::string::npos) << msl;
  EXPECT_NE(msl.find("arg0[v2] = v5;"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsNestedDiamondToOuterMerge) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @pad_like(ptr %arg0, ptr %arg1, ptr %arg2, ptr %arg3) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %row = udiv i32 %tid, 3
  %col = urem i32 %tid, 3
  %in_outer = icmp sle i32 %row, 2
  br i1 %in_outer, label %check_inner, label %outer_else

check_inner:
  %row_ok = icmp sge i32 %row, 1
  %col_ok = icmp sle i32 %col, 1
  %in_inner = and i1 %row_ok, %col_ok
  br i1 %in_inner, label %inner_then, label %inner_else

inner_then:
  %source_index = add i32 %tid, -2
  %lhs_gep = getelementptr inbounds [6 x float], ptr %arg1, i32 0, i32 %source_index
  %lhs = load float, ptr %lhs_gep, align 4
  %rhs_gep = getelementptr inbounds [6 x float], ptr %arg2, i32 0, i32 %source_index
  %rhs = load float, ptr %rhs_gep, align 4
  %base_gep = getelementptr inbounds [6 x float], ptr %arg0, i32 0, i32 %source_index
  %base = load float, ptr %base_gep, align 4
  %product = fmul float %lhs, %rhs
  %sum = fadd float %base, %product
  br label %inner_merge

inner_else:
  br label %inner_merge

inner_merge:
  %inner_value = phi float [ 0.000000e+00, %inner_else ], [ %sum, %inner_then ]
  br label %outer_then_exit

outer_then_exit:
  br label %outer_merge

outer_else:
  br label %outer_merge

outer_merge:
  %value = phi float [ 1.000000e+00, %outer_else ], [ %inner_value, %outer_then_exit ]
  br label %store

store:
  %out_gep = getelementptr inbounds [18 x float], ptr %arg3, i32 0, i32 %tid
  store float %value, ptr %out_gep, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @pad_like, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("} else {"), std::string::npos) << msl;
  EXPECT_NE(msl.find("float(1)"), std::string::npos) << msl;
  EXPECT_NE(msl.find("arg3[v0] = "), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsAtomicFloatAdd) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @scatter_add(ptr %arg0, ptr %arg1) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %value_gep = getelementptr inbounds [8 x float], ptr %arg1, i32 0, i32 %tid
  %value = load float, ptr %value_gep, align 4
  %out_gep = getelementptr inbounds [8 x float], ptr %arg0, i32 0, i32 %tid
  %old = atomicrmw fadd ptr %out_gep, float %value monotonic, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @scatter_add, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("atomic_fetch_add_explicit"), std::string::npos) << msl;
  EXPECT_NE(msl.find("reinterpret_cast<device atomic_float*>(&arg0[v0])"),
            std::string::npos)
      << msl;
  EXPECT_NE(msl.find("memory_order_relaxed"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsCompareExchangeLoop) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @scatter_min(ptr %arg0, ptr %arg1, ptr %arg2) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %index_gep = getelementptr inbounds [3 x i32], ptr %arg1, i32 0, i32 %tid
  %index = load i32, ptr %index_gep, align 4
  %update_gep = getelementptr inbounds [3 x float], ptr %arg2, i32 0, i32 %tid
  %update = load float, ptr %update_gep, align 4
  %out = getelementptr inbounds [6 x float], ptr %arg0, i32 0, i32 %index
  %old = load i32, ptr %out, align 4
  br label %loop

loop:
  %current = phi i32 [ %old, %entry ], [ %actual, %loop ]
  %current_float = bitcast i32 %current to float
  %next_float = call float @llvm.minimum.f32(float %current_float, float %update)
  %next = bitcast float %next_float to i32
  %cas = cmpxchg ptr %out, i32 %current, i32 %next monotonic monotonic, align 4
  %actual = extractvalue { i32, i1 } %cas, 0
  %success = extractvalue { i32, i1 } %cas, 1
  %retry = xor i1 %success, true
  br i1 %retry, label %loop, label %exit

exit:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare float @llvm.minimum.f32(float, float)

!nvvm.annotations = !{!0}
!0 = !{ptr @scatter_min, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("do {"), std::string::npos) << msl;
  EXPECT_NE(msl.find("atomic_compare_exchange_weak_explicit"),
            std::string::npos)
      << msl;
  EXPECT_NE(msl.find("cmpxchg_old0"), std::string::npos) << msl;
  EXPECT_NE(msl.find("cmpxchg_success0"), std::string::npos) << msl;
  EXPECT_NE(msl.find("} while (v"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsLocalStackSlot) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @local_buffer(ptr %arg0) {
entry:
  %slot = alloca i8, align 1
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %is_first = icmp eq i32 %tid, 0
  call void @llvm.assume(i1 %is_first)
  %value = zext i1 %is_first to i8
  store i8 %value, ptr %slot, align 1
  %loaded = load i8, ptr %slot, align 1
  %out = getelementptr inbounds [1 x i8], ptr %arg0, i32 0, i32 0
  store i8 %loaded, ptr %out, align 1
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.assume(i1)

!nvvm.annotations = !{!0}
!0 = !{ptr @local_buffer, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("char local0[1] = {0};"), std::string::npos) << msl;
  EXPECT_NE(msl.find("local0[0] = v2;"), std::string::npos) << msl;
  EXPECT_NE(msl.find("v3 = local0[0];"), std::string::npos) << msl;
  EXPECT_EQ(msl.find("llvm.assume"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsThreadgroupGlobal) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

@tile = private addrspace(3) global [64 x float] undef, align 4

define void @shared_tile(ptr %arg0) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %in = getelementptr inbounds [64 x float], ptr addrspace(3) @tile, i32 0, i32 %tid
  store float 1.000000e+00, ptr addrspace(3) %in, align 4
  %loaded = load float, ptr addrspace(3) %in, align 4
  %out = getelementptr inbounds [64 x float], ptr %arg0, i32 0, i32 %tid
  store float %loaded, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @shared_tile, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("threadgroup float tile[64];"), std::string::npos) << msl;
  EXPECT_NE(msl.find("tile[v0] = float(1);"), std::string::npos) << msl;
  EXPECT_NE(msl.find("v1 = tile[v0];"), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, InlinesVoidHelperWithOutputBuffer) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

@nan_bits = constant [4 x i8] c"\00\00\C0\7F", align 4

define void @caller(ptr %arg0, ptr %arg1) {
entry:
  %ret = alloca i8, align 1
  call void @compare(ptr %arg0, ptr %ret)
  %value = load i8, ptr %ret, align 1
  store i8 %value, ptr %arg1, align 1
  ret void
}

define internal void @compare(ptr %lhs, ptr %out) {
entry:
  %slot = alloca float, align 4
  %value = load float, ptr %lhs, align 4
  %nan = load float, ptr @nan_bits, align 4
  %sum = fadd float %value, %nan
  store float %sum, ptr %slot, align 4
  %bits = load i32, ptr %slot, align 4
  %is_negative = icmp slt i32 %bits, 0
  %out_value = zext i1 %is_negative to i8
  store i8 %out_value, ptr %out, align 1
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @caller, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("as_type<float>(2143289344u)"), std::string::npos)
      << msl;
  EXPECT_NE(msl.find("as_type<int>("), std::string::npos) << msl;
  EXPECT_NE(msl.find("local0[0] = static_cast<char>"), std::string::npos)
      << msl;
  EXPECT_EQ(msl.find("compare("), std::string::npos) << msl;
}

TEST(MetalMslEmitterTest, EmitsTupleReducerArgmax) {
  constexpr absl::string_view kLlvmIr = R"(
target datalayout = "e-p:64:64-i64:64-n32:64-S128"
target triple = "nvptx64-nvidia-cuda"

define void @argmax(ptr %arg0, ptr %arg1, ptr %arg2, ptr %arg3) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %in_bounds = icmp sle i32 %tid, 7
  br i1 %in_bounds, label %load, label %empty

load:
  %value_gep = getelementptr inbounds [8 x float], ptr %arg0, i32 0, i32 %tid
  %value = load float, ptr %value_gep, align 4
  %index_gep = getelementptr inbounds [8 x i32], ptr %arg1, i32 0, i32 %tid
  %index = load i32, ptr %index_gep, align 4
  %pair = call { float, i32 } @argmax_region(float 0xFFF0000000000000, i32 0, float %value, i32 %index)
  %pair_value = extractvalue { float, i32 } %pair, 0
  %pair_index = extractvalue { float, i32 } %pair, 1
  br label %merge

empty:
  br label %merge

merge:
  %value_phi = phi float [ %pair_value, %load ], [ 0xFFF0000000000000, %empty ]
  %index_phi = phi i32 [ %pair_index, %load ], [ 0, %empty ]
  %peer_value = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %value_phi, i32 1, i32 31)
  %peer_index = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %index_phi, i32 1, i32 31)
  %reduced = call { float, i32 } @argmax_region(float %value_phi, i32 %index_phi, float %peer_value, i32 %peer_index)
  %out_value = extractvalue { float, i32 } %reduced, 0
  %out_index = extractvalue { float, i32 } %reduced, 1
  %is_first = icmp eq i32 %tid, 0
  br i1 %is_first, label %store, label %exit

store:
  %out_value_gep = getelementptr inbounds [1 x float], ptr %arg2, i32 0, i32 0
  store float %out_value, ptr %out_value_gep, align 4
  %out_index_gep = getelementptr inbounds [1 x i32], ptr %arg3, i32 0, i32 0
  store i32 %out_index, ptr %out_index_gep, align 4
  br label %exit

exit:
  ret void
}

define internal { float, i32 } @argmax_region(float %lhs_value, i32 %lhs_index, float %rhs_value, i32 %rhs_index) {
entry:
  %lhs_greater = fcmp ogt float %lhs_value, %rhs_value
  %lhs_nan = fcmp une float %lhs_value, %lhs_value
  %take_lhs_value = or i1 %lhs_greater, %lhs_nan
  %value = select i1 %take_lhs_value, float %lhs_value, float %rhs_value
  %values_equal = fcmp oeq float %lhs_value, %rhs_value
  %lhs_index_lower = icmp slt i32 %lhs_index, %rhs_index
  %take_lhs_tie = and i1 %values_equal, %lhs_index_lower
  %take_lhs_index = or i1 %take_lhs_value, %take_lhs_tie
  %index = select i1 %take_lhs_index, i32 %lhs_index, i32 %rhs_index
  %with_value = insertvalue { float, i32 } poison, float %value, 0
  %with_index = insertvalue { float, i32 } %with_value, i32 %index, 1
  ret { float, i32 } %with_index
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare float @llvm.nvvm.shfl.sync.down.f32(i32, float, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)

!nvvm.annotations = !{!0}
!0 = !{ptr @argmax, !"kernel", i32 1}
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseModule(kLlvmIr, context);
  ASSERT_NE(module, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string msl, EmitMslFromLlvmModule(*module));

  EXPECT_NE(msl.find("} else {"), std::string::npos) << msl;
  EXPECT_NE(msl.find("simd_shuffle_down(v6, 1)"), std::string::npos) << msl;
  EXPECT_NE(msl.find("simd_shuffle_down(v7, 1)"), std::string::npos) << msl;
  EXPECT_NE(msl.find("arg2[0] = v10;"), std::string::npos) << msl;
  EXPECT_NE(msl.find("arg3[0] = v11;"), std::string::npos) << msl;
  EXPECT_EQ(msl.find("{ float, i32 }"), std::string::npos) << msl;
}

}  // namespace
}  // namespace gpu
}  // namespace xla
