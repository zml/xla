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

#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

std::string CorpusPath(const std::string& name) {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  const char* test_workspace = std::getenv("TEST_WORKSPACE");
  EXPECT_NE(test_srcdir, nullptr);
  EXPECT_NE(test_workspace, nullptr);
  return absl::StrCat(test_srcdir == nullptr ? "" : test_srcdir, "/",
                      test_workspace == nullptr ? "" : test_workspace,
                      "/xla/service/gpu/musa/testdata/llvm14_compatibility/",
                      name);
}

std::string ReadCorpus(const std::string& name) {
  std::ifstream input(CorpusPath(name), std::ios::binary);
  EXPECT_TRUE(input.is_open()) << name;
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::string Module(absl::string_view body, absl::string_view declarations = "",
                   absl::string_view globals = "",
                   absl::string_view metadata = "",
                   absl::string_view argument = "ptr addrspace(1) %out") {
  return absl::StrCat(
      "source_filename = \"test-input\"\n", "target datalayout = \"",
      kMusaDataLayout, "\"\n", "target triple = \"", kMusaTargetTriple,
      "\"\n\n", globals, "\ndefine void @kernel(", argument, ") #0 {\nentry:\n",
      body, "\n  ret void\n}\n\n", declarations,
      "\nattributes #0 = { \"xla.musa.kernel.v1\" }\n", metadata);
}

TEST(MusaLlvm14CompatibilityTest,
     NormalizesOpaquePointerCorpusToExactLlvm14Golden) {
  const std::string input = ReadCorpus("elemental.current.ll");
  const std::string expected = ReadCorpus("elemental.llvm14.ll");
  absl::StatusOr<MusaLlvm14CompatibilityResult> result =
      NormalizeMusaLlvmTextForLlvm14(input, "compatibility_corpus");
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->normalized_llvm, expected);
  EXPECT_EQ(result->normalized_llvm_sha256, MusaBridgeSha256Hex(expected));
  EXPECT_EQ(result->metadata.module_name, "compatibility_corpus");
  EXPECT_EQ(result->metadata.kernel_entry_names,
            std::vector<std::string>{"kernel"});
  EXPECT_TRUE(result->metadata.exported_globals.empty());
  EXPECT_FALSE(
      absl::StrContains(result->normalized_llvm, kMusaLlvmKernelMarker));
  EXPECT_FALSE(absl::StrContains(result->normalized_llvm, "memory("));
  EXPECT_FALSE(absl::StrContains(result->normalized_llvm, "ignored/source"));
}

TEST(MusaLlvm14CompatibilityTest, IsDeterministicAndDoesNotMutateInput) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module = llvm::parseAssemblyString(
      ReadCorpus("elemental.current.ll"), diagnostic, context);
  ASSERT_NE(module, nullptr);
  ASSERT_NE(module->getFunction("kernel"), nullptr);
  EXPECT_TRUE(
      module->getFunction("kernel")->hasFnAttribute(kMusaLlvmKernelMarker));

  absl::StatusOr<MusaLlvm14CompatibilityResult> first =
      NormalizeMusaLlvmForLlvm14(*module, "deterministic");
  absl::StatusOr<MusaLlvm14CompatibilityResult> second =
      NormalizeMusaLlvmForLlvm14(*module, "deterministic");
  ASSERT_THAT(first, IsOk());
  ASSERT_THAT(second, IsOk());
  EXPECT_EQ(first->normalized_llvm, second->normalized_llvm);
  EXPECT_EQ(first->normalized_llvm_sha256, second->normalized_llvm_sha256);
  EXPECT_TRUE(
      module->getFunction("kernel")->hasFnAttribute(kMusaLlvmKernelMarker));
  EXPECT_EQ(module->getSourceFileName(), "ignored/source/path");
}

TEST(MusaLlvm14CompatibilityTest,
     PreservesExactMappingV3AtomicCmpXchgForVendorLlvm14) {
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(
          Module("  %pair = cmpxchg ptr addrspace(1) %out, i32 0, i32 1 "
                 "monotonic monotonic, align 4"),
          "atomic_cmpxchg_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_THAT(normalized->normalized_llvm,
              HasSubstr("cmpxchg ptr addrspace(1)"));
  EXPECT_THAT(normalized->normalized_llvm,
              HasSubstr("monotonic monotonic, align 4"));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest, StripsIdentDebugFlagsAndInvariantLoadOnly) {
  const std::string metadata =
      "!llvm.ident = !{!1}\n"
      "!llvm.module.flags = !{!2, !3}\n"
      "!llvm.dbg.cu = !{!4}\n"
      "!0 = !{}\n"
      "!1 = !{!\"producer/path\"}\n"
      "!2 = !{i32 2, !\"Dwarf Version\", i32 4}\n"
      "!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n"
      "!4 = distinct !DICompileUnit(language: DW_LANG_C, file: !5, "
      "producer: \"producer/path\", isOptimized: false, "
      "runtimeVersion: 0, emissionKind: FullDebug)\n"
      "!5 = !DIFile(filename: \"source.cc\", "
      "directory: \"sensitive/debug/path\")\n";
  const std::string input = Module(
      "  %value = load i32, ptr addrspace(1) %out, align 4, "
      "!invariant.load !0",
      "", "", metadata);
  absl::StatusOr<MusaLlvm14CompatibilityResult> result =
      NormalizeMusaLlvmTextForLlvm14(input, "metadata_strip");
  ASSERT_THAT(result, IsOk());
  EXPECT_FALSE(absl::StrContains(result->normalized_llvm, "llvm.ident"));
  EXPECT_FALSE(absl::StrContains(result->normalized_llvm, "llvm.module.flags"));
  EXPECT_FALSE(absl::StrContains(result->normalized_llvm, "invariant.load"));
  EXPECT_FALSE(absl::StrContains(result->normalized_llvm, "producer/path"));
  EXPECT_FALSE(absl::StrContains(result->normalized_llvm, "DICompileUnit"));
  EXPECT_FALSE(
      absl::StrContains(result->normalized_llvm, "sensitive/debug/path"));
}

TEST(MusaLlvm14CompatibilityTest, RejectsNonDebugModuleFlags) {
  const std::string metadata =
      "!llvm.module.flags = !{!0}\n"
      "!0 = !{i32 1, !\"wchar_size\", i32 4}\n";
  EXPECT_THAT(NormalizeMusaLlvmTextForLlvm14(Module("", "", "", metadata),
                                             "module_flags"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=module-flags")));
}

TEST(MusaLlvm14CompatibilityTest, KernelMarkerAndAbiFailClosed) {
  const std::string valid = Module("");
  for (const std::pair<std::string, std::string>& replacement : {
           std::pair{" #0 {", " {"},
           std::pair{"\"xla.musa.kernel.v1\"", "\"xla.musa.kernel.v1\"=\"2\""},
           std::pair{"define void @kernel", "define cc 102 void @kernel"},
       }) {
    const std::string input =
        absl::StrReplaceAll(valid, {{replacement.first, replacement.second}});
    EXPECT_THAT(NormalizeMusaLlvmTextForLlvm14(input, "kernel_marker"),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("capability=kernel-marker")));
  }

  const std::string unmarked_helper =
      absl::StrReplaceAll(valid, {{"define void @kernel",
                                   "define void @unmarked() { ret void }\n\n"
                                   "define void @kernel"}});
  EXPECT_THAT(NormalizeMusaLlvmTextForLlvm14(unmarked_helper, "kernel_marker"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=kernel-marker")));
}

TEST(MusaLlvm14CompatibilityTest, KernelArgumentPolicyFailClosed) {
  absl::StatusOr<MusaLlvm14CompatibilityResult> accepted =
      NormalizeMusaLlvmTextForLlvm14(
          Module("", "", "", "",
                 "ptr addrspace(1) noalias align 16 dereferenceable(16) %out"),
          "kernel_arguments");
  ASSERT_THAT(accepted, IsOk());
  EXPECT_THAT(accepted->normalized_llvm, Not(HasSubstr("noalias")));
  EXPECT_THAT(accepted->normalized_llvm, Not(HasSubstr("dereferenceable")));

  for (absl::string_view argument :
       {absl::string_view("ptr addrspace(1) nocapture %out"),
        absl::string_view(
            "ptr addrspace(1) dereferenceable_or_null(4) %out")}) {
    EXPECT_THAT(NormalizeMusaLlvmTextForLlvm14(Module("", "", "", "", argument),
                                               "kernel_arguments"),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("capability=kernel-attributes")));
  }

  const std::string target_attribute = absl::StrReplaceAll(
      Module(""), {{R"("xla.musa.kernel.v1")",
                    R"("xla.musa.kernel.v1" "target-cpu"="unqualified")"}});
  EXPECT_THAT(
      NormalizeMusaLlvmTextForLlvm14(target_attribute, "kernel_arguments"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=kernel-attributes")));
}

TEST(MusaLlvm14CompatibilityTest,
     StripsNoAliasFromLocalHelperPointerArgumentsOnly) {
  const std::string helper =
      "define internal i32 @helper(ptr noalias dereferenceable(4) %buffer, "
      "i32 %value) {\n"
      "entry:\n"
      "  ret i32 %value\n"
      "}\n";
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(
          Module("  %generic = addrspacecast ptr addrspace(1) %out to ptr\n"
                 "  %value = call i32 @helper(ptr %generic, i32 7)\n"
                 "  store i32 %value, ptr addrspace(1) %out, align 4",
                 helper),
          "helper_noalias");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("define internal i32"));
  EXPECT_THAT(normalized->normalized_llvm, Not(HasSubstr("ptr noalias")));
  EXPECT_THAT(normalized->normalized_llvm, Not(HasSubstr("dereferenceable")));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());

  const std::string unqualified = absl::StrReplaceAll(
      helper, {{"noalias dereferenceable(4)", "nocapture"}});
  EXPECT_THAT(
      NormalizeMusaLlvmTextForLlvm14(
          Module("  %generic = addrspacecast ptr addrspace(1) %out to ptr\n"
                 "  %value = call i32 @helper(ptr %generic, i32 7)\n"
                 "  store i32 %value, ptr addrspace(1) %out, align 4",
                 unqualified),
          "helper_nocapture"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=helper-attributes")));
}

TEST(MusaLlvm14CompatibilityTest,
     StripsOnlyGeneratedLegacyBitonicOptimizationHints) {
  const std::string text = Module(
      "  %block = call i32 @__xla_musa_v1_read_ctaid_x(), !range !0\n"
      "  %thread = call i32 @__xla_musa_v1_read_tid_x(), !range !1\n"
      "  %in_range = icmp ult i32 %thread, 8\n"
      "  call void @llvm.assume(i1 %in_range)\n"
      "  call void @__xla_musa_v1_workgroup_barrier()\n"
      "  %value = add i32 %block, %thread\n"
      "  store i32 %value, ptr addrspace(1) %out, align 4",
      "declare i32 @__xla_musa_v1_read_ctaid_x() #1\n"
      "declare i32 @__xla_musa_v1_read_tid_x() #2\n"
      "declare void @__xla_musa_v1_workgroup_barrier() #3\n"
      "declare void @llvm.assume(i1)\n",
      "",
      "!0 = !{i32 0, i32 2}\n!1 = !{i32 0, i32 8}\n\n"
      "attributes #1 = { convergent nounwind memory(none) }\n"
      "attributes #2 = { nounwind memory(none) }\n"
      "attributes #3 = { convergent nounwind }\n");
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(text, diagnostic, context);
  ASSERT_NE(module, nullptr);
  llvm::Function* assume = module->getFunction("llvm.assume");
  ASSERT_NE(assume, nullptr);
  assume->setAttributes(llvm::Intrinsic::getAttributes(
      context, assume->getIntrinsicID(), assume->getFunctionType()));

  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmForLlvm14(*module, "legacy_bitonic_hints");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_THAT(normalized->normalized_llvm, Not(HasSubstr("llvm.assume")));
  EXPECT_THAT(normalized->normalized_llvm, Not(HasSubstr("!range")));
  EXPECT_THAT(normalized->normalized_llvm, Not(HasSubstr("!0 =")));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("%in_range = icmp"));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest, RewritesEveryActiveMappingShimMemoryProfile) {
  std::string calls;
  std::string declarations;
  int result_number = 0;
  for (const MusaShimSpec& spec : MusaShimSpecs()) {
    absl::string_view return_type;
    absl::string_view arguments;
    switch (spec.signature) {
      case MusaShimSignature::kVoidVoid:
        return_type = "void";
        absl::StrAppend(&calls, "  call void @", spec.xla_symbol, "()\n");
        break;
      case MusaShimSignature::kI32Void:
        return_type = "i32";
        absl::StrAppend(&calls, "  %r", result_number++, " = call i32 @",
                        spec.xla_symbol, "()\n");
        break;
      case MusaShimSignature::kI64Void:
        return_type = "i64";
        absl::StrAppend(&calls, "  %r", result_number++, " = call i64 @",
                        spec.xla_symbol, "()\n");
        break;
      case MusaShimSignature::kI32I32I32:
        return_type = "i32";
        arguments = "i32 7, i32 0";
        absl::StrAppend(&calls, "  %r", result_number++, " = call i32 @",
                        spec.xla_symbol, "(", arguments, ")\n");
        break;
    }
    absl::StrAppend(&declarations, "declare ", return_type, " @",
                    spec.xla_symbol, "(", arguments.empty() ? "" : "i32, i32",
                    ") ");
    if (spec.convergent) absl::StrAppend(&declarations, "convergent ");
    if ((spec.required_attributes & kNoUnwind) != 0) {
      absl::StrAppend(&declarations, "nounwind ");
    }
    if ((spec.required_attributes & kWillReturn) != 0) {
      absl::StrAppend(&declarations, "willreturn ");
    }
    switch (spec.memory_effects) {
      case MusaMemoryEffects::kNone:
        absl::StrAppend(&declarations, "memory(none)");
        break;
      case MusaMemoryEffects::kReadWrite:
        break;
      case MusaMemoryEffects::kInaccessibleRead:
        absl::StrAppend(&declarations, "memory(inaccessiblemem: read)");
        break;
      case MusaMemoryEffects::kInaccessibleReadWrite:
        absl::StrAppend(&declarations, "memory(inaccessiblemem: readwrite)");
        break;
    }
    absl::StrAppend(&declarations, "\n");
  }

  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(Module(calls, declarations), "all_shims");
  ASSERT_TRUE(normalized.ok()) << normalized.status();
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "memory("));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("readnone"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("inaccessiblememonly"));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest, NormalizesReviewedSqrtProfile) {
  const std::string text = Module(
      "  %value = call float @llvm.sqrt.f32(float 4.0)\n"
      "  %bits = bitcast float %value to i32\n"
      "  store i32 %bits, ptr addrspace(1) %out, align 4",
      "declare float @llvm.sqrt.f32(float)");
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(text, diagnostic, context);
  ASSERT_NE(module, nullptr);
  llvm::Function* sqrt = module->getFunction("llvm.sqrt.f32");
  ASSERT_NE(sqrt, nullptr);
  sqrt->setAttributes(llvm::Intrinsic::getAttributes(
      context, sqrt->getIntrinsicID(), sqrt->getFunctionType()));

  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmForLlvm14(*module, "sqrt_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("nofree"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("nosync"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("nounwind"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("readnone"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("speculatable"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("willreturn"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "nocallback"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "mustprogress"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "memory("));
  EXPECT_EQ(normalized->normalized_llvm, ReadCorpus("sqrt.llvm14.ll"));
}

TEST(MusaLlvm14CompatibilityTest, NormalizesReviewedFabsProfile) {
  const std::string text = Module(
      "  %value = call float @llvm.fabs.f32(float -2.0)\n"
      "  %bits = bitcast float %value to i32\n"
      "  store i32 %bits, ptr addrspace(1) %out, align 4",
      "declare float @llvm.fabs.f32(float)");
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(text, diagnostic, context);
  ASSERT_NE(module, nullptr);
  llvm::Function* fabs = module->getFunction("llvm.fabs.f32");
  ASSERT_NE(fabs, nullptr);
  fabs->setAttributes(llvm::Intrinsic::getAttributes(
      context, fabs->getIntrinsicID(), fabs->getFunctionType()));

  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmForLlvm14(*module, "fabs_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_EQ(normalized->normalized_llvm, ReadCorpus("fabs.llvm14.ll"));
}

TEST(MusaLlvm14CompatibilityTest, NormalizesReviewedSineProfile) {
  const std::string text = Module(
      "  %value = call float @llvm.sin.f32(float 0.0)\n"
      "  %bits = bitcast float %value to i32\n"
      "  store i32 %bits, ptr addrspace(1) %out, align 4",
      "declare float @llvm.sin.f32(float)");
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(text, diagnostic, context);
  ASSERT_NE(module, nullptr);
  llvm::Function* sine = module->getFunction("llvm.sin.f32");
  ASSERT_NE(sine, nullptr);
  sine->setAttributes(llvm::Intrinsic::getAttributes(
      context, sine->getIntrinsicID(), sine->getFunctionType()));

  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmForLlvm14(*module, "sine_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "nocallback"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "mustprogress"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "memory("));
  EXPECT_EQ(normalized->normalized_llvm, ReadCorpus("sine.llvm14.ll"));
}

TEST(MusaLlvm14CompatibilityTest, NormalizesReviewedIntegerMinMaxProfiles) {
  const std::string text = Module(
      "  %smin = call i32 @llvm.smin.i32(i32 7, i32 3)\n"
      "  %smax = call i32 @llvm.smax.i32(i32 %smin, i32 5)\n"
      "  %umin = call i32 @llvm.umin.i32(i32 %smax, i32 4)\n"
      "  %umax = call i32 @llvm.umax.i32(i32 %umin, i32 6)\n"
      "  store i32 %umax, ptr addrspace(1) %out, align 4",
      "declare i32 @llvm.smin.i32(i32, i32)\n"
      "declare i32 @llvm.smax.i32(i32, i32)\n"
      "declare i32 @llvm.umin.i32(i32, i32)\n"
      "declare i32 @llvm.umax.i32(i32, i32)");
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(text, diagnostic, context);
  ASSERT_NE(module, nullptr);
  for (absl::string_view name :
       {"llvm.smin.i32", "llvm.smax.i32", "llvm.umin.i32", "llvm.umax.i32"}) {
    llvm::Function* function = module->getFunction(name);
    ASSERT_NE(function, nullptr);
    function->setAttributes(llvm::Intrinsic::getAttributes(
        context, function->getIntrinsicID(), function->getFunctionType()));
  }

  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmForLlvm14(*module, "integer_minmax_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("readnone"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "nocallback"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "mustprogress"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "memory("));
  EXPECT_EQ(normalized->normalized_llvm,
            ReadCorpus("integer_minmax.llvm14.ll"));
}

TEST(MusaLlvm14CompatibilityTest, NormalizesReviewedFloatingMinMaxProfiles) {
  const std::string text = Module(
      "  %minimum = call float @llvm.minimum.f32(float %lhs, float %rhs)\n"
      "  %maximum = call nsz float @llvm.maximum.f32(float %minimum, float "
      "%rhs)\n"
      "  %minnum = call float @llvm.minnum.f32(float %maximum, float %lhs)\n"
      "  %maxnum = call float @llvm.maxnum.f32(float %minnum, float %rhs)\n"
      "  %bits = bitcast float %maxnum to i32\n"
      "  store i32 %bits, ptr addrspace(1) %out, align 4",
      "declare float @llvm.minimum.f32(float, float)\n"
      "declare float @llvm.maximum.f32(float, float)\n"
      "declare float @llvm.minnum.f32(float, float)\n"
      "declare float @llvm.maxnum.f32(float, float)",
      "", "", "ptr addrspace(1) %out, float %lhs, float %rhs");
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(text, diagnostic, context);
  ASSERT_NE(module, nullptr);
  for (absl::string_view name : {"llvm.minimum.f32", "llvm.maximum.f32",
                                 "llvm.minnum.f32", "llvm.maxnum.f32"}) {
    llvm::Function* function = module->getFunction(name);
    ASSERT_NE(function, nullptr);
    function->setAttributes(llvm::Intrinsic::getAttributes(
        context, function->getIntrinsicID(), function->getFunctionType()));
  }

  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmForLlvm14(*module, "floating_minmax_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "llvm.minimum"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "llvm.maximum"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "llvm.minnum"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "llvm.maxnum"));
  EXPECT_EQ(normalized->normalized_llvm,
            ReadCorpus("floating_minmax.llvm14.ll"));
}

TEST(MusaLlvm14CompatibilityTest, RejectsUnqualifiedFloatingMinMaxFastMath) {
  const std::string text = Module(
      "  %value = call nnan float @llvm.maximum.f32(float %lhs, float %rhs)",
      "declare float @llvm.maximum.f32(float, float)", "", "",
      "ptr addrspace(1) %out, float %lhs, float %rhs");
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(text, diagnostic, context);
  ASSERT_NE(module, nullptr);
  llvm::Function* maximum = module->getFunction("llvm.maximum.f32");
  ASSERT_NE(maximum, nullptr);
  maximum->setAttributes(llvm::Intrinsic::getAttributes(
      context, maximum->getIntrinsicID(), maximum->getFunctionType()));

  EXPECT_THAT(NormalizeMusaLlvmForLlvm14(*module, "minmax_fast_math"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=fast-math-flags")));
}

TEST(MusaLlvm14CompatibilityTest, RejectsFloatingMinMaxCallSiteAttributes) {
  const std::string text = Module(
      "  %value = call noundef float @llvm.maximum.f32(float %lhs, float "
      "%rhs)",
      "declare float @llvm.maximum.f32(float, float)", "", "",
      "ptr addrspace(1) %out, float %lhs, float %rhs");
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(text, diagnostic, context);
  ASSERT_NE(module, nullptr);
  llvm::Function* maximum = module->getFunction("llvm.maximum.f32");
  ASSERT_NE(maximum, nullptr);
  maximum->setAttributes(llvm::Intrinsic::getAttributes(
      context, maximum->getIntrinsicID(), maximum->getFunctionType()));

  EXPECT_THAT(NormalizeMusaLlvmForLlvm14(*module, "minmax_call_attrs"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=floating-minmax-call")));
}

TEST(MusaLlvm14CompatibilityTest, RewritesCurrentFloatBitLiteralsForLlvm14) {
  const std::string text = Module(
      "  %low = fcmp ole float 0.0, f0xCF000000\n"
      "  %high = fcmp oge float 0.0, f0x4F000000\n"
      "  %both = and i1 %low, %high\n"
      "  store i1 %both, ptr addrspace(1) %out, align 1");
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(text, "float_literal_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("0xC1E0000000000000"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("0x41E0000000000000"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "f0x"));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest,
     RewritesCurrentDecimalFloatOutsideLlvm14ParserRange) {
  const std::string text = Module(
      "  %low = fcmp ole float -1.000000e+30, 0.000000e+00\n"
      "  store i1 %low, ptr addrspace(1) %out, align 1");
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(text, "decimal_float_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "-1.000000e+30"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("fcmp ole float 0x"));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest, RewritesCurrentSpecialFloatTokensForLlvm14) {
  const std::string text = Module(
      "  %positive = fcmp one float 1.0, +inf\n"
      "  %negative = fcmp one float 1.0, -inf\n"
      "  %result = and i1 %positive, %negative\n"
      "  store i1 %result, ptr addrspace(1) %out, align 1");
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(text, "special_float_profile");
  ASSERT_TRUE(normalized.ok()) << normalized.status();
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("0x7FF0000000000000"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("0xFFF0000000000000"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "+inf"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "-inf"));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest,
     RewritesEveryInexactDecimalFloatPhiIncomingValue) {
  const std::string text = Module(
      "  br i1 true, label %left, label %right\n"
      "left:\n"
      "  br label %merge\n"
      "right:\n"
      "  br label %merge\n"
      "merge:\n"
      "  %value = phi float [ -1.000000e+30, %left ], [ "
      "1.000000e+30, %right ]\n"
      "  %low = fcmp ole float %value, 0.000000e+00\n"
      "  store i1 %low, ptr addrspace(1) %out, align 1");
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(text, "decimal_float_phi_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "1.000000e+30"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("phi float [ 0x"));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest, DoesNotRewriteFloatPatternInModuleName) {
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(Module(""), "f0xDEADBEEF_name");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_THAT(normalized->normalized_llvm,
              HasSubstr("; ModuleID = 'f0xDEADBEEF_name'"));
  EXPECT_THAT(normalized->normalized_llvm,
              HasSubstr("source_filename = \"f0xDEADBEEF_name\""));
}

TEST(MusaLlvm14CompatibilityTest, LegalizesBfloatGlobalMemoryAsI16) {
  const std::string text = Module(
      "  %input = load bfloat, ptr addrspace(1) %out, align 2\n"
      "  %wide = fpext bfloat %input to float\n"
      "  %sum = fadd float %wide, 1.000000e+00\n"
      "  %narrow = fptrunc float %sum to bfloat\n"
      "  store bfloat %narrow, ptr addrspace(1) %out, align 2");
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(text, "bfloat_memory_profile");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("load i16"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("store i16"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("bf16_round_bias"));
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("bf16_is_nan"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "load bfloat"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "store bfloat"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "fpext bfloat"));
  EXPECT_FALSE(absl::StrContains(normalized->normalized_llvm, "to bfloat"));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest, CollectsSortedTypedExportedGlobals) {
  const std::string globals =
      "@z_mutable = protected addrspace(1) global i32 0, align 4\n"
      "@a_constant = protected addrspace(2) constant [2 x i64] "
      "[i64 1, i64 2], align 16\n";
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(
          Module("  store i32 7, ptr addrspace(1) %out, align 4", "", globals),
          "globals");
  ASSERT_THAT(normalized, IsOk());
  ASSERT_EQ(normalized->metadata.exported_globals.size(), 2);
  EXPECT_EQ(normalized->metadata.exported_globals[0].name, "a_constant");
  EXPECT_EQ(normalized->metadata.exported_globals[0].kind,
            MusaExportedGlobalKind::kConstant);
  EXPECT_EQ(normalized->metadata.exported_globals[0].address_space, 2);
  EXPECT_EQ(normalized->metadata.exported_globals[0].size, 16);
  EXPECT_EQ(normalized->metadata.exported_globals[1].name, "z_mutable");
  EXPECT_EQ(normalized->metadata.exported_globals[1].kind,
            MusaExportedGlobalKind::kMutable);
}

TEST(MusaLlvm14CompatibilityTest, AcceptsGlobalsOnlyConstantsModule) {
  const std::string input =
      absl::StrCat("source_filename = \"constants\"\n",
                   "target datalayout = \"", kMusaDataLayout, "\"\n",
                   "target triple = \"", kMusaTargetTriple, "\"\n\n",
                   "@constant_data = protected addrspace(1) global [4 x i32] "
                   "[i32 1, i32 2, i32 3, i32 4], align 16\n");
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(input, "constants_only");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_TRUE(normalized->metadata.kernel_entry_names.empty());
  ASSERT_EQ(normalized->metadata.exported_globals.size(), 1);
  EXPECT_EQ(normalized->metadata.exported_globals[0].name, "constant_data");
  EXPECT_EQ(normalized->metadata.exported_globals[0].kind,
            MusaExportedGlobalKind::kMutable);
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest, PreservesActiveMappingNoSignedZeros) {
  absl::StatusOr<MusaLlvm14CompatibilityResult> normalized =
      NormalizeMusaLlvmTextForLlvm14(
          Module("  %value = fadd nsz float 1.0, 2.0\n"
                 "  store float %value, ptr addrspace(1) %out, align 4"),
          "nsz");
  ASSERT_THAT(normalized, IsOk());
  EXPECT_THAT(normalized->normalized_llvm, HasSubstr("fadd nsz float"));
  EXPECT_THAT(
      ValidateMusaBridgeIr(normalized->normalized_llvm, normalized->metadata),
      IsOk());
}

TEST(MusaLlvm14CompatibilityTest, RejectsModuleWithoutKernelOrTypedGlobal) {
  const std::string input = absl::StrCat(
      "source_filename = \"empty\"\n", "target datalayout = \"",
      kMusaDataLayout, "\"\n", "target triple = \"", kMusaTargetTriple, "\"\n");
  EXPECT_THAT(NormalizeMusaLlvmTextForLlvm14(input, "empty"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=module-exports")));
}

TEST(MusaLlvm14CompatibilityTest, RejectsUnsupportedCurrentConstructs) {
  struct Rejection {
    std::string input;
    const char* capability;
  };
  const std::vector<Rejection> cases = {
      {absl::StrReplaceAll(Module(""),
                           {{kMusaTargetTriple, "nvptx64-nvidia-cuda"}}),
       "target-triple"},
      {Module("  call void asm sideeffect \"\", \"\"()"), "inline-assembly"},
      {Module("  call void @host_call()", "declare void @host_call()"),
       "unresolved-call"},
      {Module("  %value = fadd fast float 1.0, 2.0"), "fast-math-flags"},
      {Module("  %value = call float @llvm.fmuladd.f32("
              "float 2.0, float 3.0, float 4.0)",
              "declare float @llvm.fmuladd.f32(float, float, float)"),
       "intrinsic-compatibility"},
      {Module("  %value = load i32, ptr addrspace(1) %out, align 4, "
              "!range !0",
              "", "", "!0 = !{i32 0, i32 4}\n"),
       "legacy-bitonic-hints"},
  };
  for (const Rejection& rejection : cases) {
    EXPECT_THAT(
        NormalizeMusaLlvmTextForLlvm14(rejection.input,
                                       "unsupported_construct"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr(absl::StrCat("capability=", rejection.capability))));
  }
}

TEST(MusaLlvm14CompatibilityTest,
     TypedPointerCorpusIsUpgradedToOpaquePointers) {
  const std::string typed = absl::StrReplaceAll(
      Module("  store i32 0, ptr addrspace(1) %out, align 4"),
      {{"ptr addrspace(1) %out", "i32 addrspace(1)* %out"},
       {"ptr addrspace(1) %out, align 4", "i32 addrspace(1)* %out, align 4"}});
  absl::StatusOr<MusaLlvm14CompatibilityResult> result =
      NormalizeMusaLlvmTextForLlvm14(typed, "typed_pointer");
  ASSERT_THAT(result, IsOk());
  EXPECT_FALSE(absl::StrContains(result->normalized_llvm, "i32 addrspace"));
  EXPECT_THAT(ValidateMusaBridgeIr(result->normalized_llvm, result->metadata),
              IsOk());
}

TEST(MusaLlvm14CompatibilityTest, DiagnosticsDoNotEchoIrOrPaths) {
  constexpr absl::string_view kSensitive =
      "sensitive/current/llvm/source/token";
  absl::Status status =
      NormalizeMusaLlvmTextForLlvm14(kSensitive, "diagnostic").status();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=current-parse")));
  EXPECT_FALSE(absl::StrContains(status.message(), kSensitive));

  status = NormalizeMusaLlvmTextForLlvm14(Module(""), kSensitive).status();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=module-name")));
  EXPECT_FALSE(absl::StrContains(status.message(), kSensitive));
}

}  // namespace
}  // namespace xla::gpu::musa
