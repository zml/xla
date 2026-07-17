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

#include "xla/service/gpu/musa/musa_bridge_ir_validator.h"

#include <cstdint>
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
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/musa/protocol.pb.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

MusaBridgeIrMetadata Metadata(std::vector<std::string> kernels = {"kernel"}) {
  MusaBridgeIrMetadata metadata;
  metadata.module_name = "validator_test";
  metadata.kernel_entry_names = std::move(kernels);
  return metadata;
}

std::string Module(absl::string_view body, absl::string_view declarations = "",
                   absl::string_view globals = "") {
  return absl::StrCat("target datalayout = \"", kMusaDataLayout,
                      "\"\ntarget triple = \"", kMusaTargetTriple, "\"\n",
                      globals,
                      "\ndefine void @kernel(ptr "
                      "addrspace(1) %out) {\nentry:\n",
                      body, "\n  ret void\n}\n", declarations, "\n");
}

std::string InsertBeforeFirstFunction(std::string ir,
                                      absl::string_view declaration) {
  const size_t position = ir.find("\ndefine ");
  if (position == std::string::npos) return ir;
  ir.insert(position, absl::StrCat("\n", declaration));
  return ir;
}

std::string MinimalModule() {
  return Module(
      "  %tid = call i32 @__xla_musa_v1_read_tid_x()\n"
      "  store i32 %tid, ptr addrspace(1) %out, align 4",
      "declare i32 @__xla_musa_v1_read_tid_x() #0\n"
      "attributes #0 = { nounwind memory(none) }");
}

absl::Status Validate(absl::string_view ir,
                      MusaBridgeIrMetadata metadata = Metadata()) {
  return ValidateMusaBridgeIr(ir, metadata);
}

MusaBridgeCompileRequest RequestForIr(absl::string_view ir) {
  MusaBridgeCompileRequest request;
  request.set_protocol_version(kMusaBridgeProtocolVersion);
  request.set_shim_abi_version(kMusaShimAbiVersion);
  request.set_mapping_version(kMusaShimMappingVersion);
  request.set_mapping_fingerprint(kMusaShimMappingSha256);
  request.set_module_name("validator_test");
  request.set_normalized_llvm(ir);
  request.set_normalized_llvm_bytes(ir.size());
  request.set_normalized_llvm_sha256(MusaBridgeSha256Hex(ir));
  request.add_kernel_entry_names("kernel");
  request.add_exported_symbol_names("kernel");
  request.set_target_triple(kMusaTargetTriple);
  request.set_architecture(kMusaTargetArchitecture);
  request.set_data_layout(kMusaDataLayout);
  request.set_pointer_model(MUSA_BRIDGE_POINTER_MODEL_OPAQUE);
  request.set_pointer_width_bits(kMusaInterchangePointerWidth);
  request.set_byte_order(MUSA_BRIDGE_BYTE_ORDER_LITTLE_ENDIAN);
  request.mutable_numerical_flags();
  request.set_optimization_level(2);
  request.set_deterministic(true);
  request.set_xla_revision("ca08f32db3");
  request.set_current_llvm_revision("llvm-openxla");
  request.set_provider_name("bridge-contract-v1");
  request.set_provider_fingerprint(std::string(64, '1'));
  request.set_bridge_fingerprint(std::string(64, '2'));
  request.set_toolchain_fingerprint(std::string(64, '3'));
  return request;
}

TEST(MusaBridgeIrValidatorTest, AcceptsMinimalVersionedInterchange) {
  EXPECT_THAT(Validate(MinimalModule()), IsOk());
}

TEST(MusaBridgeIrValidatorTest, AcceptsSafeDashedKernelSymbols) {
  std::string ir =
      absl::StrReplaceAll(MinimalModule(), {{"@kernel", "@kernel-name"}});
  EXPECT_THAT(Validate(ir, Metadata({"kernel-name"})), IsOk());
}

TEST(MusaBridgeIrValidatorTest, VerifiesBeforeTraversalWithoutEchoingIr) {
  const std::string malformed =
      absl::StrCat("target datalayout = \"", kMusaDataLayout,
                   "\"\ntarget triple = \"", kMusaTargetTriple,
                   "\"\ndefine void @kernel(ptr addrspace(1) %out) {\n"
                   "entry:\n"
                   "  br i1 true, label %left, label %right\n"
                   "left:\n"
                   "  %sensitive_value = add i32 1, 2\n"
                   "  br label %right\n"
                   "right:\n"
                   "  store i32 %sensitive_value, ptr addrspace(1) %out\n"
                   "  ret void\n"
                   "}\n");
  absl::Status status = Validate(malformed);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=llvm-verifier")));
  EXPECT_FALSE(absl::StrContains(status.message(), "sensitive_value"));

  status = Validate("sensitive_parse_token\n");
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=llvm-parse")));
  EXPECT_FALSE(absl::StrContains(status.message(), "sensitive_parse_token"));
}

TEST(MusaBridgeIrValidatorTest, SanitizesParsedNamesInDiagnostics) {
  absl::Status status = Validate(
      Module("",
             "define internal void @\"sensitive/function/path\"() #1 {\n"
             "  ret void\n"
             "}\n"
             "attributes #1 = { nounwind }"));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=function-attributes")));
  EXPECT_FALSE(absl::StrContains(status.message(), "sensitive/function/path"));

  status = Validate(Module(
      "", "",
      "@\"sensitive/global/path\" = internal thread_local global i32 0"));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=thread-local-global")));
  EXPECT_FALSE(absl::StrContains(status.message(), "sensitive/global/path"));

  status = Validate(InsertBeforeFirstFunction(
      MinimalModule(),
      "!sensitive_named_metadata_token = !{!0}\n!0 = !{i32 7}\n"));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=named-metadata")));
  EXPECT_FALSE(
      absl::StrContains(status.message(), "sensitive_named_metadata_token"));

  status = Validate(absl::StrReplaceAll(
      MinimalModule(), {{kMusaTargetTriple, "sensitive/triple/path"}}));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=target-triple")));
  EXPECT_FALSE(absl::StrContains(status.message(), "sensitive/triple/path"));

  MusaBridgeIrMetadata metadata = Metadata();
  metadata.architecture = "sensitive/architecture/path";
  status = Validate(MinimalModule(), metadata);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=architecture")));
  EXPECT_FALSE(
      absl::StrContains(status.message(), "sensitive/architecture/path"));
}

TEST(MusaBridgeIrValidatorTest, ComposedRequestValidationBindsWireToIr) {
  MusaBridgeCompileRequest request = RequestForIr(MinimalModule());
  EXPECT_THAT(ValidateMusaBridgeCompileRequestIr(request), IsOk());

  request.set_module_name("1module+part");
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request), IsOk());
  EXPECT_THAT(ValidateMusaBridgeCompileRequestIr(request), IsOk());

  request.add_exported_symbol_names("unbound_symbol");
  EXPECT_THAT(ValidateMusaBridgeCompileRequestIr(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=exported-symbols")));
}

TEST(MusaBridgeIrValidatorTest, AcceptsEveryMappedShimWithExactSemantics) {
  std::string calls;
  std::string declarations;
  int result = 0;
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
        absl::StrAppend(&calls, "  %result", result++, " = call i32 @",
                        spec.xla_symbol, "()\n");
        break;
      case MusaShimSignature::kI64Void:
        return_type = "i64";
        absl::StrAppend(&calls, "  %result", result++, " = call i64 @",
                        spec.xla_symbol, "()\n");
        break;
      case MusaShimSignature::kI32I32I32:
        return_type = "i32";
        arguments = "i32 7, i32 0";
        absl::StrAppend(&calls, "  %result", result++, " = call i32 @",
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
        // No memory attribute means unrestricted read/write effects.
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
  EXPECT_THAT(Validate(Module(calls, declarations)), IsOk());
}

TEST(MusaBridgeIrValidatorTest, AcceptsTypedExportedGlobals) {
  MusaBridgeIrMetadata metadata = Metadata();
  metadata.exported_globals = {
      {"constant_data", MusaExportedGlobalKind::kConstant, 2, 16, 16},
      {"mutable_data", MusaExportedGlobalKind::kMutable, 1, 4, 4},
  };
  std::string ir =
      Module("  store i32 7, ptr addrspace(1) %out, align 4", "",
             "@constant_data = protected addrspace(2) constant [4 x i32] "
             "[i32 1, i32 2, i32 3, i32 4], align 16\n"
             "@mutable_data = protected addrspace(1) global i32 0, align 4\n"
             "@private_data = internal constant i32 9, align 4");
  EXPECT_THAT(Validate(ir, metadata), IsOk());
}

TEST(MusaBridgeIrValidatorTest, AcceptsGlobalsOnlyConstantsModule) {
  MusaBridgeIrMetadata metadata = Metadata({});
  metadata.exported_globals = {
      {"constant_data", MusaExportedGlobalKind::kMutable, 1, 16, 16}};
  const std::string ir =
      absl::StrCat("target datalayout = \"", kMusaDataLayout, "\"\n",
                   "target triple = \"", kMusaTargetTriple, "\"\n",
                   "@constant_data = protected addrspace(1) global [4 x i32] "
                   "[i32 1, i32 2, i32 3, i32 4], align 16\n");
  EXPECT_THAT(Validate(ir, metadata), IsOk());
}

TEST(MusaBridgeIrValidatorTest, RejectsModuleWithoutKernelOrTypedGlobal) {
  MusaBridgeIrMetadata metadata = Metadata({});
  const std::string ir =
      absl::StrCat("target datalayout = \"", kMusaDataLayout, "\"\n",
                   "target triple = \"", kMusaTargetTriple, "\"\n");
  EXPECT_THAT(Validate(ir, metadata),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=module-exports")));
}

TEST(MusaBridgeIrValidatorTest, RejectsContractAndTargetMismatch) {
  MusaBridgeIrMetadata metadata = Metadata();
  metadata.mapping_version = 1;
  EXPECT_THAT(Validate(MinimalModule(), metadata),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("version mismatch")));

  metadata = Metadata();
  metadata.architecture = "mp_99";
  EXPECT_THAT(Validate(MinimalModule(), metadata),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=architecture")));

  EXPECT_THAT(Validate(absl::StrReplaceAll(
                  MinimalModule(), {{"mtgpu-mt-musa", "nvptx64-nvidia-cuda"}})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=target-triple")));
  EXPECT_THAT(Validate(absl::StrReplaceAll(MinimalModule(),
                                           {{kMusaDataLayout, "e-p:64:64"}})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=data-layout")));
}

class ForeignTargetTest
    : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(ForeignTargetTest, RejectsForeignTargetFunction) {
  const auto& [name, signature] = GetParam();
  std::string ir =
      Module(absl::StrCat("  %value = call i32 @", name, "()"),
             absl::StrCat("declare i32 @", name, "() ", signature));
  EXPECT_THAT(Validate(ir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=foreign-target-construct")));
}

INSTANTIATE_TEST_SUITE_P(
    VendorFamilies, ForeignTargetTest,
    ::testing::Values(std::pair{"llvm.musa.read.ptx.sreg.tid.x", "nounwind"},
                      std::pair{"llvm.nvvm.read.ptx.sreg.tid.x", "nounwind"},
                      std::pair{"llvm.amdgcn.workitem.id.x", "nounwind"},
                      std::pair{"__ockl_get_local_id", "nounwind"}));

TEST(MusaBridgeIrValidatorTest, RejectsRawNativeCallingConvention) {
  std::string ir = absl::StrReplaceAll(
      MinimalModule(), {{"define void @kernel", "define cc 102 void @kernel"}});
  EXPECT_THAT(Validate(ir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=calling-convention")));
}

TEST(MusaBridgeIrValidatorTest, RejectsModuleAndCallInlineAssembly) {
  std::string module_asm = Module("  store i32 0, ptr addrspace(1) %out",
                                  "module asm \"bar.sync 0;\"");
  EXPECT_THAT(Validate(module_asm),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=inline-assembly")));

  EXPECT_THAT(Validate(Module("  call void asm sideeffect \"\", \"\"()")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=inline-assembly")));
}

TEST(MusaBridgeIrValidatorTest, RejectsOperandBundlesAndIndirectCalls) {
  EXPECT_THAT(Validate(Module("  call void @helper() [ \"deopt\"() ]",
                              "define internal void @helper() { ret void }")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=operand-bundle")));
  EXPECT_THAT(Validate(absl::StrCat(
                  "target datalayout = \"", kMusaDataLayout,
                  "\"\ntarget triple = \"", kMusaTargetTriple,
                  "\"\ndefine void @kernel(ptr %callee) {\nentry:\n  call void "
                  "%callee()\n  ret void\n}\n")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=indirect-call")));
}

TEST(MusaBridgeIrValidatorTest, RejectsAttributesAndMetadata) {
  EXPECT_THAT(
      Validate(absl::StrReplaceAll(
          MinimalModule(),
          {{"define void @kernel", "define void @kernel"},
           {"ptr addrspace(1) %out) {", "ptr addrspace(1) %out) #1 {"},
           {"attributes #0 =",
            "attributes #1 = { \"target-cpu\"=\"sm_80\" }\nattributes #0 ="}})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=function-attributes")));

  EXPECT_THAT(
      Validate(absl::StrReplaceAll(
          MinimalModule(), {{"ptr addrspace(1) %out)",
                             "ptr addrspace(1) byval(i32) align 4 %out)"}})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=function-attributes")));

  EXPECT_THAT(Validate(Module("  call void @helper() #1",
                              "define internal void @helper() { ret void }\n"
                              "attributes #1 = { nounwind }")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=call-site-attributes")));

  EXPECT_THAT(
      Validate(absl::StrCat(MinimalModule(),
                            "!musa.annotations = !{!0}\n!0 = !{i32 1}\n")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=named-metadata")));
  EXPECT_THAT(Validate(absl::StrCat(MinimalModule(),
                                    "!llvm.module.flags = !{!0}\n"
                                    "!0 = !{i32 1, !\"flag\", i32 1}\n")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=named-metadata")));

  EXPECT_THAT(
      Validate(absl::StrReplaceAll(
          MinimalModule(),
          {{"ptr addrspace(1) %out) {", "ptr addrspace(1) %out) !custom !0 {"},
           {"attributes #0 =", "!0 = !{i32 1}\nattributes #0 ="}})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=function-metadata")));
}

TEST(MusaBridgeIrValidatorTest, RejectsUnversionedObjectAndLinkerState) {
  EXPECT_THAT(Validate(InsertBeforeFirstFunction(MinimalModule(),
                                                 "$unused = comdat any\n")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=module-comdat")));

  EXPECT_THAT(Validate(absl::StrReplaceAll(
                  MinimalModule(),
                  {{"ptr addrspace(1) %out) {",
                    "ptr addrspace(1) %out) section \".text.injected\" {"}})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=function-object-state")));

  EXPECT_THAT(Validate(Module("", "",
                              "@hidden = internal global i32 0, section "
                              "\".data.injected\"")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=global-object-state")));

  EXPECT_THAT(Validate(absl::StrReplaceAll(
                  MinimalModule(),
                  {{"define void @kernel", "define weak void @kernel"}})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=kernel-list")));
}

TEST(MusaBridgeIrValidatorTest, RejectsAliasesAndIfuncs) {
  EXPECT_THAT(
      Validate(absl::StrCat(MinimalModule(),
                            "@kernel_alias = alias void (ptr addrspace(1)), "
                            "ptr @kernel\n")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=global-alias")));
}

TEST(MusaBridgeIrValidatorTest, RejectsUnknownWrongOrDefinedShim) {
  EXPECT_THAT(Validate(Module("  %v = call i32 @__xla_musa_v1_unknown()",
                              "declare i32 @__xla_musa_v1_unknown() nounwind "
                              "willreturn memory(none)")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=unknown-shim")));

  EXPECT_THAT(
      Validate(Module("  %v = call i64 @__xla_musa_v1_read_tid_x()",
                      "declare i64 @__xla_musa_v1_read_tid_x() nounwind "
                      "willreturn memory(none)")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=shim-signature")));

  EXPECT_THAT(Validate(Module(
                  "  %v = call i32 @__xla_musa_v1_read_tid_x()",
                  "define internal i32 @__xla_musa_v1_read_tid_x() nounwind "
                  "willreturn memory(none) { ret i32 0 }")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=shim-definition")));

  EXPECT_THAT(
      Validate(Module("  %v = call i32 @__xla_musa_v1_read_tid_x()",
                      "declare i32 @__xla_musa_v1_read_tid_x() nounwind "
                      "willreturn memory(none)")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=shim-attributes")));
}

TEST(MusaBridgeIrValidatorTest, RejectsShimAddressTaking) {
  std::string ir =
      Module("  %tid = call i32 @__xla_musa_v1_read_tid_x()",
             "declare i32 @__xla_musa_v1_read_tid_x() nounwind memory(none)",
             "@shim_pointer = internal constant ptr "
             "@__xla_musa_v1_read_tid_x");
  EXPECT_THAT(Validate(ir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=shim-address-taken")));
}

TEST(MusaBridgeIrValidatorTest, RejectsUnknownExternalAndAddressSpaceFour) {
  EXPECT_THAT(Validate(Module("  call void @host_function()",
                              "declare void @host_function()")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=unresolved-call")));
  EXPECT_THAT(Validate(absl::StrCat("target datalayout = \"", kMusaDataLayout,
                                    "\"\ntarget triple = \"", kMusaTargetTriple,
                                    "\"\ndefine void @kernel(ptr addrspace(4) "
                                    "%reserved) { ret void }\n")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=address-space")));

  EXPECT_THAT(
      Validate(absl::StrReplaceAll(
          MinimalModule(), {{"ptr addrspace(1) %out) {",
                             "ptr addrspace(1) %out) addrspace(1) {"}})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=function-address-space")));

  EXPECT_THAT(
      Validate(Module(
          "  store ptr addrspacecast (ptr addrspace(4) inttoptr (i32 0 to "
          "ptr addrspace(4)) to ptr), ptr addrspace(1) %out")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=address-space")));

  EXPECT_THAT(
      Validate(Module("", "", "@hidden = internal addrspace(4) global i32 0")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=address-space")));

  EXPECT_THAT(
      Validate(Module("", "",
                      "@hidden_gep = internal global ptr getelementptr ({ ptr "
                      "addrspace(4), i32 }, ptr null, i32 0, i32 1)")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=address-space")));

  EXPECT_THAT(
      Validate(InsertBeforeFirstFunction(
          absl::StrReplaceAll(
              MinimalModule(),
              {{"entry:\n", "entry:\n  %hidden = alloca %reserved_type\n"}}),
          "%reserved_type = type { ptr addrspace(4) }\n")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=address-space")));

  EXPECT_THAT(
      Validate(absl::StrReplaceAll(
          MinimalModule(),
          {{"entry:\n", "entry:\n  %hidden = alloca { ptr addrspace(4) }\n"}})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=address-space")));
}

TEST(MusaBridgeIrValidatorTest, AcceptsAddressSpaceFivePrivateScratch) {
  EXPECT_THAT(Validate(Module(
                  "  %scratch = alloca i32, addrspace(5)\n"
                  "  store i32 7, ptr addrspace(5) %scratch, align 4\n"
                  "  %value = load i32, ptr addrspace(5) %scratch, align 4\n"
                  "  store i32 %value, ptr addrspace(1) %out, align 4")),
              IsOk());
}

TEST(MusaBridgeIrValidatorTest, RejectsUnlistedGenericIntrinsicAndCallCc) {
  EXPECT_THAT(Validate(Module("  call void @llvm.sideeffect()",
                              "declare void @llvm.sideeffect()")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=llvm-intrinsic")));
  EXPECT_THAT(Validate(Module("  call fastcc void @helper()",
                              "define internal void @helper() { ret void }")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=calling-convention")));
}

TEST(MusaBridgeIrValidatorTest, AcceptsMappingV2NoSignedZeros) {
  EXPECT_THAT(
      Validate(Module("  %value = fadd nsz float 1.0, 2.0\n"
                      "  store float %value, ptr addrspace(1) %out, align 4")),
      IsOk());
}

TEST(MusaBridgeIrValidatorTest, RejectsUncontractedFloatingPointSemantics) {
  EXPECT_THAT(
      Validate(Module("  %value = fadd fast float 1.0, 2.0\n"
                      "  store float %value, ptr addrspace(1) %out, align 4")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=fast-math-flags")));
  EXPECT_THAT(
      Validate(Module("  %value = fadd contract float 1.0, 2.0\n"
                      "  store float %value, ptr addrspace(1) %out, align 4")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=fast-math-flags")));
  EXPECT_THAT(
      Validate(Module(
          "  %value = call float @llvm.fmuladd.f32(float 2.0, float 3.0, "
          "float 4.0)\n"
          "  store float %value, ptr addrspace(1) %out, align 4",
          "declare float @llvm.fmuladd.f32(float, float, float)")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=llvm-intrinsic")));
}

TEST(MusaBridgeIrValidatorTest, RejectsUnqualifiedAtomics) {
  EXPECT_THAT(
      Validate(Module("  %old = atomicrmw add ptr addrspace(1) %out, i32 1 "
                      "monotonic")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("capability=atomics")));
}

TEST(MusaBridgeIrValidatorTest, RejectsInvalidKernelList) {
  MusaBridgeIrMetadata metadata = Metadata({"kernel", "kernel"});
  EXPECT_THAT(Validate(MinimalModule(), metadata),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("sorted and unique")));

  metadata = Metadata({"missing"});
  EXPECT_THAT(Validate(MinimalModule(), metadata),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not listed as a kernel")));

  EXPECT_THAT(
      Validate(absl::StrReplaceAll(
          MinimalModule(), {{"define void @kernel", "define i32 @kernel"},
                            {"  ret void\n}", "  ret i32 0\n}"}})),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("return void")));
}

TEST(MusaBridgeIrValidatorTest, RejectsExportedGlobalMismatch) {
  MusaBridgeIrMetadata metadata = Metadata();
  metadata.exported_globals = {
      {"mutable_data", MusaExportedGlobalKind::kMutable, 1, 8, 4}};
  EXPECT_THAT(
      Validate(Module("", "",
                      "@mutable_data = protected addrspace(1) global i32 0, "
                      "align 4"),
               metadata),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("does not match kind/address-space/size/alignment")));
}

}  // namespace
}  // namespace xla::gpu::musa
