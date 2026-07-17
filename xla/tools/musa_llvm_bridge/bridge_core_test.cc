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

#include "xla/tools/musa_llvm_bridge/bridge_core.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/musa/protocol.pb.h"
#include "xla/tools/musa_llvm_bridge/mubin_validator.h"

namespace xla::gpu::musa::bridge {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

std::string TestdataPath(const std::string& name) {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  const char* test_workspace = std::getenv("TEST_WORKSPACE");
  EXPECT_NE(test_srcdir, nullptr);
  EXPECT_NE(test_workspace, nullptr);
  return absl::StrCat(test_srcdir == nullptr ? "" : test_srcdir, "/",
                      test_workspace == nullptr ? "" : test_workspace,
                      "/xla/tools/musa_llvm_bridge/testdata/", name);
}

std::string ReadTestdata(const std::string& name) {
  std::ifstream input(TestdataPath(name), std::ios::binary);
  EXPECT_TRUE(input.is_open()) << name;
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::string ReadCompatibilityCorpus(const std::string& name) {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  const char* test_workspace = std::getenv("TEST_WORKSPACE");
  EXPECT_NE(test_srcdir, nullptr);
  EXPECT_NE(test_workspace, nullptr);
  const std::string path = absl::StrCat(
      test_srcdir == nullptr ? "" : test_srcdir, "/",
      test_workspace == nullptr ? "" : test_workspace,
      "/xla/service/gpu/musa/testdata/llvm14_compatibility/", name);
  std::ifstream input(path, std::ios::binary);
  EXPECT_TRUE(input.is_open()) << name;
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

MusaBridgeCompileRequest RequestForIr(const std::string& ir,
                                      bool with_global = false) {
  MusaBridgeCompileRequest request;
  request.set_protocol_version(kMusaBridgeProtocolVersion);
  request.set_shim_abi_version(kMusaShimAbiVersion);
  request.set_mapping_version(kMusaShimMappingVersion);
  request.set_mapping_fingerprint(kMusaShimMappingSha256);
  request.set_module_name("vendor_bridge_test");
  request.set_normalized_llvm(ir);
  request.set_normalized_llvm_bytes(ir.size());
  request.set_normalized_llvm_sha256(MusaBridgeSha256Hex(ir));
  request.add_kernel_entry_names("kernel");
  request.add_exported_symbol_names("kernel");
  if (with_global) {
    request.add_exported_symbol_names("mutable_data");
    MusaBridgeExportedGlobal* global = request.add_exported_globals();
    global->set_name("mutable_data");
    global->set_kind(MUSA_BRIDGE_GLOBAL_KIND_MUTABLE);
    global->set_address_space(1);
    global->set_size_bytes(4);
    global->set_alignment_bytes(4);
  }
  request.set_target_triple(kMusaTargetTriple);
  request.set_architecture(kMusaTargetArchitecture);
  request.set_data_layout(kMusaDataLayout);
  request.set_pointer_model(MUSA_BRIDGE_POINTER_MODEL_OPAQUE);
  request.set_pointer_width_bits(kMusaInterchangePointerWidth);
  request.set_byte_order(MUSA_BRIDGE_BYTE_ORDER_LITTLE_ENDIAN);
  request.mutable_numerical_flags();
  request.set_optimization_level(2);
  request.set_deterministic(true);
  request.set_xla_revision("vendor-bridge-test");
  request.set_current_llvm_revision("current-llvm-test");
  request.set_provider_name("vendor-llvm14-test");
  request.set_provider_fingerprint(std::string(64, '1'));
  request.set_bridge_fingerprint(std::string(64, '2'));
  request.set_toolchain_fingerprint(std::string(64, '3'));
  return request;
}

void WriteU16(std::vector<uint8_t>& bytes, size_t offset, uint16_t value) {
  ASSERT_LE(offset + 2, bytes.size());
  bytes[offset] = value & 0xff;
  bytes[offset + 1] = (value >> 8) & 0xff;
}

void WriteU32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  ASSERT_LE(offset + 4, bytes.size());
  for (int i = 0; i < 4; ++i) {
    bytes[offset + i] = (value >> (8 * i)) & 0xff;
  }
}

void WriteU64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
  ASSERT_LE(offset + 8, bytes.size());
  for (int i = 0; i < 8; ++i) {
    bytes[offset + i] = (value >> (8 * i)) & 0xff;
  }
}

// Builds an SDK-independent ELF64 ET_DYN fixture with the measured MTGPU
// envelope and a minimal .dynsym. It contains no vendor code or SDK binary.
std::vector<uint8_t> SyntheticMubin() {
  constexpr size_t kProgramHeaders = 64;
  constexpr size_t kNote = 176;
  constexpr size_t kDynstr = 200;
  constexpr size_t kDynsym = 224;
  constexpr size_t kText = 296;
  constexpr size_t kData = 304;
  constexpr size_t kSectionHeaders = 312;
  constexpr size_t kSectionCount = 5;
  constexpr size_t kSize = kSectionHeaders + kSectionCount * 64;
  std::vector<uint8_t> bytes(kSize, 0);

  bytes[0] = 0x7f;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = 2;              // ELFCLASS64
  bytes[5] = 1;              // ELFDATA2LSB
  bytes[6] = 1;              // EV_CURRENT
  WriteU16(bytes, 16, 3);    // ET_DYN
  WriteU16(bytes, 18, 253);  // EM_MTGPU
  WriteU32(bytes, 20, 1);    // EV_CURRENT
  WriteU64(bytes, 32, kProgramHeaders);
  WriteU64(bytes, 40, kSectionHeaders);
  WriteU16(bytes, 52, 64);
  WriteU16(bytes, 54, 56);
  WriteU16(bytes, 56, 2);
  WriteU16(bytes, 58, 64);
  WriteU16(bytes, 60, kSectionCount);
  WriteU16(bytes, 62, 0);  // Section names intentionally omitted.

  // Executable PT_LOAD.
  WriteU32(bytes, kProgramHeaders + 0, 1);
  WriteU32(bytes, kProgramHeaders + 4, 5);
  WriteU64(bytes, kProgramHeaders + 8, 0);
  WriteU64(bytes, kProgramHeaders + 32, kSize);
  WriteU64(bytes, kProgramHeaders + 40, kSize);
  WriteU64(bytes, kProgramHeaders + 48, 8);

  // PT_NOTE with owner MTGPU\0 and measured note type 0x40.
  constexpr size_t kNoteProgramHeader = kProgramHeaders + 56;
  WriteU32(bytes, kNoteProgramHeader + 0, 4);
  WriteU32(bytes, kNoteProgramHeader + 4, 4);
  WriteU64(bytes, kNoteProgramHeader + 8, kNote);
  WriteU64(bytes, kNoteProgramHeader + 32, 20);
  WriteU64(bytes, kNoteProgramHeader + 40, 20);
  WriteU64(bytes, kNoteProgramHeader + 48, 4);
  WriteU32(bytes, kNote + 0, 6);
  WriteU32(bytes, kNote + 4, 0);
  WriteU32(bytes, kNote + 8, 0x40);
  const std::string note_owner("MTGPU\0", 6);
  std::copy(note_owner.begin(), note_owner.end(), bytes.begin() + kNote + 12);

  const std::string dynstr("\0kernel\0mutable_data\0", 21);
  std::copy(dynstr.begin(), dynstr.end(), bytes.begin() + kDynstr);

  // dynsym[1] is protected global function @kernel in section 3.
  constexpr size_t kKernelSymbol = kDynsym + 24;
  WriteU32(bytes, kKernelSymbol + 0, 1);
  bytes[kKernelSymbol + 4] = 0x12;  // STB_GLOBAL | STT_FUNC
  bytes[kKernelSymbol + 5] = 3;     // STV_PROTECTED
  WriteU16(bytes, kKernelSymbol + 6, 3);
  WriteU64(bytes, kKernelSymbol + 16, 8);

  // dynsym[2] is protected global object @mutable_data in section 4.
  constexpr size_t kGlobalSymbol = kDynsym + 48;
  WriteU32(bytes, kGlobalSymbol + 0, 8);
  bytes[kGlobalSymbol + 4] = 0x11;  // STB_GLOBAL | STT_OBJECT
  bytes[kGlobalSymbol + 5] = 3;     // STV_PROTECTED
  WriteU16(bytes, kGlobalSymbol + 6, 4);
  WriteU64(bytes, kGlobalSymbol + 16, 4);

  // Section 1: .dynstr (names are omitted from the section-name table).
  constexpr size_t kDynstrSection = kSectionHeaders + 64;
  WriteU32(bytes, kDynstrSection + 4, 3);  // SHT_STRTAB
  WriteU64(bytes, kDynstrSection + 8, 2);  // SHF_ALLOC
  WriteU64(bytes, kDynstrSection + 24, kDynstr);
  WriteU64(bytes, kDynstrSection + 32, dynstr.size());
  WriteU64(bytes, kDynstrSection + 48, 1);

  // Section 2: .dynsym, linked to section 1.
  constexpr size_t kDynsymSection = kSectionHeaders + 128;
  WriteU32(bytes, kDynsymSection + 4, 11);  // SHT_DYNSYM
  WriteU64(bytes, kDynsymSection + 8, 2);   // SHF_ALLOC
  WriteU64(bytes, kDynsymSection + 24, kDynsym);
  WriteU64(bytes, kDynsymSection + 32, 72);
  WriteU32(bytes, kDynsymSection + 40, 1);
  WriteU32(bytes, kDynsymSection + 44, 1);
  WriteU64(bytes, kDynsymSection + 48, 8);
  WriteU64(bytes, kDynsymSection + 56, 24);

  // Section 3: executable code placeholder.
  constexpr size_t kTextSection = kSectionHeaders + 192;
  WriteU32(bytes, kTextSection + 4, 1);  // SHT_PROGBITS
  WriteU64(bytes, kTextSection + 8, 6);  // SHF_ALLOC | SHF_EXECINSTR
  WriteU64(bytes, kTextSection + 24, kText);
  WriteU64(bytes, kTextSection + 32, 8);
  WriteU64(bytes, kTextSection + 48, 4);

  // Section 4: mutable object placeholder.
  constexpr size_t kDataSection = kSectionHeaders + 256;
  WriteU32(bytes, kDataSection + 4, 1);  // SHT_PROGBITS
  WriteU64(bytes, kDataSection + 8, 3);  // SHF_WRITE | SHF_ALLOC
  WriteU64(bytes, kDataSection + 24, kData);
  WriteU64(bytes, kDataSection + 32, 4);
  WriteU64(bytes, kDataSection + 48, 4);
  return bytes;
}

TEST(MusaLlvmBridgeCoreTest, TranslatesMinimalModuleStructurally) {
  MusaBridgeCompileRequest request = RequestForIr(ReadTestdata("minimal.ll"));
  absl::StatusOr<VendorLlvmModule> translated =
      TranslateMusaBridgeRequestToVendorLlvm(request);
  ASSERT_THAT(translated, IsOk());
  EXPECT_EQ(translated->translated_shim_calls, 1);
  EXPECT_EQ(translated->kernel_count, 1);
  EXPECT_THAT(translated->llvm_ir,
              HasSubstr("define protected mtgpu_kernel void @kernel"));
  EXPECT_THAT(translated->llvm_ir, HasSubstr("@llvm.musa.read.ptx.sreg.tid.x"));
  EXPECT_THAT(translated->llvm_ir, HasSubstr("!musa.annotations = !{"));
  EXPECT_FALSE(absl::StrContains(translated->llvm_ir, "__xla_musa_"));
}

TEST(MusaLlvmBridgeCoreTest,
     ParsesAndTranslatesCurrentLlvmCompatibilityGolden) {
  MusaBridgeCompileRequest request =
      RequestForIr(ReadCompatibilityCorpus("elemental.llvm14.ll"));
  absl::StatusOr<VendorLlvmModule> translated =
      TranslateMusaBridgeRequestToVendorLlvm(request);
  ASSERT_THAT(translated, IsOk());
  EXPECT_EQ(translated->translated_shim_calls, 1);
  EXPECT_EQ(translated->kernel_count, 1);
  EXPECT_THAT(translated->llvm_ir,
              HasSubstr("define protected mtgpu_kernel void @kernel"));
  EXPECT_THAT(translated->llvm_ir, HasSubstr("@llvm.musa.read.ptx.sreg.tid.x"));
}

TEST(MusaLlvmBridgeCoreTest, ParsesAndTranslatesSqrtCompatibilityGolden) {
  MusaBridgeCompileRequest request =
      RequestForIr(ReadCompatibilityCorpus("sqrt.llvm14.ll"));
  absl::StatusOr<VendorLlvmModule> translated =
      TranslateMusaBridgeRequestToVendorLlvm(request);
  ASSERT_THAT(translated, IsOk());
  EXPECT_EQ(translated->translated_shim_calls, 0);
  EXPECT_EQ(translated->kernel_count, 1);
  EXPECT_THAT(translated->llvm_ir,
              HasSubstr("declare float @llvm.sqrt.f32(float)"));
  EXPECT_THAT(translated->llvm_ir,
              HasSubstr("define protected mtgpu_kernel void @kernel"));
}

TEST(MusaLlvmBridgeCoreTest, TranslatesEveryMappingV1Shim) {
  MusaBridgeCompileRequest request = RequestForIr(ReadTestdata("all_shims.ll"));
  absl::StatusOr<VendorLlvmModule> translated =
      TranslateMusaBridgeRequestToVendorLlvm(request);
  ASSERT_THAT(translated, IsOk());
  EXPECT_EQ(translated->translated_shim_calls, MusaShimSpecs().size());
  for (const MusaShimSpec& spec : MusaShimSpecs()) {
    EXPECT_THAT(translated->llvm_ir, HasSubstr(spec.vendor_intrinsic));
  }
  EXPECT_FALSE(absl::StrContains(translated->llvm_ir, "__xla_musa_"));
}

TEST(MusaLlvmBridgeCoreTest, InstallsProtectedAbiForExportedGlobals) {
  std::string ir = ReadTestdata("minimal.ll");
  const size_t definition = ir.find("define void @kernel");
  ASSERT_NE(definition, std::string::npos);
  ir.insert(definition,
            "@mutable_data = addrspace(1) global i32 0, align 4\n\n");
  MusaBridgeCompileRequest request = RequestForIr(ir, /*with_global=*/true);
  absl::StatusOr<VendorLlvmModule> translated =
      TranslateMusaBridgeRequestToVendorLlvm(request);
  ASSERT_THAT(translated, IsOk());
  EXPECT_THAT(translated->llvm_ir,
              HasSubstr("@mutable_data = protected addrspace(1) global i32 0"));
}

TEST(MusaLlvmBridgeCoreTest, TranslationIsDeterministic) {
  MusaBridgeCompileRequest request = RequestForIr(ReadTestdata("minimal.ll"));
  absl::StatusOr<VendorLlvmModule> first =
      TranslateMusaBridgeRequestToVendorLlvm(request);
  absl::StatusOr<VendorLlvmModule> second =
      TranslateMusaBridgeRequestToVendorLlvm(request);
  ASSERT_THAT(first, IsOk());
  ASSERT_THAT(second, IsOk());
  EXPECT_EQ(first->llvm_ir, second->llvm_ir);
}

TEST(MusaLlvmBridgeCoreTest, RejectsCapabilitiesOutsideMappingV1) {
  struct Rejection {
    const char* fixture;
    const char* capability;
  };
  for (const Rejection& rejection : {
           Rejection{"atomic.ll", "capability=atomics"},
           Rejection{"unknown_shim.ll", "capability=unknown-shim"},
           Rejection{"reserved_as4.ll", "capability=address-space"},
           Rejection{"raw_vendor_intrinsic.ll",
                     "capability=foreign-target-construct"},
       }) {
    MusaBridgeCompileRequest request =
        RequestForIr(ReadTestdata(rejection.fixture));
    EXPECT_THAT(TranslateMusaBridgeRequestToVendorLlvm(request),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr(rejection.capability)))
        << rejection.fixture;
  }
}

TEST(MusaLlvmBridgeCoreTest, RejectsUnversionedFloatingPointSemantics) {
  constexpr absl::string_view kPrefix =
      "target datalayout = \"e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-"
      "p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-"
      "v96:128\"\n"
      "target triple = \"mtgpu-mt-musa\"\n";
  for (absl::string_view flag :
       {absl::string_view("fast"), absl::string_view("contract")}) {
    const std::string ir =
        absl::StrCat(kPrefix, "define void @kernel(ptr addrspace(1) %out) {\n",
                     "  %value = fadd ", flag, " float 1.0, 2.0\n",
                     "  store float %value, ptr addrspace(1) %out, align 4\n",
                     "  ret void\n}\n");
    MusaBridgeCompileRequest request = RequestForIr(ir);
    EXPECT_THAT(TranslateMusaBridgeRequestToVendorLlvm(request),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("capability=fast-math-flags")));
  }

  const std::string fmuladd_ir = absl::StrCat(
      kPrefix,
      "define void @kernel(ptr addrspace(1) %out) {\n"
      "  %value = call float @llvm.fmuladd.f32(float 2.0, float 3.0, float "
      "4.0)\n"
      "  store float %value, ptr addrspace(1) %out, align 4\n"
      "  ret void\n"
      "}\n"
      "declare float @llvm.fmuladd.f32(float, float, float)\n");
  MusaBridgeCompileRequest request = RequestForIr(fmuladd_ir);
  EXPECT_THAT(TranslateMusaBridgeRequestToVendorLlvm(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("capability=llvm-intrinsic")));
}

TEST(MusaLlvmBridgeCoreTest, ParserFailureDoesNotEchoUntrustedIr) {
  const std::string ir = "sensitive_vendor_bridge_token\n";
  MusaBridgeCompileRequest request = RequestForIr(ir);
  absl::Status status =
      TranslateMusaBridgeRequestToVendorLlvm(request).status();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("capability=llvm14-parse")));
  EXPECT_FALSE(
      absl::StrContains(status.message(), "sensitive_vendor_bridge_token"));
}

TEST(MusaLlvmBridgeCoreTest, ValidatesMubinEnvelopeAndRequestedSymbols) {
  MusaBridgeCompileRequest request =
      RequestForIr(ReadTestdata("minimal.ll"), /*with_global=*/true);
  EXPECT_THAT(ValidateMubinOutput(SyntheticMubin(), request), IsOk());
}

TEST(MusaLlvmBridgeCoreTest, RejectsMubinSymbolContractViolations) {
  MusaBridgeCompileRequest request =
      RequestForIr(ReadTestdata("minimal.ll"), /*with_global=*/true);

  std::vector<uint8_t> missing_kernel = SyntheticMubin();
  WriteU32(missing_kernel, 224 + 24, 0);
  EXPECT_THAT(ValidateMubinOutput(missing_kernel, request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("kernel is absent")));

  std::vector<uint8_t> wrong_kernel_type = SyntheticMubin();
  wrong_kernel_type[224 + 24 + 4] = 0x11;
  EXPECT_THAT(ValidateMubinOutput(wrong_kernel_type, request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("wrong ELF symbol type")));

  std::vector<uint8_t> wrong_global_size = SyntheticMubin();
  WriteU64(wrong_global_size, 224 + 48 + 16, 8);
  EXPECT_THAT(ValidateMubinOutput(wrong_global_size, request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("wrong ELF size")));

  std::vector<uint8_t> wrong_envelope = SyntheticMubin();
  wrong_envelope[0] = 0;
  EXPECT_THAT(
      ValidateMubinOutput(wrong_envelope, request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("ELF magic")));
}

}  // namespace
}  // namespace xla::gpu::musa::bridge
