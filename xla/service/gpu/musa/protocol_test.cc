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

#include "xla/service/gpu/musa/protocol.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/unknown_field_set.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/service/gpu/musa/protocol.pb.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr absl::string_view kNormalizedLlvm =
    "; ModuleID = 'add_module'\n"
    "source_filename = \"add_module\"\n"
    "target datalayout = \"e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:"
    "64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-"
    "v96:128\"\n"
    "target triple = \"mtgpu-mt-musa\"\n"
    "define void @add(ptr %out) {\n"
    "  ret void\n"
    "}\n";

void SetNormalizedLlvm(MusaBridgeCompileRequest* request,
                       absl::string_view llvm_ir) {
  request->set_normalized_llvm(llvm_ir);
  request->set_normalized_llvm_bytes(llvm_ir.size());
  request->set_normalized_llvm_sha256(MusaBridgeSha256Hex(llvm_ir));
}

MusaBridgeCompileRequest MakeValidRequest() {
  MusaBridgeCompileRequest request;
  request.set_protocol_version(kMusaBridgeProtocolVersion);
  request.set_shim_abi_version(kMusaShimAbiVersion);
  request.set_mapping_version(kMusaShimMappingVersion);
  request.set_module_name("add_module");
  SetNormalizedLlvm(&request, kNormalizedLlvm);
  request.add_kernel_entry_names("add");
  request.add_exported_symbol_names("add");
  request.add_exported_symbol_names("bias");
  MusaBridgeExportedGlobal* global = request.add_exported_globals();
  global->set_name("bias");
  global->set_kind(MUSA_BRIDGE_GLOBAL_KIND_CONSTANT);
  global->set_address_space(2);
  global->set_size_bytes(4);
  global->set_alignment_bytes(4);
  request.set_target_triple(kMusaTargetTriple);
  request.set_architecture(kMusaTargetArchitecture);
  request.set_data_layout(kMusaDataLayout);
  request.set_pointer_model(MUSA_BRIDGE_POINTER_MODEL_OPAQUE);
  request.set_pointer_width_bits(kMusaInterchangePointerWidth);
  request.set_byte_order(MUSA_BRIDGE_BYTE_ORDER_LITTLE_ENDIAN);
  MusaBridgeNumericalFlags* numerical_flags = request.mutable_numerical_flags();
  numerical_flags->set_allow_fp_contract(true);
  request.set_optimization_level(2);
  request.set_emit_debug_information(false);
  request.set_deterministic(true);
  request.set_xla_revision("ed8f8caf84");
  request.set_current_llvm_revision("llvm-18-openxla");
  request.set_provider_name("subprocess-v1");
  request.set_provider_fingerprint(std::string(64, '1'));
  request.set_bridge_fingerprint(std::string(64, '2'));
  request.set_toolchain_fingerprint(std::string(64, '3'));
  request.set_mapping_fingerprint(kMusaShimMappingSha256);
  return request;
}

MusaBridgeCompileResponse MakeValidResponse(
    const MusaBridgeCompileRequest& request) {
  MusaBridgeCompileResponse response;
  response.set_protocol_version(kMusaBridgeProtocolVersion);
  response.set_shim_abi_version(kMusaShimAbiVersion);
  response.set_mapping_version(kMusaShimMappingVersion);
  response.set_status(MUSA_BRIDGE_STATUS_OK);
  response.set_request_sha256(MusaBridgeCompileRequestSha256(request).value());
  response.set_provider_name(request.provider_name());
  response.set_provider_fingerprint(request.provider_fingerprint());
  response.set_bridge_fingerprint(request.bridge_fingerprint());
  response.set_toolchain_fingerprint(request.toolchain_fingerprint());
  response.set_mapping_fingerprint(request.mapping_fingerprint());
  MusaBridgeDiagnostic* diagnostic = response.add_diagnostics();
  diagnostic->set_severity(MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_NOTE);
  diagnostic->set_code("compile-complete");
  diagnostic->set_message("compiled one kernel");
  diagnostic->set_component("musa-llvm-bridge");
  diagnostic->set_line(1);
  diagnostic->set_column(1);
  diagnostic->set_symbol_name("add");
  MusaBridgeCompileStats* stats = response.mutable_stats();
  stats->set_input_llvm_bytes(request.normalized_llvm_bytes());
  stats->set_kernel_count(request.kernel_entry_names_size());
  stats->set_exported_symbol_count(request.exported_symbol_names_size());
  stats->set_diagnostic_count(response.diagnostics_size());
  stats->set_bridge_wall_time_microseconds(1200);
  stats->set_peak_memory_bytes(4 * 1024 * 1024);
  const std::string mubin = {'\x7f', 'E', 'L', 'F', '\0',
                             'M',    'U', 'B', 'I', 'N'};
  response.set_mubin(mubin);
  response.set_mubin_sha256(MusaBridgeSha256Hex(mubin));
  stats->set_output_mubin_bytes(mubin.size());
  return response;
}

MusaBridgeCompileResponse MakeFailureResponse(
    const MusaBridgeCompileRequest& request) {
  MusaBridgeCompileResponse response = MakeValidResponse(request);
  response.set_status(MUSA_BRIDGE_STATUS_REJECTED);
  response.clear_diagnostics();
  MusaBridgeDiagnostic* diagnostic = response.add_diagnostics();
  diagnostic->set_severity(MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_ERROR);
  diagnostic->set_code("unknown-shim");
  diagnostic->set_message("the module references an unknown shim");
  diagnostic->set_component("shim-validator");
  diagnostic->set_symbol_name("add");
  response.mutable_stats()->set_diagnostic_count(1);
  response.mutable_stats()->set_output_mubin_bytes(0);
  response.clear_mubin();
  response.clear_mubin_sha256();
  return response;
}

TEST(MusaBridgeProtocolTest, Sha256UsesLowerCaseUnprefixedHex) {
  EXPECT_EQ(MusaBridgeSha256Hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(MusaBridgeProtocolTest, RequestCanonicalRoundTripIsStable) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  absl::StatusOr<std::string> first = EncodeMusaBridgeCompileRequest(request);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_TRUE(absl::StartsWith(*first, kMusaBridgeRequestMagic));
  absl::StatusOr<MusaBridgeCompileRequest> decoded =
      DecodeMusaBridgeCompileRequest(*first);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->SerializeAsString(), request.SerializeAsString());
  EXPECT_EQ(EncodeMusaBridgeCompileRequest(*decoded), first);
  EXPECT_EQ(MusaBridgeCompileRequestSha256(*decoded),
            MusaBridgeCompileRequestSha256(request));
  EXPECT_EQ(MusaBridgeSha256Hex(*first),
            "06f40056f9ba836960dfb2ffd3054df68b88bf555d9e8fd33d4a7d13414a969d");
}

TEST(MusaBridgeProtocolTest, ResponseCanonicalRoundTripPreservesBinaryMubin) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  MusaBridgeCompileResponse response = MakeValidResponse(request);
  absl::StatusOr<std::string> wire = EncodeMusaBridgeCompileResponse(response);
  ASSERT_TRUE(wire.ok()) << wire.status();
  EXPECT_TRUE(absl::StartsWith(*wire, kMusaBridgeResponseMagic));
  absl::StatusOr<MusaBridgeCompileResponse> decoded =
      DecodeMusaBridgeCompileResponse(*wire);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->SerializeAsString(), response.SerializeAsString());
  EXPECT_EQ(decoded->mubin(), response.mubin());
  EXPECT_EQ(EncodeMusaBridgeCompileResponse(*decoded), wire);
  EXPECT_EQ(MusaBridgeSha256Hex(*wire),
            "06482c4e81cb8ccda8bc49c6b698de155a09c3d61df24505ccecfa6c3cea6bb8");
}

TEST(MusaBridgeProtocolTest, WireBoundsCoverWorstCaseTextExpansion) {
  EXPECT_GE(kMusaBridgeMaxRequestWireBytes, 4 * kMusaBridgeMaxLlvmBytes);
  EXPECT_GE(kMusaBridgeMaxResponseWireBytes,
            4 * kMusaBridgeMaxMubinBytes + 4 * kMusaBridgeMaxDiagnosticBytes);
}

TEST(MusaBridgeProtocolTest, CompleteExchangeValidationSucceeds) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  EXPECT_TRUE(
      ValidateMusaBridgeExchange(request, MakeValidResponse(request)).ok());
  EXPECT_TRUE(
      ValidateMusaBridgeExchange(request, MakeFailureResponse(request)).ok());
}

TEST(MusaBridgeProtocolTest, EveryVersionMustMatchExactly) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  request.set_protocol_version(kMusaBridgeProtocolVersion + 1);
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("protocol_version")));
  request = MakeValidRequest();
  request.set_shim_abi_version(kMusaShimAbiVersion + 1);
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("shim_abi_version")));
  request = MakeValidRequest();
  request.set_mapping_version(kMusaShimMappingVersion + 1);
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("mapping_version")));

  MusaBridgeCompileResponse response = MakeValidResponse(MakeValidRequest());
  response.set_mapping_version(kMusaShimMappingVersion - 1);
  EXPECT_THAT(ValidateMusaBridgeCompileResponse(response),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("mapping_version")));
}

TEST(MusaBridgeProtocolTest,
     DecoderRejectsWrongMagicMalformedAndNoncanonicalText) {
  absl::StatusOr<std::string> wire =
      EncodeMusaBridgeCompileRequest(MakeValidRequest());
  ASSERT_TRUE(wire.ok());
  EXPECT_THAT(DecodeMusaBridgeCompileRequest("protocol_version: 1\n"),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("magic")));
  EXPECT_THAT(DecodeMusaBridgeCompileRequest(
                  absl::StrCat(kMusaBridgeRequestMagic, "not a proto\n")),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("parse")));
  std::string noncanonical = *wire;
  noncanonical.insert(kMusaBridgeRequestMagic.size(), "\n");
  EXPECT_THAT(
      DecodeMusaBridgeCompileRequest(noncanonical),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("canonical")));
}

TEST(MusaBridgeProtocolTest, ClosedSchemaRejectsProcessControlInjection) {
  absl::StatusOr<std::string> wire =
      EncodeMusaBridgeCompileRequest(MakeValidRequest());
  ASSERT_TRUE(wire.ok());
  for (absl::string_view injected_field :
       {"working_directory: \"/tmp\"\n", "environment: \"LD_PRELOAD=x\"\n",
        "arguments: \"--arbitrary-flag\"\n",
        "output_filename: \"out.mubin\"\n"}) {
    EXPECT_THAT(
        DecodeMusaBridgeCompileRequest(absl::StrCat(*wire, injected_field)),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("parse")));
  }
}

TEST(MusaBridgeProtocolTest, ProgrammaticUnknownFieldsAlsoFailClosed) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  request.GetReflection()->MutableUnknownFields(&request)->AddVarint(999, 1);
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unknown")));

  request = MakeValidRequest();
  MusaBridgeExportedGlobal* global = request.mutable_exported_globals(0);
  global->GetReflection()->MutableUnknownFields(global)->AddVarint(999, 1);
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unknown")));

  request = MakeValidRequest();
  MusaBridgeCompileResponse response = MakeValidResponse(request);
  response.GetReflection()->MutableUnknownFields(&response)->AddVarint(999, 1);
  EXPECT_THAT(
      ValidateMusaBridgeCompileResponse(response),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unknown")));
}

TEST(MusaBridgeProtocolTest, NormalizedLlvmSizeDigestAndControlsAreChecked) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  request.set_normalized_llvm_bytes(request.normalized_llvm_bytes() + 1);
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("normalized_llvm_bytes")));

  request = MakeValidRequest();
  request.set_normalized_llvm_sha256(std::string(64, '0'));
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match")));

  request = MakeValidRequest();
  SetNormalizedLlvm(&request, "define void @add() {\r\n}\n");
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("control")));

  request = MakeValidRequest();
  SetNormalizedLlvm(&request, "define void @add() { \n}\n");
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("trailing")));

  request = MakeValidRequest();
  std::string nul_injected = "define void @add() {\n}";
  nul_injected.push_back('\0');
  nul_injected.push_back('\n');
  SetNormalizedLlvm(&request, nul_injected);
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("control")));
}

TEST(MusaBridgeProtocolTest, OversizeIrAndCountsAreRejectedBeforeParsing) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  std::string oversized_ir(kMusaBridgeMaxLlvmBytes, 'a');
  oversized_ir.push_back('\n');
  SetNormalizedLlvm(&request, oversized_ir);
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("too long")));

  request = MakeValidRequest();
  request.clear_kernel_entry_names();
  for (size_t i = 0; i <= kMusaBridgeMaxKernelCount; ++i) {
    request.add_kernel_entry_names(absl::StrCat("k", i));
  }
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("too many")));

  MusaBridgeCompileResponse response = MakeValidResponse(MakeValidRequest());
  response.mutable_diagnostics(0)->set_message(
      std::string(kMusaBridgeMaxDiagnosticMessageBytes + 1, 'x'));
  EXPECT_THAT(ValidateMusaBridgeCompileResponse(response),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("diagnostics.message")));

  response = MakeValidResponse(MakeValidRequest());
  response.clear_diagnostics();
  for (size_t i = 0; i <= kMusaBridgeMaxDiagnosticBytes /
                              kMusaBridgeMaxDiagnosticMessageBytes;
       ++i) {
    MusaBridgeDiagnostic* diagnostic = response.add_diagnostics();
    diagnostic->set_severity(MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_NOTE);
    diagnostic->set_code("bounded-note");
    diagnostic->set_message(
        std::string(kMusaBridgeMaxDiagnosticMessageBytes, 'x'));
    diagnostic->set_component("protocol-test");
  }
  response.mutable_stats()->set_diagnostic_count(response.diagnostics_size());
  EXPECT_THAT(ValidateMusaBridgeCompileResponse(response),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("aggregate text")));
}

TEST(MusaBridgeProtocolTest, TargetContractIsExact) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  request.set_target_triple("nvptx64-nvidia-cuda");
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("target_triple")));
  request = MakeValidRequest();
  request.set_architecture("mp_31");
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("architecture")));
  request = MakeValidRequest();
  request.set_data_layout("e-p:64:64");
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("data_layout")));
  request = MakeValidRequest();
  request.set_pointer_width_bits(32);
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pointer_width_bits")));
  request = MakeValidRequest();
  request.set_byte_order(MUSA_BRIDGE_BYTE_ORDER_UNSPECIFIED);
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("byte_order")));
}

TEST(MusaBridgeProtocolTest, TokensFingerprintsAndListsAreCanonical) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  request.set_provider_name("../../bridge");
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("provider_name")));
  request = MakeValidRequest();
  request.set_toolchain_fingerprint(std::string(64, 'A'));
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("toolchain_fingerprint")));
  request = MakeValidRequest();
  request.set_mapping_fingerprint(std::string(64, '4'));
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("mapping_fingerprint")));
  request = MakeValidRequest();
  request.add_kernel_entry_names("add");
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("sorted")));
  request = MakeValidRequest();
  request.add_target_features("+aaa");
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("target_features")));
}

TEST(MusaBridgeProtocolTest, ExportedGlobalMetadataIsTypedAndBounded) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  request.mutable_exported_globals(0)->set_kind(
      MUSA_BRIDGE_GLOBAL_KIND_UNSPECIFIED);
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("kind")));
  request = MakeValidRequest();
  request.mutable_exported_globals(0)->set_address_space(6);
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("address_space")));
  request = MakeValidRequest();
  request.mutable_exported_globals(0)->set_address_space(1);
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("global kind")));
  request = MakeValidRequest();
  request.mutable_exported_globals(0)->set_alignment_bytes(3);
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("alignment")));
  request = MakeValidRequest();
  request.mutable_exported_globals(0)->set_name("missing");
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exported_symbol_names")));
}

TEST(MusaBridgeProtocolTest, UnknownRequestEnumsFailClosed) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  request.set_pointer_model(static_cast<MusaBridgePointerModel>(99));
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("pointer_model")));
  request = MakeValidRequest();
  request.set_byte_order(static_cast<MusaBridgeByteOrder>(99));
  EXPECT_THAT(
      ValidateMusaBridgeCompileRequest(request),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("byte_order")));
  request = MakeValidRequest();
  request.mutable_exported_globals(0)->set_kind(
      static_cast<MusaBridgeGlobalKind>(99));
  EXPECT_THAT(ValidateMusaBridgeCompileRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("kind")));
}

TEST(MusaBridgeProtocolTest,
     ResponseStatusAndDiagnosticsCombinationsFailClosed) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  MusaBridgeCompileResponse response = MakeValidResponse(request);
  response.set_status(static_cast<MusaBridgeStatus>(99));
  EXPECT_THAT(
      ValidateMusaBridgeCompileResponse(response),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("status")));

  response = MakeValidResponse(request);
  response.mutable_diagnostics(0)->set_severity(
      MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_THAT(
      ValidateMusaBridgeCompileResponse(response),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("OK response")));

  response = MakeFailureResponse(request);
  response.mutable_diagnostics(0)->set_severity(
      MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_WARNING);
  EXPECT_THAT(ValidateMusaBridgeCompileResponse(response),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("requires an error")));

  response = MakeFailureResponse(request);
  response.set_mubin("not-allowed");
  EXPECT_THAT(ValidateMusaBridgeCompileResponse(response),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be absent")));

  response = MakeValidResponse(request);
  response.mutable_diagnostics(0)->set_severity(
      static_cast<MusaBridgeDiagnosticSeverity>(99));
  EXPECT_THAT(
      ValidateMusaBridgeCompileResponse(response),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("severity")));
}

TEST(MusaBridgeProtocolTest, MubinSizeAndDigestMustMatchActualBytes) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  MusaBridgeCompileResponse response = MakeValidResponse(request);
  response.mutable_stats()->set_output_mubin_bytes(response.mubin().size() + 1);
  EXPECT_THAT(ValidateMusaBridgeCompileResponse(response),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("output_mubin_bytes")));
  response = MakeValidResponse(request);
  response.set_mubin_sha256(std::string(64, '0'));
  EXPECT_THAT(ValidateMusaBridgeCompileResponse(response),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match MUBIN")));
}

TEST(MusaBridgeProtocolTest, ExchangeBindsRequestIdentityAndStatistics) {
  MusaBridgeCompileRequest request = MakeValidRequest();
  MusaBridgeCompileResponse response = MakeValidResponse(request);
  response.set_request_sha256(std::string(64, '0'));
  EXPECT_THAT(ValidateMusaBridgeExchange(request, response),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("does not bind")));

  response = MakeValidResponse(request);
  response.set_bridge_fingerprint(std::string(64, '4'));
  EXPECT_THAT(
      ValidateMusaBridgeExchange(request, response),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("identity")));

  response = MakeValidResponse(request);
  response.mutable_stats()->set_kernel_count(2);
  EXPECT_THAT(
      ValidateMusaBridgeExchange(request, response),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("statistics")));

  response = MakeValidResponse(request);
  response.mutable_diagnostics(0)->set_symbol_name("unknown_symbol");
  EXPECT_THAT(
      ValidateMusaBridgeExchange(request, response),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("outside")));
}

}  // namespace
}  // namespace xla::gpu::musa
