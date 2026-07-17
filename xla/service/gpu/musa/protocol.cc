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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/unknown_field_set.h"
#include <openssl/sha.h>

namespace xla::gpu::musa {
namespace {

constexpr uint64_t kMaxExportedGlobalBytes = uint64_t{1} << 40;
constexpr uint32_t kMaxExportedGlobalAlignment = uint32_t{1} << 16;
constexpr uint64_t kMaxBridgeWallTimeMicroseconds =
    uint64_t{24} * 60 * 60 * 1000 * 1000;
constexpr uint64_t kMaxBridgePeakMemoryBytes = uint64_t{1} << 50;
constexpr int kTextFormatRecursionLimit = 16;

absl::Status InvalidField(absl::string_view field, absl::string_view reason) {
  return absl::InvalidArgumentError(absl::StrCat(field, ": ", reason));
}

absl::Status ValidateVersion(absl::string_view field, uint32_t actual,
                             uint32_t expected) {
  if (actual != expected) {
    return absl::FailedPreconditionError(
        absl::StrCat(field, " is ", actual, "; expected ", expected));
  }
  return absl::OkStatus();
}

bool IsAsciiAlpha(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool IsAsciiDigit(unsigned char c) { return c >= '0' && c <= '9'; }

bool IsTokenBody(unsigned char c) {
  return IsAsciiAlpha(c) || IsAsciiDigit(c) || c == '_' || c == '-' ||
         c == '.' || c == '+';
}

absl::Status ValidateToken(absl::string_view value, absl::string_view field,
                           size_t max_bytes) {
  if (value.empty()) return InvalidField(field, "must not be empty");
  if (value.size() > max_bytes) return InvalidField(field, "is too long");
  if (!IsAsciiAlpha(value.front()) && !IsAsciiDigit(value.front()) &&
      value.front() != '_') {
    return InvalidField(field, "has an invalid first character");
  }
  for (unsigned char c : value) {
    if (!IsTokenBody(c)) {
      return InvalidField(field, "contains a non-token character");
    }
  }
  return absl::OkStatus();
}

bool IsLlvmSymbolInitial(unsigned char c) {
  return IsAsciiAlpha(c) || c == '_' || c == '$' || c == '.';
}

bool IsLlvmSymbolBody(unsigned char c) {
  return IsLlvmSymbolInitial(c) || IsAsciiDigit(c) || c == '-';
}

absl::Status ValidateSymbol(absl::string_view value, absl::string_view field) {
  if (value.empty()) return InvalidField(field, "must not be empty");
  if (value.size() > kMusaBridgeMaxSymbolNameBytes) {
    return InvalidField(field, "is too long");
  }
  if (!IsLlvmSymbolInitial(value.front())) {
    return InvalidField(field, "has an invalid first character");
  }
  for (unsigned char c : value) {
    if (!IsLlvmSymbolBody(c)) {
      return InvalidField(field, "contains an invalid LLVM symbol character");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateNoUnknownFields(const google::protobuf::Message& message,
                                     absl::string_view path) {
  const google::protobuf::Reflection* reflection = message.GetReflection();
  if (reflection->GetUnknownFields(message).field_count() != 0) {
    return InvalidField(path, "contains unknown protobuf fields");
  }
  std::vector<const google::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (const google::protobuf::FieldDescriptor* field : fields) {
    if (field->cpp_type() !=
        google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
      continue;
    }
    const std::string child_path = absl::StrCat(path, ".", field->name());
    if (field->is_repeated()) {
      const int count = reflection->FieldSize(message, field);
      for (int i = 0; i < count; ++i) {
        absl::Status status = ValidateNoUnknownFields(
            reflection->GetRepeatedMessage(message, field, i), child_path);
        if (!status.ok()) return status;
      }
    } else {
      absl::Status status = ValidateNoUnknownFields(
          reflection->GetMessage(message, field), child_path);
      if (!status.ok()) return status;
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateSha256(absl::string_view value, absl::string_view field) {
  if (value.size() != 64) {
    return InvalidField(field, "must contain 64 lower-case hexadecimal digits");
  }
  for (unsigned char c : value) {
    if (!IsAsciiDigit(c) && !(c >= 'a' && c <= 'f')) {
      return InvalidField(field,
                          "must contain 64 lower-case hexadecimal digits");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateSafeText(absl::string_view value, absl::string_view field,
                              size_t max_bytes, bool require_nonempty) {
  if (require_nonempty && value.empty()) {
    return InvalidField(field, "must not be empty");
  }
  if (value.size() > max_bytes) return InvalidField(field, "is too long");
  for (unsigned char c : value) {
    if ((c < 0x20 && c != '\n' && c != '\t') || c == 0x7f) {
      return InvalidField(field, "contains a forbidden control byte");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateNormalizedLlvm(absl::string_view llvm_ir) {
  absl::Status status = ValidateSafeText(llvm_ir, "normalized_llvm",
                                         kMusaBridgeMaxLlvmBytes, true);
  if (!status.ok()) return status;
  if (llvm_ir.back() != '\n') {
    return InvalidField("normalized_llvm",
                        "must end with one LF-terminated line");
  }
  for (size_t newline = llvm_ir.find('\n'); newline != absl::string_view::npos;
       newline = llvm_ir.find('\n', newline + 1)) {
    if (newline > 0 &&
        (llvm_ir[newline - 1] == ' ' || llvm_ir[newline - 1] == '\t')) {
      return InvalidField("normalized_llvm",
                          "contains trailing horizontal whitespace");
    }
  }
  return absl::OkStatus();
}

template <typename RepeatedStrings>
absl::Status ValidateSortedUniqueSymbols(const RepeatedStrings& values,
                                         absl::string_view field,
                                         size_t max_count,
                                         bool require_nonempty) {
  if (require_nonempty && values.empty()) {
    return InvalidField(field, "must not be empty");
  }
  if (values.size() > max_count)
    return InvalidField(field, "has too many items");
  for (int i = 0; i < values.size(); ++i) {
    absl::Status status = ValidateSymbol(values.Get(i), field);
    if (!status.ok()) return status;
    if (i > 0 && values.Get(i - 1) >= values.Get(i)) {
      return InvalidField(field, "must be strictly sorted and unique");
    }
  }
  return absl::OkStatus();
}

bool ContainsSortedSymbol(
    const google::protobuf::RepeatedPtrField<std::string>& symbols,
    absl::string_view symbol) {
  return std::binary_search(symbols.begin(), symbols.end(), symbol);
}

absl::Status ValidateExportedGlobals(const MusaBridgeCompileRequest& request) {
  if (request.exported_globals_size() > kMusaBridgeMaxExportedGlobalCount) {
    return InvalidField("exported_globals", "has too many items");
  }
  for (int i = 0; i < request.exported_globals_size(); ++i) {
    const MusaBridgeExportedGlobal& global = request.exported_globals(i);
    absl::Status status =
        ValidateSymbol(global.name(), "exported_globals.name");
    if (!status.ok()) return status;
    if (i > 0 && request.exported_globals(i - 1).name() >= global.name()) {
      return InvalidField("exported_globals",
                          "must be strictly sorted and unique by name");
    }
    switch (global.kind()) {
      case MUSA_BRIDGE_GLOBAL_KIND_MUTABLE:
      case MUSA_BRIDGE_GLOBAL_KIND_CONSTANT:
        break;
      case MUSA_BRIDGE_GLOBAL_KIND_UNSPECIFIED:
      default:
        return InvalidField("exported_globals.kind",
                            "contains an unsupported enum value");
    }
    const MusaAddressSpaceSpec* address_space =
        FindMusaAddressSpace(global.address_space());
    if (address_space == nullptr || !address_space->allowed_in_interchange) {
      return InvalidField("exported_globals.address_space",
                          "is not allowed in the interchange ABI");
    }
    const MusaAddressSpaceKind expected_address_space =
        global.kind() == MUSA_BRIDGE_GLOBAL_KIND_CONSTANT
            ? MusaAddressSpaceKind::kConstant
            : MusaAddressSpaceKind::kGlobal;
    // Shared GPU codegen emits runtime-resolved constant allocations in the
    // generic address space. Preserve that representation across the bridge.
    if (address_space->kind != MusaAddressSpaceKind::kGeneric &&
        address_space->kind != expected_address_space) {
      return InvalidField("exported_globals.address_space",
                          "does not match the exported global kind");
    }
    if (global.size_bytes() == 0 ||
        global.size_bytes() > kMaxExportedGlobalBytes) {
      return InvalidField("exported_globals.size_bytes", "is outside bounds");
    }
    if (global.alignment_bytes() == 0 ||
        global.alignment_bytes() > kMaxExportedGlobalAlignment ||
        (global.alignment_bytes() & (global.alignment_bytes() - 1)) != 0) {
      return InvalidField("exported_globals.alignment_bytes",
                          "must be a bounded power of two");
    }
    if (!ContainsSortedSymbol(request.exported_symbol_names(), global.name())) {
      return InvalidField("exported_globals.name",
                          "is absent from exported_symbol_names");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateDiagnostic(const MusaBridgeDiagnostic& diagnostic) {
  switch (diagnostic.severity()) {
    case MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_NOTE:
    case MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_WARNING:
    case MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_ERROR:
      break;
    case MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_UNSPECIFIED:
    default:
      return InvalidField("diagnostics.severity",
                          "contains an unsupported enum value");
  }
  absl::Status status =
      ValidateToken(diagnostic.code(), "diagnostics.code", 128);
  if (!status.ok()) return status;
  status = ValidateSafeText(diagnostic.message(), "diagnostics.message",
                            kMusaBridgeMaxDiagnosticMessageBytes, true);
  if (!status.ok()) return status;
  status = ValidateToken(diagnostic.component(), "diagnostics.component", 128);
  if (!status.ok()) return status;
  if ((diagnostic.line() == 0) != (diagnostic.column() == 0)) {
    return InvalidField(
        "diagnostics.location",
        "line and column must either both be zero or both nonzero");
  }
  if (!diagnostic.symbol_name().empty()) {
    status =
        ValidateSymbol(diagnostic.symbol_name(), "diagnostics.symbol_name");
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}

absl::Status PrintCanonical(const google::protobuf::Message& message,
                            std::string* output) {
  google::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(false);
  printer.SetUseShortRepeatedPrimitives(false);
  printer.SetUseUtf8StringEscaping(false);
  if (!printer.PrintToString(message, output)) {
    return absl::InternalError(
        "failed to print canonical MUSA bridge TextFormat");
  }
  return absl::OkStatus();
}

template <typename Message, typename Validator>
absl::StatusOr<std::string> EncodeCanonical(const Message& message,
                                            absl::string_view magic,
                                            size_t max_wire_bytes,
                                            Validator validator) {
  absl::Status status = validator(message);
  if (!status.ok()) return status;
  std::string payload;
  status = PrintCanonical(message, &payload);
  if (!status.ok()) return status;
  if (magic.size() + payload.size() > max_wire_bytes) {
    return absl::ResourceExhaustedError(
        "canonical MUSA bridge wire is too large");
  }
  return absl::StrCat(magic, payload);
}

template <typename Message, typename Validator>
absl::StatusOr<Message> DecodeCanonical(absl::string_view wire,
                                        absl::string_view magic,
                                        size_t max_wire_bytes,
                                        Validator validator) {
  if (wire.size() > max_wire_bytes) {
    return absl::ResourceExhaustedError("MUSA bridge wire is too large");
  }
  if (!absl::StartsWith(wire, magic)) {
    return absl::InvalidArgumentError(
        "MUSA bridge wire has the wrong magic line");
  }
  absl::string_view payload = wire.substr(magic.size());
  Message message;
  google::protobuf::TextFormat::Parser parser;
  parser.AllowUnknownField(false);
  parser.AllowUnknownExtension(false);
  parser.AllowCaseInsensitiveField(false);
  parser.AllowFieldNumber(false);
  parser.SetRecursionLimit(kTextFormatRecursionLimit);
  if (!parser.ParseFromString(payload, &message)) {
    return absl::InvalidArgumentError("failed to parse MUSA bridge TextFormat");
  }
  absl::Status status = validator(message);
  if (!status.ok()) return status;
  std::string canonical_payload;
  status = PrintCanonical(message, &canonical_payload);
  if (!status.ok()) return status;
  if (payload != canonical_payload) {
    return absl::InvalidArgumentError(
        "MUSA bridge TextFormat is valid but not canonical");
  }
  return message;
}

}  // namespace

std::string MusaBridgeSha256Hex(absl::string_view data) {
  std::array<uint8_t, SHA256_DIGEST_LENGTH> digest;
  SHA256(reinterpret_cast<const uint8_t*>(data.data()), data.size(),
         digest.data());
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(64, '0');
  for (size_t i = 0; i < digest.size(); ++i) {
    result[2 * i] = kHex[digest[i] >> 4];
    result[2 * i + 1] = kHex[digest[i] & 0x0f];
  }
  return result;
}

absl::Status ValidateMusaBridgeCompileRequest(
    const MusaBridgeCompileRequest& request) {
  absl::Status status = ValidateNoUnknownFields(request, "request");
  if (!status.ok()) return status;
  status = ValidateVersion("protocol_version", request.protocol_version(),
                           kMusaBridgeProtocolVersion);
  if (!status.ok()) return status;
  status = ValidateVersion("shim_abi_version", request.shim_abi_version(),
                           kMusaShimAbiVersion);
  if (!status.ok()) return status;
  status = ValidateVersion("mapping_version", request.mapping_version(),
                           kMusaShimMappingVersion);
  if (!status.ok()) return status;

  status = ValidateToken(request.module_name(), "module_name",
                         kMusaBridgeMaxModuleNameBytes);
  if (!status.ok()) return status;
  status = ValidateNormalizedLlvm(request.normalized_llvm());
  if (!status.ok()) return status;
  if (request.normalized_llvm_bytes() != request.normalized_llvm().size()) {
    return InvalidField("normalized_llvm_bytes",
                        "does not match normalized_llvm size");
  }
  status = ValidateSha256(request.normalized_llvm_sha256(),
                          "normalized_llvm_sha256");
  if (!status.ok()) return status;
  if (request.normalized_llvm_sha256() !=
      MusaBridgeSha256Hex(request.normalized_llvm())) {
    return InvalidField("normalized_llvm_sha256",
                        "does not match normalized_llvm");
  }

  status = ValidateSortedUniqueSymbols(request.kernel_entry_names(),
                                       "kernel_entry_names",
                                       kMusaBridgeMaxKernelCount, false);
  if (!status.ok()) return status;
  status = ValidateSortedUniqueSymbols(request.exported_symbol_names(),
                                       "exported_symbol_names",
                                       kMusaBridgeMaxExportedSymbolCount, true);
  if (!status.ok()) return status;
  for (const std::string& kernel : request.kernel_entry_names()) {
    if (!ContainsSortedSymbol(request.exported_symbol_names(), kernel)) {
      return InvalidField(
          "kernel_entry_names",
          "contains a symbol absent from exported_symbol_names");
    }
  }
  status = ValidateExportedGlobals(request);
  if (!status.ok()) return status;
  if (request.kernel_entry_names().empty() &&
      request.exported_globals().empty()) {
    return InvalidField("module_exports",
                        "must contain a kernel or typed global");
  }

  if (request.target_triple() != kMusaTargetTriple) {
    return InvalidField("target_triple", "is not mtgpu-mt-musa");
  }
  if (request.architecture() != kMusaTargetArchitecture) {
    return InvalidField("architecture", "is not the qualified mp_21 target");
  }
  if (!request.target_features().empty()) {
    return InvalidField(
        "target_features",
        "protocol version 1 does not define target feature overrides");
  }
  if (request.data_layout() != kMusaDataLayout) {
    return InvalidField("data_layout",
                        "does not match the qualified S80 layout");
  }
  switch (request.pointer_model()) {
    case MUSA_BRIDGE_POINTER_MODEL_OPAQUE:
      break;
    case MUSA_BRIDGE_POINTER_MODEL_UNSPECIFIED:
    default:
      return InvalidField("pointer_model", "must be the opaque pointer model");
  }
  if (request.pointer_width_bits() != kMusaInterchangePointerWidth) {
    return InvalidField("pointer_width_bits", "must be 64");
  }
  switch (request.byte_order()) {
    case MUSA_BRIDGE_BYTE_ORDER_LITTLE_ENDIAN:
      if (!kMusaInterchangeIsLittleEndian) {
        return InvalidField("byte_order",
                            "does not match the shared interchange ABI");
      }
      break;
    case MUSA_BRIDGE_BYTE_ORDER_UNSPECIFIED:
    default:
      return InvalidField("byte_order", "must be little endian");
  }
  if (!request.has_numerical_flags()) {
    return InvalidField("numerical_flags", "must be present explicitly");
  }
  if (request.optimization_level() > 3) {
    return InvalidField("optimization_level", "must be in the range 0..3");
  }

  status = ValidateToken(request.xla_revision(), "xla_revision",
                         kMusaBridgeMaxRevisionBytes);
  if (!status.ok()) return status;
  status = ValidateToken(request.current_llvm_revision(),
                         "current_llvm_revision", kMusaBridgeMaxRevisionBytes);
  if (!status.ok()) return status;
  status = ValidateToken(request.provider_name(), "provider_name",
                         kMusaBridgeMaxProviderNameBytes);
  if (!status.ok()) return status;
  status =
      ValidateSha256(request.provider_fingerprint(), "provider_fingerprint");
  if (!status.ok()) return status;
  status = ValidateSha256(request.bridge_fingerprint(), "bridge_fingerprint");
  if (!status.ok()) return status;
  status =
      ValidateSha256(request.toolchain_fingerprint(), "toolchain_fingerprint");
  if (!status.ok()) return status;
  status = ValidateSha256(request.mapping_fingerprint(), "mapping_fingerprint");
  if (!status.ok()) return status;
  if (request.mapping_fingerprint() != kMusaShimMappingSha256) {
    return InvalidField("mapping_fingerprint",
                        "does not match the compiled shim mapping");
  }
  return absl::OkStatus();
}

absl::Status ValidateMusaBridgeCompileResponse(
    const MusaBridgeCompileResponse& response) {
  absl::Status status = ValidateNoUnknownFields(response, "response");
  if (!status.ok()) return status;
  status = ValidateVersion("protocol_version", response.protocol_version(),
                           kMusaBridgeProtocolVersion);
  if (!status.ok()) return status;
  status = ValidateVersion("shim_abi_version", response.shim_abi_version(),
                           kMusaShimAbiVersion);
  if (!status.ok()) return status;
  status = ValidateVersion("mapping_version", response.mapping_version(),
                           kMusaShimMappingVersion);
  if (!status.ok()) return status;

  bool success = false;
  switch (response.status()) {
    case MUSA_BRIDGE_STATUS_OK:
      success = true;
      break;
    case MUSA_BRIDGE_STATUS_REJECTED:
    case MUSA_BRIDGE_STATUS_COMPILATION_ERROR:
    case MUSA_BRIDGE_STATUS_INTERNAL_ERROR:
      break;
    case MUSA_BRIDGE_STATUS_UNSPECIFIED:
    default:
      return InvalidField("status", "contains an unsupported enum value");
  }
  status = ValidateSha256(response.request_sha256(), "request_sha256");
  if (!status.ok()) return status;
  status = ValidateToken(response.provider_name(), "provider_name",
                         kMusaBridgeMaxProviderNameBytes);
  if (!status.ok()) return status;
  status =
      ValidateSha256(response.provider_fingerprint(), "provider_fingerprint");
  if (!status.ok()) return status;
  status = ValidateSha256(response.bridge_fingerprint(), "bridge_fingerprint");
  if (!status.ok()) return status;
  status =
      ValidateSha256(response.toolchain_fingerprint(), "toolchain_fingerprint");
  if (!status.ok()) return status;
  status =
      ValidateSha256(response.mapping_fingerprint(), "mapping_fingerprint");
  if (!status.ok()) return status;
  if (response.mapping_fingerprint() != kMusaShimMappingSha256) {
    return InvalidField("mapping_fingerprint",
                        "does not match the compiled shim mapping");
  }

  if (response.diagnostics_size() > kMusaBridgeMaxDiagnosticCount) {
    return InvalidField("diagnostics", "has too many items");
  }
  bool has_error = false;
  size_t diagnostic_bytes = 0;
  for (const MusaBridgeDiagnostic& diagnostic : response.diagnostics()) {
    status = ValidateDiagnostic(diagnostic);
    if (!status.ok()) return status;
    diagnostic_bytes += diagnostic.code().size() + diagnostic.message().size() +
                        diagnostic.component().size() +
                        diagnostic.symbol_name().size();
    if (diagnostic_bytes > kMusaBridgeMaxDiagnosticBytes) {
      return InvalidField("diagnostics", "aggregate text is too large");
    }
    has_error |= diagnostic.severity() == MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_ERROR;
  }
  if (!response.has_stats()) {
    return InvalidField("stats", "must be present explicitly");
  }
  const MusaBridgeCompileStats& stats = response.stats();
  if (stats.input_llvm_bytes() == 0 ||
      stats.input_llvm_bytes() > kMusaBridgeMaxLlvmBytes) {
    return InvalidField("stats.input_llvm_bytes", "is outside bounds");
  }
  if (stats.kernel_count() > kMusaBridgeMaxKernelCount) {
    return InvalidField("stats.kernel_count", "is outside bounds");
  }
  if (stats.exported_symbol_count() < stats.kernel_count() ||
      stats.exported_symbol_count() > kMusaBridgeMaxExportedSymbolCount) {
    return InvalidField("stats.exported_symbol_count", "is outside bounds");
  }
  if (stats.diagnostic_count() != response.diagnostics_size()) {
    return InvalidField("stats.diagnostic_count",
                        "does not match diagnostics size");
  }
  if (stats.bridge_wall_time_microseconds() > kMaxBridgeWallTimeMicroseconds) {
    return InvalidField("stats.bridge_wall_time_microseconds",
                        "is outside bounds");
  }
  if (stats.peak_memory_bytes() > kMaxBridgePeakMemoryBytes) {
    return InvalidField("stats.peak_memory_bytes", "is outside bounds");
  }

  if (success) {
    if (has_error) {
      return InvalidField("diagnostics",
                          "an OK response cannot contain an error diagnostic");
    }
    if (response.mubin().empty() ||
        response.mubin().size() > kMusaBridgeMaxMubinBytes) {
      return InvalidField("mubin", "is empty or outside bounds");
    }
    if (stats.output_mubin_bytes() != response.mubin().size()) {
      return InvalidField("stats.output_mubin_bytes",
                          "does not match MUBIN size");
    }
    status = ValidateSha256(response.mubin_sha256(), "mubin_sha256");
    if (!status.ok()) return status;
    if (response.mubin_sha256() != MusaBridgeSha256Hex(response.mubin())) {
      return InvalidField("mubin_sha256", "does not match MUBIN");
    }
  } else {
    if (!has_error) {
      return InvalidField("diagnostics",
                          "a failed response requires an error diagnostic");
    }
    if (!response.mubin().empty() || !response.mubin_sha256().empty() ||
        stats.output_mubin_bytes() != 0) {
      return InvalidField("mubin", "must be absent from every failed response");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateMusaBridgeExchange(
    const MusaBridgeCompileRequest& request,
    const MusaBridgeCompileResponse& response) {
  absl::Status status = ValidateMusaBridgeCompileRequest(request);
  if (!status.ok()) return status;
  status = ValidateMusaBridgeCompileResponse(response);
  if (!status.ok()) return status;
  absl::StatusOr<std::string> request_sha =
      MusaBridgeCompileRequestSha256(request);
  if (!request_sha.ok()) return request_sha.status();
  if (response.request_sha256() != *request_sha) {
    return absl::FailedPreconditionError(
        "response request_sha256 does not bind to the canonical request");
  }
  if (response.provider_name() != request.provider_name() ||
      response.provider_fingerprint() != request.provider_fingerprint() ||
      response.bridge_fingerprint() != request.bridge_fingerprint() ||
      response.toolchain_fingerprint() != request.toolchain_fingerprint() ||
      response.mapping_fingerprint() != request.mapping_fingerprint()) {
    return absl::FailedPreconditionError(
        "response provider, toolchain, or mapping identity differs from the "
        "request");
  }
  const MusaBridgeCompileStats& stats = response.stats();
  if (stats.input_llvm_bytes() != request.normalized_llvm_bytes() ||
      stats.kernel_count() != request.kernel_entry_names_size() ||
      stats.exported_symbol_count() != request.exported_symbol_names_size()) {
    return absl::FailedPreconditionError(
        "response statistics do not describe the bound request");
  }
  for (const MusaBridgeDiagnostic& diagnostic : response.diagnostics()) {
    if (!diagnostic.symbol_name().empty() &&
        !ContainsSortedSymbol(request.exported_symbol_names(),
                              diagnostic.symbol_name())) {
      return absl::FailedPreconditionError(
          "response diagnostic names a symbol outside the bound request");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeMusaBridgeCompileRequest(
    const MusaBridgeCompileRequest& request) {
  return EncodeCanonical(request, kMusaBridgeRequestMagic,
                         kMusaBridgeMaxRequestWireBytes,
                         ValidateMusaBridgeCompileRequest);
}

absl::StatusOr<MusaBridgeCompileRequest> DecodeMusaBridgeCompileRequest(
    absl::string_view wire) {
  return DecodeCanonical<MusaBridgeCompileRequest>(
      wire, kMusaBridgeRequestMagic, kMusaBridgeMaxRequestWireBytes,
      ValidateMusaBridgeCompileRequest);
}

absl::StatusOr<std::string> EncodeMusaBridgeCompileResponse(
    const MusaBridgeCompileResponse& response) {
  return EncodeCanonical(response, kMusaBridgeResponseMagic,
                         kMusaBridgeMaxResponseWireBytes,
                         ValidateMusaBridgeCompileResponse);
}

absl::StatusOr<MusaBridgeCompileResponse> DecodeMusaBridgeCompileResponse(
    absl::string_view wire) {
  return DecodeCanonical<MusaBridgeCompileResponse>(
      wire, kMusaBridgeResponseMagic, kMusaBridgeMaxResponseWireBytes,
      ValidateMusaBridgeCompileResponse);
}

absl::StatusOr<std::string> MusaBridgeCompileRequestSha256(
    const MusaBridgeCompileRequest& request) {
  absl::StatusOr<std::string> wire = EncodeMusaBridgeCompileRequest(request);
  if (!wire.ok()) return wire.status();
  return MusaBridgeSha256Hex(*wire);
}

}  // namespace xla::gpu::musa
