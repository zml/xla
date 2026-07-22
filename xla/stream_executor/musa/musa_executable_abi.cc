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

#include "xla/stream_executor/musa/musa_executable_abi.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include <openssl/sha.h>
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/musa/musa_mubin.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_target_contract.h"

namespace stream_executor::musa {
namespace {

using MusaVersion = ExecutableAbiVersionProto::MusaPlatformVersion;

constexpr size_t kMaxProvenanceValueBytes = 256;
constexpr size_t kMaxOptionalLibraryNameBytes = 128;
constexpr size_t kMaxOptionalLibraryAbiVersionBytes = 128;
constexpr int kMaxRequiredOptionalLibraries = 64;

absl::Status InvalidAbi(absl::string_view reason) {
  return absl::InvalidArgumentError(
      absl::StrCat("invalid MUSA executable ABI: ", reason));
}

bool IsLowerHexSha256(absl::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return std::isdigit(c) != 0 || (c >= 'a' && c <= 'f');
         });
}

absl::Status RequireSha256(absl::string_view value, absl::string_view field) {
  if (!IsLowerHexSha256(value)) {
    return InvalidAbi(absl::StrCat(field, " is not lower-case SHA-256"));
  }
  return absl::OkStatus();
}

absl::Status RequireBoundedText(absl::string_view value,
                                absl::string_view field, size_t max_bytes) {
  if (value.empty()) {
    return InvalidAbi(absl::StrCat(field, " is empty"));
  }
  if (value.size() > max_bytes) {
    return InvalidAbi(absl::StrCat(field, " exceeds ", max_bytes, " bytes"));
  }
  for (unsigned char c : value) {
    if (c < 0x20 || c == 0x7f) {
      return InvalidAbi(absl::StrCat(field, " contains control characters"));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateCompilerProvenanceShape(const MusaVersion& musa) {
  if (musa.bridge_protocol_version() == 0 || musa.shim_abi_version() == 0 ||
      musa.shim_mapping_version() == 0) {
    return InvalidAbi(
        "compiler bridge, shim, and mapping versions must be nonzero");
  }
  for (const auto& [value, field] :
       {std::pair<absl::string_view, absl::string_view>(
            musa.llvm14_compatibility_revision(),
            "llvm14_compatibility_revision"),
        {musa.xla_revision(), "xla_revision"},
        {musa.current_llvm_revision(), "current_llvm_revision"},
        {musa.provider_name(), "provider_name"},
        {musa.driver_compatibility(), "driver_compatibility"},
        {musa.runtime_compatibility(), "runtime_compatibility"}}) {
    absl::Status status =
        RequireBoundedText(value, field, kMaxProvenanceValueBytes);
    if (!status.ok()) return status;
  }
  for (const auto& [value, field] :
       {std::pair<absl::string_view, absl::string_view>(
            musa.shim_mapping_sha256(), "shim_mapping_sha256"),
        {musa.provider_fingerprint(), "provider_fingerprint"},
        {musa.bridge_fingerprint(), "bridge_fingerprint"},
        {musa.toolchain_fingerprint(), "toolchain_fingerprint"},
        {musa.libdevice_fingerprint(), "libdevice_fingerprint"}}) {
    absl::Status status = RequireSha256(value, field);
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}

absl::Status ValidateNumericalOptions(
    const MusaVersion::NumericalOptions& options) {
  if (options.optimization_level() != 2 || options.emit_debug_information() ||
      !options.deterministic() || options.fast_math() ||
      options.flush_denormals_to_zero() || options.finite_math_only() ||
      options.unsafe_math_optimizations() || options.no_signed_zeros() ||
      options.allow_fp_contract()) {
    return InvalidAbi(
        "numerical options are outside the qualified deterministic O2 "
        "profile");
  }
  return absl::OkStatus();
}

absl::Status ValidateOptionalLibraries(const MusaVersion& musa) {
  if (musa.required_optional_library_abis_size() >
      kMaxRequiredOptionalLibraries) {
    return InvalidAbi("too many required optional-library entries");
  }
  absl::flat_hash_set<std::string> library_names;
  std::string previous_name;
  for (const MusaVersion::OptionalLibraryAbi& library :
       musa.required_optional_library_abis()) {
    absl::Status status = RequireBoundedText(
        library.name(), "optional library name", kMaxOptionalLibraryNameBytes);
    if (!status.ok()) return status;
    status = RequireBoundedText(library.abi_version(),
                                "optional library ABI version",
                                kMaxOptionalLibraryAbiVersionBytes);
    if (!status.ok()) return status;
    if (!library_names.insert(library.name()).second) {
      return InvalidAbi("required optional-library name is duplicated");
    }
    if (!previous_name.empty() && library.name() <= previous_name) {
      return InvalidAbi(
          "required optional-library entries are not canonically sorted");
    }
    previous_name = library.name();
    if (!library.fingerprint().empty()) {
      status =
          RequireSha256(library.fingerprint(), "optional library fingerprint");
      if (!status.ok()) return status;
    }
  }
  return absl::OkStatus();
}

}  // namespace

std::string MusaExecutableBinarySha256(absl::Span<const uint8_t> main_binary) {
  std::array<uint8_t, SHA256_DIGEST_LENGTH> digest;
  SHA256(main_binary.data(), main_binary.size(), digest.data());
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(64, '0');
  for (size_t i = 0; i < digest.size(); ++i) {
    result[2 * i] = kHex[digest[i] >> 4];
    result[2 * i + 1] = kHex[digest[i] & 0x0f];
  }
  return result;
}

absl::Status ValidateMusaExecutableAbi(
    const ExecutableAbiVersion& executable_abi_version,
    absl::Span<const uint8_t> main_binary) {
  const ExecutableAbiVersionProto& proto = executable_abi_version.proto();
  if (proto.platform_name() != kMusaPlatformId->ToName()) {
    return InvalidAbi(
        absl::StrCat("platform is ", proto.platform_name(), ", expected MUSA"));
  }
  if (!proto.has_musa_platform_version()) {
    return InvalidAbi("MUSA platform metadata is missing");
  }
  const MusaVersion& musa = proto.musa_platform_version();
  if (musa.envelope_version() != kMusaExecutableAbiEnvelopeVersion) {
    return InvalidAbi(
        absl::StrCat("unsupported envelope version ", musa.envelope_version()));
  }
  if (musa.binary_kind() != kMusaExecutableBinaryKind ||
      musa.mubin_loader_abi_version() != kMubinLoaderAbiVersion) {
    return InvalidAbi("binary kind or MUBIN loader ABI does not match");
  }
  if (musa.target_triple() != kMusaTargetTriple ||
      musa.architecture() != kS80TargetArchitecture ||
      musa.target_features() != kS80TargetFeatures ||
      musa.data_layout() != kMusaTargetDataLayout ||
      musa.pointer_width() != kMusaPointerWidth ||
      musa.little_endian() != kMusaIsLittleEndian ||
      musa.hardware_warp_size() != kS80HardwareWarpSize ||
      musa.logical_subgroup_size() != kS80CompilerLogicalSubgroupSize) {
    return InvalidAbi("target ABI does not match the qualified S80");
  }
  absl::Status provenance = ValidateCompilerProvenanceShape(musa);
  if (!provenance.ok()) return provenance;
  absl::Status main_digest =
      RequireSha256(musa.main_binary_sha256(), "main_binary_sha256");
  if (!main_digest.ok()) return main_digest;
  if (musa.main_binary_sha256() != MusaExecutableBinarySha256(main_binary)) {
    return InvalidAbi("main MUBIN digest does not match its bytes");
  }
  if (!main_binary.empty()) {
    absl::StatusOr<MusaMubinMetadata> metadata = ValidateMusaMubin(main_binary);
    if (!metadata.ok()) return metadata.status();
  }
  if (!musa.has_numerical_options()) {
    return InvalidAbi("numerical options are missing");
  }
  absl::Status numerical = ValidateNumericalOptions(musa.numerical_options());
  if (!numerical.ok()) return numerical;
  return ValidateOptionalLibraries(musa);
}

}  // namespace stream_executor::musa
