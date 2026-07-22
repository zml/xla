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

#include "xla/stream_executor/musa/musa_runtime_abi_version.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/stream_executor/musa/musa_mubin.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_runtime_abi_version.pb.h"
#include "xla/stream_executor/musa/musa_target_contract.h"
#include "xla/stream_executor/musa/musa_version_parser.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::musa {
namespace {

using MusaExecutableVersion = ExecutableAbiVersionProto::MusaPlatformVersion;

absl::Status RuntimeProtoError(absl::string_view reason) {
  return absl::InvalidArgumentError(
      absl::StrCat("invalid MUSA runtime ABI version: ", reason));
}

absl::Status Incompatible(absl::string_view reason) {
  return absl::FailedPreconditionError(
      absl::StrCat("incompatible MUSA executable ABI: ", reason));
}

absl::Status ValidateAvailableLibraries(
    const std::vector<MusaOptionalLibraryAbi>& libraries);

absl::StatusOr<SemanticVersion> ParseRuntimeVersion(absl::string_view value,
                                                    absl::string_view field) {
  absl::StatusOr<SemanticVersion> version =
      SemanticVersion::ParseFromString(value);
  if (!version.ok()) {
    return RuntimeProtoError(
        absl::StrCat(field, " is not a semantic version: ", value));
  }
  if (!version->IsValid()) {
    return RuntimeProtoError(absl::StrCat(field, " must be known and nonzero"));
  }
  return *version;
}

absl::Status ValidateRuntimeFacts(
    const SemanticVersion& runtime_version,
    const SemanticVersion& driver_version,
    const SemanticVersion& kernel_driver_version,
    const SemanticVersion& toolkit_version,
    const std::vector<MusaOptionalLibraryAbi>& libraries) {
  for (const auto& [version, field] :
       {std::pair<const SemanticVersion*, absl::string_view>(&runtime_version,
                                                             "runtime_version"),
        {&driver_version, "driver_version"},
        {&kernel_driver_version, "kernel_driver_version"},
        {&toolkit_version, "toolkit_version"}}) {
    if (!version->IsValid()) {
      return RuntimeProtoError(
          absl::StrCat(field, " must be known and nonzero"));
    }
  }
  return ValidateAvailableLibraries(libraries);
}

absl::StatusOr<SemanticVersion> ParseExecutableVersion(
    absl::string_view value, absl::string_view field) {
  if (value.empty()) {
    return Incompatible(absl::StrCat(field, " is missing"));
  }
  absl::StatusOr<SemanticVersion> version =
      SemanticVersion::ParseFromString(value);
  if (!version.ok()) {
    return Incompatible(
        absl::StrCat(field, " is not a semantic version: ", value));
  }
  if (!version->IsValid()) {
    return Incompatible(absl::StrCat(field, " must be known and nonzero"));
  }
  return *version;
}

absl::Status ValidateAvailableLibraries(
    const std::vector<MusaOptionalLibraryAbi>& libraries) {
  std::set<std::string> names;
  for (const MusaOptionalLibraryAbi& library : libraries) {
    if (library.name.empty() || library.abi_version.empty()) {
      return RuntimeProtoError(
          "available optional-library name and ABI version must be nonempty");
    }
    if (!names.insert(library.name).second) {
      return RuntimeProtoError(absl::StrCat(
          "available optional-library name is duplicated: ", library.name));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateRange(const SemanticVersion& current,
                           absl::string_view minimum_text,
                           absl::string_view maximum_text,
                           absl::string_view name) {
  ASSIGN_OR_RETURN(
      SemanticVersion minimum,
      ParseExecutableVersion(minimum_text,
                             absl::StrCat("minimum_", name, "_version")));
  ASSIGN_OR_RETURN(
      SemanticVersion maximum,
      ParseExecutableVersion(maximum_text,
                             absl::StrCat("maximum_", name, "_version")));
  if (maximum < minimum) {
    return Incompatible(absl::StrCat(
        name, " version range is inverted: ", minimum, "..", maximum));
  }
  if (current < minimum || current > maximum) {
    return Incompatible(absl::StrCat(name, " version ", current,
                                     " is outside required inclusive range ",
                                     minimum, "..", maximum));
  }
  return absl::OkStatus();
}

absl::Status ValidateTargetContract(const MusaExecutableVersion& version) {
  if (version.envelope_version() != kMusaExecutableAbiEnvelopeVersion) {
    return Incompatible(absl::StrCat("unsupported envelope version ",
                                     version.envelope_version(), "; expected ",
                                     kMusaExecutableAbiEnvelopeVersion));
  }
  if (version.binary_kind() != kMusaExecutableBinaryKind) {
    return Incompatible(absl::StrCat("binary kind is ", version.binary_kind(),
                                     "; expected ", kMusaExecutableBinaryKind));
  }
  if (version.mubin_loader_abi_version() != kMubinLoaderAbiVersion) {
    return Incompatible(absl::StrCat("MUBIN loader ABI is ",
                                     version.mubin_loader_abi_version(),
                                     "; expected ", kMubinLoaderAbiVersion));
  }
  if (version.target_triple() != kMusaTargetTriple ||
      version.architecture() != kS80TargetArchitecture ||
      version.target_features() != kS80TargetFeatures ||
      version.data_layout() != kMusaTargetDataLayout ||
      version.pointer_width() != kMusaPointerWidth ||
      version.little_endian() != kMusaIsLittleEndian ||
      version.hardware_warp_size() != kS80HardwareWarpSize ||
      version.logical_subgroup_size() != kS80CompilerLogicalSubgroupSize) {
    return Incompatible(
        "target contract does not match the qualified S80 mp_21 ABI");
  }
  return absl::OkStatus();
}

absl::Status ValidateRequiredLibraries(
    const MusaExecutableVersion& executable,
    const std::vector<MusaOptionalLibraryAbi>& available_libraries) {
  std::map<std::string, const MusaOptionalLibraryAbi*> available_by_name;
  for (const MusaOptionalLibraryAbi& library : available_libraries) {
    available_by_name.emplace(library.name, &library);
  }

  std::set<std::string> required_names;
  for (const MusaExecutableVersion::OptionalLibraryAbi& required :
       executable.required_optional_library_abis()) {
    if (required.name().empty() || required.abi_version().empty()) {
      return Incompatible(
          "required optional-library name and ABI version must be nonempty");
    }
    if (!required_names.insert(required.name()).second) {
      return Incompatible(absl::StrCat(
          "required optional-library name is duplicated: ", required.name()));
    }
  }

  for (const MusaExecutableVersion::OptionalLibraryAbi& required :
       executable.required_optional_library_abis()) {
    auto available = available_by_name.find(required.name());
    if (available == available_by_name.end()) {
      return Incompatible(absl::StrCat(
          "required optional library is unavailable: ", required.name()));
    }
    if (available->second->abi_version != required.abi_version()) {
      return Incompatible(
          absl::StrCat("optional library ", required.name(),
                       " has ABI version ", available->second->abi_version,
                       "; executable requires ", required.abi_version()));
    }
    if (!required.fingerprint().empty() &&
        available->second->fingerprint != required.fingerprint()) {
      return Incompatible(absl::StrCat(
          "optional library fingerprint mismatch: ", required.name()));
    }
  }
  return absl::OkStatus();
}

}  // namespace

MusaRuntimeAbiVersion::MusaRuntimeAbiVersion(
    ValidatedTag, SemanticVersion runtime_version,
    SemanticVersion driver_version, SemanticVersion kernel_driver_version,
    SemanticVersion toolkit_version,
    std::vector<MusaOptionalLibraryAbi> available_optional_library_abis)
    : runtime_version_(runtime_version),
      driver_version_(driver_version),
      kernel_driver_version_(kernel_driver_version),
      toolkit_version_(toolkit_version),
      available_optional_library_abis_(
          std::move(available_optional_library_abis)) {}

absl::StatusOr<MusaRuntimeAbiVersion>
MusaRuntimeAbiVersion::CreateFromApiVersions(
    int runtime_version, int driver_version,
    SemanticVersion kernel_driver_version, int toolkit_version) {
  ASSIGN_OR_RETURN(SemanticVersion parsed_runtime_version,
                   ParseMusaVersion(runtime_version));
  ASSIGN_OR_RETURN(SemanticVersion parsed_driver_version,
                   ParseMusaVersion(driver_version));
  ASSIGN_OR_RETURN(SemanticVersion parsed_toolkit_version,
                   ParseMusaVersion(toolkit_version));
  return Create(parsed_runtime_version, parsed_driver_version,
                kernel_driver_version, parsed_toolkit_version);
}

absl::StatusOr<MusaRuntimeAbiVersion> MusaRuntimeAbiVersion::Create(
    SemanticVersion runtime_version, SemanticVersion driver_version,
    SemanticVersion kernel_driver_version, SemanticVersion toolkit_version,
    std::vector<MusaOptionalLibraryAbi> available_optional_library_abis) {
  RETURN_IF_ERROR(ValidateRuntimeFacts(runtime_version, driver_version,
                                       kernel_driver_version, toolkit_version,
                                       available_optional_library_abis));
  return MusaRuntimeAbiVersion(ValidatedTag{}, runtime_version, driver_version,
                               kernel_driver_version, toolkit_version,
                               std::move(available_optional_library_abis));
}

absl::StatusOr<std::unique_ptr<MusaRuntimeAbiVersion>>
MusaRuntimeAbiVersion::FromProto(const MusaRuntimeAbiVersionProto& proto) {
  ASSIGN_OR_RETURN(
      SemanticVersion runtime_version,
      ParseRuntimeVersion(proto.runtime_version(), "runtime_version"));
  ASSIGN_OR_RETURN(
      SemanticVersion driver_version,
      ParseRuntimeVersion(proto.driver_version(), "driver_version"));
  ASSIGN_OR_RETURN(SemanticVersion kernel_driver_version,
                   ParseRuntimeVersion(proto.kernel_driver_version(),
                                       "kernel_driver_version"));
  ASSIGN_OR_RETURN(
      SemanticVersion toolkit_version,
      ParseRuntimeVersion(proto.toolkit_version(), "toolkit_version"));

  std::vector<MusaOptionalLibraryAbi> libraries;
  libraries.reserve(proto.available_optional_library_abis_size());
  for (const MusaRuntimeAbiVersionProto::OptionalLibraryAbi& library :
       proto.available_optional_library_abis()) {
    libraries.push_back(
        {library.name(), library.abi_version(), library.fingerprint()});
  }
  ASSIGN_OR_RETURN(
      MusaRuntimeAbiVersion version,
      Create(runtime_version, driver_version, kernel_driver_version,
             toolkit_version, std::move(libraries)));
  return std::make_unique<MusaRuntimeAbiVersion>(std::move(version));
}

absl::StatusOr<std::unique_ptr<MusaRuntimeAbiVersion>>
MusaRuntimeAbiVersion::FromSerializedProto(absl::string_view proto) {
  MusaRuntimeAbiVersionProto musa_proto;
  if (!musa_proto.ParseFromString(proto)) {
    return RuntimeProtoError("serialized payload cannot be parsed");
  }
  return FromProto(musa_proto);
}

absl::Status MusaRuntimeAbiVersion::IsCompatibleWith(
    const ExecutableAbiVersion& executable_abi_version) const {
  RETURN_IF_ERROR(ValidateRuntimeFacts(runtime_version_, driver_version_,
                                       kernel_driver_version_, toolkit_version_,
                                       available_optional_library_abis_));

  const ExecutableAbiVersionProto& executable_proto =
      executable_abi_version.proto();
  if (executable_proto.platform_name() != kMusaPlatformId->ToName()) {
    return Incompatible(absl::StrCat("platform is ",
                                     executable_proto.platform_name(),
                                     "; expected ", kMusaPlatformId->ToName()));
  }
  if (!executable_proto.has_musa_platform_version()) {
    return Incompatible("MUSA platform version is missing");
  }

  const MusaExecutableVersion& musa = executable_proto.musa_platform_version();
  RETURN_IF_ERROR(ValidateTargetContract(musa));

  ASSIGN_OR_RETURN(SemanticVersion required_toolkit,
                   ParseExecutableVersion(musa.required_toolkit_version(),
                                          "required_toolkit_version"));
  if (toolkit_version_ != required_toolkit) {
    return Incompatible(absl::StrCat("toolkit version is ", toolkit_version_,
                                     "; executable requires ",
                                     required_toolkit));
  }

  RETURN_IF_ERROR(ValidateRange(runtime_version_,
                                musa.minimum_runtime_version(),
                                musa.maximum_runtime_version(), "runtime"));
  RETURN_IF_ERROR(ValidateRange(driver_version_, musa.minimum_driver_version(),
                                musa.maximum_driver_version(), "driver"));
  RETURN_IF_ERROR(ValidateRange(
      kernel_driver_version_, musa.minimum_kernel_driver_version(),
      musa.maximum_kernel_driver_version(), "kernel_driver"));
  return ValidateRequiredLibraries(musa, available_optional_library_abis_);
}

absl::StatusOr<RuntimeAbiVersionProto> MusaRuntimeAbiVersion::ToProto() const {
  RETURN_IF_ERROR(ValidateRuntimeFacts(runtime_version_, driver_version_,
                                       kernel_driver_version_, toolkit_version_,
                                       available_optional_library_abis_));

  MusaRuntimeAbiVersionProto musa_proto;
  musa_proto.set_runtime_version(runtime_version_.ToString());
  musa_proto.set_driver_version(driver_version_.ToString());
  musa_proto.set_kernel_driver_version(kernel_driver_version_.ToString());
  musa_proto.set_toolkit_version(toolkit_version_.ToString());
  for (const MusaOptionalLibraryAbi& library :
       available_optional_library_abis_) {
    MusaRuntimeAbiVersionProto::OptionalLibraryAbi* proto_library =
        musa_proto.add_available_optional_library_abis();
    proto_library->set_name(library.name);
    proto_library->set_abi_version(library.abi_version);
    proto_library->set_fingerprint(library.fingerprint);
  }

  std::string serialized_proto;
  if (!musa_proto.SerializeToString(&serialized_proto)) {
    return absl::InternalError(
        "Failed to serialize MusaRuntimeAbiVersionProto to string.");
  }

  RuntimeAbiVersionProto proto;
  proto.set_platform_name(kMusaPlatformId->ToName());
  proto.set_platform_specific_version(std::move(serialized_proto));
  return proto;
}

absl::StatusOr<PlatformId> MusaRuntimeAbiVersion::platform_id() const {
  return kMusaPlatformId;
}

}  // namespace stream_executor::musa
