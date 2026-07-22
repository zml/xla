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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/musa/musa_mubin.h"
#include "xla/stream_executor/musa/musa_target_contract.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using MusaVersion = ExecutableAbiVersionProto::MusaPlatformVersion;

ExecutableAbiVersionProto CompleteProto(
    absl::Span<const uint8_t> main_binary = {}) {
  ExecutableAbiVersionProto proto;
  proto.set_platform_name("MUSA");
  MusaVersion* musa = proto.mutable_musa_platform_version();
  musa->set_envelope_version(kMusaExecutableAbiEnvelopeVersion);
  musa->set_binary_kind(kMusaExecutableBinaryKind);
  musa->set_mubin_loader_abi_version(kMubinLoaderAbiVersion);
  musa->set_main_binary_sha256(MusaExecutableBinarySha256(main_binary));
  musa->set_target_triple(kMusaTargetTriple);
  musa->set_architecture(kS80TargetArchitecture);
  musa->set_target_features(kS80TargetFeatures);
  musa->set_data_layout(kMusaTargetDataLayout);
  musa->set_pointer_width(kMusaPointerWidth);
  musa->set_little_endian(kMusaIsLittleEndian);
  musa->set_hardware_warp_size(kS80HardwareWarpSize);
  musa->set_logical_subgroup_size(kS80CompilerLogicalSubgroupSize);

  // These values intentionally do not import compiler-service constants.
  musa->set_bridge_protocol_version(7);
  musa->set_shim_abi_version(8);
  musa->set_shim_mapping_version(9);
  musa->set_shim_mapping_sha256(std::string(64, '1'));
  musa->set_llvm14_compatibility_revision("compat-revision-a");
  musa->set_xla_revision("xla-revision-a");
  musa->set_current_llvm_revision("llvm-revision-a");
  musa->set_provider_name("provider-a");
  musa->set_provider_fingerprint(std::string(64, '2'));
  musa->set_bridge_fingerprint(std::string(64, '3'));
  musa->set_toolchain_fingerprint(std::string(64, '4'));
  musa->set_libdevice_fingerprint(std::string(64, '5'));
  musa->set_driver_compatibility("driver-contract-a");
  musa->set_runtime_compatibility("runtime-contract-a");

  MusaVersion::NumericalOptions* numerical = musa->mutable_numerical_options();
  numerical->set_optimization_level(2);
  numerical->set_deterministic(true);
  musa->set_required_toolkit_version("4.0.1");
  musa->set_minimum_runtime_version("1.5.4");
  musa->set_maximum_runtime_version("1.5.4");
  musa->set_minimum_driver_version("3.0.0");
  musa->set_maximum_driver_version("3.0.0");
  musa->set_minimum_kernel_driver_version("3.0.0");
  musa->set_maximum_kernel_driver_version("3.0.0");
  return proto;
}

absl::Status ValidationStatus(const ExecutableAbiVersionProto& proto,
                              absl::Span<const uint8_t> main_binary = {}) {
  absl::StatusOr<ExecutableAbiVersion> executable =
      ExecutableAbiVersion::FromProto(proto);
  if (!executable.ok()) return executable.status();
  return ValidateMusaExecutableAbi(*executable, main_binary);
}

TEST(MusaExecutableAbiTest, AcceptsBoundedOpaqueCompilerProvenanceDrift) {
  ExecutableAbiVersionProto proto = CompleteProto();
  EXPECT_OK(ValidationStatus(proto));

  MusaVersion* musa = proto.mutable_musa_platform_version();
  musa->set_bridge_protocol_version(70);
  musa->set_shim_abi_version(80);
  musa->set_shim_mapping_version(90);
  musa->set_shim_mapping_sha256(std::string(64, 'a'));
  musa->set_llvm14_compatibility_revision("compat-revision-b");
  EXPECT_OK(ValidationStatus(proto));
}

TEST(MusaExecutableAbiTest, RejectsCorruptBinaryAndDigestMetadata) {
  const std::vector<uint8_t> not_mubin = {0x7f, 'E', 'L', 'F'};
  ExecutableAbiVersionProto proto = CompleteProto(not_mubin);
  EXPECT_THAT(ValidationStatus(proto, not_mubin),
              StatusIs(absl::StatusCode::kInvalidArgument));

  proto = CompleteProto();
  proto.mutable_musa_platform_version()->set_main_binary_sha256(
      std::string(64, 'f'));
  EXPECT_THAT(ValidationStatus(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("digest does not match")));
}

TEST(MusaExecutableAbiTest, RejectsMalformedOrUnboundedProvenanceMetadata) {
  ExecutableAbiVersionProto proto = CompleteProto();
  proto.mutable_musa_platform_version()->set_provider_fingerprint("not-a-sha");
  EXPECT_THAT(ValidationStatus(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("provider_fingerprint")));

  proto = CompleteProto();
  proto.mutable_musa_platform_version()->set_xla_revision(
      std::string(257, 'x'));
  EXPECT_THAT(ValidationStatus(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds 256 bytes")));
}

TEST(MusaExecutableAbiTest, RejectsNoncanonicalOptionsAndLibraries) {
  ExecutableAbiVersionProto proto = CompleteProto();
  proto.mutable_musa_platform_version()
      ->mutable_numerical_options()
      ->set_fast_math(true);
  EXPECT_THAT(ValidationStatus(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("numerical options")));

  proto = CompleteProto();
  MusaVersion* musa = proto.mutable_musa_platform_version();
  auto* library = musa->add_required_optional_library_abis();
  library->set_name("mublas");
  library->set_abi_version("1");
  library->set_fingerprint("not-a-sha");
  EXPECT_THAT(ValidationStatus(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("optional library fingerprint")));
}

}  // namespace
}  // namespace stream_executor::musa
