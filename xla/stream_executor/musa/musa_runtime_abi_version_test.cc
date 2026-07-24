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

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/stream_executor/abi/runtime_abi_version_manager.h"
#include "xla/stream_executor/musa/musa_mubin.h"
#include "xla/stream_executor/musa/musa_runtime_abi_version.pb.h"
#include "xla/stream_executor/musa/musa_target_contract.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using MusaExecutableVersion = ExecutableAbiVersionProto::MusaPlatformVersion;

MusaRuntimeAbiVersion CurrentRuntime(
    std::vector<MusaOptionalLibraryAbi> libraries = {}) {
  return MusaRuntimeAbiVersion::Create(
             SemanticVersion(1, 5, 4), SemanticVersion(3, 0, 0),
             SemanticVersion(3, 0, 0), SemanticVersion(4, 0, 1),
             std::move(libraries))
      .value();
}

ExecutableAbiVersionProto CompatibleExecutableProto() {
  ExecutableAbiVersionProto proto;
  proto.set_platform_name("MUSA");
  MusaExecutableVersion* musa = proto.mutable_musa_platform_version();
  musa->set_envelope_version(kMusaExecutableAbiEnvelopeVersion);
  musa->set_binary_kind(kMusaExecutableBinaryKind);
  musa->set_mubin_loader_abi_version(kMubinLoaderAbiVersion);
  musa->set_target_triple(kMusaTargetTriple);
  musa->set_architecture(kS80TargetArchitecture);
  musa->set_target_features(kS80TargetFeatures);
  musa->set_data_layout(kMusaTargetDataLayout);
  musa->set_pointer_width(kMusaPointerWidth);
  musa->set_little_endian(kMusaIsLittleEndian);
  musa->set_hardware_warp_size(kS80HardwareWarpSize);
  musa->set_logical_subgroup_size(kS80CompilerLogicalSubgroupSize);
  musa->set_required_toolkit_version("4.0.1");
  musa->set_minimum_runtime_version("1.0.0");
  musa->set_maximum_runtime_version("2.0.0");
  musa->set_minimum_driver_version("2.0.0");
  musa->set_maximum_driver_version("4.0.0");
  musa->set_minimum_kernel_driver_version("3.0.0");
  musa->set_maximum_kernel_driver_version("3.0.0");
  return proto;
}

absl::Status CompatibilityStatus(const MusaRuntimeAbiVersion& runtime,
                                 const ExecutableAbiVersionProto& proto) {
  absl::StatusOr<ExecutableAbiVersion> executable =
      ExecutableAbiVersion::FromProto(proto);
  if (!executable.ok()) return executable.status();
  return runtime.IsCompatibleWith(*executable);
}

TEST(MusaRuntimeAbiVersionTest, CreatesSnapshotWithoutDeviceDiscovery) {
  ASSERT_OK_AND_ASSIGN(
      MusaRuntimeAbiVersion runtime,
      MusaRuntimeAbiVersion::CreateFromApiVersions(
          /*runtime_version=*/10504, /*driver_version=*/10504,
          SemanticVersion(3, 0, 0), /*toolkit_version=*/40001));
  EXPECT_EQ(runtime.runtime_version(), SemanticVersion(1, 5, 4));
  EXPECT_EQ(runtime.driver_version(), SemanticVersion(1, 5, 4));
  EXPECT_EQ(runtime.kernel_driver_version(), SemanticVersion(3, 0, 0));
  EXPECT_EQ(runtime.toolkit_version(), SemanticVersion(4, 0, 1));
}

TEST(MusaRuntimeAbiVersionTest, ApiSnapshotPreservesOptionalLibraries) {
  const std::vector<MusaOptionalLibraryAbi> libraries = {
      {kMusaMuBlasLibraryAbiName, kMusaMuBlasLibraryAbiVersion, ""}};
  ASSERT_OK_AND_ASSIGN(
      MusaRuntimeAbiVersion runtime,
      MusaRuntimeAbiVersion::CreateFromApiVersions(
          /*runtime_version=*/10504, /*driver_version=*/10504,
          SemanticVersion(3, 0, 0), /*toolkit_version=*/40001, libraries));
  EXPECT_EQ(runtime.available_optional_library_abis(), libraries);
}

TEST(MusaRuntimeAbiVersionTest, ProtoRoundTripPreservesRuntimeFacts) {
  const std::vector<MusaOptionalLibraryAbi> libraries = {
      {"mublas", "4.0", "mublas-fingerprint"},
      {"mufft", "2.1", "mufft-fingerprint"},
  };
  MusaRuntimeAbiVersion runtime = CurrentRuntime(libraries);

  ASSERT_OK_AND_ASSIGN(RuntimeAbiVersionProto outer, runtime.ToProto());
  EXPECT_EQ(outer.platform_name(), "MUSA");

  MusaRuntimeAbiVersionProto serialized;
  ASSERT_TRUE(serialized.ParseFromString(outer.platform_specific_version()));
  EXPECT_EQ(serialized.runtime_version(), "1.5.4");
  EXPECT_EQ(serialized.driver_version(), "3.0.0");
  EXPECT_EQ(serialized.kernel_driver_version(), "3.0.0");
  EXPECT_EQ(serialized.toolkit_version(), "4.0.1");
  ASSERT_EQ(serialized.available_optional_library_abis_size(), 2);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MusaRuntimeAbiVersion> restored,
                       MusaRuntimeAbiVersion::FromSerializedProto(
                           outer.platform_specific_version()));
  EXPECT_EQ(restored->runtime_version(), SemanticVersion(1, 5, 4));
  EXPECT_EQ(restored->driver_version(), SemanticVersion(3, 0, 0));
  EXPECT_EQ(restored->kernel_driver_version(), SemanticVersion(3, 0, 0));
  EXPECT_EQ(restored->toolkit_version(), SemanticVersion(4, 0, 1));
  EXPECT_EQ(restored->available_optional_library_abis(), libraries);
}

TEST(MusaRuntimeAbiVersionTest, RejectsUnknownRuntimeFactsAtConstruction) {
  const std::array<const char*, 4> fields = {
      "runtime_version", "driver_version", "kernel_driver_version",
      "toolkit_version"};
  for (int invalid = 0; invalid < fields.size(); ++invalid) {
    std::array<SemanticVersion, 4> versions = {
        SemanticVersion(1, 5, 4), SemanticVersion(3, 0, 0),
        SemanticVersion(3, 0, 0), SemanticVersion(4, 0, 1)};
    versions[invalid] = SemanticVersion(0, 0, 0);
    EXPECT_THAT(MusaRuntimeAbiVersion::Create(versions[0], versions[1],
                                              versions[2], versions[3]),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr(fields[invalid])));
  }
}

TEST(MusaRuntimeAbiVersionTest, RejectsMalformedRuntimeProto) {
  EXPECT_THAT(MusaRuntimeAbiVersion::FromSerializedProto("not a proto"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  MusaRuntimeAbiVersionProto proto;
  proto.set_runtime_version("not-a-version");
  proto.set_driver_version("3.0.0");
  proto.set_kernel_driver_version("3.0.0");
  proto.set_toolkit_version("4.0.1");
  EXPECT_THAT(MusaRuntimeAbiVersion::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("runtime_version")));

  proto.set_runtime_version("1.5.4");
  for (int invalid = 0; invalid < 4; ++invalid) {
    MusaRuntimeAbiVersionProto zero_proto = proto;
    switch (invalid) {
      case 0:
        zero_proto.set_runtime_version("0.0.0");
        break;
      case 1:
        zero_proto.set_driver_version("0.0.0");
        break;
      case 2:
        zero_proto.set_kernel_driver_version("0.0.0");
        break;
      case 3:
        zero_proto.set_toolkit_version("0.0.0");
        break;
    }
    EXPECT_THAT(MusaRuntimeAbiVersion::FromProto(zero_proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("must be known and nonzero")));
  }

  for (int i = 0; i < 2; ++i) {
    auto* library = proto.add_available_optional_library_abis();
    library->set_name("mublas");
    library->set_abi_version("4.0");
  }
  EXPECT_THAT(
      MusaRuntimeAbiVersion::FromProto(proto),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("duplicated")));
}

TEST(MusaRuntimeAbiVersionTest, RegisteredFactoryDeserializesMusa) {
  ASSERT_OK_AND_ASSIGN(RuntimeAbiVersionProto proto,
                       CurrentRuntime().ToProto());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RuntimeAbiVersion> restored,
      RuntimeAbiVersionManager::GetInstance().GetRuntimeAbiVersion(proto));
  ASSERT_OK_AND_ASSIGN(PlatformId platform_id, restored->platform_id());
  EXPECT_EQ(platform_id->ToName(), "MUSA");
}

TEST(MusaRuntimeAbiVersionTest, AcceptsQualifiedInclusiveRanges) {
  MusaRuntimeAbiVersion runtime = CurrentRuntime();
  ExecutableAbiVersionProto proto = CompatibleExecutableProto();
  EXPECT_OK(CompatibilityStatus(runtime, proto));

  MusaExecutableVersion* musa = proto.mutable_musa_platform_version();
  musa->set_minimum_runtime_version("1.5.4");
  musa->set_maximum_runtime_version("1.5.4");
  musa->set_minimum_driver_version("3.0.0");
  musa->set_maximum_driver_version("3.0.0");
  EXPECT_OK(CompatibilityStatus(runtime, proto));
}

TEST(MusaRuntimeAbiVersionTest, RejectsPlatformAndTargetMismatches) {
  MusaRuntimeAbiVersion runtime = CurrentRuntime();

  ExecutableAbiVersionProto proto = CompatibleExecutableProto();
  proto.set_platform_name("CUDA");
  EXPECT_THAT(
      CompatibilityStatus(runtime, proto),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("platform")));

  proto = CompatibleExecutableProto();
  proto.clear_musa_platform_version();
  EXPECT_THAT(
      CompatibilityStatus(runtime, proto),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("missing")));

  using Mutation = std::function<void(MusaExecutableVersion*)>;
  const std::vector<std::pair<std::string, Mutation>> mutations = {
      {"envelope",
       [](MusaExecutableVersion* musa) {
         musa->set_envelope_version(kMusaExecutableAbiEnvelopeVersion + 1);
       }},
      {"binary",
       [](MusaExecutableVersion* musa) { musa->set_binary_kind("cubin"); }},
      {"loader",
       [](MusaExecutableVersion* musa) {
         musa->set_mubin_loader_abi_version(kMubinLoaderAbiVersion + 1);
       }},
      {"triple",
       [](MusaExecutableVersion* musa) {
         musa->set_target_triple("nvptx64-nvidia-cuda");
       }},
      {"architecture",
       [](MusaExecutableVersion* musa) { musa->set_architecture("mp_22"); }},
      {"features",
       [](MusaExecutableVersion* musa) {
         musa->set_target_features("+unqualified");
       }},
      {"layout",
       [](MusaExecutableVersion* musa) { musa->set_data_layout("e-p:32:32"); }},
      {"pointer",
       [](MusaExecutableVersion* musa) { musa->set_pointer_width(32); }},
      {"endianness",
       [](MusaExecutableVersion* musa) { musa->set_little_endian(false); }},
      {"warp",
       [](MusaExecutableVersion* musa) { musa->set_hardware_warp_size(32); }},
      {"subgroup",
       [](MusaExecutableVersion* musa) {
         musa->set_logical_subgroup_size(128);
       }},
  };
  for (const auto& [name, mutate] : mutations) {
    SCOPED_TRACE(name);
    proto = CompatibleExecutableProto();
    mutate(proto.mutable_musa_platform_version());
    EXPECT_THAT(CompatibilityStatus(runtime, proto),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
}

TEST(MusaRuntimeAbiVersionTest, RejectsMalformedAndUnsatisfiedVersions) {
  MusaRuntimeAbiVersion runtime = CurrentRuntime();
  using Mutation = std::function<void(MusaExecutableVersion*)>;
  const std::vector<std::pair<std::string, Mutation>> mutations = {
      {"toolkit",
       [](MusaExecutableVersion* musa) {
         musa->set_required_toolkit_version("4.1.0");
       }},
      {"runtime too new",
       [](MusaExecutableVersion* musa) {
         musa->set_minimum_runtime_version("1.5.5");
       }},
      {"runtime too old",
       [](MusaExecutableVersion* musa) {
         musa->set_maximum_runtime_version("1.5.3");
       }},
      {"driver too new",
       [](MusaExecutableVersion* musa) {
         musa->set_minimum_driver_version("3.0.1");
       }},
      {"driver too old",
       [](MusaExecutableVersion* musa) {
         musa->set_maximum_driver_version("2.9.9");
       }},
      {"kernel too new",
       [](MusaExecutableVersion* musa) {
         musa->set_minimum_kernel_driver_version("3.0.1");
       }},
      {"kernel too old",
       [](MusaExecutableVersion* musa) {
         musa->set_maximum_kernel_driver_version("2.9.9");
       }},
      {"missing bound",
       [](MusaExecutableVersion* musa) {
         musa->clear_minimum_runtime_version();
       }},
      {"malformed bound",
       [](MusaExecutableVersion* musa) {
         musa->set_minimum_driver_version("unknown");
       }},
      {"inverted bound",
       [](MusaExecutableVersion* musa) {
         musa->set_minimum_runtime_version("2.0.0");
         musa->set_maximum_runtime_version("1.0.0");
       }},
      {"unknown toolkit",
       [](MusaExecutableVersion* musa) {
         musa->set_required_toolkit_version("0.0.0");
       }},
      {"unknown bound",
       [](MusaExecutableVersion* musa) {
         musa->set_minimum_driver_version("0.0.0");
       }},
  };
  for (const auto& [name, mutate] : mutations) {
    SCOPED_TRACE(name);
    ExecutableAbiVersionProto proto = CompatibleExecutableProto();
    mutate(proto.mutable_musa_platform_version());
    EXPECT_THAT(CompatibilityStatus(runtime, proto),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
}

TEST(MusaRuntimeAbiVersionTest, ChecksOnlyExplicitlyRequiredLibraries) {
  MusaRuntimeAbiVersion runtime = CurrentRuntime({
      {"mublas", "4.0", "mublas-sha"},
      {"unused", "9", "unused-sha"},
  });
  ExecutableAbiVersionProto proto = CompatibleExecutableProto();

  // Extra available libraries do not become implicit executable requirements.
  EXPECT_OK(CompatibilityStatus(runtime, proto));

  MusaExecutableVersion::OptionalLibraryAbi* required =
      proto.mutable_musa_platform_version()
          ->add_required_optional_library_abis();
  required->set_name("mublas");
  required->set_abi_version("4.0");
  required->set_fingerprint("mublas-sha");
  EXPECT_OK(CompatibilityStatus(runtime, proto));

  required->set_fingerprint("");
  EXPECT_OK(CompatibilityStatus(runtime, proto));

  required->set_abi_version("5.0");
  EXPECT_THAT(CompatibilityStatus(runtime, proto),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("ABI version")));

  required->set_abi_version("4.0");
  required->set_fingerprint("different-sha");
  EXPECT_THAT(CompatibilityStatus(runtime, proto),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("fingerprint")));

  required->set_name("musolver");
  EXPECT_THAT(CompatibilityStatus(runtime, proto),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("unavailable")));

  *proto.mutable_musa_platform_version()->add_required_optional_library_abis() =
      *required;
  EXPECT_THAT(
      CompatibilityStatus(runtime, proto),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("duplicated")));
}

}  // namespace
}  // namespace stream_executor::musa
