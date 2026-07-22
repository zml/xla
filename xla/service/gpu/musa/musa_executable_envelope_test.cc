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

#include "xla/service/gpu/musa/musa_executable_envelope.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_executable_abi.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

stream_executor::DeviceDescription S80Device() {
  stream_executor::DeviceDescription device;
  device.set_gpu_compute_capability(stream_executor::GpuComputeCapability(
      stream_executor::MusaComputeCapability("mp_21", 2, 1,
                                             /*hardware_warp_size=*/128,
                                             /*logical_subgroup_size=*/32)));
  device.set_threads_per_warp(128);
  device.set_device_address_bits(64);
  device.set_runtime_version(stream_executor::SemanticVersion(1, 5, 4));
  device.set_driver_version(stream_executor::SemanticVersion(1, 5, 4));
  device.set_kernel_mode_driver_version(
      stream_executor::SemanticVersion(3, 0, 0));
  device.set_compile_time_toolkit_version(
      stream_executor::SemanticVersion(4, 0, 1));
  return device;
}

MusaCompilationIdentity Identity() {
  return MusaCompilationIdentity{
      .xla_revision = "xla-test-revision",
      .current_llvm_revision = "llvm-test-revision",
      .provider_name = "subprocess",
      .provider_fingerprint = std::string(64, '1'),
      .bridge_fingerprint = std::string(64, '2'),
      .toolchain_fingerprint = std::string(64, '3'),
      .libdevice_fingerprint = std::string(64, '4'),
      .driver_compatibility = "musa-driver-3.0-compatible",
      .runtime_compatibility = "musa-runtime-4.0.1-compatible",
  };
}

absl::StatusOr<stream_executor::ExecutableAbiVersion> EmptyEnvelope() {
  return BuildMusaExecutableEnvelope(
      S80Device(), Identity(), MusaCompilationCapabilities(),
      MusaCompilationOptions(), /*main_binary=*/{});
}

TEST(MusaExecutableEnvelopeTest, BuildsCompleteDeterministicIdentity) {
  ASSERT_OK_AND_ASSIGN(stream_executor::ExecutableAbiVersion first,
                       EmptyEnvelope());
  ASSERT_OK_AND_ASSIGN(stream_executor::ExecutableAbiVersion second,
                       EmptyEnvelope());
  EXPECT_EQ(first.proto().SerializeAsString(),
            second.proto().SerializeAsString());

  const auto& musa = first.proto().musa_platform_version();
  EXPECT_EQ(musa.envelope_version(), kMusaExecutableEnvelopeVersion);
  EXPECT_EQ(musa.binary_kind(), "mubin");
  EXPECT_EQ(musa.architecture(), "mp_21");
  EXPECT_EQ(musa.hardware_warp_size(), 128);
  EXPECT_EQ(musa.logical_subgroup_size(), 32);
  EXPECT_EQ(musa.required_optional_library_abis_size(), 0);
  EXPECT_THAT(ValidateMusaExecutableEnvelope(first, {}), IsOk());
}

TEST(MusaExecutableEnvelopeTest, RejectsDeviceFactsOutsideQualifiedS80) {
  const std::vector<stream_executor::MusaComputeCapability> unsupported = {
      {"mp_22", 2, 1, 128, 32}, {"mp_21", 3, 1, 128, 32},
      {"mp_21", 2, 0, 128, 32}, {"mp_21", 2, 1, 64, 32},
      {"mp_21", 2, 1, 128, 16},
  };
  for (const stream_executor::MusaComputeCapability& capability : unsupported) {
    SCOPED_TRACE(capability.architecture());
    stream_executor::DeviceDescription device = S80Device();
    device.set_gpu_compute_capability(
        stream_executor::GpuComputeCapability(capability));
    EXPECT_THAT(BuildMusaExecutableEnvelope(
                    device, Identity(), MusaCompilationCapabilities(),
                    MusaCompilationOptions(), /*main_binary=*/{}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("qualified S80")));
  }

  stream_executor::DeviceDescription device = S80Device();
  device.set_threads_per_warp(32);
  EXPECT_THAT(
      BuildMusaExecutableEnvelope(device, Identity(),
                                  MusaCompilationCapabilities(),
                                  MusaCompilationOptions(), /*main_binary=*/{}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("qualified S80")));

  device = S80Device();
  device.set_device_address_bits(32);
  EXPECT_THAT(
      BuildMusaExecutableEnvelope(device, Identity(),
                                  MusaCompilationCapabilities(),
                                  MusaCompilationOptions(), /*main_binary=*/{}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("qualified S80")));
}

TEST(MusaExecutableEnvelopeTest,
     RequiresCanonicalIdentityAndIsolatedCompileProvider) {
  MusaCompilationIdentity identity = Identity();
  identity.provider_fingerprint = "not-a-sha256";
  EXPECT_THAT(
      BuildMusaExecutableEnvelope(S80Device(), identity,
                                  MusaCompilationCapabilities(),
                                  MusaCompilationOptions(), /*main_binary=*/{}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("fingerprints")));

  MusaCompilationCapabilities capabilities;
  capabilities.supports_compile = false;
  EXPECT_THAT(
      BuildMusaExecutableEnvelope(S80Device(), Identity(), capabilities,
                                  MusaCompilationOptions(), /*main_binary=*/{}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("does not support compilation")));

  capabilities = MusaCompilationCapabilities();
  capabilities.vendor_llvm_isolated = false;
  EXPECT_THAT(
      BuildMusaExecutableEnvelope(S80Device(), Identity(), capabilities,
                                  MusaCompilationOptions(), /*main_binary=*/{}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("isolated out of process")));
}

TEST(MusaExecutableEnvelopeTest, RejectsVersionDigestAndShimMutations) {
  ASSERT_OK_AND_ASSIGN(stream_executor::ExecutableAbiVersion envelope,
                       EmptyEnvelope());
  for (int mutation = 0; mutation < 3; ++mutation) {
    auto proto = envelope.proto();
    if (mutation == 0) {
      proto.mutable_musa_platform_version()->set_envelope_version(2);
    } else if (mutation == 1) {
      proto.mutable_musa_platform_version()->set_main_binary_sha256(
          std::string(64, 'f'));
    } else {
      proto.mutable_musa_platform_version()->set_shim_mapping_version(99);
    }
    ASSERT_OK_AND_ASSIGN(
        auto changed,
        stream_executor::ExecutableAbiVersion::FromProto(std::move(proto)));
    EXPECT_THAT(ValidateMusaExecutableEnvelope(changed, {}),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(MusaExecutableEnvelopeTest,
     CompilerRejectsProductionConstantDriftAcceptedByLoader) {
  ASSERT_OK_AND_ASSIGN(stream_executor::ExecutableAbiVersion envelope,
                       EmptyEnvelope());
  auto proto = envelope.proto();
  auto* musa = proto.mutable_musa_platform_version();
  musa->set_bridge_protocol_version(musa->bridge_protocol_version() + 1);
  musa->set_shim_abi_version(musa->shim_abi_version() + 1);
  musa->set_shim_mapping_version(musa->shim_mapping_version() + 1);
  musa->set_shim_mapping_sha256(std::string(64, 'a'));
  musa->set_llvm14_compatibility_revision("future-compatible-revision");
  ASSERT_OK_AND_ASSIGN(
      auto changed,
      stream_executor::ExecutableAbiVersion::FromProto(std::move(proto)));

  EXPECT_THAT(stream_executor::musa::ValidateMusaExecutableAbi(changed, {}),
              IsOk());
  EXPECT_THAT(ValidateMusaExecutableEnvelope(changed, {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("compiler bridge or shim identity")));
}

TEST(MusaExecutableEnvelopeTest, RejectsUnqualifiedNumericalProfile) {
  ASSERT_OK_AND_ASSIGN(stream_executor::ExecutableAbiVersion envelope,
                       EmptyEnvelope());
  auto proto = envelope.proto();
  proto.mutable_musa_platform_version()
      ->mutable_numerical_options()
      ->set_fast_math(true);
  ASSERT_OK_AND_ASSIGN(
      auto changed,
      stream_executor::ExecutableAbiVersion::FromProto(std::move(proto)));
  EXPECT_THAT(ValidateMusaExecutableEnvelope(changed, {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("numerical options")));
}

TEST(MusaExecutableEnvelopeTest, RequiresCanonicalOptionalLibraries) {
  std::vector<MusaOptionalLibraryAbi> libraries = {
      {.name = "musolver", .abi_version = "1", .fingerprint = {}},
      {.name = "mublas", .abi_version = "1", .fingerprint = {}},
  };
  EXPECT_THAT(BuildMusaExecutableEnvelope(
                  S80Device(), Identity(), MusaCompilationCapabilities(),
                  MusaCompilationOptions(), /*main_binary=*/{}, libraries),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("canonically sorted")));
}

}  // namespace
}  // namespace xla::gpu::musa
