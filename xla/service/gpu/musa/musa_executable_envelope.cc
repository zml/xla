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

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_executable_abi.h"
#include "xla/stream_executor/musa/musa_mubin.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_target_contract.h"

namespace xla::gpu::musa {
namespace {

using MusaVersion =
    stream_executor::ExecutableAbiVersionProto::MusaPlatformVersion;

absl::Status InvalidEnvelope(absl::string_view reason) {
  return absl::InvalidArgumentError(
      absl::StrCat("invalid MUSA executable envelope: ", reason));
}

}  // namespace

absl::StatusOr<stream_executor::ExecutableAbiVersion>
BuildMusaExecutableEnvelope(
    const stream_executor::DeviceDescription& device_description,
    const MusaCompilationIdentity& identity,
    const MusaCompilationCapabilities& capabilities,
    const MusaCompilationOptions& compilation_options,
    absl::Span<const uint8_t> main_binary,
    absl::Span<const MusaOptionalLibraryAbi> required_optional_libraries) {
  if (!device_description.gpu_compute_capability().IsMusa()) {
    return InvalidEnvelope("device description is not MUSA");
  }
  const stream_executor::MusaComputeCapability& compute_capability =
      device_description.musa_compute_capability();
  if (compute_capability.architecture() !=
          stream_executor::musa::kS80TargetArchitecture ||
      compute_capability.major() != 2 || compute_capability.minor() != 1 ||
      compute_capability.hardware_warp_size() !=
          stream_executor::musa::kS80HardwareWarpSize ||
      compute_capability.logical_subgroup_size() !=
          stream_executor::musa::kS80CompilerLogicalSubgroupSize ||
      device_description.threads_per_warp() !=
          stream_executor::musa::kS80HardwareWarpSize ||
      device_description.device_address_bits() !=
          kMusaInterchangePointerWidth) {
    return InvalidEnvelope(
        "device capability does not match the qualified S80 mp_21 target");
  }
  absl::Status identity_status = ValidateMusaCompilationIdentity(identity);
  if (!identity_status.ok()) return identity_status;
  if (!capabilities.supports_compile) {
    return InvalidEnvelope("compiler provider does not support compilation");
  }
  if (!capabilities.vendor_llvm_isolated) {
    return InvalidEnvelope("vendor LLVM must remain isolated out of process");
  }
  if (capabilities.binary_kind != "mubin") {
    return InvalidEnvelope(
        absl::StrCat("compiler produced unsupported binary kind ",
                     capabilities.binary_kind));
  }
  if (capabilities.target != "mtgpu-mt-musa/mp_21") {
    return InvalidEnvelope(absl::StrCat("compiler selected unsupported target ",
                                        capabilities.target));
  }

  stream_executor::ExecutableAbiVersionProto proto;
  proto.set_platform_name(stream_executor::musa::kMusaPlatformId->ToName());
  MusaVersion* musa = proto.mutable_musa_platform_version();
  musa->set_envelope_version(kMusaExecutableEnvelopeVersion);
  musa->set_binary_kind(capabilities.binary_kind);
  musa->set_mubin_loader_abi_version(
      stream_executor::musa::kMubinLoaderAbiVersion);
  musa->set_main_binary_sha256(
      stream_executor::musa::MusaExecutableBinarySha256(main_binary));
  musa->set_target_triple(stream_executor::musa::kMusaTargetTriple);
  musa->set_architecture(stream_executor::musa::kS80TargetArchitecture);
  musa->set_target_features("none");
  musa->set_data_layout(stream_executor::musa::kMusaTargetDataLayout);
  musa->set_pointer_width(kMusaInterchangePointerWidth);
  musa->set_little_endian(kMusaInterchangeIsLittleEndian);

  musa->set_hardware_warp_size(compute_capability.hardware_warp_size());
  musa->set_logical_subgroup_size(compute_capability.logical_subgroup_size());
  musa->set_bridge_protocol_version(kMusaBridgeProtocolVersion);
  musa->set_shim_abi_version(kMusaShimAbiVersion);
  musa->set_shim_mapping_version(kMusaShimMappingVersion);
  musa->set_shim_mapping_sha256(kMusaShimMappingSha256);
  musa->set_llvm14_compatibility_revision(kMusaLlvm14CompatibilityRevision);

  musa->set_xla_revision(identity.xla_revision);
  musa->set_current_llvm_revision(identity.current_llvm_revision);
  musa->set_provider_name(identity.provider_name);
  musa->set_provider_fingerprint(identity.provider_fingerprint);
  musa->set_bridge_fingerprint(identity.bridge_fingerprint);
  musa->set_toolchain_fingerprint(identity.toolchain_fingerprint);
  musa->set_libdevice_fingerprint(identity.libdevice_fingerprint);
  musa->set_driver_compatibility(identity.driver_compatibility);
  musa->set_runtime_compatibility(identity.runtime_compatibility);

  MusaVersion::NumericalOptions* numerical = musa->mutable_numerical_options();
  numerical->set_optimization_level(compilation_options.optimization_level);
  numerical->set_emit_debug_information(
      compilation_options.emit_debug_information);
  numerical->set_deterministic(compilation_options.deterministic);
  numerical->set_fast_math(compilation_options.fast_math);
  numerical->set_flush_denormals_to_zero(
      compilation_options.flush_denormals_to_zero);
  numerical->set_finite_math_only(compilation_options.finite_math_only);
  numerical->set_unsafe_math_optimizations(
      compilation_options.unsafe_math_optimizations);
  numerical->set_no_signed_zeros(compilation_options.no_signed_zeros);
  numerical->set_allow_fp_contract(compilation_options.allow_fp_contract);

  const std::string toolkit =
      device_description.compile_time_toolkit_version().ToString();
  const std::string runtime = device_description.runtime_version().ToString();
  const std::string driver = device_description.driver_version().ToString();
  const std::string kernel_driver =
      device_description.kernel_mode_driver_version().ToString();
  if (!device_description.compile_time_toolkit_version().IsValid() ||
      !device_description.runtime_version().IsValid() ||
      !device_description.driver_version().IsValid() ||
      !device_description.kernel_mode_driver_version().IsValid()) {
    return InvalidEnvelope(
        "toolkit, runtime, driver API, and kernel-driver versions must all "
        "be known before serialization");
  }
  musa->set_required_toolkit_version(toolkit);
  musa->set_minimum_runtime_version(runtime);
  musa->set_maximum_runtime_version(runtime);
  musa->set_minimum_driver_version(driver);
  musa->set_maximum_driver_version(driver);
  musa->set_minimum_kernel_driver_version(kernel_driver);
  musa->set_maximum_kernel_driver_version(kernel_driver);

  for (const MusaOptionalLibraryAbi& library : required_optional_libraries) {
    MusaVersion::OptionalLibraryAbi* required =
        musa->add_required_optional_library_abis();
    required->set_name(library.name);
    required->set_abi_version(library.abi_version);
    required->set_fingerprint(library.fingerprint);
  }

  absl::StatusOr<stream_executor::ExecutableAbiVersion> result =
      stream_executor::ExecutableAbiVersion::FromProto(std::move(proto));
  if (!result.ok()) return result.status();
  absl::Status validation =
      ValidateMusaExecutableEnvelope(*result, main_binary);
  if (!validation.ok()) return validation;
  return result;
}

absl::Status ValidateMusaExecutableEnvelope(
    const stream_executor::ExecutableAbiVersion& executable_abi_version,
    absl::Span<const uint8_t> main_binary) {
  absl::Status loader_validation =
      stream_executor::musa::ValidateMusaExecutableAbi(executable_abi_version,
                                                       main_binary);
  if (!loader_validation.ok()) return loader_validation;

  const stream_executor::ExecutableAbiVersionProto& proto =
      executable_abi_version.proto();
  const MusaVersion& musa = proto.musa_platform_version();
  if (musa.bridge_protocol_version() != kMusaBridgeProtocolVersion ||
      musa.shim_abi_version() != kMusaShimAbiVersion ||
      musa.shim_mapping_version() != kMusaShimMappingVersion ||
      musa.shim_mapping_sha256() != kMusaShimMappingSha256 ||
      musa.llvm14_compatibility_revision() !=
          kMusaLlvm14CompatibilityRevision) {
    return InvalidEnvelope("compiler bridge or shim identity does not match");
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu::musa
