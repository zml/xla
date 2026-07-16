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

#include "xla/stream_executor/musa/musa_device_description.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_device_properties.h"
#include "xla/stream_executor/musa/musa_target_contract.h"
#include "xla/stream_executor/musa/musa_version_parser.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::musa {
namespace {

absl::Status ValidatePositive(int64_t value, const char* name) {
  if (value <= 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("MUSA device reported invalid ", name, ": ", value));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> MusaArchitectureFromComputeCapability(int major,
                                                                  int minor) {
  if (major != kS80ComputeCapabilityMajor ||
      minor != kS80ComputeCapabilityMinor) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unqualified MUSA compute capability ", major, ".", minor,
                     "; C01 supports only S80 mp_21"));
  }
  return "mp_21";
}

absl::StatusOr<DeviceDescription> BuildMusaDeviceDescription(
    const MusaDeviceProperties& properties,
    const MusaDeviceVersions& versions) {
  if (properties.name != kS80DeviceName) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unqualified MUSA device '", properties.name,
                     "'; C01 supports only ", kS80DeviceName));
  }
  if (properties.pci_bus_id.empty()) {
    return absl::FailedPreconditionError("MUSA PCI bus ID is empty");
  }
  RETURN_IF_ERROR(
      ValidatePositive(properties.total_memory_bytes, "total memory"));
  RETURN_IF_ERROR(ValidatePositive(properties.max_threads_per_block,
                                   "maximum threads per block"));
  RETURN_IF_ERROR(ValidatePositive(properties.max_block_dim_x,
                                   "maximum block dimension X"));
  RETURN_IF_ERROR(ValidatePositive(properties.max_block_dim_y,
                                   "maximum block dimension Y"));
  RETURN_IF_ERROR(ValidatePositive(properties.max_block_dim_z,
                                   "maximum block dimension Z"));
  RETURN_IF_ERROR(
      ValidatePositive(properties.max_grid_dim_x, "maximum grid dimension X"));
  RETURN_IF_ERROR(
      ValidatePositive(properties.max_grid_dim_y, "maximum grid dimension Y"));
  RETURN_IF_ERROR(
      ValidatePositive(properties.max_grid_dim_z, "maximum grid dimension Z"));
  if (properties.hardware_warp_size != kS80HardwareWarpSize) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unqualified MUSA hardware warp size: ", properties.hardware_warp_size,
        "; expected ", kS80HardwareWarpSize, " for S80 mp_21"));
  }
  RETURN_IF_ERROR(ValidatePositive(properties.multiprocessor_count,
                                   "multiprocessor count"));
  RETURN_IF_ERROR(ValidatePositive(properties.max_threads_per_multiprocessor,
                                   "maximum threads per multiprocessor"));
  RETURN_IF_ERROR(ValidatePositive(properties.max_shared_memory_per_block,
                                   "shared memory per block"));
  RETURN_IF_ERROR(
      ValidatePositive(properties.max_shared_memory_per_multiprocessor,
                       "shared memory per multiprocessor"));
  RETURN_IF_ERROR(ValidatePositive(properties.max_registers_per_block,
                                   "registers per block"));
  RETURN_IF_ERROR(ValidatePositive(properties.max_registers_per_multiprocessor,
                                   "registers per multiprocessor"));
  RETURN_IF_ERROR(
      ValidatePositive(properties.clock_rate_khz, "core clock rate"));
  RETURN_IF_ERROR(
      ValidatePositive(properties.memory_clock_rate_khz, "memory clock rate"));
  RETURN_IF_ERROR(
      ValidatePositive(properties.memory_bus_width_bits, "memory bus width"));
  RETURN_IF_ERROR(
      ValidatePositive(properties.l2_cache_size_bytes, "L2 cache size"));
  RETURN_IF_ERROR(ValidatePositive(properties.texture_alignment_bytes,
                                   "texture alignment"));

  ASSIGN_OR_RETURN(std::string architecture,
                   MusaArchitectureFromComputeCapability(
                       properties.compute_capability_major,
                       properties.compute_capability_minor));
  ASSIGN_OR_RETURN(SemanticVersion runtime_version,
                   ParseMusaVersion(versions.runtime_api));
  ASSIGN_OR_RETURN(SemanticVersion driver_version,
                   ParseMusaVersion(versions.driver_api));
  if (versions.compile_time_toolkit != kQualifiedMusaToolkitVersion) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unqualified MUSA toolkit version ", versions.compile_time_toolkit,
        "; C01 supports only 4.0.1 (raw 40001)"));
  }
  ASSIGN_OR_RETURN(SemanticVersion toolkit_version,
                   ParseMusaVersion(versions.compile_time_toolkit));

  DeviceDescription description;
  description.set_device_vendor("Moore Threads");
  description.set_name(properties.name);
  description.set_pci_bus_id(properties.pci_bus_id);
  description.set_platform_version(absl::StrFormat(
      "MUSA Compute Capability %d.%d (%s)", properties.compute_capability_major,
      properties.compute_capability_minor, architecture));
  description.set_gpu_compute_capability(
      GpuComputeCapability(MusaComputeCapability(
          architecture, properties.compute_capability_major,
          properties.compute_capability_minor, properties.hardware_warp_size,
          kS80CompilerLogicalSubgroupSize)));

  description.set_thread_dim_limit(ThreadDim(properties.max_block_dim_x,
                                             properties.max_block_dim_y,
                                             properties.max_block_dim_z));
  description.set_block_dim_limit(BlockDim(properties.max_grid_dim_x,
                                           properties.max_grid_dim_y,
                                           properties.max_grid_dim_z));
  description.set_threads_per_block_limit(properties.max_threads_per_block);
  description.set_threads_per_core_limit(
      properties.max_threads_per_multiprocessor);
  // Preserve the generic DeviceDescription contract: this is the hardware
  // warp/wavefront width. MUSA lowering must read the distinct logical subgroup
  // width from MusaComputeCapability.
  description.set_threads_per_warp(properties.hardware_warp_size);
  description.set_registers_per_block_limit(properties.max_registers_per_block);
  description.set_registers_per_core_limit(
      properties.max_registers_per_multiprocessor);
  description.set_shared_memory_per_block(
      properties.max_shared_memory_per_block);
  description.set_shared_memory_per_core(
      properties.max_shared_memory_per_multiprocessor);
  description.set_shared_memory_per_block_optin(
      properties.max_shared_memory_per_block_optin);
  description.set_reserved_shared_memory_per_block(
      properties.reserved_shared_memory_per_block);
  // The runtime reports a raw value for maximum resident blocks, but its
  // precise scheduling semantics have not yet been independently qualified.
  // Keep the shared scheduling field unknown until that qualification exists.
  description.set_core_count(properties.multiprocessor_count);
  description.set_device_address_bits(64);
  description.set_device_memory_size(properties.total_memory_bytes);
  description.set_l2_cache_size(properties.l2_cache_size_bytes);
  description.set_clock_rate_ghz(static_cast<float>(properties.clock_rate_khz) /
                                 1.0e6f);
  description.set_mem_clock_ghz(
      static_cast<float>(properties.memory_clock_rate_khz) / 1.0e6f);
  description.set_memory_bandwidth(
      2 * int64_t{properties.memory_clock_rate_khz} * 1000 *
      int64_t{properties.memory_bus_width_bits} / 8);
  description.set_ecc_enabled(properties.ecc_enabled);
  description.set_runtime_version(runtime_version);
  description.set_driver_version(driver_version);
  description.set_compile_time_toolkit_version(toolkit_version);
  if (versions.kernel_mode_driver.has_value()) {
    description.set_kernel_mode_driver_version(*versions.kernel_mode_driver);
  }
  description.set_model_str(absl::StrFormat(
      "%s with %dB RAM, %d multiprocessors, %dKHz clock, %dKHz memory "
      "clock, %dB L2",
      architecture, properties.total_memory_bytes,
      properties.multiprocessor_count, properties.clock_rate_khz,
      properties.memory_clock_rate_khz, properties.l2_cache_size_bytes));

  return description;
}

}  // namespace stream_executor::musa
