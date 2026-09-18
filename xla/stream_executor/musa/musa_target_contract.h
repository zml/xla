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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_TARGET_CONTRACT_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_TARGET_CONTRACT_H_

#include <cstdint>
#include <string_view>

namespace stream_executor::musa {

// C01 qualifies exactly this initial hardware/toolkit contract. These values
// must not be treated as defaults for a future MUSA architecture or SDK.
inline constexpr char kS80DeviceName[] = "MTT S80";
inline constexpr int kS80ComputeCapabilityMajor = 2;
inline constexpr int kS80ComputeCapabilityMinor = 1;
inline constexpr int kS80HardwareWarpSize = 128;
// Vendor LLVM uses 32 as its compiler-visible subgroup ABI input for mp_21.
// Subgroup-operation semantics remain gated on the C07/C08 conformance probes.
inline constexpr int kS80CompilerLogicalSubgroupSize = 32;
inline constexpr int kQualifiedMusaToolkitVersion = 40001;  // 4.0.1

// S4000 keeps the 128-thread hardware warp and 32-lane compiler-visible
// subgroup used by the bridge, but has a distinct device architecture and SDK
// contract.
inline constexpr char kS4000DeviceName[] = "MTT S4000";
inline constexpr int kS4000ComputeCapabilityMajor = 2;
inline constexpr int kS4000ComputeCapabilityMinor = 2;
inline constexpr int kS4000HardwareWarpSize = 128;
inline constexpr int kS4000CompilerLogicalSubgroupSize = 32;
inline constexpr int kS4000QualifiedMusaToolkitVersion = 50100;  // 5.1.0
// The qualified S4000 advertises 25 TFLOP/s FP32. The live 56-MP part at its
// reported 1.791 GHz maximum clock reaches 25.68 TFLOP/s with 128 scalar FPUs
// per MP (counting an FMA as two operations), matching the published rating.
inline constexpr int kS4000FpusPerCore = 128;

// StreamExecutor-owned executable envelope facts. Compiler bridge and shim
// identities are deliberately not part of this contract: they are validated
// by the compiler service that produces the executable.
inline constexpr uint32_t kMusaExecutableAbiEnvelopeVersion = 1;
inline constexpr char kMusaExecutableBinaryKind[] = "mubin";
inline constexpr char kS80TargetFeatures[] = "none";
inline constexpr uint32_t kMusaPointerWidth = 64;
inline constexpr bool kMusaIsLittleEndian = true;

// Compiler identities shared by the current-LLVM interchange validator and
// the isolated vendor-LLVM bridge. The data layout is installed before any
// pointer-size or alignment-sensitive transformation.
inline constexpr char kMusaTargetTriple[] = "mtgpu-mt-musa";
inline constexpr char kS80TargetArchitecture[] = "mp_21";
inline constexpr char kS4000TargetArchitecture[] = "mp_22";
inline constexpr char kMusaTargetDataLayout[] =
    "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:"
    "32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128";

// Measured MUBIN/kernel ABI facts. The versioned textual compiler boundary is
// defined separately in xla/service/gpu/musa/musa_shim_abi.h.
inline constexpr int kMusaKernelCallingConvention = 102;
inline constexpr int kMubinElfMachine = 253;

inline constexpr bool IsQualifiedMusaArchitecture(
    std::string_view architecture) {
  return architecture == kS80TargetArchitecture ||
         architecture == kS4000TargetArchitecture;
}

inline constexpr std::string_view QualifiedMusaDeviceName(
    std::string_view architecture) {
  return architecture == kS80TargetArchitecture
             ? std::string_view(kS80DeviceName)
         : architecture == kS4000TargetArchitecture
             ? std::string_view(kS4000DeviceName)
             : std::string_view();
}

inline constexpr int QualifiedMusaToolkitVersion(
    std::string_view architecture) {
  return architecture == kS80TargetArchitecture
             ? kQualifiedMusaToolkitVersion
         : architecture == kS4000TargetArchitecture
             ? kS4000QualifiedMusaToolkitVersion
             : 0;
}

inline constexpr int QualifiedMusaHardwareWarpSize(
    std::string_view architecture) {
  return architecture == kS80TargetArchitecture
             ? kS80HardwareWarpSize
         : architecture == kS4000TargetArchitecture
             ? kS4000HardwareWarpSize
             : 0;
}

inline constexpr int QualifiedMusaLogicalSubgroupSize(
    std::string_view architecture) {
  return architecture == kS80TargetArchitecture
             ? kS80CompilerLogicalSubgroupSize
         : architecture == kS4000TargetArchitecture
             ? kS4000CompilerLogicalSubgroupSize
             : 0;
}

inline constexpr bool IsQualifiedMusaTargetContract(
    std::string_view architecture, int major, int minor,
    int hardware_warp_size, int logical_subgroup_size) {
  return (architecture == kS80TargetArchitecture &&
          major == kS80ComputeCapabilityMajor &&
          minor == kS80ComputeCapabilityMinor &&
          hardware_warp_size == kS80HardwareWarpSize &&
          logical_subgroup_size == kS80CompilerLogicalSubgroupSize) ||
         (architecture == kS4000TargetArchitecture &&
          major == kS4000ComputeCapabilityMajor &&
          minor == kS4000ComputeCapabilityMinor &&
          hardware_warp_size == kS4000HardwareWarpSize &&
          logical_subgroup_size == kS4000CompilerLogicalSubgroupSize);
}

inline constexpr bool IsQualifiedMusaAbiTarget(
    std::string_view architecture, int hardware_warp_size,
    int logical_subgroup_size) {
  return IsQualifiedMusaArchitecture(architecture) &&
         hardware_warp_size == QualifiedMusaHardwareWarpSize(architecture) &&
         logical_subgroup_size ==
             QualifiedMusaLogicalSubgroupSize(architecture);
}

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_TARGET_CONTRACT_H_
