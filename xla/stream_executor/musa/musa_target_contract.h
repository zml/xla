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
inline constexpr char kMusaTargetDataLayout[] =
    "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:"
    "32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128";

// Measured MUBIN/kernel ABI facts. The versioned textual compiler boundary is
// defined separately in xla/service/gpu/musa/musa_shim_abi.h.
inline constexpr int kMusaKernelCallingConvention = 102;
inline constexpr int kMubinElfMachine = 253;

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_TARGET_CONTRACT_H_
