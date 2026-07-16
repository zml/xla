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

// Measured MUBIN/kernel ABI facts. The textual compiler bridge ABI is separate
// and intentionally deferred to C05.
inline constexpr int kMusaKernelCallingConvention = 102;
inline constexpr int kMubinElfMachine = 253;

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_TARGET_CONTRACT_H_
