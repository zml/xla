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

#ifndef XLA_TOOLS_MUSA_LLVM_BRIDGE_BRIDGE_CORE_H_
#define XLA_TOOLS_MUSA_LLVM_BRIDGE_BRIDGE_CORE_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "xla/service/gpu/musa/protocol.pb.h"

namespace xla::gpu::musa::bridge {

// A vendor-LLVM-14-native module serialized after the complete mapping-v1
// translation. Keeping this interface textual prevents vendor LLVM C++ types
// from escaping the isolated bridge process or leaking into the PJRT plugin.
struct VendorLlvmModule {
  std::string llvm_ir;
  uint32_t translated_shim_calls;
  uint32_t kernel_count;
};

// Returns the canonical path of the shared object that supplies the vendor
// LLVM parser used by this process. The bridge compares it with the measured
// `--libclang-cpp` component before accepting a request.
absl::StatusOr<std::string> LoadedVendorLlvmSharedObjectPath();

// Parses and verifies the normalized interchange module with the MUSA SDK's
// LLVM 14, validates mapping-v1 again inside the isolated process, replaces
// each XLA shim call structurally with its registered vendor intrinsic, and
// installs the native MUSA kernel ABI. The translated module is verified a
// second time before it is returned.
absl::StatusOr<VendorLlvmModule> TranslateMusaBridgeRequestToVendorLlvm(
    const MusaBridgeCompileRequest& request);

}  // namespace xla::gpu::musa::bridge

#endif  // XLA_TOOLS_MUSA_LLVM_BRIDGE_BRIDGE_CORE_H_
