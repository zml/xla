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

#ifndef XLA_SERVICE_GPU_MUSA_MUSA_BRIDGE_IR_VALIDATOR_H_
#define XLA_SERVICE_GPU_MUSA_MUSA_BRIDGE_IR_VALIDATOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"

namespace xla::gpu::musa {

inline constexpr uint64_t kMaxMusaInterchangeIrBytes = kMusaBridgeMaxLlvmBytes;
inline constexpr uint64_t kMaxMusaKernelEntries = kMusaBridgeMaxKernelCount;
inline constexpr uint64_t kMaxMusaExportedGlobals =
    kMusaBridgeMaxExportedGlobalCount;
inline constexpr uint64_t kMaxMusaSymbolBytes = kMusaBridgeMaxSymbolNameBytes;

enum class MusaExportedGlobalKind : uint8_t { kMutable, kConstant };

struct MusaExportedGlobal {
  std::string name;
  MusaExportedGlobalKind kind;
  uint32_t address_space;
  uint64_t size;
  uint64_t alignment;
};

struct MusaBridgeIrMetadata {
  uint32_t protocol_version = kMusaBridgeProtocolVersion;
  uint32_t shim_abi_version = kMusaShimAbiVersion;
  uint32_t mapping_version = kMusaShimMappingVersion;
  std::string module_name;
  std::string architecture = kMusaTargetArchitecture;
  std::vector<std::string> kernel_entry_names;
  std::vector<MusaExportedGlobal> exported_globals;
};

// Parses and structurally validates the current-LLVM textual interchange.
// This is a fail-closed contract validator, not the LLVM-14 normalization pass
// added by C08. It never invokes vendor code and never returns the IR in an
// error message.
absl::Status ValidateMusaBridgeIr(absl::string_view llvm_ir,
                                  const MusaBridgeIrMetadata& metadata);

// Validates both the canonical wire fields and their structural relationship
// to the normalized LLVM module. C06 providers must call this composed entry
// point before invoking the vendor process.
absl::Status ValidateMusaBridgeCompileRequestIr(
    const MusaBridgeCompileRequest& request);

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_MUSA_BRIDGE_IR_VALIDATOR_H_
