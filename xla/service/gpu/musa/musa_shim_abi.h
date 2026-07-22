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

#ifndef XLA_SERVICE_GPU_MUSA_MUSA_SHIM_ABI_H_
#define XLA_SERVICE_GPU_MUSA_MUSA_SHIM_ABI_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/musa/musa_target_contract.h"

namespace xla::gpu::musa {

inline constexpr uint32_t kMusaBridgeProtocolVersion = 1;
inline constexpr uint32_t kMusaShimAbiVersion = 1;
inline constexpr uint32_t kMusaShimMappingVersion = 3;
inline constexpr uint32_t kMusaInterchangePointerWidth = 64;
inline constexpr bool kMusaInterchangeIsLittleEndian = true;
// Mapping v3 admits exactly one atomic instruction shape. Both bridge
// validators independently enforce the rest of the contract (scalar i32,
// strong, non-volatile, system scope, monotonic/monotonic).
inline constexpr uint32_t kMusaAtomicCmpXchgAddressSpace = 1;
inline constexpr uint32_t kMusaAtomicCmpXchgBitWidth = 32;
inline constexpr uint32_t kMusaAtomicCmpXchgAlignment = 4;
inline constexpr char kMusaShimSymbolPrefix[] = "__xla_musa_v1_";
// Current LLVM marks qualified entry definitions with this string attribute.
// The C08 compatibility boundary consumes it before textual serialization;
// vendor LLVM must never observe it as an unversioned target attribute.
inline constexpr char kMusaLlvmKernelMarker[] = "xla.musa.kernel.v1";
inline constexpr auto& kMusaTargetTriple =
    stream_executor::musa::kMusaTargetTriple;
inline constexpr auto& kMusaTargetArchitecture =
    stream_executor::musa::kS80TargetArchitecture;
inline constexpr auto& kMusaDataLayout =
    stream_executor::musa::kMusaTargetDataLayout;

// A checked-in digest makes changes to the canonical table an explicit ABI
// review event. musa_shim_abi_test recomputes it from MusaShimCanonicalText().
inline constexpr char kMusaShimMappingSha256[] =
    "ef5630e3dc4fef8f23b650aa8b92dbcd0838a859d598c4b54d406c743676dd64";

enum class MusaShimId : uint8_t {
#define MUSA_SHIM(id, ...) id,
#include "xla/service/gpu/musa/musa_shim_abi.def"
#undef MUSA_SHIM
};

enum class MusaShimSignature : uint8_t {
  kVoidVoid,
  kI32Void,
  kI64Void,
  kI32I32I32,
};

enum class MusaMemoryEffects : uint8_t {
  kNone,
  kReadWrite,
  kInaccessibleRead,
  kInaccessibleReadWrite,
};

enum MusaShimAttribute : uint8_t {
  kNoAttributes = 0,
  kNoUnwind = 1 << 0,
  kWillReturn = 1 << 1,
};

struct MusaShimSpec {
  MusaShimId id;
  absl::string_view xla_symbol;
  absl::string_view vendor_intrinsic;
  MusaShimSignature signature;
  MusaMemoryEffects memory_effects;
  uint8_t required_attributes;
  bool convergent;
  uint32_t minimum_mapping_version;
};

enum class MusaAddressSpaceKind : uint8_t {
  kGeneric,
  kGlobal,
  kConstant,
  kWorkgroup,
  kPrivate,
  kVendorInternal,
};

struct MusaAddressSpaceSpec {
  MusaAddressSpaceKind kind;
  uint32_t number;
  uint32_t pointer_width;
  absl::string_view name;
  bool allowed_in_interchange;
};

// These categories are deliberately named and rejected by the active mapping.
// Adding one requires compiler-source and end-to-end probes, then a mapping
// version change. They must never fall through to another GPU target.
struct MusaUnsupportedCapability {
  absl::string_view name;
  absl::string_view reason;
};

absl::Span<const MusaShimSpec> MusaShimSpecs();
absl::Span<const MusaAddressSpaceSpec> MusaAddressSpaceSpecs();
absl::Span<const MusaUnsupportedCapability> MusaUnsupportedCapabilities();

const MusaShimSpec* FindMusaShim(absl::string_view symbol);
const MusaAddressSpaceSpec* FindMusaAddressSpace(uint32_t address_space);

absl::Status ValidateMusaShimTable();
std::string MusaShimCanonicalText();
absl::string_view MusaShimSignatureText(MusaShimSignature signature);
absl::string_view MusaMemoryEffectsText(MusaMemoryEffects effects);

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_MUSA_SHIM_ABI_H_
