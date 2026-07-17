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

#include "xla/service/gpu/musa/musa_shim_abi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace xla::gpu::musa {
namespace {

constexpr MusaShimSpec kShimSpecs[] = {
#define MUSA_SHIM(id, symbol, intrinsic, signature, memory, attributes, \
                  convergent, mapping_version)                          \
  {MusaShimId::id,                                                      \
   symbol,                                                              \
   intrinsic,                                                           \
   MusaShimSignature::signature,                                        \
   MusaMemoryEffects::memory,                                           \
   static_cast<uint8_t>(attributes),                                    \
   convergent,                                                          \
   mapping_version},
#include "xla/service/gpu/musa/musa_shim_abi.def"
#undef MUSA_SHIM
};

constexpr MusaAddressSpaceSpec kAddressSpaceSpecs[] = {
    {MusaAddressSpaceKind::kGeneric, 0, 64, "generic", true},
    {MusaAddressSpaceKind::kGlobal, 1, 64, "global", true},
    {MusaAddressSpaceKind::kConstant, 2, 64, "constant", true},
    {MusaAddressSpaceKind::kWorkgroup, 3, 32, "workgroup", true},
    {MusaAddressSpaceKind::kVendorInternal, 4, 32, "reserved", false},
    {MusaAddressSpaceKind::kPrivate, 5, 64, "private_scratch", true},
};

constexpr MusaUnsupportedCapability kUnsupportedCapabilities[] = {
    {"atomics",
     "ordering, scope, type, and address-space mappings require C06 probes"},
    {"non_generic_math",
     "device-library and numerical behavior require C06/C08 probes"},
    {"subgroup_barrier",
     "the S80 logical subgroup ABI requires C06/C07 conformance"},
    {"subgroup_vote",
     "mask width and convergence semantics require C06/C07 conformance"},
};

bool IsValidSymbol(absl::string_view symbol) {
  if (symbol.empty()) return false;
  const auto valid_first = [](char c) {
    return absl::ascii_isalpha(c) || c == '_' || c == '.' || c == '$';
  };
  const auto valid_rest = [&](char c) {
    return valid_first(c) || absl::ascii_isdigit(c);
  };
  return valid_first(symbol.front()) &&
         std::all_of(symbol.begin() + 1, symbol.end(), valid_rest);
}

absl::string_view AddressSpaceKindText(MusaAddressSpaceKind kind) {
  switch (kind) {
    case MusaAddressSpaceKind::kGeneric:
      return "generic";
    case MusaAddressSpaceKind::kGlobal:
      return "global";
    case MusaAddressSpaceKind::kConstant:
      return "constant";
    case MusaAddressSpaceKind::kWorkgroup:
      return "workgroup";
    case MusaAddressSpaceKind::kPrivate:
      return "private";
    case MusaAddressSpaceKind::kVendorInternal:
      return "vendor-internal";
  }
  return {};
}

}  // namespace

absl::Span<const MusaShimSpec> MusaShimSpecs() { return kShimSpecs; }

absl::Span<const MusaAddressSpaceSpec> MusaAddressSpaceSpecs() {
  return kAddressSpaceSpecs;
}

absl::Span<const MusaUnsupportedCapability> MusaUnsupportedCapabilities() {
  return kUnsupportedCapabilities;
}

const MusaShimSpec* FindMusaShim(absl::string_view symbol) {
  auto it = std::lower_bound(
      std::begin(kShimSpecs), std::end(kShimSpecs), symbol,
      [](const MusaShimSpec& spec, absl::string_view candidate) {
        return spec.xla_symbol < candidate;
      });
  return it != std::end(kShimSpecs) && it->xla_symbol == symbol ? it : nullptr;
}

const MusaAddressSpaceSpec* FindMusaAddressSpace(uint32_t address_space) {
  auto it =
      std::find_if(std::begin(kAddressSpaceSpecs), std::end(kAddressSpaceSpecs),
                   [address_space](const MusaAddressSpaceSpec& spec) {
                     return spec.number == address_space;
                   });
  return it == std::end(kAddressSpaceSpecs) ? nullptr : it;
}

absl::Status ValidateMusaShimTable() {
  absl::flat_hash_set<absl::string_view> symbols;
  absl::flat_hash_set<absl::string_view> intrinsics;
  absl::string_view previous;
  for (const MusaShimSpec& spec : kShimSpecs) {
    if (!IsValidSymbol(spec.xla_symbol) ||
        !absl::StartsWith(spec.xla_symbol, kMusaShimSymbolPrefix)) {
      return absl::InternalError(absl::StrCat(
          "invalid or unversioned MUSA shim symbol: ", spec.xla_symbol));
    }
    if (!absl::StartsWith(spec.vendor_intrinsic, "llvm.musa.")) {
      return absl::InternalError(
          absl::StrCat("MUSA shim has a non-vendor target: ", spec.xla_symbol,
                       " -> ", spec.vendor_intrinsic));
    }
    if (!previous.empty() && previous >= spec.xla_symbol) {
      return absl::InternalError(
          "MUSA shim table must be strictly sorted by XLA symbol");
    }
    if (!symbols.insert(spec.xla_symbol).second ||
        !intrinsics.insert(spec.vendor_intrinsic).second) {
      return absl::InternalError(
          absl::StrCat("duplicate MUSA shim mapping: ", spec.xla_symbol));
    }
    if (spec.minimum_mapping_version == 0 ||
        spec.minimum_mapping_version > kMusaShimMappingVersion) {
      return absl::InternalError(absl::StrCat(
          "invalid mapping version for MUSA shim: ", spec.xla_symbol));
    }
    previous = spec.xla_symbol;
  }

  uint32_t previous_space = 0;
  bool first = true;
  for (const MusaAddressSpaceSpec& spec : kAddressSpaceSpecs) {
    if (!first && spec.number <= previous_space) {
      return absl::InternalError(
          "MUSA address-space table must be strictly sorted");
    }
    if (AddressSpaceKindText(spec.kind).empty() || spec.name.empty() ||
        (spec.pointer_width != 32 && spec.pointer_width != 64)) {
      return absl::InternalError("invalid MUSA address-space specification");
    }
    previous_space = spec.number;
    first = false;
  }
  return absl::OkStatus();
}

absl::string_view MusaShimSignatureText(MusaShimSignature signature) {
  switch (signature) {
    case MusaShimSignature::kVoidVoid:
      return "void ()";
    case MusaShimSignature::kI32Void:
      return "i32 ()";
    case MusaShimSignature::kI64Void:
      return "i64 ()";
    case MusaShimSignature::kI32I32I32:
      return "i32 (i32, i32)";
  }
}

absl::string_view MusaMemoryEffectsText(MusaMemoryEffects effects) {
  switch (effects) {
    case MusaMemoryEffects::kNone:
      return "none";
    case MusaMemoryEffects::kReadWrite:
      return "read-write";
    case MusaMemoryEffects::kInaccessibleRead:
      return "inaccessible-read";
    case MusaMemoryEffects::kInaccessibleReadWrite:
      return "inaccessible-read-write";
  }
}

std::string MusaShimCanonicalText() {
  std::string text;
  absl::StrAppend(&text, "shim_abi=", kMusaShimAbiVersion,
                  "\nmapping=", kMusaShimMappingVersion,
                  "\ntriple=", kMusaTargetTriple,
                  "\narch=", kMusaTargetArchitecture,
                  "\ndata_layout=", kMusaDataLayout, "\n");
  for (const MusaAddressSpaceSpec& spec : kAddressSpaceSpecs) {
    absl::StrAppend(
        &text, "as\t", spec.number, "\t", AddressSpaceKindText(spec.kind), "\t",
        spec.name, "\t", spec.pointer_width, "\t",
        spec.allowed_in_interchange ? "interchange" : "vendor", "\n");
  }
  for (const MusaShimSpec& spec : kShimSpecs) {
    absl::StrAppend(&text, "shim\t", spec.xla_symbol, "\t",
                    spec.vendor_intrinsic, "\t",
                    MusaShimSignatureText(spec.signature), "\t",
                    MusaMemoryEffectsText(spec.memory_effects), "\t",
                    static_cast<int>(spec.required_attributes), "\t",
                    spec.convergent ? "convergent" : "nonconvergent", "\t",
                    spec.minimum_mapping_version, "\n");
  }
  for (const MusaUnsupportedCapability& capability : kUnsupportedCapabilities) {
    absl::StrAppend(&text, "unsupported\t", capability.name, "\t",
                    capability.reason, "\n");
  }
  return text;
}

}  // namespace xla::gpu::musa
