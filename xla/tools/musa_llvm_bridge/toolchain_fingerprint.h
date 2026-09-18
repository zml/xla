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

#ifndef XLA_TOOLS_MUSA_LLVM_BRIDGE_TOOLCHAIN_FINGERPRINT_H_
#define XLA_TOOLS_MUSA_LLVM_BRIDGE_TOOLCHAIN_FINGERPRINT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla::gpu::musa::bridge {

struct MusaBridgeToolchainPaths {
  std::string bridge_executable;
  std::string toolchain_identity;
  std::string libclang_cpp;
  std::string mcc;
  std::string clang_offload_bundler;
  std::string lld;
  std::string llvm_readobj;
  std::string libdevice;
  std::string intrinsics_musa_td;
  std::string builtins_mtgpu_def;
  // Optional for legacy bundles. When present, its bytes enter the identity.
  std::string llvm_pass_plugin;
};

struct MusaBridgeComponentFingerprint {
  std::string name;
  uint64_t size_bytes;
  std::string sha256;
};

struct MusaBridgeFingerprints {
  std::string provider_name;
  std::string bridge_fingerprint;
  std::string provider_fingerprint;
  std::string toolchain_fingerprint;
  std::string canonical_toolchain_manifest;
  // The validated identity text from the same read used for its fingerprint.
  std::string toolchain_identity_text;
  std::vector<MusaBridgeComponentFingerprint> components;
};

// Bounded, streaming content hash. Rejects empty/non-regular files and files
// changed while hashing; never loads a compiler shared library into this
// process.
absl::StatusOr<std::string> MusaCompilerFileSha256(absl::string_view path);

// Hashes immutable bridge/toolchain inputs by content. Paths never enter a
// fingerprint. `provider_contract` is the normalized, LF-terminated text that
// freezes the provider's argv and output-selection policy.
absl::StatusOr<MusaBridgeFingerprints> FingerprintMusaBridgeToolchain(
    const MusaBridgeToolchainPaths& paths, std::string provider_name,
    std::string provider_contract);

}  // namespace xla::gpu::musa::bridge

#endif  // XLA_TOOLS_MUSA_LLVM_BRIDGE_TOOLCHAIN_FINGERPRINT_H_
