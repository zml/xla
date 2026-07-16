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

#ifndef XLA_SERVICE_GPU_MUSA_MCC_BUNDLE_CODEGEN_H_
#define XLA_SERVICE_GPU_MUSA_MCC_BUNDLE_CODEGEN_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/musa/bounded_subprocess.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/stream_executor/musa/musa_target_contract.h"

namespace xla::gpu::musa {

// Closed configuration for the initially qualified MCC bundle route. Fields
// describing code generation are deliberately validated against the frozen
// S80 contract rather than accepted as general MCC flags.
struct MccBundleCodegenOptions {
  // Trusted tool paths supplied by bridge startup. They must be absolute.
  std::string mcc_path;
  std::string clang_offload_bundler_path;

  std::string target_triple = stream_executor::musa::kMusaTargetTriple;
  std::string architecture = stream_executor::musa::kS80TargetArchitecture;
  int optimization_level = 2;
  bool opaque_pointers = true;

  // A fresh mode-0700 directory is created below this absolute directory for
  // every invocation and recursively removed before the call returns.
  std::string temporary_directory_root = "/tmp";

  MusaSubprocessLimits subprocess_limits;
  size_t max_llvm_bytes = kMusaBridgeMaxLlvmBytes;
  size_t max_mubin_bytes = kMusaBridgeMaxMubinBytes;
  size_t max_diagnostic_bytes = kMusaBridgeMaxDiagnosticMessageBytes;
};

struct MccBundleCodegenResult {
  std::vector<uint8_t> mubin;
  // Exact identifier reported by clang-offload-bundler --list and used for
  // extraction. Its punctuation is intentionally not synthesized by XLA.
  std::string selected_bundle_id;
  // Sanitized, path-redacted, bounded output from successful tool stages.
  std::string diagnostics;
};

// Canonical, path-independent provider contract used as a toolchain
// fingerprint input. It freezes the provider ID, argv ordering, controlled
// environment, and output-bundle selection policy implemented below.
absl::string_view MccBundleProviderName();
absl::string_view MccBundleProviderCanonicalText();

// Compiles bridge-verified vendor LLVM text through the measured MCC offload
// bundle route. This function does not parse LLVM and has no LLVM dependency.
absl::StatusOr<MccBundleCodegenResult> CompileVerifiedMusaLlvmWithMcc(
    absl::string_view verified_vendor_llvm,
    const MccBundleCodegenOptions& options);

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_MCC_BUNDLE_CODEGEN_H_
