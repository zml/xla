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

#ifndef XLA_SERVICE_GPU_MUSA_MUSA_COMPILATION_PROVIDER_H_
#define XLA_SERVICE_GPU_MUSA_MUSA_COMPILATION_PROVIDER_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/musa/bounded_subprocess.h"
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"
#include "xla/service/gpu/musa/protocol.pb.h"

namespace xla::gpu::musa {

inline constexpr char kMusaCompilationCacheKeyRevision[] =
    "xla-musa-compilation-key-v1";

struct MusaCompilationIdentity {
  std::string xla_revision;
  std::string current_llvm_revision;
  std::string provider_name;
  std::string provider_fingerprint;
  std::string bridge_fingerprint;
  std::string toolchain_fingerprint;
  // Recorded separately even though it also contributes to the full
  // toolchain fingerprint, so cache reviews can prove libdevice changes are
  // observable without reverse engineering a sidecar manifest.
  std::string libdevice_fingerprint;
  // Canonical compatibility-rule tokens, not the live version values alone.
  std::string driver_compatibility;
  std::string runtime_compatibility;
};

struct MusaCompilationOptions {
  uint32_t optimization_level = 2;
  bool emit_debug_information = false;
  bool deterministic = true;
  bool fast_math = false;
  bool flush_denormals_to_zero = false;
  bool finite_math_only = false;
  bool unsafe_math_optimizations = false;
  bool no_signed_zeros = false;
  bool allow_fp_contract = false;

  // Must be thread-safe and nonblocking. A true result cancels a queued or
  // running request and returns absl::CancelledError.
  std::function<bool()> cancellation_requested;
};

struct MusaCompilationCapabilities {
  bool supports_compile = true;
  bool supports_relocatable = false;
  bool supports_compile_and_link = false;
  bool vendor_llvm_isolated = true;
  std::string binary_kind = "mubin";
  std::string target = "mtgpu-mt-musa/mp_21";
};

struct MusaCompilationArtifact {
  std::vector<uint8_t> mubin;
  std::string mubin_sha256;
  std::string cache_key;
  std::vector<MusaBridgeDiagnostic> diagnostics;
  bool cache_hit = false;
  bool recovered_invalid_cache_entry = false;
};

class MusaCompilationProvider {
 public:
  virtual ~MusaCompilationProvider() = default;

  // Thread-safe. Implementations must validate the complete request/response
  // exchange and return only an explicit checked MUBIN.
  virtual absl::StatusOr<MusaCompilationArtifact> Compile(
      const MusaLlvm14CompatibilityResult& module,
      const MusaCompilationOptions& options) const = 0;

  virtual const MusaCompilationIdentity& identity() const = 0;
  virtual MusaCompilationCapabilities capabilities() const = 0;
  virtual absl::string_view name() const = 0;
};

struct MusaSubprocessBridgePaths {
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
};

struct MusaSubprocessCompilationProviderOptions {
  MusaCompilationIdentity identity;
  MusaSubprocessBridgePaths paths;

  // Both directories must already exist and be absolute. Each bridge request
  // gets a private mode-0700 child below temporary_directory_root. An empty
  // cache_directory disables the persistent cache.
  std::string temporary_directory_root = "/tmp";
  std::string cache_directory;

  size_t max_concurrent_compilations = 4;
  MusaSubprocessLimits subprocess_limits = [] {
    MusaSubprocessLimits limits;
    limits.timeout = std::chrono::minutes(5);
    limits.max_stdin_bytes = kMusaBridgeMaxRequestWireBytes;
    limits.max_stdout_bytes = kMusaBridgeMaxResponseWireBytes;
    limits.max_stderr_bytes = kMusaBridgeMaxDiagnosticBytes;
    limits.max_file_bytes = uint64_t{128} << 20;
    limits.max_address_space_bytes = uint64_t{8} << 30;
    return limits;
  }();
};

enum class MusaCompilationProviderKind : uint8_t {
  kSubprocess = 1,
  // Diagnostic-only selections. They deliberately return a structured
  // unavailable status rather than loading vendor compiler ABIs in-process.
  kMccBundleInProcess = 2,
  kDirectInternalTools = 3,
};

struct MusaCompilationProviderSelection {
  MusaCompilationProviderKind kind = MusaCompilationProviderKind::kSubprocess;
  MusaSubprocessCompilationProviderOptions subprocess;
};

absl::StatusOr<std::unique_ptr<MusaCompilationProvider>>
AssembleMusaCompilationProvider(
    const MusaCompilationProviderSelection& selection);

// Public for cache-key conformance tests and future executable envelopes.
absl::StatusOr<MusaBridgeCompileRequest> BuildMusaBridgeCompileRequest(
    const MusaLlvm14CompatibilityResult& module,
    const MusaCompilationIdentity& identity,
    const MusaCompilationOptions& options);

absl::StatusOr<std::string> MusaCompilationCacheKey(
    const MusaBridgeCompileRequest& request,
    const MusaCompilationIdentity& identity);

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_MUSA_COMPILATION_PROVIDER_H_
