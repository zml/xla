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

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include "xla/service/gpu/musa/mcc_bundle_codegen.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/musa/protocol.pb.h"
#include "xla/tools/musa_llvm_bridge/bridge_core.h"
#include "xla/tools/musa_llvm_bridge/mubin_validator.h"
#include "xla/tools/musa_llvm_bridge/toolchain_fingerprint.h"

namespace xla::gpu::musa::bridge {
namespace {

constexpr size_t kMaxPathBytes = 4096;
constexpr size_t kDiagnosticLimit = 16 << 10;

struct MainOptions {
  MusaBridgeToolchainPaths paths;
  std::string temporary_directory_root;
};

struct Identity {
  std::string provider_name;
  std::string provider_fingerprint;
  std::string bridge_fingerprint;
  std::string toolchain_fingerprint;
};

bool IsSafeAbsolutePath(absl::string_view path) {
  if (path.empty() || path.size() > kMaxPathBytes || path.front() != '/' ||
      path.find('\0') != absl::string_view::npos) {
    return false;
  }
  return std::all_of(path.begin(), path.end(),
                     [](unsigned char c) { return c >= 0x20 && c != 0x7f; });
}

std::string Dirname(absl::string_view path) {
  const size_t slash = path.rfind('/');
  if (slash == 0) return "/";
  return std::string(path.substr(0, slash));
}

std::string JoinPath(absl::string_view directory, absl::string_view relative) {
  return absl::StrCat(
      directory, directory.size() == 1 && directory.front() == '/' ? "" : "/",
      relative);
}

absl::Status ValidatePinnedSdkClosure(const MusaBridgeToolchainPaths& paths) {
  const std::string sdk_root = Dirname(Dirname(paths.mcc));
  struct ExpectedPath {
    absl::string_view component;
    const std::string* actual;
    absl::string_view relative;
  };
  const ExpectedPath expected[] = {
      {"mcc", &paths.mcc, "bin/clang-14"},
      {"clang-offload-bundler", &paths.clang_offload_bundler,
       "bin/clang-offload-bundler"},
      {"lld", &paths.lld, "bin/lld"},
      {"llvm-readobj", &paths.llvm_readobj, "bin/llvm-readobj"},
      {"libclang-cpp", &paths.libclang_cpp, "lib/libclang-cpp.so.14"},
      {"libdevice", &paths.libdevice, "mtgpu/bitcode/libdevice.bc"},
      {"IntrinsicsMUSA.td", &paths.intrinsics_musa_td,
       "include/llvm/IR/IntrinsicsMUSA.td"},
      {"BuiltinsMTGPU.def", &paths.builtins_mtgpu_def,
       "include/clang/Basic/BuiltinsMTGPU.def"},
  };
  for (const ExpectedPath& component : expected) {
    if (*component.actual != JoinPath(sdk_root, component.relative)) {
      return absl::FailedPreconditionError(
          absl::StrCat("bridge component ", component.component,
                       " is not from the canonical pinned MCC SDK root"));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<MainOptions> ParseCommandLine(int argc, char** argv) {
  MainOptions options;
  struct Flag {
    absl::string_view name;
    std::string* value;
  };
  const Flag flags[] = {
      {"bridge-executable", &options.paths.bridge_executable},
      {"toolchain-identity", &options.paths.toolchain_identity},
      {"libclang-cpp", &options.paths.libclang_cpp},
      {"mcc", &options.paths.mcc},
      {"clang-offload-bundler", &options.paths.clang_offload_bundler},
      {"lld", &options.paths.lld},
      {"llvm-readobj", &options.paths.llvm_readobj},
      {"libdevice", &options.paths.libdevice},
      {"intrinsics-musa-td", &options.paths.intrinsics_musa_td},
      {"builtins-mtgpu-def", &options.paths.builtins_mtgpu_def},
      {"temp-root", &options.temporary_directory_root},
  };
  absl::flat_hash_set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const absl::string_view argument(argv[i]);
    if (!absl::StartsWith(argument, "--")) {
      return absl::InvalidArgumentError(
          "musa-llvm-bridge accepts named flags only");
    }
    const size_t separator = argument.find('=');
    if (separator == absl::string_view::npos || separator <= 2 ||
        separator + 1 == argument.size()) {
      return absl::InvalidArgumentError(
          "every musa-llvm-bridge flag requires a nonempty value");
    }
    const absl::string_view name = argument.substr(2, separator - 2);
    const absl::string_view value = argument.substr(separator + 1);
    auto flag = std::find_if(
        std::begin(flags), std::end(flags),
        [&](const Flag& candidate) { return candidate.name == name; });
    if (flag == std::end(flags)) {
      return absl::InvalidArgumentError(
          "musa-llvm-bridge received an unknown flag");
    }
    if (!seen.insert(std::string(name)).second) {
      return absl::InvalidArgumentError(
          "musa-llvm-bridge received a duplicate flag");
    }
    *flag->value = std::string(value);
  }
  if (seen.size() != std::size(flags)) {
    return absl::InvalidArgumentError(
        "musa-llvm-bridge requires every toolchain and temporary-root flag");
  }
  for (const Flag& flag : flags) {
    if (!IsSafeAbsolutePath(*flag.value)) {
      return absl::InvalidArgumentError(
          "musa-llvm-bridge flag values must be safe absolute paths");
    }
  }
  if (options.temporary_directory_root.find(',') != std::string::npos) {
    return absl::InvalidArgumentError(
        "musa-llvm-bridge temporary root must not contain a comma");
  }
  return options;
}

absl::StatusOr<std::string> ResolvePath(absl::string_view value,
                                        absl::string_view component,
                                        bool directory, bool executable) {
  char* resolved_raw = realpath(std::string(value).c_str(), nullptr);
  if (resolved_raw == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("cannot resolve bridge component ", component));
  }
  std::string resolved(resolved_raw);
  std::free(resolved_raw);
  struct stat metadata;
  if (stat(resolved.c_str(), &metadata) != 0 ||
      (directory ? !S_ISDIR(metadata.st_mode) : !S_ISREG(metadata.st_mode))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "bridge component ", component, " has the wrong filesystem type"));
  }
  const int access_mode =
      directory ? (W_OK | X_OK) : (R_OK | (executable ? X_OK : 0));
  if (access(resolved.c_str(), access_mode) != 0) {
    return absl::PermissionDeniedError(
        absl::StrCat("bridge component ", component,
                     " does not have the required access mode"));
  }
  return resolved;
}

absl::StatusOr<MainOptions> ResolveAndValidatePaths(MainOptions options) {
  struct Component {
    absl::string_view name;
    std::string* path;
    bool executable;
  };
  const Component components[] = {
      {"bridge", &options.paths.bridge_executable, true},
      {"toolchain-identity", &options.paths.toolchain_identity, false},
      {"libclang-cpp", &options.paths.libclang_cpp, false},
      {"mcc", &options.paths.mcc, true},
      {"clang-offload-bundler", &options.paths.clang_offload_bundler, true},
      {"lld", &options.paths.lld, true},
      {"llvm-readobj", &options.paths.llvm_readobj, true},
      {"libdevice", &options.paths.libdevice, false},
      {"IntrinsicsMUSA.td", &options.paths.intrinsics_musa_td, false},
      {"BuiltinsMTGPU.def", &options.paths.builtins_mtgpu_def, false},
  };
  for (const Component& component : components) {
    absl::StatusOr<std::string> resolved =
        ResolvePath(*component.path, component.name, /*directory=*/false,
                    component.executable);
    if (!resolved.ok()) return resolved.status();
    *component.path = *std::move(resolved);
  }
  if (absl::Status status = ValidatePinnedSdkClosure(options.paths);
      !status.ok()) {
    return status;
  }
  absl::StatusOr<std::string> temp =
      ResolvePath(options.temporary_directory_root, "temporary-root",
                  /*directory=*/true, /*executable=*/false);
  if (!temp.ok()) return temp.status();
  options.temporary_directory_root = *std::move(temp);

  absl::StatusOr<std::string> self =
      ResolvePath("/proc/self/exe", "running-bridge", /*directory=*/false,
                  /*executable=*/true);
  if (!self.ok()) return self.status();
  if (*self != options.paths.bridge_executable) {
    return absl::FailedPreconditionError(
        "bridge-executable does not identify the running process image");
  }
  absl::StatusOr<std::string> loaded_vendor_llvm =
      LoadedVendorLlvmSharedObjectPath();
  if (!loaded_vendor_llvm.ok()) return loaded_vendor_llvm.status();
  if (*loaded_vendor_llvm != options.paths.libclang_cpp) {
    return absl::FailedPreconditionError(
        "libclang-cpp does not identify the loaded vendor LLVM image");
  }
  return options;
}

absl::StatusOr<std::string> ReadBoundedStdin() {
  struct stat metadata;
  if (fstat(STDIN_FILENO, &metadata) == 0 && S_ISREG(metadata.st_mode) &&
      metadata.st_size >= 0 &&
      static_cast<uint64_t>(metadata.st_size) >
          kMusaBridgeMaxRequestWireBytes) {
    return absl::ResourceExhaustedError(
        "MUSA bridge request exceeds its wire limit");
  }
  std::string input;
  input.reserve(std::min<size_t>(kMusaBridgeMaxRequestWireBytes, 1 << 20));
  char buffer[64 << 10];
  while (true) {
    const ssize_t count = read(STDIN_FILENO, buffer, sizeof(buffer));
    if (count > 0) {
      if (static_cast<size_t>(count) >
          kMusaBridgeMaxRequestWireBytes - input.size()) {
        return absl::ResourceExhaustedError(
            "MUSA bridge request exceeds its wire limit");
      }
      input.append(buffer, count);
      continue;
    }
    if (count == 0) break;
    if (errno == EINTR) continue;
    return absl::InternalError("failed to read MUSA bridge request");
  }
  if (input.empty()) {
    return absl::InvalidArgumentError("MUSA bridge request is empty");
  }
  return input;
}

bool WriteAll(int fd, absl::string_view value) {
  while (!value.empty()) {
    const ssize_t count = write(fd, value.data(), value.size());
    if (count > 0) {
      value.remove_prefix(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

Identity RequestIdentity(const MusaBridgeCompileRequest& request) {
  return Identity{.provider_name = request.provider_name(),
                  .provider_fingerprint = request.provider_fingerprint(),
                  .bridge_fingerprint = request.bridge_fingerprint(),
                  .toolchain_fingerprint = request.toolchain_fingerprint()};
}

Identity ActualIdentity(const MusaBridgeFingerprints& fingerprints) {
  return Identity{.provider_name = fingerprints.provider_name,
                  .provider_fingerprint = fingerprints.provider_fingerprint,
                  .bridge_fingerprint = fingerprints.bridge_fingerprint,
                  .toolchain_fingerprint = fingerprints.toolchain_fingerprint};
}

std::vector<std::string> RedactedPaths(const MainOptions& options) {
  return {Dirname(Dirname(options.paths.mcc)),
          options.paths.bridge_executable,
          options.paths.toolchain_identity,
          options.paths.libclang_cpp,
          options.paths.mcc,
          options.paths.clang_offload_bundler,
          options.paths.lld,
          options.paths.llvm_readobj,
          options.paths.libdevice,
          options.paths.intrinsics_musa_td,
          options.paths.builtins_mtgpu_def,
          options.temporary_directory_root};
}

std::string SanitizeDiagnostic(absl::string_view message,
                               const std::vector<std::string>& paths) {
  std::string result;
  result.reserve(std::min(message.size(), kDiagnosticLimit));
  size_t cursor = 0;
  while (cursor < message.size() && result.size() < kDiagnosticLimit) {
    bool redacted = false;
    for (const std::string& path : paths) {
      if (!path.empty() && message.substr(cursor, path.size()) == path) {
        constexpr absl::string_view replacement = "<path>";
        result.append(
            replacement.data(),
            std::min(replacement.size(), kDiagnosticLimit - result.size()));
        cursor += path.size();
        redacted = true;
        break;
      }
    }
    if (redacted) continue;
    if (message[cursor] == '/') {
      constexpr absl::string_view replacement = "<path>";
      result.append(
          replacement.data(),
          std::min(replacement.size(), kDiagnosticLimit - result.size()));
      ++cursor;
      while (cursor < message.size()) {
        const unsigned char path_byte = message[cursor];
        if (path_byte <= 0x20 || path_byte == 0x7f || path_byte == '"' ||
            path_byte == '\'' || path_byte == '<' || path_byte == '>' ||
            path_byte == '[' || path_byte == ']' || path_byte == '(' ||
            path_byte == ')') {
          break;
        }
        ++cursor;
      }
      continue;
    }
    const unsigned char c = message[cursor++];
    result.push_back(c == '\n' || c == '\t' || (c >= 0x20 && c < 0x7f)
                         ? static_cast<char>(c)
                         : '?');
  }
  if (result.empty()) result = "operation failed";
  return result;
}

uint64_t ElapsedMicros(std::chrono::steady_clock::time_point start) {
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  return elapsed <= 0 ? 1 : static_cast<uint64_t>(elapsed);
}

uint64_t PeakMemoryBytes() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss <= 0) return 0;
  const uint64_t kib = static_cast<uint64_t>(usage.ru_maxrss);
  if (kib > std::numeric_limits<uint64_t>::max() / 1024) return 0;
  return kib * 1024;
}

absl::StatusOr<MusaBridgeCompileResponse> BaseResponse(
    const MusaBridgeCompileRequest& request, const Identity& identity) {
  absl::StatusOr<std::string> request_sha =
      MusaBridgeCompileRequestSha256(request);
  if (!request_sha.ok()) return request_sha.status();
  MusaBridgeCompileResponse response;
  response.set_protocol_version(kMusaBridgeProtocolVersion);
  response.set_shim_abi_version(kMusaShimAbiVersion);
  response.set_mapping_version(kMusaShimMappingVersion);
  response.set_request_sha256(*request_sha);
  response.set_provider_name(identity.provider_name);
  response.set_provider_fingerprint(identity.provider_fingerprint);
  response.set_bridge_fingerprint(identity.bridge_fingerprint);
  response.set_toolchain_fingerprint(identity.toolchain_fingerprint);
  response.set_mapping_fingerprint(kMusaShimMappingSha256);
  MusaBridgeCompileStats* stats = response.mutable_stats();
  stats->set_input_llvm_bytes(request.normalized_llvm_bytes());
  stats->set_kernel_count(request.kernel_entry_names_size());
  stats->set_exported_symbol_count(request.exported_symbol_names_size());
  return response;
}

void FinishStats(MusaBridgeCompileResponse* response,
                 std::chrono::steady_clock::time_point start) {
  MusaBridgeCompileStats* stats = response->mutable_stats();
  stats->set_diagnostic_count(response->diagnostics_size());
  stats->set_bridge_wall_time_microseconds(ElapsedMicros(start));
  stats->set_peak_memory_bytes(PeakMemoryBytes());
}

void AddDiagnostic(MusaBridgeCompileResponse* response,
                   MusaBridgeDiagnosticSeverity severity,
                   absl::string_view code, absl::string_view component,
                   absl::string_view message,
                   const std::vector<std::string>& paths) {
  MusaBridgeDiagnostic* diagnostic = response->add_diagnostics();
  diagnostic->set_severity(severity);
  diagnostic->set_code(code);
  diagnostic->set_component(component);
  diagnostic->set_message(SanitizeDiagnostic(message, paths));
}

absl::Status ValidateRequestForProvider(
    const MusaBridgeCompileRequest& request) {
  const MusaBridgeNumericalFlags& numerical = request.numerical_flags();
  if (request.optimization_level() != 2 || request.emit_debug_information() ||
      !request.deterministic() || numerical.fast_math() ||
      numerical.flush_denormals_to_zero() || numerical.finite_math_only() ||
      numerical.unsafe_math_optimizations() || numerical.no_signed_zeros() ||
      numerical.allow_fp_contract()) {
    return absl::InvalidArgumentError(
        "MCC bundle provider supports only deterministic O2 with debug "
        "disabled and "
        "all unimplemented numerical controls disabled");
  }
  return absl::OkStatus();
}

absl::StatusOr<MusaBridgeCompileResponse> FailureResponse(
    const MusaBridgeCompileRequest& request, const Identity& identity,
    MusaBridgeStatus bridge_status, absl::string_view code,
    absl::string_view component, const absl::Status& failure,
    const std::vector<std::string>& paths,
    std::chrono::steady_clock::time_point start) {
  absl::StatusOr<MusaBridgeCompileResponse> response =
      BaseResponse(request, identity);
  if (!response.ok()) return response.status();
  response->set_status(bridge_status);
  AddDiagnostic(&*response, MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_ERROR, code,
                component, failure.message(), paths);
  FinishStats(&*response, start);
  return response;
}

absl::StatusOr<MusaBridgeCompileResponse> CompileOneRequest(
    const MusaBridgeCompileRequest& request,
    const absl::StatusOr<MainOptions>& parsed_options,
    std::chrono::steady_clock::time_point start) {
  if (!parsed_options.ok()) {
    return FailureResponse(request, RequestIdentity(request),
                           MUSA_BRIDGE_STATUS_INTERNAL_ERROR, "startup-config",
                           "musa-llvm-bridge", parsed_options.status(), {},
                           start);
  }
  absl::StatusOr<MainOptions> options =
      ResolveAndValidatePaths(*parsed_options);
  if (!options.ok()) {
    return FailureResponse(request, RequestIdentity(request),
                           MUSA_BRIDGE_STATUS_INTERNAL_ERROR, "startup-config",
                           "musa-llvm-bridge", options.status(), {}, start);
  }
  const std::vector<std::string> redactions = RedactedPaths(*options);

  absl::StatusOr<MusaBridgeFingerprints> fingerprints =
      FingerprintMusaBridgeToolchain(
          options->paths, std::string(MccBundleProviderName()),
          std::string(MccBundleProviderCanonicalText()));
  if (!fingerprints.ok()) {
    return FailureResponse(request, RequestIdentity(request),
                           MUSA_BRIDGE_STATUS_INTERNAL_ERROR,
                           "fingerprint-failure", "toolchain-fingerprint",
                           fingerprints.status(), redactions, start);
  }
  const Identity actual = ActualIdentity(*fingerprints);
  if (request.provider_name() != actual.provider_name ||
      request.provider_fingerprint() != actual.provider_fingerprint ||
      request.bridge_fingerprint() != actual.bridge_fingerprint ||
      request.toolchain_fingerprint() != actual.toolchain_fingerprint) {
    return FailureResponse(
        request, actual, MUSA_BRIDGE_STATUS_REJECTED, "identity-mismatch",
        "toolchain-fingerprint",
        absl::FailedPreconditionError(
            "request identity does not match the running bridge toolchain"),
        redactions, start);
  }
  if (absl::Status status = ValidateRequestForProvider(request); !status.ok()) {
    return FailureResponse(request, actual, MUSA_BRIDGE_STATUS_REJECTED,
                           "unsupported-options", "mcc-bundle-provider", status,
                           redactions, start);
  }

  absl::StatusOr<VendorLlvmModule> vendor_module =
      TranslateMusaBridgeRequestToVendorLlvm(request);
  if (!vendor_module.ok()) {
    const bool rejected =
        vendor_module.status().code() == absl::StatusCode::kInvalidArgument;
    return FailureResponse(
        request, actual,
        rejected ? MUSA_BRIDGE_STATUS_REJECTED
                 : MUSA_BRIDGE_STATUS_INTERNAL_ERROR,
        rejected ? "interchange-rejected" : "translation-failure",
        "vendor-llvm14", vendor_module.status(), redactions, start);
  }

  MccBundleCodegenOptions codegen_options;
  codegen_options.mcc_path = options->paths.mcc;
  codegen_options.clang_offload_bundler_path =
      options->paths.clang_offload_bundler;
  codegen_options.temporary_directory_root = options->temporary_directory_root;
  absl::StatusOr<MccBundleCodegenResult> codegen =
      CompileVerifiedMusaLlvmWithMcc(vendor_module->llvm_ir, codegen_options);
  if (!codegen.ok()) {
    return FailureResponse(request, actual,
                           MUSA_BRIDGE_STATUS_COMPILATION_ERROR,
                           "codegen-failure", "mcc-bundle-provider",
                           codegen.status(), redactions, start);
  }
  if (absl::Status status = ValidateMubinOutput(codegen->mubin, request);
      !status.ok()) {
    return FailureResponse(
        request, actual, MUSA_BRIDGE_STATUS_COMPILATION_ERROR, "invalid-mubin",
        "mubin-validator", status, redactions, start);
  }

  absl::StatusOr<MusaBridgeCompileResponse> response =
      BaseResponse(request, actual);
  if (!response.ok()) return response.status();
  response->set_status(MUSA_BRIDGE_STATUS_OK);
  response->set_mubin(codegen->mubin.data(), codegen->mubin.size());
  response->set_mubin_sha256(MusaBridgeSha256Hex(response->mubin()));
  response->mutable_stats()->set_output_mubin_bytes(codegen->mubin.size());
  if (!codegen->diagnostics.empty()) {
    AddDiagnostic(&*response, MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_WARNING,
                  "provider-output", "mcc-bundle-provider",
                  codegen->diagnostics, redactions);
  }
  FinishStats(&*response, start);
  return response;
}

int ReportTransportFailure(const absl::Status& status) {
  const std::string message =
      SanitizeDiagnostic(status.message(), /*paths=*/{});
  const std::string line = absl::StrCat("musa-llvm-bridge: ", message, "\n");
  (void)WriteAll(STDERR_FILENO, line);
  return 2;
}

}  // namespace
}  // namespace xla::gpu::musa::bridge

int main(int argc, char** argv) {
  using namespace xla::gpu::musa;
  using namespace xla::gpu::musa::bridge;
  const auto start = std::chrono::steady_clock::now();
  absl::StatusOr<MainOptions> options = ParseCommandLine(argc, argv);
  absl::StatusOr<std::string> wire = ReadBoundedStdin();
  if (!wire.ok()) return ReportTransportFailure(wire.status());
  absl::StatusOr<MusaBridgeCompileRequest> request =
      DecodeMusaBridgeCompileRequest(*wire);
  if (!request.ok()) return ReportTransportFailure(request.status());

  absl::StatusOr<MusaBridgeCompileResponse> response =
      CompileOneRequest(*request, options, start);
  if (!response.ok()) return ReportTransportFailure(response.status());
  absl::StatusOr<std::string> encoded =
      EncodeMusaBridgeCompileResponse(*response);
  if (!encoded.ok()) return ReportTransportFailure(encoded.status());
  if (!WriteAll(STDOUT_FILENO, *encoded)) {
    return ReportTransportFailure(
        absl::InternalError("failed to write MUSA bridge response"));
  }
  return 0;
}
