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

#include "xla/service/gpu/musa/mcc_bundle_codegen.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/musa/bounded_subprocess.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/stream_executor/musa/musa_target_contract.h"
#include "xla/tsl/platform/env.h"

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace xla::gpu::musa {
namespace {

constexpr char kHostBundleTarget[] = "host-x86_64-unknown-linux-gnu";
constexpr size_t kMaxPathBytes = 4096;
constexpr size_t kMaxBundleListBytes = size_t{64} << 10;
constexpr size_t kMaxBundleIds = 64;
constexpr size_t kMaxBundleIdBytes = 512;

constexpr char kMccBundleProviderName[] = "mcc-bundle-v1";
constexpr char kMccBundleProviderCanonicalText[] =
    "schema=xla-musa-mcc-bundle-v1\n"
    "target=mtgpu-mt-musa/mp_21\n"
    "input_bundle_argv=--type=ll|--targets=host-x86_64-unknown-linux-gnu,"
    "musa-{triple}-{architecture}|--inputs=/dev/null,{llvm}|"
    "--outputs={input_bundle}\n"
    "mcc_argv=-mtgpu|--musa-device-only|--offload-arch={architecture}|-O2|"
    "-mllvm|-opaque-pointers|-x|ir|{input_bundle}|-c|-o|{output_bundle}\n"
    "list_argv=--type=o|--inputs={output_bundle}|--list\n"
    "selection=prefix:musa-{triple};separator:one-or-more-hyphens;"
    "architecture:{architecture};cardinality:exactly-one\n"
    "unbundle_argv=--unbundle|--type=o|--targets={reported_bundle_id}|"
    "--inputs={output_bundle}|--outputs={mubin}\n"
    "environment=HOME={temp}|LANG=C|LC_ALL=C|PATH={mcc_directory}|"
    "SOURCE_DATE_EPOCH=0|TMPDIR={temp}|TZ=UTC\n"
    "temporary_directory=mode-0700;per-request;recursive-cleanup\n"
    "bounds=protocol-v1\n";

bool ContainsNul(absl::string_view value) {
  return value.find('\0') != absl::string_view::npos;
}

bool IsSafeAbsolutePath(absl::string_view path) {
  if (path.empty() || path.front() != '/' || path.size() > kMaxPathBytes ||
      ContainsNul(path)) {
    return false;
  }
  for (unsigned char c : path) {
    if (c < 0x20 || c == 0x7f) return false;
  }
  return true;
}

std::string Dirname(absl::string_view path) {
  const size_t slash = path.rfind('/');
  if (slash == 0) return "/";
  return std::string(path.substr(0, slash));
}

std::string JoinPath(absl::string_view directory, absl::string_view basename) {
  return absl::StrCat(
      directory, directory.size() == 1 && directory.front() == '/' ? "" : "/",
      basename);
}

absl::Status ValidateOptions(absl::string_view verified_vendor_llvm,
                             const MccBundleCodegenOptions& options) {
  if (!IsSafeAbsolutePath(options.mcc_path) ||
      !IsSafeAbsolutePath(options.clang_offload_bundler_path)) {
    return absl::InvalidArgumentError(
        "MCC and clang-offload-bundler paths must be trusted absolute paths");
  }
  if (!IsSafeAbsolutePath(options.temporary_directory_root) ||
      options.temporary_directory_root.find(',') != std::string::npos) {
    return absl::InvalidArgumentError(
        "MCC temporary directory root must be an absolute path without a "
        "comma");
  }
  if (options.target_triple != stream_executor::musa::kMusaTargetTriple ||
      options.architecture != stream_executor::musa::kS80TargetArchitecture ||
      options.optimization_level != 2 || !options.opaque_pointers) {
    return absl::InvalidArgumentError(
        "MCC bundle codegen supports only frozen mtgpu-mt-musa/mp_21/O2 "
        "opaque-pointer options");
  }
  if (verified_vendor_llvm.empty() || ContainsNul(verified_vendor_llvm) ||
      options.max_llvm_bytes == 0 ||
      options.max_llvm_bytes > kMusaBridgeMaxLlvmBytes ||
      verified_vendor_llvm.size() > options.max_llvm_bytes) {
    return absl::InvalidArgumentError(
        "verified vendor LLVM text is empty, non-textual, or too large");
  }
  if (options.max_mubin_bytes == 0 ||
      options.max_mubin_bytes > kMusaBridgeMaxMubinBytes ||
      options.max_diagnostic_bytes == 0 ||
      options.max_diagnostic_bytes > kMusaBridgeMaxDiagnosticMessageBytes ||
      options.max_llvm_bytes > options.subprocess_limits.max_file_bytes ||
      options.max_mubin_bytes > options.subprocess_limits.max_file_bytes) {
    return absl::InvalidArgumentError(
        "MCC input, MUBIN, diagnostic, or subprocess file bounds are "
        "inconsistent");
  }
  return absl::OkStatus();
}

class DiagnosticAccumulator {
 public:
  DiagnosticAccumulator(size_t limit, absl::string_view private_directory)
      : limit_(limit), private_directory_(private_directory) {}

  void Append(absl::string_view stage, absl::string_view stream,
              absl::string_view text) {
    if (text.empty() || output_.size() >= limit_) return;
    AppendSanitized(absl::StrCat(stage, " ", stream, ":\n"));
    AppendSanitized(text);
    AppendSanitized("\n");
  }

  const std::string& value() const { return output_; }

 private:
  void AppendSanitized(absl::string_view text) {
    size_t index = 0;
    while (index < text.size() && output_.size() < limit_) {
      if (!private_directory_.empty() &&
          text.substr(index, private_directory_.size()) == private_directory_) {
        AppendLiteral("<temp>");
        index += private_directory_.size();
        continue;
      }
      const unsigned char c = text[index++];
      output_.push_back(c == '\n' || c == '\t' || (c >= 0x20 && c < 0x7f)
                            ? static_cast<char>(c)
                            : '?');
    }
  }

  void AppendLiteral(absl::string_view text) {
    const size_t available = limit_ - output_.size();
    output_.append(text.data(), std::min(available, text.size()));
  }

  size_t limit_;
  std::string private_directory_;
  std::string output_;
};

#if defined(__linux__)

class ScopedTempDirectory {
 public:
  static absl::StatusOr<std::unique_ptr<ScopedTempDirectory>> Create(
      absl::string_view root) {
    std::string pattern = JoinPath(root, "xla-musa-mcc-XXXXXX");
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    char* created = mkdtemp(mutable_pattern.data());
    if (created == nullptr) {
      return absl::InternalError(absl::StrCat(
          "failed to create private MCC directory: ", std::strerror(errno)));
    }
    char* canonical = realpath(created, nullptr);
    if (canonical == nullptr) {
      int64_t undeleted_files = 0;
      int64_t undeleted_directories = 0;
      (void)tsl::Env::Default()->DeleteRecursively(created, &undeleted_files,
                                                   &undeleted_directories);
      return absl::InternalError(absl::StrCat(
          "failed to resolve private MCC directory: ", std::strerror(errno)));
    }
    std::string canonical_path(canonical);
    std::free(canonical);
    if (canonical_path.find(',') != std::string::npos) {
      int64_t undeleted_files = 0;
      int64_t undeleted_directories = 0;
      (void)tsl::Env::Default()->DeleteRecursively(
          canonical_path, &undeleted_files, &undeleted_directories);
      return absl::InvalidArgumentError(
          "resolved MCC temporary directory contains a comma");
    }
    return std::unique_ptr<ScopedTempDirectory>(
        new ScopedTempDirectory(std::move(canonical_path)));
  }

  ~ScopedTempDirectory() {
    int64_t undeleted_files = 0;
    int64_t undeleted_directories = 0;
    (void)tsl::Env::Default()->DeleteRecursively(path_, &undeleted_files,
                                                 &undeleted_directories);
  }

  const std::string& path() const { return path_; }

 private:
  explicit ScopedTempDirectory(std::string path) : path_(std::move(path)) {}

  std::string path_;
};

class ScopedFd {
 public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ~ScopedFd() {
    // Linux releases the descriptor even when close reports EINTR. Retrying
    // in a multithreaded process could close an unrelated, newly reused fd.
    if (fd_ >= 0) close(fd_);
  }
  int get() const { return fd_; }

 private:
  int fd_;
};

absl::Status WriteFileExclusive(absl::string_view path,
                                absl::string_view contents) {
  const int fd =
      open(std::string(path).c_str(),
           O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) {
    return absl::InternalError(
        absl::StrCat("failed to create MCC input: ", std::strerror(errno)));
  }
  ScopedFd scoped_fd(fd);
  while (!contents.empty()) {
    const ssize_t count = write(fd, contents.data(), contents.size());
    if (count > 0) {
      contents.remove_prefix(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return absl::InternalError(
        absl::StrCat("failed to write MCC input: ", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<uint8_t>> ReadBoundedMubin(absl::string_view path,
                                                      size_t max_bytes) {
  const int fd = open(std::string(path).c_str(),
                      O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (fd < 0) {
    return absl::DataLossError(absl::StrCat(
        "MCC did not produce a readable MUBIN: ", std::strerror(errno)));
  }
  ScopedFd scoped_fd(fd);
  struct stat metadata;
  if (fstat(fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
      metadata.st_size <= 0) {
    return absl::DataLossError("MCC MUBIN is empty or not a regular file");
  }
  if (static_cast<uint64_t>(metadata.st_size) > max_bytes) {
    return absl::ResourceExhaustedError(
        "MCC MUBIN exceeds the configured byte limit");
  }

  std::vector<uint8_t> bytes(static_cast<size_t>(metadata.st_size));
  size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        read(fd, bytes.data() + offset, bytes.size() - offset);
    if (count > 0) {
      offset += count;
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return absl::DataLossError("failed while reading MCC MUBIN");
  }
  uint8_t extra = 0;
  ssize_t extra_count;
  do {
    extra_count = read(fd, &extra, 1);
  } while (extra_count < 0 && errno == EINTR);
  if (extra_count != 0) {
    return absl::DataLossError("MCC MUBIN changed while it was being read");
  }
  return bytes;
}

#endif  // defined(__linux__)

std::vector<std::pair<std::string, std::string>> ControlledEnvironment(
    const MccBundleCodegenOptions& options, absl::string_view temp_directory) {
  return {
      {"HOME", std::string(temp_directory)},
      {"LANG", "C"},
      {"LC_ALL", "C"},
      {"PATH", Dirname(options.mcc_path)},
      {"SOURCE_DATE_EPOCH", "0"},
      {"TMPDIR", std::string(temp_directory)},
      {"TZ", "UTC"},
  };
}

MusaSubprocessLimits StageLimits(const MccBundleCodegenOptions& options,
                                 bool bundle_list) {
  MusaSubprocessLimits limits = options.subprocess_limits;
  if (bundle_list) {
    limits.max_stdout_bytes =
        std::min(limits.max_stdout_bytes, kMaxBundleListBytes);
  }
  return limits;
}

absl::Status ToolFailure(absl::string_view stage,
                         const MusaSubprocessResult& result,
                         const DiagnosticAccumulator& diagnostics) {
  const std::string detail = diagnostics.value().empty()
                                 ? std::string()
                                 : absl::StrCat("\n", diagnostics.value());
  if (result.output_limit_exceeded) {
    return absl::ResourceExhaustedError(
        absl::StrCat(stage, " exceeded its output limit", detail));
  }
  if (result.timed_out) {
    return absl::DeadlineExceededError(
        absl::StrCat(stage, " timed out", detail));
  }
  if (result.terminating_signal != 0) {
    return absl::InternalError(absl::StrCat(stage, " terminated by signal ",
                                            result.terminating_signal, detail));
  }
  return absl::InternalError(
      absl::StrCat(stage, " exited with code ", result.exit_code, detail));
}

absl::StatusOr<MusaSubprocessResult> RunStage(
    absl::string_view stage, absl::string_view executable,
    std::vector<std::string> arguments, absl::string_view working_directory,
    const std::vector<std::pair<std::string, std::string>>& environment,
    const MccBundleCodegenOptions& options, bool bundle_list,
    DiagnosticAccumulator* diagnostics) {
  MusaSubprocessOptions subprocess;
  subprocess.executable = std::string(executable);
  subprocess.arguments = std::move(arguments);
  subprocess.working_directory = std::string(working_directory);
  subprocess.environment = environment;
  subprocess.limits = StageLimits(options, bundle_list);
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(subprocess);
  if (!result.ok()) {
    return absl::Status(result.status().code(),
                        absl::StrCat(stage, " could not be started: ",
                                     result.status().message()));
  }
  if (!bundle_list) {
    diagnostics->Append(stage, "stdout", result->stdout_text);
  }
  diagnostics->Append(stage, "stderr", result->stderr_text);
  if (!result->exited_successfully()) {
    return ToolFailure(stage, *result, *diagnostics);
  }
  return result;
}

bool IsBundleIdCharacter(unsigned char c) {
  return absl::ascii_isalnum(c) || c == '-' || c == '_' || c == '.' || c == '+';
}

bool IsMatchingMusaBundleId(absl::string_view id,
                            const MccBundleCodegenOptions& options) {
  const std::string prefix = absl::StrCat("musa-", options.target_triple);
  if (id.size() < prefix.size() || id.substr(0, prefix.size()) != prefix) {
    return false;
  }
  absl::string_view suffix = id.substr(prefix.size());
  size_t separators = 0;
  while (!suffix.empty() && suffix.front() == '-') {
    suffix.remove_prefix(1);
    ++separators;
  }
  return separators > 0 && suffix == options.architecture;
}

absl::StatusOr<std::string> SelectBundleId(
    absl::string_view list_output, const MccBundleCodegenOptions& options) {
  std::vector<std::string> matching_ids;
  size_t total_ids = 0;
  while (!list_output.empty()) {
    const size_t newline = list_output.find('\n');
    absl::string_view line = list_output.substr(0, newline);
    list_output = newline == absl::string_view::npos
                      ? absl::string_view()
                      : list_output.substr(newline + 1);
    line = absl::StripAsciiWhitespace(line);
    if (line.empty()) continue;
    if (++total_ids > kMaxBundleIds || line.size() > kMaxBundleIdBytes ||
        !std::all_of(line.begin(), line.end(), IsBundleIdCharacter)) {
      return absl::DataLossError(
          "clang-offload-bundler returned a malformed or oversized target "
          "list");
    }
    if (IsMatchingMusaBundleId(line, options)) {
      matching_ids.emplace_back(line);
    }
  }
  if (matching_ids.size() != 1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "clang-offload-bundler reported ", matching_ids.size(),
        " matching mtgpu-mt-musa/mp_21 images; exactly one is required"));
  }
  return std::move(matching_ids.front());
}

}  // namespace

absl::string_view MccBundleProviderName() { return kMccBundleProviderName; }

absl::string_view MccBundleProviderCanonicalText() {
  return kMccBundleProviderCanonicalText;
}

absl::StatusOr<MccBundleCodegenResult> CompileVerifiedMusaLlvmWithMcc(
    absl::string_view verified_vendor_llvm,
    const MccBundleCodegenOptions& options) {
  absl::Status status = ValidateOptions(verified_vendor_llvm, options);
  if (!status.ok()) return status;
#if !defined(__linux__)
  return absl::UnimplementedError("MCC bundle codegen requires Linux");
#else
  absl::StatusOr<std::unique_ptr<ScopedTempDirectory>> temp_or =
      ScopedTempDirectory::Create(options.temporary_directory_root);
  if (!temp_or.ok()) return temp_or.status();
  std::unique_ptr<ScopedTempDirectory> temp = *std::move(temp_or);
  DiagnosticAccumulator diagnostics(options.max_diagnostic_bytes, temp->path());
  const auto environment = ControlledEnvironment(options, temp->path());

  const std::string input_path = JoinPath(temp->path(), "module.ll");
  const std::string input_bundle_path =
      JoinPath(temp->path(), "module.input.bundle.ll");
  const std::string output_bundle_path =
      JoinPath(temp->path(), "module.output.bundle.o");
  const std::string mubin_path = JoinPath(temp->path(), "module.mubin");
  status = WriteFileExclusive(input_path, verified_vendor_llvm);
  if (!status.ok()) return status;

  const std::string input_device_target =
      absl::StrCat("musa-", options.target_triple, "-", options.architecture);
  absl::StatusOr<MusaSubprocessResult> result = RunStage(
      "LLVM bundling", options.clang_offload_bundler_path,
      {"--type=ll",
       absl::StrCat("--targets=", kHostBundleTarget, ",", input_device_target),
       absl::StrCat("--inputs=/dev/null,", input_path),
       absl::StrCat("--outputs=", input_bundle_path)},
      temp->path(), environment, options, /*bundle_list=*/false, &diagnostics);
  if (!result.ok()) return result.status();

  result = RunStage("MCC codegen", options.mcc_path,
                    {"-mtgpu", "--musa-device-only",
                     absl::StrCat("--offload-arch=", options.architecture),
                     "-O2", "-mllvm", "-opaque-pointers", "-x", "ir",
                     input_bundle_path, "-c", "-o", output_bundle_path},
                    temp->path(), environment, options, /*bundle_list=*/false,
                    &diagnostics);
  if (!result.ok()) return result.status();

  result = RunStage(
      "bundle target listing", options.clang_offload_bundler_path,
      {"--type=o", absl::StrCat("--inputs=", output_bundle_path), "--list"},
      temp->path(), environment, options, /*bundle_list=*/true, &diagnostics);
  if (!result.ok()) return result.status();
  absl::StatusOr<std::string> bundle_id =
      SelectBundleId(result->stdout_text, options);
  if (!bundle_id.ok()) return bundle_id.status();

  result = RunStage(
      "MUBIN extraction", options.clang_offload_bundler_path,
      {"--unbundle", "--type=o", absl::StrCat("--targets=", *bundle_id),
       absl::StrCat("--inputs=", output_bundle_path),
       absl::StrCat("--outputs=", mubin_path)},
      temp->path(), environment, options, /*bundle_list=*/false, &diagnostics);
  if (!result.ok()) return result.status();

  absl::StatusOr<std::vector<uint8_t>> mubin =
      ReadBoundedMubin(mubin_path, options.max_mubin_bytes);
  if (!mubin.ok()) return mubin.status();
  return MccBundleCodegenResult{
      .mubin = *std::move(mubin),
      .selected_bundle_id = *std::move(bundle_id),
      .diagnostics = diagnostics.value(),
  };
#endif
}

}  // namespace xla::gpu::musa
