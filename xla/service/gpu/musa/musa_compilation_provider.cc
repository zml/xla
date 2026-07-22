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

#include "xla/service/gpu/musa/musa_compilation_provider.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "xla/service/gpu/musa/bounded_subprocess.h"
#include "xla/service/gpu/musa/musa_bridge_ir_validator.h"
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/musa/protocol.pb.h"
#include "xla/stream_executor/musa/musa_mubin.h"
#include "xla/tsl/platform/env.h"

namespace xla::gpu::musa {
namespace {

constexpr size_t kMaxPathBytes = 4096;
constexpr size_t kMaxConcurrentCompilations = 64;
constexpr size_t kMaxSanitizedProcessDiagnosticBytes = 16 << 10;
constexpr absl::string_view kCacheMagic = "XLA_MUSA_COMPILATION_CACHE_V1\n";
constexpr size_t kMaxCacheHeaderBytes = 512;
std::atomic<uint64_t> cache_temporary_counter{0};

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

bool IsSha256(absl::string_view value) {
  if (value.size() != 64) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return absl::ascii_isdigit(c) || (c >= 'a' && c <= 'f');
  });
}

bool IsCompatibilityToken(absl::string_view value) {
  if (value.empty() || value.size() > kMusaBridgeMaxRevisionBytes ||
      !(absl::ascii_isalpha(value.front()) ||
        absl::ascii_isdigit(value.front()) || value.front() == '_')) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return absl::ascii_isalnum(c) || c == '_' || c == '-' || c == '.' ||
           c == '+';
  });
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

absl::StatusOr<std::string> CanonicalDirectory(absl::string_view path,
                                               absl::string_view field) {
  if (!IsSafeAbsolutePath(path)) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, " must be an absolute safe path"));
  }
  char* resolved_raw = realpath(std::string(path).c_str(), nullptr);
  if (resolved_raw == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("cannot resolve ", field, ": ", std::strerror(errno)));
  }
  std::string resolved(resolved_raw);
  std::free(resolved_raw);
  struct stat metadata;
  if (stat(resolved.c_str(), &metadata) != 0 || !S_ISDIR(metadata.st_mode)) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, " is not a directory"));
  }
  return resolved;
}

absl::Status ValidatePaths(const MusaSubprocessBridgePaths& paths) {
  const absl::string_view values[] = {
      paths.bridge_executable,     paths.toolchain_identity,
      paths.libclang_cpp,          paths.mcc,
      paths.clang_offload_bundler, paths.lld,
      paths.llvm_readobj,          paths.libdevice,
      paths.intrinsics_musa_td,    paths.builtins_mtgpu_def,
  };
  if (!std::all_of(std::begin(values), std::end(values), IsSafeAbsolutePath)) {
    return absl::InvalidArgumentError(
        "every MUSA bridge/toolchain path must be absolute and safe");
  }
  return absl::OkStatus();
}

class ScopedTempDirectory {
 public:
  static absl::StatusOr<std::unique_ptr<ScopedTempDirectory>> Create(
      absl::string_view root) {
    std::string pattern = JoinPath(root, "xla-musa-provider-XXXXXX");
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    char* created = mkdtemp(mutable_pattern.data());
    if (created == nullptr) {
      return absl::InternalError(
          absl::StrCat("failed to create private MUSA provider directory: ",
                       std::strerror(errno)));
    }
    char* resolved_raw = realpath(created, nullptr);
    if (resolved_raw == nullptr) {
      int64_t files = 0;
      int64_t directories = 0;
      (void)tsl::Env::Default()->DeleteRecursively(created, &files,
                                                   &directories);
      return absl::InternalError(
          "failed to resolve private MUSA provider directory");
    }
    std::string resolved(resolved_raw);
    std::free(resolved_raw);
    return std::unique_ptr<ScopedTempDirectory>(
        new ScopedTempDirectory(std::move(resolved)));
  }

  ~ScopedTempDirectory() {
    int64_t files = 0;
    int64_t directories = 0;
    (void)tsl::Env::Default()->DeleteRecursively(path_, &files, &directories);
  }

  const std::string& path() const { return path_; }

 private:
  explicit ScopedTempDirectory(std::string path) : path_(std::move(path)) {}
  std::string path_;
};

class CompilationLimiter {
 public:
  explicit CompilationLimiter(size_t limit) : limit_(limit) {}

  class Permit {
   public:
    explicit Permit(CompilationLimiter* limiter) : limiter_(limiter) {}
    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;
    Permit(Permit&& other) noexcept
        : limiter_(std::exchange(other.limiter_, nullptr)) {}
    Permit& operator=(Permit&&) = delete;
    ~Permit() {
      if (limiter_ != nullptr) limiter_->Release();
    }

   private:
    CompilationLimiter* limiter_;
  };

  absl::StatusOr<Permit> Acquire(
      const std::function<bool()>& cancellation_requested) {
    std::unique_lock<std::mutex> lock(mu_);
    while (active_ >= limit_) {
      if (cancellation_requested && cancellation_requested()) {
        return absl::CancelledError(
            "MUSA compilation cancelled while waiting for a provider slot");
      }
      changed_.wait_for(lock, std::chrono::milliseconds(20));
    }
    if (cancellation_requested && cancellation_requested()) {
      return absl::CancelledError(
          "MUSA compilation cancelled before provider launch");
    }
    ++active_;
    return Permit(this);
  }

 private:
  void Release() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      --active_;
    }
    changed_.notify_one();
  }

  const size_t limit_;
  size_t active_ = 0;
  std::mutex mu_;
  std::condition_variable changed_;
};

class ScopedFd {
 public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ~ScopedFd() {
    if (fd_ >= 0) close(fd_);
  }
  int get() const { return fd_; }

 private:
  int fd_;
};

absl::Status WriteAll(int fd, absl::string_view contents) {
  while (!contents.empty()) {
    const ssize_t count = write(fd, contents.data(), contents.size());
    if (count > 0) {
      contents.remove_prefix(static_cast<size_t>(count));
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return absl::InternalError(
        absl::StrCat("MUSA cache write failed: ", std::strerror(errno)));
  }
  return absl::OkStatus();
}

std::string CachePath(absl::string_view directory, absl::string_view key) {
  return JoinPath(directory, absl::StrCat(key, ".musa-cache-v1"));
}

absl::Status RecoverInvalidCache(absl::string_view path, bool* recovered) {
  if (unlink(std::string(path).c_str()) != 0 && errno != ENOENT) {
    return absl::InternalError(absl::StrCat(
        "cannot remove invalid MUSA cache entry: ", std::strerror(errno)));
  }
  *recovered = true;
  return absl::OkStatus();
}

absl::StatusOr<std::optional<std::vector<uint8_t>>> ReadCache(
    absl::string_view directory, absl::string_view key, bool* recovered) {
  if (directory.empty()) return std::nullopt;
  const std::string path = CachePath(directory, key);
  ScopedFd fd(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (fd.get() < 0) {
    if (errno == ENOENT) return std::nullopt;
    return absl::InternalError(
        absl::StrCat("cannot open MUSA cache entry: ", std::strerror(errno)));
  }
  struct stat metadata;
  const uint64_t max_bytes = kMusaBridgeMaxMubinBytes + kMaxCacheHeaderBytes;
  if (fstat(fd.get(), &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
      metadata.st_size <= 0 ||
      static_cast<uint64_t>(metadata.st_size) > max_bytes) {
    absl::Status status = RecoverInvalidCache(path, recovered);
    if (!status.ok()) return status;
    return std::nullopt;
  }

  std::string entry(static_cast<size_t>(metadata.st_size), '\0');
  size_t offset = 0;
  while (offset < entry.size()) {
    const ssize_t count =
        read(fd.get(), entry.data() + offset, entry.size() - offset);
    if (count > 0) {
      offset += static_cast<size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    absl::Status status = RecoverInvalidCache(path, recovered);
    if (!status.ok()) return status;
    return std::nullopt;
  }

  const size_t header_end = entry.find("\n\n");
  if (header_end == std::string::npos || header_end > kMaxCacheHeaderBytes ||
      !absl::StartsWith(entry, kCacheMagic)) {
    absl::Status status = RecoverInvalidCache(path, recovered);
    if (!status.ok()) return status;
    return std::nullopt;
  }
  std::vector<std::string> lines = absl::StrSplit(
      absl::string_view(entry).substr(kCacheMagic.size(),
                                      header_end + 1 - kCacheMagic.size()),
      '\n', absl::SkipEmpty());
  if (lines.size() != 3 || !absl::StartsWith(lines[0], "key=") ||
      !absl::StartsWith(lines[1], "mubin_sha256=") ||
      !absl::StartsWith(lines[2], "mubin_bytes=")) {
    absl::Status status = RecoverInvalidCache(path, recovered);
    if (!status.ok()) return status;
    return std::nullopt;
  }
  const absl::string_view stored_key =
      absl::string_view(lines[0]).substr(strlen("key="));
  const absl::string_view stored_sha =
      absl::string_view(lines[1]).substr(strlen("mubin_sha256="));
  uint64_t stored_size = 0;
  const bool size_ok = absl::SimpleAtoi(
      absl::string_view(lines[2]).substr(strlen("mubin_bytes=")), &stored_size);
  const size_t payload_offset = header_end + 2;
  absl::string_view payload(entry.data() + payload_offset,
                            entry.size() - payload_offset);
  if (stored_key != key || !IsSha256(stored_sha) || !size_ok ||
      stored_size != payload.size() ||
      MusaBridgeSha256Hex(payload) != stored_sha) {
    absl::Status status = RecoverInvalidCache(path, recovered);
    if (!status.ok()) return status;
    return std::nullopt;
  }
  std::vector<uint8_t> mubin(payload.begin(), payload.end());
  if (!stream_executor::musa::ValidateMusaMubin(mubin).ok()) {
    absl::Status status = RecoverInvalidCache(path, recovered);
    if (!status.ok()) return status;
    return std::nullopt;
  }
  return std::optional<std::vector<uint8_t>>(std::move(mubin));
}

absl::Status WriteCache(absl::string_view directory, absl::string_view key,
                        absl::string_view mubin_sha,
                        const std::vector<uint8_t>& mubin) {
  if (directory.empty()) return absl::OkStatus();
  const std::string payload(reinterpret_cast<const char*>(mubin.data()),
                            mubin.size());
  const std::string entry =
      absl::StrCat(kCacheMagic, "key=", key, "\n", "mubin_sha256=", mubin_sha,
                   "\n", "mubin_bytes=", mubin.size(), "\n\n", payload);
  const uint64_t counter =
      cache_temporary_counter.fetch_add(1, std::memory_order_relaxed);
  const std::string temporary = JoinPath(
      directory, absl::StrCat(".", key, ".", getpid(), ".", counter, ".tmp"));
  const std::string destination = CachePath(directory, key);
  ScopedFd fd(open(temporary.c_str(),
                   O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
  if (fd.get() < 0) {
    return absl::InternalError(absl::StrCat(
        "cannot create atomic MUSA cache entry: ", std::strerror(errno)));
  }
  absl::Status status = WriteAll(fd.get(), entry);
  if (status.ok() && fsync(fd.get()) != 0) {
    status = absl::InternalError(
        absl::StrCat("cannot sync MUSA cache entry: ", std::strerror(errno)));
  }
  if (!status.ok()) {
    (void)unlink(temporary.c_str());
    return status;
  }
  if (rename(temporary.c_str(), destination.c_str()) != 0) {
    const int saved_errno = errno;
    (void)unlink(temporary.c_str());
    return absl::InternalError(absl::StrCat("cannot publish MUSA cache entry: ",
                                            std::strerror(saved_errno)));
  }
  ScopedFd directory_fd(
      open(std::string(directory).c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY));
  if (directory_fd.get() < 0 || fsync(directory_fd.get()) != 0) {
    return absl::InternalError("cannot sync MUSA cache directory");
  }
  return absl::OkStatus();
}

std::string SanitizedProcessDiagnostic(absl::string_view input) {
  std::string output;
  output.reserve(std::min(input.size(), kMaxSanitizedProcessDiagnosticBytes));
  size_t cursor = 0;
  while (cursor < input.size() &&
         output.size() < kMaxSanitizedProcessDiagnosticBytes) {
    if (input[cursor] == '/') {
      constexpr absl::string_view replacement = "<path>";
      output.append(
          replacement.data(),
          std::min(replacement.size(),
                   kMaxSanitizedProcessDiagnosticBytes - output.size()));
      ++cursor;
      while (cursor < input.size()) {
        const unsigned char c = input[cursor];
        if (c <= 0x20 || c == 0x7f || c == '"' || c == '\'' || c == '<' ||
            c == '>' || c == '[' || c == ']' || c == '(' || c == ')') {
          break;
        }
        ++cursor;
      }
      continue;
    }
    const unsigned char c = input[cursor++];
    output.push_back(c == '\n' || c == '\t' || (c >= 0x20 && c < 0x7f)
                         ? static_cast<char>(c)
                         : '?');
  }
  return output.empty() ? "no sanitized bridge diagnostic" : output;
}

absl::Status ProcessFailure(const MusaSubprocessResult& result) {
  if (result.cancelled) {
    return absl::CancelledError("MUSA bridge compilation was cancelled");
  }
  if (result.output_limit_exceeded) {
    return absl::ResourceExhaustedError(
        "MUSA bridge exceeded its stdout or stderr bound");
  }
  if (result.timed_out) {
    return absl::DeadlineExceededError("MUSA bridge compilation timed out");
  }
  const std::string diagnostic = SanitizedProcessDiagnostic(result.stderr_text);
  if (result.terminating_signal != 0) {
    return absl::InternalError(absl::StrCat("MUSA bridge terminated by signal ",
                                            result.terminating_signal, ": ",
                                            diagnostic));
  }
  return absl::InternalError(absl::StrCat("MUSA bridge exited with code ",
                                          result.exit_code, ": ", diagnostic));
}

std::string ResponseDiagnostic(const MusaBridgeCompileResponse& response) {
  std::vector<std::string> diagnostics;
  diagnostics.reserve(response.diagnostics_size());
  for (const MusaBridgeDiagnostic& diagnostic : response.diagnostics()) {
    diagnostics.push_back(absl::StrCat(diagnostic.code(), "@",
                                       diagnostic.component(), ": ",
                                       diagnostic.message()));
  }
  return absl::StrJoin(diagnostics, "; ");
}

absl::Status StructuredBridgeFailure(
    const MusaBridgeCompileResponse& response) {
  const std::string message = absl::StrCat("MUSA bridge rejected compilation: ",
                                           ResponseDiagnostic(response));
  switch (response.status()) {
    case MUSA_BRIDGE_STATUS_REJECTED:
      return absl::InvalidArgumentError(message);
    case MUSA_BRIDGE_STATUS_COMPILATION_ERROR:
      return absl::InternalError(message);
    case MUSA_BRIDGE_STATUS_INTERNAL_ERROR:
      return absl::InternalError(message);
    case MUSA_BRIDGE_STATUS_OK:
    case MUSA_BRIDGE_STATUS_UNSPECIFIED:
    default:
      return absl::InternalError("invalid MUSA bridge failure status");
  }
}

std::vector<std::string> BridgeArguments(const MusaSubprocessBridgePaths& paths,
                                         absl::string_view temp_root) {
  return {
      absl::StrCat("--bridge-executable=", paths.bridge_executable),
      absl::StrCat("--toolchain-identity=", paths.toolchain_identity),
      absl::StrCat("--libclang-cpp=", paths.libclang_cpp),
      absl::StrCat("--mcc=", paths.mcc),
      absl::StrCat("--clang-offload-bundler=", paths.clang_offload_bundler),
      absl::StrCat("--lld=", paths.lld),
      absl::StrCat("--llvm-readobj=", paths.llvm_readobj),
      absl::StrCat("--libdevice=", paths.libdevice),
      absl::StrCat("--intrinsics-musa-td=", paths.intrinsics_musa_td),
      absl::StrCat("--builtins-mtgpu-def=", paths.builtins_mtgpu_def),
      absl::StrCat("--temp-root=", temp_root),
  };
}

std::vector<std::pair<std::string, std::string>> BridgeEnvironment(
    const MusaSubprocessBridgePaths& paths, absl::string_view temp_root) {
  return {
      {"HOME", std::string(temp_root)},
      {"LANG", "C"},
      {"LC_ALL", "C"},
      {"PATH", Dirname(paths.bridge_executable)},
      {"SOURCE_DATE_EPOCH", "0"},
      {"TMPDIR", std::string(temp_root)},
      {"TZ", "UTC"},
  };
}

class SubprocessCompilationProvider final : public MusaCompilationProvider {
 public:
  explicit SubprocessCompilationProvider(
      MusaSubprocessCompilationProviderOptions options)
      : options_(std::move(options)),
        limiter_(options_.max_concurrent_compilations) {}

  absl::StatusOr<MusaCompilationArtifact> Compile(
      const MusaLlvm14CompatibilityResult& module,
      const MusaCompilationOptions& compilation_options) const override {
    if (compilation_options.cancellation_requested &&
        compilation_options.cancellation_requested()) {
      return absl::CancelledError(
          "MUSA compilation cancelled before request construction");
    }
    absl::StatusOr<MusaBridgeCompileRequest> request =
        BuildMusaBridgeCompileRequest(module, options_.identity,
                                      compilation_options);
    if (!request.ok()) return request.status();
    absl::StatusOr<std::string> cache_key =
        MusaCompilationCacheKey(*request, options_.identity);
    if (!cache_key.ok()) return cache_key.status();

    bool recovered = false;
    absl::StatusOr<std::optional<std::vector<uint8_t>>> cached =
        ReadCache(options_.cache_directory, *cache_key, &recovered);
    if (!cached.ok()) return cached.status();
    if (cached->has_value()) {
      const std::string mubin_sha256 = MusaBridgeSha256Hex(absl::string_view(
          reinterpret_cast<const char*>((*cached)->data()), (*cached)->size()));
      return MusaCompilationArtifact{
          .mubin = std::move(**cached),
          .mubin_sha256 = mubin_sha256,
          .cache_key = *cache_key,
          .cache_hit = true,
          .recovered_invalid_cache_entry = recovered,
      };
    }

    absl::StatusOr<CompilationLimiter::Permit> permit =
        limiter_.Acquire(compilation_options.cancellation_requested);
    if (!permit.ok()) return permit.status();

    // A compilation that held a slot may have populated the cache while this
    // request was queued.
    cached = ReadCache(options_.cache_directory, *cache_key, &recovered);
    if (!cached.ok()) return cached.status();
    if (cached->has_value()) {
      const std::string mubin_sha256 = MusaBridgeSha256Hex(absl::string_view(
          reinterpret_cast<const char*>((*cached)->data()), (*cached)->size()));
      return MusaCompilationArtifact{
          .mubin = std::move(**cached),
          .mubin_sha256 = mubin_sha256,
          .cache_key = *cache_key,
          .cache_hit = true,
          .recovered_invalid_cache_entry = recovered,
      };
    }

    absl::StatusOr<std::string> request_wire =
        EncodeMusaBridgeCompileRequest(*request);
    if (!request_wire.ok()) return request_wire.status();
    absl::StatusOr<std::unique_ptr<ScopedTempDirectory>> temporary =
        ScopedTempDirectory::Create(options_.temporary_directory_root);
    if (!temporary.ok()) return temporary.status();

    MusaSubprocessOptions subprocess;
    subprocess.executable = options_.paths.bridge_executable;
    subprocess.arguments =
        BridgeArguments(options_.paths, (*temporary)->path());
    subprocess.stdin_data = *std::move(request_wire);
    subprocess.working_directory = (*temporary)->path();
    subprocess.environment =
        BridgeEnvironment(options_.paths, (*temporary)->path());
    subprocess.limits = options_.subprocess_limits;
    subprocess.cancellation_requested =
        compilation_options.cancellation_requested;

    absl::StatusOr<MusaSubprocessResult> process =
        RunMusaBoundedSubprocess(subprocess);
    if (!process.ok()) return process.status();
    if (!process->exited_successfully()) return ProcessFailure(*process);
    if (!process->stderr_text.empty()) {
      return absl::DataLossError(
          "successful MUSA bridge process emitted unexpected stderr");
    }
    absl::StatusOr<MusaBridgeCompileResponse> response =
        DecodeMusaBridgeCompileResponse(process->stdout_text);
    if (!response.ok()) {
      return absl::DataLossError(
          "MUSA bridge returned a malformed or noncanonical response");
    }
    absl::Status exchange = ValidateMusaBridgeExchange(*request, *response);
    if (!exchange.ok()) {
      return absl::Status(
          exchange.code(),
          absl::StrCat("MUSA bridge response identity check failed: ",
                       exchange.message()));
    }
    if (response->status() != MUSA_BRIDGE_STATUS_OK) {
      return StructuredBridgeFailure(*response);
    }
    std::vector<uint8_t> mubin(response->mubin().begin(),
                               response->mubin().end());
    absl::StatusOr<stream_executor::musa::MusaMubinMetadata> metadata =
        stream_executor::musa::ValidateMusaMubin(mubin);
    if (!metadata.ok()) {
      return absl::DataLossError(
          "MUSA bridge returned bytes that are not a checked MUBIN");
    }
    absl::Status cache_status = WriteCache(options_.cache_directory, *cache_key,
                                           response->mubin_sha256(), mubin);
    if (!cache_status.ok()) return cache_status;

    std::vector<MusaBridgeDiagnostic> diagnostics(
        response->diagnostics().begin(), response->diagnostics().end());
    return MusaCompilationArtifact{
        .mubin = std::move(mubin),
        .mubin_sha256 = response->mubin_sha256(),
        .cache_key = *cache_key,
        .diagnostics = std::move(diagnostics),
        .cache_hit = false,
        .recovered_invalid_cache_entry = recovered,
    };
  }

  const MusaCompilationIdentity& identity() const override {
    return options_.identity;
  }

  MusaCompilationCapabilities capabilities() const override { return {}; }

  absl::string_view name() const override {
    return options_.identity.provider_name;
  }

 private:
  MusaSubprocessCompilationProviderOptions options_;
  mutable CompilationLimiter limiter_;
};

absl::StatusOr<MusaSubprocessCompilationProviderOptions>
ValidateAndCanonicalizeOptions(
    MusaSubprocessCompilationProviderOptions options) {
  absl::Status status = ValidateMusaCompilationIdentity(options.identity);
  if (!status.ok()) return status;
  status = ValidatePaths(options.paths);
  if (!status.ok()) return status;
  if (options.max_concurrent_compilations == 0 ||
      options.max_concurrent_compilations > kMaxConcurrentCompilations) {
    return absl::InvalidArgumentError(
        "MUSA compilation concurrency is outside the range 1..64");
  }
  absl::StatusOr<std::string> temporary = CanonicalDirectory(
      options.temporary_directory_root, "temporary_directory_root");
  if (!temporary.ok()) return temporary.status();
  options.temporary_directory_root = *std::move(temporary);
  if (!options.cache_directory.empty()) {
    absl::StatusOr<std::string> cache =
        CanonicalDirectory(options.cache_directory, "cache_directory");
    if (!cache.ok()) return cache.status();
    options.cache_directory = *std::move(cache);
  }
  return options;
}

}  // namespace

absl::Status ValidateMusaCompilationIdentity(
    const MusaCompilationIdentity& identity) {
  if (!IsCompatibilityToken(identity.xla_revision) ||
      !IsCompatibilityToken(identity.current_llvm_revision) ||
      !IsCompatibilityToken(identity.provider_name)) {
    return absl::InvalidArgumentError(
        "MUSA XLA, LLVM, and provider identities must be canonical tokens");
  }
  if (!IsSha256(identity.provider_fingerprint) ||
      !IsSha256(identity.bridge_fingerprint) ||
      !IsSha256(identity.toolchain_fingerprint) ||
      !IsSha256(identity.libdevice_fingerprint)) {
    return absl::InvalidArgumentError(
        "MUSA compilation identity fingerprints must be lowercase SHA-256");
  }
  if (!IsCompatibilityToken(identity.driver_compatibility) ||
      !IsCompatibilityToken(identity.runtime_compatibility)) {
    return absl::InvalidArgumentError(
        "MUSA driver/runtime compatibility rules must be canonical tokens");
  }
  return absl::OkStatus();
}

absl::StatusOr<MusaBridgeCompileRequest> BuildMusaBridgeCompileRequest(
    const MusaLlvm14CompatibilityResult& module,
    const MusaCompilationIdentity& identity,
    const MusaCompilationOptions& options) {
  absl::Status status = ValidateMusaCompilationIdentity(identity);
  if (!status.ok()) return status;
  if (module.normalized_llvm.empty() ||
      MusaBridgeSha256Hex(module.normalized_llvm) !=
          module.normalized_llvm_sha256) {
    return absl::InvalidArgumentError(
        "MUSA compatibility result has an invalid LLVM fingerprint");
  }

  MusaBridgeCompileRequest request;
  request.set_protocol_version(module.metadata.protocol_version);
  request.set_shim_abi_version(module.metadata.shim_abi_version);
  request.set_mapping_version(module.metadata.mapping_version);
  request.set_module_name(module.metadata.module_name);
  request.set_normalized_llvm(module.normalized_llvm);
  request.set_normalized_llvm_bytes(module.normalized_llvm.size());
  request.set_normalized_llvm_sha256(module.normalized_llvm_sha256);
  for (const std::string& kernel : module.metadata.kernel_entry_names) {
    request.add_kernel_entry_names(kernel);
  }

  std::vector<std::string> exported_symbols =
      module.metadata.kernel_entry_names;
  for (const MusaExportedGlobal& global : module.metadata.exported_globals) {
    if (global.alignment > std::numeric_limits<uint32_t>::max()) {
      return absl::InvalidArgumentError(
          "MUSA exported global alignment exceeds the wire ABI");
    }
    exported_symbols.push_back(global.name);
    MusaBridgeExportedGlobal* wire_global = request.add_exported_globals();
    wire_global->set_name(global.name);
    wire_global->set_kind(global.kind == MusaExportedGlobalKind::kConstant
                              ? MUSA_BRIDGE_GLOBAL_KIND_CONSTANT
                              : MUSA_BRIDGE_GLOBAL_KIND_MUTABLE);
    wire_global->set_address_space(global.address_space);
    wire_global->set_size_bytes(global.size);
    wire_global->set_alignment_bytes(static_cast<uint32_t>(global.alignment));
  }
  std::sort(exported_symbols.begin(), exported_symbols.end());
  if (std::adjacent_find(exported_symbols.begin(), exported_symbols.end()) !=
      exported_symbols.end()) {
    return absl::InvalidArgumentError(
        "MUSA exported kernel/global symbols are not unique");
  }
  for (const std::string& symbol : exported_symbols) {
    request.add_exported_symbol_names(symbol);
  }

  request.set_target_triple(kMusaTargetTriple);
  request.set_architecture(module.metadata.architecture);
  request.set_data_layout(kMusaDataLayout);
  request.set_pointer_model(MUSA_BRIDGE_POINTER_MODEL_OPAQUE);
  request.set_pointer_width_bits(kMusaInterchangePointerWidth);
  request.set_byte_order(MUSA_BRIDGE_BYTE_ORDER_LITTLE_ENDIAN);
  MusaBridgeNumericalFlags* numerical = request.mutable_numerical_flags();
  numerical->set_fast_math(options.fast_math);
  numerical->set_flush_denormals_to_zero(options.flush_denormals_to_zero);
  numerical->set_finite_math_only(options.finite_math_only);
  numerical->set_unsafe_math_optimizations(options.unsafe_math_optimizations);
  numerical->set_no_signed_zeros(options.no_signed_zeros);
  numerical->set_allow_fp_contract(options.allow_fp_contract);
  request.set_optimization_level(options.optimization_level);
  request.set_emit_debug_information(options.emit_debug_information);
  request.set_deterministic(options.deterministic);
  request.set_xla_revision(identity.xla_revision);
  request.set_current_llvm_revision(identity.current_llvm_revision);
  request.set_provider_name(identity.provider_name);
  request.set_provider_fingerprint(identity.provider_fingerprint);
  request.set_bridge_fingerprint(identity.bridge_fingerprint);
  request.set_toolchain_fingerprint(identity.toolchain_fingerprint);
  request.set_mapping_fingerprint(kMusaShimMappingSha256);

  status = ValidateMusaBridgeCompileRequestIr(request);
  if (!status.ok()) return status;
  return request;
}

absl::StatusOr<std::string> MusaCompilationCacheKey(
    const MusaBridgeCompileRequest& request,
    const MusaCompilationIdentity& identity) {
  absl::Status status = ValidateMusaBridgeCompileRequestIr(request);
  if (!status.ok()) return status;
  status = ValidateMusaCompilationIdentity(identity);
  if (!status.ok()) return status;
  if (request.xla_revision() != identity.xla_revision ||
      request.current_llvm_revision() != identity.current_llvm_revision ||
      request.provider_name() != identity.provider_name ||
      request.provider_fingerprint() != identity.provider_fingerprint ||
      request.bridge_fingerprint() != identity.bridge_fingerprint ||
      request.toolchain_fingerprint() != identity.toolchain_fingerprint) {
    return absl::InvalidArgumentError(
        "MUSA cache-key identity does not match the compile request");
  }
  absl::StatusOr<std::string> request_sha =
      MusaBridgeCompileRequestSha256(request);
  if (!request_sha.ok()) return request_sha.status();

  const std::string canonical = absl::StrCat(
      "schema=", kMusaCompilationCacheKeyRevision, "\n",
      "request_sha256=", *request_sha, "\n",
      "normalized_llvm_sha256=", request.normalized_llvm_sha256(), "\n",
      "compatibility_revision=", kMusaLlvm14CompatibilityRevision, "\n",
      "protocol_version=", request.protocol_version(), "\n",
      "shim_abi_version=", request.shim_abi_version(), "\n",
      "mapping_version=", request.mapping_version(), "\n",
      "mubin_loader_abi_version=",
      stream_executor::musa::kMubinLoaderAbiVersion, "\n",
      "provider=", identity.provider_name, "\n",
      "provider_fingerprint=", identity.provider_fingerprint, "\n",
      "bridge_fingerprint=", identity.bridge_fingerprint, "\n",
      "toolchain_fingerprint=", identity.toolchain_fingerprint, "\n",
      "libdevice_fingerprint=", identity.libdevice_fingerprint, "\n",
      "driver_compatibility=", identity.driver_compatibility, "\n",
      "runtime_compatibility=", identity.runtime_compatibility, "\n");
  return MusaBridgeSha256Hex(canonical);
}

absl::StatusOr<std::unique_ptr<MusaCompilationProvider>>
AssembleMusaCompilationProvider(
    const MusaCompilationProviderSelection& selection) {
  switch (selection.kind) {
    case MusaCompilationProviderKind::kSubprocess: {
      absl::StatusOr<MusaSubprocessCompilationProviderOptions> options =
          ValidateAndCanonicalizeOptions(selection.subprocess);
      if (!options.ok()) return options.status();
      return std::make_unique<SubprocessCompilationProvider>(
          *std::move(options));
    }
    case MusaCompilationProviderKind::kMccBundleInProcess:
      return absl::UnavailableError(
          "the MCC bundle route is qualified only behind the isolated "
          "musa-llvm-bridge subprocess");
    case MusaCompilationProviderKind::kDirectInternalTools:
      return absl::UnavailableError(
          "direct MUSA internal-tool compilation is a diagnostic probe, not "
          "a qualified PJRT provider");
  }
  return absl::InvalidArgumentError(
      "unknown MUSA compilation provider selection");
}

}  // namespace xla::gpu::musa
