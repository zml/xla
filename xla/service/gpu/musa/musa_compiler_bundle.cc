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

#include "xla/service/gpu/musa/musa_compiler_bundle.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace xla::gpu::musa {
namespace {

constexpr size_t kMaxManifestBytes = 16 << 10;
constexpr size_t kMaxPathBytes = 4096;
constexpr char kDefaultManifestBasename[] = "musa_compiler_bundle.conf";
constexpr char kTemporaryRootEnvironment[] =
    "XLA_MUSA_COMPILATION_TEMP_ROOT";
constexpr char kCacheDirectoryEnvironment[] =
    "XLA_MUSA_COMPILATION_CACHE_DIR";

constexpr std::array<absl::string_view, 20> kManifestFields = {
    "schema",
    "xla_revision",
    "current_llvm_revision",
    "provider_name",
    "provider_fingerprint",
    "bridge_fingerprint",
    "toolchain_fingerprint",
    "libdevice_fingerprint",
    "driver_compatibility",
    "runtime_compatibility",
    "bridge_executable",
    "toolchain_identity",
    "libclang_cpp",
    "mcc",
    "clang_offload_bundler",
    "lld",
    "llvm_readobj",
    "libdevice",
    "intrinsics_musa_td",
    "builtins_mtgpu_def",
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

std::string Dirname(absl::string_view path) {
  const size_t slash = path.rfind('/');
  if (slash == absl::string_view::npos) return ".";
  if (slash == 0) return "/";
  return std::string(path.substr(0, slash));
}

std::string JoinPath(absl::string_view directory, absl::string_view basename) {
  return absl::StrCat(
      directory, directory.size() == 1 && directory.front() == '/' ? "" : "/",
      basename);
}

bool IsSafeManifestValue(absl::string_view value) {
  if (value.empty() || value.size() > kMaxPathBytes) return false;
  for (unsigned char c : value) {
    if (c <= 0x20 || c == 0x7f || c == '=' || c == '\\') return false;
  }
  return true;
}

bool IsSafeRelativePath(absl::string_view path) {
  if (!IsSafeManifestValue(path) || path.front() == '/') return false;
  while (!path.empty()) {
    const size_t slash = path.find('/');
    const absl::string_view component = path.substr(0, slash);
    if (component.empty() || component == "." || component == "..") {
      return false;
    }
    if (slash == absl::string_view::npos) break;
    path.remove_prefix(slash + 1);
  }
  return true;
}

absl::StatusOr<std::string> CanonicalPath(absl::string_view path,
                                          absl::string_view description) {
  if (path.empty() || path.front() != '/' || path.size() > kMaxPathBytes) {
    return absl::InvalidArgumentError(
        absl::StrCat(description, " must be an absolute safe path"));
  }
  char* resolved_raw = realpath(std::string(path).c_str(), nullptr);
  if (resolved_raw == nullptr) {
    return absl::NotFoundError(absl::StrCat(
        "cannot resolve ", description, ": ", std::strerror(errno)));
  }
  std::string resolved(resolved_raw);
  std::free(resolved_raw);
  return resolved;
}

absl::StatusOr<std::string> DefaultManifestPath() {
  const char* configured = std::getenv(kMusaCompilerBundleEnvironment);
  if (configured != nullptr && configured[0] != '\0') {
    return CanonicalPath(configured, kMusaCompilerBundleEnvironment);
  }

  Dl_info info = {};
  if (dladdr(reinterpret_cast<void*>(
                 &LoadMusaCompilationProviderFromBundle),
             &info) == 0 ||
      info.dli_fname == nullptr) {
    return absl::InternalError(
        "failed to locate the MUSA PJRT compiler bundle shared object");
  }
  absl::StatusOr<std::string> library =
      CanonicalPath(info.dli_fname, "MUSA PJRT shared object");
  if (!library.ok()) return library.status();
  return CanonicalPath(JoinPath(Dirname(*library), kDefaultManifestBasename),
                       "MUSA compiler bundle manifest");
}

absl::StatusOr<std::string> ReadManifest(absl::string_view path) {
  ScopedFd fd(open(std::string(path).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (fd.get() < 0) {
    return absl::NotFoundError(absl::StrCat(
        "cannot open MUSA compiler bundle manifest: ", std::strerror(errno)));
  }
  struct stat metadata;
  if (fstat(fd.get(), &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
      metadata.st_size <= 0 ||
      static_cast<size_t>(metadata.st_size) > kMaxManifestBytes) {
    return absl::InvalidArgumentError(
        "MUSA compiler bundle manifest is not a bounded regular file");
  }
  std::string contents(static_cast<size_t>(metadata.st_size), '\0');
  size_t offset = 0;
  while (offset < contents.size()) {
    const ssize_t count =
        read(fd.get(), contents.data() + offset, contents.size() - offset);
    if (count > 0) {
      offset += static_cast<size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return absl::InternalError("failed to read MUSA compiler bundle manifest");
  }
  return contents;
}

absl::StatusOr<std::array<std::string, kManifestFields.size()>> ParseManifest(
    absl::string_view contents) {
  std::array<std::string, kManifestFields.size()> values;
  for (size_t index = 0; index < kManifestFields.size(); ++index) {
    const size_t newline = contents.find('\n');
    const absl::string_view line = contents.substr(0, newline);
    const std::string prefix = absl::StrCat(kManifestFields[index], "=");
    if (!absl::StartsWith(line, prefix)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "MUSA compiler bundle expected field ", kManifestFields[index]));
    }
    const absl::string_view value = line.substr(prefix.size());
    if (!IsSafeManifestValue(value)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "MUSA compiler bundle has an unsafe value for ",
          kManifestFields[index]));
    }
    values[index] = std::string(value);
    if (newline == absl::string_view::npos) {
      contents = {};
    } else {
      contents.remove_prefix(newline + 1);
    }
  }
  if (!contents.empty()) {
    return absl::InvalidArgumentError(
        "MUSA compiler bundle has unknown or duplicate fields");
  }
  if (values[0] != kMusaCompilerBundleSchema) {
    return absl::InvalidArgumentError(
        "unsupported MUSA compiler bundle schema");
  }
  return values;
}

absl::StatusOr<std::string> ResolveBundleFile(absl::string_view directory,
                                              absl::string_view relative,
                                              bool executable) {
  if (!IsSafeRelativePath(relative)) {
    return absl::InvalidArgumentError(
        "MUSA compiler bundle tool paths must be safe and relative");
  }
  absl::StatusOr<std::string> resolved =
      CanonicalPath(JoinPath(directory, relative), "MUSA compiler bundle file");
  if (!resolved.ok()) return resolved.status();
  struct stat metadata;
  if (stat(resolved->c_str(), &metadata) != 0 || !S_ISREG(metadata.st_mode)) {
    return absl::InvalidArgumentError(
        "MUSA compiler bundle component is not a regular file");
  }
  if (executable && access(resolved->c_str(), X_OK) != 0) {
    return absl::PermissionDeniedError(
        "MUSA compiler bridge is not executable");
  }
  return resolved;
}

absl::StatusOr<std::string> OptionalDirectoryEnvironment(
    absl::string_view name, absl::string_view default_value) {
  const char* configured = std::getenv(std::string(name).c_str());
  if (configured == nullptr || configured[0] == '\0') {
    return std::string(default_value);
  }
  return CanonicalPath(configured, name);
}

}  // namespace

absl::StatusOr<std::unique_ptr<MusaCompilationProvider>>
LoadMusaCompilationProviderFromBundle(absl::string_view manifest_path) {
  absl::StatusOr<std::string> canonical_manifest = manifest_path.empty()
                                                       ? DefaultManifestPath()
                                                       : CanonicalPath(
                                                             manifest_path,
                                                             "MUSA compiler "
                                                             "bundle manifest");
  if (!canonical_manifest.ok()) return canonical_manifest.status();
  absl::StatusOr<std::string> contents = ReadManifest(*canonical_manifest);
  if (!contents.ok()) return contents.status();
  absl::StatusOr<std::array<std::string, kManifestFields.size()>> values =
      ParseManifest(*contents);
  if (!values.ok()) return values.status();

  MusaCompilationProviderSelection selection;
  MusaCompilationIdentity& identity = selection.subprocess.identity;
  identity.xla_revision = (*values)[1];
  identity.current_llvm_revision = (*values)[2];
  identity.provider_name = (*values)[3];
  identity.provider_fingerprint = (*values)[4];
  identity.bridge_fingerprint = (*values)[5];
  identity.toolchain_fingerprint = (*values)[6];
  identity.libdevice_fingerprint = (*values)[7];
  identity.driver_compatibility = (*values)[8];
  identity.runtime_compatibility = (*values)[9];

  const std::string directory = Dirname(*canonical_manifest);
  MusaSubprocessBridgePaths& paths = selection.subprocess.paths;
  std::string* output_paths[] = {
      &paths.bridge_executable,     &paths.toolchain_identity,
      &paths.libclang_cpp,          &paths.mcc,
      &paths.clang_offload_bundler, &paths.lld,
      &paths.llvm_readobj,          &paths.libdevice,
      &paths.intrinsics_musa_td,    &paths.builtins_mtgpu_def,
  };
  for (size_t index = 0; index < std::size(output_paths); ++index) {
    absl::StatusOr<std::string> resolved = ResolveBundleFile(
        directory, (*values)[index + 10], /*executable=*/index == 0);
    if (!resolved.ok()) return resolved.status();
    *output_paths[index] = *std::move(resolved);
  }

  absl::StatusOr<std::string> temporary = OptionalDirectoryEnvironment(
      kTemporaryRootEnvironment, "/tmp");
  if (!temporary.ok()) return temporary.status();
  selection.subprocess.temporary_directory_root = *std::move(temporary);
  absl::StatusOr<std::string> cache =
      OptionalDirectoryEnvironment(kCacheDirectoryEnvironment, "");
  if (!cache.ok()) return cache.status();
  selection.subprocess.cache_directory = *std::move(cache);
  return AssembleMusaCompilationProvider(selection);
}

}  // namespace xla::gpu::musa
