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
#include "xla/service/gpu/musa/mcc_bundle_codegen.h"
#include "xla/stream_executor/musa/musa_embedded_runtime.h"
#include "xla/tools/musa_llvm_bridge/toolchain_fingerprint.h"
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

absl::StatusOr<std::string> CanonicalLayoutPath(absl::string_view path,
                                                absl::string_view description,
                                                bool directory = false,
                                                bool executable = false) {
  if (path.empty() || path.front() != '/' || path.size() > kMaxPathBytes) {
    return absl::InvalidArgumentError(
        absl::StrCat(description, " must be an absolute safe path"));
  }
  for (unsigned char c : path) {
    if (c < 0x20 || c == 0x7f) {
      return absl::InvalidArgumentError(
          absl::StrCat(description, " contains a control byte"));
    }
  }
  char* resolved_raw = realpath(std::string(path).c_str(), nullptr);
  if (resolved_raw == nullptr) {
    return absl::NotFoundError(absl::StrCat("cannot resolve ", description,
                                            ": ", std::strerror(errno)));
  }
  std::string resolved(resolved_raw);
  std::free(resolved_raw);
  struct stat metadata;
  if (stat(resolved.c_str(), &metadata) != 0 ||
      (directory ? !S_ISDIR(metadata.st_mode) : !S_ISREG(metadata.st_mode))) {
    return absl::InvalidArgumentError(
        absl::StrCat(description, " has the wrong filesystem type"));
  }
  if (access(resolved.c_str(), R_OK | (executable ? X_OK : 0)) != 0) {
    return absl::PermissionDeniedError(
        absl::StrCat(description, " is not readable/executable"));
  }
  return resolved;
}

absl::StatusOr<std::string> PluginPath() {
  Dl_info info = {};
  if (dladdr(reinterpret_cast<void*>(&LoadMusaCompilationProviderFromBundle),
             &info) == 0 ||
      info.dli_fname == nullptr) {
    return absl::InternalError("failed to locate the MUSA PJRT shared object");
  }
  return CanonicalLayoutPath(info.dli_fname, "MUSA PJRT shared object");
}

absl::StatusOr<std::string> ResolveLayoutFile(absl::string_view root,
                                              absl::string_view relative,
                                              bool executable = false) {
  absl::StatusOr<std::string> resolved = CanonicalLayoutPath(
      JoinPath(root, relative), relative, false, executable);
  if (!resolved.ok()) return resolved.status();
  if (!absl::StartsWith(*resolved, JoinPath(root, ""))) {
    return absl::InvalidArgumentError(
        absl::StrCat("MUSA compiler component escapes its bundle: ", relative));
  }
  return resolved;
}

absl::StatusOr<std::string> DirectoryEnvironment(
    const char* name, absl::string_view default_value) {
  const char* configured = std::getenv(name);
  if (configured == nullptr || configured[0] == '\0') {
    if (default_value.empty()) return std::string();
    return CanonicalLayoutPath(default_value, name, /*directory=*/true);
  }
  return CanonicalLayoutPath(configured, name, /*directory=*/true);
}

absl::Status ValidateQualifiedSdk(absl::string_view identity) {
  // Preserve the qualification previously enforced by ZML's generator.
  // FingerprintMusaBridgeToolchain already rejects duplicate/malformed fields.
  const absl::string_view required[] = {
      "schema=xla-musa-toolchain-v1",
      "musa_version=5.1.0",
      "musa_version_number=50100",
      "musa_device=S4000",
      "musa_gpu_architectures=mp_22",
  };
  const std::string lines = absl::StrCat("\n", identity);
  for (absl::string_view field : required) {
    if (!absl::StrContains(lines, absl::StrCat("\n", field, "\n"))) {
      return absl::FailedPreconditionError(absl::StrCat(
          "MUSA compiler requires the qualified SDK 5.1.0 S4000/mp_22 redist; "
          "missing ",
          field));
    }
  }
  // Both archives contain the same qualified SDK compiler components.
  const absl::string_view qualified_hashes[] = {
      "5407266eab8fe42caee83f6a7a979edeaa9ea6e542e197bf65fb5f394f1980b2",
      "afa05b1e73c4816e063fb695c889e37877599aa021a4ef8dba08998c1f3b1f9f",
  };
  for (absl::string_view hash : qualified_hashes) {
    if (absl::StrContains(lines,
                         absl::StrCat("\ndistro_sha256=", hash, "\n"))) {
      return absl::OkStatus();
    }
  }
  return absl::FailedPreconditionError(
      "MUSA compiler requires the qualified SDK 5.1.0 S4000/mp_22 redist; "
      "unrecognized distro_sha256");
}

}  // namespace

absl::StatusOr<std::unique_ptr<MusaCompilationProvider>>
LoadMusaCompilationProviderFromLayout(absl::string_view plugin_path,
                                      absl::string_view toolkit_root) {
  const bool embedded = plugin_path.empty();
  absl::StatusOr<std::string> plugin =
      plugin_path.empty()
          ? PluginPath()
          : CanonicalLayoutPath(plugin_path, "MUSA PJRT shared object");
  if (!plugin.ok()) return plugin.status();
  const std::string directory = Dirname(*plugin);
  absl::StatusOr<std::string> sdk =
      toolkit_root.empty()
          ? DirectoryEnvironment("MUSA_PATH", JoinPath(directory, "musa-sdk"))
          : CanonicalLayoutPath(toolkit_root, "MUSA toolkit root",
                                /*directory=*/true);
  if (!sdk.ok()) return sdk.status();

  MusaCompilationProviderSelection selection;
  MusaSubprocessBridgePaths& paths = selection.subprocess.paths;
  std::string helper_directory = directory;
  if (embedded) {
    auto runtime = stream_executor::musa::internal::GetMusaEmbeddedRuntime(*sdk);
    if (!runtime.ok()) return runtime.status();
    helper_directory = (*runtime)->directory();
  }
  absl::StatusOr<std::string> bridge_path = ResolveLayoutFile(
      helper_directory, "musa-compiler/musa-llvm-bridge", /*executable=*/true);
  if (!bridge_path.ok()) return bridge_path.status();
  paths.bridge_executable = *std::move(bridge_path);
  constexpr char kPassGuard[] = "musa-compiler/libmusa_llvm_pass_guard.so";
  struct stat guard_metadata;
  if (lstat(JoinPath(helper_directory, kPassGuard).c_str(), &guard_metadata) ==
      0) {
    auto guard =
        ResolveLayoutFile(helper_directory, kPassGuard, /*executable=*/false);
    if (!guard.ok()) return guard.status();
    paths.llvm_pass_plugin = *std::move(guard);
  } else if (errno != ENOENT || embedded) {
    return absl::InternalError("cannot inspect MUSA compiler pass guard");
  }
  struct Component {
    absl::string_view relative;
    std::string* output;
    bool executable = false;
  };
  const Component components[] = {
      {"toolchain-identity.txt", &paths.toolchain_identity},
      {"lib/libclang-cpp.so.14", &paths.libclang_cpp},
      {"bin/clang-14", &paths.mcc, true},
      {"bin/clang-offload-bundler", &paths.clang_offload_bundler, true},
      {"bin/lld", &paths.lld, true},
      {"bin/llvm-readobj", &paths.llvm_readobj, true},
      {"mtgpu/bitcode/libdevice.bc", &paths.libdevice},
      {"include/llvm/IR/IntrinsicsMUSA.td", &paths.intrinsics_musa_td},
      {"include/clang/Basic/BuiltinsMTGPU.def", &paths.builtins_mtgpu_def},
  };
  for (const Component& component : components) {
    absl::StatusOr<std::string> path =
        ResolveLayoutFile(*sdk, component.relative, component.executable);
    if (!path.ok()) return path.status();
    *component.output = *std::move(path);
  }

  const bridge::MusaBridgeToolchainPaths fingerprint_paths = {
      paths.bridge_executable,     paths.toolchain_identity,
      paths.libclang_cpp,          paths.mcc,
      paths.clang_offload_bundler, paths.lld,
      paths.llvm_readobj,          paths.libdevice,
      paths.intrinsics_musa_td,    paths.builtins_mtgpu_def,
      paths.llvm_pass_plugin};
  absl::StatusOr<bridge::MusaBridgeFingerprints> fingerprints =
      bridge::FingerprintMusaBridgeToolchain(
          fingerprint_paths,
          std::string(MccBundleProviderName(!paths.llvm_pass_plugin.empty())),
          std::string(
              MccBundleProviderCanonicalText(!paths.llvm_pass_plugin.empty())));
  if (!fingerprints.ok()) return fingerprints.status();
  absl::Status status =
      ValidateQualifiedSdk(fingerprints->toolchain_identity_text);
  if (!status.ok()) return status;

  absl::StatusOr<std::string> plugin_sha =
      bridge::MusaCompilerFileSha256(*plugin);
  if (!plugin_sha.ok()) return plugin_sha.status();
  MusaCompilationIdentity& identity = selection.subprocess.identity;
  identity.xla_revision = *std::move(plugin_sha);
  // LLVM is statically linked into PJRT. Its actual build is covered by the
  // plugin content hash, rather than a separately maintained revision string.
  identity.current_llvm_revision =
      absl::StrCat("linked-", identity.xla_revision);
  identity.provider_name = fingerprints->provider_name;
  identity.provider_fingerprint = fingerprints->provider_fingerprint;
  identity.bridge_fingerprint = fingerprints->bridge_fingerprint;
  identity.toolchain_fingerprint = fingerprints->toolchain_fingerprint;
  for (const bridge::MusaBridgeComponentFingerprint& component :
       fingerprints->components) {
    if (component.name == "libdevice") {
      identity.libdevice_fingerprint = component.sha256;
    }
  }
  identity.driver_compatibility = "musa-driver-5.1-compatible";
  identity.runtime_compatibility = "musa-runtime-5.1.0-compatible";

  absl::StatusOr<std::string> temporary =
      DirectoryEnvironment("XLA_MUSA_COMPILATION_TEMP_ROOT", "/tmp");
  if (!temporary.ok()) return temporary.status();
  selection.subprocess.temporary_directory_root = *std::move(temporary);
  absl::StatusOr<std::string> cache =
      DirectoryEnvironment("XLA_MUSA_COMPILATION_CACHE_DIR", "");
  if (!cache.ok()) return cache.status();
  selection.subprocess.cache_directory = *std::move(cache);
  return AssembleMusaCompilationProvider(selection);
}

absl::StatusOr<std::unique_ptr<MusaCompilationProvider>>
LoadMusaCompilationProviderFromBundle(absl::string_view manifest_path) {
  if (manifest_path.empty()) {
    const char* configured = std::getenv(kMusaCompilerBundleEnvironment);
    if (configured == nullptr || configured[0] == '\0') {
      auto plugin = PluginPath();
      if (!plugin.ok()) return plugin.status();
      const std::string adjacent_manifest =
          JoinPath(Dirname(*plugin), kDefaultManifestBasename);
      struct stat metadata;
      if (lstat(adjacent_manifest.c_str(), &metadata) != 0 && errno == ENOENT) {
        return LoadMusaCompilationProviderFromLayout();
      }
    }
  }
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
