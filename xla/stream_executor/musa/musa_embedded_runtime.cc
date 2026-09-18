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

#include "xla/stream_executor/musa/musa_embedded_runtime.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <openssl/sha.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#ifdef XLA_MUSA_EMBED_RUNTIME
#include "xla/stream_executor/musa/musa_embedded_runtime_data.h"
#endif

namespace stream_executor::musa::internal {
namespace {

struct Component {
  absl::string_view name;
  absl::string_view relative;
  bool executable;
};
constexpr std::array<Component, 5> kComponents = {{
    {"musa-llvm-bridge", "musa-compiler/musa-llvm-bridge", true},
    {"libmusa_llvm_pass_guard.so", "musa-compiler/libmusa_llvm_pass_guard.so",
     false},
    {"libxla_musa_mublas_shim.so", "libxla_musa_mublas_shim.so.1", false},
    {"libxla_musa_mudnn_shim.so", "libxla_musa_mudnn_shim.so.1", false},
    {"libxla_musa_mufft_shim.so", "libxla_musa_mufft_shim.so.1", false},
}};

class Fd {
 public:
  explicit Fd(int value) : value_(value) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() {
    if (value_ >= 0) close(value_);
  }
  int get() const { return value_; }

 private:
  int value_;
};

absl::Status FileError(absl::string_view operation, absl::string_view path) {
  const int error = errno;
  const std::string message =
      absl::StrCat("MUSA embedded runtime: ", operation, " ", path, ": ",
                   std::strerror(error));
  if (error == EACCES || error == EPERM || error == EROFS) {
    return absl::PermissionDeniedError(message);
  }
  return absl::InternalError(message);
}

absl::StatusOr<std::string> CanonicalDirectory(absl::string_view path) {
  if (path.empty() || path.front() != '/' ||
      path.find('\0') != absl::string_view::npos) {
    return absl::InvalidArgumentError(
        "MUSA runtime directory must be absolute");
  }
  char* resolved = realpath(std::string(path).c_str(), nullptr);
  if (resolved == nullptr) return FileError("resolve directory", path);
  std::string result(resolved);
  std::free(resolved);
  struct stat metadata;
  if (stat(result.c_str(), &metadata) != 0) return FileError("stat", result);
  if (!S_ISDIR(metadata.st_mode)) {
    return absl::InvalidArgumentError(
        absl::StrCat("not a MUSA runtime directory: ", result));
  }
  return result;
}

absl::StatusOr<std::unique_ptr<MusaEmbeddedRuntime>> CreateDefaultRuntime(
    absl::string_view sdk_override) {
#ifdef XLA_MUSA_EMBED_RUNTIME
  const char* configured = std::getenv("MUSA_PATH");
  std::string sdk;
  if (!sdk_override.empty()) {
    sdk = std::string(sdk_override);
  } else if (configured != nullptr && configured[0] != '\0') {
    sdk = configured;
  } else {
    Dl_info info = {};
    if (dladdr(reinterpret_cast<void*>(&GetMusaEmbeddedRuntime), &info) == 0 ||
        info.dli_fname == nullptr) {
      return absl::InternalError("cannot locate the MUSA PJRT shared object");
    }
    char* resolved = realpath(info.dli_fname, nullptr);
    if (resolved == nullptr) return FileError("resolve PJRT", info.dli_fname);
    const std::string plugin(resolved);
    std::free(resolved);
    sdk = absl::StrCat(plugin.substr(0, plugin.rfind('/')), "/musa-sdk");
  }
  const char* temporary = std::getenv("XLA_MUSA_COMPILATION_TEMP_ROOT");
  return MusaEmbeddedRuntime::Create(
      sdk, temporary != nullptr && temporary[0] != '\0' ? temporary : "/tmp",
      absl::MakeConstSpan(musa_embedded_runtime_data_create(),
                          musa_embedded_runtime_data_size()));
#else
  return absl::UnavailableError("MUSA helpers are not embedded in this build");
#endif
}

}  // namespace

MusaEmbeddedRuntime::MusaEmbeddedRuntime(std::string directory,
                                         std::string sdk_root)
    : directory_(std::move(directory)), sdk_root_(std::move(sdk_root)) {}

MusaEmbeddedRuntime::~MusaEmbeddedRuntime() {
  // Only unlink known entries. In particular, never traverse the SDK symlink.
  for (const Component& component : kComponents) {
    unlink(Path(component.relative).c_str());
  }
  unlink(Path("musa-sdk").c_str());
  rmdir(Path("musa-compiler").c_str());
  rmdir(directory_.c_str());
}

std::string MusaEmbeddedRuntime::Path(absl::string_view relative) const {
  return absl::StrCat(directory_, "/", relative);
}

absl::StatusOr<std::unique_ptr<MusaEmbeddedRuntime>>
MusaEmbeddedRuntime::Create(absl::string_view sdk_root,
                            absl::string_view temporary_root,
                            absl::Span<const FileToc> files) {
  if (files.size() != kComponents.size()) {
    return absl::InvalidArgumentError(
        "MUSA embedded runtime requires five components");
  }
  auto sdk = CanonicalDirectory(sdk_root);
  if (!sdk.ok()) return sdk.status();
  auto temporary = CanonicalDirectory(temporary_root);
  if (!temporary.ok()) return temporary.status();
  struct statvfs mount;
  if (statvfs(temporary->c_str(), &mount) != 0)
    return FileError("statvfs", *temporary);
  if ((mount.f_flag & ST_NOEXEC) != 0) {
    return absl::PermissionDeniedError(absl::StrCat(
        "MUSA helper extraction requires an executable filesystem: ",
        *temporary,
        "; set XLA_MUSA_COMPILATION_TEMP_ROOT to an executable directory"));
  }
  std::string directory = absl::StrCat(*temporary, "/xla-musa-XXXXXX");
  if (mkdtemp(directory.data()) == nullptr)
    return FileError("create private directory", directory);
  auto runtime = std::unique_ptr<MusaEmbeddedRuntime>(
      new MusaEmbeddedRuntime(directory, *sdk));
  if (mkdir(runtime->Path("musa-compiler").c_str(), 0700) != 0) {
    return FileError("create compiler directory", directory);
  }
  if (symlink(sdk->c_str(), runtime->Path("musa-sdk").c_str()) != 0) {
    return FileError("link SDK", *sdk);
  }
  for (const Component& component : kComponents) {
    const FileToc* input = nullptr;
    for (const FileToc& file : files) {
      if (file.name != nullptr && file.name == component.name) {
        if (input != nullptr)
          return absl::InvalidArgumentError("duplicate MUSA payload component");
        input = &file;
      }
    }
    if (input == nullptr || input->data == nullptr || input->size == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("missing MUSA payload component: ", component.name));
    }
    const std::string path = runtime->Path(component.relative);
    Fd fd(open(path.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    if (fd.get() < 0) return FileError("create component", path);
    size_t offset = 0;
    while (offset < input->size) {
      const ssize_t count =
          write(fd.get(), input->data + offset, input->size - offset);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return FileError("write component", path);
      offset += count;
    }
    if (fchmod(fd.get(), component.executable ? 0500 : 0400) != 0) {
      return FileError("set component permissions", path);
    }
    ExtractedFile extracted{path, input->size, {}};
    SHA256(reinterpret_cast<const unsigned char*>(input->data), input->size,
           extracted.sha256.data());
    runtime->files_.push_back(std::move(extracted));
  }
  absl::Status verified = runtime->Verify();
  if (!verified.ok()) return verified;
  return runtime;
}

absl::Status MusaEmbeddedRuntime::Verify() const {
  for (const ExtractedFile& file : files_) {
    Fd fd(open(file.path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
    if (fd.get() < 0) return FileError("verify component", file.path);
    struct stat metadata;
    if (fstat(fd.get(), &metadata) != 0)
      return FileError("stat component", file.path);
    if (!S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
        static_cast<size_t>(metadata.st_size) != file.size) {
      return absl::DataLossError(
          absl::StrCat("MUSA component size changed: ", file.path));
    }
    SHA256_CTX digest;
    SHA256_Init(&digest);
    std::array<unsigned char, 64 << 10> buffer;
    size_t total = 0;
    for (;;) {
      const ssize_t count = read(fd.get(), buffer.data(), buffer.size());
      if (count < 0 && errno == EINTR) continue;
      if (count < 0) return FileError("read component", file.path);
      if (count == 0) break;
      total += count;
      if (total > file.size)
        return absl::DataLossError("MUSA component grew during verification");
      SHA256_Update(&digest, buffer.data(), count);
    }
    std::array<unsigned char, 32> actual;
    SHA256_Final(actual.data(), &digest);
    if (total != file.size || actual != file.sha256) {
      return absl::DataLossError(
          absl::StrCat("MUSA component digest mismatch: ", file.path));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<const MusaEmbeddedRuntime*> GetMusaEmbeddedRuntime(
    absl::string_view sdk_root) {
  static const auto runtime = CreateDefaultRuntime(sdk_root);
  if (!runtime.ok()) return runtime.status();
  if (!sdk_root.empty()) {
    auto canonical = CanonicalDirectory(sdk_root);
    if (!canonical.ok()) return canonical.status();
    if (*canonical != (*runtime)->sdk_root()) {
      return absl::FailedPreconditionError(
          "MUSA SDK changed after runtime initialization");
    }
  }
  return runtime->get();
}

}  // namespace stream_executor::musa::internal
