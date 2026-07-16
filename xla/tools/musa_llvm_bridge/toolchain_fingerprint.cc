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

#include "xla/tools/musa_llvm_bridge/toolchain_fingerprint.h"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include <fcntl.h>
#include <openssl/sha.h>
#include <sys/stat.h>
#include <unistd.h>
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"

namespace xla::gpu::musa::bridge {
namespace {

constexpr uint64_t kMaxToolBytes = uint64_t{1} << 30;
constexpr uint64_t kMaxSourceContractBytes = uint64_t{64} << 20;
constexpr uint64_t kMaxIdentityBytes = uint64_t{1} << 20;
constexpr size_t kMaxProviderContractBytes = 64 << 10;

struct HashedFile {
  uint64_t size_bytes;
  std::string sha256;
  std::string contents;
};

class Fd {
 public:
  explicit Fd(int fd) : fd_(fd) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() {
    // Linux releases the descriptor even when close reports EINTR. Retrying
    // in a multithreaded process could close an unrelated, newly reused fd.
    if (fd_ >= 0) close(fd_);
  }
  int get() const { return fd_; }

 private:
  int fd_;
};

std::string DigestHex(const uint8_t* digest, size_t size) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(size * 2, '0');
  for (size_t i = 0; i < size; ++i) {
    result[2 * i] = kHex[digest[i] >> 4];
    result[2 * i + 1] = kHex[digest[i] & 0x0f];
  }
  return result;
}

absl::StatusOr<HashedFile> HashFile(absl::string_view path, uint64_t max_bytes,
                                    bool retain_contents) {
  if (path.empty() || path.front() != '/' ||
      path.find('\0') != absl::string_view::npos) {
    return absl::InvalidArgumentError(
        "MUSA fingerprint input must be an absolute NUL-free path");
  }
  char* resolved_raw = realpath(std::string(path).c_str(), nullptr);
  if (resolved_raw == nullptr) {
    return absl::NotFoundError(absl::StrCat(
        "cannot resolve MUSA fingerprint input: ", std::strerror(errno)));
  }
  std::string resolved(resolved_raw);
  std::free(resolved_raw);
  Fd fd(open(resolved.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (fd.get() < 0) {
    return absl::NotFoundError(absl::StrCat(
        "cannot open MUSA fingerprint input: ", std::strerror(errno)));
  }
  struct stat before;
  if (fstat(fd.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 ||
      static_cast<uint64_t>(before.st_size) > max_bytes) {
    return absl::InvalidArgumentError(
        "MUSA fingerprint input is not a bounded nonempty regular file");
  }

  SHA256_CTX context;
  SHA256_Init(&context);
  std::string contents;
  if (retain_contents) contents.reserve(before.st_size);
  std::array<char, 64 << 10> buffer;
  uint64_t total = 0;
  while (true) {
    const ssize_t count = read(fd.get(), buffer.data(), buffer.size());
    if (count > 0) {
      total += count;
      if (total > max_bytes) {
        return absl::ResourceExhaustedError(
            "MUSA fingerprint input grew beyond its bound");
      }
      SHA256_Update(&context, buffer.data(), count);
      if (retain_contents) contents.append(buffer.data(), count);
      continue;
    }
    if (count == 0) break;
    if (errno == EINTR) continue;
    return absl::InternalError(absl::StrCat(
        "cannot read MUSA fingerprint input: ", std::strerror(errno)));
  }
  struct stat after;
  if (fstat(fd.get(), &after) != 0 || after.st_dev != before.st_dev ||
      after.st_ino != before.st_ino || after.st_size != before.st_size ||
      after.st_mtim.tv_sec != before.st_mtim.tv_sec ||
      after.st_mtim.tv_nsec != before.st_mtim.tv_nsec ||
      after.st_ctim.tv_sec != before.st_ctim.tv_sec ||
      after.st_ctim.tv_nsec != before.st_ctim.tv_nsec ||
      total != static_cast<uint64_t>(before.st_size)) {
    return absl::AbortedError(
        "MUSA fingerprint input changed while it was being hashed");
  }
  std::array<uint8_t, SHA256_DIGEST_LENGTH> digest;
  SHA256_Final(digest.data(), &context);
  return HashedFile{.size_bytes = total,
                    .sha256 = DigestHex(digest.data(), digest.size()),
                    .contents = std::move(contents)};
}

bool IsToken(absl::string_view value) {
  if (value.empty() || value.size() > 128 ||
      !(absl::ascii_isalpha(value.front()) || value.front() == '_')) {
    return false;
  }
  for (unsigned char c : value) {
    if (!absl::ascii_isalnum(c) && c != '_' && c != '-' && c != '.') {
      return false;
    }
  }
  return true;
}

absl::Status ValidateIdentity(absl::string_view identity) {
  if (!absl::StartsWith(identity, "schema=xla-musa-toolchain-v1\n") ||
      identity.back() != '\n' ||
      identity.find('\r') != absl::string_view::npos ||
      identity.find('\0') != absl::string_view::npos) {
    return absl::InvalidArgumentError(
        "MUSA toolchain identity is not canonical schema v1 text");
  }
  absl::flat_hash_set<absl::string_view> names;
  for (absl::string_view line : absl::StrSplit(identity, '\n')) {
    if (line.empty()) continue;
    const size_t separator = line.find('=');
    if (separator == absl::string_view::npos ||
        !IsToken(line.substr(0, separator)) ||
        !names.insert(line.substr(0, separator)).second) {
      return absl::InvalidArgumentError(
          "MUSA toolchain identity has an invalid or duplicate field");
    }
    const absl::string_view value = line.substr(separator + 1);
    if (value.size() > 4096 || absl::StartsWith(value, "/") ||
        value.find("..") != absl::string_view::npos) {
      return absl::InvalidArgumentError(
          "MUSA toolchain identity contains an unsafe value");
    }
    for (unsigned char c : value) {
      if (c < 0x20 || c == 0x7f) {
        return absl::InvalidArgumentError(
            "MUSA toolchain identity contains a control byte");
      }
    }
  }
  return absl::OkStatus();
}

void AppendComponent(std::string* manifest,
                     const MusaBridgeComponentFingerprint& component) {
  absl::StrAppend(manifest, "component\t", component.name, "\t",
                  component.size_bytes, "\t", component.sha256, "\n");
}

}  // namespace

absl::StatusOr<MusaBridgeFingerprints> FingerprintMusaBridgeToolchain(
    const MusaBridgeToolchainPaths& paths, std::string provider_name,
    std::string provider_contract) {
  if (!IsToken(provider_name) || provider_contract.empty() ||
      provider_contract.size() > kMaxProviderContractBytes ||
      provider_contract.back() != '\n' ||
      provider_contract.find('\r') != provider_contract.npos ||
      provider_contract.find('\0') != provider_contract.npos) {
    return absl::InvalidArgumentError(
        "MUSA provider identity or contract is not canonical");
  }

  const std::pair<absl::string_view, absl::string_view> component_paths[] = {
      {"bridge", paths.bridge_executable},
      {"toolchain_identity", paths.toolchain_identity},
      {"libclang_cpp", paths.libclang_cpp},
      {"mcc", paths.mcc},
      {"clang_offload_bundler", paths.clang_offload_bundler},
      {"lld", paths.lld},
      {"llvm_readobj", paths.llvm_readobj},
      {"libdevice", paths.libdevice},
      {"intrinsics_musa_td", paths.intrinsics_musa_td},
      {"builtins_mtgpu_def", paths.builtins_mtgpu_def},
  };

  MusaBridgeFingerprints result;
  result.provider_name = std::move(provider_name);
  result.components.reserve(std::size(component_paths));
  std::string identity_contents;
  for (const auto& [name, path] : component_paths) {
    const bool identity = name == "toolchain_identity";
    const bool source_contract =
        name == "intrinsics_musa_td" || name == "builtins_mtgpu_def";
    const uint64_t limit = identity          ? kMaxIdentityBytes
                           : source_contract ? kMaxSourceContractBytes
                                             : kMaxToolBytes;
    absl::StatusOr<HashedFile> hashed = HashFile(path, limit, identity);
    if (!hashed.ok()) {
      return absl::Status(
          hashed.status().code(),
          absl::StrCat("component ", name, ": ", hashed.status().message()));
    }
    if (identity) identity_contents = hashed->contents;
    result.components.push_back(MusaBridgeComponentFingerprint{
        .name = std::string(name),
        .size_bytes = hashed->size_bytes,
        .sha256 = std::move(hashed->sha256),
    });
  }
  absl::Status status = ValidateIdentity(identity_contents);
  if (!status.ok()) return status;

  result.bridge_fingerprint = result.components.front().sha256;
  const std::string provider_contract_sha =
      MusaBridgeSha256Hex(provider_contract);
  std::string provider_manifest;
  absl::StrAppend(&provider_manifest, "schema=xla-musa-provider-v1\n",
                  "provider=", result.provider_name, "\n",
                  "contract_sha256=", provider_contract_sha, "\n",
                  "mapping_sha256=", kMusaShimMappingSha256, "\n");
  for (const MusaBridgeComponentFingerprint& component : result.components) {
    if (component.name == "mcc" || component.name == "clang_offload_bundler") {
      AppendComponent(&provider_manifest, component);
    }
  }
  result.provider_fingerprint = MusaBridgeSha256Hex(provider_manifest);

  absl::StrAppend(&result.canonical_toolchain_manifest,
                  "schema=xla-musa-bridge-toolchain-v1\n",
                  "provider=", result.provider_name, "\n",
                  "provider_fingerprint=", result.provider_fingerprint, "\n",
                  "provider_contract_sha256=", provider_contract_sha, "\n",
                  "mapping_sha256=", kMusaShimMappingSha256, "\n");
  for (const MusaBridgeComponentFingerprint& component : result.components) {
    if (component.name != "bridge") {
      AppendComponent(&result.canonical_toolchain_manifest, component);
    }
  }
  result.toolchain_fingerprint =
      MusaBridgeSha256Hex(result.canonical_toolchain_manifest);
  return result;
}

}  // namespace xla::gpu::musa::bridge
