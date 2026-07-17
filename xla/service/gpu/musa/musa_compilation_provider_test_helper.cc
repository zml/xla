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
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/musa/protocol.pb.h"

namespace xla::gpu::musa {
namespace {

constexpr size_t kElfHeaderSize = 64;
constexpr size_t kProgramHeaderSize = 56;
constexpr size_t kLoadHeader = kElfHeaderSize;
constexpr size_t kNoteHeader = kLoadHeader + kProgramHeaderSize;
constexpr size_t kLoadOffset = 192;
constexpr size_t kNoteOffset = 208;

void WriteU16(std::vector<uint8_t>& bytes, size_t offset, uint16_t value) {
  bytes[offset] = value;
  bytes[offset + 1] = value >> 8;
}

void WriteU32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  for (int i = 0; i < 4; ++i) bytes[offset + i] = value >> (8 * i);
}

void WriteU64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
  for (int i = 0; i < 8; ++i) bytes[offset + i] = value >> (8 * i);
}

std::vector<uint8_t> MakeValidMubin() {
  std::vector<uint8_t> bytes(240);
  bytes[0] = 0x7f;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = 2;
  bytes[5] = 1;
  bytes[6] = 1;
  bytes[8] = 7;
  WriteU16(bytes, 16, 3);
  WriteU16(bytes, 18, 253);
  WriteU32(bytes, 20, 1);
  WriteU64(bytes, 32, kElfHeaderSize);
  WriteU16(bytes, 52, kElfHeaderSize);
  WriteU16(bytes, 54, kProgramHeaderSize);
  WriteU16(bytes, 56, 2);

  WriteU32(bytes, kLoadHeader, 1);
  WriteU32(bytes, kLoadHeader + 4, 5);
  WriteU64(bytes, kLoadHeader + 8, kLoadOffset);
  WriteU64(bytes, kLoadHeader + 32, 8);
  WriteU64(bytes, kLoadHeader + 40, 8);

  WriteU32(bytes, kNoteHeader, 4);
  WriteU32(bytes, kNoteHeader + 4, 4);
  WriteU64(bytes, kNoteHeader + 8, kNoteOffset);
  WriteU64(bytes, kNoteHeader + 32, 24);
  WriteU64(bytes, kNoteHeader + 40, 24);

  WriteU32(bytes, kNoteOffset, 6);
  WriteU32(bytes, kNoteOffset + 4, 4);
  WriteU32(bytes, kNoteOffset + 8, 0x40);
  const std::string owner = "MTGPU";
  for (size_t i = 0; i < owner.size(); ++i) {
    bytes[kNoteOffset + 12 + i] = owner[i];
  }
  bytes[kNoteOffset + 20] = 0xde;
  bytes[kNoteOffset + 21] = 0xad;
  bytes[kNoteOffset + 22] = 0xbe;
  bytes[kNoteOffset + 23] = 0xef;
  return bytes;
}

bool WriteAll(int fd, std::string_view value) {
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

struct State {
  int active = 0;
  int maximum = 0;
  int calls = 0;
};

bool UpdateState(const std::string& path, int active_delta) {
  const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0 || flock(fd, LOCK_EX) != 0) {
    if (fd >= 0) close(fd);
    return false;
  }
  char buffer[128] = {};
  const ssize_t count = pread(fd, buffer, sizeof(buffer) - 1, 0);
  State state;
  if (count > 0) {
    (void)std::sscanf(buffer, "%d %d %d", &state.active, &state.maximum,
                      &state.calls);
  }
  state.active += active_delta;
  if (active_delta > 0) {
    state.maximum = std::max(state.maximum, state.active);
    ++state.calls;
  }
  const std::string value = std::to_string(state.active) + " " +
                            std::to_string(state.maximum) + " " +
                            std::to_string(state.calls) + "\n";
  const bool ok = ftruncate(fd, 0) == 0 &&
                  pwrite(fd, value.data(), value.size(), 0) ==
                      static_cast<ssize_t>(value.size()) &&
                  fsync(fd) == 0;
  (void)flock(fd, LOCK_UN);
  close(fd);
  return ok;
}

std::string ArgumentValue(int argc, char** argv, std::string_view prefix) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (absl::StartsWith(argument, prefix)) {
      return std::string(argument.substr(prefix.size()));
    }
  }
  return "";
}

void FillBoundFields(const MusaBridgeCompileRequest& request,
                     MusaBridgeCompileResponse* response) {
  response->set_protocol_version(request.protocol_version());
  response->set_shim_abi_version(request.shim_abi_version());
  response->set_mapping_version(request.mapping_version());
  response->set_provider_name(request.provider_name());
  response->set_provider_fingerprint(request.provider_fingerprint());
  response->set_bridge_fingerprint(request.bridge_fingerprint());
  response->set_toolchain_fingerprint(request.toolchain_fingerprint());
  response->set_mapping_fingerprint(request.mapping_fingerprint());
  absl::StatusOr<std::string> request_sha =
      MusaBridgeCompileRequestSha256(request);
  if (request_sha.ok()) response->set_request_sha256(*request_sha);
  MusaBridgeCompileStats* stats = response->mutable_stats();
  stats->set_input_llvm_bytes(request.normalized_llvm_bytes());
  stats->set_kernel_count(request.kernel_entry_names_size());
  stats->set_exported_symbol_count(request.exported_symbol_names_size());
  stats->set_bridge_wall_time_microseconds(1);
  stats->set_peak_memory_bytes(4096);
}

}  // namespace
}  // namespace xla::gpu::musa

int main(int argc, char** argv) {
  using namespace xla::gpu::musa;
  if (argc != 12) return 80;
  const std::string state_path =
      ArgumentValue(argc, argv, "--toolchain-identity=");
  if (state_path.empty() || !UpdateState(state_path, 1)) return 81;
  struct StateGuard {
    std::string path;
    ~StateGuard() { (void)UpdateState(path, -1); }
  } state_guard{state_path};

  const std::string wire(std::istreambuf_iterator<char>(std::cin), {});
  absl::StatusOr<MusaBridgeCompileRequest> request =
      DecodeMusaBridgeCompileRequest(wire);
  if (!request.ok()) return 82;
  const std::string& mode = request->module_name();

  if (mode == "crash") std::raise(SIGABRT);
  if (mode == "malformed") {
    return WriteAll(STDOUT_FILENO, "not-a-canonical-response") ? 0 : 83;
  }
  if (mode == "oversized") {
    return WriteAll(STDOUT_FILENO, std::string(16384, 'x')) ? 0 : 84;
  }
  if (mode == "timeout") {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  if (mode == "cancel" || mode == "slow") {
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }
  if (absl::StartsWith(mode, "concurrent")) {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
  }
  if (mode == "stderr") {
    if (!WriteAll(STDERR_FILENO, "/private/provider/path\n")) return 85;
  }

  MusaBridgeCompileResponse response;
  FillBoundFields(*request, &response);
  MusaBridgeStatus status = MUSA_BRIDGE_STATUS_OK;
  if (mode == "rejected") status = MUSA_BRIDGE_STATUS_REJECTED;
  if (mode == "compile_error") status = MUSA_BRIDGE_STATUS_COMPILATION_ERROR;
  if (mode == "internal_error") status = MUSA_BRIDGE_STATUS_INTERNAL_ERROR;
  response.set_status(status);

  if (status == MUSA_BRIDGE_STATUS_OK) {
    const std::vector<uint8_t> bytes = mode == "invalid_mubin"
                                           ? std::vector<uint8_t>{1, 2, 3, 4}
                                           : MakeValidMubin();
    response.set_mubin(bytes.data(), bytes.size());
    response.set_mubin_sha256(MusaBridgeSha256Hex(response.mubin()));
    response.mutable_stats()->set_output_mubin_bytes(bytes.size());
  } else {
    MusaBridgeDiagnostic* diagnostic = response.add_diagnostics();
    diagnostic->set_severity(MUSA_BRIDGE_DIAGNOSTIC_SEVERITY_ERROR);
    diagnostic->set_code("test-failure");
    diagnostic->set_message("controlled provider rejection");
    diagnostic->set_component("test-helper");
    response.mutable_stats()->set_diagnostic_count(1);
  }
  if (mode == "identity_mismatch") {
    response.set_toolchain_fingerprint(std::string(64, 'f'));
  }

  absl::StatusOr<std::string> response_wire =
      EncodeMusaBridgeCompileResponse(response);
  if (!response_wire.ok()) return 86;
  return WriteAll(STDOUT_FILENO, *response_wire) ? 0 : 87;
}
