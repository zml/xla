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

// This is a manual S80 qualification test. It deliberately executes the
// standalone vendor-LLVM process as a data dependency: vendor LLVM 14 is never
// loaded into this current-LLVM/StreamExecutor test process.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tools/cpp/runfiles/runfiles.h"
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "xla/service/gpu/musa/bounded_subprocess.h"
#include "xla/service/gpu/musa/mcc_bundle_codegen.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/musa/protocol.pb.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_executor.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/musa_llvm_bridge/toolchain_fingerprint.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu::musa::bridge {
namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;
using ::stream_executor::BlockDim;
using ::stream_executor::DeviceAddressBase;
using ::stream_executor::Kernel;
using ::stream_executor::KernelArgsPackedArray;
using ::stream_executor::KernelLoaderSpec;
using ::stream_executor::ModuleHandle;
using ::stream_executor::MultiModuleLoaderSpec;
using ::stream_executor::Stream;
using ::stream_executor::ThreadDim;
using ::stream_executor::musa::MusaExecutor;

constexpr absl::string_view kPositiveIr = R"llvm(
source_filename = "c06_live_mapping_v1"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

@constant_probe = addrspace(2) constant [4 x i32] [i32 3, i32 5, i32 7, i32 11], align 16
@mutable_probe = addrspace(1) global [4 x i32] [i32 10, i32 20, i32 30, i32 40], align 16
@shared_probe = internal addrspace(3) global [4 x float] undef, align 16

define void @barrier_reverse(ptr addrspace(1) %input, ptr addrspace(1) %output) {
entry:
  %tid = call i32 @__xla_musa_v1_read_tid_x()
  %input_element = getelementptr inbounds float, ptr addrspace(1) %input, i32 %tid
  %value = load float, ptr addrspace(1) %input_element, align 4
  %shared_element = getelementptr inbounds [4 x float], ptr addrspace(3) @shared_probe, i32 0, i32 %tid
  store float %value, ptr addrspace(3) %shared_element, align 4
  call void @__xla_musa_v1_workgroup_barrier()
  %reverse_tid = sub i32 3, %tid
  %reverse_shared_element = getelementptr inbounds [4 x float], ptr addrspace(3) @shared_probe, i32 0, i32 %reverse_tid
  %reverse_value = load float, ptr addrspace(3) %reverse_shared_element, align 4
  %output_element = getelementptr inbounds float, ptr addrspace(1) %output, i32 %tid
  store float %reverse_value, ptr addrspace(1) %output_element, align 4
  ret void
}

define void @global_probe(ptr addrspace(1) %output) {
entry:
  %tid = call i32 @__xla_musa_v1_read_tid_x()
  %mutable_element = getelementptr inbounds [4 x i32], ptr addrspace(1) @mutable_probe, i32 0, i32 %tid
  %mutable_value = load i32, ptr addrspace(1) %mutable_element, align 4
  %constant_element = getelementptr inbounds [4 x i32], ptr addrspace(2) @constant_probe, i32 0, i32 %tid
  %constant_value = load i32, ptr addrspace(2) %constant_element, align 4
  %mutable_output = getelementptr inbounds i32, ptr addrspace(1) %output, i32 %tid
  store i32 %mutable_value, ptr addrspace(1) %mutable_output, align 4
  %constant_index = add i32 %tid, 4
  %constant_output = getelementptr inbounds i32, ptr addrspace(1) %output, i32 %constant_index
  store i32 %constant_value, ptr addrspace(1) %constant_output, align 4
  ret void
}

define void @mapping_probe(ptr addrspace(1) %output) {
entry:
  %ctaid_x = call i32 @__xla_musa_v1_read_ctaid_x()
  %ctaid_y = call i32 @__xla_musa_v1_read_ctaid_y()
  %ctaid_z = call i32 @__xla_musa_v1_read_ctaid_z()
  %nctaid_x = call i32 @__xla_musa_v1_read_nctaid_x()
  %nctaid_y = call i32 @__xla_musa_v1_read_nctaid_y()
  %nctaid_z = call i32 @__xla_musa_v1_read_nctaid_z()
  %ntid_x = call i32 @__xla_musa_v1_read_ntid_x()
  %ntid_y = call i32 @__xla_musa_v1_read_ntid_y()
  %ntid_z = call i32 @__xla_musa_v1_read_ntid_z()
  %tid_x = call i32 @__xla_musa_v1_read_tid_x()
  %tid_y = call i32 @__xla_musa_v1_read_tid_y()
  %tid_z = call i32 @__xla_musa_v1_read_tid_z()
  call void @__xla_musa_v1_workgroup_barrier()
  %block_x_selected = icmp eq i32 %ctaid_x, 1
  %block_y_selected = icmp eq i32 %ctaid_y, 1
  %block_z_selected = icmp eq i32 %ctaid_z, 1
  %thread_x_selected = icmp eq i32 %tid_x, 1
  %thread_y_selected = icmp eq i32 %tid_y, 1
  %thread_z_selected = icmp eq i32 %tid_z, 1
  %selected_block_xy = and i1 %block_x_selected, %block_y_selected
  %selected_block = and i1 %selected_block_xy, %block_z_selected
  %selected_thread_xy = and i1 %thread_x_selected, %thread_y_selected
  %selected_thread = and i1 %selected_thread_xy, %thread_z_selected
  %selected_lane = and i1 %selected_block, %selected_thread
  br i1 %selected_lane, label %write, label %exit

write:
  %v0 = zext i32 %ctaid_x to i64
  %out0 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 0
  store i64 %v0, ptr addrspace(1) %out0, align 8
  %v1 = zext i32 %ctaid_y to i64
  %out1 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 1
  store i64 %v1, ptr addrspace(1) %out1, align 8
  %v2 = zext i32 %ctaid_z to i64
  %out2 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 2
  store i64 %v2, ptr addrspace(1) %out2, align 8
  %v3 = zext i32 %nctaid_x to i64
  %out3 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 3
  store i64 %v3, ptr addrspace(1) %out3, align 8
  %v4 = zext i32 %nctaid_y to i64
  %out4 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 4
  store i64 %v4, ptr addrspace(1) %out4, align 8
  %v5 = zext i32 %nctaid_z to i64
  %out5 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 5
  store i64 %v5, ptr addrspace(1) %out5, align 8
  %v6 = zext i32 %ntid_x to i64
  %out6 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 6
  store i64 %v6, ptr addrspace(1) %out6, align 8
  %v7 = zext i32 %ntid_y to i64
  %out7 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 7
  store i64 %v7, ptr addrspace(1) %out7, align 8
  %v8 = zext i32 %ntid_z to i64
  %out8 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 8
  store i64 %v8, ptr addrspace(1) %out8, align 8
  %v9 = zext i32 %tid_x to i64
  %out9 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 9
  store i64 %v9, ptr addrspace(1) %out9, align 8
  %v10 = zext i32 %tid_y to i64
  %out10 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 10
  store i64 %v10, ptr addrspace(1) %out10, align 8
  %v11 = zext i32 %tid_z to i64
  %out11 = getelementptr inbounds i64, ptr addrspace(1) %output, i32 11
  store i64 %v11, ptr addrspace(1) %out11, align 8
  br label %exit

exit:
  ret void
}

define void @sqrt_probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
entry:
  %tid = call i32 @__xla_musa_v1_read_tid_x()
  %input_element = getelementptr inbounds float, ptr addrspace(1) %input, i32 %tid
  %value = load float, ptr addrspace(1) %input_element, align 4
  %result = call float @llvm.sqrt.f32(float %value)
  %output_element = getelementptr inbounds float, ptr addrspace(1) %output, i32 %tid
  store float %result, ptr addrspace(1) %output_element, align 4
  ret void
}

define void @vector_add(ptr addrspace(1) %left, ptr addrspace(1) %right, ptr addrspace(1) %output, i64 %size) {
entry:
  %block = call i32 @__xla_musa_v1_read_ctaid_x()
  %block64 = zext i32 %block to i64
  %width = call i32 @__xla_musa_v1_read_ntid_x()
  %width64 = zext i32 %width to i64
  %block_offset = mul i64 %block64, %width64
  %thread = call i32 @__xla_musa_v1_read_tid_x()
  %thread64 = zext i32 %thread to i64
  %index = add i64 %block_offset, %thread64
  %in_bounds = icmp ult i64 %index, %size
  br i1 %in_bounds, label %body, label %exit

body:
  %left_element = getelementptr inbounds float, ptr addrspace(1) %left, i64 %index
  %right_element = getelementptr inbounds float, ptr addrspace(1) %right, i64 %index
  %output_element = getelementptr inbounds float, ptr addrspace(1) %output, i64 %index
  %left_value = load float, ptr addrspace(1) %left_element, align 4
  %right_value = load float, ptr addrspace(1) %right_element, align 4
  %sum = fadd float %left_value, %right_value
  store float %sum, ptr addrspace(1) %output_element, align 4
  br label %exit

exit:
  ret void
}

declare i32 @__xla_musa_v1_read_ctaid_x() #0
declare i32 @__xla_musa_v1_read_ctaid_y() #0
declare i32 @__xla_musa_v1_read_ctaid_z() #0
declare i32 @__xla_musa_v1_read_nctaid_x() #1
declare i32 @__xla_musa_v1_read_nctaid_y() #1
declare i32 @__xla_musa_v1_read_nctaid_z() #1
declare i32 @__xla_musa_v1_read_ntid_x() #1
declare i32 @__xla_musa_v1_read_ntid_y() #1
declare i32 @__xla_musa_v1_read_ntid_z() #1
declare i32 @__xla_musa_v1_read_tid_x() #1
declare i32 @__xla_musa_v1_read_tid_y() #1
declare i32 @__xla_musa_v1_read_tid_z() #1
declare void @__xla_musa_v1_workgroup_barrier() #2
declare float @llvm.sqrt.f32(float) #3

attributes #0 = { convergent nounwind readnone }
attributes #1 = { nounwind readnone }
attributes #2 = { convergent nounwind }
attributes #3 = { nofree nosync nounwind readnone speculatable willreturn }
)llvm";

constexpr absl::string_view kAtomicIr = R"llvm(
source_filename = "c06_atomic_negative"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @atomic_probe(ptr addrspace(1) %output) {
entry:
  %old = atomicrmw add ptr addrspace(1) %output, i32 1 monotonic
  ret void
}
)llvm";

constexpr absl::string_view kClock32Ir = R"llvm(
source_filename = "c06_clock32_provider_negative"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @clock_probe(ptr addrspace(1) %output) {
entry:
  %clock = call i32 @__xla_musa_v1_clock32()
  store i32 %clock, ptr addrspace(1) %output, align 4
  ret void
}

declare i32 @__xla_musa_v1_clock32() #0
attributes #0 = { inaccessiblememonly nounwind }
)llvm";

constexpr absl::string_view kClock64Ir = R"llvm(
source_filename = "c06_clock64_provider_negative"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @clock_probe(ptr addrspace(1) %output) {
entry:
  %clock = call i64 @__xla_musa_v1_clock64()
  store i64 %clock, ptr addrspace(1) %output, align 8
  ret void
}

declare i64 @__xla_musa_v1_clock64() #0
attributes #0 = { inaccessiblememonly nounwind }
)llvm";

struct BridgeProcessResult {
  int exit_code = -1;
  int terminating_signal = 0;
  bool timed_out = false;
  bool output_limit_exceeded = false;
  std::string stdout_text;
  std::string stderr_text;
};

struct QualifiedToolchain {
  MusaBridgeToolchainPaths paths;
  MusaBridgeFingerprints fingerprints;
};

std::string RequiredEnvironment(absl::string_view name) {
  const char* value = std::getenv(std::string(name).c_str());
  EXPECT_NE(value, nullptr) << name;
  EXPECT_NE(value == nullptr ? '\0' : value[0], '\0') << name;
  return value == nullptr ? "" : value;
}

std::string ResolveRunfile(Runfiles& runfiles, absl::string_view variable) {
  const std::string logical = RequiredEnvironment(variable);
  const std::string resolved = runfiles.Rlocation(logical);
  EXPECT_FALSE(resolved.empty()) << variable << "=" << logical;
  EXPECT_TRUE(std::filesystem::is_regular_file(resolved))
      << variable << "=" << resolved;
  return resolved;
}

absl::StatusOr<std::string> MakeTemporaryRoot(absl::string_view suffix) {
  const std::string parent = RequiredEnvironment("TEST_TMPDIR");
  std::string pattern =
      absl::StrCat(parent, "/musa-bridge-live-", suffix, "-XXXXXX");
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char* directory = mkdtemp(writable.data());
  if (directory == nullptr) {
    return absl::InternalError(
        absl::StrCat("mkdtemp failed: ", std::strerror(errno)));
  }
  return std::string(directory);
}

absl::Status SetNonBlocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
    return absl::InternalError(
        absl::StrCat("fcntl failed: ", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::Status DrainFd(int* fd, size_t limit, std::string* output,
                     bool* overflow) {
  std::array<char, 8192> buffer;
  while (*fd >= 0) {
    const ssize_t count = read(*fd, buffer.data(), buffer.size());
    if (count > 0) {
      if (output->size() > limit ||
          static_cast<size_t>(count) > limit - output->size()) {
        *overflow = true;
        close(*fd);
        *fd = -1;
        return absl::OkStatus();
      }
      output->append(buffer.data(), count);
      continue;
    }
    if (count == 0) {
      close(*fd);
      *fd = -1;
      return absl::OkStatus();
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return absl::OkStatus();
    return absl::InternalError(
        absl::StrCat("bridge pipe read failed: ", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::Status WriteRequest(int fd, absl::string_view request) {
  while (!request.empty()) {
    const ssize_t count = write(fd, request.data(), request.size());
    if (count > 0) {
      request.remove_prefix(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return absl::InternalError(
        absl::StrCat("bridge request write failed: ", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::StatusOr<BridgeProcessResult> RunBridgeProcess(
    const MusaBridgeToolchainPaths& paths, absl::string_view temporary_root,
    absl::string_view request_wire, const MusaSubprocessLimits& limits) {
  int stdin_pipe[2] = {-1, -1};
  int stdout_pipe[2] = {-1, -1};
  int stderr_pipe[2] = {-1, -1};
  const auto close_fd = [](int* fd) {
    if (*fd >= 0) close(std::exchange(*fd, -1));
  };
  const auto close_all = [&] {
    for (int* fd : {&stdin_pipe[0], &stdin_pipe[1], &stdout_pipe[0],
                    &stdout_pipe[1], &stderr_pipe[0], &stderr_pipe[1]}) {
      close_fd(fd);
    }
  };
  if (pipe2(stdin_pipe, O_CLOEXEC) != 0 || pipe2(stdout_pipe, O_CLOEXEC) != 0 ||
      pipe2(stderr_pipe, O_CLOEXEC) != 0) {
    const int saved_errno = errno;
    close_all();
    return absl::InternalError(
        absl::StrCat("pipe2 failed: ", std::strerror(saved_errno)));
  }

  std::vector<std::string> storage = {
      paths.bridge_executable,
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
      absl::StrCat("--temp-root=", temporary_root),
  };
  std::vector<char*> arguments;
  arguments.reserve(storage.size() + 1);
  for (std::string& value : storage) arguments.push_back(value.data());
  arguments.push_back(nullptr);
  std::array<std::string, 4> environment_storage = {
      "HOME=/nonexistent", "LC_ALL=C", "PATH=/usr/bin:/bin",
      absl::StrCat("TMPDIR=", temporary_root)};
  std::array<char*, 5> environment = {
      environment_storage[0].data(), environment_storage[1].data(),
      environment_storage[2].data(), environment_storage[3].data(), nullptr};

  const pid_t pid = fork();
  if (pid < 0) {
    const int saved_errno = errno;
    close_all();
    return absl::InternalError(
        absl::StrCat("fork failed: ", std::strerror(saved_errno)));
  }
  if (pid == 0) {
    setpgid(0, 0);
    if (dup2(stdin_pipe[0], STDIN_FILENO) < 0 ||
        dup2(stdout_pipe[1], STDOUT_FILENO) < 0 ||
        dup2(stderr_pipe[1], STDERR_FILENO) < 0) {
      _exit(126);
    }
    close(stdin_pipe[0]);
    close(stdin_pipe[1]);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    execve(paths.bridge_executable.c_str(), arguments.data(),
           environment.data());
    _exit(127);
  }

  close_fd(&stdin_pipe[0]);
  close_fd(&stdout_pipe[1]);
  close_fd(&stderr_pipe[1]);
  (void)setpgid(pid, pid);
  const auto kill_and_reap = [&] {
    (void)kill(-pid, SIGKILL);
    (void)kill(pid, SIGKILL);
    while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
    }
  };
  absl::Status write_status = WriteRequest(stdin_pipe[1], request_wire);
  close_fd(&stdin_pipe[1]);
  if (!write_status.ok()) {
    kill_and_reap();
    close_all();
    return write_status;
  }
  if (absl::Status status = SetNonBlocking(stdout_pipe[0]); !status.ok()) {
    kill_and_reap();
    close_all();
    return status;
  }
  if (absl::Status status = SetNonBlocking(stderr_pipe[0]); !status.ok()) {
    kill_and_reap();
    close_all();
    return status;
  }

  const auto deadline = std::chrono::steady_clock::now() + limits.timeout;
  BridgeProcessResult result;
  bool exited = false;
  bool killed = false;
  std::optional<std::chrono::steady_clock::time_point> close_pipes_deadline;
  int wait_status = 0;
  const auto start_kill = [&] {
    if (killed || exited) return;
    (void)kill(-pid, SIGKILL);
    (void)kill(pid, SIGKILL);
    killed = true;
    close_pipes_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
  };
  while (!exited || stdout_pipe[0] >= 0 || stderr_pipe[0] >= 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      result.timed_out = true;
      start_kill();
    }
    if (close_pipes_deadline.has_value() &&
        std::chrono::steady_clock::now() >= *close_pipes_deadline) {
      close_fd(&stdout_pipe[0]);
      close_fd(&stderr_pipe[0]);
    }

    pollfd descriptors[2];
    nfds_t descriptor_count = 0;
    if (stdout_pipe[0] >= 0) {
      descriptors[descriptor_count++] = {
          .fd = stdout_pipe[0], .events = POLLIN | POLLHUP, .revents = 0};
    }
    if (stderr_pipe[0] >= 0) {
      descriptors[descriptor_count++] = {
          .fd = stderr_pipe[0], .events = POLLIN | POLLHUP, .revents = 0};
    }
    if (descriptor_count > 0) {
      const int poll_result = poll(descriptors, descriptor_count, 20);
      if (poll_result < 0 && errno != EINTR) {
        start_kill();
        kill_and_reap();
        close_all();
        return absl::InternalError(
            absl::StrCat("bridge poll failed: ", std::strerror(errno)));
      }
    }

    bool overflow = false;
    if (absl::Status status = DrainFd(&stdout_pipe[0], limits.max_stdout_bytes,
                                      &result.stdout_text, &overflow);
        !status.ok()) {
      start_kill();
      kill_and_reap();
      close_all();
      return status;
    }
    result.output_limit_exceeded |= overflow;
    overflow = false;
    if (absl::Status status = DrainFd(&stderr_pipe[0], limits.max_stderr_bytes,
                                      &result.stderr_text, &overflow);
        !status.ok()) {
      start_kill();
      kill_and_reap();
      close_all();
      return status;
    }
    result.output_limit_exceeded |= overflow;
    if (result.output_limit_exceeded) {
      start_kill();
    }

    if (!exited) {
      const pid_t observed = waitpid(pid, &wait_status, WNOHANG);
      if (observed == pid) {
        exited = true;
        if (stdout_pipe[0] >= 0 || stderr_pipe[0] >= 0) {
          close_pipes_deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(1);
        }
      }
      if (observed < 0 && errno != EINTR) {
        start_kill();
        close_all();
        return absl::InternalError(
            absl::StrCat("waitpid failed: ", std::strerror(errno)));
      }
    }
  }
  if (WIFEXITED(wait_status)) result.exit_code = WEXITSTATUS(wait_status);
  if (WIFSIGNALED(wait_status)) {
    result.terminating_signal = WTERMSIG(wait_status);
  }
  return result;
}

absl::StatusOr<QualifiedToolchain> LocateQualifiedToolchain() {
  std::string error;
  std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));
  if (runfiles == nullptr) {
    return absl::InternalError(absl::StrCat("runfiles init failed: ", error));
  }
  MusaBridgeToolchainPaths paths;
  paths.bridge_executable = ResolveRunfile(*runfiles, "MUSA_BRIDGE_RUNFILE");
  paths.toolchain_identity =
      ResolveRunfile(*runfiles, "MUSA_TOOLCHAIN_IDENTITY_RUNFILE");
  paths.mcc = ResolveRunfile(*runfiles, "MUSA_MCC_RUNFILE");
  paths.clang_offload_bundler =
      ResolveRunfile(*runfiles, "MUSA_BUNDLER_RUNFILE");
  paths.libclang_cpp = ResolveRunfile(*runfiles, "MUSA_LIBCLANG_RUNFILE");
  paths.lld = ResolveRunfile(*runfiles, "MUSA_LLD_RUNFILE");
  paths.llvm_readobj = ResolveRunfile(*runfiles, "MUSA_READOBJ_RUNFILE");
  paths.libdevice = ResolveRunfile(*runfiles, "MUSA_LIBDEVICE_RUNFILE");

  const std::filesystem::path sdk_root =
      std::filesystem::path(paths.mcc).parent_path().parent_path();
  paths.intrinsics_musa_td =
      (sdk_root / "include/llvm/IR/IntrinsicsMUSA.td").string();
  paths.builtins_mtgpu_def =
      (sdk_root / "include/clang/Basic/BuiltinsMTGPU.def").string();
  for (const std::string* path :
       {&paths.intrinsics_musa_td, &paths.builtins_mtgpu_def}) {
    if (!std::filesystem::is_regular_file(*path)) {
      return absl::NotFoundError("qualified SDK runfile is missing");
    }
  }
  absl::StatusOr<MusaBridgeFingerprints> fingerprints =
      FingerprintMusaBridgeToolchain(
          paths, std::string(MccBundleProviderName()),
          std::string(MccBundleProviderCanonicalText()));
  if (!fingerprints.ok()) return fingerprints.status();
  return QualifiedToolchain{.paths = std::move(paths),
                            .fingerprints = *std::move(fingerprints)};
}

MusaBridgeCompileRequest MakeRequest(
    absl::string_view module_name, absl::string_view llvm_ir,
    const std::vector<std::string>& kernels,
    const std::vector<MusaBridgeExportedGlobal>& globals,
    const MusaBridgeFingerprints& fingerprints) {
  MusaBridgeCompileRequest request;
  request.set_protocol_version(kMusaBridgeProtocolVersion);
  request.set_shim_abi_version(kMusaShimAbiVersion);
  request.set_mapping_version(kMusaShimMappingVersion);
  request.set_mapping_fingerprint(kMusaShimMappingSha256);
  request.set_module_name(std::string(module_name));
  request.set_normalized_llvm(std::string(llvm_ir));
  request.set_normalized_llvm_bytes(llvm_ir.size());
  request.set_normalized_llvm_sha256(MusaBridgeSha256Hex(llvm_ir));
  for (const std::string& kernel : kernels)
    request.add_kernel_entry_names(kernel);

  std::vector<std::string> exported(kernels.begin(), kernels.end());
  for (const MusaBridgeExportedGlobal& global : globals) {
    *request.add_exported_globals() = global;
    exported.push_back(global.name());
  }
  std::sort(exported.begin(), exported.end());
  for (const std::string& symbol : exported) {
    request.add_exported_symbol_names(symbol);
  }
  request.set_target_triple(kMusaTargetTriple);
  request.set_architecture(kMusaTargetArchitecture);
  request.set_data_layout(kMusaDataLayout);
  request.set_pointer_model(MUSA_BRIDGE_POINTER_MODEL_OPAQUE);
  request.set_pointer_width_bits(kMusaInterchangePointerWidth);
  request.set_byte_order(MUSA_BRIDGE_BYTE_ORDER_LITTLE_ENDIAN);
  request.mutable_numerical_flags();
  request.set_optimization_level(2);
  request.set_deterministic(true);
  request.set_xla_revision("c06-live-test");
  request.set_current_llvm_revision("openxla-current");
  request.set_provider_name(fingerprints.provider_name);
  request.set_provider_fingerprint(fingerprints.provider_fingerprint);
  request.set_bridge_fingerprint(fingerprints.bridge_fingerprint);
  request.set_toolchain_fingerprint(fingerprints.toolchain_fingerprint);
  return request;
}

std::vector<MusaBridgeExportedGlobal> PositiveGlobals() {
  std::vector<MusaBridgeExportedGlobal> globals(2);
  globals[0].set_name("constant_probe");
  globals[0].set_kind(MUSA_BRIDGE_GLOBAL_KIND_CONSTANT);
  globals[0].set_address_space(2);
  globals[0].set_size_bytes(4 * sizeof(uint32_t));
  globals[0].set_alignment_bytes(16);
  globals[1].set_name("mutable_probe");
  globals[1].set_kind(MUSA_BRIDGE_GLOBAL_KIND_MUTABLE);
  globals[1].set_address_space(1);
  globals[1].set_size_bytes(4 * sizeof(uint32_t));
  globals[1].set_alignment_bytes(16);
  return globals;
}

absl::StatusOr<MusaBridgeCompileResponse> InvokeBridge(
    const QualifiedToolchain& toolchain,
    const MusaBridgeCompileRequest& request, absl::string_view suffix) {
  absl::StatusOr<std::string> request_wire =
      EncodeMusaBridgeCompileRequest(request);
  if (!request_wire.ok()) return request_wire.status();
  absl::StatusOr<std::string> root = MakeTemporaryRoot(suffix);
  if (!root.ok()) return root.status();
  MusaSubprocessLimits limits;
  limits.timeout = std::chrono::minutes(5);
  limits.max_stdout_bytes = 64 << 20;
  limits.max_stderr_bytes = 1 << 20;
  absl::StatusOr<BridgeProcessResult> process =
      RunBridgeProcess(toolchain.paths, *root, *request_wire, limits);
  if (!process.ok()) return process.status();
  if (process->timed_out || process->output_limit_exceeded ||
      process->terminating_signal != 0 || process->exit_code != 0) {
    return absl::InternalError(absl::StrCat(
        "bridge process failed: exit=", process->exit_code, " signal=",
        process->terminating_signal, " timeout=", process->timed_out,
        " output-limit=", process->output_limit_exceeded,
        " stderr=", process->stderr_text));
  }
  if (!process->stderr_text.empty()) {
    return absl::DataLossError(
        "successful bridge transport wrote unexpected stderr");
  }
  absl::StatusOr<MusaBridgeCompileResponse> response =
      DecodeMusaBridgeCompileResponse(process->stdout_text);
  if (!response.ok()) return response.status();
  if (absl::Status status = ValidateMusaBridgeExchange(request, *response);
      !status.ok()) {
    return status;
  }
  return response;
}

std::unique_ptr<Kernel> LoadKernel(MusaExecutor& executor,
                                   const std::vector<uint8_t>& mubin,
                                   absl::string_view name, int arity) {
  KernelLoaderSpec spec = KernelLoaderSpec::CreateMusaMubinInMemorySpec(
      mubin, std::string(name), arity);
  absl::StatusOr<std::unique_ptr<Kernel>> kernel = executor.LoadKernel(spec);
  EXPECT_TRUE(kernel.ok()) << kernel.status();
  return kernel.ok() ? *std::move(kernel) : nullptr;
}

TEST(MusaLlvmBridgeLiveTest,
     CompilesDeterministicallyAndExecutesMappingV2OnS80) {
  TF_ASSERT_OK_AND_ASSIGN(QualifiedToolchain toolchain,
                          LocateQualifiedToolchain());
  const std::vector<std::string> kernels = {"barrier_reverse", "global_probe",
                                            "mapping_probe", "sqrt_probe",
                                            "vector_add"};
  MusaBridgeCompileRequest request =
      MakeRequest("c06_live_mapping_v1", kPositiveIr, kernels,
                  PositiveGlobals(), toolchain.fingerprints);
  // LLVM's legacy attribute spellings are intentional for the vendor LLVM 14
  // process. Current-LLVM validation has its own isolated unit target: linking
  // current LLVM into this runtime process would collide with LLVM code inside
  // the vendor driver when MusaExecutor dlopens libmusa.
  TF_ASSERT_OK(ValidateMusaBridgeCompileRequest(request));

  std::vector<MusaBridgeCompileResponse> responses;
  for (int invocation = 0; invocation < 3; ++invocation) {
    TF_ASSERT_OK_AND_ASSIGN(
        MusaBridgeCompileResponse response,
        InvokeBridge(toolchain, request, absl::StrCat(invocation)));
    ASSERT_EQ(response.status(), MUSA_BRIDGE_STATUS_OK)
        << response.ShortDebugString();
    ASSERT_FALSE(response.mubin().empty());
    responses.push_back(std::move(response));
  }
  EXPECT_EQ(responses[0].mubin(), responses[1].mubin());
  EXPECT_EQ(responses[0].mubin(), responses[2].mubin());
  EXPECT_EQ(responses[0].mubin_sha256(), responses[1].mubin_sha256());
  EXPECT_EQ(responses[0].mubin_sha256(), responses[2].mubin_sha256());
  std::cout << "MUSA_C06_QUALIFICATION"
            << " bridge_sha256=" << toolchain.fingerprints.bridge_fingerprint
            << " provider_name=" << toolchain.fingerprints.provider_name
            << " provider_sha256="
            << toolchain.fingerprints.provider_fingerprint
            << " toolchain_sha256="
            << toolchain.fingerprints.toolchain_fingerprint
            << " mapping_sha256=" << kMusaShimMappingSha256
            << " mubin_sha256=" << responses[0].mubin_sha256()
            << " mubin_bytes=" << responses[0].mubin().size() << std::endl;

  const std::vector<uint8_t> mubin(responses[0].mubin().begin(),
                                   responses[0].mubin().end());
  MusaExecutor executor(/*platform=*/nullptr, /*device_ordinal=*/0);
  TF_ASSERT_OK(executor.Init());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor.CreateStream(/*priority=*/std::nullopt));

  MultiModuleLoaderSpec module_spec;
  module_spec.AddMusaMubinInMemory(mubin);
  TF_ASSERT_OK_AND_ASSIGN(ModuleHandle module,
                          executor.LoadModule(module_spec));
  TF_ASSERT_OK_AND_ASSIGN(DeviceAddressBase mutable_data,
                          executor.GetSymbol("mutable_probe", module));
  TF_ASSERT_OK_AND_ASSIGN(DeviceAddressBase constant,
                          executor.GetSymbol("constant_probe", module));
  std::array<uint32_t, 4> mutable_host = {};
  std::array<uint32_t, 4> constant_host = {};
  TF_ASSERT_OK(executor.SynchronousMemcpy(mutable_host.data(), mutable_data,
                                          sizeof(mutable_host)));
  TF_ASSERT_OK(executor.SynchronousMemcpy(constant_host.data(), constant,
                                          sizeof(constant_host)));
  EXPECT_EQ(mutable_host, (std::array<uint32_t, 4>{10, 20, 30, 40}));
  EXPECT_EQ(constant_host, (std::array<uint32_t, 4>{3, 5, 7, 11}));

  DeviceAddressBase global_output = executor.Allocate(8 * sizeof(uint32_t), 0);
  ASSERT_FALSE(global_output.is_null());
  std::unique_ptr<Kernel> global_probe =
      LoadKernel(executor, mubin, "global_probe", 1);
  ASSERT_NE(global_probe, nullptr);
  KernelArgsPackedArray global_arguments(/*num_args=*/1);
  global_arguments.add_argument(global_output);
  TF_ASSERT_OK(global_probe->Launch(ThreadDim(4), BlockDim(1), stream.get(),
                                    global_arguments));
  std::array<uint32_t, 8> global_host = {};
  TF_ASSERT_OK(
      stream->Memcpy(global_host.data(), global_output, sizeof(global_host)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(global_host,
            (std::array<uint32_t, 8>{10, 20, 30, 40, 3, 5, 7, 11}));

  DeviceAddressBase left = executor.Allocate(4 * sizeof(float), 0);
  DeviceAddressBase right = executor.Allocate(4 * sizeof(float), 0);
  DeviceAddressBase output = executor.Allocate(4 * sizeof(float), 0);
  ASSERT_FALSE(left.is_null());
  ASSERT_FALSE(right.is_null());
  ASSERT_FALSE(output.is_null());
  const std::array<float, 4> left_host = {1, 2, 3, 4};
  const std::array<float, 4> right_host = {10, 20, 30, 40};
  TF_ASSERT_OK(stream->Memcpy(&left, left_host.data(), sizeof(left_host)));
  TF_ASSERT_OK(stream->Memcpy(&right, right_host.data(), sizeof(right_host)));
  std::unique_ptr<Kernel> vector_add =
      LoadKernel(executor, mubin, "vector_add", 4);
  ASSERT_NE(vector_add, nullptr);
  KernelArgsPackedArray vector_arguments(/*num_args=*/4);
  vector_arguments.add_argument(left);
  vector_arguments.add_argument(right);
  vector_arguments.add_argument(output);
  vector_arguments.add_argument(int64_t{4});
  TF_ASSERT_OK(vector_add->Launch(ThreadDim(4), BlockDim(1), stream.get(),
                                  vector_arguments));
  std::array<float, 4> output_host = {};
  TF_ASSERT_OK(stream->Memcpy(output_host.data(), output, sizeof(output_host)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(output_host, (std::array<float, 4>{11, 22, 33, 44}));

  const std::array<float, 4> sqrt_input = {4, 9, 16, 25};
  TF_ASSERT_OK(stream->Memcpy(&left, sqrt_input.data(), sizeof(sqrt_input)));
  std::unique_ptr<Kernel> sqrt = LoadKernel(executor, mubin, "sqrt_probe", 2);
  ASSERT_NE(sqrt, nullptr);
  KernelArgsPackedArray sqrt_arguments(/*num_args=*/2);
  sqrt_arguments.add_argument(left);
  sqrt_arguments.add_argument(output);
  TF_ASSERT_OK(
      sqrt->Launch(ThreadDim(4), BlockDim(1), stream.get(), sqrt_arguments));
  output_host.fill(0);
  TF_ASSERT_OK(stream->Memcpy(output_host.data(), output, sizeof(output_host)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(output_host, (std::array<float, 4>{2, 3, 4, 5}));

  TF_ASSERT_OK(stream->Memcpy(&left, left_host.data(), sizeof(left_host)));
  std::unique_ptr<Kernel> barrier =
      LoadKernel(executor, mubin, "barrier_reverse", 2);
  ASSERT_NE(barrier, nullptr);
  KernelArgsPackedArray barrier_arguments(/*num_args=*/2);
  barrier_arguments.add_argument(left);
  barrier_arguments.add_argument(output);
  TF_ASSERT_OK(barrier->Launch(ThreadDim(4), BlockDim(1), stream.get(),
                               barrier_arguments));
  output_host.fill(0);
  TF_ASSERT_OK(stream->Memcpy(output_host.data(), output, sizeof(output_host)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(output_host, (std::array<float, 4>{4, 3, 2, 1}));

  DeviceAddressBase mapping_output =
      executor.Allocate(12 * sizeof(uint64_t), 0);
  ASSERT_FALSE(mapping_output.is_null());
  TF_ASSERT_OK(stream->MemZero(&mapping_output, mapping_output.size()));
  std::unique_ptr<Kernel> mapping =
      LoadKernel(executor, mubin, "mapping_probe", 1);
  ASSERT_NE(mapping, nullptr);
  KernelArgsPackedArray mapping_arguments(/*num_args=*/1);
  mapping_arguments.add_argument(mapping_output);
  TF_ASSERT_OK(mapping->Launch(ThreadDim(2, 2, 2), BlockDim(2, 2, 2),
                               stream.get(), mapping_arguments));
  std::array<uint64_t, 12> mapping_host = {};
  TF_ASSERT_OK(stream->Memcpy(mapping_host.data(), mapping_output,
                              sizeof(mapping_host)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ((std::array<uint64_t, 3>{mapping_host[0], mapping_host[1],
                                     mapping_host[2]}),
            (std::array<uint64_t, 3>{1, 1, 1}));
  EXPECT_EQ((std::array<uint64_t, 6>{mapping_host[3], mapping_host[4],
                                     mapping_host[5], mapping_host[6],
                                     mapping_host[7], mapping_host[8]}),
            (std::array<uint64_t, 6>{2, 2, 2, 2, 2, 2}));
  EXPECT_EQ((std::array<uint64_t, 3>{mapping_host[9], mapping_host[10],
                                     mapping_host[11]}),
            (std::array<uint64_t, 3>{1, 1, 1}));

  mapping.reset();
  barrier.reset();
  sqrt.reset();
  vector_add.reset();
  global_probe.reset();
  EXPECT_TRUE(executor.UnloadModule(module));
  stream.reset();
  executor.Deallocate(&mapping_output);
  executor.Deallocate(&global_output);
  executor.Deallocate(&output);
  executor.Deallocate(&right);
  executor.Deallocate(&left);
}

TEST(MusaLlvmBridgeLiveTest, RejectsAtomicsUntilMappingVersionBump) {
  TF_ASSERT_OK_AND_ASSIGN(QualifiedToolchain toolchain,
                          LocateQualifiedToolchain());
  MusaBridgeCompileRequest request =
      MakeRequest("c06_atomic_negative", kAtomicIr, {"atomic_probe"}, {},
                  toolchain.fingerprints);
  TF_ASSERT_OK_AND_ASSIGN(MusaBridgeCompileResponse response,
                          InvokeBridge(toolchain, request, "atomic-negative"));
  EXPECT_EQ(response.status(), MUSA_BRIDGE_STATUS_REJECTED);
  EXPECT_TRUE(response.mubin().empty());
  ASSERT_GT(response.diagnostics_size(), 0);
  EXPECT_EQ(response.diagnostics(0).code(), "interchange-rejected");
  EXPECT_TRUE(absl::StrContains(response.diagnostics(0).message(), "atomics"));
}

TEST(MusaLlvmBridgeLiveTest, BoundsKnownSdkClockCodegenFailures) {
  // MUSA 4.0.1 accepts both base-mapping clock intrinsics, then its selector
  // repeats a DAG diagnostic indefinitely. The provider's output bound must
  // turn that SDK defect into a structured failure instead of a bridge hang.
  TF_ASSERT_OK_AND_ASSIGN(QualifiedToolchain toolchain,
                          LocateQualifiedToolchain());
  const std::array<std::pair<absl::string_view, absl::string_view>, 2> cases = {
      std::pair{"clock32", kClock32Ir}, std::pair{"clock64", kClock64Ir}};
  for (const auto& [name, llvm_ir] : cases) {
    MusaBridgeCompileRequest request =
        MakeRequest(absl::StrCat("c06_", name, "_provider_negative"), llvm_ir,
                    {"clock_probe"}, {}, toolchain.fingerprints);
    TF_ASSERT_OK_AND_ASSIGN(MusaBridgeCompileResponse response,
                            InvokeBridge(toolchain, request, name));
    EXPECT_EQ(response.status(), MUSA_BRIDGE_STATUS_COMPILATION_ERROR) << name;
    EXPECT_TRUE(response.mubin().empty()) << name;
    ASSERT_GT(response.diagnostics_size(), 0) << name;
    EXPECT_EQ(response.diagnostics(0).code(), "codegen-failure") << name;
    EXPECT_TRUE(
        absl::StrContains(response.diagnostics(0).message(), "output limit"))
        << name;
    const std::string sdk_root = std::filesystem::path(toolchain.paths.mcc)
                                     .parent_path()
                                     .parent_path()
                                     .string();
    EXPECT_FALSE(absl::StrContains(response.diagnostics(0).message(), sdk_root))
        << name;
  }
}

}  // namespace
}  // namespace xla::gpu::musa::bridge
