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

#ifndef XLA_SERVICE_GPU_MUSA_BOUNDED_SUBPROCESS_H_
#define XLA_SERVICE_GPU_MUSA_BOUNDED_SUBPROCESS_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"

namespace xla::gpu::musa {

struct MusaSubprocessLimits {
  std::chrono::milliseconds timeout = std::chrono::minutes(2);
  size_t max_stdout_bytes = 1 << 20;
  size_t max_stderr_bytes = 1 << 20;
  uint64_t max_file_bytes = uint64_t{128} << 20;
  uint64_t max_address_space_bytes = uint64_t{8} << 30;
};

// A closed subprocess description. `arguments` excludes argv[0]. The child
// receives exactly `environment`; it never inherits the bridge environment.
struct MusaSubprocessOptions {
  std::string executable;
  std::vector<std::string> arguments;
  std::string working_directory;
  std::vector<std::pair<std::string, std::string>> environment;
  MusaSubprocessLimits limits;
};

struct MusaSubprocessResult {
  int exit_code = -1;
  int terminating_signal = 0;
  bool timed_out = false;
  bool output_limit_exceeded = false;
  std::string stdout_text;
  std::string stderr_text;

  bool exited_successfully() const {
    return !timed_out && !output_limit_exceeded && terminating_signal == 0 &&
           exit_code == 0;
  }
};

// Executes one absolute program without a shell. On Linux, the child becomes
// a process-group leader, receives a clean signal mask/disposition set, keeps
// only stdin/stdout/stderr, and receives CPU, address-space, file-size, and
// core limits. Timeout or output overflow kills the process group and direct
// child; copied output descriptors are forcibly abandoned after a bounded
// post-kill drain. Pre-fork setup failures are Status values. Child-side setup
// and exec failures are started-child results with exit code 127.
//
// RLIMIT_AS and RLIMIT_FSIZE are per-process/per-file limits, not aggregate
// cgroup limits. Process-group cleanup contains ordinary descendants, but this
// API is not a security sandbox for an executable that deliberately creates a
// new session/process group. Callers must execute a trusted bridge binary and
// must not concurrently reap this API's child from a process-wide SIGCHLD
// handler or waitpid(-1) loop. SIGCHLD dispositions that auto-reap children are
// rejected before fork.
absl::StatusOr<MusaSubprocessResult> RunMusaBoundedSubprocess(
    const MusaSubprocessOptions& options);

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_BOUNDED_SUBPROCESS_H_
