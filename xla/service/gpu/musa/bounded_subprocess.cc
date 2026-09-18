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

#include "xla/service/gpu/musa/bounded_subprocess.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"

#if defined(__linux__)
#include <fcntl.h>
#include <linux/memfd.h>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace xla::gpu::musa {
namespace {

constexpr size_t kMaxArgumentCount = 256;
constexpr size_t kMaxArgumentBytes = 1 << 20;
constexpr size_t kMaxEnvironmentCount = 128;
constexpr size_t kMaxEnvironmentBytes = 1 << 20;
constexpr size_t kMaxPathBytes = 4096;
constexpr size_t kMaxEnvironmentNameBytes = 256;
constexpr size_t kMaxEnvironmentValueBytes = 64 << 10;
constexpr size_t kMaxInputStreamBytes = size_t{512} << 20;
constexpr size_t kMaxCapturedStreamBytes = size_t{512} << 20;
constexpr uint64_t kMaxFileBytes = uint64_t{4} << 30;
constexpr uint64_t kMaxAddressSpaceBytes = uint64_t{64} << 30;
constexpr auto kMaxTimeout = std::chrono::hours(1);
constexpr auto kPostKillPipeGrace = std::chrono::seconds(1);
constexpr size_t kDrainBudgetBytes = 64 << 10;

bool ContainsNul(const std::string& value) {
  return value.find('\0') != std::string::npos;
}

bool IsEnvironmentName(const std::string& name) {
  if (name.empty() ||
      !(absl::ascii_isalpha(name.front()) || name.front() == '_')) {
    return false;
  }
  for (unsigned char c : name) {
    if (!absl::ascii_isalnum(c) && c != '_') return false;
  }
  return true;
}

absl::Status ValidateOptions(const MusaSubprocessOptions& options) {
  if (options.executable.empty() || options.executable.size() > kMaxPathBytes ||
      options.executable.front() != '/' || ContainsNul(options.executable)) {
    return absl::InvalidArgumentError(
        "MUSA subprocess executable must be an absolute NUL-free path");
  }
  if (options.working_directory.empty() ||
      options.working_directory.size() > kMaxPathBytes ||
      options.working_directory.front() != '/' ||
      ContainsNul(options.working_directory)) {
    return absl::InvalidArgumentError(
        "MUSA subprocess working directory must be an absolute NUL-free path");
  }
  if (options.arguments.size() > kMaxArgumentCount) {
    return absl::InvalidArgumentError("MUSA subprocess has too many arguments");
  }
  size_t argument_bytes = options.executable.size() + 1;
  for (const std::string& argument : options.arguments) {
    const size_t remaining = kMaxArgumentBytes - argument_bytes;
    if (ContainsNul(argument) || argument.size() >= remaining) {
      return absl::InvalidArgumentError(
          "MUSA subprocess arguments are invalid or too large");
    }
    argument_bytes += argument.size() + 1;
  }
  if (options.environment.size() > kMaxEnvironmentCount) {
    return absl::InvalidArgumentError(
        "MUSA subprocess environment has too many entries");
  }
  size_t environment_bytes = 0;
  std::string previous_name;
  for (const auto& [name, value] : options.environment) {
    if (!IsEnvironmentName(name) || name.size() > kMaxEnvironmentNameBytes ||
        value.size() > kMaxEnvironmentValueBytes || ContainsNul(value) ||
        (!previous_name.empty() && previous_name >= name) ||
        name.size() + value.size() + 2 >
            kMaxEnvironmentBytes - environment_bytes) {
      return absl::InvalidArgumentError(
          "MUSA subprocess environment must be bounded, sorted, and unique");
    }
    environment_bytes += name.size() + value.size() + 2;
    previous_name = name;
  }
  const MusaSubprocessLimits& limits = options.limits;
  if (limits.timeout <= std::chrono::milliseconds::zero() ||
      limits.timeout > kMaxTimeout || limits.max_stdin_bytes == 0 ||
      limits.max_stdin_bytes > kMaxInputStreamBytes ||
      options.stdin_data.size() > limits.max_stdin_bytes ||
      limits.max_stdout_bytes == 0 ||
      limits.max_stdout_bytes > kMaxCapturedStreamBytes ||
      limits.max_stderr_bytes == 0 ||
      limits.max_stderr_bytes > kMaxCapturedStreamBytes ||
      limits.max_file_bytes == 0 || limits.max_file_bytes > kMaxFileBytes ||
      limits.max_address_space_bytes == 0 ||
      limits.max_address_space_bytes > kMaxAddressSpaceBytes) {
    return absl::InvalidArgumentError(
        "MUSA subprocess resource limits are outside supported bounds");
  }
  return absl::OkStatus();
}

#if defined(__linux__)

struct Pipe {
  int read = -1;
  int write = -1;
};

void CloseFd(int* fd) {
  if (*fd >= 0) {
    // On Linux, close() releases the descriptor even when it reports EINTR.
    // Retrying in a multi-threaded parent can therefore close an unrelated
    // descriptor that another thread has since allocated with the same number.
    close(*fd);
    *fd = -1;
  }
}

absl::StatusOr<Pipe> MakePipe() {
  int fds[2];
  if (pipe2(fds, O_CLOEXEC) != 0) {
    return absl::InternalError(
        absl::StrCat("pipe2 failed: ", std::strerror(errno)));
  }
  for (int i = 0; i < 2; ++i) {
    if (fds[i] >= STDERR_FILENO + 1) continue;
    const int replacement = fcntl(fds[i], F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
    if (replacement < 0) {
      const int saved_errno = errno;
      close(fds[0]);
      close(fds[1]);
      return absl::InternalError(
          absl::StrCat("failed to move MUSA subprocess pipe above stdio: ",
                       std::strerror(saved_errno)));
    }
    close(fds[i]);
    fds[i] = replacement;
  }
  return Pipe{.read = fds[0], .write = fds[1]};
}

absl::StatusOr<int> MakeInputFile(const std::string& input) {
#if defined(SYS_memfd_create)
  int fd = syscall(SYS_memfd_create, "xla-musa-subprocess-input", MFD_CLOEXEC);
  if (fd < 0) {
    return absl::InternalError(
        absl::StrCat("memfd_create failed: ", std::strerror(errno)));
  }
  if (fd <= STDERR_FILENO) {
    const int replacement = fcntl(fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
    if (replacement < 0) {
      const int saved_errno = errno;
      close(fd);
      return absl::InternalError(
          absl::StrCat("failed to move MUSA subprocess input above stdio: ",
                       std::strerror(saved_errno)));
    }
    close(fd);
    fd = replacement;
  }
  size_t written = 0;
  while (written < input.size()) {
    const ssize_t count =
        write(fd, input.data() + written, input.size() - written);
    if (count > 0) {
      written += static_cast<size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    const int saved_errno = errno;
    close(fd);
    return absl::InternalError(absl::StrCat(
        "write to MUSA subprocess input failed: ", std::strerror(saved_errno)));
  }
  if (lseek(fd, 0, SEEK_SET) != 0) {
    const int saved_errno = errno;
    close(fd);
    return absl::InternalError(absl::StrCat(
        "rewind MUSA subprocess input failed: ", std::strerror(saved_errno)));
  }
  return fd;
#else
  return absl::UnimplementedError(
      "bounded MUSA subprocess stdin requires memfd_create");
#endif
}

bool SetLimit(int resource, uint64_t value) {
  const uint64_t max_rlim = std::numeric_limits<rlim_t>::max();
  const rlim_t bounded = static_cast<rlim_t>(std::min(value, max_rlim));
  const rlimit limit{.rlim_cur = bounded, .rlim_max = bounded};
  return setrlimit(resource, &limit) == 0;
}

template <size_t N>
[[noreturn]] void ChildFailure(int fd, const char (&message)[N]) {
  size_t written = 0;
  while (written < N - 1) {
    const ssize_t count = write(fd, message + written, N - 1 - written);
    if (count > 0) {
      written += count;
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    break;
  }
  _exit(127);
}

void KillGroupAndLeader(pid_t pid, bool leader_is_unreaped) {
  if (pid > 0 && kill(-pid, SIGKILL) != 0 && errno != ESRCH) {
    // Fall through to the direct-child kill. The caller still waits on the
    // direct child, so cleanup does not depend on process-group setup.
  }
  // An unreaped child still owns its pid, so this cannot target a reused pid.
  if (pid > 0 && leader_is_unreaped && kill(pid, SIGKILL) != 0 &&
      errno != ESRCH) {
    // Best effort; the caller must still reap the direct child.
  }
}

bool ResetChildSignalState() {
  sigset_t empty;
  if (sigemptyset(&empty) != 0 ||
      sigprocmask(SIG_SETMASK, &empty, nullptr) != 0) {
    return false;
  }
  struct sigaction action = {};
  action.sa_handler = SIG_DFL;
  if (sigemptyset(&action.sa_mask) != 0) return false;
  for (int signal = 1; signal < NSIG; ++signal) {
    if (signal == SIGKILL || signal == SIGSTOP) continue;
    if (sigaction(signal, &action, nullptr) != 0 && errno != EINVAL) {
      return false;
    }
  }
  return true;
}

bool CloseInheritedFileDescriptors(int max_fd) {
#if defined(SYS_close_range)
  if (syscall(SYS_close_range, static_cast<unsigned int>(STDERR_FILENO + 1),
              std::numeric_limits<unsigned int>::max(), 0) == 0) {
    return true;
  }
  if (errno != ENOSYS && errno != EINVAL) return false;
#endif
  for (int fd = STDERR_FILENO + 1; fd < max_fd; ++fd) close(fd);
  return true;
}

bool SetNonBlocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

absl::Status DrainPipe(int* fd, size_t limit, std::string* output,
                       bool* overflowed) {
  char buffer[8192];
  size_t drained = 0;
  while (*fd >= 0 && drained < kDrainBudgetBytes) {
    const size_t requested =
        std::min(sizeof(buffer), kDrainBudgetBytes - drained);
    const ssize_t count = read(*fd, buffer, requested);
    if (count > 0) {
      drained += static_cast<size_t>(count);
      const size_t available =
          output->size() < limit ? limit - output->size() : 0;
      const size_t kept = std::min<size_t>(count, available);
      output->append(buffer, kept);
      if (kept != static_cast<size_t>(count)) {
        *overflowed = true;
        CloseFd(fd);
        return absl::OkStatus();
      }
      continue;
    }
    if (count == 0) {
      CloseFd(fd);
      return absl::OkStatus();
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return absl::OkStatus();
    return absl::InternalError(absl::StrCat(
        "read from MUSA subprocess failed: ", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::Status ObserveChildExit(pid_t pid, bool* exited) {
  while (true) {
    siginfo_t info = {};
    if (waitid(P_PID, pid, &info, WEXITED | WNOHANG | WNOWAIT) == 0) {
      if (info.si_pid == pid) *exited = true;
      return absl::OkStatus();
    }
    if (errno == EINTR) continue;
    if (errno == ECHILD) {
      return absl::FailedPreconditionError(
          "MUSA subprocess child was reaped by another component");
    }
    return absl::InternalError(
        absl::StrCat("waitid failed: ", std::strerror(errno)));
  }
}

absl::Status ReapExitedChild(pid_t pid, int* status) {
  while (true) {
    const pid_t result = waitpid(pid, status, 0);
    if (result == pid) return absl::OkStatus();
    if (result < 0 && errno == EINTR) continue;
    if (result < 0 && errno == ECHILD) {
      return absl::FailedPreconditionError(
          "MUSA subprocess child was reaped by another component");
    }
    return absl::InternalError(
        absl::StrCat("waitpid failed: ", std::strerror(errno)));
  }
}

absl::Status ReapAfterFailure(pid_t pid, int* status) {
  KillGroupAndLeader(pid, /*leader_is_unreaped=*/true);
  while (waitpid(pid, status, 0) < 0) {
    if (errno == EINTR) continue;
    if (errno == ECHILD) return absl::OkStatus();
    return absl::InternalError(
        absl::StrCat("waitpid cleanup failed: ", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::StatusOr<MusaSubprocessResult> RunLinux(
    const MusaSubprocessOptions& options) {
  struct sigaction sigchld_action = {};
  if (sigaction(SIGCHLD, nullptr, &sigchld_action) != 0) {
    return absl::InternalError(absl::StrCat(
        "failed to inspect SIGCHLD disposition: ", std::strerror(errno)));
  }
  if (sigchld_action.sa_handler == SIG_IGN ||
      (sigchld_action.sa_flags & SA_NOCLDWAIT) != 0) {
    return absl::FailedPreconditionError(
        "bounded MUSA subprocesses require waitable child processes");
  }

  absl::StatusOr<int> input_fd_or = MakeInputFile(options.stdin_data);
  if (!input_fd_or.ok()) return input_fd_or.status();
  int input_fd = *input_fd_or;
  absl::StatusOr<Pipe> stdout_pipe_or = MakePipe();
  if (!stdout_pipe_or.ok()) {
    CloseFd(&input_fd);
    return stdout_pipe_or.status();
  }
  Pipe stdout_pipe = *stdout_pipe_or;
  absl::StatusOr<Pipe> stderr_pipe_or = MakePipe();
  if (!stderr_pipe_or.ok()) {
    CloseFd(&input_fd);
    CloseFd(&stdout_pipe.read);
    CloseFd(&stdout_pipe.write);
    return stderr_pipe_or.status();
  }
  Pipe stderr_pipe = *stderr_pipe_or;

  std::vector<std::string> argv_storage;
  argv_storage.reserve(options.arguments.size() + 1);
  argv_storage.push_back(options.executable);
  argv_storage.insert(argv_storage.end(), options.arguments.begin(),
                      options.arguments.end());
  std::vector<char*> argv;
  argv.reserve(argv_storage.size() + 1);
  for (std::string& value : argv_storage) argv.push_back(value.data());
  argv.push_back(nullptr);

  std::vector<std::string> environment_storage;
  environment_storage.reserve(options.environment.size());
  for (const auto& [name, value] : options.environment) {
    environment_storage.push_back(absl::StrCat(name, "=", value));
  }
  std::vector<char*> environment;
  environment.reserve(environment_storage.size() + 1);
  for (std::string& value : environment_storage) {
    environment.push_back(value.data());
  }
  environment.push_back(nullptr);

  const long open_max = sysconf(_SC_OPEN_MAX);
  if (open_max <= STDERR_FILENO || open_max > std::numeric_limits<int>::max()) {
    CloseFd(&input_fd);
    CloseFd(&stdout_pipe.read);
    CloseFd(&stdout_pipe.write);
    CloseFd(&stderr_pipe.read);
    CloseFd(&stderr_pipe.write);
    return absl::InternalError(
        "failed to determine a safe subprocess file-descriptor ceiling");
  }
  const int max_fd = static_cast<int>(open_max);
  const pid_t parent_pid = getpid();

  const pid_t pid = fork();
  if (pid < 0) {
    const absl::Status status = absl::InternalError(
        absl::StrCat("fork failed: ", std::strerror(errno)));
    CloseFd(&input_fd);
    CloseFd(&stdout_pipe.read);
    CloseFd(&stdout_pipe.write);
    CloseFd(&stderr_pipe.read);
    CloseFd(&stderr_pipe.write);
    return status;
  }
  if (pid == 0) {
    CloseFd(&stdout_pipe.read);
    CloseFd(&stderr_pipe.read);
    if (setpgid(0, 0) != 0) {
      ChildFailure(stderr_pipe.write, "setpgid failed\n");
    }
    if (syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0) != 0 ||
        getppid() != parent_pid) {
      ChildFailure(stderr_pipe.write, "parent-death setup failed\n");
    }
    if (!ResetChildSignalState()) {
      ChildFailure(stderr_pipe.write, "signal reset failed\n");
    }
    umask(077);
    if (chdir(options.working_directory.c_str()) != 0) {
      ChildFailure(stderr_pipe.write, "chdir failed\n");
    }
    const uint64_t cpu_seconds =
        static_cast<uint64_t>((options.limits.timeout.count() + 999) / 1000) +
        1;
    if (!SetLimit(RLIMIT_CORE, 0) ||
        !SetLimit(RLIMIT_FSIZE, options.limits.max_file_bytes) ||
        !SetLimit(RLIMIT_AS, options.limits.max_address_space_bytes) ||
        !SetLimit(RLIMIT_CPU, cpu_seconds)) {
      ChildFailure(stderr_pipe.write, "setrlimit failed\n");
    }
    if (dup2(input_fd, STDIN_FILENO) < 0 ||
        dup2(stdout_pipe.write, STDOUT_FILENO) < 0 ||
        dup2(stderr_pipe.write, STDERR_FILENO) < 0) {
      ChildFailure(stderr_pipe.write, "dup2 failed\n");
    }
    CloseFd(&input_fd);
    CloseFd(&stdout_pipe.write);
    CloseFd(&stderr_pipe.write);
    if (!CloseInheritedFileDescriptors(max_fd)) {
      ChildFailure(STDERR_FILENO, "descriptor cleanup failed\n");
    }
    execve(options.executable.c_str(), argv.data(), environment.data());
    ChildFailure(STDERR_FILENO, "execve failed\n");
  }

  CloseFd(&input_fd);
  CloseFd(&stdout_pipe.write);
  CloseFd(&stderr_pipe.write);
  // Close a race where the child reaches exec before its own setpgid call.
  if (setpgid(pid, pid) != 0 && errno != EACCES && errno != ESRCH) {
    const int setpgid_errno = errno;
    int status = 0;
    absl::Status cleanup = ReapAfterFailure(pid, &status);
    CloseFd(&stdout_pipe.read);
    CloseFd(&stderr_pipe.read);
    if (!cleanup.ok()) return cleanup;
    return absl::InternalError(
        absl::StrCat("parent setpgid failed: ", std::strerror(setpgid_errno)));
  }
  if (!SetNonBlocking(stdout_pipe.read) || !SetNonBlocking(stderr_pipe.read)) {
    int status = 0;
    absl::Status cleanup = ReapAfterFailure(pid, &status);
    CloseFd(&stdout_pipe.read);
    CloseFd(&stderr_pipe.read);
    if (!cleanup.ok()) return cleanup;
    return absl::InternalError("failed to make subprocess pipes nonblocking");
  }

  MusaSubprocessResult result;
  const auto deadline =
      std::chrono::steady_clock::now() + options.limits.timeout;
  // Observe the leader with WNOWAIT and retain its zombie until all output is
  // handled. This reserves its pid/pgid, so a delayed process-group kill can
  // never target an unrelated group after numeric-id reuse.
  bool child_exited = false;
  bool killed = false;
  std::chrono::steady_clock::time_point close_pipes_deadline;
  int wait_status = 0;
  const auto start_kill = [&](std::chrono::steady_clock::time_point now) {
    if (killed) return;
    KillGroupAndLeader(pid, /*leader_is_unreaped=*/true);
    killed = true;
    close_pipes_deadline = now + kPostKillPipeGrace;
  };
  while (!child_exited || stdout_pipe.read >= 0 || stderr_pipe.read >= 0) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline &&
        (!child_exited || stdout_pipe.read >= 0 || stderr_pipe.read >= 0)) {
      result.timed_out = true;
      start_kill(now);
    }
    if (!result.cancelled && options.cancellation_requested &&
        options.cancellation_requested()) {
      result.cancelled = true;
      start_kill(now);
    }
    if (result.output_limit_exceeded) start_kill(now);
    // A deliberately detached descendant can retain a copied output fd even
    // after the process group and direct child are gone. Never let that evade
    // the public wall/output bound and hold this call open indefinitely.
    if (killed && now >= close_pipes_deadline) {
      CloseFd(&stdout_pipe.read);
      CloseFd(&stderr_pipe.read);
    }

    pollfd poll_fds[2];
    nfds_t count = 0;
    if (stdout_pipe.read >= 0) {
      poll_fds[count++] = pollfd{
          .fd = stdout_pipe.read, .events = POLLIN | POLLHUP, .revents = 0};
    }
    if (stderr_pipe.read >= 0) {
      poll_fds[count++] = pollfd{
          .fd = stderr_pipe.read, .events = POLLIN | POLLHUP, .revents = 0};
    }
    int timeout_ms = 20;
    if (!child_exited && !result.timed_out) {
      const auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      timeout_ms = static_cast<int>(
          std::max<int64_t>(0, std::min<int64_t>(20, remaining.count())));
    }
    if (count > 0) {
      const int poll_result = poll(poll_fds, count, timeout_ms);
      if (poll_result < 0 && errno != EINTR) {
        absl::Status status = absl::InternalError(
            absl::StrCat("poll failed: ", std::strerror(errno)));
        absl::Status cleanup = ReapAfterFailure(pid, &wait_status);
        CloseFd(&stdout_pipe.read);
        CloseFd(&stderr_pipe.read);
        return cleanup.ok() ? status : cleanup;
      }
    } else if (!child_exited) {
      usleep(1000);
    }

    bool stream_overflowed = false;
    absl::Status status =
        DrainPipe(&stdout_pipe.read, options.limits.max_stdout_bytes,
                  &result.stdout_text, &stream_overflowed);
    if (!status.ok()) {
      absl::Status cleanup = ReapAfterFailure(pid, &wait_status);
      CloseFd(&stdout_pipe.read);
      CloseFd(&stderr_pipe.read);
      return cleanup.ok() ? status : cleanup;
    }
    result.output_limit_exceeded |= stream_overflowed;
    if (stream_overflowed) start_kill(std::chrono::steady_clock::now());
    stream_overflowed = false;
    status = DrainPipe(&stderr_pipe.read, options.limits.max_stderr_bytes,
                       &result.stderr_text, &stream_overflowed);
    if (!status.ok()) {
      absl::Status cleanup = ReapAfterFailure(pid, &wait_status);
      CloseFd(&stdout_pipe.read);
      CloseFd(&stderr_pipe.read);
      return cleanup.ok() ? status : cleanup;
    }
    result.output_limit_exceeded |= stream_overflowed;
    if (stream_overflowed) start_kill(std::chrono::steady_clock::now());
    if (!child_exited) {
      status = ObserveChildExit(pid, &child_exited);
      if (!status.ok()) {
        // ECHILD means another component may already have released and reused
        // this numeric pid. Do not signal it or its former process group.
        if (absl::IsFailedPrecondition(status)) {
          CloseFd(&stdout_pipe.read);
          CloseFd(&stderr_pipe.read);
          return status;
        }
        absl::Status cleanup = ReapAfterFailure(pid, &wait_status);
        CloseFd(&stdout_pipe.read);
        CloseFd(&stderr_pipe.read);
        return cleanup.ok() ? status : cleanup;
      }
    }
  }

  absl::Status reap_status = ReapExitedChild(pid, &wait_status);
  if (!reap_status.ok()) return reap_status;

  if (WIFEXITED(wait_status)) {
    result.exit_code = WEXITSTATUS(wait_status);
  } else if (WIFSIGNALED(wait_status)) {
    result.terminating_signal = WTERMSIG(wait_status);
  }
  return result;
}

#endif  // defined(__linux__)

}  // namespace

absl::StatusOr<MusaSubprocessResult> RunMusaBoundedSubprocess(
    const MusaSubprocessOptions& options) {
  absl::Status status = ValidateOptions(options);
  if (!status.ok()) return status;
#if defined(__linux__)
  return RunLinux(options);
#else
  return absl::UnimplementedError(
      "bounded MUSA compiler subprocesses require Linux");
#endif
}

}  // namespace xla::gpu::musa
