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
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

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

bool WriteBytes(int fd, size_t bytes) {
  const std::string chunk(4096, 'x');
  while (bytes > 0) {
    const size_t count = std::min(bytes, chunk.size());
    if (!WriteAll(fd, std::string_view(chunk).substr(0, count))) return false;
    bytes -= count;
  }
  return true;
}

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.substr(0, prefix.size()) == prefix;
}

int ParseInt(std::string_view value) {
  return std::strtol(std::string(value).c_str(), nullptr, 10);
}

size_t ParseSize(std::string_view value) {
  return std::strtoull(std::string(value).c_str(), nullptr, 10);
}

}  // namespace

int main(int argc, char** argv) {
  int exit_code = 0;
  int sleep_ms = 0;
  int child_delay_ms = 0;
  std::string child_sentinel;
  bool child_detach = false;
  size_t file_bytes = 0;
  size_t allocate_bytes = 0;
  bool stdout_forever = false;
  bool check_stdin_eof = false;
  bool check_signal_state = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (StartsWith(argument, "--stdout=")) {
      if (!WriteAll(STDOUT_FILENO, argument.substr(strlen("--stdout=")))) {
        return 90;
      }
    } else if (StartsWith(argument, "--stderr=")) {
      if (!WriteAll(STDERR_FILENO, argument.substr(strlen("--stderr=")))) {
        return 91;
      }
    } else if (StartsWith(argument, "--stdout-bytes=")) {
      if (!WriteBytes(STDOUT_FILENO,
                      ParseSize(argument.substr(strlen("--stdout-bytes="))))) {
        return 92;
      }
    } else if (StartsWith(argument, "--stderr-bytes=")) {
      if (!WriteBytes(STDERR_FILENO,
                      ParseSize(argument.substr(strlen("--stderr-bytes="))))) {
        return 99;
      }
    } else if (argument == "--stdout-forever") {
      stdout_forever = true;
    } else if (StartsWith(argument, "--sleep-ms=")) {
      sleep_ms = ParseInt(argument.substr(strlen("--sleep-ms=")));
    } else if (StartsWith(argument, "--exit=")) {
      exit_code = ParseInt(argument.substr(strlen("--exit=")));
    } else if (StartsWith(argument, "--print-env=")) {
      const std::string name(argument.substr(strlen("--print-env=")));
      const char* value = std::getenv(name.c_str());
      if (!WriteAll(STDOUT_FILENO, value == nullptr ? "<unset>" : value)) {
        return 93;
      }
    } else if (argument == "--print-cwd") {
      char* cwd = getcwd(nullptr, 0);
      if (cwd == nullptr) return 94;
      const bool ok = WriteAll(STDOUT_FILENO, cwd);
      std::free(cwd);
      if (!ok) return 95;
    } else if (StartsWith(argument, "--check-fd=")) {
      const int fd = ParseInt(argument.substr(strlen("--check-fd=")));
      if (!WriteAll(STDOUT_FILENO, fcntl(fd, F_GETFD) < 0 && errno == EBADF
                                       ? "closed"
                                       : "open")) {
        return 100;
      }
    } else if (argument == "--check-stdin-eof") {
      check_stdin_eof = true;
    } else if (argument == "--check-signal-state") {
      check_signal_state = true;
    } else if (StartsWith(argument, "--file-bytes=")) {
      file_bytes = ParseSize(argument.substr(strlen("--file-bytes=")));
    } else if (StartsWith(argument, "--allocate-bytes=")) {
      allocate_bytes = ParseSize(argument.substr(strlen("--allocate-bytes=")));
    } else if (StartsWith(argument, "--child-sentinel=")) {
      child_sentinel = argument.substr(strlen("--child-sentinel="));
    } else if (StartsWith(argument, "--child-delay-ms=")) {
      child_delay_ms = ParseInt(argument.substr(strlen("--child-delay-ms=")));
    } else if (argument == "--child-detach") {
      child_detach = true;
    } else {
      return 96;
    }
  }

  if (check_stdin_eof) {
    char byte;
    const ssize_t count = read(STDIN_FILENO, &byte, 1);
    const std::string state = count == 0  ? "eof"
                              : count > 0 ? "data"
                                          : "error-" + std::to_string(errno);
    if (!WriteAll(STDOUT_FILENO, state)) return 101;
  }
  if (check_signal_state) {
    sigset_t mask;
    struct sigaction pipe_action = {};
    if (sigprocmask(SIG_SETMASK, nullptr, &mask) != 0 ||
        sigaction(SIGPIPE, nullptr, &pipe_action) != 0) {
      return 102;
    }
    const bool clean =
        sigismember(&mask, SIGUSR1) == 0 && pipe_action.sa_handler == SIG_DFL;
    if (!WriteAll(STDOUT_FILENO, clean ? "clean" : "inherited")) return 103;
  }
  if (file_bytes > 0) {
    const int fd = open("bounded-subprocess-output",
                        O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0 || !WriteBytes(fd, file_bytes)) return 104;
    if (close(fd) != 0) return 105;
  }
  if (allocate_bytes > 0) {
    void* allocation = mmap(nullptr, allocate_bytes, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocation == MAP_FAILED) {
      if (!WriteAll(STDOUT_FILENO, "allocation-failed")) return 106;
    } else {
      if (!WriteAll(STDOUT_FILENO, "allocation-succeeded")) return 107;
      munmap(allocation, allocate_bytes);
    }
  }
  if (stdout_forever) {
    const std::string chunk(4096, 'x');
    while (WriteAll(STDOUT_FILENO, chunk)) {
    }
    return 108;
  }

  pid_t child = -1;
  if (!child_sentinel.empty()) {
    child = fork();
    if (child < 0) return 97;
    if (child == 0) {
      if (child_detach && setsid() < 0) _exit(109);
      std::this_thread::sleep_for(std::chrono::milliseconds(child_delay_ms));
      const int fd =
          open(child_sentinel.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
      if (fd >= 0) close(fd);
      _exit(fd >= 0 ? 0 : 98);
    }
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  if (child > 0) {
    int status;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
  }
  return exit_code;
}
