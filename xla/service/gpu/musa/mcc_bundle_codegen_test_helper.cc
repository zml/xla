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
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

extern char** environ;

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

bool WriteFile(const std::string& path, std::string_view contents) {
  const int fd =
      open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) return false;
  const bool result = WriteAll(fd, contents);
  while (close(fd) < 0 && errno == EINTR) {
  }
  return result;
}

bool WriteRepeated(const std::string& path, size_t bytes) {
  const int fd =
      open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) return false;
  const std::string block(4096, 'M');
  bool result = true;
  while (bytes > 0) {
    const size_t count = std::min(bytes, block.size());
    if (!WriteAll(fd, std::string_view(block).substr(0, count))) {
      result = false;
      break;
    }
    bytes -= count;
  }
  while (close(fd) < 0 && errno == EINTR) {
  }
  return result;
}

std::string ReadFile(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return {};
  std::string contents;
  char buffer[4096];
  while (true) {
    const ssize_t count = read(fd, buffer, sizeof(buffer));
    if (count > 0) {
      contents.append(buffer, count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    break;
  }
  while (close(fd) < 0 && errno == EINTR) {
  }
  return contents;
}

std::string CurrentDirectory() {
  char* directory = getcwd(nullptr, 0);
  if (directory == nullptr) return {};
  std::string result(directory);
  std::free(directory);
  return result;
}

std::string Dirname(std::string_view path) {
  const size_t slash = path.rfind('/');
  if (slash == 0) return "/";
  return std::string(path.substr(0, slash));
}

std::string Mode(std::string_view contents) {
  constexpr std::string_view prefix = "FAKE_MODE=";
  const size_t start = contents.find(prefix);
  if (start == std::string_view::npos) return "success";
  const size_t value_start = start + prefix.size();
  const size_t end = contents.find_first_of("\r\n ", value_start);
  return std::string(contents.substr(value_start, end - value_start));
}

int Fail(const std::string& directory, std::string_view stage, int exit_code) {
  WriteAll(STDERR_FILENO,
           "\x01"
           "fake ");
  WriteAll(STDERR_FILENO, stage);
  WriteAll(STDERR_FILENO, " failure at ");
  WriteAll(STDERR_FILENO, directory);
  WriteAll(STDERR_FILENO, "/secret\n");
  return exit_code;
}

bool HasExactEnvironment(const std::string& directory,
                         std::string_view executable) {
  const std::vector<std::string> expected = {
      "HOME=" + directory,
      "LANG=C",
      "LC_ALL=C",
      "PATH=" + Dirname(executable),
      "SOURCE_DATE_EPOCH=0",
      "TMPDIR=" + directory,
      "TZ=UTC",
  };
  size_t count = 0;
  while (environ[count] != nullptr) ++count;
  if (count != expected.size()) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != environ[i]) return false;
  }
  return true;
}

bool ExactArguments(int argc, char** argv,
                    const std::vector<std::string>& expected) {
  if (argc != expected.size() + 1) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (argv[i + 1] != expected[i]) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string directory = CurrentDirectory();
  if (directory.empty()) return 90;
  if (!HasExactEnvironment(directory, argv[0])) {
    return Fail(directory, "environment", 91);
  }

  const std::string input = directory + "/module.ll";
  const std::string input_bundle = directory + "/module.input.bundle.ll";
  const std::string output_bundle = directory + "/module.output.bundle.o";
  const std::string mubin = directory + "/module.mubin";

  if (ExactArguments(
          argc, argv,
          {"--type=ll",
           "--targets=host-x86_64-unknown-linux-gnu,"
           "musa-mtgpu-mt-musa-mp_21",
           "--inputs=/dev/null," + input, "--outputs=" + input_bundle})) {
    const std::string contents = ReadFile(input);
    if (Mode(contents) == "bundle_fail") {
      return Fail(directory, "bundle", 20);
    }
    return WriteFile(input_bundle, contents) ? 0 : 92;
  }

  if (ExactArguments(argc, argv,
                     {"-mtgpu", "--musa-device-only", "--offload-arch=mp_21",
                      "-O2", "-mllvm", "-opaque-pointers", "-x", "ir",
                      input_bundle, "-c", "-o", output_bundle})) {
    const std::string contents = ReadFile(input_bundle);
    const std::string mode = Mode(contents);
    if (mode == "timeout") {
      std::this_thread::sleep_for(std::chrono::seconds(10));
    } else if (mode == "output") {
      const std::string block(4096, 'x');
      for (int i = 0; i < 1024; ++i) {
        if (!WriteAll(STDOUT_FILENO, block)) break;
      }
    } else if (mode == "mcc_fail") {
      return Fail(directory, "mcc", 21);
    } else if (mode == "diagnostic") {
      WriteAll(STDERR_FILENO, "warning: \x02path=");
      WriteAll(STDERR_FILENO, directory);
      WriteAll(STDERR_FILENO, "/module.ll\n");
    }
    return WriteFile(output_bundle, contents) ? 0 : 93;
  }

  if (ExactArguments(argc, argv,
                     {"--type=o", "--inputs=" + output_bundle, "--list"})) {
    const std::string mode = Mode(ReadFile(output_bundle));
    if (mode == "list_fail") return Fail(directory, "list", 22);
    if (mode == "wrong_bundle") {
      return WriteAll(STDOUT_FILENO, "musa-mtgpu-mt-musa--mp_22\n") ? 0 : 94;
    }
    if (mode == "duplicate_bundle") {
      return WriteAll(STDOUT_FILENO,
                      "musa-mtgpu-mt-musa--mp_21\n"
                      "musa-mtgpu-mt-musa---mp_21\n")
                 ? 0
                 : 95;
    }
    if (mode == "malformed_bundle") {
      return WriteAll(STDOUT_FILENO, "musa-mtgpu-mt-musa mp_21\n") ? 0 : 96;
    }
    return WriteAll(STDOUT_FILENO, "musa-mtgpu-mt-musa---mp_21\n") ? 0 : 97;
  }

  const std::string mode = Mode(ReadFile(output_bundle));
  if (ExactArguments(
          argc, argv,
          {"--unbundle", "--type=o", "--targets=musa-mtgpu-mt-musa---mp_21",
           "--inputs=" + output_bundle, "--outputs=" + mubin})) {
    if (mode == "unbundle_fail") {
      return Fail(directory, "unbundle", 23);
    }
    if (mode == "fifo_mubin") return mkfifo(mubin.c_str(), 0600) == 0 ? 0 : 98;
    if (mode == "large_mubin") return WriteRepeated(mubin, 4096) ? 0 : 98;
    return WriteFile(mubin, "MUBIN:" + directory) ? 0 : 99;
  }

  return Fail(directory, "argv", 100);
}
