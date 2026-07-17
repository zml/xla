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

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "xla/tsl/platform/resource_loader.h"

namespace xla::gpu::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

std::string HelperPath() {
  return tsl::GetDataDependencyFilepath(
      "xla/service/gpu/musa/bounded_subprocess_test_helper");
}

MusaSubprocessOptions Options(std::vector<std::string> arguments = {}) {
  MusaSubprocessOptions options;
  options.executable = HelperPath();
  options.arguments = std::move(arguments);
  options.working_directory = ::testing::TempDir();
  while (options.working_directory.size() > 1 &&
         options.working_directory.back() == '/') {
    options.working_directory.pop_back();
  }
  return options;
}

TEST(MusaBoundedSubprocessTest, CapturesStreamsAndExitCode) {
  absl::StatusOr<MusaSubprocessResult> result = RunMusaBoundedSubprocess(
      Options({"--stdout=output", "--stderr=diagnostic", "--exit=7"}));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->exit_code, 7);
  EXPECT_EQ(result->terminating_signal, 0);
  EXPECT_EQ(result->stdout_text, "output");
  EXPECT_EQ(result->stderr_text, "diagnostic");
  EXPECT_FALSE(result->exited_successfully());
}

TEST(MusaBoundedSubprocessTest, DeliversBoundedStdinWithoutARequestFile) {
  MusaSubprocessOptions options = Options({"--copy-stdin"});
  options.stdin_data = std::string("request\0payload", 15);
  options.limits.max_stdin_bytes = options.stdin_data.size();
  options.limits.max_stdout_bytes = options.stdin_data.size();
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->exited_successfully());
  EXPECT_EQ(result->stdout_text, options.stdin_data);

  options.limits.max_stdin_bytes = options.stdin_data.size() - 1;
  EXPECT_THAT(RunMusaBoundedSubprocess(options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("resource limits")));
}

TEST(MusaBoundedSubprocessTest, CancellationKillsTheProcessGroup) {
  std::atomic<bool> cancel = false;
  MusaSubprocessOptions options = Options({"--sleep-ms=5000"});
  options.limits.timeout = std::chrono::seconds(30);
  options.cancellation_requested = [&] { return cancel.load(); };
  std::thread canceller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cancel.store(true);
  });
  const auto started = std::chrono::steady_clock::now();
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  canceller.join();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->cancelled);
  EXPECT_FALSE(result->timed_out);
  EXPECT_FALSE(result->exited_successfully());
  EXPECT_LT(elapsed, std::chrono::seconds(2));
}

TEST(MusaBoundedSubprocessTest, UsesOnlyExplicitEnvironmentAndDirectory) {
  MusaSubprocessOptions options = Options({"--print-env=ONLY"});
  options.environment = {{"ONLY", "explicit"}};
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->exited_successfully());
  EXPECT_EQ(result->stdout_text, "explicit");

  options.arguments = {"--print-env=PATH"};
  result = RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->stdout_text, "<unset>");

  options.arguments = {"--print-cwd"};
  result = RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->stdout_text, options.working_directory);
}

TEST(MusaBoundedSubprocessTest, OutputOverflowKillsTheProcess) {
  MusaSubprocessOptions options = Options({"--stdout-bytes=1048576"});
  options.limits.max_stdout_bytes = 1024;
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->output_limit_exceeded);
  EXPECT_EQ(result->stdout_text.size(), 1024);
  EXPECT_FALSE(result->exited_successfully());
}

TEST(MusaBoundedSubprocessTest, OutputExactlyAtTheBoundIsAccepted) {
  MusaSubprocessOptions options =
      Options({"--stdout-bytes=1024", "--stderr-bytes=513"});
  options.limits.max_stdout_bytes = 1024;
  options.limits.max_stderr_bytes = 513;
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->exited_successfully());
  EXPECT_FALSE(result->output_limit_exceeded);
  EXPECT_EQ(result->stdout_text.size(), 1024);
  EXPECT_EQ(result->stderr_text.size(), 513);
}

TEST(MusaBoundedSubprocessTest,
     OutputOverflowCannotStarveKillWithContinuousWrites) {
  MusaSubprocessOptions options = Options({"--stdout-forever"});
  options.limits.max_stdout_bytes = 1024;
  options.limits.timeout = std::chrono::seconds(30);
  const auto started = std::chrono::steady_clock::now();
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->output_limit_exceeded);
  EXPECT_EQ(result->stdout_text.size(), 1024);
  EXPECT_LT(elapsed, std::chrono::seconds(2));
}

TEST(MusaBoundedSubprocessTest, StderrHasAnIndependentBound) {
  MusaSubprocessOptions options = Options({"--stderr-bytes=4096"});
  options.limits.max_stderr_bytes = 513;
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->output_limit_exceeded);
  EXPECT_TRUE(result->stdout_text.empty());
  EXPECT_EQ(result->stderr_text.size(), 513);
}

TEST(MusaBoundedSubprocessTest, TimeoutKillsTheProcessGroup) {
  const std::string sentinel =
      absl::StrCat(::testing::TempDir(), "/musa-subprocess-sentinel");
  unlink(sentinel.c_str());
  MusaSubprocessOptions options =
      Options({"--child-sentinel=" + sentinel, "--child-delay-ms=500",
               "--sleep-ms=2000"});
  options.limits.timeout = std::chrono::milliseconds(50);
  const auto started = std::chrono::steady_clock::now();
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->timed_out);
  EXPECT_FALSE(result->exited_successfully());
  EXPECT_LT(elapsed, std::chrono::seconds(2));
  std::this_thread::sleep_for(std::chrono::milliseconds(650));
  EXPECT_NE(access(sentinel.c_str(), F_OK), 0);
}

TEST(MusaBoundedSubprocessTest,
     DetachedDescendantCannotHoldRunnerOpenIndefinitely) {
  // A process group is not a security sandbox: a deliberately hostile child
  // can create a new session. Verify the documented residual limitation does
  // not let copied output descriptors defeat the public wall-time bound.
  const std::string sentinel =
      absl::StrCat(::testing::TempDir(), "/musa-detached-sentinel");
  unlink(sentinel.c_str());
  MusaSubprocessOptions options =
      Options({"--child-sentinel=" + sentinel, "--child-delay-ms=1300",
               "--child-detach", "--sleep-ms=5000"});
  options.limits.timeout = std::chrono::milliseconds(50);
  const auto started = std::chrono::steady_clock::now();
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->timed_out);
  EXPECT_LT(elapsed, std::chrono::seconds(2));
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  EXPECT_EQ(access(sentinel.c_str(), F_OK), 0);
  unlink(sentinel.c_str());
}

TEST(MusaBoundedSubprocessTest, RejectsAutoReapedChildren) {
  struct sigaction ignore = {};
  struct sigaction previous = {};
  ignore.sa_handler = SIG_IGN;
  ASSERT_EQ(sigemptyset(&ignore.sa_mask), 0);
  ASSERT_EQ(sigaction(SIGCHLD, &ignore, &previous), 0);
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(Options());
  const int restore_result = sigaction(SIGCHLD, &previous, nullptr);
  ASSERT_EQ(restore_result, 0);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kFailedPrecondition,
                               HasSubstr("waitable child")));
}

TEST(MusaBoundedSubprocessTest, ClosesEveryInheritedNonStdioDescriptor) {
  const int base = open("/dev/null", O_RDONLY);
  ASSERT_GE(base, 0);
  const int inherited = fcntl(base, F_DUPFD, 200);
  close(base);
  ASSERT_GE(inherited, 200);
  ASSERT_EQ(fcntl(inherited, F_SETFD, 0), 0);

  absl::StatusOr<MusaSubprocessResult> result = RunMusaBoundedSubprocess(
      Options({absl::StrCat("--check-fd=", inherited)}));
  close(inherited);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->exited_successfully());
  EXPECT_EQ(result->stdout_text, "closed");
}

TEST(MusaBoundedSubprocessTest, WorksWhenParentStdinIsClosed) {
  const int saved_stdin = fcntl(STDIN_FILENO, F_DUPFD_CLOEXEC, 3);
  if (saved_stdin < 0) GTEST_SKIP() << "test process has no stdin to save";
  ASSERT_EQ(close(STDIN_FILENO), 0);
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(Options({"--check-stdin-eof"}));
  const int restore_result = dup2(saved_stdin, STDIN_FILENO);
  close(saved_stdin);
  ASSERT_EQ(restore_result, STDIN_FILENO);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->exited_successfully());
  EXPECT_EQ(result->stdout_text, "eof");
}

TEST(MusaBoundedSubprocessTest, ResetsInheritedSignalMaskAndIgnoredSignals) {
  sigset_t blocked;
  sigset_t previous_mask;
  ASSERT_EQ(sigemptyset(&blocked), 0);
  ASSERT_EQ(sigaddset(&blocked, SIGUSR1), 0);
  ASSERT_EQ(pthread_sigmask(SIG_BLOCK, &blocked, &previous_mask), 0);

  struct sigaction ignore = {};
  struct sigaction previous_pipe = {};
  ignore.sa_handler = SIG_IGN;
  ASSERT_EQ(sigemptyset(&ignore.sa_mask), 0);
  ASSERT_EQ(sigaction(SIGPIPE, &ignore, &previous_pipe), 0);

  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(Options({"--check-signal-state"}));
  const int restore_pipe = sigaction(SIGPIPE, &previous_pipe, nullptr);
  const int restore_mask =
      pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr);
  ASSERT_EQ(restore_pipe, 0);
  ASSERT_EQ(restore_mask, 0);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->exited_successfully());
  EXPECT_EQ(result->stdout_text, "clean");
}

TEST(MusaBoundedSubprocessTest, FileSizeLimitIsEnforced) {
  const std::string output =
      absl::StrCat(::testing::TempDir(), "/bounded-subprocess-output");
  unlink(output.c_str());
  MusaSubprocessOptions options = Options({"--file-bytes=8192"});
  options.limits.max_file_bytes = 1024;
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->terminating_signal, SIGXFSZ);
  struct stat metadata = {};
  ASSERT_EQ(stat(output.c_str(), &metadata), 0);
  EXPECT_LE(metadata.st_size, 1024);
  unlink(output.c_str());
}

TEST(MusaBoundedSubprocessTest, AddressSpaceLimitRejectsOversizedMapping) {
  MusaSubprocessOptions options = Options({"--allocate-bytes=536870912"});
  options.limits.max_address_space_bytes = uint64_t{128} << 20;
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->exited_successfully());
  EXPECT_EQ(result->stdout_text, "allocation-failed");
}

TEST(MusaBoundedSubprocessTest, ExecFailureIsAStartedChildResult) {
  MusaSubprocessOptions options = Options();
  options.executable = "/definitely/missing/musa-compiler";
  absl::StatusOr<MusaSubprocessResult> result =
      RunMusaBoundedSubprocess(options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->exit_code, 127);
  EXPECT_THAT(result->stderr_text, HasSubstr("execve failed"));
}

TEST(MusaBoundedSubprocessTest, RejectsOpenEndedProcessControls) {
  MusaSubprocessOptions options = Options();
  options.executable = "relative-compiler";
  EXPECT_THAT(
      RunMusaBoundedSubprocess(options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("absolute")));

  options = Options();
  options.environment = {{"DUP", "one"}, {"DUP", "two"}};
  EXPECT_THAT(
      RunMusaBoundedSubprocess(options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("sorted")));

  options = Options();
  options.limits.timeout = std::chrono::milliseconds::zero();
  EXPECT_THAT(RunMusaBoundedSubprocess(options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("resource limits")));

  options = Options({std::string(1 << 20, 'x')});
  EXPECT_THAT(
      RunMusaBoundedSubprocess(options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("arguments")));

  options = Options();
  options.environment = {{"TOO_LARGE", std::string((64 << 10) + 1, 'x')}};
  EXPECT_THAT(
      RunMusaBoundedSubprocess(options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("environment")));

  options = Options();
  options.limits.max_stdout_bytes = (size_t{512} << 20) + 1;
  EXPECT_THAT(RunMusaBoundedSubprocess(options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("resource limits")));
}

TEST(MusaBoundedSubprocessTest, ConcurrentRunsKeepResultsIsolated) {
  constexpr int kWorkers = 8;
  std::vector<MusaSubprocessOptions> options;
  options.reserve(kWorkers);
  for (int i = 0; i < kWorkers; ++i) {
    options.push_back(Options({absl::StrCat("--stdout=worker-", i)}));
  }
  std::vector<int> ok(kWorkers, 0);
  std::vector<std::string> output(kWorkers);
  std::vector<std::thread> workers;
  for (int i = 0; i < kWorkers; ++i) {
    workers.emplace_back([&, i] {
      absl::StatusOr<MusaSubprocessResult> result =
          RunMusaBoundedSubprocess(options[i]);
      if (result.ok() && result->exited_successfully()) {
        output[i] = result->stdout_text;
        ok[i] = 1;
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  for (int i = 0; i < kWorkers; ++i) {
    EXPECT_EQ(ok[i], 1);
    EXPECT_EQ(output[i], absl::StrCat("worker-", i));
  }
}

}  // namespace
}  // namespace xla::gpu::musa
