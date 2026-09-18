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
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>
#include "xla/tsl/platform/env.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/tsl/util/file_toc.h"

namespace stream_executor::musa::internal {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class MusaEmbeddedRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto root = tsl::testing::TemporaryDirectory::CreateForCurrentTestcase();
    ASSERT_THAT(root, IsOk());
    root_ =
        std::make_unique<tsl::testing::TemporaryDirectory>(*std::move(root));
    sdk_ = absl::StrCat(root_->path(), "/sdk");
    temporary_ = absl::StrCat(root_->path(), "/temporary");
    ASSERT_THAT(tsl::Env::Default()->CreateDir(sdk_), IsOk());
    ASSERT_THAT(tsl::Env::Default()->CreateDir(temporary_), IsOk());
  }
  auto Create() {
    return MusaEmbeddedRuntime::Create(sdk_, temporary_, files_);
  }
  void ExpectEmptyTemporaryRoot() {
    std::vector<std::string> entries;
    ASSERT_THAT(tsl::Env::Default()->GetChildren(temporary_, &entries), IsOk());
    EXPECT_TRUE(entries.empty());
  }
  std::unique_ptr<tsl::testing::TemporaryDirectory> root_;
  std::string sdk_;
  std::string temporary_;
  std::array<FileToc, 5> files_ = {{
      {"musa-llvm-bridge", "payload", 7, {}},
      {"libmusa_llvm_pass_guard.so", "payload", 7, {}},
      {"libxla_musa_mublas_shim.so", "payload", 7, {}},
      {"libxla_musa_mudnn_shim.so", "payload", 7, {}},
      {"libxla_musa_mufft_shim.so", "payload", 7, {}},
  }};
};

TEST_F(MusaEmbeddedRuntimeTest,
       ExtractsPrivatelyAndCleansWithoutFollowingSdkLink) {
  auto runtime = Create();
  ASSERT_THAT(runtime, IsOk());
  struct stat metadata;
  ASSERT_EQ(stat((*runtime)->directory().c_str(), &metadata), 0);
  EXPECT_EQ(metadata.st_mode & 0777, 0700);
  ASSERT_EQ(stat((*runtime)->Path("musa-compiler/musa-llvm-bridge").c_str(),
                 &metadata),
            0);
  EXPECT_EQ(metadata.st_mode & 0777, 0500);
  ASSERT_EQ(
      stat((*runtime)->Path("libxla_musa_mublas_shim.so.1").c_str(), &metadata),
      0);
  EXPECT_EQ(metadata.st_mode & 0777, 0400);
  char* resolved = realpath((*runtime)->Path("musa-sdk").c_str(), nullptr);
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(std::string(resolved), sdk_);
  std::free(resolved);
  EXPECT_THAT((*runtime)->Verify(), IsOk());
  runtime->reset();
  ExpectEmptyTemporaryRoot();
  EXPECT_EQ(access(sdk_.c_str(), F_OK), 0);
}

TEST_F(MusaEmbeddedRuntimeTest,
       DetectsCorruptionAndDoesNotFollowReplacementSymlink) {
  auto runtime = Create();
  ASSERT_THAT(runtime, IsOk());
  const std::string path = (*runtime)->Path("musa-compiler/musa-llvm-bridge");
  ASSERT_EQ(chmod(path.c_str(), 0600), 0);
  ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), path, "changed"),
              IsOk());
  EXPECT_THAT((*runtime)->Verify(), StatusIs(absl::StatusCode::kDataLoss,
                                             HasSubstr("digest mismatch")));
  ASSERT_EQ(unlink(path.c_str()), 0);
  ASSERT_EQ(symlink("/dev/zero", path.c_str()), 0);
  EXPECT_FALSE((*runtime)->Verify().ok());
}

TEST_F(MusaEmbeddedRuntimeTest,
       RejectsIncompleteAndDuplicatePayloadsAndCleansPartialWrites) {
  files_[4].name = "../../escaped";
  EXPECT_THAT(Create(), StatusIs(absl::StatusCode::kInvalidArgument));
  ExpectEmptyTemporaryRoot();
  files_[4].name = files_[0].name;
  EXPECT_THAT(Create(), StatusIs(absl::StatusCode::kInvalidArgument));
  ExpectEmptyTemporaryRoot();
  EXPECT_EQ(access(absl::StrCat(root_->path(), "/escaped").c_str(), F_OK), -1);
}

TEST_F(MusaEmbeddedRuntimeTest, RejectsInvalidSdkAndUnwritableTemporaryRoot) {
  EXPECT_THAT(MusaEmbeddedRuntime::Create("relative", temporary_, files_),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_FALSE(
      MusaEmbeddedRuntime::Create("/nonexistent-musa-sdk", temporary_, files_)
          .ok());
  if (geteuid() == 0)
    GTEST_SKIP() << "permission check requires an unprivileged user";
  ASSERT_EQ(chmod(temporary_.c_str(), 0500), 0);
  EXPECT_THAT(Create(), StatusIs(absl::StatusCode::kPermissionDenied));
  ASSERT_EQ(chmod(temporary_.c_str(), 0700), 0);
}

TEST_F(MusaEmbeddedRuntimeTest, ReportsNoexecFilesystem) {
  for (const char* directory : {"/dev/shm", "/run", "/proc"}) {
    struct statvfs mount;
    if (statvfs(directory, &mount) != 0 || (mount.f_flag & ST_NOEXEC) == 0)
      continue;
    EXPECT_THAT(MusaEmbeddedRuntime::Create(sdk_, directory, files_),
                StatusIs(absl::StatusCode::kPermissionDenied,
                         HasSubstr("XLA_MUSA_COMPILATION_TEMP_ROOT")));
    return;
  }
  GTEST_SKIP() << "no noexec test filesystem";
}

TEST_F(MusaEmbeddedRuntimeTest,
       ConcurrentDefaultInitializationExtractsOnceAndCleansAtExit) {
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    close(descriptors[0]);
    setenv("MUSA_PATH", sdk_.c_str(), 1);
    setenv("XLA_MUSA_COMPILATION_TEMP_ROOT", temporary_.c_str(), 1);
    std::array<const MusaEmbeddedRuntime*, 8> results = {};
    std::vector<std::thread> workers;
    for (size_t i = 0; i < results.size(); ++i) {
      workers.emplace_back([&, i] {
        auto runtime = GetMusaEmbeddedRuntime();
        if (runtime.ok()) results[i] = *runtime;
      });
    }
    for (auto& worker : workers) worker.join();
    for (const auto* runtime : results) {
      if (runtime == nullptr || runtime != results[0]) std::exit(2);
    }
    if (GetMusaEmbeddedRuntime(temporary_).status().code() !=
        absl::StatusCode::kFailedPrecondition)
      std::exit(4);
    const std::string directory = results[0]->directory();
    if (write(descriptors[1], directory.data(), directory.size()) !=
        directory.size())
      std::exit(3);
    close(descriptors[1]);
    std::exit(0);  // Exercise process-lifetime runtime destruction.
  }
  close(descriptors[1]);
  std::string directory;
  char buffer[4096];
  for (;;) {
    const ssize_t count = read(descriptors[0], buffer, sizeof(buffer));
    if (count < 0 && errno == EINTR) continue;
    ASSERT_GE(count, 0);
    if (count == 0) break;
    directory.append(buffer, count);
  }
  close(descriptors[0]);
  int status;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  ASSERT_FALSE(directory.empty());
  EXPECT_EQ(access(directory.c_str(), F_OK), -1);
  ExpectEmptyTemporaryRoot();
}

}  // namespace
}  // namespace stream_executor::musa::internal
