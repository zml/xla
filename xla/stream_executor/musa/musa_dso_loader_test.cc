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

#include "xla/stream_executor/musa/musa_dso_loader.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include <unistd.h>
#include "xla/tsl/platform/env.h"

namespace stream_executor::musa::internal {
namespace {

TEST(MusaDsoLoaderTest, ExpandsBareSonamesAfterOriginalCandidates) {
  const std::vector<std::string> candidates = {
      "libfirst.so.1", "/opt/musa/lib/libabsolute.so", "relative/libpath.so",
      "liblast.so"};

  EXPECT_EQ(
      ExpandMusaDsoCandidates(candidates),
      (std::vector<std::string>{"libfirst.so.1", "/opt/musa/lib/libabsolute.so",
                                "relative/libpath.so", "liblast.so",
                                "/usr/local/musa/lib/libfirst.so.1",
                                "/usr/local/musa/lib/liblast.so"}));
}

TEST(MusaDsoLoaderTest, CachesExactLoadFailure) {
  std::unique_ptr<MusaSymbolLoader> loader = CreateMusaDsoLoader({});

  const absl::Status first = loader->Load();
  const absl::Status second = loader->Load();
  EXPECT_EQ(first, absl::InvalidArgumentError(
                       "No candidate MUSA shared libraries were provided"));
  EXPECT_EQ(second, first);
  EXPECT_TRUE(loader->loaded_path().empty());

  auto resolved = loader->Resolve("muInit");
  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status(), first);
}

TEST(MusaDsoLoaderTest,
     DistinguishesAbsenceFromConfiguredBrokenOrUninspectableDso) {
  tsl::Env* env = tsl::Env::Default();
  std::string invalid_dso;
  ASSERT_TRUE(env->LocalTempFilename(&invalid_dso));

  std::unique_ptr<MusaSymbolLoader> absent = CreateMusaDsoLoader({invalid_dso});
  EXPECT_TRUE(absl::IsNotFound(absent->Load()));

  std::unique_ptr<MusaSymbolLoader> configured_missing =
      CreateMusaDsoLoader({invalid_dso}, /*fail_if_not_found=*/true);
  EXPECT_TRUE(absl::IsFailedPrecondition(configured_missing->Load()));

  ASSERT_TRUE(tsl::WriteStringToFile(env, invalid_dso, "not an ELF DSO").ok());
  std::unique_ptr<MusaSymbolLoader> existing_but_invalid =
      CreateMusaDsoLoader({invalid_dso, "libc.so.6"});
  const absl::Status invalid_status = existing_but_invalid->Load();
  EXPECT_TRUE(absl::IsFailedPrecondition(invalid_status)) << invalid_status;
  EXPECT_TRUE(existing_but_invalid->loaded_path().empty());
  EXPECT_TRUE(env->DeleteFile(invalid_dso).ok());

  std::string missing_target;
  ASSERT_TRUE(env->LocalTempFilename(&missing_target));
  ASSERT_EQ(symlink(missing_target.c_str(), invalid_dso.c_str()), 0);
  std::unique_ptr<MusaSymbolLoader> dangling_symlink =
      CreateMusaDsoLoader({invalid_dso, "libc.so.6"});
  const absl::Status dangling_status = dangling_symlink->Load();
  EXPECT_TRUE(absl::IsFailedPrecondition(dangling_status)) << dangling_status;
  EXPECT_TRUE(dangling_symlink->loaded_path().empty());
  EXPECT_TRUE(env->DeleteFile(invalid_dso).ok());

  std::string loop_a;
  std::string loop_b;
  ASSERT_TRUE(env->LocalTempFilename(&loop_a));
  ASSERT_TRUE(env->LocalTempFilename(&loop_b));
  ASSERT_EQ(symlink(loop_b.c_str(), loop_a.c_str()), 0);
  ASSERT_EQ(symlink(loop_a.c_str(), loop_b.c_str()), 0);
  std::unique_ptr<MusaSymbolLoader> uninspectable_path =
      CreateMusaDsoLoader({loop_a + "/libshim.so", "libc.so.6"});
  const absl::Status inspect_status = uninspectable_path->Load();
  EXPECT_TRUE(absl::IsFailedPrecondition(inspect_status)) << inspect_status;
  EXPECT_TRUE(uninspectable_path->loaded_path().empty());
  EXPECT_TRUE(env->DeleteFile(loop_a).ok());
  EXPECT_TRUE(env->DeleteFile(loop_b).ok());
}

}  // namespace
}  // namespace stream_executor::musa::internal
