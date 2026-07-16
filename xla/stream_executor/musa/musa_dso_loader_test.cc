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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

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

}  // namespace
}  // namespace stream_executor::musa::internal
