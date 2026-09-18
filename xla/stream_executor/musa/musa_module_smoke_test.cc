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

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::musa {
namespace {

TEST(MusaModuleSmokeTest, LoadsDeduplicatesAndUnloadsExternalMubin) {
  const char* path = std::getenv("MUSA_TEST_MUBIN");
  if (path == nullptr || path[0] == '\0') {
    GTEST_SKIP() << "Set MUSA_TEST_MUBIN to an external qualified MUBIN";
  }

  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input) << "Unable to open MUSA_TEST_MUBIN=" << path;
  std::vector<uint8_t> image((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
  ASSERT_FALSE(image.empty());

  MusaExecutor executor(/*platform=*/nullptr, /*device_ordinal=*/0);
  ASSERT_TRUE(executor.Init().ok());

  MultiModuleLoaderSpec spec;
  spec.AddMusaMubinInMemory(image);
  TF_ASSERT_OK_AND_ASSIGN(ModuleHandle first, executor.LoadModule(spec));
  TF_ASSERT_OK_AND_ASSIGN(ModuleHandle second, executor.LoadModule(spec));
  EXPECT_EQ(first, second);

  EXPECT_TRUE(executor.UnloadModule(first));
  EXPECT_TRUE(executor.UnloadModule(second));
  EXPECT_FALSE(executor.UnloadModule(second));
  EXPECT_TRUE(executor.SynchronizeAllActivity());
}

}  // namespace
}  // namespace stream_executor::musa
