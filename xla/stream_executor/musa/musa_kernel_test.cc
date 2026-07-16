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

#include "xla/stream_executor/musa/musa_kernel.h"

#include <atomic>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/mock_stream_executor.h"

namespace stream_executor::musa {
namespace {

TEST(MusaKernelTest, RetainsParentModuleAndDefersExecutionToC04) {
  std::atomic<bool> module_released = false;
  auto* module_pointer = reinterpret_cast<MusaModule*>(uintptr_t{0x1234});
  std::shared_ptr<MusaModule> module(
      module_pointer,
      [&module_released](MusaModule*) { module_released = true; });
  std::weak_ptr<MusaModule> weak_module = module;
  MockStreamExecutor executor;
  auto function = reinterpret_cast<MUfunction>(uintptr_t{0x5678});

  {
    MusaKernel kernel(&executor, function, module, /*arity=*/3);
    module.reset();

    EXPECT_EQ(kernel.Arity(), 3);
    EXPECT_EQ(kernel.function(), function);
    EXPECT_EQ(kernel.module().get(), module_pointer);
    EXPECT_FALSE(weak_module.expired());
    EXPECT_FALSE(module_released);
    EXPECT_EQ(
        kernel.GetMaxOccupiedBlocksPerCore(ThreadDim(), 0).status().code(),
        absl::StatusCode::kUnimplemented);
  }

  EXPECT_TRUE(weak_module.expired());
  EXPECT_TRUE(module_released);
}

}  // namespace
}  // namespace stream_executor::musa
