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

#include "xla/service/gpu/gpu_module_globals.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Return;

void ExpectModuleBinaryFormat(stream_executor::PlatformId platform_id,
                              bool expect_mubin) {
  stream_executor::MockPlatform platform;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  std::vector<uint8_t> binary = {0x7f, 'E', 'L', 'F'};
  std::vector<GpuModuleGlobals::ConstantInfo> constants;

  EXPECT_CALL(platform, id).WillRepeatedly(Return(platform_id));
  EXPECT_CALL(executor, GetPlatform).WillRepeatedly(Return(&platform));
  EXPECT_CALL(stream, parent).WillRepeatedly(Return(&executor));
  EXPECT_CALL(executor, LoadModule)
      .WillOnce(
          [expect_mubin](const stream_executor::MultiModuleLoaderSpec& spec)
              -> absl::StatusOr<stream_executor::ModuleHandle> {
            EXPECT_EQ(spec.has_musa_mubin_in_memory(), expect_mubin);
            EXPECT_EQ(spec.has_cuda_cubin_in_memory(), !expect_mubin);
            if (expect_mubin && spec.has_musa_mubin_in_memory()) {
              EXPECT_THAT(spec.musa_mubin_in_memory(),
                          ElementsAre(0x7f, 'E', 'L', 'F'));
            } else if (!expect_mubin && spec.has_cuda_cubin_in_memory()) {
              EXPECT_THAT(spec.cuda_cubin_in_memory(),
                          ElementsAre(0x7f, 'E', 'L', 'F'));
            }
            return stream_executor::ModuleHandle{};
          });

  GpuModuleGlobals module_globals(binary, constants);
  absl::StatusOr<const GpuModuleGlobals::BufferAllocToDeviceMemoryMap*>
      resolved = module_globals.Resolve(&stream);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_TRUE((*resolved)->empty());
}

TEST(GpuModuleGlobalsTest, RoutesMusaBinaryToMubinModuleSpec) {
  ExpectModuleBinaryFormat(stream_executor::musa::kMusaPlatformId,
                           /*expect_mubin=*/true);
}

TEST(GpuModuleGlobalsTest, PreservesRocmCubinModuleSpecCompatibility) {
  ExpectModuleBinaryFormat(stream_executor::rocm::kROCmPlatformId,
                           /*expect_mubin=*/false);
}

TEST(GpuModuleGlobalsTest, SkipsEmptyMusaModule) {
  stream_executor::MockPlatform platform;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;

  EXPECT_CALL(platform, id)
      .WillRepeatedly(Return(stream_executor::musa::kMusaPlatformId));
  EXPECT_CALL(executor, GetPlatform).WillRepeatedly(Return(&platform));
  EXPECT_CALL(stream, parent).WillRepeatedly(Return(&executor));
  EXPECT_CALL(executor, LoadModule).Times(0);

  std::vector<uint8_t> binary;
  std::vector<GpuModuleGlobals::ConstantInfo> constants;
  GpuModuleGlobals module_globals(binary, constants);
  absl::StatusOr<const GpuModuleGlobals::BufferAllocToDeviceMemoryMap*>
      resolved = module_globals.Resolve(&stream);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_TRUE((*resolved)->empty());
}

}  // namespace
}  // namespace xla::gpu
