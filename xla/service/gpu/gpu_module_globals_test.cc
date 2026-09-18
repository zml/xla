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

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Return;

auto DeviceDestination() {
  return testing::A<stream_executor::DeviceAddressBase*>();
}

auto HostSource() { return testing::A<const void*>(); }

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

TEST(GpuModuleGlobalsTest, SymbolFailureReturnsStatusAndUnloadsModule) {
  stream_executor::MockPlatform platform;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  std::vector<uint8_t> binary = {0x7f, 'E', 'L', 'F'};
  GpuModuleGlobals::ConstantInfo constant;
  constant.symbol_name = "missing_global";
  std::vector<GpuModuleGlobals::ConstantInfo> constants = {constant};
  stream_executor::ModuleHandle module(
      reinterpret_cast<void*>(uintptr_t{0x1234}));

  EXPECT_CALL(platform, id)
      .WillRepeatedly(Return(stream_executor::musa::kMusaPlatformId));
  EXPECT_CALL(executor, GetPlatform).WillRepeatedly(Return(&platform));
  EXPECT_CALL(stream, parent).WillRepeatedly(Return(&executor));
  EXPECT_CALL(executor, LoadModule(_)).WillOnce(Return(module));
  EXPECT_CALL(executor, GetSymbol("missing_global", module))
      .WillOnce(Return(absl::NotFoundError("injected missing symbol")));
  EXPECT_CALL(executor, UnloadModule(module)).WillOnce(Return(true));

  GpuModuleGlobals module_globals(binary, constants);
  absl::StatusOr<const GpuModuleGlobals::BufferAllocToDeviceMemoryMap*>
      resolved = module_globals.Resolve(&stream);
  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(resolved.status().message(),
              testing::HasSubstr("missing_global"));
}

TEST(GpuModuleGlobalsTest, OversizedInitializerReturnsStatusAndUnloadsModule) {
  stream_executor::MockPlatform platform;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  std::vector<uint8_t> binary = {0x7f, 'E', 'L', 'F'};
  GpuModuleGlobals::ConstantInfo constant;
  constant.symbol_name = "small_global";
  constant.content = DenseDataIntermediate::Own({1, 2, 3, 4, 5});
  std::vector<GpuModuleGlobals::ConstantInfo> constants = {constant};
  stream_executor::ModuleHandle module(
      reinterpret_cast<void*>(uintptr_t{0x1234}));

  EXPECT_CALL(platform, id)
      .WillRepeatedly(Return(stream_executor::musa::kMusaPlatformId));
  EXPECT_CALL(executor, GetPlatform).WillRepeatedly(Return(&platform));
  EXPECT_CALL(stream, parent).WillRepeatedly(Return(&executor));
  EXPECT_CALL(executor, LoadModule(_)).WillOnce(Return(module));
  EXPECT_CALL(executor, GetSymbol("small_global", module))
      .WillOnce(Return(stream_executor::DeviceAddressBase(
          reinterpret_cast<void*>(uintptr_t{0x2000}), 4)));
  EXPECT_CALL(executor, UnloadModule(module)).WillOnce(Return(true));

  GpuModuleGlobals module_globals(binary, constants);
  absl::StatusOr<const GpuModuleGlobals::BufferAllocToDeviceMemoryMap*>
      resolved = module_globals.Resolve(&stream);
  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(resolved.status().message(), testing::HasSubstr("5 bytes"));
}

TEST(GpuModuleGlobalsTest, ResolvesAllSymbolsBeforeSubmittingCopies) {
  stream_executor::MockPlatform platform;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  std::vector<uint8_t> binary = {0x7f, 'E', 'L', 'F'};
  GpuModuleGlobals::ConstantInfo first;
  first.symbol_name = "first_global";
  first.content = DenseDataIntermediate::Own({1, 2, 3, 4});
  GpuModuleGlobals::ConstantInfo missing;
  missing.symbol_name = "missing_global";
  std::vector<GpuModuleGlobals::ConstantInfo> constants = {first, missing};
  stream_executor::ModuleHandle module(
      reinterpret_cast<void*>(uintptr_t{0x1234}));

  EXPECT_CALL(platform, id)
      .WillRepeatedly(Return(stream_executor::musa::kMusaPlatformId));
  EXPECT_CALL(executor, GetPlatform).WillRepeatedly(Return(&platform));
  EXPECT_CALL(stream, parent).WillRepeatedly(Return(&executor));
  EXPECT_CALL(executor, LoadModule(_)).WillOnce(Return(module));
  {
    testing::InSequence sequence;
    EXPECT_CALL(executor, GetSymbol("first_global", module))
        .WillOnce(Return(stream_executor::DeviceAddressBase(
            reinterpret_cast<void*>(uintptr_t{0x2000}), 4)));
    EXPECT_CALL(executor, GetSymbol("missing_global", module))
        .WillOnce(Return(absl::NotFoundError("injected missing symbol")));
  }
  EXPECT_CALL(stream, Memcpy(DeviceDestination(), HostSource(), _)).Times(0);
  EXPECT_CALL(executor, UnloadModule(module)).WillOnce(Return(true));

  GpuModuleGlobals module_globals(binary, constants);
  auto resolved = module_globals.Resolve(&stream);
  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status().code(), absl::StatusCode::kNotFound);
}

TEST(GpuModuleGlobalsTest, SuccessfulInitializationRetainsBalancedHandle) {
  stream_executor::MockPlatform platform;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  std::vector<uint8_t> binary = {0x7f, 'E', 'L', 'F'};
  GpuModuleGlobals::ConstantInfo constant;
  constant.symbol_name = "initialized_global";
  constant.content = DenseDataIntermediate::Own({1, 2, 3, 4});
  constant.allocation_index = 7;
  std::vector<GpuModuleGlobals::ConstantInfo> constants = {constant};
  stream_executor::ModuleHandle module(
      reinterpret_cast<void*>(uintptr_t{0x1234}));
  stream_executor::DeviceAddressBase address(
      reinterpret_cast<void*>(uintptr_t{0x2000}), 4);

  EXPECT_CALL(platform, id)
      .WillRepeatedly(Return(stream_executor::musa::kMusaPlatformId));
  EXPECT_CALL(executor, GetPlatform).WillRepeatedly(Return(&platform));
  EXPECT_CALL(stream, parent).WillRepeatedly(Return(&executor));
  EXPECT_CALL(executor, LoadModule(_)).WillOnce(Return(module));
  EXPECT_CALL(executor, GetSymbol("initialized_global", module))
      .WillOnce(Return(address));
  EXPECT_CALL(stream, Memcpy(DeviceDestination(), HostSource(), 4))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(stream, BlockHostUntilDone).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(executor, UnloadModule(module)).WillOnce(Return(true));

  {
    GpuModuleGlobals module_globals(binary, constants);
    auto resolved = module_globals.Resolve(&stream);
    ASSERT_TRUE(resolved.ok()) << resolved.status();
    ASSERT_EQ((*resolved)->size(), 1);
    EXPECT_EQ((*resolved)->at(7), address);
  }
}

TEST(GpuModuleGlobalsTest, CopyFailureSynchronizesThenUnloads) {
  stream_executor::MockPlatform platform;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  std::vector<uint8_t> binary = {0x7f, 'E', 'L', 'F'};
  GpuModuleGlobals::ConstantInfo constant;
  constant.symbol_name = "copy_failure_global";
  constant.content = DenseDataIntermediate::Own({1});
  std::vector<GpuModuleGlobals::ConstantInfo> constants = {constant};
  stream_executor::ModuleHandle module(
      reinterpret_cast<void*>(uintptr_t{0x1234}));

  EXPECT_CALL(platform, id)
      .WillRepeatedly(Return(stream_executor::musa::kMusaPlatformId));
  EXPECT_CALL(executor, GetPlatform).WillRepeatedly(Return(&platform));
  EXPECT_CALL(stream, parent).WillRepeatedly(Return(&executor));
  EXPECT_CALL(executor, LoadModule(_)).WillOnce(Return(module));
  EXPECT_CALL(executor, GetSymbol("copy_failure_global", module))
      .WillOnce(Return(stream_executor::DeviceAddressBase(
          reinterpret_cast<void*>(uintptr_t{0x2000}), 1)));
  EXPECT_CALL(stream, Memcpy(DeviceDestination(), HostSource(), 1))
      .WillOnce(Return(absl::InternalError("injected copy failure")));
  EXPECT_CALL(stream, BlockHostUntilDone).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(executor, UnloadModule(module)).WillOnce(Return(true));

  GpuModuleGlobals module_globals(binary, constants);
  auto resolved = module_globals.Resolve(&stream);
  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(resolved.status().message(),
              testing::HasSubstr("injected copy failure"));
}

TEST(GpuModuleGlobalsTest, MusaSyncFailureReleasesHandleToExecutor) {
  stream_executor::MockPlatform platform;
  stream_executor::MockStreamExecutor executor;
  stream_executor::MockStream stream;
  std::vector<uint8_t> binary = {0x7f, 'E', 'L', 'F'};
  GpuModuleGlobals::ConstantInfo constant;
  constant.symbol_name = "sync_failure_global";
  constant.content = DenseDataIntermediate::Own({1});
  std::vector<GpuModuleGlobals::ConstantInfo> constants = {constant};
  stream_executor::ModuleHandle module(
      reinterpret_cast<void*>(uintptr_t{0x1234}));

  EXPECT_CALL(platform, id)
      .WillRepeatedly(Return(stream_executor::musa::kMusaPlatformId));
  EXPECT_CALL(executor, GetPlatform).WillRepeatedly(Return(&platform));
  EXPECT_CALL(stream, parent).WillRepeatedly(Return(&executor));
  EXPECT_CALL(executor, LoadModule(_)).WillOnce(Return(module));
  EXPECT_CALL(executor, GetSymbol("sync_failure_global", module))
      .WillOnce(Return(stream_executor::DeviceAddressBase(
          reinterpret_cast<void*>(uintptr_t{0x2000}), 1)));
  EXPECT_CALL(stream, Memcpy(DeviceDestination(), HostSource(), 1))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(stream, BlockHostUntilDone)
      .WillOnce(Return(absl::InternalError("injected sync failure")));
  EXPECT_CALL(executor, UnloadModule).Times(0);

  GpuModuleGlobals module_globals(binary, constants);
  auto resolved = module_globals.Resolve(&stream);
  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(resolved.status().message(),
              testing::HasSubstr("retaining the MUSA module"));
}

}  // namespace
}  // namespace xla::gpu
