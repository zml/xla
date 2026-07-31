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

#include "xla/stream_executor/vulkan/vulkan_platform.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/vulkan/vulkan_platform_id.h"
#include "xla/tsl/platform/status_matchers.h"

namespace stream_executor::vulkan {
namespace {

TEST(VulkanPlatformTest, RegistersWithPlatformManager) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("vulkan"));
  EXPECT_EQ(platform->id(), kVulkanPlatformId);
  EXPECT_EQ(platform->Name(), "VULKAN");
  EXPECT_GE(platform->VisibleDeviceCount(), 0);
}

TEST(VulkanPlatformTest, DeviceDescriptionIsCompleteOrActionable) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("vulkan"));
  if (platform->VisibleDeviceCount() == 0) {
    GTEST_SKIP() << "No Vulkan device is visible";
  }

  absl::StatusOr<std::unique_ptr<DeviceDescription>> description =
      platform->DescriptionForDevice(0);
  if (!description.ok()) {
    EXPECT_EQ(description.status().code(),
              absl::StatusCode::kFailedPrecondition);
    EXPECT_NE(description.status().message().find(
                  "Vulkan device performance description is incomplete"),
              std::string::npos);
    EXPECT_NE(description.status().message().find("missing:"),
              std::string::npos);
    EXPECT_NE(description.status().message().find("Explicit overrides"),
              std::string::npos);
    return;
  }

  EXPECT_GT((*description)->core_count(), 0);
  EXPECT_GT((*description)->threads_per_core_limit(), 0);
  EXPECT_GT((*description)->fpus_per_core(), 0);
  EXPECT_GT((*description)->memory_bandwidth(), 0);
  EXPECT_GT((*description)->l2_cache_size(), 0);
  EXPECT_GT((*description)->clock_rate_ghz(), 0.0f);
}

}  // namespace
}  // namespace stream_executor::vulkan
