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
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/vulkan/vulkan_executor.h"
#include "xla/stream_executor/vulkan/vulkan_platform_id.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::vulkan {

VulkanPlatform::VulkanPlatform() : name_(kVulkanPlatformId->ToName()) {}

Platform::Id VulkanPlatform::id() const { return kVulkanPlatformId; }

const std::string& VulkanPlatform::Name() const { return name_; }

int VulkanPlatform::VisibleDeviceCount() const {
  absl::StatusOr<int> count = VulkanExecutor::GetDeviceCount();
  if (!count.ok()) {
    LOG(ERROR) << "Failed to enumerate Vulkan devices: " << count.status();
    return -1;
  }
  return *count;
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
VulkanPlatform::DescriptionForDevice(int ordinal) const {
  return VulkanExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> VulkanPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal] { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<StreamExecutor*> VulkanPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
VulkanPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<VulkanExecutor>(this, ordinal);
  RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}

}  // namespace stream_executor::vulkan

namespace stream_executor {
namespace {

void InitializeVulkanPlatform() {
  TF_CHECK_OK(PlatformManager::RegisterPlatform(
      std::make_unique<vulkan::VulkanPlatform>()));
}

}  // namespace
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    vulkan_platform, stream_executor::InitializeVulkanPlatform());
