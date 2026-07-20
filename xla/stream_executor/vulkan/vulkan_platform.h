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

#ifndef XLA_STREAM_EXECUTOR_VULKAN_VULKAN_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_VULKAN_VULKAN_PLATFORM_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::vulkan {

// Vulkan platform entry in StreamExecutor's global PlatformManager registry.
// Device discovery and executor construction are added in later milestones.
class VulkanPlatform final : public Platform {
 public:
  VulkanPlatform();

  PlatformId id() const override;
  const std::string& Name() const override;
  int VisibleDeviceCount() const override;
  absl::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;
  absl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;

 private:
  std::string name_;
};

}  // namespace stream_executor::vulkan

#endif  // XLA_STREAM_EXECUTOR_VULKAN_VULKAN_PLATFORM_H_


