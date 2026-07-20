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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/vulkan/vulkan_platform_id.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::vulkan {
namespace {

using VkResult = int32_t;
using VkFlags = uint32_t;
using VkStructureType = uint32_t;
using VkInstance = struct VkInstance_T*;
using VkPhysicalDevice = struct VkPhysicalDevice_T*;
using PFN_vkVoidFunction = void (*)();
using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction (*)(VkInstance,
                                                         const char*);
using PFN_vkCreateInstance = VkResult (*)(const struct VkInstanceCreateInfo*,
                                          const void*, VkInstance*);
using PFN_vkEnumerateInstanceVersion = VkResult (*)(uint32_t*);
using PFN_vkEnumeratePhysicalDevices = VkResult (*)(VkInstance, uint32_t*,
                                                     VkPhysicalDevice*);

constexpr VkResult kVkSuccess = 0;
constexpr VkStructureType kVkStructureTypeApplicationInfo = 0;
constexpr VkStructureType kVkStructureTypeInstanceCreateInfo = 1;
constexpr uint32_t kVkApiVersion10 = 1u << 22;

struct VkApplicationInfo {
  VkStructureType sType;
  const void* pNext;
  const char* pApplicationName;
  uint32_t applicationVersion;
  const char* pEngineName;
  uint32_t engineVersion;
  uint32_t apiVersion;
};

struct VkInstanceCreateInfo {
  VkStructureType sType;
  const void* pNext;
  VkFlags flags;
  const VkApplicationInfo* pApplicationInfo;
  uint32_t enabledLayerCount;
  const char* const* ppEnabledLayerNames;
  uint32_t enabledExtensionCount;
  const char* const* ppEnabledExtensionNames;
};

struct VulkanInstance {
  // Keep the DSO open for at least as long as instance is alive. Vulkan's function pointers are owned by this DSO.
  void* loader = nullptr;
  VkInstance instance = nullptr;
  PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
  uint32_t api_version = kVkApiVersion10;
};

absl::StatusOr<VulkanInstance> CreateVulkanInstance() {
  VulkanInstance result;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->LoadDynamicLibrary("libvulkan.so.1",
                                                              &result.loader));

  void* get_instance_proc_addr_symbol = nullptr;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetSymbolFromLibrary(
      result.loader, "vkGetInstanceProcAddr", &get_instance_proc_addr_symbol));
  result.get_instance_proc_addr =
      reinterpret_cast<PFN_vkGetInstanceProcAddr>(get_instance_proc_addr_symbol);

  auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(
      result.get_instance_proc_addr(nullptr, "vkCreateInstance"));
  if (create_instance == nullptr) {
    return absl::NotFoundError(
        "libvulkan.so.1 does not export vkCreateInstance");
  }

  // vkEnumerateInstanceVersion was added in Vulkan 1.1. A Vulkan 1.0 loader remains valid, so use 1.0 when that optional query is unavailable.
  if (auto enumerate_instance_version =
          reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
              result.get_instance_proc_addr(nullptr,
                                            "vkEnumerateInstanceVersion"));
      enumerate_instance_version != nullptr) {
    VkResult query_result = enumerate_instance_version(&result.api_version);
    if (query_result != kVkSuccess) {
      return absl::InternalError(absl::StrCat(
          "vkEnumerateInstanceVersion failed with VkResult ", query_result));
    }
  }

  VkApplicationInfo application_info = {
      kVkStructureTypeApplicationInfo,  // sType
      nullptr,                           // pNext
      "XLA StreamExecutor",             // pApplicationName
      1,                                 // applicationVersion
      "XLA",                            // pEngineName
      1,                                 // engineVersion
      result.api_version,                // apiVersion
  };
  VkInstanceCreateInfo create_info = {
      kVkStructureTypeInstanceCreateInfo,  // sType
      nullptr,                             // pNext
      0,                                   // flags
      &application_info,                   // pApplicationInfo
      0,                                   // enabledLayerCount
      nullptr,                             // ppEnabledLayerNames
      0,                                   // enabledExtensionCount
      nullptr,                             // ppEnabledExtensionNames
  };

  VkResult create_result =
      create_instance(&create_info, nullptr, &result.instance);
  if (create_result != kVkSuccess) {
    return absl::InternalError(
        absl::StrCat("vkCreateInstance failed with VkResult ", create_result));
  }
  return result;
}

absl::StatusOr<const VulkanInstance*> GetVulkanInstance() {
  // A Platform is process-global. Retaining this object intentionally keeps
  // both the VkInstance and libvulkan loaded until process exit; no executor
  // or physical-device state is created here.
  static const absl::StatusOr<const VulkanInstance*> instance =
      []() -> absl::StatusOr<const VulkanInstance*> {
    absl::StatusOr<VulkanInstance> instance = CreateVulkanInstance();
    if (!instance.ok()) return instance.status();
    return new VulkanInstance(std::move(*instance));
  }();
  return instance;
}

}  // namespace

VulkanPlatform::VulkanPlatform() : name_(kVulkanPlatformId->ToName()) {}

PlatformId VulkanPlatform::id() const { return kVulkanPlatformId; }

const std::string& VulkanPlatform::Name() const { return name_; }

int VulkanPlatform::VisibleDeviceCount() const {
  absl::StatusOr<const VulkanInstance*> instance = GetVulkanInstance();
  if (!instance.ok()) {
    LOG(ERROR) << "Failed to initialize the Vulkan loader/instance: "
               << instance.status();
    return -1;
  }
  const VulkanInstance* vulkan_instance = *instance;

  auto enumerate_physical_devices =
      reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
          vulkan_instance->get_instance_proc_addr(
              vulkan_instance->instance, "vkEnumeratePhysicalDevices"));
  if (enumerate_physical_devices == nullptr) {
    LOG(ERROR) << "libvulkan.so.1 does not export "
               << "vkEnumeratePhysicalDevices";
    return -1;
  }

  uint32_t num_devices = 0;
  VkResult result = enumerate_physical_devices(vulkan_instance->instance,
                                                &num_devices, nullptr);

  if (result != kVkSuccess) {
    LOG(ERROR) << "Unable to enumerate Vulkan physical devices; VkResult: "
               << result;
    return -1;
  }

  return num_devices;
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
VulkanPlatform::DescriptionForDevice(int /*ordinal*/) const {
  return absl::UnimplementedError(
      "Vulkan device discovery is not implemented yet");
}

absl::StatusOr<StreamExecutor*> VulkanPlatform::ExecutorForDevice(
    int /*ordinal*/) {
  return absl::UnimplementedError(
      "Vulkan StreamExecutor construction is not implemented yet");
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


