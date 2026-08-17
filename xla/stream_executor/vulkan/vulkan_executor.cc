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

#include "xla/stream_executor/vulkan/vulkan_executor.h"

#include <dlfcn.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::vulkan {
namespace {

constexpr char kValidationLayer[] = "VK_LAYER_KHRONOS_validation";

struct VulkanDevicePerformanceProperties {
  std::optional<int> core_count;
  std::optional<int64_t> threads_per_warp;
  std::optional<int64_t> threads_per_core;
  std::optional<int> fpus_per_core;
  std::optional<int64_t> memory_bandwidth;
  std::optional<int64_t> l2_cache_size;
  std::optional<float> clock_rate_ghz;
  std::array<uint8_t, VK_UUID_SIZE> device_uuid = {};
  bool has_device_uuid = false;
  std::string pci_bus_id;
  std::vector<std::string> probe_diagnostics;
};

bool HasExtension(const std::set<std::string>& extensions,
                  absl::string_view name) {
  return extensions.find(std::string(name)) != extensions.end();
}

template <typename T>
absl::Status ApplyPositiveIntegerOverride(const char* name,
                                          std::optional<T>* value,
                                          std::vector<std::string>* diagnostics) {
  const char* text = std::getenv(name);
  if (text == nullptr) return absl::OkStatus();
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || parsed == 0 ||
      parsed > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s must be a positive integer, got '%s'", name, text));
  }
  *value = static_cast<T>(parsed);
  diagnostics->push_back(absl::StrCat(name, ": explicit override applied"));
  return absl::OkStatus();
}

absl::Status ApplyPositiveFloatOverride(const char* name,
                                        std::optional<float>* value,
                                        std::vector<std::string>* diagnostics) {
  const char* text = std::getenv(name);
  if (text == nullptr) return absl::OkStatus();
  char* end = nullptr;
  float parsed = std::strtof(text, &end);
  if (end == text || *end != '\0' || !(parsed > 0.0f)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s must be a positive number, got '%s'", name, text));
  }
  *value = parsed;
  diagnostics->push_back(absl::StrCat(name, ": explicit override applied"));
  return absl::OkStatus();
}

std::string FormatUuid(
    const std::array<uint8_t, VK_UUID_SIZE>& uuid,
    absl::string_view prefix = {}) {
  std::string result(prefix);
  for (size_t i = 0; i < uuid.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) result.push_back('-');
    absl::StrAppendFormat(&result, "%02x", uuid[i]);
  }
  return result;
}

std::string NormalizeUuid(absl::string_view uuid) {
  std::string result;
  result.reserve(2 * VK_UUID_SIZE);
  for (unsigned char character : uuid) {
    if (std::isxdigit(character)) {
      result.push_back(static_cast<char>(std::tolower(character)));
    }
  }
  if (result.size() > 2 * VK_UUID_SIZE) {
    result.erase(0, result.size() - 2 * VK_UUID_SIZE);
  }
  return result;
}

using CuDevice = int;
using CuResult = int;
struct CuUuid {
  char bytes[VK_UUID_SIZE];
};

constexpr CuResult kCudaSuccess = 0;
constexpr int kCudaAttributeClockRate = 13;
constexpr int kCudaAttributeMultiprocessorCount = 16;
constexpr int kCudaAttributeMemoryClockRate = 36;
constexpr int kCudaAttributeGlobalMemoryBusWidth = 37;
constexpr int kCudaAttributeL2CacheSize = 38;
constexpr int kCudaAttributeMaxThreadsPerMultiprocessor = 39;

struct CudaApi {
  using Init = CuResult (*)(unsigned int);
  using DeviceGetCount = CuResult (*)(int*);
  using DeviceGet = CuResult (*)(CuDevice*, int);
  using DeviceGetUuid = CuResult (*)(CuUuid*, CuDevice);
  using DeviceGetAttribute = CuResult (*)(int*, int, CuDevice);
  using DeviceGetPciBusId = CuResult (*)(char*, int, CuDevice);

  void* library = nullptr;
  Init init = nullptr;
  DeviceGetCount device_get_count = nullptr;
  DeviceGet device_get = nullptr;
  DeviceGetUuid device_get_uuid = nullptr;
  DeviceGetAttribute device_get_attribute = nullptr;
  DeviceGetPciBusId device_get_pci_bus_id = nullptr;
  std::string error;
};

const CudaApi& GetCudaApi() {
  static const CudaApi* api = [] {
    CudaApi* result = new CudaApi;
    result->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (result->library == nullptr) {
      const char* error = dlerror();
      result->error = error == nullptr ? "dlopen failed" : error;
      return result;
    }
    auto load = [result](const char* name) {
      return dlsym(result->library, name);
    };
    result->init = reinterpret_cast<CudaApi::Init>(load("cuInit"));
    result->device_get_count = reinterpret_cast<CudaApi::DeviceGetCount>(
        load("cuDeviceGetCount"));
    result->device_get =
        reinterpret_cast<CudaApi::DeviceGet>(load("cuDeviceGet"));
    result->device_get_uuid = reinterpret_cast<CudaApi::DeviceGetUuid>(
        load("cuDeviceGetUuid_v2"));
    if (result->device_get_uuid == nullptr) {
      result->device_get_uuid = reinterpret_cast<CudaApi::DeviceGetUuid>(
          load("cuDeviceGetUuid"));
    }
    result->device_get_attribute =
        reinterpret_cast<CudaApi::DeviceGetAttribute>(
            load("cuDeviceGetAttribute"));
    result->device_get_pci_bus_id =
        reinterpret_cast<CudaApi::DeviceGetPciBusId>(
            load("cuDeviceGetPCIBusId"));
    if (result->init == nullptr || result->device_get_count == nullptr ||
        result->device_get == nullptr ||
        result->device_get_attribute == nullptr ||
        result->device_get_pci_bus_id == nullptr) {
      result->error = "required discovery symbols are missing";
    } else if (result->init(0) != kCudaSuccess) {
      result->error = "initialization failed";
    }
    return result;
  }();
  return *api;
}

void ProbeCuda(VulkanDevicePerformanceProperties* performance) {
  const CudaApi& api = GetCudaApi();
  if (!api.error.empty()) {
    performance->probe_diagnostics.push_back(
        absl::StrCat("CUDA unavailable: ", api.error));
    return;
  }

  int device_count = 0;
  if (api.device_get_count(&device_count) != kCudaSuccess) {
    performance->probe_diagnostics.push_back(
        "CUDA could not query the device count");
    return;
  }

  std::optional<CuDevice> matched_device;
  if (performance->has_device_uuid && api.device_get_uuid != nullptr) {
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
      CuDevice device = 0;
      CuUuid uuid = {};
      if (api.device_get(&device, ordinal) == kCudaSuccess &&
          api.device_get_uuid(&uuid, device) == kCudaSuccess &&
          std::memcmp(uuid.bytes, performance->device_uuid.data(),
                      VK_UUID_SIZE) == 0) {
        matched_device = device;
        performance->probe_diagnostics.push_back(
            "CUDA device matched by UUID");
        break;
      }
    }
  } else if (api.device_get_uuid == nullptr) {
    performance->probe_diagnostics.push_back(
        "CUDA UUID query symbol is unavailable");
  }

  if (!matched_device.has_value() && !performance->pci_bus_id.empty()) {
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
      CuDevice device = 0;
      std::array<char, 32> pci_bus_id = {};
      if (api.device_get(&device, ordinal) != kCudaSuccess ||
          api.device_get_pci_bus_id(pci_bus_id.data(),
                                    static_cast<int>(pci_bus_id.size()),
                                    device) != kCudaSuccess ||
          performance->pci_bus_id != pci_bus_id.data()) {
        continue;
      }
      if (performance->has_device_uuid && api.device_get_uuid != nullptr) {
        CuUuid uuid = {};
        if (api.device_get_uuid(&uuid, device) != kCudaSuccess ||
            std::memcmp(uuid.bytes, performance->device_uuid.data(),
                        VK_UUID_SIZE) != 0) {
          performance->probe_diagnostics.push_back(
              "CUDA PCI identity matched, but the device UUID did not match");
          return;
        }
      }
      matched_device = device;
      performance->probe_diagnostics.push_back(
          "CUDA device matched by PCI identity");
      break;
    }
  }

  if (!matched_device.has_value()) {
    performance->probe_diagnostics.push_back(
        "CUDA did not find the Vulkan device by UUID or PCI identity");
    return;
  }

  auto query_attribute = [&](int attribute, absl::string_view name,
                             auto* output) {
    int value = 0;
    if (api.device_get_attribute(&value, attribute, *matched_device) ==
            kCudaSuccess &&
        value > 0) {
      *output = value;
      performance->probe_diagnostics.push_back(
          absl::StrFormat("CUDA %s=%d", name, value));
    } else {
      performance->probe_diagnostics.push_back(
          absl::StrCat("CUDA did not report ", name));
    }
  };
  query_attribute(kCudaAttributeMultiprocessorCount, "multiprocessor count",
                  &performance->core_count);
  query_attribute(kCudaAttributeMaxThreadsPerMultiprocessor,
                  "maximum threads per multiprocessor",
                  &performance->threads_per_core);
  query_attribute(kCudaAttributeL2CacheSize, "L2 cache size",
                  &performance->l2_cache_size);

  std::optional<int64_t> clock_rate_khz;
  query_attribute(kCudaAttributeClockRate, "clock rate", &clock_rate_khz);
  if (clock_rate_khz.has_value()) {
    performance->clock_rate_ghz =
        static_cast<float>(*clock_rate_khz) / 1'000'000.0f;
  }

  std::optional<int64_t> memory_clock_khz;
  std::optional<int64_t> memory_bus_width_bits;
  query_attribute(kCudaAttributeMemoryClockRate, "memory clock rate",
                  &memory_clock_khz);
  query_attribute(kCudaAttributeGlobalMemoryBusWidth, "memory bus width",
                  &memory_bus_width_bits);
  if (memory_clock_khz.has_value() && memory_bus_width_bits.has_value()) {
    performance->memory_bandwidth =
        2 * *memory_clock_khz * 1000 * *memory_bus_width_bits / 8;
    performance->probe_diagnostics.push_back(absl::StrFormat(
        "CUDA derived memory bandwidth=%d bytes/s",
        *performance->memory_bandwidth));
  }
}

void ProbeNvml(VulkanDevicePerformanceProperties* performance) {
  using NvmlDevice = struct nvmlDevice_st*;
  using NvmlReturn = int;
  constexpr NvmlReturn kNvmlSuccess = 0;

  using Init = NvmlReturn (*)();
  using GetHandle = NvmlReturn (*)(const char*, NvmlDevice*);
  using GetUuid = NvmlReturn (*)(NvmlDevice, char*, unsigned int);
  using GetNumGpuCores = NvmlReturn (*)(NvmlDevice, unsigned int*);
  struct NvmlApi {
    void* library = nullptr;
    GetHandle get_handle_by_uuid = nullptr;
    GetHandle get_handle_by_pci_bus_id = nullptr;
    GetUuid get_uuid = nullptr;
    GetNumGpuCores get_num_gpu_cores = nullptr;
    std::string error;
  };
  static const NvmlApi* api = [] {
    NvmlApi* result = new NvmlApi;
    result->library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (result->library == nullptr) {
      const char* error = dlerror();
      result->error = error == nullptr ? "dlopen failed" : error;
      return result;
    }
    auto load = [result](const char* name) {
      return dlsym(result->library, name);
    };
    Init init = reinterpret_cast<Init>(load("nvmlInit_v2"));
    result->get_handle_by_uuid = reinterpret_cast<GetHandle>(
        load("nvmlDeviceGetHandleByUUID"));
    result->get_handle_by_pci_bus_id = reinterpret_cast<GetHandle>(
        load("nvmlDeviceGetHandleByPciBusId_v2"));
    result->get_uuid =
        reinterpret_cast<GetUuid>(load("nvmlDeviceGetUUID"));
    result->get_num_gpu_cores = reinterpret_cast<GetNumGpuCores>(
        load("nvmlDeviceGetNumGpuCores"));
    if (init == nullptr || result->get_handle_by_uuid == nullptr ||
        result->get_handle_by_pci_bus_id == nullptr ||
        result->get_uuid == nullptr || result->get_num_gpu_cores == nullptr) {
      result->error = "required discovery symbols are missing";
    } else if (init() != kNvmlSuccess) {
      result->error = "initialization failed";
    }
    return result;
  }();
  if (!api->error.empty()) {
    performance->probe_diagnostics.push_back(
        absl::StrCat("NVML unavailable: ", api->error));
    return;
  }

  NvmlDevice device = nullptr;
  if (performance->has_device_uuid) {
    const std::string uuid = FormatUuid(performance->device_uuid, "GPU-");
    if (api->get_handle_by_uuid(uuid.c_str(), &device) == kNvmlSuccess) {
      performance->probe_diagnostics.push_back(
          "NVML device matched by UUID");
    }
  }
  if (device == nullptr && !performance->pci_bus_id.empty() &&
      api->get_handle_by_pci_bus_id(performance->pci_bus_id.c_str(), &device) ==
          kNvmlSuccess) {
    std::array<char, 96> nvml_uuid = {};
    if (performance->has_device_uuid &&
        (api->get_uuid(device, nvml_uuid.data(),
                       static_cast<unsigned int>(nvml_uuid.size())) !=
             kNvmlSuccess ||
         NormalizeUuid(nvml_uuid.data()) !=
             NormalizeUuid(FormatUuid(performance->device_uuid)))) {
      performance->probe_diagnostics.push_back(
          "NVML PCI identity matched, but the device UUID did not match");
      return;
    }
    performance->probe_diagnostics.push_back(
        "NVML device matched by PCI identity");
  }
  if (device == nullptr) {
    performance->probe_diagnostics.push_back(
        "NVML did not find the Vulkan device by UUID or PCI identity");
    return;
  }

  unsigned int gpu_core_count = 0;
  if (api->get_num_gpu_cores(device, &gpu_core_count) != kNvmlSuccess ||
      gpu_core_count == 0) {
    performance->probe_diagnostics.push_back(
        "NVML did not report the GPU core count");
    return;
  }
  if (!performance->core_count.has_value() ||
      gpu_core_count % *performance->core_count != 0) {
    performance->probe_diagnostics.push_back(
        "NVML GPU core count cannot be divided by the CUDA multiprocessor "
        "count");
    return;
  }
  performance->fpus_per_core = gpu_core_count / *performance->core_count;
  performance->probe_diagnostics.push_back(absl::StrFormat(
      "NVML GPU core count=%u; FPUs per multiprocessor=%d", gpu_core_count,
      *performance->fpus_per_core));
}

absl::Status VulkanError(absl::string_view operation, VkResult result) {
  return absl::InternalError(
      absl::StrFormat("%s failed with VkResult %d", operation, result));
}

bool ValidationRequested() {
  const char* value = std::getenv("XLA_VULKAN_ENABLE_VALIDATION");
  return value != nullptr && std::strcmp(value, "0") != 0 &&
         std::strcmp(value, "false") != 0;
}

VKAPI_ATTR VkBool32 VKAPI_CALL ValidationCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*) {
  const char* message = callback_data == nullptr ? nullptr
                                                  : callback_data->pMessage;
  const char* id = callback_data == nullptr ? nullptr
                                             : callback_data->pMessageIdName;
  std::string text = absl::StrFormat(
      "Vulkan validation [%s, type=0x%x]: %s", id == nullptr ? "unknown" : id,
      type, message == nullptr ? "(no message)" : message);
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
    LOG(ERROR) << text;
    // Ask the validation layer to abort the offending Vulkan call. Commands
    // returning VkResult will report VK_ERROR_VALIDATION_FAILED, which the
    // executor converts to an absl::Status instead of continuing with invalid
    // Vulkan state.
    return VK_TRUE;
  } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) !=
             0) {
    LOG(WARNING) << text;
  } else {
    VLOG(1) << text;
  }
  return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT DebugMessengerCreateInfo() {
  VkDebugUtilsMessengerCreateInfoEXT create_info = {};
  create_info.sType =
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  create_info.pfnUserCallback = ValidationCallback;
  return create_info;
}

#define RETURN_IF_VK_ERROR(expr)            \
  do {                                      \
    VkResult result = (expr);               \
    if (result != VK_SUCCESS) {             \
      return VulkanError(#expr, result);    \
    }                                       \
  } while (false)

template <typename T>
absl::Status LoadGlobalProc(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                            const char* name, T* function) {
  *function = reinterpret_cast<T>(get_instance_proc_addr(VK_NULL_HANDLE, name));
  if (*function == nullptr) {
    return absl::NotFoundError(absl::StrCat("Vulkan loader is missing ", name));
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status LoadInstanceProc(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                              VkInstance instance, const char* name,
                              T* function) {
  *function = reinterpret_cast<T>(get_instance_proc_addr(instance, name));
  if (*function == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("Vulkan instance is missing ", name));
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status LoadDeviceProc(PFN_vkGetDeviceProcAddr get_device_proc_addr,
                            VkDevice device, const char* name, T* function) {
  *function = reinterpret_cast<T>(get_device_proc_addr(device, name));
  if (*function == nullptr) {
    return absl::NotFoundError(absl::StrCat("Vulkan device is missing ", name));
  }
  return absl::OkStatus();
}

struct VulkanShaderFeatures {
  bool shader_bfloat16 = false;
  bool storage_buffer_8bit_access = false;
  bool storage_buffer_16bit_access = false;
  bool shader_int8 = false;
  bool shader_int16 = false;
  bool shader_int64 = false;
  bool storage_buffer_array_dynamic_indexing = false;
  bool variable_pointers_storage_buffer = false;
};

class VulkanDriver {
 public:
  ~VulkanDriver() = default;

  absl::Status Initialize() {
    std::call_once(once_, [this] { status_ = InitializeOnce(); });
    return status_;
  }

  absl::Span<const VkPhysicalDevice> physical_devices() const {
    return physical_devices_;
  }

  absl::StatusOr<VulkanShaderFeatures> GetShaderFeatures(
      VkPhysicalDevice physical_device) const {
    uint32_t extension_count = 0;
    RETURN_IF_VK_ERROR(enumerate_device_extension_properties(
        physical_device, /*pLayerName=*/nullptr, &extension_count,
        /*pProperties=*/nullptr));
    std::vector<VkExtensionProperties> extensions(extension_count);
    if (extension_count != 0) {
      RETURN_IF_VK_ERROR(enumerate_device_extension_properties(
          physical_device, /*pLayerName=*/nullptr, &extension_count,
          extensions.data()));
    }
    const bool has_extension = std::any_of(
        extensions.begin(), extensions.end(), [](const auto& extension) {
          return std::strcmp(extension.extensionName,
                             VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) == 0;
        });
    VkPhysicalDevice8BitStorageFeatures storage_8bit_features = {};
    storage_8bit_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    VkPhysicalDevice16BitStorageFeatures storage_16bit_features = {};
    storage_16bit_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceShaderFloat16Int8Features float16_int8_features = {};
    float16_int8_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    VkPhysicalDeviceVariablePointersFeatures variable_pointer_features = {};
    variable_pointer_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;
    storage_8bit_features.pNext = &storage_16bit_features;
    storage_16bit_features.pNext = &float16_int8_features;
    float16_int8_features.pNext = &variable_pointer_features;
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features = {};
    if (has_extension) {
      bfloat16_features.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
      variable_pointer_features.pNext = &bfloat16_features;
    }
    VkPhysicalDeviceFeatures2 features = {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &storage_8bit_features;
    get_physical_device_features2(physical_device, &features);
    VulkanShaderFeatures shader_features;
    shader_features.shader_bfloat16 =
        has_extension && bfloat16_features.shaderBFloat16Type == VK_TRUE;
    shader_features.storage_buffer_8bit_access =
        storage_8bit_features.storageBuffer8BitAccess == VK_TRUE;
    shader_features.storage_buffer_16bit_access =
        storage_16bit_features.storageBuffer16BitAccess == VK_TRUE;
    shader_features.shader_int8 =
        float16_int8_features.shaderInt8 == VK_TRUE;
    shader_features.shader_int16 = features.features.shaderInt16 == VK_TRUE;
    shader_features.shader_int64 = features.features.shaderInt64 == VK_TRUE;
    shader_features.storage_buffer_array_dynamic_indexing =
        features.features.shaderStorageBufferArrayDynamicIndexing == VK_TRUE;
    shader_features.variable_pointers_storage_buffer =
        variable_pointer_features.variablePointersStorageBuffer == VK_TRUE;
    return shader_features;
  }

  PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
  PFN_vkGetPhysicalDeviceProperties get_physical_device_properties = nullptr;
  PFN_vkGetPhysicalDeviceProperties2 get_physical_device_properties2 = nullptr;
  PFN_vkGetPhysicalDeviceFeatures2 get_physical_device_features2 = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties get_physical_device_memory_properties =
      nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties2
      get_physical_device_memory_properties2 = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties
      get_physical_device_queue_family_properties = nullptr;
  PFN_vkEnumerateDeviceExtensionProperties
      enumerate_device_extension_properties = nullptr;
  PFN_vkCreateDevice create_device = nullptr;

 private:
  absl::Status InitializeOnce() {
    library_ = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("Unable to load libvulkan.so.1: ", dlerror()));
    }

    auto get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        dlsym(library_, "vkGetInstanceProcAddr"));
    if (get_instance_proc_addr == nullptr) {
      return absl::NotFoundError(
          "libvulkan.so.1 does not export vkGetInstanceProcAddr");
    }

    PFN_vkCreateInstance create_instance = nullptr;
    RETURN_IF_ERROR(LoadGlobalProc(get_instance_proc_addr, "vkCreateInstance",
                                   &create_instance));

    const bool validation_requested = ValidationRequested();
    std::vector<const char*> enabled_layers;
    std::vector<const char*> enabled_extensions;
    bool synchronization_validation = false;
    if (validation_requested) {
      PFN_vkEnumerateInstanceLayerProperties enumerate_layers = nullptr;
      PFN_vkEnumerateInstanceExtensionProperties enumerate_extensions =
          nullptr;
      RETURN_IF_ERROR(LoadGlobalProc(get_instance_proc_addr,
                                     "vkEnumerateInstanceLayerProperties",
                                     &enumerate_layers));
      RETURN_IF_ERROR(LoadGlobalProc(get_instance_proc_addr,
                                     "vkEnumerateInstanceExtensionProperties",
                                     &enumerate_extensions));

      uint32_t layer_count = 0;
      RETURN_IF_VK_ERROR(enumerate_layers(&layer_count, nullptr));
      std::vector<VkLayerProperties> layers(layer_count);
      RETURN_IF_VK_ERROR(enumerate_layers(&layer_count, layers.data()));
      bool has_validation_layer =
          std::any_of(layers.begin(), layers.end(), [](const auto& layer) {
            return std::strcmp(layer.layerName, kValidationLayer) == 0;
          });
      if (!has_validation_layer) {
        return absl::FailedPreconditionError(absl::StrCat(
            "XLA_VULKAN_ENABLE_VALIDATION is set, but ", kValidationLayer,
            " is not installed"));
      }
      enabled_layers.push_back(kValidationLayer);

      uint32_t extension_count = 0;
      RETURN_IF_VK_ERROR(
          enumerate_extensions(nullptr, &extension_count, nullptr));
      std::vector<VkExtensionProperties> extensions(extension_count);
      RETURN_IF_VK_ERROR(enumerate_extensions(nullptr, &extension_count,
                                              extensions.data()));
      bool has_debug_utils = std::any_of(
          extensions.begin(), extensions.end(), [](const auto& extension) {
            return std::strcmp(extension.extensionName,
                               VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
          });
      if (!has_debug_utils) {
        return absl::FailedPreconditionError(
            "Vulkan validation requested, but VK_EXT_debug_utils is missing");
      }
      enabled_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

      extension_count = 0;
      RETURN_IF_VK_ERROR(enumerate_extensions(
          kValidationLayer, &extension_count, nullptr));
      extensions.resize(extension_count);
      RETURN_IF_VK_ERROR(enumerate_extensions(
          kValidationLayer, &extension_count, extensions.data()));
      synchronization_validation = std::any_of(
          extensions.begin(), extensions.end(), [](const auto& extension) {
            return std::strcmp(extension.extensionName,
                               VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) == 0;
          });
      if (synchronization_validation) {
        enabled_extensions.push_back(
            VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
      }
    }

    VkApplicationInfo application_info = {};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "XLA Vulkan StreamExecutor";
    application_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application_info.pEngineName = "XLA";
    application_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &application_info;
    create_info.enabledLayerCount =
        static_cast<uint32_t>(enabled_layers.size());
    create_info.ppEnabledLayerNames =
        enabled_layers.empty() ? nullptr : enabled_layers.data();
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(enabled_extensions.size());
    create_info.ppEnabledExtensionNames =
        enabled_extensions.empty() ? nullptr : enabled_extensions.data();

    // Put the debug messenger in the instance create chain so validation can
    // report errors raised by vkCreateInstance and vkDestroyInstance too.
    VkDebugUtilsMessengerCreateInfoEXT debug_create_info =
        DebugMessengerCreateInfo();
    VkValidationFeatureEnableEXT synchronization_feature =
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT validation_features = {};
    if (validation_requested) {
      create_info.pNext = &debug_create_info;
      if (synchronization_validation) {
        validation_features.sType =
            VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        validation_features.enabledValidationFeatureCount = 1;
        validation_features.pEnabledValidationFeatures =
            &synchronization_feature;
        debug_create_info.pNext = &validation_features;
      }
    }
    RETURN_IF_VK_ERROR(
        create_instance(&create_info, /*pAllocator=*/nullptr, &instance_));

    if (validation_requested) {
      PFN_vkCreateDebugUtilsMessengerEXT create_debug_messenger = nullptr;
      RETURN_IF_ERROR(LoadInstanceProc(
          get_instance_proc_addr, instance_, "vkCreateDebugUtilsMessengerEXT",
          &create_debug_messenger));
      RETURN_IF_VK_ERROR(create_debug_messenger(
          instance_, &debug_create_info, nullptr, &debug_messenger_));
      LOG(INFO) << "Enabled " << kValidationLayer
                << (synchronization_validation
                        ? " with synchronization validation"
                        : " (synchronization validation unavailable)");
    }

    PFN_vkEnumeratePhysicalDevices enumerate_physical_devices = nullptr;
    RETURN_IF_ERROR(LoadInstanceProc(get_instance_proc_addr, instance_,
                                     "vkEnumeratePhysicalDevices",
                                     &enumerate_physical_devices));
    RETURN_IF_ERROR(LoadInstanceProc(get_instance_proc_addr, instance_,
                                     "vkGetPhysicalDeviceProperties",
                                     &get_physical_device_properties));
    RETURN_IF_ERROR(LoadInstanceProc(get_instance_proc_addr, instance_,
                                     "vkGetPhysicalDeviceProperties2",
                                     &get_physical_device_properties2));
    RETURN_IF_ERROR(LoadInstanceProc(get_instance_proc_addr, instance_,
                                     "vkGetPhysicalDeviceFeatures2",
                                     &get_physical_device_features2));
    RETURN_IF_ERROR(LoadInstanceProc(
        get_instance_proc_addr, instance_,
        "vkGetPhysicalDeviceMemoryProperties",
        &get_physical_device_memory_properties));
    RETURN_IF_ERROR(LoadInstanceProc(
        get_instance_proc_addr, instance_,
        "vkGetPhysicalDeviceMemoryProperties2",
        &get_physical_device_memory_properties2));
    RETURN_IF_ERROR(LoadInstanceProc(
        get_instance_proc_addr, instance_,
        "vkGetPhysicalDeviceQueueFamilyProperties",
        &get_physical_device_queue_family_properties));
    RETURN_IF_ERROR(LoadInstanceProc(
        get_instance_proc_addr, instance_,
        "vkEnumerateDeviceExtensionProperties",
        &enumerate_device_extension_properties));
    RETURN_IF_ERROR(LoadInstanceProc(get_instance_proc_addr, instance_,
                                     "vkCreateDevice", &create_device));
    RETURN_IF_ERROR(LoadInstanceProc(get_instance_proc_addr, instance_,
                                     "vkGetDeviceProcAddr",
                                     &get_device_proc_addr));

    uint32_t count = 0;
    RETURN_IF_VK_ERROR(
        enumerate_physical_devices(instance_, &count, /*pPhysicalDevices=*/nullptr));
    physical_devices_.resize(count);
    if (count != 0) {
      RETURN_IF_VK_ERROR(enumerate_physical_devices(
          instance_, &count, physical_devices_.data()));
      physical_devices_.resize(count);
    }

    physical_devices_.erase(
        std::remove_if(
            physical_devices_.begin(), physical_devices_.end(),
            [this](VkPhysicalDevice device) {
              VkPhysicalDeviceProperties properties = {};
              get_physical_device_properties(device, &properties);
              if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
                return false;
              }
              LOG(INFO) << "Ignoring CPU Vulkan device "
                        << properties.deviceName;
              return true;
            }),
        physical_devices_.end());
    return absl::OkStatus();
  }

  std::once_flag once_;
  absl::Status status_;
  void* library_ = nullptr;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
  std::vector<VkPhysicalDevice> physical_devices_;
};

VulkanDriver& Driver() {
  static VulkanDriver* driver = new VulkanDriver;
  return *driver;
}

absl::StatusOr<VulkanDevicePerformanceProperties>
QueryDevicePerformanceProperties(VkPhysicalDevice physical_device,
                                 VkPhysicalDeviceProperties* properties) {
  VkPhysicalDeviceProperties2 properties2 = {};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  Driver().get_physical_device_properties2(physical_device, &properties2);
  *properties = properties2.properties;
  if (properties->apiVersion < VK_API_VERSION_1_2) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Vulkan device %s supports Vulkan %d.%d, but Vulkan 1.2 is required",
        properties->deviceName, VK_API_VERSION_MAJOR(properties->apiVersion),
        VK_API_VERSION_MINOR(properties->apiVersion)));
  }

  uint32_t extension_count = 0;
  RETURN_IF_VK_ERROR(Driver().enumerate_device_extension_properties(
      physical_device, /*pLayerName=*/nullptr, &extension_count,
      /*pProperties=*/nullptr));
  std::vector<VkExtensionProperties> extension_properties(extension_count);
  if (extension_count != 0) {
    RETURN_IF_VK_ERROR(Driver().enumerate_device_extension_properties(
        physical_device, /*pLayerName=*/nullptr, &extension_count,
        extension_properties.data()));
  }
  std::set<std::string> extensions;
  for (const VkExtensionProperties& extension : extension_properties) {
    extensions.insert(extension.extensionName);
  }

  VkPhysicalDeviceSubgroupProperties subgroup = {};
  subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceIDProperties device_id = {};
  device_id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {};
  pci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;

  VkBaseOutStructure* tail =
      reinterpret_cast<VkBaseOutStructure*>(&properties2);
  auto append = [&tail](void* property) {
    VkBaseOutStructure* next = reinterpret_cast<VkBaseOutStructure*>(property);
    tail->pNext = next;
    tail = next;
  };
  append(&subgroup);
  append(&device_id);
  const bool has_pci =
      HasExtension(extensions, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
  if (has_pci) append(&pci);
  Driver().get_physical_device_properties2(physical_device, &properties2);

  VulkanDevicePerformanceProperties performance;
  if (subgroup.subgroupSize != 0) {
    performance.threads_per_warp = subgroup.subgroupSize;
    performance.probe_diagnostics.push_back(absl::StrFormat(
        "Vulkan subgroup size=%u", subgroup.subgroupSize));
  } else {
    performance.probe_diagnostics.push_back(
        "Vulkan did not report a subgroup size");
  }
  if (std::any_of(device_id.deviceUUID,
                  device_id.deviceUUID + VK_UUID_SIZE,
                  [](uint8_t byte) { return byte != 0; })) {
    std::copy_n(device_id.deviceUUID, VK_UUID_SIZE,
                performance.device_uuid.begin());
    performance.has_device_uuid = true;
    performance.probe_diagnostics.push_back(absl::StrCat(
        "Vulkan device UUID=", FormatUuid(performance.device_uuid)));
  } else {
    performance.probe_diagnostics.push_back(
        "Vulkan did not report a device UUID");
  }

  if (has_pci) {
    performance.pci_bus_id =
        absl::StrFormat("%04x:%02x:%02x.%x", pci.pciDomain, pci.pciBus,
                        pci.pciDevice, pci.pciFunction);
    performance.probe_diagnostics.push_back(
        absl::StrCat("Vulkan PCI identity=", performance.pci_bus_id));
  } else {
    performance.probe_diagnostics.push_back(
        "VK_EXT_pci_bus_info is not supported");
  }

  if (properties->vendorID == 0x10de) {
    ProbeCuda(&performance);
    ProbeNvml(&performance);
  } else {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Vulkan performance discovery supports NVIDIA devices only; device "
        "%s has vendor ID 0x%04x",
        properties->deviceName, properties->vendorID));
  }

  RETURN_IF_ERROR(ApplyPositiveIntegerOverride(
      "XLA_VULKAN_CORE_COUNT", &performance.core_count,
      &performance.probe_diagnostics));
  RETURN_IF_ERROR(ApplyPositiveIntegerOverride(
      "XLA_VULKAN_THREADS_PER_CORE", &performance.threads_per_core,
      &performance.probe_diagnostics));
  RETURN_IF_ERROR(ApplyPositiveIntegerOverride(
      "XLA_VULKAN_FPUS_PER_CORE", &performance.fpus_per_core,
      &performance.probe_diagnostics));
  RETURN_IF_ERROR(ApplyPositiveIntegerOverride(
      "XLA_VULKAN_MEMORY_BANDWIDTH_BYTES_PER_SECOND",
      &performance.memory_bandwidth, &performance.probe_diagnostics));
  RETURN_IF_ERROR(ApplyPositiveIntegerOverride(
      "XLA_VULKAN_L2_CACHE_SIZE_BYTES", &performance.l2_cache_size,
      &performance.probe_diagnostics));
  RETURN_IF_ERROR(ApplyPositiveFloatOverride(
      "XLA_VULKAN_CLOCK_RATE_GHZ", &performance.clock_rate_ghz,
      &performance.probe_diagnostics));
  return performance;
}

absl::Status ValidateDevicePerformanceProperties(
    const VkPhysicalDeviceProperties& properties,
    const VulkanDevicePerformanceProperties& performance) {
  std::vector<absl::string_view> missing;
  if (!performance.core_count.has_value()) missing.push_back("core_count");
  if (!performance.threads_per_warp.has_value()) {
    missing.push_back("threads_per_warp");
  }
  if (!performance.threads_per_core.has_value()) {
    missing.push_back("threads_per_core_limit");
  }
  if (!performance.fpus_per_core.has_value()) {
    missing.push_back("fpus_per_core");
  }
  if (!performance.memory_bandwidth.has_value()) {
    missing.push_back("memory_bandwidth");
  }
  if (!performance.l2_cache_size.has_value()) {
    missing.push_back("l2_cache_size");
  }
  if (!performance.clock_rate_ghz.has_value()) {
    missing.push_back("clock_rate_ghz");
  }
  if (missing.empty()) return absl::OkStatus();

  std::string message = absl::StrFormat(
      "Vulkan device performance description is incomplete:\n"
      "  device: %s\n"
      "  vendor/device: %04x:%04x\n"
      "  UUID: %s\n"
      "  PCI: %s\n"
      "  missing:",
      properties.deviceName, properties.vendorID, properties.deviceID,
      performance.has_device_uuid
          ? FormatUuid(performance.device_uuid)
          : "unavailable",
      performance.pci_bus_id.empty() ? "unavailable"
                                     : performance.pci_bus_id);
  for (absl::string_view field : missing) {
    absl::StrAppend(&message, "\n    ", field);
  }
  absl::StrAppend(&message, "\n  probes:");
  for (const std::string& diagnostic : performance.probe_diagnostics) {
    absl::StrAppend(&message, "\n    ", diagnostic);
  }
  absl::StrAppend(
      &message,
      "\n  Explicit overrides (positive values only):"
      "\n    XLA_VULKAN_CORE_COUNT"
      "\n    XLA_VULKAN_THREADS_PER_CORE"
      "\n    XLA_VULKAN_FPUS_PER_CORE"
      "\n    XLA_VULKAN_MEMORY_BANDWIDTH_BYTES_PER_SECOND"
      "\n    XLA_VULKAN_L2_CACHE_SIZE_BYTES"
      "\n    XLA_VULKAN_CLOCK_RATE_GHZ");
  return absl::FailedPreconditionError(message);
}

class VulkanEvent final : public Event {
 public:
  Status PollForStatus() override {
    if (executor_ == nullptr || value_ == 0) return Status::kComplete;
    absl::StatusOr<uint64_t> current = executor_->TimelineValue();
    if (!current.ok()) return Status::kError;
    return *current >= value_ ? Status::kComplete : Status::kPending;
  }
  absl::Status Synchronize() override {
    if (executor_ == nullptr || value_ == 0) return absl::OkStatus();
    return executor_->WaitTimeline(value_);
  }
  void SetSignal(VulkanExecutor* executor, uint64_t value) {
    executor_ = executor;
    value_ = value;
  }
  VulkanExecutor* executor() const { return executor_; }
  uint64_t value() const { return value_; }

 private:
  VulkanExecutor* executor_ = nullptr;
  uint64_t value_ = 0;
};

}  // namespace

class VulkanStream;

class VulkanKernel final : public Kernel {
 public:
  VulkanKernel(VulkanExecutor* executor, unsigned arity,
               std::vector<VulkanDescriptorBinding> bindings)
      : executor_(executor), arity_(arity), bindings_(std::move(bindings)) {}
  ~VulkanKernel() override { executor_->UnloadKernel(this); }

  unsigned Arity() const override { return arity_; }
  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override {
    return absl::UnimplementedError(
        "Vulkan occupancy calculation is not implemented");
  }

 private:
  friend class VulkanExecutor;

  absl::Status Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                      const std::optional<ClusterDim>& cluster_dims,
                      Stream* stream, const KernelArgs& args) override;

  VulkanExecutor* executor_;
  unsigned arity_;
  std::vector<VulkanDescriptorBinding> bindings_;
  VkShaderModule shader_module_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

struct VulkanExecutor::Impl {
  struct Allocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
  };

  struct AllocationView {
    Allocation* allocation;
    VkDeviceSize offset;
    VkDeviceSize size;
  };

  struct Submission {
    uint64_t value = 0;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  };

  absl::StatusOr<uint64_t> CurrentTimelineValue() const {
    uint64_t value = 0;
    VkResult result = vkGetSemaphoreCounterValue(device, timeline, &value);
    if (result != VK_SUCCESS) {
      return VulkanError("vkGetSemaphoreCounterValue", result);
    }
    return value;
  }

  absl::Status WaitForTimeline(uint64_t value) const {
    if (value == 0) return absl::OkStatus();
    VLOG(2) << "Waiting for Vulkan timeline value " << value;
    VkSemaphoreWaitInfo wait_info = {};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &timeline;
    wait_info.pValues = &value;
    VkResult result = vkWaitSemaphores(device, &wait_info, UINT64_MAX);
    if (result == VK_SUCCESS) {
      VLOG(2) << "Vulkan timeline reached value " << value;
    }
    return result == VK_SUCCESS
               ? absl::OkStatus()
               : VulkanError("vkWaitSemaphores", result);
  }

  void ReclaimCompleted() {
    absl::StatusOr<uint64_t> completed = CurrentTimelineValue();
    if (!completed.ok()) return;
    std::lock_guard<std::mutex> lock(queue_mutex);
    while (!submissions.empty() && submissions.front().value <= *completed) {
      Submission submission = submissions.front();
      submissions.pop_front();
      vkFreeCommandBuffers(device, command_pool, 1,
                           &submission.command_buffer);
      if (submission.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, submission.descriptor_pool, nullptr);
      }
    }
  }

  absl::StatusOr<uint64_t> Submit(VkCommandBuffer command_buffer,
                                  VkDescriptorPool descriptor_pool,
                                  uint64_t wait_value) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    uint64_t signal_value = ++next_timeline_value;
    VkTimelineSemaphoreSubmitInfo timeline_info = {};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = wait_value == 0 ? 0 : 1;
    timeline_info.pWaitSemaphoreValues = wait_value == 0 ? nullptr : &wait_value;
    timeline_info.signalSemaphoreValueCount = 1;
    timeline_info.pSignalSemaphoreValues = &signal_value;

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = &timeline_info;
    submit_info.waitSemaphoreCount = wait_value == 0 ? 0 : 1;
    submit_info.pWaitSemaphores = wait_value == 0 ? nullptr : &timeline;
    submit_info.pWaitDstStageMask = wait_value == 0 ? nullptr : &wait_stage;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &timeline;
    VkResult result = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) return VulkanError("vkQueueSubmit", result);
    VLOG(1) << "Submitted Vulkan command buffer: wait=" << wait_value
            << ", signal=" << signal_value;
    submissions.push_back(
        Submission{signal_value, command_buffer, descriptor_pool});
    return signal_value;
  }

  void ScheduleCallback(
      uint64_t value,
      absl::AnyInvocable<absl::Status() &&> callback) {
    {
      std::lock_guard<std::mutex> lock(callback_mutex);
      callbacks.emplace(value, std::move(callback));
    }
    callback_cv.notify_one();
  }

  void CompletionLoop() {
    while (true) {
      uint64_t value = 0;
      absl::AnyInvocable<absl::Status() &&> callback;
      {
        std::unique_lock<std::mutex> lock(callback_mutex);
        callback_cv.wait(lock,
                         [this] { return stop_callbacks || !callbacks.empty(); });
        if (stop_callbacks && callbacks.empty()) return;
        auto it = callbacks.begin();
        value = it->first;
        callback = std::move(it->second);
        callbacks.erase(it);
      }
      absl::Status status = WaitForTimeline(value);
      if (status.ok()) status = std::move(callback)();
      if (!status.ok()) {
        LOG(ERROR) << "Vulkan completion callback failed: " << status;
      }
      ReclaimCompleted();
    }
  }

  absl::Status Initialize(int ordinal) {
    RETURN_IF_ERROR(Driver().Initialize());
    if (ordinal < 0 || ordinal >= Driver().physical_devices().size()) {
      return absl::OutOfRangeError(
          absl::StrFormat("Vulkan device ordinal %d is out of range", ordinal));
    }
    physical_device = Driver().physical_devices()[ordinal];
    Driver().get_physical_device_properties(physical_device, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_2) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Vulkan 1.2 is required; device %s exposes %d.%d",
          properties.deviceName, VK_API_VERSION_MAJOR(properties.apiVersion),
          VK_API_VERSION_MINOR(properties.apiVersion)));
    }
    Driver().get_physical_device_memory_properties(physical_device,
                                                   &memory_properties);

    uint32_t extension_count = 0;
    RETURN_IF_VK_ERROR(Driver().enumerate_device_extension_properties(
        physical_device, /*pLayerName=*/nullptr, &extension_count,
        /*pProperties=*/nullptr));
    std::vector<VkExtensionProperties> extensions(extension_count);
    if (extension_count != 0) {
      RETURN_IF_VK_ERROR(Driver().enumerate_device_extension_properties(
          physical_device, /*pLayerName=*/nullptr, &extension_count,
          extensions.data()));
      extensions.resize(extension_count);
    }
    const bool has_memory_budget = std::any_of(
        extensions.begin(), extensions.end(), [](const auto& extension) {
          return std::strcmp(extension.extensionName,
                             VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
        });
    if (!has_memory_budget) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Vulkan device %s does not support required extension %s",
          properties.deviceName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME));
    }
    TF_ASSIGN_OR_RETURN(VulkanShaderFeatures shader_features,
                        Driver().GetShaderFeatures(physical_device));
    shader_bfloat16 = shader_features.shader_bfloat16;
    storage_buffer_16bit_access =
        shader_features.storage_buffer_16bit_access;
    std::string missing_features;
    auto add_missing_feature = [&missing_features](absl::string_view feature) {
      if (!missing_features.empty()) {
        absl::StrAppend(&missing_features, ", ");
      }
      absl::StrAppend(&missing_features, feature);
    };
    if (!shader_features.shader_int8) {
      add_missing_feature("shaderInt8");
    }
    if (!shader_features.storage_buffer_8bit_access) {
      add_missing_feature("storageBuffer8BitAccess");
    }
    if (!shader_features.shader_int16) {
      add_missing_feature("shaderInt16");
    }
    if (!shader_features.shader_int64) {
      add_missing_feature("shaderInt64");
    }
    if (!shader_features.storage_buffer_array_dynamic_indexing) {
      add_missing_feature("shaderStorageBufferArrayDynamicIndexing");
    }
    if (!shader_features.variable_pointers_storage_buffer) {
      add_missing_feature("variablePointersStorageBuffer");
    }
    if (!missing_features.empty()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Vulkan device %s does not support required features: %s",
          properties.deviceName, missing_features));
    }

    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features = {};
    timeline_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceFeatures2 features = {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &timeline_features;
    Driver().get_physical_device_features2(physical_device, &features);
    if (timeline_features.timelineSemaphore != VK_TRUE) {
      return absl::FailedPreconditionError(
          "Vulkan timelineSemaphore support is required");
    }
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features = {};
    VkPhysicalDevice8BitStorageFeatures storage_8bit_features = {};
    storage_8bit_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    storage_8bit_features.storageBuffer8BitAccess = VK_TRUE;
    VkPhysicalDevice16BitStorageFeatures storage_16bit_features = {};
    VkPhysicalDeviceShaderFloat16Int8Features float16_int8_features = {};
    float16_int8_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    float16_int8_features.shaderInt8 = VK_TRUE;
    VkPhysicalDeviceVariablePointersFeatures variable_pointer_features = {};
    variable_pointer_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;
    variable_pointer_features.variablePointersStorageBuffer = VK_TRUE;
    timeline_features.pNext = &storage_8bit_features;
    storage_8bit_features.pNext = &float16_int8_features;
    float16_int8_features.pNext = &variable_pointer_features;
    if (shader_bfloat16) {
      bfloat16_features.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
      bfloat16_features.shaderBFloat16Type = VK_TRUE;
    }
    if (storage_buffer_16bit_access) {
      storage_16bit_features.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
      storage_16bit_features.storageBuffer16BitAccess = VK_TRUE;
      storage_16bit_features.pNext =
          shader_bfloat16 ? &bfloat16_features : nullptr;
      variable_pointer_features.pNext = &storage_16bit_features;
    } else if (shader_bfloat16) {
      variable_pointer_features.pNext = &bfloat16_features;
    }

    uint32_t family_count = 0;
    Driver().get_physical_device_queue_family_properties(
        physical_device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    Driver().get_physical_device_queue_family_properties(
        physical_device, &family_count, families.data());
    auto family = std::find_if(
        families.begin(), families.end(), [](const VkQueueFamilyProperties& p) {
          return p.queueCount > 0 && (p.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        });
    if (family == families.end()) {
      return absl::NotFoundError("Vulkan device has no compute queue family");
    }
    queue_family_index = std::distance(families.begin(), family);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &timeline_features;
    VkPhysicalDeviceFeatures enabled_features = {};
    enabled_features.shaderInt16 = VK_TRUE;
    enabled_features.shaderInt64 = VK_TRUE;
    enabled_features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    device_info.pEnabledFeatures = &enabled_features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    std::vector<const char*> enabled_extensions = {
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME};
    if (shader_bfloat16) {
      enabled_extensions.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
    }
    device_info.enabledExtensionCount =
        static_cast<uint32_t>(enabled_extensions.size());
    device_info.ppEnabledExtensionNames = enabled_extensions.data();
    RETURN_IF_VK_ERROR(Driver().create_device(
        physical_device, &device_info, /*pAllocator=*/nullptr, &device));

#define LOAD_DEVICE_PROC(name)                                                \
  RETURN_IF_ERROR(LoadDeviceProc(Driver().get_device_proc_addr, device, #name, \
                                 &name))
    LOAD_DEVICE_PROC(vkDestroyDevice);
    LOAD_DEVICE_PROC(vkGetDeviceQueue);
    LOAD_DEVICE_PROC(vkCreateBuffer);
    LOAD_DEVICE_PROC(vkDestroyBuffer);
    LOAD_DEVICE_PROC(vkGetBufferMemoryRequirements);
    LOAD_DEVICE_PROC(vkAllocateMemory);
    LOAD_DEVICE_PROC(vkFreeMemory);
    LOAD_DEVICE_PROC(vkBindBufferMemory);
    LOAD_DEVICE_PROC(vkMapMemory);
    LOAD_DEVICE_PROC(vkUnmapMemory);
    LOAD_DEVICE_PROC(vkCreateCommandPool);
    LOAD_DEVICE_PROC(vkDestroyCommandPool);
    LOAD_DEVICE_PROC(vkAllocateCommandBuffers);
    LOAD_DEVICE_PROC(vkFreeCommandBuffers);
    LOAD_DEVICE_PROC(vkBeginCommandBuffer);
    LOAD_DEVICE_PROC(vkEndCommandBuffer);
    LOAD_DEVICE_PROC(vkQueueSubmit);
    LOAD_DEVICE_PROC(vkDeviceWaitIdle);
    LOAD_DEVICE_PROC(vkCreateSemaphore);
    LOAD_DEVICE_PROC(vkDestroySemaphore);
    LOAD_DEVICE_PROC(vkGetSemaphoreCounterValue);
    LOAD_DEVICE_PROC(vkWaitSemaphores);
    LOAD_DEVICE_PROC(vkCreateShaderModule);
    LOAD_DEVICE_PROC(vkDestroyShaderModule);
    LOAD_DEVICE_PROC(vkCreateDescriptorSetLayout);
    LOAD_DEVICE_PROC(vkDestroyDescriptorSetLayout);
    LOAD_DEVICE_PROC(vkCreatePipelineLayout);
    LOAD_DEVICE_PROC(vkDestroyPipelineLayout);
    LOAD_DEVICE_PROC(vkCreateComputePipelines);
    LOAD_DEVICE_PROC(vkDestroyPipeline);
    LOAD_DEVICE_PROC(vkCreateDescriptorPool);
    LOAD_DEVICE_PROC(vkDestroyDescriptorPool);
    LOAD_DEVICE_PROC(vkAllocateDescriptorSets);
    LOAD_DEVICE_PROC(vkUpdateDescriptorSets);
    LOAD_DEVICE_PROC(vkCmdBindPipeline);
    LOAD_DEVICE_PROC(vkCmdBindDescriptorSets);
    LOAD_DEVICE_PROC(vkCmdDispatch);
    LOAD_DEVICE_PROC(vkCmdPipelineBarrier);
#undef LOAD_DEVICE_PROC

    vkGetDeviceQueue(device, queue_family_index, 0, &queue);
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_index;
    RETURN_IF_VK_ERROR(vkCreateCommandPool(device, &pool_info, nullptr,
                                            &command_pool));

    VkSemaphoreTypeCreateInfo semaphore_type = {};
    semaphore_type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semaphore_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphore_type.initialValue = 0;
    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = &semaphore_type;
    RETURN_IF_VK_ERROR(
        vkCreateSemaphore(device, &semaphore_info, nullptr, &timeline));
    completion_thread = std::thread([this] { CompletionLoop(); });
    return absl::OkStatus();
  }

  ~Impl() {
    if (device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device);
    {
      std::lock_guard<std::mutex> lock(callback_mutex);
      stop_callbacks = true;
    }
    callback_cv.notify_all();
    if (completion_thread.joinable()) completion_thread.join();
    ReclaimCompleted();
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      for (const Submission& submission : submissions) {
        vkFreeCommandBuffers(device, command_pool, 1,
                             &submission.command_buffer);
        if (submission.descriptor_pool != VK_NULL_HANDLE) {
          vkDestroyDescriptorPool(device, submission.descriptor_pool, nullptr);
        }
      }
      submissions.clear();
    }
    for (auto& [address, allocation] : allocations) {
      if (allocation->mapped != nullptr) {
        vkUnmapMemory(device, allocation->memory);
      }
      vkDestroyBuffer(device, allocation->buffer, nullptr);
      vkFreeMemory(device, allocation->memory, nullptr);
    }
    allocations.clear();
    if (timeline != VK_NULL_HANDLE) {
      vkDestroySemaphore(device, timeline, nullptr);
    }
    if (command_pool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device, command_pool, nullptr);
    }
    vkDestroyDevice(device, nullptr);
  }

  absl::StatusOr<uint32_t> FindMemoryType(uint32_t memory_type_bits) const {
    std::optional<uint32_t> host_visible;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((memory_type_bits & (1u << i)) == 0) continue;
      VkMemoryPropertyFlags flags =
          memory_properties.memoryTypes[i].propertyFlags;
      constexpr VkMemoryPropertyFlags required =
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      if ((flags & required) != required) continue;
      if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) return i;
      if (!host_visible.has_value()) host_visible = i;
    }
    if (host_visible.has_value()) return *host_visible;
    return absl::NotFoundError(
        "Vulkan buffer has no HOST_VISIBLE | HOST_COHERENT compatible "
        "memory type");
  }

  absl::StatusOr<AllocationView> FindAllocation(const void* address,
                                                 uint64_t size) {
    uintptr_t value = reinterpret_cast<uintptr_t>(address);
    absl::MutexLock lock(&allocations_mutex);
    auto it = allocations.upper_bound(value);
    if (it == allocations.begin()) {
      return absl::NotFoundError("Unknown Vulkan device address");
    }
    --it;
    Allocation* allocation = it->second.get();
    uintptr_t end = it->first + allocation->size;
    if (value >= end || size > end - value) {
      return absl::OutOfRangeError("Vulkan device address range is invalid");
    }
    return AllocationView{allocation, value - it->first, size};
  }

  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkSemaphore timeline = VK_NULL_HANDLE;
  uint32_t queue_family_index = 0;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties = {};
  VkPhysicalDeviceMemoryProperties memory_properties = {};
  bool shader_bfloat16 = false;
  bool storage_buffer_16bit_access = false;
  absl::Mutex allocations_mutex;
  std::map<uintptr_t, std::unique_ptr<Allocation>> allocations;
  std::mutex queue_mutex;
  uint64_t next_timeline_value = 0;
  std::deque<Submission> submissions;
  std::mutex callback_mutex;
  std::condition_variable callback_cv;
  std::multimap<uint64_t, absl::AnyInvocable<absl::Status() &&>> callbacks;
  bool stop_callbacks = false;
  std::thread completion_thread;

  PFN_vkDestroyDevice vkDestroyDevice = nullptr;
  PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
  PFN_vkCreateBuffer vkCreateBuffer = nullptr;
  PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
  PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
  PFN_vkAllocateMemory vkAllocateMemory = nullptr;
  PFN_vkFreeMemory vkFreeMemory = nullptr;
  PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
  PFN_vkMapMemory vkMapMemory = nullptr;
  PFN_vkUnmapMemory vkUnmapMemory = nullptr;
  PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
  PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
  PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
  PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
  PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
  PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
  PFN_vkQueueSubmit vkQueueSubmit = nullptr;
  PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
  PFN_vkCreateSemaphore vkCreateSemaphore = nullptr;
  PFN_vkDestroySemaphore vkDestroySemaphore = nullptr;
  PFN_vkGetSemaphoreCounterValue vkGetSemaphoreCounterValue = nullptr;
  PFN_vkWaitSemaphores vkWaitSemaphores = nullptr;
  PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
  PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
  PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
  PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
  PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
  PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
  PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
  PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
  PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
  PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
  PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
  PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
  PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
  PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
  PFN_vkCmdDispatch vkCmdDispatch = nullptr;
  PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
};

class VulkanStream final : public StreamCommon {
 public:
  VulkanStream(VulkanExecutor* executor,
               std::optional<std::variant<StreamPriority, int>> priority)
      : StreamCommon(executor, priority), executor_(executor) {}
  ~VulkanStream() override {
    absl::Status status = BlockHostUntilDone();
    if (!status.ok()) LOG(ERROR) << "Failed to drain Vulkan stream: " << status;
    executor_->DeallocateStream(this);
  }

  absl::Status WaitFor(Stream* other) override {
    auto* vulkan_stream = dynamic_cast<VulkanStream*>(other);
    if (vulkan_stream == nullptr || vulkan_stream->executor_ != executor_) {
      return other->BlockHostUntilDone();
    }
    wait_value_ = std::max(wait_value_, vulkan_stream->DependencyValue());
    return absl::OkStatus();
  }
  absl::Status WaitFor(Event* event) override {
    if (event == nullptr) return absl::OkStatus();
    auto* vulkan_event = dynamic_cast<VulkanEvent*>(event);
    if (vulkan_event == nullptr ||
        (vulkan_event->executor() != nullptr &&
         vulkan_event->executor() != executor_)) {
      return event->Synchronize();
    }
    wait_value_ = std::max(wait_value_, vulkan_event->value());
    return absl::OkStatus();
  }
  absl::Status RecordEvent(Event* event) override {
    auto* vulkan_event = dynamic_cast<VulkanEvent*>(event);
    if (vulkan_event == nullptr) {
      return absl::InvalidArgumentError("Expected a VulkanEvent");
    }
    vulkan_event->SetSignal(executor_, DependencyValue());
    return absl::OkStatus();
  }
  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& gpu_src,
                      uint64_t size) override {
    RETURN_IF_ERROR(BlockHostUntilDone());
    return executor_->SynchronousMemcpy(host_dst, gpu_src, size);
  }
  absl::Status Memcpy(DeviceAddressBase* gpu_dst, const void* host_src,
                      uint64_t size) override {
    RETURN_IF_ERROR(BlockHostUntilDone());
    return executor_->SynchronousMemcpy(gpu_dst, host_src, size);
  }
  absl::Status Memcpy(DeviceAddressBase* gpu_dst,
                      const DeviceAddressBase& gpu_src,
                      uint64_t size) override {
    RETURN_IF_ERROR(BlockHostUntilDone());
    return executor_->MemcpyDeviceToDevice(gpu_dst, gpu_src, size);
  }
  absl::Status MemZero(DeviceAddressBase* location, uint64_t size) override {
    RETURN_IF_ERROR(BlockHostUntilDone());
    return executor_->Memset(location, 0, size);
  }
  absl::Status Memset32(DeviceAddressBase* location, uint32_t pattern,
                        uint64_t size) override {
    RETURN_IF_ERROR(BlockHostUntilDone());
    return executor_->Memset32(location, pattern, size);
  }
  absl::Status BlockHostUntilDone() override {
    RETURN_IF_ERROR(executor_->WaitTimeline(DependencyValue()));
    wait_value_ = 0;
    return absl::OkStatus();
  }
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override {
    executor_->ScheduleCallback(DependencyValue(), std::move(callback));
    return absl::OkStatus();
  }
  PlatformSpecificHandle platform_specific_handle() const override {
    return {};
  }

  uint64_t wait_value() const { return wait_value_; }
  uint64_t DependencyValue() const {
    return std::max(last_submitted_value_, wait_value_);
  }
  void Submitted(uint64_t value) {
    last_submitted_value_ = value;
    wait_value_ = 0;
  }

 private:
  VulkanExecutor* executor_;
  uint64_t last_submitted_value_ = 0;
  uint64_t wait_value_ = 0;
};

absl::Status VulkanKernel::Launch(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, Stream* stream,
    const KernelArgs& args) {
  if (cluster_dims.has_value() &&
      (cluster_dims->x != 1 || cluster_dims->y != 1 ||
       cluster_dims->z != 1)) {
    return absl::UnimplementedError(
        "Vulkan does not support XLA cluster launch dimensions");
  }
  auto* vulkan_stream = dynamic_cast<VulkanStream*>(stream);
  if (vulkan_stream == nullptr) {
    return absl::InvalidArgumentError("Vulkan kernel requires VulkanStream");
  }
  ASSIGN_OR_RETURN(
      uint64_t value,
      executor_->Launch(*this, thread_dims, block_dims, args,
                        vulkan_stream->wait_value()));
  vulkan_stream->Submitted(value);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<DeviceAddressBase>> GetDeviceArguments(
    const KernelArgs& args) {
  if (const auto* device_args =
          DynCast<KernelArgsDeviceAddressArray>(&args)) {
    return std::vector<DeviceAddressBase>(device_args->device_addr_args().begin(),
                                          device_args->device_addr_args().end());
  }
  if (const auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    if (const auto* packed_array =
            dynamic_cast<const KernelArgsPackedArray*>(packed);
        packed_array != nullptr &&
        packed_array->device_addresses().size() ==
            packed->argument_addresses().size()) {
      return std::vector<DeviceAddressBase>(
          packed_array->device_addresses().begin(),
          packed_array->device_addresses().end());
    }
    std::vector<DeviceAddressBase> addresses;
    addresses.reserve(packed->argument_addresses().size());
    for (const void* argument_address : packed->argument_addresses()) {
      void* address = nullptr;
      std::memcpy(&address, argument_address, sizeof(address));
      addresses.emplace_back(address, 0);
    }
    return addresses;
  }
  return absl::InvalidArgumentError("Unsupported Vulkan kernel argument type");
}

VulkanExecutor::VulkanExecutor(Platform* platform, int device_ordinal)
    : GpuExecutor(platform, device_ordinal), impl_(std::make_unique<Impl>()) {}

VulkanExecutor::~VulkanExecutor() = default;

absl::StatusOr<int> VulkanExecutor::GetDeviceCount() {
  RETURN_IF_ERROR(Driver().Initialize());
  return Driver().physical_devices().size();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
VulkanExecutor::CreateDeviceDescription(int device_ordinal) {
  RETURN_IF_ERROR(Driver().Initialize());
  if (device_ordinal < 0 ||
      device_ordinal >= Driver().physical_devices().size()) {
    return absl::OutOfRangeError("Vulkan device ordinal is out of range");
  }
  VkPhysicalDevice physical_device = Driver().physical_devices()[device_ordinal];
  VkPhysicalDeviceProperties properties = {};
  VkPhysicalDeviceMemoryProperties memory_properties = {};
  Driver().get_physical_device_memory_properties(physical_device,
                                                 &memory_properties);
  TF_ASSIGN_OR_RETURN(VulkanShaderFeatures shader_features,
                      Driver().GetShaderFeatures(physical_device));
  TF_ASSIGN_OR_RETURN(
      VulkanDevicePerformanceProperties performance,
      QueryDevicePerformanceProperties(physical_device, &properties));
  RETURN_IF_ERROR(
      ValidateDevicePerformanceProperties(properties, performance));

  auto description = std::make_unique<DeviceDescription>();
  description->set_name(properties.deviceName);
  description->set_model_str(properties.deviceName);
  description->set_device_vendor(
      absl::StrFormat("Vulkan vendor 0x%04x", properties.vendorID));
  description->set_platform_version(absl::StrFormat(
      "Vulkan %d.%d.%d", VK_API_VERSION_MAJOR(properties.apiVersion),
      VK_API_VERSION_MINOR(properties.apiVersion),
      VK_API_VERSION_PATCH(properties.apiVersion)));
  description->set_vulkan_compute_capability(
      VK_API_VERSION_MAJOR(properties.apiVersion),
      VK_API_VERSION_MINOR(properties.apiVersion),
      shader_features.shader_bfloat16,
      shader_features.storage_buffer_16bit_access);
  description->set_thread_dim_limit(ThreadDim(
      properties.limits.maxComputeWorkGroupSize[0],
      properties.limits.maxComputeWorkGroupSize[1],
      properties.limits.maxComputeWorkGroupSize[2]));
  description->set_block_dim_limit(BlockDim(
      properties.limits.maxComputeWorkGroupCount[0],
      properties.limits.maxComputeWorkGroupCount[1],
      properties.limits.maxComputeWorkGroupCount[2]));
  description->set_threads_per_block_limit(
      properties.limits.maxComputeWorkGroupInvocations);
  description->set_threads_per_warp(*performance.threads_per_warp);
  description->set_threads_per_core_limit(*performance.threads_per_core);
  description->set_core_count(*performance.core_count);
  description->set_fpus_per_core(*performance.fpus_per_core);
  description->set_clock_rate_ghz(*performance.clock_rate_ghz);
  description->set_memory_bandwidth(*performance.memory_bandwidth);
  description->set_l2_cache_size(*performance.l2_cache_size);
  if (!performance.pci_bus_id.empty()) {
    description->set_pci_bus_id(performance.pci_bus_id);
  }
  description->set_device_address_bits(sizeof(void*) * 8);
  description->set_shared_memory_per_block(
      properties.limits.maxComputeSharedMemorySize);
  int64_t device_memory_size = 0;
  for (uint32_t i = 0; i < memory_properties.memoryHeapCount; ++i) {
    if ((memory_properties.memoryHeaps[i].flags &
         VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
      device_memory_size += memory_properties.memoryHeaps[i].size;
    }
  }
  description->set_device_memory_size(device_memory_size);
  return description;
}

absl::Status VulkanExecutor::Init() {
  return impl_->Initialize(device_ordinal());
}

absl::StatusOr<std::unique_ptr<Stream>> VulkanExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  // PJRT uses distinct priorities for internal compute, transfer, and callback
  // streams. Vulkan v1 exposes one queue per executor, so preserve the logical
  // priority on StreamCommon while all streams share that queue.
  return std::make_unique<VulkanStream>(this, priority);
}

absl::StatusOr<std::unique_ptr<Event>> VulkanExecutor::CreateEvent() {
  return std::make_unique<VulkanEvent>();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
VulkanExecutor::CreateDeviceDescription() const {
  return CreateDeviceDescription(device_ordinal());
}

absl::StatusOr<std::unique_ptr<Kernel>> VulkanExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  std::optional<VulkanSpirvInMemory> spirv = spec.vulkan_spirv_in_memory();
  if (!spirv.has_value()) {
    return absl::InvalidArgumentError(
        "VulkanExecutor requires a VulkanSpirvInMemory kernel");
  }
  if (spirv->target_environment != VulkanSpirvProto::VULKAN_1_2) {
    return absl::InvalidArgumentError("Unsupported Vulkan target environment");
  }
  if (spirv->spirv_bytes.size() < sizeof(uint32_t) ||
      spirv->spirv_bytes.size() % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("Invalid Vulkan SPIR-V byte size");
  }

  auto kernel = std::make_unique<VulkanKernel>(
      this, spec.arity(),
      std::vector<VulkanDescriptorBinding>(spirv->descriptor_bindings.begin(),
                                           spirv->descriptor_bindings.end()));
  kernel->set_name(spec.kernel_name());

  VkShaderModuleCreateInfo shader_info = {};
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.codeSize = spirv->spirv_bytes.size();
  shader_info.pCode =
      reinterpret_cast<const uint32_t*>(spirv->spirv_bytes.data());
  LOG(INFO) << "Vulkan kernel " << kernel->name()
            << ": creating shader module from " << shader_info.codeSize
            << " SPIR-V bytes";
  RETURN_IF_VK_ERROR(impl_->vkCreateShaderModule(
      impl_->device, &shader_info, nullptr, &kernel->shader_module_));
  LOG(INFO) << "Vulkan kernel " << kernel->name()
            << ": shader module created";

  std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
  layout_bindings.reserve(kernel->bindings_.size());
  for (const VulkanDescriptorBinding& binding : kernel->bindings_) {
    if (binding.descriptor_set != 0) {
      return absl::UnimplementedError(
          "Initial Vulkan StreamExecutor supports descriptor set 0 only");
    }
    VkDescriptorSetLayoutBinding layout_binding = {};
    layout_binding.binding = binding.binding;
    layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layout_binding.descriptorCount = 1;
    layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layout_bindings.push_back(layout_binding);
  }
  VkDescriptorSetLayoutCreateInfo layout_info = {};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = layout_bindings.size();
  layout_info.pBindings = layout_bindings.data();
  RETURN_IF_VK_ERROR(impl_->vkCreateDescriptorSetLayout(
      impl_->device, &layout_info, nullptr, &kernel->descriptor_set_layout_));

  VkPipelineLayoutCreateInfo pipeline_layout_info = {};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &kernel->descriptor_set_layout_;
  RETURN_IF_VK_ERROR(impl_->vkCreatePipelineLayout(
      impl_->device, &pipeline_layout_info, nullptr, &kernel->pipeline_layout_));

  VkComputePipelineCreateInfo pipeline_info = {};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = kernel->shader_module_;
  pipeline_info.stage.pName = kernel->name().data();
  pipeline_info.layout = kernel->pipeline_layout_;
  LOG(INFO) << "Vulkan kernel " << kernel->name()
            << ": creating compute pipeline";
  RETURN_IF_VK_ERROR(impl_->vkCreateComputePipelines(
      impl_->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
      &kernel->pipeline_));
  LOG(INFO) << "Vulkan kernel " << kernel->name()
            << ": compute pipeline created";

  if (std::holds_alternative<KernelLoaderSpec::KernelArgsPackingFunc>(
          spec.kernel_args_packing())) {
    kernel->set_args_packing(
        std::get<KernelLoaderSpec::KernelArgsPackingFunc>(
            spec.kernel_args_packing()));
  }
  return std::move(kernel);
}

void VulkanExecutor::UnloadKernel(const Kernel* kernel) {
  auto* vulkan_kernel = const_cast<VulkanKernel*>(
      dynamic_cast<const VulkanKernel*>(kernel));
  if (vulkan_kernel == nullptr || impl_->device == VK_NULL_HANDLE) return;
  if (vulkan_kernel->pipeline_ != VK_NULL_HANDLE) {
    impl_->vkDestroyPipeline(impl_->device, vulkan_kernel->pipeline_, nullptr);
    vulkan_kernel->pipeline_ = VK_NULL_HANDLE;
  }
  if (vulkan_kernel->pipeline_layout_ != VK_NULL_HANDLE) {
    impl_->vkDestroyPipelineLayout(impl_->device,
                                   vulkan_kernel->pipeline_layout_, nullptr);
    vulkan_kernel->pipeline_layout_ = VK_NULL_HANDLE;
  }
  if (vulkan_kernel->descriptor_set_layout_ != VK_NULL_HANDLE) {
    impl_->vkDestroyDescriptorSetLayout(
        impl_->device, vulkan_kernel->descriptor_set_layout_, nullptr);
    vulkan_kernel->descriptor_set_layout_ = VK_NULL_HANDLE;
  }
  if (vulkan_kernel->shader_module_ != VK_NULL_HANDLE) {
    impl_->vkDestroyShaderModule(impl_->device, vulkan_kernel->shader_module_,
                                 nullptr);
    vulkan_kernel->shader_module_ = VK_NULL_HANDLE;
  }
}

DeviceAddressBase VulkanExecutor::Allocate(uint64_t size,
                                           int64_t memory_space) {
  if (static_cast<MemorySpace>(memory_space) == MemorySpace::kCollective) {
    LOG(ERROR) << "Vulkan collective memory is not implemented";
    return DeviceAddressBase();
  }
  if (size == 0) return DeviceAddressBase();
  auto allocation = std::make_unique<Impl::Allocation>();
  allocation->size = size;

  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkResult result = impl_->vkCreateBuffer(impl_->device, &buffer_info, nullptr,
                                          &allocation->buffer);
  if (result != VK_SUCCESS) {
    LOG(ERROR) << VulkanError("vkCreateBuffer", result);
    return DeviceAddressBase();
  }
  VkMemoryRequirements requirements = {};
  impl_->vkGetBufferMemoryRequirements(impl_->device, allocation->buffer,
                                       &requirements);
  absl::StatusOr<uint32_t> memory_type =
      impl_->FindMemoryType(requirements.memoryTypeBits);
  if (!memory_type.ok()) {
    LOG(ERROR) << memory_type.status();
    impl_->vkDestroyBuffer(impl_->device, allocation->buffer, nullptr);
    return DeviceAddressBase();
  }
  VkMemoryAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = *memory_type;
  result = impl_->vkAllocateMemory(impl_->device, &allocate_info, nullptr,
                                   &allocation->memory);
  if (result != VK_SUCCESS) {
    LOG(ERROR) << VulkanError("vkAllocateMemory", result);
    impl_->vkDestroyBuffer(impl_->device, allocation->buffer, nullptr);
    return DeviceAddressBase();
  }
  result = impl_->vkBindBufferMemory(impl_->device, allocation->buffer,
                                     allocation->memory, 0);
  if (result == VK_SUCCESS) {
    result = impl_->vkMapMemory(impl_->device, allocation->memory, 0, size, 0,
                                &allocation->mapped);
  }
  if (result != VK_SUCCESS) {
    LOG(ERROR) << VulkanError("Vulkan buffer bind/map", result);
    impl_->vkDestroyBuffer(impl_->device, allocation->buffer, nullptr);
    impl_->vkFreeMemory(impl_->device, allocation->memory, nullptr);
    return DeviceAddressBase();
  }
  void* mapped = allocation->mapped;
  {
    absl::MutexLock lock(&impl_->allocations_mutex);
    impl_->allocations.emplace(reinterpret_cast<uintptr_t>(mapped),
                               std::move(allocation));
  }
  return DeviceAddressBase(mapped, size);
}

void VulkanExecutor::Deallocate(DeviceAddressBase* mem) {
  if (mem == nullptr || mem->is_null()) return;
  std::unique_ptr<Impl::Allocation> allocation;
  {
    absl::MutexLock lock(&impl_->allocations_mutex);
    auto it = impl_->allocations.find(reinterpret_cast<uintptr_t>(mem->opaque()));
    if (it == impl_->allocations.end()) {
      LOG(ERROR) << "Attempted to free unknown or sliced Vulkan allocation "
                 << mem->opaque();
      return;
    }
    allocation = std::move(it->second);
    impl_->allocations.erase(it);
  }
  impl_->vkUnmapMemory(impl_->device, allocation->memory);
  impl_->vkDestroyBuffer(impl_->device, allocation->buffer, nullptr);
  impl_->vkFreeMemory(impl_->device, allocation->memory, nullptr);
  *mem = DeviceAddressBase();
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
VulkanExecutor::CreateMemoryAllocator(MemorySpace memory_space) {
  if (memory_space == MemorySpace::kCollective) {
    return absl::UnimplementedError(
        "Vulkan collective memory space is not implemented");
  }
  return std::make_unique<GenericMemoryAllocator>(
      [this](uint64_t size)
          -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        DeviceAddressBase address = Allocate(size, 0);
        if (address.is_null() && size != 0) {
          return absl::ResourceExhaustedError(absl::StrCat(
              "Failed to allocate ", size, " bytes of Vulkan memory"));
        }
        return std::make_unique<GenericMemoryAllocation>(
            address.opaque(), size, [this](void* pointer, uint64_t bytes) {
              DeviceAddressBase address(pointer, bytes);
              Deallocate(&address);
            });
      });
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
VulkanExecutor::HostMemoryAllocate(uint64_t size) {
  void* memory = std::malloc(size);
  if (memory == nullptr && size != 0) {
    return absl::ResourceExhaustedError("Failed to allocate host memory");
  }
  return std::make_unique<GenericMemoryAllocation>(
      memory, size, [](void* pointer, uint64_t) { std::free(pointer); });
}

bool VulkanExecutor::SynchronizeAllActivity() {
  uint64_t value = 0;
  {
    std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    value = impl_->next_timeline_value;
  }
  absl::Status status = impl_->WaitForTimeline(value);
  if (status.ok()) impl_->ReclaimCompleted();
  return status.ok();
}

absl::StatusOr<uint64_t> VulkanExecutor::TimelineValue() const {
  return impl_->CurrentTimelineValue();
}

absl::Status VulkanExecutor::WaitTimeline(uint64_t value) const {
  return impl_->WaitForTimeline(value);
}

void VulkanExecutor::ScheduleCallback(
    uint64_t value,
    absl::AnyInvocable<absl::Status() &&> callback) {
  impl_->ScheduleCallback(value, std::move(callback));
}

absl::Status VulkanExecutor::SynchronousMemcpy(DeviceAddressBase* device_dst,
                                               const void* host_src,
                                               uint64_t size) {
  ASSIGN_OR_RETURN(Impl::AllocationView view,
                   impl_->FindAllocation(device_dst->opaque(), size));
  std::memcpy(static_cast<std::byte*>(view.allocation->mapped) + view.offset,
              host_src, size);
  VLOG(2) << "Copied " << size << " host bytes to Vulkan buffer "
          << view.allocation->buffer << " at offset " << view.offset;
  return absl::OkStatus();
}

absl::Status VulkanExecutor::SynchronousMemcpy(
    void* host_dst, const DeviceAddressBase& device_src, uint64_t size) {
  ASSIGN_OR_RETURN(Impl::AllocationView view,
                   impl_->FindAllocation(device_src.opaque(), size));
  std::memcpy(host_dst,
              static_cast<std::byte*>(view.allocation->mapped) + view.offset,
              size);
  VLOG(2) << "Copied " << size << " bytes from Vulkan buffer "
          << view.allocation->buffer << " at offset " << view.offset
          << " after synchronization";
  return absl::OkStatus();
}

absl::Status VulkanExecutor::MemcpyDeviceToDevice(
    DeviceAddressBase* device_dst, const DeviceAddressBase& device_src,
    uint64_t size) {
  ASSIGN_OR_RETURN(Impl::AllocationView dst,
                   impl_->FindAllocation(device_dst->opaque(), size));
  ASSIGN_OR_RETURN(Impl::AllocationView src,
                   impl_->FindAllocation(device_src.opaque(), size));
  std::memmove(static_cast<std::byte*>(dst.allocation->mapped) + dst.offset,
               static_cast<std::byte*>(src.allocation->mapped) + src.offset,
               size);
  return absl::OkStatus();
}

absl::Status VulkanExecutor::Memset(DeviceAddressBase* location, uint8_t value,
                                    uint64_t size) {
  ASSIGN_OR_RETURN(Impl::AllocationView view,
                   impl_->FindAllocation(location->opaque(), size));
  std::memset(static_cast<std::byte*>(view.allocation->mapped) + view.offset,
              value, size);
  return absl::OkStatus();
}

absl::Status VulkanExecutor::Memset32(DeviceAddressBase* location,
                                      uint32_t value, uint64_t size) {
  if (size % sizeof(uint32_t) != 0 ||
      reinterpret_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) != 0) {
    return absl::InvalidArgumentError(
        "Vulkan Memset32 requires 4-byte size and alignment");
  }
  ASSIGN_OR_RETURN(Impl::AllocationView view,
                   impl_->FindAllocation(location->opaque(), size));
  auto* begin = reinterpret_cast<uint32_t*>(
      static_cast<std::byte*>(view.allocation->mapped) + view.offset);
  std::fill(begin, begin + size / sizeof(uint32_t), value);
  return absl::OkStatus();
}

void VulkanExecutor::DeallocateStream(Stream* stream) {}

absl::Status VulkanExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  return absl::UnimplementedError("Vulkan peer access is not implemented");
}

bool VulkanExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  return false;
}

bool VulkanExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_properties = {};
  budget_properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
  VkPhysicalDeviceMemoryProperties2 memory_properties = {};
  memory_properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
  memory_properties.pNext = &budget_properties;
  Driver().get_physical_device_memory_properties2(
      impl_->physical_device, &memory_properties);

  constexpr VkMemoryPropertyFlags required =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t compatible_heaps = 0;
  uint32_t device_local_heaps = 0;
  for (uint32_t i = 0;
       i < memory_properties.memoryProperties.memoryTypeCount; ++i) {
    const VkMemoryType& type =
        memory_properties.memoryProperties.memoryTypes[i];
    if ((type.propertyFlags & required) == required) {
      compatible_heaps |= 1u << type.heapIndex;
      if ((type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
        device_local_heaps |= 1u << type.heapIndex;
      }
    }
  }
  if (device_local_heaps != 0) compatible_heaps = device_local_heaps;

  uint64_t budget = 0;
  uint64_t available = 0;
  for (uint32_t i = 0;
       i < memory_properties.memoryProperties.memoryHeapCount; ++i) {
    if ((compatible_heaps & (1u << i)) == 0) continue;
    const uint64_t heap_budget = budget_properties.heapBudget[i];
    const uint64_t heap_usage = budget_properties.heapUsage[i];
    const uint64_t heap_available =
        heap_usage < heap_budget ? heap_budget - heap_usage : 0;
    budget = heap_budget > std::numeric_limits<uint64_t>::max() - budget
                 ? std::numeric_limits<uint64_t>::max()
                 : budget + heap_budget;
    available =
        heap_available > std::numeric_limits<uint64_t>::max() - available
            ? std::numeric_limits<uint64_t>::max()
            : available + heap_available;
    VLOG(1) << "Vulkan memory heap " << i
            << ": size="
            << memory_properties.memoryProperties.memoryHeaps[i].size
            << ", budget=" << heap_budget << ", usage=" << heap_usage
            << ", available=" << heap_available;
  }

  const int64_t configured_limit = GetMemoryLimitBytes();
  if (configured_limit > 0) {
    budget = std::min(budget, static_cast<uint64_t>(configured_limit));
    available = std::min(available, budget);
  }
  constexpr uint64_t kMaxInt64 =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  const int64_t reported_total =
      static_cast<int64_t>(std::min(budget, kMaxInt64));
  const int64_t reported_free =
      static_cast<int64_t>(std::min(available, kMaxInt64));
  VLOG(1) << "Vulkan memory usage: total budget=" << reported_total
          << ", free budget=" << reported_free;
  if (total != nullptr) *total = reported_total;
  if (free != nullptr) *free = reported_free;
  return true;
}

absl::StatusOr<uint64_t> VulkanExecutor::Launch(
    const VulkanKernel& kernel, const ThreadDim& thread_dims,
    const BlockDim& block_dims, const KernelArgs& args,
    uint64_t wait_value) {
  ASSIGN_OR_RETURN(std::vector<DeviceAddressBase> arguments,
                   GetDeviceArguments(args));
  if (arguments.size() < kernel.Arity()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Vulkan kernel %s expected %d arguments but got %d", kernel.name(),
        kernel.Arity(), arguments.size()));
  }

  std::vector<VkDescriptorBufferInfo> buffer_infos;
  std::vector<VkWriteDescriptorSet> writes;
  buffer_infos.reserve(kernel.bindings_.size());
  writes.reserve(kernel.bindings_.size());
  for (const VulkanDescriptorBinding& binding : kernel.bindings_) {
    if (binding.argument_index >= arguments.size()) {
      return absl::InvalidArgumentError("Vulkan descriptor argument is missing");
    }
    const DeviceAddressBase& argument = arguments[binding.argument_index];
    ASSIGN_OR_RETURN(Impl::AllocationView view,
                     impl_->FindAllocation(argument.opaque(), argument.size()));
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = view.allocation->buffer;
    buffer_info.offset = view.offset;
    buffer_info.range = argument.size() == 0 ? view.allocation->size - view.offset
                                             : argument.size();
    if (buffer_info.offset %
            impl_->properties.limits.minStorageBufferOffsetAlignment !=
        0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Vulkan kernel %s argument %d offset %d is not aligned to %d",
          kernel.name(), binding.argument_index, buffer_info.offset,
          impl_->properties.limits.minStorageBufferOffsetAlignment));
    }
    if (buffer_info.range == 0 ||
        buffer_info.range > impl_->properties.limits.maxStorageBufferRange) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Vulkan kernel %s argument %d range %d exceeds device limit %d",
          kernel.name(), binding.argument_index, buffer_info.range,
          impl_->properties.limits.maxStorageBufferRange));
    }
    VLOG(1) << "Vulkan kernel " << kernel.name() << " descriptor set="
            << binding.descriptor_set << ", binding=" << binding.binding
            << ", argument=" << binding.argument_index
            << ", slice=" << binding.slice_index
            << ", read_only=" << binding.read_only
            << ", buffer=" << buffer_info.buffer
            << ", offset=" << buffer_info.offset
            << ", range=" << buffer_info.range;
    buffer_infos.push_back(buffer_info);
  }

  VkDescriptorPoolSize pool_size = {};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = kernel.bindings_.size();
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;

  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  RETURN_IF_VK_ERROR(impl_->vkCreateDescriptorPool(
      impl_->device, &pool_info, nullptr, &descriptor_pool));
  auto destroy_pool = [&] {
    impl_->vkDestroyDescriptorPool(impl_->device, descriptor_pool, nullptr);
  };

  VkDescriptorSetAllocateInfo set_info = {};
  set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  set_info.descriptorPool = descriptor_pool;
  set_info.descriptorSetCount = 1;
  set_info.pSetLayouts = &kernel.descriptor_set_layout_;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VkResult result =
      impl_->vkAllocateDescriptorSets(impl_->device, &set_info, &descriptor_set);
  if (result != VK_SUCCESS) {
    destroy_pool();
    return VulkanError("vkAllocateDescriptorSets", result);
  }

  for (size_t i = 0; i < kernel.bindings_.size(); ++i) {
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set;
    write.dstBinding = kernel.bindings_[i].binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_infos[i];
    writes.push_back(write);
  }
  impl_->vkUpdateDescriptorSets(impl_->device, writes.size(), writes.data(), 0,
                                nullptr);

  VkCommandBufferAllocateInfo command_info = {};
  command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_info.commandPool = impl_->command_pool;
  command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_info.commandBufferCount = 1;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  {
    std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    result = impl_->vkAllocateCommandBuffers(impl_->device, &command_info,
                                             &command_buffer);
  }
  if (result != VK_SUCCESS) {
    destroy_pool();
    return VulkanError("vkAllocateCommandBuffers", result);
  }
  auto cleanup = [&] {
    {
      std::lock_guard<std::mutex> lock(impl_->queue_mutex);
      impl_->vkFreeCommandBuffers(impl_->device, impl_->command_pool, 1,
                                  &command_buffer);
    }
    destroy_pool();
  };

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  result = impl_->vkBeginCommandBuffer(command_buffer, &begin_info);
  if (result != VK_SUCCESS) {
    cleanup();
    return VulkanError("vkBeginCommandBuffer", result);
  }
  VkMemoryBarrier before = {};
  before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  before.srcAccessMask =
      VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  impl_->vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0,
      nullptr);
  impl_->vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           kernel.pipeline_);
  impl_->vkCmdBindDescriptorSets(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline_layout_,
      0, 1, &descriptor_set, 0, nullptr);
  VLOG(1) << "Dispatching Vulkan kernel " << kernel.name() << " with groups=("
          << block_dims.x << ", " << block_dims.y << ", " << block_dims.z
          << ") and local size=(" << thread_dims.x << ", " << thread_dims.y
          << ", " << thread_dims.z << ")";
  impl_->vkCmdDispatch(command_buffer, block_dims.x, block_dims.y, block_dims.z);
  VkMemoryBarrier after = {};
  after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  impl_->vkCmdPipelineBarrier(command_buffer,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0,
                              nullptr, 0, nullptr);
  result = impl_->vkEndCommandBuffer(command_buffer);
  if (result != VK_SUCCESS) {
    cleanup();
    return VulkanError("vkEndCommandBuffer", result);
  }
  absl::StatusOr<uint64_t> submitted =
      impl_->Submit(command_buffer, descriptor_pool, wait_value);
  if (!submitted.ok()) {
    cleanup();
    return submitted.status();
  }
  impl_->ReclaimCompleted();
  return *submitted;
}

}  // namespace stream_executor::vulkan
