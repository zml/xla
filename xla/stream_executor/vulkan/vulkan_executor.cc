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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
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
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::vulkan {
namespace {

absl::Status VulkanError(absl::string_view operation, VkResult result) {
  return absl::InternalError(
      absl::StrFormat("%s failed with VkResult %d", operation, result));
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

  PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
  PFN_vkGetPhysicalDeviceProperties get_physical_device_properties = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties get_physical_device_memory_properties =
      nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties
      get_physical_device_queue_family_properties = nullptr;
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
    RETURN_IF_VK_ERROR(
        create_instance(&create_info, /*pAllocator=*/nullptr, &instance_));

    PFN_vkEnumeratePhysicalDevices enumerate_physical_devices = nullptr;
    RETURN_IF_ERROR(LoadInstanceProc(get_instance_proc_addr, instance_,
                                     "vkEnumeratePhysicalDevices",
                                     &enumerate_physical_devices));
    RETURN_IF_ERROR(LoadInstanceProc(get_instance_proc_addr, instance_,
                                     "vkGetPhysicalDeviceProperties",
                                     &get_physical_device_properties));
    RETURN_IF_ERROR(LoadInstanceProc(
        get_instance_proc_addr, instance_,
        "vkGetPhysicalDeviceMemoryProperties",
        &get_physical_device_memory_properties));
    RETURN_IF_ERROR(LoadInstanceProc(
        get_instance_proc_addr, instance_,
        "vkGetPhysicalDeviceQueueFamilyProperties",
        &get_physical_device_queue_family_properties));
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
    return absl::OkStatus();
  }

  std::once_flag once_;
  absl::Status status_;
  void* library_ = nullptr;
  VkInstance instance_ = VK_NULL_HANDLE;
  std::vector<VkPhysicalDevice> physical_devices_;
};

VulkanDriver& Driver() {
  static VulkanDriver* driver = new VulkanDriver;
  return *driver;
}

class VulkanActivateContext final : public ActivateContext {};

class VulkanEvent final : public Event {
 public:
  Status PollForStatus() override {
    return complete_.load(std::memory_order_acquire) ? Status::kComplete
                                                     : Status::kPending;
  }
  absl::Status Synchronize() override {
    return complete_.load(std::memory_order_acquire)
               ? absl::OkStatus()
               : absl::FailedPreconditionError("Vulkan event is not recorded");
  }
  void Record() { complete_.store(true, std::memory_order_release); }

 private:
  std::atomic<bool> complete_{false};
};

}  // namespace

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
                      Stream* stream, const KernelArgs& args) override {
    if (cluster_dims.has_value() &&
        (cluster_dims->x != 1 || cluster_dims->y != 1 ||
         cluster_dims->z != 1)) {
      return absl::UnimplementedError(
          "Vulkan does not support XLA cluster launch dimensions");
    }
    return executor_->Launch(*this, thread_dims, block_dims, args);
  }

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
    bool coherent = false;
  };

  struct AllocationView {
    Allocation* allocation;
    VkDeviceSize offset;
    VkDeviceSize size;
  };

  absl::Status Initialize(int ordinal) {
    RETURN_IF_ERROR(Driver().Initialize());
    if (ordinal < 0 || ordinal >= Driver().physical_devices().size()) {
      return absl::OutOfRangeError(
          absl::StrFormat("Vulkan device ordinal %d is out of range", ordinal));
    }
    physical_device = Driver().physical_devices()[ordinal];
    Driver().get_physical_device_properties(physical_device, &properties);
    Driver().get_physical_device_memory_properties(physical_device,
                                                   &memory_properties);

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
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
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
    LOAD_DEVICE_PROC(vkFlushMappedMemoryRanges);
    LOAD_DEVICE_PROC(vkInvalidateMappedMemoryRanges);
    LOAD_DEVICE_PROC(vkCreateCommandPool);
    LOAD_DEVICE_PROC(vkDestroyCommandPool);
    LOAD_DEVICE_PROC(vkAllocateCommandBuffers);
    LOAD_DEVICE_PROC(vkFreeCommandBuffers);
    LOAD_DEVICE_PROC(vkBeginCommandBuffer);
    LOAD_DEVICE_PROC(vkEndCommandBuffer);
    LOAD_DEVICE_PROC(vkQueueSubmit);
    LOAD_DEVICE_PROC(vkQueueWaitIdle);
    LOAD_DEVICE_PROC(vkDeviceWaitIdle);
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
    return absl::OkStatus();
  }

  ~Impl() {
    if (device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device);
    for (auto& [address, allocation] : allocations) {
      if (allocation->mapped != nullptr) {
        vkUnmapMemory(device, allocation->memory);
      }
      vkDestroyBuffer(device, allocation->buffer, nullptr);
      vkFreeMemory(device, allocation->memory, nullptr);
    }
    allocations.clear();
    if (command_pool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device, command_pool, nullptr);
    }
    vkDestroyDevice(device, nullptr);
  }

  absl::StatusOr<uint32_t> FindMemoryType(uint32_t memory_type_bits) const {
    std::optional<uint32_t> host_visible;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((memory_type_bits & (1u << i)) == 0) continue;
      VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
      if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
      if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) return i;
      host_visible = i;
    }
    if (host_visible.has_value()) return *host_visible;
    return absl::NotFoundError(
        "Vulkan buffer has no host-visible compatible memory type");
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
    if (value > end || size > end - value) {
      return absl::OutOfRangeError("Vulkan device address range is invalid");
    }
    return AllocationView{allocation, value - it->first, size};
  }

  absl::Status Flush(const AllocationView& view) {
    if (view.allocation->coherent) return absl::OkStatus();
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = view.allocation->memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    RETURN_IF_VK_ERROR(vkFlushMappedMemoryRanges(device, 1, &range));
    return absl::OkStatus();
  }

  absl::Status Invalidate(const AllocationView& view) {
    if (view.allocation->coherent) return absl::OkStatus();
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = view.allocation->memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    RETURN_IF_VK_ERROR(vkInvalidateMappedMemoryRanges(device, 1, &range));
    return absl::OkStatus();
  }

  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family_index = 0;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties = {};
  VkPhysicalDeviceMemoryProperties memory_properties = {};
  absl::Mutex allocations_mutex;
  std::map<uintptr_t, std::unique_ptr<Allocation>> allocations;
  std::mutex queue_mutex;

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
  PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges = nullptr;
  PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges = nullptr;
  PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
  PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
  PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
  PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
  PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
  PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
  PFN_vkQueueSubmit vkQueueSubmit = nullptr;
  PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
  PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
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

namespace {

class VulkanStream final : public StreamCommon {
 public:
  VulkanStream(VulkanExecutor* executor,
               std::optional<std::variant<StreamPriority, int>> priority)
      : StreamCommon(executor, priority), executor_(executor) {}
  ~VulkanStream() override { executor_->DeallocateStream(this); }

  absl::Status WaitFor(Stream* other) override {
    return other->BlockHostUntilDone();
  }
  absl::Status WaitFor(Event* event) override { return event->Synchronize(); }
  absl::Status RecordEvent(Event* event) override {
    static_cast<VulkanEvent*>(event)->Record();
    return absl::OkStatus();
  }
  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& gpu_src,
                      uint64_t size) override {
    return executor_->SynchronousMemcpy(host_dst, gpu_src, size);
  }
  absl::Status Memcpy(DeviceAddressBase* gpu_dst, const void* host_src,
                      uint64_t size) override {
    return executor_->SynchronousMemcpy(gpu_dst, host_src, size);
  }
  absl::Status Memcpy(DeviceAddressBase* gpu_dst,
                      const DeviceAddressBase& gpu_src,
                      uint64_t size) override {
    return executor_->MemcpyDeviceToDevice(gpu_dst, gpu_src, size);
  }
  absl::Status MemZero(DeviceAddressBase* location, uint64_t size) override {
    return executor_->Memset(location, 0, size);
  }
  absl::Status Memset32(DeviceAddressBase* location, uint32_t pattern,
                        uint64_t size) override {
    return executor_->Memset32(location, pattern, size);
  }
  absl::Status BlockHostUntilDone() override {
    return executor_->SynchronizeAllActivity()
               ? absl::OkStatus()
               : absl::InternalError("Vulkan queue synchronization failed");
  }
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override {
    RETURN_IF_ERROR(BlockHostUntilDone());
    return std::move(callback)();
  }
  PlatformSpecificHandle platform_specific_handle() const override {
    return {};
  }

 private:
  VulkanExecutor* executor_;
};

absl::StatusOr<std::vector<DeviceAddressBase>> GetDeviceArguments(
    const KernelArgs& args) {
  if (const auto* device_args =
          DynCast<KernelArgsDeviceAddressArray>(&args)) {
    return std::vector<DeviceAddressBase>(device_args->device_addr_args().begin(),
                                          device_args->device_addr_args().end());
  }
  if (const auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
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

}  // namespace

VulkanExecutor::VulkanExecutor(Platform* platform, int device_ordinal)
    : platform_(platform),
      device_ordinal_(device_ordinal),
      impl_(std::make_unique<Impl>()) {}

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
  Driver().get_physical_device_properties(physical_device, &properties);
  Driver().get_physical_device_memory_properties(physical_device,
                                                 &memory_properties);

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
      VK_API_VERSION_MINOR(properties.apiVersion));
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
  description->set_threads_per_warp(32);
  description->set_core_count(1);
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

std::unique_ptr<ActivateContext> VulkanExecutor::Activate() {
  return std::make_unique<VulkanActivateContext>();
}

const Platform* VulkanExecutor::GetPlatform() const { return platform_; }

absl::Status VulkanExecutor::Init() {
  RETURN_IF_ERROR(impl_->Initialize(device_ordinal_));
  ASSIGN_OR_RETURN(device_description_,
                   CreateDeviceDescription(device_ordinal_));
  return absl::OkStatus();
}

int VulkanExecutor::device_ordinal() const { return device_ordinal_; }

absl::StatusOr<std::unique_ptr<Stream>> VulkanExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  if (priority.has_value() && std::holds_alternative<StreamPriority>(*priority) &&
      std::get<StreamPriority>(*priority) != StreamPriority::Default) {
    return absl::UnimplementedError(
        "Vulkan StreamExecutor supports only default stream priority");
  }
  return std::make_unique<VulkanStream>(this, priority);
}

absl::StatusOr<std::unique_ptr<Event>> VulkanExecutor::CreateEvent() {
  return std::make_unique<VulkanEvent>();
}

const DeviceDescription& VulkanExecutor::GetDeviceDescription() const {
  return *device_description_;
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
VulkanExecutor::CreateDeviceDescription() const {
  return CreateDeviceDescription(device_ordinal_);
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
  RETURN_IF_VK_ERROR(impl_->vkCreateShaderModule(
      impl_->device, &shader_info, nullptr, &kernel->shader_module_));

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
  RETURN_IF_VK_ERROR(impl_->vkCreateComputePipelines(
      impl_->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
      &kernel->pipeline_));

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
  std::lock_guard<std::mutex> lock(impl_->queue_mutex);
  impl_->vkDeviceWaitIdle(impl_->device);
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
  allocation->coherent =
      (impl_->memory_properties.memoryTypes[*memory_type].propertyFlags &
       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
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
  impl_->vkDeviceWaitIdle(impl_->device);
  impl_->vkUnmapMemory(impl_->device, allocation->memory);
  impl_->vkDestroyBuffer(impl_->device, allocation->buffer, nullptr);
  impl_->vkFreeMemory(impl_->device, allocation->memory, nullptr);
  *mem = DeviceAddressBase();
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
  std::lock_guard<std::mutex> lock(impl_->queue_mutex);
  return impl_->vkDeviceWaitIdle(impl_->device) == VK_SUCCESS;
}

absl::Status VulkanExecutor::SynchronousMemcpy(DeviceAddressBase* device_dst,
                                               const void* host_src,
                                               uint64_t size) {
  ASSIGN_OR_RETURN(Impl::AllocationView view,
                   impl_->FindAllocation(device_dst->opaque(), size));
  std::memcpy(static_cast<std::byte*>(view.allocation->mapped) + view.offset,
              host_src, size);
  return impl_->Flush(view);
}

absl::Status VulkanExecutor::SynchronousMemcpy(
    void* host_dst, const DeviceAddressBase& device_src, uint64_t size) {
  ASSIGN_OR_RETURN(Impl::AllocationView view,
                   impl_->FindAllocation(device_src.opaque(), size));
  RETURN_IF_ERROR(impl_->Invalidate(view));
  std::memcpy(host_dst,
              static_cast<std::byte*>(view.allocation->mapped) + view.offset,
              size);
  return absl::OkStatus();
}

absl::Status VulkanExecutor::MemcpyDeviceToDevice(
    DeviceAddressBase* device_dst, const DeviceAddressBase& device_src,
    uint64_t size) {
  ASSIGN_OR_RETURN(Impl::AllocationView dst,
                   impl_->FindAllocation(device_dst->opaque(), size));
  ASSIGN_OR_RETURN(Impl::AllocationView src,
                   impl_->FindAllocation(device_src.opaque(), size));
  RETURN_IF_ERROR(impl_->Invalidate(src));
  std::memmove(static_cast<std::byte*>(dst.allocation->mapped) + dst.offset,
               static_cast<std::byte*>(src.allocation->mapped) + src.offset,
               size);
  return impl_->Flush(dst);
}

absl::Status VulkanExecutor::Memset(DeviceAddressBase* location, uint8_t value,
                                    uint64_t size) {
  ASSIGN_OR_RETURN(Impl::AllocationView view,
                   impl_->FindAllocation(location->opaque(), size));
  std::memset(static_cast<std::byte*>(view.allocation->mapped) + view.offset,
              value, size);
  return impl_->Flush(view);
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
  return impl_->Flush(view);
}

void VulkanExecutor::DeallocateStream(Stream* stream) {}

absl::Status VulkanExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  return absl::UnimplementedError("Vulkan peer access is not implemented");
}

bool VulkanExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  return false;
}

int64_t VulkanExecutor::GetMemoryLimitBytes() const {
  return device_description_->device_memory_size();
}

bool VulkanExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  if (total != nullptr) *total = GetMemoryLimitBytes();
  if (free != nullptr) *free = GetMemoryLimitBytes();
  return true;
}

absl::Status VulkanExecutor::Launch(const VulkanKernel& kernel,
                                    const ThreadDim& thread_dims,
                                    const BlockDim& block_dims,
                                    const KernelArgs& args) {
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

  std::lock_guard<std::mutex> lock(impl_->queue_mutex);
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
  result = impl_->vkAllocateCommandBuffers(impl_->device, &command_info,
                                           &command_buffer);
  if (result != VK_SUCCESS) {
    destroy_pool();
    return VulkanError("vkAllocateCommandBuffers", result);
  }
  auto cleanup = [&] {
    impl_->vkFreeCommandBuffers(impl_->device, impl_->command_pool, 1,
                                &command_buffer);
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
  before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  impl_->vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                              &before, 0, nullptr, 0, nullptr);
  impl_->vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           kernel.pipeline_);
  impl_->vkCmdBindDescriptorSets(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline_layout_,
      0, 1, &descriptor_set, 0, nullptr);
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
  if (result == VK_SUCCESS) {
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    result = impl_->vkQueueSubmit(impl_->queue, 1, &submit_info, VK_NULL_HANDLE);
  }
  if (result == VK_SUCCESS) result = impl_->vkQueueWaitIdle(impl_->queue);
  cleanup();
  if (result != VK_SUCCESS) return VulkanError("Vulkan kernel dispatch", result);
  return absl::OkStatus();
}

}  // namespace stream_executor::vulkan
