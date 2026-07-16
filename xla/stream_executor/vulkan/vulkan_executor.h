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

#ifndef XLA_STREAM_EXECUTOR_VULKAN_VULKAN_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_VULKAN_VULKAN_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::vulkan {

class VulkanKernel;

// Minimal Vulkan compute StreamExecutor. It deliberately uses host-visible
// Vulkan storage buffers and synchronous queue submission. This keeps the
// initial implementation correct for single-device buffer execution while
// leaving asynchronous staging and command-buffer reuse as follow-up work.
class VulkanExecutor final : public StreamExecutor {
 public:
  VulkanExecutor(Platform* platform, int device_ordinal);
  ~VulkanExecutor() override;

  static absl::StatusOr<int> GetDeviceCount();
  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  std::unique_ptr<ActivateContext> Activate() override;
  const Platform* GetPlatform() const override;
  absl::Status Init() override;
  int device_ordinal() const override;

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;
  const DeviceDescription& GetDeviceDescription() const override;
  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override;

  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;
  void UnloadKernel(const Kernel* kernel) override;

  DeviceAddressBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceAddressBase* mem) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override;
  bool SynchronizeAllActivity() override;
  absl::Status SynchronousMemcpy(DeviceAddressBase* device_dst,
                                 const void* host_src,
                                 uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceAddressBase& device_src,
                                 uint64_t size) override;
  void DeallocateStream(Stream* stream) override;
  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(StreamExecutor* other) override;
  int64_t GetMemoryLimitBytes() const override;
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  absl::Status MemcpyDeviceToDevice(DeviceAddressBase* device_dst,
                                    const DeviceAddressBase& device_src,
                                    uint64_t size);
  absl::Status Memset(DeviceAddressBase* location, uint8_t value,
                      uint64_t size);
  absl::Status Memset32(DeviceAddressBase* location, uint32_t value,
                        uint64_t size);

 private:
  friend class VulkanKernel;

  struct Impl;
  absl::Status Launch(const VulkanKernel& kernel, const ThreadDim& thread_dims,
                      const BlockDim& block_dims, const KernelArgs& args);

  Platform* platform_;
  int device_ordinal_;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<DeviceDescription> device_description_;
};

}  // namespace stream_executor::vulkan

#endif  // XLA_STREAM_EXECUTOR_VULKAN_VULKAN_EXECUTOR_H_
