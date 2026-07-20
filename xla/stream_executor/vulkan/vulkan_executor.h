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

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::vulkan {

class VulkanKernel;

// Vulkan compute StreamExecutor backed by one queue, host-visible coherent
// storage buffers, and timeline-semaphore completion events.
class VulkanExecutor final : public gpu::GpuExecutor {
 public:
  VulkanExecutor(Platform* platform, int device_ordinal);
  ~VulkanExecutor() override;

  static absl::StatusOr<int> GetDeviceCount();
  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  absl::Status Init() override;

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;
  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override;

  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;
  void UnloadKernel(const Kernel* kernel) override;

  DeviceAddressBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceAddressBase* mem) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocator>> CreateMemoryAllocator(
      MemorySpace memory_space) override;
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
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  absl::Status MemcpyDeviceToDevice(DeviceAddressBase* device_dst,
                                    const DeviceAddressBase& device_src,
                                    uint64_t size);
  absl::Status Memset(DeviceAddressBase* location, uint8_t value,
                      uint64_t size);
  absl::Status Memset32(DeviceAddressBase* location, uint32_t value,
                        uint64_t size);
  absl::StatusOr<uint64_t> TimelineValue() const;
  absl::Status WaitTimeline(uint64_t value) const;
  void ScheduleCallback(
      uint64_t value,
      absl::AnyInvocable<absl::Status() &&> callback);

 private:
  friend class VulkanKernel;

  struct Impl;
  absl::StatusOr<uint64_t> Launch(const VulkanKernel& kernel,
                                  const ThreadDim& thread_dims,
                                  const BlockDim& block_dims,
                                  const KernelArgs& args,
                                  uint64_t wait_value);

  std::unique_ptr<Impl> impl_;
};

}  // namespace stream_executor::vulkan

#endif  // XLA_STREAM_EXECUTOR_VULKAN_VULKAN_EXECUTOR_H_
