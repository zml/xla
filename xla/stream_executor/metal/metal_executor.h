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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::metal {

class MetalExecutor : public gpu::GpuExecutor {
 public:
  MetalExecutor(Platform* platform, int device_ordinal)
      : GpuExecutor(platform, device_ordinal) {}
  ~MetalExecutor() override;

  absl::Status Init() override;

  blas::BlasSupport* AsBlas() override { return nullptr; }
  dnn::DnnSupport* AsDnn() override { return nullptr; }
  fft::FftSupport* AsFft() override { return nullptr; }

  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;
  void UnloadKernel(const Kernel* kernel) override {}
  absl::StatusOr<ModuleHandle> LoadModule(
      const MultiModuleLoaderSpec& spec) override;
  bool UnloadModule(ModuleHandle module_handle) override;

  DeviceAddressBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceAddressBase* mem) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocator>> CreateMemoryAllocator(
      MemorySpace memory_space) override;
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override;

  absl::Status SynchronousMemcpy(DeviceAddressBase* device_dst,
                                 const void* host_src,
                                 uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceAddressBase& device_src,
                                 uint64_t size) override;
  bool SynchronizeAllActivity() override;

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;
  void DeallocateStream(Stream* stream) override {}

  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(StreamExecutor* other) override { return false; }
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override;

  absl::StatusOr<DeviceAddressBase> GetMemoryRange(
      const DeviceAddressBase& location) const override;
  absl::StatusOr<MemorySpace> GetPointerMemorySpace(const void* ptr) override;

  void* device() const { return device_; }
  void* command_queue() const { return command_queue_; }

  struct Allocation {
    void* buffer = nullptr;
    void* contents = nullptr;
    uint64_t size = 0;
  };

  absl::StatusOr<Allocation> ResolveAllocation(const void* ptr) const;

 private:
  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernelFromLibraryPayload(
      absl::Span<const uint8_t> payload, const std::string& kernel_name,
      size_t arity);

  void* device_ = nullptr;
  void* command_queue_ = nullptr;

  mutable absl::Mutex allocations_mu_;
  absl::flat_hash_map<const void*, Allocation> allocations_
      ABSL_GUARDED_BY(allocations_mu_);

  mutable absl::Mutex modules_mu_;
  absl::flat_hash_map<const void*, void*> loaded_modules_
      ABSL_GUARDED_BY(modules_mu_);
};

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_EXECUTOR_H_
