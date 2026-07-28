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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/host_callback_registry.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

class MusaContext;
class MusaDriver;
class MusaModule;
class MusaModuleCache;
class MusaModuleReaper;

class MusaExecutor : public gpu::GpuExecutor {
 public:
  MusaExecutor(Platform* platform, int device_ordinal,
               absl::Duration callback_poll_interval = absl::Seconds(5));

  ~MusaExecutor() override;

  std::unique_ptr<ActivateContext> Activate() override;

  absl::Status Init() override;

  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;

  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;

  DeviceAddressBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceAddressBase* mem) override;

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override;

  absl::StatusOr<std::unique_ptr<MemoryAllocator>> CreateMemoryAllocator(
      MemorySpace memory_space) override;

  bool SynchronizeAllActivity() override;

  absl::Status SynchronousMemcpy(DeviceAddressBase* device_dst,
                                 const void* host_src, uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceAddressBase& device_src,
                                 uint64_t size) override;

  void DeallocateStream(Stream* stream) override;

  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(StreamExecutor* other) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override;

  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  absl::StatusOr<std::shared_ptr<DeviceAddressBase>> CreateOrShareConstant(
      Stream* stream, absl::Span<const uint8_t> content) override;

  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;

  absl::StatusOr<ModuleHandle> LoadModule(
      const MultiModuleLoaderSpec& spec) override;

  bool UnloadModule(ModuleHandle module_handle) override;

  absl::StatusOr<DeviceAddressBase> GetSymbol(
      const std::string& symbol_name, ModuleHandle module_handle) override;

  Stream* FindAllocatedStream(void* device_stream) override;

  blas::BlasSupport* AsBlas() override;

  fft::FftSupport* AsFft() override;

  gpu::HostCallbackRegistry* host_callback_registry() const {
    return host_callback_registry_.get();
  }

 private:
  MusaDriver* driver_;
  // Allocators and RAII allocations may outlive their executor. They retain
  // the primary context needed by their allocation/free callbacks.
  std::shared_ptr<MusaContext> context_;
  // Declared after context_ so cached modules unload before the primary
  // context reference is released during executor teardown.
  std::unique_ptr<MusaModuleCache> module_cache_;
  // Declared before the callback registry so it remains available while the
  // registry tears down outstanding stream-ordered completion callbacks.
  std::unique_ptr<MusaModuleReaper> module_reaper_;
  std::unique_ptr<gpu::HostCallbackRegistry> host_callback_registry_;

  mutable absl::Mutex support_mu_;
  std::unique_ptr<blas::BlasSupport> blas_ ABSL_GUARDED_BY(support_mu_);
  std::unique_ptr<fft::FftSupport> fft_ ABSL_GUARDED_BY(support_mu_);

  mutable absl::Mutex alive_streams_mu_;
  absl::flat_hash_map<void*, Stream*> alive_streams_
      ABSL_GUARDED_BY(alive_streams_mu_);
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_EXECUTOR_H_
