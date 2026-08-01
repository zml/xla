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

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

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
#include "xla/stream_executor/metal/metal_runtime.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::metal {

class MetalStream;

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
  // Loads a kernel from in-memory metallib bytes, specializing it with Metal
  // function constants (newFunctionWithName:constantValues:). For kernels whose
  // [[function_constant]] values have no default — e.g. the imported llama.cpp
  // flash_attn_ext_vec. A thunk reaches this by static_cast'ing its
  // se::StreamExecutor* to metal::MetalExecutor* (the MPSGraph/metalBLAS pattern).
  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernelWithConstants(
      absl::Span<const uint8_t> metallib, const std::string& kernel_name,
      size_t arity, absl::Span<const MetalFunctionConstant> constants);
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

  // Makes any staged residency changes take effect. Called before each launch,
  // so allocations can be batched into the set for free during loading and turn
  // resident in a single pass before the GPU first reads them. Safe to call
  // with work in flight: Metal does not synchronize a residency set between CPU
  // and GPU, and explicitly permits mutating one while a command buffer that
  // touches its allocations is running.
  void FlushResidency();

  // The per-device MTLSharedEvent used to order dependent executes on the GPU
  // (see metal_runtime.h). Shared by all streams on this device's single queue.
  void* shared_event() const { return shared_event_; }
  // The per-device MTLSharedEventListener (one private serial queue) that drives
  // PJRT host callbacks (SetStateConcrete) off shared-event values — the Metal
  // analog of cuLaunchHostFunc. Null if the device has no shared event.
  void* shared_event_listener() const { return shared_event_listener_; }
  // Allocates the next monotonic value to signal on `shared_event()`. Atomic:
  // PJRT may dispatch executes concurrently. Values start at 1 (0 == "never
  // signaled / no GPU work to wait on").
  uint64_t NextEventValue() {
    return next_event_value_.fetch_add(1, std::memory_order_relaxed) + 1;
  }

  // Cross-stream commit hook (free-execute). On one MTLCommandQueue a command
  // buffer that GPU-WAITS a value must be committed AFTER the buffer that
  // SIGNALS it, else the queue deadlocks (metal_runtime.h). When a consumer
  // stream is about to encode a wait for value `N` produced on another stream,
  // CommitOpenBufferThrough(N) force-commits whichever registered stream still
  // holds an OPEN buffer carrying signal N — so the signaler commits first.
  // No-op while RecordEvent commits per-execute (nothing stays open); becomes
  // load-bearing once per-execute commits are deferred. Streams self-register.
  void RegisterStream(MetalStream* stream);
  void UnregisterStream(MetalStream* stream);
  void CommitOpenBufferThrough(uint64_t value);

  struct Allocation {
    void* buffer = nullptr;
    void* contents = nullptr;
    uint64_t size = 0;
    // Whether this buffer made it into the residency set; false once the cap is
    // reached, and Deallocate must not try to remove what was never added.
    bool wired = false;
  };

  absl::StatusOr<Allocation> ResolveAllocation(const void* ptr) const;

 private:
  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernelFromLibraryPayload(
      absl::Span<const uint8_t> payload, const std::string& kernel_name,
      size_t arity);

  // Commits the staged residency changes and asks for residency. Caller holds
  // allocations_mu_.
  void CommitResidencyLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(allocations_mu_);

  void* device_ = nullptr;
  void* command_queue_ = nullptr;
  // Keeps allocations out of the OS memory compressor; see NewResidencySet in
  // metal_runtime.h. Null when the OS/device does not support residency sets,
  // in which case every residency call below is a no-op.
  void* residency_set_ = nullptr;
  // Wiring is capped at the device's recommended working set, since wiring past
  // it starves the rest of the system. Bytes beyond the cap simply stay
  // unwired -- correct, just compressible.
  uint64_t residency_capacity_ = 0;
  uint64_t residency_bytes_ ABSL_GUARDED_BY(allocations_mu_) = 0;
  // Set when allocations have been added to or removed from the residency set
  // but not yet committed. Read outside the lock on the launch fast path; see
  // FlushResidency.
  std::atomic<bool> residency_staged_ = false;
  uint64_t residency_staged_bytes_ ABSL_GUARDED_BY(allocations_mu_) = 0;
  void* shared_event_ = nullptr;
  void* shared_event_listener_ = nullptr;
  std::atomic<uint64_t> next_event_value_{0};

  // Live device allocations, keyed by the buffer's base address (contents
  // pointer as uintptr_t) and kept ORDERED so ResolveAllocation can map an
  // interior pointer (base + offset, as kernel args carry) to its owning
  // allocation in O(log N) via upper_bound + predecessor, instead of an O(N)
  // scan. Allocations are distinct Shared MTLBuffers with non-overlapping
  // [base, base + size) ranges, so the predecessor is the unique candidate
  // (still containment-checked). Ordered lookup also keeps the eventual
  // zero-copy mmap weight path cheap, where every weight tensor registers here.
  mutable absl::Mutex allocations_mu_;
  std::map<uintptr_t, Allocation> allocations_ ABSL_GUARDED_BY(allocations_mu_);

  mutable absl::Mutex modules_mu_;
  absl::flat_hash_map<const void*, void*> loaded_modules_
      ABSL_GUARDED_BY(modules_mu_);

  // Registered streams, for CommitOpenBufferThrough (cross-stream commit hook).
  mutable absl::Mutex streams_mu_;
  std::vector<MetalStream*> streams_ ABSL_GUARDED_BY(streams_mu_);
};

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_EXECUTOR_H_
