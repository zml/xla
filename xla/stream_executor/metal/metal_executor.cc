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

#include "xla/stream_executor/metal/metal_executor.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/metal/metal_event.h"
#include "xla/stream_executor/metal/metal_kernel.h"
#include "xla/stream_executor/metal/metal_runtime.h"
#include "xla/stream_executor/metal/metal_stream.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::metal {
namespace {

bool Contains(const MetalExecutor::Allocation& allocation, const void* ptr) {
  auto base = reinterpret_cast<uintptr_t>(allocation.contents);
  auto value = reinterpret_cast<uintptr_t>(ptr);
  return value >= base && value < base + allocation.size;
}

bool IsMetallib(absl::Span<const uint8_t> payload) {
  constexpr char kMetallibMagic[] = "MTLB";
  return payload.size() >= 4 &&
         std::memcmp(payload.data(), kMetallibMagic, 4) == 0;
}

absl::StatusOr<void*> LoadMetalLibrary(void* device,
                                       absl::Span<const uint8_t> payload) {
  if (IsMetallib(payload)) {
    return LoadLibraryFromData(device, payload);
  }
  std::string source(reinterpret_cast<const char*>(payload.data()),
                     payload.size());
  return CompileLibrary(device, source);
}

}  // namespace

MetalExecutor::~MetalExecutor() {
  {
    absl::MutexLock lock(allocations_mu_);
    for (const auto& [_, allocation] : allocations_) {
      ReleaseObject(allocation.buffer);
    }
  }
  {
    absl::MutexLock lock(modules_mu_);
    for (const auto& [_, module] : loaded_modules_) {
      ReleaseObject(module);
    }
  }
  ReleaseObject(shared_event_listener_);
  ReleaseObject(shared_event_);
  ResidencySetEndResidency(residency_set_);
  ReleaseObject(residency_set_);
  ReleaseObject(command_queue_);
  ReleaseObject(device_);
}

absl::Status MetalExecutor::Init() {
  TF_ASSIGN_OR_RETURN(device_, RetainDevice(device_ordinal()));
  TF_ASSIGN_OR_RETURN(command_queue_, NewCommandQueue(device_));
  TF_ASSIGN_OR_RETURN(residency_set_, NewResidencySet(device_));
  if (residency_set_ != nullptr) {
    residency_capacity_ = RecommendedMaxWorkingSetSize(device_);
    if (const char* env = std::getenv("METAL_RESIDENCY_LIMIT_MB")) {
      residency_capacity_ = static_cast<uint64_t>(std::atoll(env)) << 20;
    }
    CommandQueueAddResidencySet(command_queue_, residency_set_);
  }
  shared_event_ = NewSharedEvent(device_);
  if (shared_event_ == nullptr) {
    LOG(WARNING) << "Metal: newSharedEvent failed; falling back to host-side "
                    "execute ordering (no GPU pipelining).";
  } else {
    shared_event_listener_ = NewSharedEventListener();
    if (shared_event_listener_ == nullptr) {
      LOG(WARNING) << "Metal: MTLSharedEventListener creation failed; host "
                      "callbacks fall back to per-buffer completion handlers.";
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> MetalExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  if (!spec.has_metal_library_in_memory()) {
    return absl::UnimplementedError(
        "Metal only loads in-memory metallib data.");
  }
  auto payload = spec.metal_library_in_memory();
  if (!payload.has_value()) {
    return absl::InvalidArgumentError("Metal kernel payload is empty.");
  }
  TF_ASSIGN_OR_RETURN(auto kernel,
                      LoadKernelFromLibraryPayload(payload->metallib_bytes,
                                                   spec.kernel_name(),
                                                   spec.arity()));
  if (std::holds_alternative<KernelLoaderSpec::KernelArgsPackingFunc>(
          spec.kernel_args_packing())) {
    kernel->set_args_packing(
        std::get<KernelLoaderSpec::KernelArgsPackingFunc>(
            spec.kernel_args_packing()));
  } else {
    const auto& packing_spec =
        std::get<KernelArgsPackingSpec>(spec.kernel_args_packing());
    kernel->set_args_packing(
        [packing_spec](const Kernel& kernel, const KernelArgs& args) {
          const auto& mem_args = Cast<KernelArgsDeviceAddressArray>(&args);
          return packing_spec.BuildArguments(mem_args->packed_args(),
                                             args.number_of_shared_bytes());
        });
  }
  return kernel;
}

absl::StatusOr<std::unique_ptr<Kernel>>
MetalExecutor::LoadKernelFromLibraryPayload(
    absl::Span<const uint8_t> payload, const std::string& kernel_name,
    size_t arity) {
  TF_ASSIGN_OR_RETURN(void* library, LoadMetalLibrary(device_, payload));
  TF_ASSIGN_OR_RETURN(void* function, NewFunction(library, kernel_name));
  TF_ASSIGN_OR_RETURN(void* pipeline, NewComputePipeline(device_, function));

  auto kernel = std::make_unique<MetalKernel>(this);
  kernel->set_name(kernel_name);
  kernel->set_arity(arity);
  kernel->set_library(library);
  kernel->set_function(function);
  kernel->set_pipeline(pipeline);
  kernel->set_uses_argument_buffer(arity > 31);
  return kernel;
}

absl::StatusOr<std::unique_ptr<Kernel>> MetalExecutor::LoadKernelWithConstants(
    absl::Span<const uint8_t> metallib, const std::string& kernel_name,
    size_t arity, absl::Span<const MetalFunctionConstant> constants) {
  TF_ASSIGN_OR_RETURN(void* library, LoadMetalLibrary(device_, metallib));
  TF_ASSIGN_OR_RETURN(
      void* function,
      NewFunctionWithConstants(library, kernel_name, constants));
  TF_ASSIGN_OR_RETURN(void* pipeline, NewComputePipeline(device_, function));

  auto kernel = std::make_unique<MetalKernel>(this);
  kernel->set_name(kernel_name);
  kernel->set_arity(arity);
  kernel->set_library(library);
  kernel->set_function(function);
  kernel->set_pipeline(pipeline);
  kernel->set_uses_argument_buffer(arity > 31);
  return kernel;
}

absl::StatusOr<ModuleHandle> MetalExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  if (!spec.has_metal_library_in_memory()) {
    return absl::UnimplementedError(
        "Metal only loads in-memory metallib data.");
  }
  absl::Span<const uint8_t> payload = spec.metal_library_in_memory();
  TF_ASSIGN_OR_RETURN(void* library, LoadMetalLibrary(device_, payload));
  ModuleHandle handle(library);
  absl::MutexLock lock(modules_mu_);
  loaded_modules_.emplace(handle.id(), library);
  return handle;
}

bool MetalExecutor::UnloadModule(ModuleHandle module_handle) {
  absl::MutexLock lock(modules_mu_);
  auto it = loaded_modules_.find(module_handle.id());
  if (it == loaded_modules_.end()) {
    return false;
  }
  ReleaseObject(it->second);
  loaded_modules_.erase(it);
  return true;
}

DeviceAddressBase MetalExecutor::Allocate(uint64_t size, int64_t memory_space) {
  auto memory_space_kind = static_cast<MemorySpace>(memory_space);
  if (memory_space_kind == MemorySpace::kCollective) {
    LOG(ERROR) << "Metal does not support collective memory yet.";
    return DeviceAddressBase(nullptr, 0);
  }
  void* contents = nullptr;
  auto buffer = NewSharedBuffer(device_, size, &contents);
  if (!buffer.ok()) {
    LOG(ERROR) << buffer.status();
    return DeviceAddressBase(nullptr, 0);
  }
  if (size != 0) {
    absl::MutexLock lock(allocations_mu_);
    constexpr uint64_t kMinWiredBytes = 1 << 20;  // 1 MiB
    const uint64_t wired_bytes =
        residency_set_ != nullptr ? BufferAllocatedSize(*buffer) : 0;
    const bool wired = residency_set_ != nullptr && size >= kMinWiredBytes &&
                       residency_bytes_ + wired_bytes <= residency_capacity_;
    if (wired) {
      ResidencySetAddAllocation(residency_set_, *buffer);
      residency_staged_ = true;
      residency_staged_bytes_ += wired_bytes;
      residency_bytes_ += wired_bytes;
      constexpr uint64_t kResidencyFlushBytes = uint64_t{1} << 30;  // 1 GiB
      if (residency_staged_bytes_ >= kResidencyFlushBytes) {
        CommitResidencyLocked();
      }
    }
    allocations_.emplace(reinterpret_cast<uintptr_t>(contents),
                         Allocation{*buffer, contents, size, wired});
  }
  return DeviceAddressBase(contents, size);
}

void MetalExecutor::CommitResidencyLocked() {
  ResidencySetCommit(residency_set_);
  ResidencySetRequestResidency(residency_set_);
  residency_staged_ = false;
  residency_staged_bytes_ = 0;
  residency_bytes_ = ResidencySetAllocatedSize(residency_set_);
  VLOG(2) << "Metal residency: " << (residency_bytes_ >> 20) << " MiB in set of "
          << (residency_capacity_ >> 20) << " MiB cap";
}

void MetalExecutor::FlushResidency() {
  if (residency_set_ == nullptr || !residency_staged_) return;
  absl::MutexLock lock(allocations_mu_);
  if (!residency_staged_) return;
  CommitResidencyLocked();
}

void MetalExecutor::Deallocate(DeviceAddressBase* mem) {
  if (mem == nullptr || mem->opaque() == nullptr) {
    return;
  }
  absl::MutexLock lock(allocations_mu_);
  auto it = allocations_.find(reinterpret_cast<uintptr_t>(mem->opaque()));
  if (it == allocations_.end()) {
    LOG(ERROR) << "Attempted to deallocate unknown Metal allocation "
               << mem->opaque();
    *mem = DeviceAddressBase(nullptr, 0);
    return;
  }
  if (it->second.wired) {
    ResidencySetRemoveAllocation(residency_set_, it->second.buffer);
    residency_staged_ = true;
    const uint64_t wired_bytes = BufferAllocatedSize(it->second.buffer);
    residency_bytes_ -= std::min(residency_bytes_, wired_bytes);
  }
  ReleaseObject(it->second.buffer);
  allocations_.erase(it);
  *mem = DeviceAddressBase(nullptr, 0);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
MetalExecutor::CreateMemoryAllocator(MemorySpace memory_space) {
  if (memory_space == MemorySpace::kCollective) {
    return absl::UnimplementedError(
        "Metal collective memory space is not implemented.");
  }
  return std::make_unique<GenericMemoryAllocator>(
      [this](uint64_t size)
          -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        DeviceAddressBase address = Allocate(size, 0);
        if (address.is_null() && size != 0) {
          return absl::ResourceExhaustedError(
              absl::StrCat("Failed to allocate ", size,
                           " bytes of Metal shared memory."));
        }
        return std::make_unique<GenericMemoryAllocation>(
            address.opaque(), size, [this](void* ptr, uint64_t size) {
              DeviceAddressBase mem(ptr, size);
              Deallocate(&mem);
            });
      });
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
MetalExecutor::HostMemoryAllocate(uint64_t size) {
  void* ptr = ::operator new(size);
  return std::make_unique<GenericMemoryAllocation>(
      ptr, size, [](void* ptr, uint64_t) { ::operator delete(ptr); });
}

absl::Status MetalExecutor::SynchronousMemcpy(DeviceAddressBase* device_dst,
                                              const void* host_src,
                                              uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (device_dst == nullptr || device_dst->opaque() == nullptr) {
    return absl::InvalidArgumentError("Metal H2D destination is null.");
  }
  std::memcpy(device_dst->opaque(), host_src, size);
  return absl::OkStatus();
}

absl::Status MetalExecutor::SynchronousMemcpy(
    void* host_dst, const DeviceAddressBase& device_src, uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (host_dst == nullptr || device_src.opaque() == nullptr) {
    return absl::InvalidArgumentError("Metal D2H source or destination is null.");
  }
  std::memcpy(host_dst, device_src.opaque(), size);
  return absl::OkStatus();
}

bool MetalExecutor::SynchronizeAllActivity() {
  return SynchronizeCommandQueue(command_queue_).ok();
}

absl::StatusOr<std::unique_ptr<Stream>> MetalExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  return std::make_unique<MetalStream>(this, priority);
}

absl::StatusOr<std::unique_ptr<Event>> MetalExecutor::CreateEvent() {
  return std::make_unique<MetalEvent>();
}

void MetalExecutor::RegisterStream(MetalStream* stream) {
  absl::MutexLock lock(streams_mu_);
  streams_.push_back(stream);
}

void MetalExecutor::UnregisterStream(MetalStream* stream) {
  absl::MutexLock lock(streams_mu_);
  streams_.erase(std::remove(streams_.begin(), streams_.end(), stream),
                 streams_.end());
}

void MetalExecutor::CommitOpenBufferThrough(uint64_t value) {
  if (value == 0) return;
  absl::MutexLock lock(streams_mu_);
  for (MetalStream* stream : streams_) {
    stream->FlushOpenBufferIfCarrying(value);
  }
}

absl::Status MetalExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  return absl::UnimplementedError("Metal peer access is not implemented.");
}

bool MetalExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  auto info = GetDeviceInfo(device_ordinal());
  if (!info.ok()) return false;
  if (total != nullptr) {
    *total = info->recommended_max_working_set_size;
  }
  if (free != nullptr) {
    *free = -1;
  }
  return true;
}

namespace {

int64_t AppleGpuMemoryBandwidth(const MetalComputeCapability& cc,
                                uint64_t core_count) {
  struct Part {
    int gen;
    int64_t max_cores;
    int64_t bytes_per_second;
  };
  static constexpr Part kParts[] = {
      {13, 8, 68'250'000'000},    // M1            128-bit LPDDR4X-4266
      {13, 16, 204'800'000'000},  // M1 Pro        256-bit LPDDR5-6400
      {13, 32, 409'600'000'000},  // M1 Max        512-bit
      {13, 64, 819'200'000'000},  // M1 Ultra     1024-bit
      {14, 10, 102'400'000'000},  // M2            128-bit LPDDR5-6400
      {14, 19, 204'800'000'000},  // M2 Pro        256-bit
      {14, 38, 409'600'000'000},  // M2 Max        512-bit
      {14, 76, 819'200'000'000},  // M2 Ultra     1024-bit
      {15, 10, 102'400'000'000},  // M3            128-bit LPDDR5-6400
      {15, 18, 153'600'000'000},  // M3 Pro        192-bit
      {15, 30, 307'200'000'000},  // M3 Max        384-bit (30-core bin)
      {15, 40, 409'600'000'000},  // M3 Max        512-bit (40-core bin)
      {15, 80, 819'200'000'000},  // M3 Ultra     1024-bit
      {16, 10, 120'000'000'000},  // M4            128-bit LPDDR5X-7500
      {16, 20, 273'100'000'000},  // M4 Pro        256-bit LPDDR5X-8533
      {16, 32, 409'600'000'000},  // M4 Max        384-bit (32-core bin)
      {16, 40, 546'100'000'000},  // M4 Max        512-bit (40-core bin)
      {17, 10, 153'600'000'000},  // M5            128-bit LPDDR5X-9600
      {17, 20, 307'200'000'000},  // M5 Pro        256-bit
      {17, 32, 460'800'000'000},  // M5 Max        384-bit (32-core bin)
      {17, 40, 614'400'000'000},  // M5 Max        512-bit (40-core bin)
  };

  const int gen = cc.architecture_gen();
  const int64_t cores = static_cast<int64_t>(core_count);
  if (cores > 0) {
    for (const Part& part : kParts) {
      if (part.gen == gen && cores <= part.max_cores) {
        return part.bytes_per_second;
      }
    }
  }

  constexpr int64_t kFallbackBytesPerSecondPerCore = 15'000'000'000;
  const int64_t assumed_cores = cores > 0 ? cores : 8;
  return std::clamp<int64_t>(assumed_cores * kFallbackBytesPerSecondPerCore,
                             60'000'000'000, 3'000'000'000'000);
}

}  // namespace

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MetalExecutor::CreateDeviceDescription() const {
  TF_ASSIGN_OR_RETURN(MetalDeviceInfo info, GetDeviceInfo(device_ordinal()));
  auto desc = std::make_unique<DeviceDescription>();
  desc->set_device_vendor("Apple");
  desc->set_platform_version("Metal");
  desc->set_gpu_compute_capability(
      GpuComputeCapability(MetalComputeCapability(info.architecture)));
  desc->set_driver_version(SemanticVersion{0, 0, 0});
  desc->set_runtime_version(SemanticVersion{0, 0, 0});
  desc->set_compile_time_toolkit_version(SemanticVersion{0, 0, 0});
  desc->set_name(info.name);
  desc->set_model_str(info.name);
  desc->set_pci_bus_id(info.registry_id);
  desc->set_numa_node(0);
  desc->set_thread_dim_limit(ThreadDim(1024, 1024, 64));
  desc->set_block_dim_limit(BlockDim(2147483647, 65535, 65535));
  desc->set_threads_per_block_limit(
      info.max_threads_per_threadgroup == 0
          ? 1024
          : static_cast<int64_t>(info.max_threads_per_threadgroup));
  desc->set_threads_per_core_limit(1024);
  desc->set_threads_per_warp(32);
  desc->set_registers_per_core_limit(0);
  desc->set_registers_per_block_limit(0);
  desc->set_device_address_bits(64);
  desc->set_device_memory_size(info.recommended_max_working_set_size);
  desc->set_l2_cache_size(0);
  desc->set_memory_bandwidth(AppleGpuMemoryBandwidth(
      MetalComputeCapability(info.architecture), info.gpu_core_count));
  desc->set_pcie_bandwidth(0);
  const int64_t tg_mem = info.max_threadgroup_memory_length == 0
                             ? 32 * 1024
                             : static_cast<int64_t>(
                                   info.max_threadgroup_memory_length);
  desc->set_shared_memory_per_core(tg_mem);
  desc->set_shared_memory_per_block(tg_mem);
  desc->set_shared_memory_per_block_optin(tg_mem);
  desc->set_clock_rate_ghz(1.4);
  desc->set_core_count(info.gpu_core_count == 0
                           ? 1
                           : static_cast<int>(info.gpu_core_count));
  desc->set_fpus_per_core(128);
  desc->set_ecc_enabled(false);
  return desc;
}

absl::StatusOr<DeviceAddressBase> MetalExecutor::GetMemoryRange(
    const DeviceAddressBase& location) const {
  TF_ASSIGN_OR_RETURN(Allocation allocation,
                      ResolveAllocation(location.opaque()));
  return DeviceAddressBase(allocation.contents, allocation.size);
}

absl::StatusOr<MemorySpace> MetalExecutor::GetPointerMemorySpace(
    const void* ptr) {
  TF_RETURN_IF_ERROR(ResolveAllocation(ptr).status());
  return MemorySpace::kDevice;
}

absl::StatusOr<MetalExecutor::Allocation> MetalExecutor::ResolveAllocation(
    const void* ptr) const {
  if (ptr == nullptr) {
    return absl::InvalidArgumentError("Cannot resolve null Metal pointer.");
  }
  absl::MutexLock lock(allocations_mu_);
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = allocations_.upper_bound(addr);
  if (it != allocations_.begin()) {
    --it;
    if (Contains(it->second, ptr)) {
      return it->second;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("No Metal allocation contains pointer %p", ptr));
}

}  // namespace stream_executor::metal
