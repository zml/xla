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
  ReleaseObject(command_queue_);
  ReleaseObject(device_);
}

absl::Status MetalExecutor::Init() {
  TF_ASSIGN_OR_RETURN(device_, RetainDevice(device_ordinal()));
  TF_ASSIGN_OR_RETURN(command_queue_, NewCommandQueue(device_));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> MetalExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  if (!spec.has_cuda_cubin_in_memory()) {
    return absl::UnimplementedError(
        "Metal only loads MSL source from the opaque in-memory binary slot.");
  }
  auto msl = spec.cuda_cubin_in_memory();
  if (!msl.has_value()) {
    return absl::InvalidArgumentError("Metal kernel payload is empty.");
  }
  TF_ASSIGN_OR_RETURN(auto kernel, LoadKernelFromMsl(msl->cubin_bytes,
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
          return packing_spec.BuildArguments(mem_args->device_addr_args(),
                                             args.number_of_shared_bytes());
        });
  }
  return kernel;
}

absl::StatusOr<std::unique_ptr<Kernel>> MetalExecutor::LoadKernelFromMsl(
    absl::Span<const uint8_t> msl, const std::string& kernel_name,
    size_t arity) {
  std::string source(reinterpret_cast<const char*>(msl.data()), msl.size());
  TF_ASSIGN_OR_RETURN(void* library, CompileLibrary(device_, source));
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

absl::StatusOr<ModuleHandle> MetalExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  if (!spec.has_cuda_cubin_in_memory()) {
    return absl::UnimplementedError(
        "Metal only loads MSL source from the opaque in-memory binary slot.");
  }
  absl::Span<const uint8_t> msl = spec.cuda_cubin_in_memory();
  std::string source(reinterpret_cast<const char*>(msl.data()), msl.size());
  TF_ASSIGN_OR_RETURN(void* library, CompileLibrary(device_, source));
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
  if (memory_space_kind == MemorySpace::kCollective ||
      memory_space_kind == MemorySpace::kP2P) {
    LOG(ERROR) << "Metal does not support collective or P2P memory yet.";
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
    allocations_.emplace(contents, Allocation{*buffer, contents, size});
  }
  return DeviceAddressBase(contents, size);
}

void MetalExecutor::Deallocate(DeviceAddressBase* mem) {
  if (mem == nullptr || mem->opaque() == nullptr) {
    return;
  }
  absl::MutexLock lock(allocations_mu_);
  auto it = allocations_.find(mem->opaque());
  if (it == allocations_.end()) {
    LOG(ERROR) << "Attempted to deallocate unknown Metal allocation "
               << mem->opaque();
    *mem = DeviceAddressBase(nullptr, 0);
    return;
  }
  ReleaseObject(it->second.buffer);
  allocations_.erase(it);
  *mem = DeviceAddressBase(nullptr, 0);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
MetalExecutor::CreateMemoryAllocator(MemorySpace memory_space) {
  if (memory_space == MemorySpace::kCollective ||
      memory_space == MemorySpace::kP2P) {
    return absl::UnimplementedError(
        "Metal collective and P2P memory spaces are not implemented.");
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

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MetalExecutor::CreateDeviceDescription() const {
  TF_ASSIGN_OR_RETURN(MetalDeviceInfo info, GetDeviceInfo(device_ordinal()));
  auto desc = std::make_unique<DeviceDescription>();
  desc->set_device_vendor("Apple");
  desc->set_platform_version("Metal");
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
  desc->set_memory_bandwidth(0);
  desc->set_pcie_bandwidth(0);
  desc->set_shared_memory_per_core(32 * 1024);
  desc->set_shared_memory_per_block(32 * 1024);
  desc->set_shared_memory_per_block_optin(32 * 1024);
  desc->set_clock_rate_ghz(0.0);
  desc->set_core_count(1);
  desc->set_fpus_per_core(1);
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
  for (const auto& [_, allocation] : allocations_) {
    if (Contains(allocation, ptr)) {
      return allocation;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("No Metal allocation contains pointer %p", ptr));
}

}  // namespace stream_executor::metal
