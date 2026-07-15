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

#include "xla/stream_executor/musa/musa_executor.h"

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_runtime.h"
#include "xla/stream_executor/musa/musa_stream.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::musa {
namespace {

std::string DefaultArchitecture() {
  const char* archs = std::getenv("MUSA_GPU_ARCHS");
  if (archs == nullptr || archs[0] == '\0') {
    return "unknown";
  }
  std::vector<std::string> split = absl::StrSplit(archs, ',');
  return split.empty() || split[0].empty() ? "unknown" : split[0];
}

SemanticVersion VersionFromRuntimeInt(int version) {
  if (version <= 0) {
    return SemanticVersion{0, 0, 0};
  }
  return SemanticVersion{static_cast<unsigned>(version / 1000),
                         static_cast<unsigned>((version % 1000) / 10),
                         static_cast<unsigned>(version % 10)};
}

void* MallocHost(uint64_t size) { return std::malloc(static_cast<size_t>(size)); }

void FreeHost(void* ptr, uint64_t) { std::free(ptr); }

}  // namespace

absl::Status MusaExecutor::Init() {
  TF_RETURN_IF_ERROR(MusaRuntime::Get()->Init());
  return MusaRuntime::Get()->SetDevice(device_ordinal());
}

absl::StatusOr<std::unique_ptr<Stream>> MusaExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_RETURN_IF_ERROR(MusaRuntime::Get()->SetDevice(device_ordinal()));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<MusaStream> stream,
                      MusaStream::Create(this, priority));
  {
    absl::MutexLock lock(&alive_streams_mu_);
    alive_streams_[stream->stream_handle()] = stream.get();
  }
  return stream;
}

absl::StatusOr<std::unique_ptr<Event>> MusaExecutor::CreateEvent() {
  TF_ASSIGN_OR_RETURN(MusaEvent event, MusaEvent::Create(this));
  return std::make_unique<MusaEvent>(std::move(event));
}

DeviceAddressBase MusaExecutor::Allocate(uint64_t size, int64_t memory_space) {
  if (size == 0) {
    return DeviceAddressBase();
  }
  auto ptr = MusaRuntime::Get()->Malloc(size);
  if (!ptr.ok()) {
    LOG(ERROR) << "MUSA allocation failed: " << ptr.status();
    return DeviceAddressBase();
  }
  return DeviceAddressBase(*ptr, size);
}

void MusaExecutor::Deallocate(DeviceAddressBase* mem) {
  if (mem == nullptr || mem->is_null()) {
    return;
  }
  absl::Status status = MusaRuntime::Get()->Free(mem->opaque());
  if (!status.ok()) {
    LOG(ERROR) << "MUSA free failed: " << status;
  }
  *mem = DeviceAddressBase();
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
MusaExecutor::HostMemoryAllocate(uint64_t size) {
  auto ptr = MusaRuntime::Get()->HostAlloc(size);
  if (ptr.ok()) {
    return std::make_unique<GenericMemoryAllocation>(
        *ptr, size, [](void* p, uint64_t) {
          (void)MusaRuntime::Get()->FreeHost(p);
        });
  }
  void* host_ptr = MallocHost(size);
  if (host_ptr == nullptr) {
    return absl::ResourceExhaustedError("Failed to allocate host memory.");
  }
  return std::make_unique<GenericMemoryAllocation>(host_ptr, size, FreeHost);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
MusaExecutor::CreateMemoryAllocator(MemorySpace memory_space) {
  switch (memory_space) {
    case MemorySpace::kHost:
      return std::make_unique<GenericMemoryAllocator>(
          [this](uint64_t size) { return HostMemoryAllocate(size); });
    case MemorySpace::kCollective:
    case MemorySpace::kDevice:
      return std::make_unique<GenericMemoryAllocator>(
          [](uint64_t size)
              -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
            TF_ASSIGN_OR_RETURN(void* ptr, MusaRuntime::Get()->Malloc(size));
            return std::make_unique<GenericMemoryAllocation>(
                ptr, size, [](void* location, uint64_t) {
                  absl::Status status = MusaRuntime::Get()->Free(location);
                  if (!status.ok()) {
                    LOG(ERROR) << "Failed to free MUSA device memory: "
                               << status;
                  }
                });
          });
    default:
      return absl::UnimplementedError("Unsupported MUSA memory space.");
  }
}

bool MusaExecutor::SynchronizeAllActivity() {
  return MusaRuntime::Get()->DeviceSynchronize().ok();
}

absl::Status MusaExecutor::SynchronousMemcpy(DeviceAddressBase* device_dst,
                                             const void* host_src,
                                             uint64_t size) {
  return MusaRuntime::Get()->Memcpy(device_dst->opaque(), host_src, size,
                                    MusaMemcpyKind::kHostToDevice);
}

absl::Status MusaExecutor::SynchronousMemcpy(
    void* host_dst, const DeviceAddressBase& device_src, uint64_t size) {
  return MusaRuntime::Get()->Memcpy(host_dst, device_src.opaque(), size,
                                    MusaMemcpyKind::kDeviceToHost);
}

void MusaExecutor::DeallocateStream(Stream* stream) {
  auto* musa_stream = dynamic_cast<MusaStream*>(stream);
  if (musa_stream == nullptr) {
    return;
  }
  absl::MutexLock lock(&alive_streams_mu_);
  alive_streams_.erase(musa_stream->stream_handle());
}

absl::Status MusaExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  return absl::UnimplementedError(
      "MUSA peer access is not implemented in PJRT GPU v1.");
}

bool MusaExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  return false;
}

bool MusaExecutor::DeviceMemoryUsage(int64_t* free,
                                     int64_t* total) const {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  absl::Status status =
      MusaRuntime::Get()->MemGetInfo(&free_bytes, &total_bytes);
  if (!status.ok()) {
    return false;
  }
  *free = static_cast<int64_t>(free_bytes);
  *total = static_cast<int64_t>(total_bytes);
  return true;
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MusaExecutor::CreateDeviceDescription() const {
  return MusaExecutor::CreateDeviceDescription(device_ordinal());
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MusaExecutor::CreateDeviceDescription(int device_ordinal) {
  TF_RETURN_IF_ERROR(MusaRuntime::Get()->SetDevice(device_ordinal));
  auto desc = std::make_unique<DeviceDescription>();
  desc->set_device_vendor("Moore Threads");
  desc->set_name("MUSA device");
  desc->set_model_str("MUSA device");
  desc->set_platform_version("MUSA");
  desc->set_musa_compute_capability(DefaultArchitecture());
  desc->set_thread_dim_limit(ThreadDim(1024, 1024, 64));
  desc->set_block_dim_limit(BlockDim(2147483647, 65535, 65535));
  desc->set_threads_per_block_limit(1024);
  desc->set_threads_per_core_limit(2048);
  desc->set_threads_per_warp(32);
  desc->set_registers_per_core_limit(65536);
  desc->set_registers_per_block_limit(65536);
  desc->set_device_address_bits(64);
  desc->set_core_count(1);
  desc->set_fpus_per_core(1);
  desc->set_shared_memory_per_core(64 * 1024);
  desc->set_shared_memory_per_block(48 * 1024);
  desc->set_shared_memory_per_block_optin(64 * 1024);

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (MusaRuntime::Get()->MemGetInfo(&free_bytes, &total_bytes).ok()) {
    desc->set_device_memory_size(static_cast<int64_t>(total_bytes));
  }
  if (auto runtime_version = MusaRuntime::Get()->RuntimeVersion();
      runtime_version.ok()) {
    desc->set_runtime_version(VersionFromRuntimeInt(*runtime_version));
  }
  if (auto driver_version = MusaRuntime::Get()->DriverVersion();
      driver_version.ok()) {
    desc->set_driver_version(VersionFromRuntimeInt(*driver_version));
  }
  return desc;
}

absl::StatusOr<std::shared_ptr<DeviceAddressBase>>
MusaExecutor::CreateOrShareConstant(Stream* stream,
                                    absl::Span<const uint8_t> content) {
  DeviceAddressBase allocation = Allocate(content.size(), /*memory_space=*/0);
  if (allocation.is_null()) {
    return absl::ResourceExhaustedError(
        "Failed to allocate MUSA constant memory.");
  }
  absl::Status status =
      SynchronousMemcpy(&allocation, content.data(), content.size());
  if (!status.ok()) {
    Deallocate(&allocation);
    return status;
  }
  return std::shared_ptr<DeviceAddressBase>(
      new DeviceAddressBase(allocation), [this](DeviceAddressBase* ptr) {
        DeviceAddressBase allocation = *ptr;
        Deallocate(&allocation);
        delete ptr;
      });
}

absl::StatusOr<std::unique_ptr<Kernel>> MusaExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  return absl::UnimplementedError(
      "MUSA kernel loading is not implemented in PJRT GPU v1.");
}

absl::StatusOr<ModuleHandle> MusaExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  return absl::UnimplementedError(
      "MUSA module loading is not implemented in PJRT GPU v1.");
}

absl::StatusOr<DeviceAddressBase> MusaExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  return absl::UnimplementedError(
      "MUSA symbol lookup is not implemented in PJRT GPU v1.");
}

Stream* MusaExecutor::FindAllocatedStream(void* device_stream) {
  absl::MutexLock lock(&alive_streams_mu_);
  auto it = alive_streams_.find(device_stream);
  return it == alive_streams_.end() ? nullptr : it->second;
}

}  // namespace stream_executor::musa
