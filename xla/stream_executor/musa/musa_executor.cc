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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_device_description.h"
#include "xla/stream_executor/musa/musa_device_properties.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_runtime.h"
#include "xla/stream_executor/musa/musa_stream.h"
#include "xla/stream_executor/musa/musa_version_parser.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"

#ifndef XLA_MUSA_TOOLKIT_VERSION
#define XLA_MUSA_TOOLKIT_VERSION 0
#endif

namespace stream_executor::musa {
namespace {

std::optional<SemanticVersion> GetKernelModeDriverVersion() {
  std::FILE* file = std::fopen("/proc/driver/musa/version", "r");
  if (file == nullptr) {
    LOG(WARNING) << "Could not open /proc/driver/musa/version";
    return std::nullopt;
  }
  std::array<char, 256> contents = {};
  const char* read_result = std::fgets(contents.data(), contents.size(), file);
  std::fclose(file);
  if (read_result == nullptr) {
    LOG(WARNING) << "Could not read /proc/driver/musa/version";
    return std::nullopt;
  }
  auto version = ParseMusaKernelDriverVersion(contents.data());
  if (!version.ok()) {
    LOG(WARNING) << "Could not parse MUSA kernel driver version: "
                 << version.status();
    return std::nullopt;
  }
  return *version;
}

void* MallocHost(uint64_t size) {
  return std::malloc(static_cast<size_t>(size));
}

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
        *ptr, size,
        [](void* p, uint64_t) { (void)MusaRuntime::Get()->FreeHost(p); });
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
                    LOG(ERROR)
                        << "Failed to free MUSA device memory: " << status;
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

bool MusaExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
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
  TF_ASSIGN_OR_RETURN(MusaDeviceProperties properties,
                      MusaRuntime::Get()->GetDeviceProperties(device_ordinal));
  TF_ASSIGN_OR_RETURN(int runtime_version,
                      MusaRuntime::Get()->RuntimeVersion());
  TF_ASSIGN_OR_RETURN(int driver_version, MusaRuntime::Get()->DriverVersion());

  MusaDeviceVersions versions{
      .runtime_api = runtime_version,
      .driver_api = driver_version,
      .compile_time_toolkit = XLA_MUSA_TOOLKIT_VERSION,
      .kernel_mode_driver = GetKernelModeDriverVersion(),
  };
  TF_ASSIGN_OR_RETURN(DeviceDescription description,
                      BuildMusaDeviceDescription(properties, versions));
  return std::make_unique<DeviceDescription>(std::move(description));
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
