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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/musa/musa_device_description.h"
#include "xla/stream_executor/musa/musa_device_properties.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_module.h"
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

absl::StatusOr<std::unique_ptr<MemoryAllocation>> AllocateHostMemory(
    std::shared_ptr<MusaContext> context, uint64_t size) {
  gpu::ScopedActivateContext activation(context.get());
  auto ptr = MusaRuntime::Get()->HostAlloc(size);
  if (ptr.ok()) {
    return std::make_unique<GenericMemoryAllocation>(
        *ptr, size, [context = std::move(context)](void* p, uint64_t) {
          gpu::ScopedActivateContext activation(context.get());
          absl::Status status = MusaRuntime::Get()->FreeHost(p);
          if (!status.ok()) {
            LOG(ERROR) << "Failed to free MUSA host memory: " << status;
          }
        });
  }
  void* host_ptr = MallocHost(size);
  if (host_ptr == nullptr) {
    return absl::ResourceExhaustedError("Failed to allocate host memory.");
  }
  return std::make_unique<GenericMemoryAllocation>(host_ptr, size, FreeHost);
}

}  // namespace

MusaExecutor::MusaExecutor(Platform* platform, int device_ordinal)
    : gpu::GpuExecutor(platform, device_ordinal),
      driver_(&MusaDriver::Instance()) {}

MusaExecutor::~MusaExecutor() = default;

std::unique_ptr<ActivateContext> MusaExecutor::Activate() {
  CHECK(context_ != nullptr) << "MUSA executor has not been initialized";
  return std::make_unique<gpu::ScopedActivateContext>(context_.get());
}

absl::Status MusaExecutor::Init() {
  TF_RETURN_IF_ERROR(driver_->Init());
  TF_ASSIGN_OR_RETURN(context_, MusaContext::Create(device_ordinal(), driver_));
  std::unique_ptr<ActivateContext> activation = Activate();
  TF_RETURN_IF_ERROR(MusaRuntime::Get()->Init());
  TF_RETURN_IF_ERROR(MusaRuntime::Get()->SetDevice(device_ordinal()));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<DeviceDescription> description,
                      CreateDeviceDescription());
  const MusaComputeCapability* capability =
      description->gpu_compute_capability().musa_compute_capability();
  if (capability == nullptr || capability->architecture().empty()) {
    return absl::InternalError(
        "Live MUSA device description has no architecture");
  }
  module_cache_ = std::make_unique<MusaModuleCache>(driver_, context_,
                                                    capability->architecture());
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Stream>> MusaExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
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
  std::unique_ptr<ActivateContext> activation = Activate();
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
  std::unique_ptr<ActivateContext> activation = Activate();
  absl::Status status = MusaRuntime::Get()->Free(mem->opaque());
  if (!status.ok()) {
    LOG(ERROR) << "MUSA free failed: " << status;
  }
  *mem = DeviceAddressBase();
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
MusaExecutor::HostMemoryAllocate(uint64_t size) {
  return AllocateHostMemory(context_, size);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
MusaExecutor::CreateMemoryAllocator(MemorySpace memory_space) {
  switch (memory_space) {
    case MemorySpace::kHost:
      return std::make_unique<GenericMemoryAllocator>(
          [context = context_](uint64_t size) {
            return AllocateHostMemory(context, size);
          });
    case MemorySpace::kCollective:
    case MemorySpace::kDevice:
      return std::make_unique<GenericMemoryAllocator>(
          [context = context_](uint64_t size)
              -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
            gpu::ScopedActivateContext activation(context.get());
            TF_ASSIGN_OR_RETURN(void* ptr, MusaRuntime::Get()->Malloc(size));
            return std::make_unique<GenericMemoryAllocation>(
                ptr, size, [context](void* location, uint64_t) {
                  gpu::ScopedActivateContext activation(context.get());
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
  return context_->Synchronize().ok();
}

absl::Status MusaExecutor::SynchronousMemcpy(DeviceAddressBase* device_dst,
                                             const void* host_src,
                                             uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  return MusaRuntime::Get()->Memcpy(device_dst->opaque(), host_src, size,
                                    MusaMemcpyKind::kHostToDevice);
}

absl::Status MusaExecutor::SynchronousMemcpy(
    void* host_dst, const DeviceAddressBase& device_src, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
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
  gpu::ScopedActivateContext activation(context_.get());
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
  MusaDriver& driver = MusaDriver::Instance();
  TF_RETURN_IF_ERROR(driver.Init());
  TF_ASSIGN_OR_RETURN(MUcontext previous_context, driver.CurrentContext());
  absl::StatusOr<std::unique_ptr<DeviceDescription>> discovered =
      [&]() -> absl::StatusOr<std::unique_ptr<DeviceDescription>> {
    TF_RETURN_IF_ERROR(MusaRuntime::Get()->SetDevice(device_ordinal));
    TF_ASSIGN_OR_RETURN(
        MusaDeviceProperties properties,
        MusaRuntime::Get()->GetDeviceProperties(device_ordinal));
    TF_ASSIGN_OR_RETURN(int runtime_version,
                        MusaRuntime::Get()->RuntimeVersion());
    TF_ASSIGN_OR_RETURN(int driver_version, driver.DriverVersion());

    MusaDeviceVersions versions{
        .runtime_api = runtime_version,
        .driver_api = driver_version,
        .compile_time_toolkit = XLA_MUSA_TOOLKIT_VERSION,
        .kernel_mode_driver = GetKernelModeDriverVersion(),
    };
    TF_ASSIGN_OR_RETURN(DeviceDescription description,
                        BuildMusaDeviceDescription(properties, versions));
    return std::make_unique<DeviceDescription>(std::move(description));
  }();
  absl::Status restore_status = driver.SetCurrentContext(previous_context);
  if (!restore_status.ok()) {
    return absl::Status(
        restore_status.code(),
        absl::StrCat("Failed to restore the MUSA context after device "
                     "description discovery: ",
                     restore_status.message(),
                     discovered.ok()
                         ? ""
                         : absl::StrCat("; discovery also failed: ",
                                        discovered.status().message())));
  }
  return discovered;
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
      new DeviceAddressBase(allocation),
      [context = context_](DeviceAddressBase* ptr) {
        gpu::ScopedActivateContext activation(context.get());
        absl::Status status = MusaRuntime::Get()->Free(ptr->opaque());
        if (!status.ok()) {
          LOG(ERROR) << "Failed to free MUSA constant memory: " << status;
        }
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
  if (module_cache_ == nullptr) {
    return absl::FailedPreconditionError(
        "MUSA executor must be initialized before loading a module");
  }
  if (!spec.has_musa_mubin_in_memory()) {
    return absl::InvalidArgumentError(
        "MUSA module loading requires an explicit MUBIN artifact");
  }
  return module_cache_->AcquireModuleHandle(spec.musa_mubin_in_memory());
}

bool MusaExecutor::UnloadModule(ModuleHandle module_handle) {
  return module_cache_ != nullptr &&
         module_cache_->ReleaseModuleHandle(module_handle);
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
