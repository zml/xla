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

#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/musa/musa_device_description.h"
#include "xla/stream_executor/musa/musa_device_properties.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_kernel.h"
#include "xla/stream_executor/musa/musa_module.h"
#include "xla/stream_executor/musa/musa_module_reaper.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_runtime.h"
#include "xla/stream_executor/musa/musa_stream.h"
#include "xla/stream_executor/musa/musa_version_parser.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"

#ifndef XLA_MUSA_TOOLKIT_VERSION
#define XLA_MUSA_TOOLKIT_VERSION 0
#endif

namespace stream_executor::musa {
namespace {

void* MallocHost(uint64_t size) {
  return std::malloc(static_cast<size_t>(size));
}

void FreeHost(void* ptr, uint64_t) { std::free(ptr); }

struct RetainedExecutorModules {
  std::vector<std::shared_ptr<MusaModule>> orphan_modules;
  std::unique_ptr<MusaModuleCache> module_cache;
};

void RetainModulesAfterSynchronizationFailure(
    std::vector<std::shared_ptr<MusaModule>> modules,
    std::unique_ptr<MusaModuleCache> module_cache) {
  static auto* mutex = new absl::Mutex;
  static auto* retained = new std::vector<RetainedExecutorModules>;
  absl::MutexLock lock(mutex);
  retained->push_back(
      RetainedExecutorModules{std::move(modules), std::move(module_cache)});
}

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

MusaExecutor::MusaExecutor(Platform* platform, int device_ordinal,
                           absl::Duration callback_poll_interval)
    : gpu::GpuExecutor(platform, device_ordinal),
      driver_(&MusaDriver::Instance()),
      module_reaper_(std::make_unique<MusaModuleReaper>(device_ordinal)),
      host_callback_registry_(std::make_unique<gpu::HostCallbackRegistry>(
          device_ordinal, callback_poll_interval)) {}

MusaExecutor::~MusaExecutor() {
  const bool has_orphans =
      module_reaper_ != nullptr && module_reaper_->HasOrphans();
  const bool has_cached_modules =
      module_cache_ != nullptr && !module_cache_->IsQuiescent();
  if (!has_orphans && !has_cached_modules) return;

  // Ambiguous launch and callback-enqueue failures become reaper orphans. A
  // successful context synchronization makes them safe for the ordinary
  // reaper worker. If synchronization fails, retaining both the orphans and
  // cache is safer than racing module unload with device execution.
  absl::Status synchronization =
      context_ == nullptr ? absl::FailedPreconditionError(
                                "MUSA executor has no context during teardown")
                          : context_->Synchronize();
  if (!synchronization.ok()) {
    LOG(ERROR) << "Unable to synchronize MUSA activity during executor "
                  "teardown; retaining loaded kernel modules for process "
                  "lifetime: "
               << synchronization;
    RetainModulesAfterSynchronizationFailure(
        module_reaper_->TakeModulesForProcessLifetime(),
        std::move(module_cache_));
    return;
  }
  module_reaper_->ReleaseOrphansAfterSynchronization();
}

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

blas::BlasSupport* MusaExecutor::AsBlas() {
  absl::MutexLock lock(&support_mu_);
  if (blas_ != nullptr) return blas_.get();

  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::BlasFactory> factory =
      registry->GetFactory<PluginRegistry::BlasFactory>(kMusaPlatformId);
  if (!factory.ok()) {
    LOG(ERROR) << "Unable to retrieve MUSA BLAS factory: " << factory.status();
    return nullptr;
  }
  blas_.reset((*factory)(this));
  if (blas_ == nullptr) {
    LOG(ERROR) << "Unable to initialize optional muBLAS support";
  }
  return blas_.get();
}

absl::StatusOr<std::unique_ptr<Stream>> MusaExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<MusaStream> stream,
                      MusaStream::Create(this, host_callback_registry_.get(),
                                         module_reaper_.get(), priority));
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
  if (context_ == nullptr) return false;
  absl::Status status = context_->Synchronize();
  if (!status.ok()) return false;
  module_reaper_->ReleaseOrphansAfterSynchronization();
  return true;
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
    TF_ASSIGN_OR_RETURN(
        MusaDeviceProperties properties,
        MusaRuntime::Get()->GetDeviceProperties(device_ordinal));
    TF_ASSIGN_OR_RETURN(int runtime_version,
                        MusaRuntime::Get()->RuntimeVersion());
    TF_ASSIGN_OR_RETURN(int driver_version, driver.DriverVersion());
    absl::StatusOr<SemanticVersion> kernel_driver_version =
        GetMusaKernelDriverVersion();
    if (!kernel_driver_version.ok()) {
      LOG(WARNING) << "Could not query MUSA kernel driver version: "
                   << kernel_driver_version.status();
    }

    MusaDeviceVersions versions{
        .runtime_api = runtime_version,
        .driver_api = driver_version,
        .compile_time_toolkit = XLA_MUSA_TOOLKIT_VERSION,
        .kernel_mode_driver =
            kernel_driver_version.ok()
                ? std::optional<SemanticVersion>(*kernel_driver_version)
                : std::nullopt,
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
  if (module_cache_ == nullptr) {
    return absl::FailedPreconditionError(
        "MUSA executor must be initialized before loading a kernel");
  }
  if (!spec.has_musa_mubin_in_memory()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("MUSA kernel %s requires an explicit MUBIN artifact",
                        spec.kernel_name()));
  }
  if (spec.kernel_name().empty()) {
    return absl::InvalidArgumentError("MUSA kernel name must not be empty");
  }
  if (spec.arity() > std::numeric_limits<unsigned>::max()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("MUSA kernel %s arity %d exceeds the supported range",
                        spec.kernel_name(), spec.arity()));
  }

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<MusaModule> module,
      module_cache_->GetOrLoadModule(spec.musa_mubin_in_memory()->mubin_bytes));
  absl::StatusOr<MUfunction> function_status =
      module->GetFunction(spec.kernel_name());
  if (!function_status.ok()) {
    return absl::Status(
        function_status.status().code(),
        absl::StrFormat("Failed to resolve MUSA kernel %s on device %d: %s",
                        spec.kernel_name(), device_ordinal(),
                        function_status.status().message()));
  }
  MUfunction function = *function_status;

  auto kernel = std::make_unique<MusaKernel>(
      this, function, module, module_reaper_.get(), spec.arity());
  kernel->set_name(spec.kernel_name());
  TF_ASSIGN_OR_RETURN(KernelMetadata metadata, kernel->GetKernelMetadata());
  kernel->set_metadata(metadata);

  if (std::holds_alternative<KernelLoaderSpec::KernelArgsPackingFunc>(
          spec.kernel_args_packing())) {
    kernel->set_args_packing(std::get<KernelLoaderSpec::KernelArgsPackingFunc>(
        spec.kernel_args_packing()));
  } else {
    const KernelArgsPackingSpec packing_spec =
        std::get<KernelArgsPackingSpec>(spec.kernel_args_packing());
    kernel->set_args_packing(
        [packing_spec](const Kernel&, const KernelArgs& args)
            -> absl::StatusOr<std::unique_ptr<KernelArgsPackedArrayBase>> {
          const auto* packable = dynamic_cast<const PackableKernelArgs*>(&args);
          if (packable == nullptr) {
            return absl::InvalidArgumentError(
                "MUSA serializable argument packing requires packable "
                "kernel arguments");
          }
          return packing_spec.BuildArguments(packable->packed_args(),
                                             args.number_of_shared_bytes());
        });
  }
  module_reaper_->Observe(module);
  return kernel;
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
  if (module_cache_ == nullptr) {
    return absl::FailedPreconditionError(
        "MUSA executor must be initialized before symbol lookup");
  }
  if (symbol_name.empty()) {
    return absl::InvalidArgumentError("MUSA symbol name must not be empty");
  }
  TF_ASSIGN_OR_RETURN(std::shared_ptr<MusaModule> module,
                      module_cache_->LookupModule(module_handle));
  absl::StatusOr<MusaModuleGlobal> global_status =
      module->GetGlobal(symbol_name);
  if (!global_status.ok()) {
    return absl::Status(
        global_status.status().code(),
        absl::StrFormat("Failed to resolve MUSA global %s on device %d: %s",
                        symbol_name, device_ordinal(),
                        global_status.status().message()));
  }
  MusaModuleGlobal global = *global_status;
  static_assert(sizeof(MUdeviceptr) == sizeof(void*),
                "MUSA device pointers must match host pointer width");
  return DeviceAddressBase(absl::bit_cast<void*>(global.address), global.size);
}

Stream* MusaExecutor::FindAllocatedStream(void* device_stream) {
  absl::MutexLock lock(&alive_streams_mu_);
  auto it = alive_streams_.find(device_stream);
  return it == alive_streams_.end() ? nullptr : it->second;
}

}  // namespace stream_executor::musa
