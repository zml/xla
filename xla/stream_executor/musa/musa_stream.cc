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

#include "xla/stream_executor/musa/musa_stream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_graph_arguments.h"
#include "xla/stream_executor/musa/musa_module.h"
#include "xla/stream_executor/musa/musa_module_reaper.h"
#include "xla/stream_executor/musa/musa_runtime.h"
#include "xla/stream_executor/musa/musa_timer.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

MusaStream::MusaStream(
    StreamExecutor* executor, MusaEvent completed_event,
    MusaModuleReaper* module_reaper,
    std::optional<std::variant<StreamPriority, int>> priority,
    void* stream_handle)
    : StreamCommon(executor, priority),
      executor_(executor),
      completed_event_(std::move(completed_event)),
      module_reaper_(module_reaper),
      stream_handle_(stream_handle) {}

absl::StatusOr<std::unique_ptr<MusaStream>> MusaStream::Create(
    StreamExecutor* executor, gpu::HostCallbackRegistry* host_callback_registry,
    MusaModuleReaper* module_reaper,
    std::optional<std::variant<StreamPriority, int>> priority) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  if (host_callback_registry == nullptr) {
    return absl::InvalidArgumentError(
        "MusaStream requires a host callback registry");
  }
  if (module_reaper == nullptr) {
    return absl::InvalidArgumentError(
        "MusaStream requires a module lifetime reaper");
  }
  auto completed_event = MusaEvent::Create(executor);
  if (!completed_event.ok()) return completed_event.status();
  auto stream = MusaRuntime::Get()->StreamCreate();
  if (!stream.ok()) return stream.status();
  auto result = std::unique_ptr<MusaStream>(new MusaStream(
      executor, *std::move(completed_event), module_reaper, priority, *stream));
  MusaStream* stream_ptr = result.get();
  result->callback_registry_handle_ = host_callback_registry->CreateHandle(
      /*synchronization_callback=*/
      [stream_ptr] {
        std::unique_ptr<ActivateContext> activation =
            stream_ptr->executor_->Activate();
        return MusaRuntime::Get()->StreamSynchronize(
            stream_ptr->stream_handle_);
      },
      /*status_callback=*/
      [stream_ptr] { return stream_ptr->RefreshStatus(); });
  return result;
}

MusaStream::~MusaStream() {
  // The registry synchronizes and resolves callbacks that the driver may skip
  // after an asynchronous stream error. It must be torn down while the native
  // stream and executor are still alive.
  callback_registry_handle_.reset();
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  (void)MusaRuntime::Get()->StreamSynchronize(stream_handle_);
  parent()->DeallocateStream(this);
  (void)MusaRuntime::Get()->StreamDestroy(stream_handle_);
  stream_handle_ = nullptr;
}

absl::Status MusaStream::WaitFor(Stream* other) {
  auto* other_stream = dynamic_cast<MusaStream*>(other);
  if (other_stream == nullptr) {
    return absl::InvalidArgumentError("Expected another MUSA stream.");
  }
  RETURN_IF_ERROR(other_stream->RecordCompletedEvent());
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return MusaRuntime::Get()->StreamWaitEvent(
      stream_handle_, other_stream->completed_event_.handle());
}

absl::Status MusaStream::RecordCompletedEvent() {
  return RecordEvent(&completed_event_);
}

absl::Status MusaStream::RecordEvent(Event* event) {
  auto* musa_event = dynamic_cast<MusaEvent*>(event);
  if (musa_event == nullptr) {
    return absl::InvalidArgumentError("Expected a MUSA event.");
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return MusaRuntime::Get()->EventRecord(musa_event->handle(), stream_handle_);
}

absl::Status MusaStream::WaitFor(Event* event) {
  return event->WaitForEventOnExternalStream(
      reinterpret_cast<std::intptr_t>(stream_handle_));
}

absl::Status MusaStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const void* host_src, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return MusaRuntime::Get()->MemcpyAsync(gpu_dst->opaque(), host_src, size,
                                         MusaMemcpyKind::kHostToDevice,
                                         stream_handle_);
}

absl::Status MusaStream::Memcpy(void* host_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return MusaRuntime::Get()->MemcpyAsync(host_dst, gpu_src.opaque(), size,
                                         MusaMemcpyKind::kDeviceToHost,
                                         stream_handle_);
}

absl::Status MusaStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  if (size == 0 || gpu_dst->is_null() || gpu_src.is_null()) {
    return MusaRuntime::Get()->MemcpyAsync(
        gpu_dst->opaque(), gpu_src.opaque(), size,
        MusaMemcpyKind::kDeviceToDevice, stream_handle_);
  }

  MusaDriver& driver = MusaDriver::Instance();
  MUdeviceptr destination = absl::bit_cast<MUdeviceptr>(gpu_dst->opaque());
  MUdeviceptr source = absl::bit_cast<MUdeviceptr>(gpu_src.opaque());
  absl::StatusOr<MUcontext> destination_context =
      driver.ContextForPointer(destination);
  if (!destination_context.ok()) return destination_context.status();
  absl::StatusOr<MUcontext> source_context = driver.ContextForPointer(source);
  if (!source_context.ok()) return source_context.status();

  if (*destination_context == *source_context) {
    return MusaRuntime::Get()->MemcpyAsync(
        gpu_dst->opaque(), gpu_src.opaque(), size,
        MusaMemcpyKind::kDeviceToDevice, stream_handle_);
  }
  return driver.MemcpyPeerAsync(destination, *destination_context, source,
                                *source_context, size,
                                reinterpret_cast<MUstream>(stream_handle_));
}

absl::Status MusaStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return MusaRuntime::Get()->MemsetAsync(location->opaque(), 0, size,
                                         stream_handle_);
}

absl::Status MusaStream::Memset32(DeviceAddressBase* location, uint32_t pattern,
                                  uint64_t size) {
  if (location == nullptr || location->is_null()) {
    return absl::InvalidArgumentError(
        "MUSA Memset32 location must not be null");
  }
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) != 0) {
    return absl::InvalidArgumentError(
        "MUSA Memset32 location must be 4-byte aligned");
  }
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError(
        "MUSA Memset32 size must be a multiple of 4 bytes");
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return MusaDriver::Instance().MemsetD32Async(
      absl::bit_cast<MUdeviceptr>(location->opaque()), pattern,
      size / sizeof(uint32_t), reinterpret_cast<MUstream>(stream_handle_));
}

absl::Status MusaStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  return DoHostCallbackWithStatus(std::move(callback), /*error_cb=*/nullptr);
}

absl::Status MusaStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_cb) {
  if (callback_registry_handle_ == nullptr) {
    return absl::FailedPreconditionError(
        "MUSA stream callback registry is not initialized");
  }
  auto enqueue =
      [this](
          gpu::HostCallbackRegistry::RegistryHandle::DeviceCb device_callback,
          void* data) {
        std::unique_ptr<ActivateContext> activation = executor_->Activate();
        return MusaRuntime::Get()->LaunchHostFunc(stream_handle_,
                                                  device_callback, data);
      };
  return callback_registry_handle_->AddCallback(
      std::move(callback), std::move(error_cb), std::move(enqueue));
}

absl::Status MusaStream::BlockHostUntilDone() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  absl::Status status = MusaRuntime::Get()->StreamSynchronize(stream_handle_);
  if (!status.ok() && callback_registry_handle_ != nullptr) {
    callback_registry_handle_->FailAll(status);
  }
  return status;
}

absl::Status MusaStream::RefreshStatus() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return MusaRuntime::Get()->StreamQuery(stream_handle_);
}

absl::Status MusaStream::RecordModuleUse(std::shared_ptr<MusaModule> module) {
  if (module == nullptr) {
    return absl::InvalidArgumentError("Cannot record a null MUSA module use");
  }
  {
    absl::MutexLock lock(&graph_capture_mu_);
    if (graph_capture_active_) {
      auto duplicate = std::find_if(
          graph_capture_modules_.begin(), graph_capture_modules_.end(),
          [&module](const std::shared_ptr<MusaModule>& captured) {
            return captured.get() == module.get();
          });
      if (duplicate == graph_capture_modules_.end()) {
        graph_capture_modules_.push_back(std::move(module));
      }
      return absl::OkStatus();
    }
  }
  std::shared_ptr<MusaModuleReaper::ModuleUse> use =
      module_reaper_->Track(std::move(module));
  return DoHostCallbackWithStatus(
      [use]() -> absl::Status {
        use->Complete();
        return absl::OkStatus();
      },
      [use](absl::Status) { use->Orphan(); });
}

absl::Status MusaStream::BeginGraphCaptureModuleTracking() {
  absl::MutexLock lock(&graph_capture_mu_);
  if (graph_capture_active_) {
    return absl::FailedPreconditionError(
        "MUSA stream module capture is already active");
  }
  graph_capture_active_ = true;
  graph_capture_modules_.clear();
  graph_capture_kernel_arguments_.clear();
  return absl::OkStatus();
}

absl::StatusOr<const KernelArgsPackedArrayBase*>
MusaStream::RetainGraphCaptureKernelArguments(
    const KernelArgsPackedArrayBase& arguments) {
  absl::MutexLock lock(&graph_capture_mu_);
  if (!graph_capture_active_) return &arguments;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<KernelArgsPackedVector> cloned,
                      CloneMusaGraphKernelArguments(arguments));
  std::shared_ptr<KernelArgsPackedVector> retained(std::move(cloned));
  const KernelArgsPackedArrayBase* result = retained.get();
  graph_capture_kernel_arguments_.push_back(std::move(retained));
  return result;
}

absl::StatusOr<MusaGraphCaptureResources>
MusaStream::EndGraphCaptureModuleTracking() {
  absl::MutexLock lock(&graph_capture_mu_);
  if (!graph_capture_active_) {
    return absl::FailedPreconditionError(
        "MUSA stream module capture is not active");
  }
  graph_capture_active_ = false;
  MusaGraphCaptureResources resources;
  resources.modules.swap(graph_capture_modules_);
  resources.kernel_arguments.swap(graph_capture_kernel_arguments_);
  return resources;
}

void MusaStream::AbortGraphCaptureModuleTracking() {
  absl::MutexLock lock(&graph_capture_mu_);
  graph_capture_active_ = false;
  graph_capture_modules_.clear();
  graph_capture_kernel_arguments_.clear();
}

void MusaStream::OrphanModuleUse(std::shared_ptr<MusaModule> module) {
  module_reaper_->Orphan(std::move(module));
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>>
MusaStream::CreateEventBasedTimer(bool /*use_delay_kernel*/) {
  TF_ASSIGN_OR_RETURN(MusaTimer timer, MusaTimer::Create(executor_, this));
  return std::make_unique<MusaTimer>(std::move(timer));
}

namespace {

absl::Status ValidateLaunchDimensions(const DeviceDescription& description,
                                      const ThreadDim& thread_dims,
                                      const BlockDim& block_dims,
                                      int64_t shared_memory_bytes) {
  const ThreadDim& thread_limit = description.thread_dim_limit();
  const BlockDim& block_limit = description.block_dim_limit();
  if (thread_dims.x == 0 || thread_dims.y == 0 || thread_dims.z == 0 ||
      block_dims.x == 0 || block_dims.y == 0 || block_dims.z == 0) {
    return absl::InvalidArgumentError(
        "MUSA launch dimensions must all be nonzero");
  }
  constexpr uint64_t kNativeDimensionLimit =
      std::numeric_limits<unsigned int>::max();
  if (thread_dims.x > kNativeDimensionLimit ||
      thread_dims.y > kNativeDimensionLimit ||
      thread_dims.z > kNativeDimensionLimit ||
      block_dims.x > kNativeDimensionLimit ||
      block_dims.y > kNativeDimensionLimit ||
      block_dims.z > kNativeDimensionLimit) {
    return absl::InvalidArgumentError(
        "MUSA launch dimensions exceed the native unsigned-int ABI");
  }
  if (thread_dims.x > thread_limit.x || thread_dims.y > thread_limit.y ||
      thread_dims.z > thread_limit.z) {
    return absl::InvalidArgumentError(
        absl::StrFormat("MUSA thread dimensions %s exceed device limit %s",
                        thread_dims.ToString(), thread_limit.ToString()));
  }
  if (block_dims.x > block_limit.x || block_dims.y > block_limit.y ||
      block_dims.z > block_limit.z) {
    return absl::InvalidArgumentError(
        absl::StrFormat("MUSA grid dimensions %s exceed device limit %s",
                        block_dims.ToString(), block_limit.ToString()));
  }
  if (thread_dims.x > std::numeric_limits<uint64_t>::max() / thread_dims.y ||
      thread_dims.x * thread_dims.y >
          std::numeric_limits<uint64_t>::max() / thread_dims.z ||
      thread_dims.x * thread_dims.y * thread_dims.z >
          static_cast<uint64_t>(description.threads_per_block_limit())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MUSA thread dimensions %s exceed the %d threads-per-block limit",
        thread_dims.ToString(), description.threads_per_block_limit()));
  }
  if (shared_memory_bytes < 0 ||
      shared_memory_bytes > description.shared_memory_per_block() ||
      shared_memory_bytes > std::numeric_limits<unsigned int>::max()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MUSA dynamic shared memory %d exceeds the %d-byte device limit",
        shared_memory_bytes, description.shared_memory_per_block()));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status MusaStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes, bool use_pdl) {
  if (function == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("MUSA kernel %s has a null function", name));
  }
  if (cluster_dims.has_value()) {
    return absl::UnimplementedError(
        absl::StrFormat("MUSA kernel %s does not support cluster dimensions %s",
                        name, cluster_dims->ToString()));
  }
  if (use_pdl) {
    return absl::UnimplementedError(absl::StrFormat(
        "MUSA kernel %s does not support programmatic dependent launch", name));
  }
  RETURN_IF_ERROR(ValidateLaunchDimensions(
      executor_->GetDeviceDescription(), thread_dims, block_dims, shmem_bytes));

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  absl::Status status = MusaDriver::Instance().LaunchKernel(
      static_cast<MUfunction>(function),
      static_cast<unsigned int>(block_dims.x),
      static_cast<unsigned int>(block_dims.y),
      static_cast<unsigned int>(block_dims.z),
      static_cast<unsigned int>(thread_dims.x),
      static_cast<unsigned int>(thread_dims.y),
      static_cast<unsigned int>(thread_dims.z),
      static_cast<unsigned int>(shmem_bytes),
      reinterpret_cast<MUstream>(stream_handle_), args,
      /*extra=*/nullptr);
  if (!status.ok()) {
    return absl::Status(
        status.code(),
        absl::StrFormat(
            "MUSA kernel %s launch failed; grid=%s block=%s shared=%d: %s",
            name, block_dims.ToString(), thread_dims.ToString(), shmem_bytes,
            status.message()));
  }
  return absl::OkStatus();
}

}  // namespace stream_executor::musa
