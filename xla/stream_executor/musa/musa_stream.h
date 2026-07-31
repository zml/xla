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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_STREAM_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_STREAM_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/host_callback_registry.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_args_packed_vector.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_module_use_tracker.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

class MusaModule;
class MusaModuleReaper;

struct MusaGraphCaptureResources {
  std::vector<std::shared_ptr<MusaModule>> modules;
  std::vector<std::shared_ptr<KernelArgsPackedVector>> kernel_arguments;
};

class MusaStream : public StreamCommon, public MusaModuleUseTracker {
 public:
  absl::Status WaitFor(Stream* other) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status WaitFor(Event* event) override;

  absl::Status Memset32(DeviceAddressBase* location, uint32_t pattern,
                        uint64_t size) override;
  absl::Status MemZero(DeviceAddressBase* location, uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* gpu_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& gpu_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* gpu_dst,
                      const DeviceAddressBase& gpu_src, uint64_t size) override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback,
      absl::AnyInvocable<void(absl::Status) &&> error_cb) override;
  absl::Status BlockHostUntilDone() override;
  absl::Status RefreshStatus() override;

  absl::Status RecordModuleUse(std::shared_ptr<MusaModule> module) override;
  void OrphanModuleUse(std::shared_ptr<MusaModule> module) override;

  // Kernel launches normally retire modules with an ordered host callback.
  // Host callbacks must not be inserted while a stream is being captured, so
  // capture transfers modules and packed arguments to the resulting command
  // buffer.
  absl::Status BeginGraphCaptureModuleTracking();
  absl::StatusOr<const KernelArgsPackedArrayBase*>
  RetainGraphCaptureKernelArguments(
      const KernelArgsPackedArrayBase& arguments) override;
  absl::StatusOr<MusaGraphCaptureResources> EndGraphCaptureModuleTracking();
  void AbortGraphCaptureModuleTracking();

  Stream::PlatformSpecificHandle platform_specific_handle() const override {
    return {stream_handle_};
  }

  absl::StatusOr<std::unique_ptr<EventBasedTimer>> CreateEventBasedTimer(
      bool use_delay_kernel) override;

  static absl::StatusOr<std::unique_ptr<MusaStream>> Create(
      StreamExecutor* executor,
      gpu::HostCallbackRegistry* host_callback_registry,
      MusaModuleReaper* module_reaper,
      std::optional<std::variant<StreamPriority, int>> priority);

  ~MusaStream() override;

  void* stream_handle() const { return stream_handle_; }

 private:
  MusaStream(StreamExecutor* executor, MusaEvent completed_event,
             MusaModuleReaper* module_reaper,
             std::optional<std::variant<StreamPriority, int>> priority,
             void* stream_handle);

  absl::Status RecordCompletedEvent();

  absl::Status LaunchKernel(const ThreadDim& thread_dims,
                            const BlockDim& block_dims,
                            const std::optional<ClusterDim>& cluster_dims,
                            void* function, absl::string_view name, void** args,
                            int64_t shmem_bytes, bool use_pdl) override;

  StreamExecutor* executor_;
  MusaEvent completed_event_;
  MusaModuleReaper* module_reaper_;
  void* stream_handle_;
  std::unique_ptr<gpu::HostCallbackRegistry::RegistryHandle>
      callback_registry_handle_;
  absl::Mutex graph_capture_mu_;
  bool graph_capture_active_ ABSL_GUARDED_BY(graph_capture_mu_) = false;
  std::vector<std::shared_ptr<MusaModule>> graph_capture_modules_
      ABSL_GUARDED_BY(graph_capture_mu_);
  std::vector<std::shared_ptr<KernelArgsPackedVector>>
      graph_capture_kernel_arguments_ ABSL_GUARDED_BY(graph_capture_mu_);
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_STREAM_H_
