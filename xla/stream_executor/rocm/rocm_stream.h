/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_STREAM_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_STREAM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/base/call_once.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"

namespace stream_executor {
namespace gpu {

class RocmStream : public StreamCommon {
 public:
  class CaptureHandle;

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

  Stream::PlatformSpecificHandle platform_specific_handle() const override {
    return {stream_handle_};
  }

  absl::StatusOr<std::unique_ptr<EventBasedTimer>> CreateEventBasedTimer(
      bool use_delay_kernel) override {
    return executor_->CreateEventBasedTimer(this, use_delay_kernel);
  }

  static absl::StatusOr<std::unique_ptr<RocmStream>> Create(
      StreamExecutor* executor,
      std::optional<std::variant<StreamPriority, int>> priority);

  ~RocmStream() override;

  hipStream_t stream_handle() const { return stream_handle_; }

  // Begins capturing on a dedicated stream into the given graph. The returned
  // handle must be destroyed before this stream.
  absl::StatusOr<CaptureHandle> BeginCapture(
      hipGraph_t graph, const hipGraphNode_t* dependencies,
      size_t num_dependencies, hipStreamCaptureMode mode);

  // RAII handle for a HIP graph capture. Call EndCapture explicitly to
  // propagate errors.
  class CaptureHandle {
   public:
    CaptureHandle() = delete;
    CaptureHandle(const CaptureHandle&) = delete;
    CaptureHandle& operator=(const CaptureHandle&) = delete;
    CaptureHandle(CaptureHandle&& other);
    CaptureHandle& operator=(CaptureHandle&& other) = delete;

    absl::Status EndCapture();

    RocmStream* capturing_stream() const { return stream_; }

    ~CaptureHandle();

   private:
    friend class RocmStream;
    CaptureHandle(RocmStream* stream, hipGraph_t graph)
        : stream_(stream), graph_(graph) {}

    RocmStream* stream_;
    hipGraph_t graph_;
  };

 private:
  RocmStream(StreamExecutor* executor, RocmEvent completed_event,
             std::optional<std::variant<StreamPriority, int>> priority,
             hipStream_t stream_handle, unsigned int creation_flags,
             int creation_priority)
      : StreamCommon(executor, priority),
        executor_(executor),
        completed_event_(std::move(completed_event)),
        stream_handle_(stream_handle),
        creation_flags_(creation_flags),
        creation_priority_(creation_priority) {}

  absl::Status RecordCompletedEvent();

  absl::Status LaunchKernel(const ThreadDim& thread_dims,
                            const BlockDim& block_dims,
                            const std::optional<ClusterDim>& cluster_dims,
                            void* function, absl::string_view name, void** args,
                            int64_t shmem_bytes, bool use_pdl) override;

  StreamExecutor* executor_;
  RocmEvent completed_event_;
  hipStream_t stream_handle_;
  unsigned int creation_flags_;
  int creation_priority_;

  absl::once_flag capture_stream_once_;
  absl::StatusOr<std::unique_ptr<Stream>> capture_stream_{
      absl::InternalError("Capture stream not initialized")};
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_STREAM_H_
