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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"

namespace stream_executor::metal {

class MetalExecutor;

class MetalStream : public StreamCommon {
 public:
  explicit MetalStream(
      MetalExecutor* executor,
      std::optional<std::variant<StreamPriority, int>> priority);
  ~MetalStream() override;

  absl::Status WaitFor(Stream* other) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status WaitFor(Event* event) override;

  absl::Status Memset32(DeviceAddressBase* location, uint32_t pattern,
                        uint64_t size) override;
  absl::Status MemZero(DeviceAddressBase* location, uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* device_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& device_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* device_dst,
                      const DeviceAddressBase& device_src,
                      uint64_t size) override;

  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;
  absl::Status BlockHostUntilDone() override;
  absl::Status FlushBatchedWork() override;
  absl::Status CommitBatchedWorkNoWait() override;

  Stream::PlatformSpecificHandle platform_specific_handle() const override;

  absl::Status LaunchMetalKernel(
      const ThreadDim& thread_dims, const BlockDim& block_dims,
      const std::optional<ClusterDim>& cluster_dims, void* pipeline,
      void* function, bool use_argument_buffer, absl::string_view name,
      void** args, absl::Span<const KernelArgumentMetadata> arg_metadata,
      int64_t shmem_bytes, bool use_pdl);

  void FlushOpenBufferIfCarrying(uint64_t value);

 private:
  void EnsureOpenCommandBuffer();
  void CommitOpenBufferNoWait();

  MetalExecutor* executor_;
  void* command_buffer_ = nullptr;
  uint64_t last_signaled_value_ = 0;
  uint64_t pending_signal_high_ = 0;
  static constexpr int kMaxSignalsPerBuffer = 64;
  int signals_since_commit_ = 0;
  // Highest value already encoded as a GPU wait into the open buffer; later
  // waits for <= it are elided. Reset whenever the buffer closes: a wait in an
  // earlier buffer does not order a later one.
  uint64_t waited_value_high_ = 0;
};

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_
