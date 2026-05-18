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
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
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

  Stream::PlatformSpecificHandle platform_specific_handle() const override;

 private:
  absl::Status LaunchKernel(const ThreadDim& thread_dims,
                            const BlockDim& block_dims,
                            const std::optional<ClusterDim>& cluster_dims,
                            void* function, absl::string_view name,
                            void** args, int64_t shmem_bytes,
                            bool use_pdl) override;
  void SetLastCommandBuffer(void* command_buffer);

  MetalExecutor* executor_;
  void* last_command_buffer_ = nullptr;
};

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_STREAM_H_
