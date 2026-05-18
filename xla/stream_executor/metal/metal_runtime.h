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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_RUNTIME_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::metal {

struct MetalDeviceInfo {
  std::string name;
  std::string registry_id;
  uint64_t recommended_max_working_set_size = 0;
  uint64_t max_buffer_length = 0;
  uint64_t max_threads_per_threadgroup = 0;
  bool has_unified_memory = true;
};

struct MetalKernelArgument {
  void* buffer = nullptr;
  uint64_t offset = 0;
  const void* bytes = nullptr;
  size_t bytes_size = 0;
};

int GetDeviceCount();
absl::StatusOr<MetalDeviceInfo> GetDeviceInfo(int ordinal);
absl::StatusOr<void*> RetainDevice(int ordinal);
absl::StatusOr<void*> NewCommandQueue(void* device);

void* RetainObject(void* object);
void ReleaseObject(void* object);

absl::StatusOr<void*> NewSharedBuffer(void* device, uint64_t size,
                                      void** contents);
absl::StatusOr<void*> CompileLibrary(void* device, absl::string_view source);
absl::StatusOr<void*> LoadLibraryFromData(void* device,
                                          absl::Span<const uint8_t> data);
absl::StatusOr<void*> NewFunction(void* library, absl::string_view name);
absl::StatusOr<void*> NewComputePipeline(void* device, void* function);
absl::StatusOr<void*> Launch(void* command_queue, void* pipeline,
                             void* function, bool use_argument_buffer,
                             absl::Span<const MetalKernelArgument> arguments,
                             const ThreadDim& thread_dims,
                             const BlockDim& block_dims,
                             int64_t shmem_bytes);
absl::Status WaitUntilCompleted(void* command_buffer);
absl::Status SynchronizeCommandQueue(void* command_queue);
Event::Status PollCommandBufferStatus(void* command_buffer);

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_RUNTIME_H_
