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
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::metal {

struct MetalDeviceInfo {
  std::string name;
  std::string architecture;
  std::string registry_id;
  uint64_t recommended_max_working_set_size = 0;
  uint64_t max_buffer_length = 0;
  uint64_t max_threads_per_threadgroup = 0;
  uint64_t max_threadgroup_memory_length = 0;
  uint64_t gpu_core_count = 0;
  bool has_unified_memory = true;
};

struct MetalKernelArgument {
  enum class Kind : uint8_t { kBuffer, kBytes };

  Kind kind = Kind::kBytes;
  void* buffer = nullptr;
  uint64_t offset = 0;
  const void* bytes = nullptr;
  size_t bytes_size = 0;
};

struct MetalFunctionConstant {
  enum class Kind { kBool, kInt };
  uint32_t index;
  Kind kind;
  int64_t value;  // kBool: 0/1; kInt: the int value.
};

int GetDeviceCount();
absl::StatusOr<MetalDeviceInfo> GetDeviceInfo(int ordinal);
absl::StatusOr<void*> RetainDevice(int ordinal);
absl::StatusOr<void*> NewCommandQueue(void* device);

absl::StatusOr<void*> NewResidencySet(void* device);
void ResidencySetAddAllocation(void* residency_set, void* buffer);
void ResidencySetRemoveAllocation(void* residency_set, void* buffer);
void ResidencySetCommit(void* residency_set);
void ResidencySetRequestResidency(void* residency_set);
void ResidencySetEndResidency(void* residency_set);
uint64_t ResidencySetAllocatedSize(void* residency_set);
uint64_t BufferAllocatedSize(void* buffer);
void CommandQueueAddResidencySet(void* queue, void* residency_set);
uint64_t RecommendedMaxWorkingSetSize(void* device);

void ReleaseObject(void* object);

absl::StatusOr<void*> NewSharedBuffer(void* device, uint64_t size,
                                      void** contents);
absl::StatusOr<void*> CompileLibrary(void* device, absl::string_view source);
absl::StatusOr<void*> LoadLibraryFromData(void* device,
                                          absl::Span<const uint8_t> data);
absl::StatusOr<void*> NewFunction(void* library, absl::string_view name);
absl::StatusOr<void*> NewFunctionWithConstants(
    void* library, absl::string_view name,
    absl::Span<const MetalFunctionConstant> constants);
absl::StatusOr<void*> NewComputePipeline(void* device, void* function);

void* NewBatchCommandBuffer(void* command_queue);
absl::Status EncodeKernel(void* batch_command_buffer, void* pipeline,
                          void* function, bool use_argument_buffer,
                          absl::Span<const MetalKernelArgument> arguments,
                          absl::string_view name, const ThreadDim& thread_dims,
                          const BlockDim& block_dims, int64_t shmem_bytes);
absl::Status EncodeBlitCopy(void* batch_command_buffer, void* dst_buffer,
                            uint64_t dst_offset, void* src_buffer,
                            uint64_t src_offset, uint64_t size);
absl::Status EncodeBlitFill(void* batch_command_buffer, void* buffer,
                            uint64_t offset, uint64_t size, uint8_t value);
void* CommitBatchCommandBuffer(void* batch_command_buffer);

void CommitBatchCommandBufferWithCompletion(
    void* batch_command_buffer, absl::AnyInvocable<void() &&> on_complete);

absl::Status WaitUntilCompleted(void* command_buffer);
absl::Status SynchronizeCommandQueue(void* command_queue);

// On one command queue the signaling command buffer must be committed before
// the waiting one, or the queue deadlocks.

void* NewSharedEvent(void* device);
void EncodeSignalSharedEvent(void* batch_command_buffer, void* shared_event,
                             uint64_t value);
void EncodeWaitForSharedEvent(void* batch_command_buffer, void* shared_event,
                              uint64_t value);
uint64_t SharedEventSignaledValue(void* shared_event);
absl::Status WaitUntilSignaledValue(void* shared_event, uint64_t value);

void* NewSharedEventListener();
void NotifySharedEventListener(void* listener, void* shared_event,
                               uint64_t value,
                               absl::AnyInvocable<void() &&> callback);

struct MetalProfileEvent {
  std::string name;     // event label (kernel name + grid)
  std::string details;  // xprof kernel_details string (grid/block dims)
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint64_t bytes = 0;   // approx bytes touched (sum of bound-buffer lengths)
};

void MetalProfilingStart();
void MetalProfilingStop();
bool MetalProfilingEnabled();

void MetalProfilingResolveStep(void* device);

std::vector<MetalProfileEvent> MetalProfilingDrain();
uint64_t MetalProfilingDroppedCount();

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_RUNTIME_H_
