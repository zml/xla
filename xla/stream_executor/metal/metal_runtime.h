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
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::metal {

struct MetalDeviceInfo {
  std::string name;
  std::string registry_id;
  uint64_t recommended_max_working_set_size = 0;
  uint64_t max_buffer_length = 0;
  uint64_t max_threads_per_threadgroup = 0;
  // Device threadgroup-memory budget ([MTLDevice maxThreadgroupMemoryLength]).
  // 32 KB on every shipping Apple GPU (M1..M4) — but query it rather than
  // assume, so the scheduler's resource model and the flash-attention nsg ramp
  // key off the real device limit instead of a hardcoded 32*1024 (which would
  // silently over-allocate threadgroup memory on a smaller-smem part). 0 if the
  // query is unavailable; callers fall back to the historical 32 KB.
  uint64_t max_threadgroup_memory_length = 0;
  bool has_unified_memory = true;
};

struct MetalKernelArgument {
  void* buffer = nullptr;
  uint64_t offset = 0;
  const void* bytes = nullptr;
  size_t bytes_size = 0;
};

// A Metal function constant set at pipeline specialization
// (newFunctionWithName:constantValues:). Lets a kernel select a specialized
// variant via [[function_constant(index)]] (e.g. llama.cpp's flash_attn_ext_vec
// NSG/NWG/has_mask). `index` is the function-constant slot; `kind` picks the
// MTLDataType (Bool/Int — the only kinds we need).
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

void* RetainObject(void* object);
void ReleaseObject(void* object);

absl::StatusOr<void*> NewSharedBuffer(void* device, uint64_t size,
                                      void** contents);
absl::StatusOr<void*> CompileLibrary(void* device, absl::string_view source);
absl::StatusOr<void*> LoadLibraryFromData(void* device,
                                          absl::Span<const uint8_t> data);
absl::StatusOr<void*> NewFunction(void* library, absl::string_view name);
// Like NewFunction but specializes the function with `constants`
// (newFunctionWithName:constantValues:). Required for kernels whose
// [[function_constant]] values have no default (they will NOT instantiate via
// plain NewFunction) — e.g. the imported llama.cpp flash_attn_ext_vec.
absl::StatusOr<void*> NewFunctionWithConstants(
    void* library, absl::string_view name,
    absl::Span<const MetalFunctionConstant> constants);
absl::StatusOr<void*> NewComputePipeline(void* device, void* function);
// One-command-buffer-per-execution batching (#62). The stream holds ONE open
// MPSCommandBuffer for the duration of an execution; every AIR kernel and every
// matmul encodes into it, so the whole step is a single command buffer —
// committed and waited once — rather than one tiny command buffer per op (which
// left the GPU idle between hundreds of submissions). The open buffer is an
// MPSCommandBuffer (a thin wrapper over MTLCommandBuffer that can split via
// commitAndContinue). Correctness rests on Metal's in-command-buffer
// encoder ordering + automatic hazard tracking of the (default-tracked) shared
// buffers — the same producer→consumer guarantee the per-cb queue order gave.

// Creates an open MPSCommandBuffer on the queue (retained, +1).
void* NewBatchCommandBuffer(void* command_queue);
// Encodes one compute kernel into the open command buffer's current underlying
// MTLCommandBuffer (a fresh compute encoder per call). Does NOT commit.
absl::Status EncodeKernel(void* batch_command_buffer, void* pipeline,
                          void* function, bool use_argument_buffer,
                          absl::Span<const MetalKernelArgument> arguments,
                          absl::string_view name, const ThreadDim& thread_dims,
                          const BlockDim& block_dims, int64_t shmem_bytes);
// Encodes a device-to-device blit copy (MTLBlitCommandEncoder copyFromBuffer:..)
// into the open command buffer — no commit, no host drain. Metal hazard-tracks
// the buffers, so the copy is ordered after prior writes / before later reads in
// the same buffer; cross-execute consumers order via the stream's shared-event
// signal. Lets the decode copy thunk (lm_head RNG-state copy.5) pipeline instead
// of draining the queue. dst/src are MTLBuffers with byte offsets.
absl::Status EncodeBlitCopy(void* batch_command_buffer, void* dst_buffer,
                            uint64_t dst_offset, void* src_buffer,
                            uint64_t src_offset, uint64_t size);
// Commits the open command buffer and returns its final committed underlying
// MTLCommandBuffer (retained, +1) to wait on / record in an event. (MPSGraph may
// split via commitAndContinue; waiting the last segment drains all prior by queue
// order.) The caller still owns and must release the batch buffer itself.
void* CommitBatchCommandBuffer(void* batch_command_buffer);

// Commits the open command buffer with a GPU completion handler that runs
// `on_complete` (exactly once) AFTER the GPU finishes executing it — WITHOUT
// blocking the host. Used by MetalStream::DoHostCallbackWithStatus to signal
// PJRT's per-execute definition event (SetStateConcrete) on real GPU completion
// while the host keeps enqueueing, giving host/GPU pipelining. Command buffers
// on one queue complete in commit order, so a buffer committed after the kernels
// observes all of them. If `batch_command_buffer` is null there is no GPU work
// to wait on and `on_complete` runs immediately. The caller still owns and must
// release the batch buffer itself (Metal retains it internally until the handler
// has run).
void CommitBatchCommandBufferWithCompletion(
    void* batch_command_buffer, absl::AnyInvocable<void() &&> on_complete);

absl::Status WaitUntilCompleted(void* command_buffer);
absl::Status SynchronizeCommandQueue(void* command_queue);
Event::Status PollCommandBufferStatus(void* command_buffer);

// === GPU-side cross-command-buffer ordering (MTLSharedEvent) ================
// One MTLSharedEvent + a monotonic value per device implements producer->consumer
// ordering across PJRT executes ON the GPU, so dependent executes need not block
// the host (recovers decode pipelining lost to the per-execute BlockHostUntilDone).
// The producer encodes a signal of value N at the END of its command buffer; the
// consumer encodes a wait for N at the START of its command buffer. The consumer's
// kernels then do not run on the GPU until the producer's command buffer signals N.
//
// HARD CONSTRAINT (verified in scratch/air-ref/2026-06-04-shared_event_order.mm):
// on a single MTLCommandQueue the SIGNALER command buffer MUST be committed BEFORE
// the WAITER's — committing a waiter ahead of its signaler deadlocks the queue
// (kIOGPUCommandBufferCallbackErrorSubmissionsIgnored). PJRT guarantees this via
// BufferSequencingEvent::WaitForEventOnStream's BlockUntilReady (the consumer host-
// waits until the producer has recorded == committed its event). A wait for an
// already-signaled value is a cheap GPU no-op, so over-emitting waits is harmless.
// Note: direct (default hazard-TRACKED) buffer access is already ordered across
// command buffers by Metal; this is needed for UNtracked access (MPSGraph's
// offset-aliased MPSNDArrays), which is the decode-race source.

// Creates an MTLSharedEvent on the device (retained, +1; initial signaledValue 0).
void* NewSharedEvent(void* device);
// Encodes a signal of `value` into the batch command buffer's CURRENT underlying
// MTLCommandBuffer (call after all work is encoded; no active encoder). No-op if
// either arg is null.
void EncodeSignalSharedEvent(void* batch_command_buffer, void* shared_event,
                             uint64_t value);
// Encodes a GPU-side wait for `value` into the batch command buffer's current
// underlying MTLCommandBuffer (call before any work is encoded; no active
// encoder). No-op if either arg is null.
void EncodeWaitForSharedEvent(void* batch_command_buffer, void* shared_event,
                              uint64_t value);
// The shared event's current signaledValue (0 if null) — for event polling.
uint64_t SharedEventSignaledValue(void* shared_event);
// Blocks the host until the shared event's signaledValue reaches `value` (the
// host-block analog of waiting a command buffer, but keyed on the per-device
// MTLSharedEvent value instead of a committed cb). No-op if shared_event is null
// or value is 0. Returns an error only if the (effectively-infinite) wait times
// out. Used by MetalEvent::Synchronize so events need not own a committed cb.
absl::Status WaitUntilSignaledValue(void* shared_event, uint64_t value);

// Creates an MTLSharedEventListener backed by a private SERIAL dispatch queue
// (retained, +1; release with ReleaseObject). Serial ⇒ registered blocks fire
// in value order, matching the in-order completion-handler contract. Returns
// nullptr if creation fails. Owned one-per-device by MetalExecutor.
void* NewSharedEventListener();
// Registers `callback` to run on the listener's queue when `shared_event`
// reaches `value` — the Metal analog of cuLaunchHostFunc: a free async enqueue
// that drives PJRT's SetStateConcrete on real GPU completion WITHOUT a per-
// execute commit. The block runs exactly once and frees the callback. Fires
// immediately if `value` has already been signaled. No-op (callback dropped) if
// listener/shared_event is null.
void NotifySharedEventListener(void* listener, void* shared_event,
                               uint64_t value,
                               absl::AnyInvocable<void() &&> callback);

// ===========================================================================
// Metal GPU profiling — per-op GPU timing for the xprof device tracer.
// ===========================================================================
// Timing is collected via an MTLCounterSampleBuffer (the "timestamp" counter
// set) sampled at compute-encoder boundaries inside EncodeComputeInto — the one
// funnel every AIR compute dispatch flows through. OFF unless
// MetalProfilingStart() has been called (by the MetalTracer ProfilerInterface,
// which the existing ZML `--profile` path drives). Times are wall-clock
// nanoseconds (UNIX epoch), anchored to the host clock so GPU events line up
// with host TraceMe events on one timeline. COVERAGE: AIR compute kernels only
// (the decode hot path); MPSGraph matmuls use their own encoders and are not
// captured.
struct MetalProfileEvent {
  std::string name;     // event label (kernel name + grid)
  std::string details;  // xprof kernel_details string (grid/block dims)
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint64_t bytes = 0;   // approx bytes touched (sum of bound-buffer lengths)
};

// Enable/disable capture (called from MetalTracer::Start/Stop). Start() resets
// all accumulated state.
void MetalProfilingStart();
void MetalProfilingStop();
bool MetalProfilingEnabled();

// Resolves the GPU timestamp samples of the just-completed step into events.
// Called by the stream right after it waits its command buffer
// (BlockHostUntilDone). `device` is the void*-bridged id<MTLDevice>.
void MetalProfilingResolveStep(void* device);

// Moves out the events collected since the last Start() (clears the buffer).
std::vector<MetalProfileEvent> MetalProfilingDrain();
// Count of ops dropped (buffer full / unsupported / invalid sample) since Start.
uint64_t MetalProfilingDroppedCount();

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_RUNTIME_H_
