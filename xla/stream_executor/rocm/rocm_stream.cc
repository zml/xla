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

#include "xla/stream_executor/rocm/rocm_stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/casts.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_context.h"
#include "xla/stream_executor/rocm/rocm_kernel.h"
#include "xla/stream_executor/rocm/rocm_performance_counters.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {

ScopedRocmActivateContext ActivateRocm(StreamExecutor* executor) {
  return ScopedRocmActivateContext(executor->device_ordinal());
}

// ---------------------------------------------------------------------------
// Process-level HIP stream handle cache
//
// hipStreamCreate on ROCm is expensive (~100 ms/stream). Instead of calling
// hipStreamDestroy in RocmStream::~RocmStream() and hipStreamCreate in
// RocmStream::Create(), idle handles are kept in this cache and reused.
//
// Cache key: (device_ordinal, creation_flags, creation_priority_int). Normal
// retirement carries the immutable creation parameters with the wrapper; the
// synchronous fallback queries them from the live stream before insertion.
//
// Safety invariants:
//   - An active handle first enters a retirement queue with a tail event and is
//     inserted into the ready cache only after hipEventQuery reports complete.
//   - A stream or retirement event in an error state is destroyed rather than
//     cached.
//   - The GPU context is activated (executor->Activate()) for all HIP calls.
//
// Intentional no-destructor: the singleton holds at most one vector of handles
// per (device, flags, priority) for the process lifetime.  absl::NoDestructor
// avoids running the destructor at program exit (which would call
// hipStreamDestroy after the driver may already be torn down).
//
// Thread safety: mu guards all map accesses.
struct HipStreamHandleCache {
  using Key = std::tuple<int, unsigned int, int>;

  struct RetiredHandle {
    hipStream_t stream;
    hipEvent_t completion_event;
  };

  absl::Mutex mu;
  // Key: (device_ordinal, flags, priority_int)
  absl::btree_map<Key, std::vector<hipStream_t>> handles ABSL_GUARDED_BY(mu);
  // Handles stay here until a tail event proves that all previously enqueued
  // work, including host callbacks, has completed.
  absl::btree_map<Key, std::vector<RetiredHandle>> retired
      ABSL_GUARDED_BY(mu);
};

HipStreamHandleCache& GetHipStreamHandleCache() {
  static absl::NoDestructor<HipStreamHandleCache> cache;
  return *cache;
}

void CacheReadyStream(StreamExecutor* executor, hipStream_t stream,
                      unsigned int flags, int priority) {
  auto& cache = GetHipStreamHandleCache();
  absl::MutexLock lock(&cache.mu);
  auto key = std::make_tuple(executor->device_ordinal(), flags, priority);
  cache.handles[key].push_back(stream);
  VLOG(2) << "cached ready HIP stream " << stream << " for device "
          << executor->device_ordinal() << " flags=" << flags
          << " priority=" << priority;
}

void DestroyStreamWithoutCaching(StreamExecutor* executor,
                                 hipStream_t stream) {
  if (stream == nullptr) {
    return;
  }
  auto activation = ActivateRocm(executor);
  hipError_t result = hipStreamDestroy(stream);
  if (result != hipSuccess) {
    LOG(ERROR) << "failed to destroy ROCM stream for device "
               << executor->device_ordinal() << ": " << ToString(result);
  }
}

void HarvestRetiredStreams(StreamExecutor* executor,
                           const HipStreamHandleCache::Key& key) {
  auto activation = ActivateRocm(executor);
  auto& cache = GetHipStreamHandleCache();
  absl::MutexLock lock(&cache.mu);
  auto it = cache.retired.find(key);
  if (it == cache.retired.end()) {
    return;
  }

  auto& retired = it->second;
  for (size_t i = 0; i < retired.size();) {
    hipError_t result = hipEventQuery(retired[i].completion_event);
    if (result == hipErrorNotReady) {
      ++i;
      continue;
    }

    hipStream_t stream = retired[i].stream;
    hipEvent_t event = retired[i].completion_event;
    retired[i] = retired.back();
    retired.pop_back();

    hipError_t event_destroy_result = hipEventDestroy(event);
    if (event_destroy_result != hipSuccess) {
      LOG(ERROR) << "failed to destroy HIP stream retirement event: "
                 << ToString(event_destroy_result);
    }

    if (result == hipSuccess) {
      cache.handles[key].push_back(stream);
      VLOG(2) << "harvested retired HIP stream " << stream << " for device "
              << std::get<0>(key);
      continue;
    }

    LOG(WARNING) << "retired HIP stream completed with an error: "
                 << ToString(result) << " — destroying instead of caching";
    hipError_t stream_destroy_result = hipStreamDestroy(stream);
    if (stream_destroy_result != hipSuccess) {
      LOG(ERROR) << "failed to destroy errored HIP stream for device "
                 << std::get<0>(key) << ": "
                 << ToString(stream_destroy_result);
    }
  }

  if (retired.empty()) {
    cache.retired.erase(it);
  }
}

absl::StatusOr<hipStream_t> CreateStream(StreamExecutor* executor,
                                         int priority) {
  // StreamExecutor expresses dependencies with events and does not rely on
  // legacy null-stream synchronization. Match CUDA's nonblocking stream
  // semantics and include the flag in the cache key.
  constexpr unsigned int kFlags = hipStreamNonBlocking;
  auto key = std::make_tuple(executor->device_ordinal(), kFlags, priority);

  // Poll tail events before consulting the ready cache. This is nonblocking;
  // a handle with outstanding work remains retired and cannot be reused.
  HarvestRetiredStreams(executor, key);

  // Check the cache for an idle handle with matching (device, flags, priority).
  {
    auto& cache = GetHipStreamHandleCache();
    absl::MutexLock lock(&cache.mu);
    auto it = cache.handles.find(key);
    if (it != cache.handles.end() && !it->second.empty()) {
      hipStream_t h = it->second.back();
      it->second.pop_back();
      VLOG(2) << "Reusing cached HIP stream " << h << " for device "
              << executor->device_ordinal() << " flags=" << kFlags
              << " priority=" << priority;
      return h;
    }
  }

  // Cold path: create a new HIP stream.
  auto activation = ActivateRocm(executor);
  hipStream_t stream;
  if (priority == 0) {
    ABSL_RETURN_IF_ERROR(ToStatus(hipStreamCreateWithFlags(&stream, kFlags),
                                  "Failed to create stream"));
  } else {
    ABSL_RETURN_IF_ERROR(ToStatus(
        hipStreamCreateWithPriority(&stream, kFlags, priority),
        "Failed to create stream"));
  }

  VLOG(2) << "successfully created stream " << stream << " for device "
          << executor->device_ordinal() << " on thread";
  return stream;
}

void RetireStream(StreamExecutor* executor, hipStream_t stream,
                  hipEvent_t completion_event, unsigned int flags,
                  int priority) {
  constexpr size_t kMaxRetiredStreamsPerKey = 256;
  auto key = std::make_tuple(executor->device_ordinal(), flags, priority);

  HarvestRetiredStreams(executor, key);

  {
    auto& cache = GetHipStreamHandleCache();
    absl::MutexLock lock(&cache.mu);
    auto& retired = cache.retired[key];
    if (retired.size() < kMaxRetiredStreamsPerKey) {
      retired.push_back({stream, completion_event});
      VLOG(2) << "retired active HIP stream " << stream << " for device "
              << executor->device_ordinal();
      return;
    }
  }

  // Keep the process-level queue bounded. This path is expected only during
  // extreme stream churn; wait for this one handle rather than accumulating
  // unbounded raw HIP resources.
  auto activation = ActivateRocm(executor);
  hipError_t synchronize_result = hipEventSynchronize(completion_event);
  hipError_t event_destroy_result = hipEventDestroy(completion_event);
  if (event_destroy_result != hipSuccess) {
    LOG(ERROR) << "failed to destroy HIP stream retirement event: "
               << ToString(event_destroy_result);
  }

  if (synchronize_result != hipSuccess) {
    LOG(WARNING) << "failed to synchronize a saturated retired HIP stream: "
                 << ToString(synchronize_result)
                 << " — destroying instead of caching";
    hipError_t destroy_result = hipStreamDestroy(stream);
    if (destroy_result != hipSuccess) {
      LOG(ERROR) << "failed to destroy HIP stream for device "
                 << executor->device_ordinal() << ": "
                 << ToString(destroy_result);
    }
    return;
  }

  auto& cache = GetHipStreamHandleCache();
  absl::MutexLock lock(&cache.mu);
  cache.handles[key].push_back(stream);
}

absl::Status RecordEvent(StreamExecutor* executor, hipEvent_t event,
                         hipStream_t stream) {
  auto activation = ActivateRocm(executor);
  IncrementRocmPerformanceCounter(RocmPerformanceCounter::kEventRecord);
  hipError_t res = hipEventRecord(event, stream);
  switch (res) {
    case hipSuccess:
      return absl::OkStatus();
    case hipErrorDeinitialized:
    case hipErrorNotInitialized:
      return absl::FailedPreconditionError(
          absl::StrFormat("error recording ROCM event on stream %p: %s", stream,
                          ToString(res).c_str()));
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("error recording ROCM event on stream %p: %s", stream,
                          ToString(res).c_str()));
  }
}

absl::Status WaitStreamOnEvent(StreamExecutor* executor, hipStream_t stream,
                               hipEvent_t event) {
  auto activation = ActivateRocm(executor);
  IncrementRocmPerformanceCounter(RocmPerformanceCounter::kEventWait);
  ABSL_RETURN_IF_ERROR(ToStatus(hipStreamWaitEvent(stream, event, 0 /* = flags */),
                           "could not wait stream on event"));
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2H(StreamExecutor* executor, void* host_dst,
                                   hipDeviceptr_t gpu_src, uint64_t size,
                                   hipStream_t stream) {
  auto activation = ActivateRocm(executor);
  ABSL_RETURN_IF_ERROR(ToStatus(
      hipMemcpyDtoHAsync(host_dst, gpu_src, size, stream),
      absl::StrFormat(
          "failed to enqueue async memcpy from device to host: host dst: %p; "
          "Gpu src: %p; size: %llu=0x%llx",
          host_dst, absl::bit_cast<void*>(gpu_src), size, size)));

  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream
          << " device: " << executor->device_ordinal();
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyH2D(StreamExecutor* executor,
                                   hipDeviceptr_t gpu_dst, const void* host_src,
                                   uint64_t size, hipStream_t stream) {
  auto activation = ActivateRocm(executor);
  ABSL_RETURN_IF_ERROR(ToStatus(
      hipMemcpyHtoDAsync(gpu_dst, const_cast<void*>(host_src), size, stream),
      absl::StrFormat(
          "failed to enqueue async memcpy from host to device: Gpu dst: %p; "
          "host src: %p; size: %llu=0x%llx",
          absl::bit_cast<void*>(gpu_dst), host_src, size, size)));

  VLOG(2) << "successfully enqueued async memcpy h2d of " << size
          << " bytes from " << host_src << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream
          << " device: " << executor->device_ordinal();
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2D(StreamExecutor* executor,
                                   hipDeviceptr_t gpu_dst,
                                   hipDeviceptr_t gpu_src, uint64_t size,
                                   hipStream_t stream) {
  auto activation = ActivateRocm(executor);
  ABSL_RETURN_IF_ERROR(ToStatus(
      hipMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream),
      absl::StrFormat("failed to enqueue async memcpy from device to device: "
                      "Gpu dst: %p ; Gpu src: %p ; size: %llu=0x%llx",
                      absl::bit_cast<void*>(gpu_dst),
                      absl::bit_cast<void*>(gpu_src), size, size)));

  VLOG(2) << "successfully enqueued async memcpy d2d of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream
          << " device: " << executor->device_ordinal();
  return absl::OkStatus();
}

absl::Status SynchronizeStream(StreamExecutor* executor, hipStream_t stream) {
  auto activation = ActivateRocm(executor);
  ABSL_RETURN_IF_ERROR(ToStatus(hipStreamSynchronize(stream),
                           "Could not synchronize on ROCM stream"));
  VLOG(2) << "successfully synchronized stream " << stream << " on device "
          << executor->device_ordinal();
  return absl::OkStatus();
}

absl::StatusOr<bool> StreamIsCapturing(StreamExecutor* executor,
                                       hipStream_t stream) {
  auto activation = ActivateRocm(executor);
  hipStreamCaptureStatus status;
  ABSL_RETURN_IF_ERROR(
      ToStatus(hipStreamIsCapturing(stream, &status),
               "Failed to check HIP stream capture status"));
  return status == hipStreamCaptureStatusActive;
}

}  // namespace

RocmStream::CaptureHandle::CaptureHandle(CaptureHandle&& other)
    : stream_(std::exchange(other.stream_, nullptr)),
      graph_(std::exchange(other.graph_, nullptr)) {}

absl::Status RocmStream::CaptureHandle::EndCapture() {
  if (stream_ == nullptr || graph_ == nullptr) {
    return absl::OkStatus();
  }

  RocmStream* stream = std::exchange(stream_, nullptr);
  hipGraph_t graph = std::exchange(graph_, nullptr);
  hipGraph_t captured_graph = nullptr;
  ABSL_RETURN_IF_ERROR(ToStatus(
      hipStreamEndCapture(stream->stream_handle_, &captured_graph),
      "Failed to end HIP stream capture"));
  if (captured_graph != graph) {
    return absl::InternalError(
        "HIP stream capture should update the graph passed to BeginCapture");
  }
  return absl::OkStatus();
}

RocmStream::CaptureHandle::~CaptureHandle() { EndCapture().IgnoreError(); }

absl::StatusOr<RocmStream::CaptureHandle> RocmStream::BeginCapture(
    hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t num_dependencies, hipStreamCaptureMode mode) {
  absl::call_once(capture_stream_once_, [this]() {
    capture_stream_ = parent()->CreateStream(priority());
  });
  if (!capture_stream_.ok()) {
    return capture_stream_.status();
  }

  RocmStream* capture_stream =
      static_cast<RocmStream*>(capture_stream_->get());
  ABSL_ASSIGN_OR_RETURN(
      bool is_capturing,
      StreamIsCapturing(executor_, capture_stream->stream_handle_));
  if (is_capturing) {
    return absl::FailedPreconditionError(
        "HIP capture stream is already capturing");
  }

  ABSL_RETURN_IF_ERROR(ToStatus(
      hipStreamBeginCaptureToGraph(
          capture_stream->stream_handle_, graph, dependencies,
          /*dependencyData=*/nullptr, num_dependencies, mode),
      "Failed to begin HIP stream capture to graph"));
  return CaptureHandle(capture_stream, graph);
}

absl::StatusOr<std::unique_ptr<RocmStream>> RocmStream::Create(
    StreamExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  int stream_priority = [&]() {
    if (priority.has_value() && std::holds_alternative<int>(priority.value())) {
      return std::get<int>(priority.value());
    }
    return executor->GetGpuStreamPriority(
        std::get<StreamPriority>(priority.value_or(StreamPriority::Default)));
  }();
  ABSL_ASSIGN_OR_RETURN(auto stream_handle, CreateStream(executor, stream_priority));

  ABSL_ASSIGN_OR_RETURN(auto completed_event,
                   RocmEvent::Create(executor,
                                     /*allow_timing=*/false));

  return std::unique_ptr<RocmStream>(new RocmStream(
      executor, std::move(completed_event), priority, stream_handle,
      hipStreamNonBlocking, stream_priority));
}

absl::Status RocmStream::WaitFor(Stream* other) {
  RocmStream* other_stream = static_cast<RocmStream*>(other);

  ABSL_RETURN_IF_ERROR(other_stream->RecordCompletedEvent());

  return WaitStreamOnEvent(executor_, stream_handle_,
                           other_stream->completed_event_.GetHandle());
}

absl::Status RocmStream::RecordEvent(Event* event) {
  return stream_executor::gpu::RecordEvent(
      executor_, static_cast<RocmEvent*>(event)->GetHandle(), stream_handle_);
}

absl::Status RocmStream::WaitFor(Event* event) {
  return WaitStreamOnEvent(executor_, stream_handle_,
                           static_cast<RocmEvent*>(event)->GetHandle());
}

absl::Status RocmStream::RecordCompletedEvent() {
  return RecordEvent(&completed_event_);
}

namespace {
void DestroyStream(StreamExecutor* executor, hipStream_t stream) {
  if (stream == nullptr) {
    return;
  }

  // Activate the device context for all HIP calls below.
  auto activation = ActivateRocm(executor);

  // Verify the stream is fully idle before caching. This is the synchronous
  // fallback for poisoned streams and retirement-event failures; an error here
  // destroys the handle rather than poisoning the ready cache.
  hipError_t query_res = hipStreamQuery(stream);
  if (query_res != hipSuccess) {
    LOG(WARNING) << "stream not idle on destroy: " << ToString(query_res)
                 << " — destroying instead of caching";
    hipError_t res = hipStreamDestroy(stream);
    if (res != hipSuccess) {
      LOG(ERROR) << "failed to destroy ROCM stream for device "
                 << executor->device_ordinal() << ": " << ToString(res);
    }
    return;
  }

  // Query the stream's creation flags and priority so they can be used as the
  // cache key. This guarantees a retrieved handle always matches the exact
  // (flags, priority) the new stream would have been created with — even if
  // XLA is changed to use hipStreamNonBlocking or non-default priorities.
  unsigned int flags = 0;
  int stream_priority = 0;
  hipError_t flags_res = hipStreamGetFlags(stream, &flags);
  if (flags_res != hipSuccess) {
    LOG(WARNING) << "hipStreamGetFlags failed: " << ToString(flags_res)
                 << " — destroying stream " << stream << " instead of caching";
    hipError_t destroy_res = hipStreamDestroy(stream);
    if (destroy_res != hipSuccess) {
      LOG(ERROR) << "failed to destroy ROCM stream for device "
                 << executor->device_ordinal() << ": " << ToString(destroy_res);
    }
    return;
  }
  hipError_t prio_res = hipStreamGetPriority(stream, &stream_priority);
  if (prio_res != hipSuccess) {
    LOG(WARNING) << "hipStreamGetPriority failed: " << ToString(prio_res)
                 << " — destroying stream " << stream << " instead of caching";
    hipError_t destroy_res = hipStreamDestroy(stream);
    if (destroy_res != hipSuccess) {
      LOG(ERROR) << "failed to destroy ROCM stream for device "
                 << executor->device_ordinal() << ": " << ToString(destroy_res);
    }
    return;
  }

  // Insert the verified, idle handle into the cache for reuse.
  auto& cache = GetHipStreamHandleCache();
  absl::MutexLock lock(&cache.mu);
  auto key =
      std::make_tuple(executor->device_ordinal(), flags, stream_priority);
  cache.handles[key].push_back(stream);
  VLOG(2) << "cached HIP stream " << stream << " for device "
          << executor->device_ordinal() << " flags=" << flags
          << " priority=" << stream_priority;
}
}  // namespace

RocmStream::~RocmStream() {
  // Most streams are already idle at wrapper destruction. Query first because
  // it is cheaper than either synchronizing or recording a system-scope event.
  hipError_t query_result = hipErrorUnknown;
  if (ok()) {
    auto activation = ActivateRocm(executor_);
    query_result = hipStreamQuery(stream_handle_);
  }

  if (query_result == hipSuccess) {
    executor_->DeallocateStream(this);
    CacheReadyStream(executor_, std::exchange(stream_handle_, nullptr),
                     creation_flags_, creation_priority_);
    return;
  }

  if (!ok() || query_result != hipErrorNotReady) {
    executor_->DeallocateStream(this);
    DestroyStreamWithoutCaching(executor_,
                                std::exchange(stream_handle_, nullptr));
    return;
  }

  // An active stream gets a tail event so its wrapper can be destroyed without
  // blocking. Event completion proves all earlier device work and callbacks
  // are finished before the handle is returned to the ready cache.
  absl::Status retirement_status = RecordCompletedEvent();
  executor_->DeallocateStream(this);

  if (retirement_status.ok()) {
    RetireStream(executor_, std::exchange(stream_handle_, nullptr),
                 completed_event_.ReleaseHandle(), creation_flags_,
                 creation_priority_);
    return;
  }

  // Preserve the synchronous correctness fallback for poisoned streams and
  // event-record failures. DestroyStream will cache only if synchronization
  // and its final idle query both succeed.
  BlockHostUntilDone().IgnoreError();
  DestroyStream(executor_, std::exchange(stream_handle_, nullptr));
}

absl::Status RocmStream::Memset32(DeviceAddressBase* location, uint32_t pattern,
                                  uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) != 0) {
    return absl::InvalidArgumentError("location must be 4 byte aligned.");
  }
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("size must be a multiple of 4 bytes.");
  }
  return ToStatus(
      hipMemsetD32Async(location->opaque(), pattern, size / 4, stream_handle_),
      "Failed to memset memory");
}

absl::Status RocmStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return Memset32(location, 0x0, size);
  } else {
    auto activation = ActivateRocm(executor_);
    return ToStatus(
        hipMemsetAsync(location->opaque(), 0x0, size, stream_handle_),
        "Failed to enqueue async memset operation");
  }
}

absl::Status RocmStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2D(
      executor_, absl::bit_cast<hipDeviceptr_t>(gpu_dst->opaque()),
      absl::bit_cast<hipDeviceptr_t>(gpu_src.opaque()), size, stream_handle_);
}

absl::Status RocmStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const void* host_src, uint64_t size) {
  return AsynchronousMemcpyH2D(
      executor_, absl::bit_cast<hipDeviceptr_t>(gpu_dst->opaque()), host_src,
      size, stream_handle_);
}

absl::Status RocmStream::Memcpy(void* host_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2H(executor_, host_dst,
                               absl::bit_cast<hipDeviceptr_t>(gpu_src.opaque()),
                               size, stream_handle_);
}

namespace {
struct RocmHostCallbackState {
  absl::AnyInvocable<absl::Status() &&> callback;
  absl::AnyInvocable<void(absl::Status) &&> error_callback;
};

void ReportHostCallbackError(
    absl::AnyInvocable<void(absl::Status) &&> error_callback,
    absl::Status status) {
  if (error_callback) {
    std::move(error_callback)(std::move(status));
  } else {
    LOG(WARNING) << "Host callback failed: " << status;
  }
}

void InternalHostCallback(hipStream_t /*stream*/, hipError_t stream_status,
                          void* data) {
  std::unique_ptr<RocmHostCallbackState> state(
      static_cast<RocmHostCallbackState*>(data));
  if (stream_status != hipSuccess) {
    ReportHostCallbackError(
        std::move(state->error_callback),
        ToStatus(stream_status, "HIP stream failed before host callback"));
    return;
  }

  absl::Status status = std::move(state->callback)();
  if (!status.ok()) {
    ReportHostCallbackError(std::move(state->error_callback),
                            std::move(status));
  }
}
}  // namespace

absl::Status RocmStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  return DoHostCallbackWithStatus(std::move(callback), nullptr);
}

absl::Status RocmStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_cb) {
  auto* state =
      new RocmHostCallbackState{std::move(callback), std::move(error_cb)};
  hipError_t result =
      hipStreamAddCallback(stream_handle_, InternalHostCallback, state, 0);
  if (result == hipSuccess) {
    return absl::OkStatus();
  }

  absl::Status status = ToStatus(result, "unable to add host callback");
  ReportHostCallbackError(std::move(state->error_callback), status);
  delete state;
  return status;
}

namespace {
absl::Status LaunchRocmKernel(
    StreamExecutor* executor, absl::string_view kernel_name,
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes, hipStream_t stream,
    void** kernel_params, void** extra) {
  auto activation = ActivateRocm(executor);
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << " smem: " << shared_mem_bytes
          << " func: " << (const void*)function;

  IncrementRocmPerformanceCounter(RocmPerformanceCounter::kKernelLaunch);
  auto res = hipModuleLaunchKernel(
      function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
      block_dim_z, shared_mem_bytes, stream, kernel_params, extra);
  ABSL_RETURN_IF_ERROR(ToStatus(
      res, absl::StrCat("Failed to launch ROCm kernel: ", kernel_name,
                        "; grid: ", grid_dim_x, "x", grid_dim_y, "x",
                        grid_dim_z, "; block: ", block_dim_x, "x", block_dim_y,
                        "x", block_dim_z, "; shared_mem: ", shared_mem_bytes)));

  VLOG(2) << "successfully launched kernel";
  return absl::OkStatus();
}

absl::Status LaunchRocmKernel(
    StreamExecutor* executor, absl::string_view kernel_name,
    hipFunction_t function, unsigned int cluster_dim_x,
    unsigned int cluster_dim_y, unsigned int cluster_dim_z,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes, hipStream_t stream,
    void** kernel_params, void** extra) {
  if (cluster_dim_x != 1 || cluster_dim_y != 1 || cluster_dim_z != 1)
    return absl::UnimplementedError("Not implemented for ROCm");
  return LaunchRocmKernel(executor, kernel_name, function, grid_dim_x,
                          grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
                          block_dim_z, shared_mem_bytes, stream, kernel_params,
                          extra);
}

}  // namespace

absl::Status RocmStream::BlockHostUntilDone() {
  return SynchronizeStream(executor_, stream_handle_);
}

absl::Status RocmStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes, bool use_pdl) {
  if (cluster_dims.has_value()) {
    return LaunchRocmKernel(
        executor_, name, static_cast<hipFunction_t>(function), cluster_dims->x,
        cluster_dims->y, cluster_dims->z, block_dims.x, block_dims.y,
        block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z, shmem_bytes,
        stream_handle_, args,
        /*extra=*/nullptr);
  } else {
    return LaunchRocmKernel(
        executor_, name, static_cast<hipFunction_t>(function), block_dims.x,
        block_dims.y, block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
        shmem_bytes, stream_handle_, args,
        /*extra=*/nullptr);
  }
}

}  // namespace stream_executor::gpu
