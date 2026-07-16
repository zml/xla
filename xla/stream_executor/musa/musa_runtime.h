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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_RUNTIME_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "musa_runtime_api.h"
#include "xla/stream_executor/musa/musa_device_properties.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"

namespace stream_executor::musa {

enum class MusaMemcpyKind : int {
  kHostToDevice = 1,
  kDeviceToHost = 2,
  kDeviceToDevice = 3,
};

class MusaRuntime {
 public:
  static MusaRuntime* Get();

  // Creates an isolated runtime instance backed by an injectable symbol loader.
  // Production callers must use Get(); this factory exists for deterministic
  // loader and concurrency tests and does not alter singleton state.
  static std::unique_ptr<MusaRuntime> CreateForTesting(
      std::unique_ptr<internal::MusaSymbolLoader> loader);

  ~MusaRuntime();

  absl::Status Init();
  bool IsLoaded();

  absl::StatusOr<int> GetDeviceCount();
  absl::StatusOr<MusaDeviceProperties> GetDeviceProperties(int device_ordinal);
  absl::Status SetDevice(int device_ordinal);
  absl::Status DeviceSynchronize();

  absl::StatusOr<void*> Malloc(uint64_t bytes);
  absl::Status Free(void* ptr);
  absl::StatusOr<void*> HostAlloc(uint64_t bytes);
  absl::Status FreeHost(void* ptr);
  absl::Status Memcpy(void* dst, const void* src, uint64_t bytes,
                      MusaMemcpyKind kind);
  absl::Status MemcpyAsync(void* dst, const void* src, uint64_t bytes,
                           MusaMemcpyKind kind, void* stream);
  absl::Status MemsetAsync(void* dst, int value, uint64_t bytes, void* stream);
  absl::Status MemGetInfo(size_t* free_bytes, size_t* total_bytes) const;

  absl::StatusOr<void*> StreamCreate();
  absl::Status StreamDestroy(void* stream);
  absl::Status StreamSynchronize(void* stream);
  absl::Status StreamWaitEvent(void* stream, void* event);

  absl::StatusOr<void*> EventCreate();
  absl::Status EventDestroy(void* event);
  absl::Status EventRecord(void* event, void* stream);
  absl::Status EventSynchronize(void* event);
  int EventQuery(void* event);

  absl::StatusOr<int> RuntimeVersion();

 private:
  struct FunctionTable;

  explicit MusaRuntime(std::unique_ptr<internal::MusaSymbolLoader> loader);

  absl::Status Load() const;
  absl::Status Initialize() const;
  const char* ErrorString(musaError_t result) const;
  absl::StatusOr<int> GetDeviceAttribute(int device_ordinal, int attribute,
                                         absl::string_view name) const;

  using MusaGetDeviceCountFn = musaError_t(MUSARTAPI*)(int*);
  using MusaGetDevicePropertiesFn = musaError_t(MUSARTAPI*)(musaDeviceProp*,
                                                            int);
  using MusaDeviceGetAttributeFn = musaError_t(MUSARTAPI*)(int*, musaDeviceAttr,
                                                           int);
  using MusaDeviceGetPciBusIdFn = musaError_t(MUSARTAPI*)(char*, int, int);
  using MusaSetDeviceFn = musaError_t(MUSARTAPI*)(int);
  using MusaDeviceSynchronizeFn = musaError_t(MUSARTAPI*)();
  using MusaMallocFn = musaError_t(MUSARTAPI*)(void**, size_t);
  using MusaFreeFn = musaError_t(MUSARTAPI*)(void*);
  using MusaHostAllocFn = musaError_t(MUSARTAPI*)(void**, size_t, unsigned int);
  using MusaFreeHostFn = musaError_t(MUSARTAPI*)(void*);
  using MusaMemcpyFn = musaError_t(MUSARTAPI*)(void*, const void*, size_t,
                                               ::musaMemcpyKind);
  using MusaMemcpyAsyncFn = musaError_t(MUSARTAPI*)(void*, const void*, size_t,
                                                    ::musaMemcpyKind,
                                                    musaStream_t);
  using MusaMemsetAsyncFn = musaError_t(MUSARTAPI*)(void*, int, size_t,
                                                    musaStream_t);
  using MusaMemGetInfoFn = musaError_t(MUSARTAPI*)(size_t*, size_t*);
  using MusaStreamCreateFn = musaError_t(MUSARTAPI*)(musaStream_t*);
  using MusaStreamDestroyFn = musaError_t(MUSARTAPI*)(musaStream_t);
  using MusaStreamSynchronizeFn = musaError_t(MUSARTAPI*)(musaStream_t);
  using MusaStreamWaitEventFn = musaError_t(MUSARTAPI*)(musaStream_t,
                                                        musaEvent_t,
                                                        unsigned int);
  using MusaEventCreateFn = musaError_t(MUSARTAPI*)(musaEvent_t*);
  using MusaEventDestroyFn = musaError_t(MUSARTAPI*)(musaEvent_t);
  using MusaEventRecordFn = musaError_t(MUSARTAPI*)(musaEvent_t, musaStream_t);
  using MusaEventSynchronizeFn = musaError_t(MUSARTAPI*)(musaEvent_t);
  using MusaEventQueryFn = musaError_t(MUSARTAPI*)(musaEvent_t);
  using MusaRuntimeGetVersionFn = musaError_t(MUSARTAPI*)(int*);
  using MusaGetErrorStringFn = const char*(MUSARTAPI*)(musaError_t);

  // `load_once_` publishes both `load_status_` and the immutable function
  // table. No loader lock is held while invoking libmusart after
  // initialization.
  mutable absl::once_flag load_once_;
  mutable absl::Status load_status_;
  mutable std::unique_ptr<const FunctionTable> functions_;
  std::unique_ptr<internal::MusaSymbolLoader> loader_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_RUNTIME_H_
