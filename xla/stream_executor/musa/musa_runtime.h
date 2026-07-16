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
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "musa_runtime_api.h"
#include "xla/stream_executor/musa/musa_device_properties.h"

namespace stream_executor::musa {

enum class MusaMemcpyKind : int {
  kHostToDevice = 1,
  kDeviceToHost = 2,
  kDeviceToDevice = 3,
};

class MusaRuntime {
 public:
  static MusaRuntime* Get();

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
  absl::StatusOr<int> DriverVersion();

 private:
  MusaRuntime() = default;

  absl::Status Load() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status FailLoad(absl::Status status) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void* Resolve(absl::string_view symbol) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  const char* ErrorString(musaError_t result) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::StatusOr<int> GetDeviceAttribute(int device_ordinal, int attribute,
                                         absl::string_view name) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  template <typename Fn>
  absl::Status LoadSymbol(Fn& fn, absl::string_view symbol)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  std::optional<absl::Status> load_status_ ABSL_GUARDED_BY(mu_);
  void* handle_ ABSL_GUARDED_BY(mu_) = nullptr;

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
  using MusaDriverGetVersionFn = musaError_t(MUSARTAPI*)(int*);
  using MusaGetErrorStringFn = const char*(MUSARTAPI*)(musaError_t);

  MusaGetDeviceCountFn get_device_count_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaGetDevicePropertiesFn get_device_properties_ ABSL_GUARDED_BY(mu_) =
      nullptr;
  MusaDeviceGetAttributeFn device_get_attribute_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaDeviceGetPciBusIdFn device_get_pci_bus_id_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaSetDeviceFn set_device_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaDeviceSynchronizeFn device_synchronize_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaMallocFn malloc_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaFreeFn free_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaHostAllocFn host_alloc_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaFreeHostFn free_host_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaMemcpyFn memcpy_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaMemcpyAsyncFn memcpy_async_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaMemsetAsyncFn memset_async_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaMemGetInfoFn mem_get_info_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaStreamCreateFn stream_create_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaStreamDestroyFn stream_destroy_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaStreamSynchronizeFn stream_synchronize_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaStreamWaitEventFn stream_wait_event_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaEventCreateFn event_create_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaEventDestroyFn event_destroy_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaEventRecordFn event_record_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaEventSynchronizeFn event_synchronize_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaEventQueryFn event_query_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaRuntimeGetVersionFn runtime_get_version_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaDriverGetVersionFn driver_get_version_ ABSL_GUARDED_BY(mu_) = nullptr;
  MusaGetErrorStringFn get_error_string_ ABSL_GUARDED_BY(mu_) = nullptr;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_RUNTIME_H_
