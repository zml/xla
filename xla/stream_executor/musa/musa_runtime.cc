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

#include "xla/stream_executor/musa/musa_runtime.h"

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/musa/musa_status.h"

namespace stream_executor::musa {
namespace {

constexpr const char* kMusartNames[] = {"libmusart.so", "libmusart.so.1"};

}  // namespace

MusaRuntime* MusaRuntime::Get() {
  static auto* runtime = new MusaRuntime;
  return runtime;
}

absl::Status MusaRuntime::Init() {
  absl::MutexLock lock(mu_);
  return Load();
}

bool MusaRuntime::IsLoaded() {
  absl::MutexLock lock(mu_);
  return Load().ok();
}

absl::Status MusaRuntime::Load() {
  if (attempted_load_) {
    if (handle_ == nullptr) {
      return absl::FailedPreconditionError(
          "MUSA runtime library libmusart.so is not available.");
    }
    return absl::OkStatus();
  }

  attempted_load_ = true;
  for (const char* name : kMusartNames) {
    handle_ = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (handle_ != nullptr) {
      break;
    }
  }
  if (handle_ == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unable to load MUSA runtime library libmusart.so: ", dlerror()));
  }

  if (auto status = LoadSymbol(get_device_count_, "musaGetDeviceCount");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(set_device_, "musaSetDevice"); !status.ok()) {
    return status;
  }
  if (auto status =
          LoadSymbol(device_synchronize_, "musaDeviceSynchronize");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(malloc_, "musaMalloc"); !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(free_, "musaFree"); !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(memcpy_, "musaMemcpy"); !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(stream_create_, "musaStreamCreate");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(stream_destroy_, "musaStreamDestroy");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(stream_synchronize_, "musaStreamSynchronize");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(stream_wait_event_, "musaStreamWaitEvent");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(event_create_, "musaEventCreate");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(event_destroy_, "musaEventDestroy");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(event_record_, "musaEventRecord");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(event_synchronize_, "musaEventSynchronize");
      !status.ok()) {
    return status;
  }
  if (auto status = LoadSymbol(event_query_, "musaEventQuery"); !status.ok()) {
    return status;
  }

  host_alloc_ =
      reinterpret_cast<MusaHostAllocFn>(Resolve("musaHostAlloc"));
  free_host_ =
      reinterpret_cast<MusaFreeHostFn>(Resolve("musaFreeHost"));
  memcpy_async_ =
      reinterpret_cast<MusaMemcpyAsyncFn>(Resolve("musaMemcpyAsync"));
  memset_async_ =
      reinterpret_cast<MusaMemsetAsyncFn>(Resolve("musaMemsetAsync"));
  mem_get_info_ =
      reinterpret_cast<MusaMemGetInfoFn>(Resolve("musaMemGetInfo"));
  runtime_get_version_ = reinterpret_cast<MusaRuntimeGetVersionFn>(
      Resolve("musaRuntimeGetVersion"));
  driver_get_version_ = reinterpret_cast<MusaDriverGetVersionFn>(
      Resolve("musaDriverGetVersion"));
  get_error_string_ =
      reinterpret_cast<MusaGetErrorStringFn>(Resolve("musaGetErrorString"));
  return absl::OkStatus();
}

void* MusaRuntime::Resolve(absl::string_view symbol) const {
  return dlsym(handle_, std::string(symbol).c_str());
}

template <typename Fn>
absl::Status MusaRuntime::LoadSymbol(Fn& fn, absl::string_view symbol) {
  fn = reinterpret_cast<Fn>(Resolve(symbol));
  if (fn == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("MUSA runtime is missing required symbol ", symbol));
  }
  return absl::OkStatus();
}

absl::StatusOr<int> MusaRuntime::GetDeviceCount() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int count = 0;
  int result = get_device_count_(&count);
  if (result != 0) return ToStatus(result, "musaGetDeviceCount", ErrorString(result));
  return count;
}

absl::Status MusaRuntime::SetDevice(int device_ordinal) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = set_device_(device_ordinal);
  return ToStatus(result, "musaSetDevice", ErrorString(result));
}

absl::Status MusaRuntime::DeviceSynchronize() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = device_synchronize_();
  return ToStatus(result, "musaDeviceSynchronize", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::Malloc(uint64_t bytes) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  void* ptr = nullptr;
  int result = malloc_(&ptr, static_cast<size_t>(bytes));
  if (result != 0) return ToStatus(result, "musaMalloc", ErrorString(result));
  return ptr;
}

absl::Status MusaRuntime::Free(void* ptr) {
  if (ptr == nullptr) return absl::OkStatus();
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = free_(ptr);
  return ToStatus(result, "musaFree", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::HostAlloc(uint64_t bytes) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (host_alloc_ == nullptr) {
    return absl::UnimplementedError("musaHostAlloc is not available.");
  }
  void* ptr = nullptr;
  int result = host_alloc_(&ptr, static_cast<size_t>(bytes), 0);
  if (result != 0) return ToStatus(result, "musaHostAlloc", ErrorString(result));
  return ptr;
}

absl::Status MusaRuntime::FreeHost(void* ptr) {
  if (ptr == nullptr) return absl::OkStatus();
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (free_host_ == nullptr) {
    return absl::UnimplementedError("musaFreeHost is not available.");
  }
  int result = free_host_(ptr);
  return ToStatus(result, "musaFreeHost", ErrorString(result));
}

absl::Status MusaRuntime::Memcpy(void* dst, const void* src, uint64_t bytes,
                                 MusaMemcpyKind kind) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = memcpy_(dst, src, static_cast<size_t>(bytes),
                       static_cast<int>(kind));
  return ToStatus(result, "musaMemcpy", ErrorString(result));
}

absl::Status MusaRuntime::MemcpyAsync(void* dst, const void* src,
                                      uint64_t bytes, MusaMemcpyKind kind,
                                      void* stream) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (memcpy_async_ == nullptr) {
    int result = memcpy_(dst, src, static_cast<size_t>(bytes),
                         static_cast<int>(kind));
    return ToStatus(result, "musaMemcpy", ErrorString(result));
  }
  int result = memcpy_async_(dst, src, static_cast<size_t>(bytes),
                             static_cast<int>(kind), stream);
  return ToStatus(result, "musaMemcpyAsync", ErrorString(result));
}

absl::Status MusaRuntime::MemsetAsync(void* dst, int value, uint64_t bytes,
                                      void* stream) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (memset_async_ == nullptr) {
    return absl::UnimplementedError("musaMemsetAsync is not available.");
  }
  int result = memset_async_(dst, value, static_cast<size_t>(bytes), stream);
  return ToStatus(result, "musaMemsetAsync", ErrorString(result));
}

absl::Status MusaRuntime::MemGetInfo(size_t* free_bytes,
                                     size_t* total_bytes) const {
  absl::MutexLock lock(mu_);
  absl::Status status = const_cast<MusaRuntime*>(this)->Load();
  if (!status.ok()) return status;
  if (mem_get_info_ == nullptr) {
    return absl::UnimplementedError("musaMemGetInfo is not available.");
  }
  int result = mem_get_info_(free_bytes, total_bytes);
  return ToStatus(result, "musaMemGetInfo", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::StreamCreate() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  void* stream = nullptr;
  int result = stream_create_(&stream);
  if (result != 0) return ToStatus(result, "musaStreamCreate", ErrorString(result));
  return stream;
}

absl::Status MusaRuntime::StreamDestroy(void* stream) {
  if (stream == nullptr) return absl::OkStatus();
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = stream_destroy_(stream);
  return ToStatus(result, "musaStreamDestroy", ErrorString(result));
}

absl::Status MusaRuntime::StreamSynchronize(void* stream) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = stream_synchronize_(stream);
  return ToStatus(result, "musaStreamSynchronize", ErrorString(result));
}

absl::Status MusaRuntime::StreamWaitEvent(void* stream, void* event) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = stream_wait_event_(stream, event, 0);
  return ToStatus(result, "musaStreamWaitEvent", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::EventCreate() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  void* event = nullptr;
  int result = event_create_(&event);
  if (result != 0) return ToStatus(result, "musaEventCreate", ErrorString(result));
  return event;
}

absl::Status MusaRuntime::EventDestroy(void* event) {
  if (event == nullptr) return absl::OkStatus();
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = event_destroy_(event);
  return ToStatus(result, "musaEventDestroy", ErrorString(result));
}

absl::Status MusaRuntime::EventRecord(void* event, void* stream) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = event_record_(event, stream);
  return ToStatus(result, "musaEventRecord", ErrorString(result));
}

absl::Status MusaRuntime::EventSynchronize(void* event) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  int result = event_synchronize_(event);
  return ToStatus(result, "musaEventSynchronize", ErrorString(result));
}

int MusaRuntime::EventQuery(void* event) {
  absl::MutexLock lock(mu_);
  if (!Load().ok()) return -1;
  return event_query_(event);
}

absl::StatusOr<int> MusaRuntime::RuntimeVersion() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (runtime_get_version_ == nullptr) {
    return absl::UnimplementedError("musaRuntimeGetVersion is not available.");
  }
  int version = 0;
  int result = runtime_get_version_(&version);
  if (result != 0) {
    return ToStatus(result, "musaRuntimeGetVersion", ErrorString(result));
  }
  return version;
}

absl::StatusOr<int> MusaRuntime::DriverVersion() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (driver_get_version_ == nullptr) {
    return absl::UnimplementedError("musaDriverGetVersion is not available.");
  }
  int version = 0;
  int result = driver_get_version_(&version);
  if (result != 0) {
    return ToStatus(result, "musaDriverGetVersion", ErrorString(result));
  }
  return version;
}

const char* MusaRuntime::ErrorString(int result) const {
  if (get_error_string_ == nullptr) {
    return "";
  }
  const char* error = get_error_string_(result);
  return error == nullptr ? "" : error;
}

}  // namespace stream_executor::musa
