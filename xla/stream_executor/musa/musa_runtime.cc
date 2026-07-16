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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include <dlfcn.h>
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
  if (load_status_.has_value()) {
    return *load_status_;
  }

  for (const char* name : kMusartNames) {
    handle_ = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (handle_ != nullptr) {
      break;
    }
  }
  if (handle_ == nullptr) {
    return FailLoad(absl::FailedPreconditionError(absl::StrCat(
        "Unable to load MUSA runtime library libmusart.so: ", dlerror())));
  }

  if (auto status = LoadSymbol(get_device_count_, "musaGetDeviceCount");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status =
          LoadSymbol(get_device_properties_, "musaGetDeviceProperties");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(device_get_attribute_, "musaDeviceGetAttribute");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(device_get_pci_bus_id_, "musaDeviceGetPCIBusId");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(set_device_, "musaSetDevice"); !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(device_synchronize_, "musaDeviceSynchronize");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(malloc_, "musaMalloc"); !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(free_, "musaFree"); !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(memcpy_, "musaMemcpy"); !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(stream_create_, "musaStreamCreate");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(stream_destroy_, "musaStreamDestroy");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(stream_synchronize_, "musaStreamSynchronize");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(stream_wait_event_, "musaStreamWaitEvent");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(event_create_, "musaEventCreate");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(event_destroy_, "musaEventDestroy");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(event_record_, "musaEventRecord");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(event_synchronize_, "musaEventSynchronize");
      !status.ok()) {
    return FailLoad(status);
  }
  if (auto status = LoadSymbol(event_query_, "musaEventQuery"); !status.ok()) {
    return FailLoad(status);
  }

  host_alloc_ = reinterpret_cast<MusaHostAllocFn>(Resolve("musaHostAlloc"));
  free_host_ = reinterpret_cast<MusaFreeHostFn>(Resolve("musaFreeHost"));
  memcpy_async_ =
      reinterpret_cast<MusaMemcpyAsyncFn>(Resolve("musaMemcpyAsync"));
  memset_async_ =
      reinterpret_cast<MusaMemsetAsyncFn>(Resolve("musaMemsetAsync"));
  mem_get_info_ = reinterpret_cast<MusaMemGetInfoFn>(Resolve("musaMemGetInfo"));
  runtime_get_version_ = reinterpret_cast<MusaRuntimeGetVersionFn>(
      Resolve("musaRuntimeGetVersion"));
  driver_get_version_ =
      reinterpret_cast<MusaDriverGetVersionFn>(Resolve("musaDriverGetVersion"));
  get_error_string_ =
      reinterpret_cast<MusaGetErrorStringFn>(Resolve("musaGetErrorString"));
  load_status_ = absl::OkStatus();
  return *load_status_;
}

absl::Status MusaRuntime::FailLoad(absl::Status status) {
  if (handle_ != nullptr) {
    dlclose(handle_);
    handle_ = nullptr;
  }
  load_status_ = status;
  return status;
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
  musaError_t result = get_device_count_(&count);
  if (result != musaSuccess) {
    return ToStatus(result, "musaGetDeviceCount", ErrorString(result));
  }
  return count;
}

absl::StatusOr<MusaDeviceProperties> MusaRuntime::GetDeviceProperties(
    int device_ordinal) {
  absl::MutexLock lock(mu_);
  RETURN_IF_ERROR(Load());

  musaDeviceProp native = {};
  musaError_t result = get_device_properties_(&native, device_ordinal);
  if (result != musaSuccess) {
    return ToStatus(result, "musaGetDeviceProperties", ErrorString(result));
  }

  MusaDeviceProperties properties;
  properties.name = native.name;
  properties.total_memory_bytes = native.totalGlobalMem;

  std::array<char, 32> pci_bus_id = {};
  result = device_get_pci_bus_id_(
      pci_bus_id.data(), static_cast<int>(pci_bus_id.size()), device_ordinal);
  if (result != musaSuccess) {
    return ToStatus(result, "musaDeviceGetPCIBusId", ErrorString(result));
  }
  properties.pci_bus_id = pci_bus_id.data();

#define XLA_MUSA_ASSIGN_ATTRIBUTE(field, attribute)                   \
  ASSIGN_OR_RETURN(                                                   \
      properties.field,                                               \
      GetDeviceAttribute(device_ordinal, static_cast<int>(attribute), \
                         #attribute))

  XLA_MUSA_ASSIGN_ATTRIBUTE(max_threads_per_block,
                            musaDevAttrMaxThreadsPerBlock);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_block_dim_x, musaDevAttrMaxBlockDimX);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_block_dim_y, musaDevAttrMaxBlockDimY);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_block_dim_z, musaDevAttrMaxBlockDimZ);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_grid_dim_x, musaDevAttrMaxGridDimX);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_grid_dim_y, musaDevAttrMaxGridDimY);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_grid_dim_z, musaDevAttrMaxGridDimZ);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_shared_memory_per_block,
                            musaDevAttrMaxSharedMemoryPerBlock);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_shared_memory_per_multiprocessor,
                            musaDevAttrMaxSharedMemoryPerMultiprocessor);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_shared_memory_per_block_optin,
                            musaDevAttrMaxSharedMemoryPerBlockOptin);
  XLA_MUSA_ASSIGN_ATTRIBUTE(reserved_shared_memory_per_block,
                            musaDevAttrReservedSharedMemoryPerBlock);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_registers_per_block,
                            musaDevAttrMaxRegistersPerBlock);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_registers_per_multiprocessor,
                            musaDevAttrMaxRegistersPerMultiprocessor);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_threads_per_multiprocessor,
                            musaDevAttrMaxThreadsPerMultiProcessor);
  XLA_MUSA_ASSIGN_ATTRIBUTE(max_blocks_per_multiprocessor,
                            musaDevAttrMaxBlocksPerMultiprocessor);
  XLA_MUSA_ASSIGN_ATTRIBUTE(multiprocessor_count,
                            musaDevAttrMultiProcessorCount);
  XLA_MUSA_ASSIGN_ATTRIBUTE(hardware_warp_size, musaDevAttrWarpSize);
  XLA_MUSA_ASSIGN_ATTRIBUTE(clock_rate_khz, musaDevAttrClockRate);
  XLA_MUSA_ASSIGN_ATTRIBUTE(memory_clock_rate_khz, musaDevAttrMemoryClockRate);
  XLA_MUSA_ASSIGN_ATTRIBUTE(memory_bus_width_bits,
                            musaDevAttrGlobalMemoryBusWidth);
  XLA_MUSA_ASSIGN_ATTRIBUTE(l2_cache_size_bytes, musaDevAttrL2CacheSize);
  XLA_MUSA_ASSIGN_ATTRIBUTE(texture_alignment_bytes,
                            musaDevAttrTextureAlignment);
  XLA_MUSA_ASSIGN_ATTRIBUTE(texture_pitch_alignment_bytes,
                            musaDevAttrTexturePitchAlignment);
  XLA_MUSA_ASSIGN_ATTRIBUTE(total_constant_memory_bytes,
                            musaDevAttrTotalConstantMemory);
  XLA_MUSA_ASSIGN_ATTRIBUTE(compute_capability_major,
                            musaDevAttrComputeCapabilityMajor);
  XLA_MUSA_ASSIGN_ATTRIBUTE(compute_capability_minor,
                            musaDevAttrComputeCapabilityMinor);
  int ecc_enabled = 0;
  ASSIGN_OR_RETURN(ecc_enabled,
                   GetDeviceAttribute(device_ordinal,
                                      static_cast<int>(musaDevAttrEccEnabled),
                                      "musaDevAttrEccEnabled"));
  properties.ecc_enabled = ecc_enabled != 0;

#undef XLA_MUSA_ASSIGN_ATTRIBUTE

  return properties;
}

absl::StatusOr<int> MusaRuntime::GetDeviceAttribute(
    int device_ordinal, int attribute, absl::string_view name) const {
  int value = 0;
  musaError_t result = device_get_attribute_(
      &value, static_cast<musaDeviceAttr>(attribute), device_ordinal);
  if (result != musaSuccess) {
    return ToStatus(result, absl::StrCat("musaDeviceGetAttribute(", name, ")"),
                    ErrorString(result));
  }
  return value;
}

absl::Status MusaRuntime::SetDevice(int device_ordinal) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result = set_device_(device_ordinal);
  return ToStatus(result, "musaSetDevice", ErrorString(result));
}

absl::Status MusaRuntime::DeviceSynchronize() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result = device_synchronize_();
  return ToStatus(result, "musaDeviceSynchronize", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::Malloc(uint64_t bytes) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  void* ptr = nullptr;
  musaError_t result = malloc_(&ptr, static_cast<size_t>(bytes));
  if (result != musaSuccess) {
    return ToStatus(result, "musaMalloc", ErrorString(result));
  }
  return ptr;
}

absl::Status MusaRuntime::Free(void* ptr) {
  if (ptr == nullptr) return absl::OkStatus();
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result = free_(ptr);
  return ToStatus(result, "musaFree", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::HostAlloc(uint64_t bytes) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (host_alloc_ == nullptr) {
    return absl::UnimplementedError("musaHostAlloc is not available.");
  }
  void* ptr = nullptr;
  musaError_t result = host_alloc_(&ptr, static_cast<size_t>(bytes), 0);
  if (result != musaSuccess) {
    return ToStatus(result, "musaHostAlloc", ErrorString(result));
  }
  return ptr;
}

absl::Status MusaRuntime::FreeHost(void* ptr) {
  if (ptr == nullptr) return absl::OkStatus();
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (free_host_ == nullptr) {
    return absl::UnimplementedError("musaFreeHost is not available.");
  }
  musaError_t result = free_host_(ptr);
  return ToStatus(result, "musaFreeHost", ErrorString(result));
}

absl::Status MusaRuntime::Memcpy(void* dst, const void* src, uint64_t bytes,
                                 MusaMemcpyKind kind) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result = memcpy_(dst, src, static_cast<size_t>(bytes),
                               static_cast<::musaMemcpyKind>(kind));
  return ToStatus(result, "musaMemcpy", ErrorString(result));
}

absl::Status MusaRuntime::MemcpyAsync(void* dst, const void* src,
                                      uint64_t bytes, MusaMemcpyKind kind,
                                      void* stream) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (memcpy_async_ == nullptr) {
    musaError_t result = memcpy_(dst, src, static_cast<size_t>(bytes),
                                 static_cast<::musaMemcpyKind>(kind));
    return ToStatus(result, "musaMemcpy", ErrorString(result));
  }
  musaError_t result = memcpy_async_(dst, src, static_cast<size_t>(bytes),
                                     static_cast<::musaMemcpyKind>(kind),
                                     reinterpret_cast<musaStream_t>(stream));
  return ToStatus(result, "musaMemcpyAsync", ErrorString(result));
}

absl::Status MusaRuntime::MemsetAsync(void* dst, int value, uint64_t bytes,
                                      void* stream) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (memset_async_ == nullptr) {
    return absl::UnimplementedError("musaMemsetAsync is not available.");
  }
  musaError_t result = memset_async_(dst, value, static_cast<size_t>(bytes),
                                     reinterpret_cast<musaStream_t>(stream));
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
  musaError_t result = mem_get_info_(free_bytes, total_bytes);
  return ToStatus(result, "musaMemGetInfo", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::StreamCreate() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaStream_t stream = nullptr;
  musaError_t result = stream_create_(&stream);
  if (result != musaSuccess) {
    return ToStatus(result, "musaStreamCreate", ErrorString(result));
  }
  return reinterpret_cast<void*>(stream);
}

absl::Status MusaRuntime::StreamDestroy(void* stream) {
  if (stream == nullptr) return absl::OkStatus();
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result = stream_destroy_(reinterpret_cast<musaStream_t>(stream));
  return ToStatus(result, "musaStreamDestroy", ErrorString(result));
}

absl::Status MusaRuntime::StreamSynchronize(void* stream) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result =
      stream_synchronize_(reinterpret_cast<musaStream_t>(stream));
  return ToStatus(result, "musaStreamSynchronize", ErrorString(result));
}

absl::Status MusaRuntime::StreamWaitEvent(void* stream, void* event) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result =
      stream_wait_event_(reinterpret_cast<musaStream_t>(stream),
                         reinterpret_cast<musaEvent_t>(event), 0);
  return ToStatus(result, "musaStreamWaitEvent", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::EventCreate() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaEvent_t event = nullptr;
  musaError_t result = event_create_(&event);
  if (result != musaSuccess) {
    return ToStatus(result, "musaEventCreate", ErrorString(result));
  }
  return reinterpret_cast<void*>(event);
}

absl::Status MusaRuntime::EventDestroy(void* event) {
  if (event == nullptr) return absl::OkStatus();
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result = event_destroy_(reinterpret_cast<musaEvent_t>(event));
  return ToStatus(result, "musaEventDestroy", ErrorString(result));
}

absl::Status MusaRuntime::EventRecord(void* event, void* stream) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result = event_record_(reinterpret_cast<musaEvent_t>(event),
                                     reinterpret_cast<musaStream_t>(stream));
  return ToStatus(result, "musaEventRecord", ErrorString(result));
}

absl::Status MusaRuntime::EventSynchronize(void* event) {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  musaError_t result = event_synchronize_(reinterpret_cast<musaEvent_t>(event));
  return ToStatus(result, "musaEventSynchronize", ErrorString(result));
}

int MusaRuntime::EventQuery(void* event) {
  absl::MutexLock lock(mu_);
  if (!Load().ok()) return -1;
  return event_query_(reinterpret_cast<musaEvent_t>(event));
}

absl::StatusOr<int> MusaRuntime::RuntimeVersion() {
  absl::MutexLock lock(mu_);
  if (auto status = Load(); !status.ok()) return status;
  if (runtime_get_version_ == nullptr) {
    return absl::UnimplementedError("musaRuntimeGetVersion is not available.");
  }
  int version = 0;
  musaError_t result = runtime_get_version_(&version);
  if (result != musaSuccess) {
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
  musaError_t result = driver_get_version_(&version);
  if (result != musaSuccess) {
    return ToStatus(result, "musaDriverGetVersion", ErrorString(result));
  }
  return version;
}

const char* MusaRuntime::ErrorString(musaError_t result) const {
  if (get_error_string_ == nullptr) {
    return "";
  }
  const char* error = get_error_string_(result);
  return error == nullptr ? "" : error;
}

}  // namespace stream_executor::musa
