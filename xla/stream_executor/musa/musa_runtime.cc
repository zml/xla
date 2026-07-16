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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_status.h"

namespace stream_executor::musa {
namespace {

constexpr const char* kMusartNames[] = {"libmusart.so.1.5", "libmusart.so"};

}  // namespace

struct MusaRuntime::FunctionTable {
  MusaGetDeviceCountFn get_device_count = nullptr;
  MusaGetDevicePropertiesFn get_device_properties = nullptr;
  MusaDeviceGetAttributeFn device_get_attribute = nullptr;
  MusaDeviceGetPciBusIdFn device_get_pci_bus_id = nullptr;
  MusaSetDeviceFn set_device = nullptr;
  MusaDeviceSynchronizeFn device_synchronize = nullptr;
  MusaMallocFn malloc = nullptr;
  MusaFreeFn free = nullptr;
  MusaHostAllocFn host_alloc = nullptr;
  MusaFreeHostFn free_host = nullptr;
  MusaMemcpyFn memcpy = nullptr;
  MusaMemcpyAsyncFn memcpy_async = nullptr;
  MusaMemsetAsyncFn memset_async = nullptr;
  MusaMemGetInfoFn mem_get_info = nullptr;
  MusaStreamCreateFn stream_create = nullptr;
  MusaStreamDestroyFn stream_destroy = nullptr;
  MusaStreamSynchronizeFn stream_synchronize = nullptr;
  MusaStreamWaitEventFn stream_wait_event = nullptr;
  MusaEventCreateFn event_create = nullptr;
  MusaEventDestroyFn event_destroy = nullptr;
  MusaEventRecordFn event_record = nullptr;
  MusaEventSynchronizeFn event_synchronize = nullptr;
  MusaEventQueryFn event_query = nullptr;
  MusaRuntimeGetVersionFn runtime_get_version = nullptr;
  MusaGetErrorStringFn get_error_string = nullptr;
};

MusaRuntime::MusaRuntime(std::unique_ptr<internal::MusaSymbolLoader> loader)
    : loader_(std::move(loader)) {}

MusaRuntime::~MusaRuntime() = default;

MusaRuntime* MusaRuntime::Get() {
  static auto* runtime = new MusaRuntime(internal::CreateMusaDsoLoader(
      std::vector<std::string>{kMusartNames[0], kMusartNames[1]}));
  return runtime;
}

std::unique_ptr<MusaRuntime> MusaRuntime::CreateForTesting(
    std::unique_ptr<internal::MusaSymbolLoader> loader) {
  return std::unique_ptr<MusaRuntime>(new MusaRuntime(std::move(loader)));
}

absl::Status MusaRuntime::Init() { return Load(); }

bool MusaRuntime::IsLoaded() { return Load().ok(); }

absl::Status MusaRuntime::Load() const {
  absl::call_once(load_once_, [this] { load_status_ = Initialize(); });
  return load_status_;
}

absl::Status MusaRuntime::Initialize() const {
  if (loader_ == nullptr) {
    return absl::InternalError("MUSA runtime symbol loader is null");
  }
  if (absl::Status status = loader_->Load(); !status.ok()) {
    return status;
  }

  auto table = std::make_unique<FunctionTable>();

#define XLA_MUSA_LOAD_REQUIRED(field, symbol)                           \
  do {                                                                  \
    absl::StatusOr<void*> resolved = loader_->Resolve(symbol);          \
    if (!resolved.ok() || *resolved == nullptr) {                       \
      return absl::FailedPreconditionError(absl::StrCat(                \
          "MUSA runtime ", loader_->loaded_path(),                      \
          " is missing required symbol ", symbol,                       \
          resolved.ok() ? "" : absl::StrCat(": ", resolved.status()))); \
    }                                                                   \
    table->field = reinterpret_cast<decltype(table->field)>(*resolved); \
  } while (false)

#define XLA_MUSA_LOAD_OPTIONAL(field, symbol)                             \
  do {                                                                    \
    absl::StatusOr<void*> resolved = loader_->Resolve(symbol);            \
    if (resolved.ok() && *resolved != nullptr) {                          \
      table->field = reinterpret_cast<decltype(table->field)>(*resolved); \
    }                                                                     \
  } while (false)

  XLA_MUSA_LOAD_REQUIRED(get_device_count, "musaGetDeviceCount");
  XLA_MUSA_LOAD_REQUIRED(get_device_properties, "musaGetDeviceProperties");
  XLA_MUSA_LOAD_REQUIRED(device_get_attribute, "musaDeviceGetAttribute");
  XLA_MUSA_LOAD_REQUIRED(device_get_pci_bus_id, "musaDeviceGetPCIBusId");
  XLA_MUSA_LOAD_REQUIRED(set_device, "musaSetDevice");
  XLA_MUSA_LOAD_REQUIRED(device_synchronize, "musaDeviceSynchronize");
  XLA_MUSA_LOAD_REQUIRED(malloc, "musaMalloc");
  XLA_MUSA_LOAD_REQUIRED(free, "musaFree");
  XLA_MUSA_LOAD_REQUIRED(memcpy, "musaMemcpy");
  XLA_MUSA_LOAD_REQUIRED(memcpy_async, "musaMemcpyAsync");
  XLA_MUSA_LOAD_REQUIRED(stream_create, "musaStreamCreate");
  XLA_MUSA_LOAD_REQUIRED(stream_destroy, "musaStreamDestroy");
  XLA_MUSA_LOAD_REQUIRED(stream_synchronize, "musaStreamSynchronize");
  XLA_MUSA_LOAD_REQUIRED(stream_wait_event, "musaStreamWaitEvent");
  XLA_MUSA_LOAD_REQUIRED(event_create, "musaEventCreate");
  XLA_MUSA_LOAD_REQUIRED(event_destroy, "musaEventDestroy");
  XLA_MUSA_LOAD_REQUIRED(event_record, "musaEventRecord");
  XLA_MUSA_LOAD_REQUIRED(event_synchronize, "musaEventSynchronize");
  XLA_MUSA_LOAD_REQUIRED(event_query, "musaEventQuery");

  XLA_MUSA_LOAD_OPTIONAL(host_alloc, "musaHostAlloc");
  XLA_MUSA_LOAD_OPTIONAL(free_host, "musaFreeHost");
  if ((table->host_alloc == nullptr) != (table->free_host == nullptr)) {
    return absl::FailedPreconditionError(
        absl::StrCat("MUSA runtime ", loader_->loaded_path(),
                     " must export musaHostAlloc and musaFreeHost as a pair"));
  }
  XLA_MUSA_LOAD_OPTIONAL(memset_async, "musaMemsetAsync");
  XLA_MUSA_LOAD_OPTIONAL(mem_get_info, "musaMemGetInfo");
  XLA_MUSA_LOAD_OPTIONAL(runtime_get_version, "musaRuntimeGetVersion");
  XLA_MUSA_LOAD_OPTIONAL(get_error_string, "musaGetErrorString");

#undef XLA_MUSA_LOAD_OPTIONAL
#undef XLA_MUSA_LOAD_REQUIRED

  functions_ = std::move(table);
  return absl::OkStatus();
}

absl::StatusOr<int> MusaRuntime::GetDeviceCount() {
  RETURN_IF_ERROR(Load());
  int count = 0;
  musaError_t result = functions_->get_device_count(&count);
  if (result != musaSuccess) {
    return ToStatus(result, "musaGetDeviceCount", ErrorString(result));
  }
  return count;
}

absl::StatusOr<MusaDeviceProperties> MusaRuntime::GetDeviceProperties(
    int device_ordinal) {
  RETURN_IF_ERROR(Load());

  musaDeviceProp native = {};
  musaError_t result =
      functions_->get_device_properties(&native, device_ordinal);
  if (result != musaSuccess) {
    return ToStatus(result, "musaGetDeviceProperties", ErrorString(result));
  }

  MusaDeviceProperties properties;
  properties.name = native.name;
  properties.total_memory_bytes = native.totalGlobalMem;

  std::array<char, 32> pci_bus_id = {};
  result = functions_->device_get_pci_bus_id(
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
  musaError_t result = functions_->device_get_attribute(
      &value, static_cast<musaDeviceAttr>(attribute), device_ordinal);
  if (result != musaSuccess) {
    return ToStatus(result, absl::StrCat("musaDeviceGetAttribute(", name, ")"),
                    ErrorString(result));
  }
  return value;
}

absl::Status MusaRuntime::SetDevice(int device_ordinal) {
  RETURN_IF_ERROR(Load());
  musaError_t result = functions_->set_device(device_ordinal);
  return ToStatus(result, "musaSetDevice", ErrorString(result));
}

absl::Status MusaRuntime::DeviceSynchronize() {
  RETURN_IF_ERROR(Load());
  musaError_t result = functions_->device_synchronize();
  return ToStatus(result, "musaDeviceSynchronize", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::Malloc(uint64_t bytes) {
  RETURN_IF_ERROR(Load());
  void* ptr = nullptr;
  musaError_t result = functions_->malloc(&ptr, static_cast<size_t>(bytes));
  if (result != musaSuccess) {
    return ToStatus(result, "musaMalloc", ErrorString(result));
  }
  return ptr;
}

absl::Status MusaRuntime::Free(void* ptr) {
  if (ptr == nullptr) return absl::OkStatus();
  RETURN_IF_ERROR(Load());
  musaError_t result = functions_->free(ptr);
  return ToStatus(result, "musaFree", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::HostAlloc(uint64_t bytes) {
  RETURN_IF_ERROR(Load());
  if (functions_->host_alloc == nullptr) {
    return absl::UnimplementedError("musaHostAlloc is not available.");
  }
  void* ptr = nullptr;
  musaError_t result =
      functions_->host_alloc(&ptr, static_cast<size_t>(bytes), 0);
  if (result != musaSuccess) {
    return ToStatus(result, "musaHostAlloc", ErrorString(result));
  }
  return ptr;
}

absl::Status MusaRuntime::FreeHost(void* ptr) {
  if (ptr == nullptr) return absl::OkStatus();
  RETURN_IF_ERROR(Load());
  if (functions_->free_host == nullptr) {
    return absl::UnimplementedError("musaFreeHost is not available.");
  }
  musaError_t result = functions_->free_host(ptr);
  return ToStatus(result, "musaFreeHost", ErrorString(result));
}

absl::Status MusaRuntime::Memcpy(void* dst, const void* src, uint64_t bytes,
                                 MusaMemcpyKind kind) {
  RETURN_IF_ERROR(Load());
  musaError_t result = functions_->memcpy(dst, src, static_cast<size_t>(bytes),
                                          static_cast<::musaMemcpyKind>(kind));
  return ToStatus(result, "musaMemcpy", ErrorString(result));
}

absl::Status MusaRuntime::MemcpyAsync(void* dst, const void* src,
                                      uint64_t bytes, MusaMemcpyKind kind,
                                      void* stream) {
  RETURN_IF_ERROR(Load());
  musaError_t result = functions_->memcpy_async(
      dst, src, static_cast<size_t>(bytes), static_cast<::musaMemcpyKind>(kind),
      reinterpret_cast<musaStream_t>(stream));
  return ToStatus(result, "musaMemcpyAsync", ErrorString(result));
}

absl::Status MusaRuntime::MemsetAsync(void* dst, int value, uint64_t bytes,
                                      void* stream) {
  RETURN_IF_ERROR(Load());
  if (functions_->memset_async == nullptr) {
    return absl::UnimplementedError("musaMemsetAsync is not available.");
  }
  musaError_t result =
      functions_->memset_async(dst, value, static_cast<size_t>(bytes),
                               reinterpret_cast<musaStream_t>(stream));
  return ToStatus(result, "musaMemsetAsync", ErrorString(result));
}

absl::Status MusaRuntime::MemGetInfo(size_t* free_bytes,
                                     size_t* total_bytes) const {
  RETURN_IF_ERROR(Load());
  if (functions_->mem_get_info == nullptr) {
    return absl::UnimplementedError("musaMemGetInfo is not available.");
  }
  musaError_t result = functions_->mem_get_info(free_bytes, total_bytes);
  return ToStatus(result, "musaMemGetInfo", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::StreamCreate() {
  RETURN_IF_ERROR(Load());
  musaStream_t stream = nullptr;
  musaError_t result = functions_->stream_create(&stream);
  if (result != musaSuccess) {
    return ToStatus(result, "musaStreamCreate", ErrorString(result));
  }
  if (stream == nullptr) {
    return absl::InternalError(
        "musaStreamCreate returned success with a null stream");
  }
  return reinterpret_cast<void*>(stream);
}

absl::Status MusaRuntime::StreamDestroy(void* stream) {
  if (stream == nullptr) return absl::OkStatus();
  RETURN_IF_ERROR(Load());
  musaError_t result =
      functions_->stream_destroy(reinterpret_cast<musaStream_t>(stream));
  return ToStatus(result, "musaStreamDestroy", ErrorString(result));
}

absl::Status MusaRuntime::StreamSynchronize(void* stream) {
  RETURN_IF_ERROR(Load());
  musaError_t result =
      functions_->stream_synchronize(reinterpret_cast<musaStream_t>(stream));
  return ToStatus(result, "musaStreamSynchronize", ErrorString(result));
}

absl::Status MusaRuntime::StreamWaitEvent(void* stream, void* event) {
  RETURN_IF_ERROR(Load());
  musaError_t result =
      functions_->stream_wait_event(reinterpret_cast<musaStream_t>(stream),
                                    reinterpret_cast<musaEvent_t>(event), 0);
  return ToStatus(result, "musaStreamWaitEvent", ErrorString(result));
}

absl::StatusOr<void*> MusaRuntime::EventCreate() {
  RETURN_IF_ERROR(Load());
  musaEvent_t event = nullptr;
  musaError_t result = functions_->event_create(&event);
  if (result != musaSuccess) {
    return ToStatus(result, "musaEventCreate", ErrorString(result));
  }
  if (event == nullptr) {
    return absl::InternalError(
        "musaEventCreate returned success with a null event");
  }
  return reinterpret_cast<void*>(event);
}

absl::Status MusaRuntime::EventDestroy(void* event) {
  if (event == nullptr) return absl::OkStatus();
  RETURN_IF_ERROR(Load());
  musaError_t result =
      functions_->event_destroy(reinterpret_cast<musaEvent_t>(event));
  return ToStatus(result, "musaEventDestroy", ErrorString(result));
}

absl::Status MusaRuntime::EventRecord(void* event, void* stream) {
  RETURN_IF_ERROR(Load());
  musaError_t result =
      functions_->event_record(reinterpret_cast<musaEvent_t>(event),
                               reinterpret_cast<musaStream_t>(stream));
  return ToStatus(result, "musaEventRecord", ErrorString(result));
}

absl::Status MusaRuntime::EventSynchronize(void* event) {
  RETURN_IF_ERROR(Load());
  musaError_t result =
      functions_->event_synchronize(reinterpret_cast<musaEvent_t>(event));
  return ToStatus(result, "musaEventSynchronize", ErrorString(result));
}

int MusaRuntime::EventQuery(void* event) {
  if (!Load().ok()) return -1;
  return static_cast<int>(
      functions_->event_query(reinterpret_cast<musaEvent_t>(event)));
}

absl::StatusOr<int> MusaRuntime::RuntimeVersion() {
  RETURN_IF_ERROR(Load());
  if (functions_->runtime_get_version == nullptr) {
    return absl::UnimplementedError("musaRuntimeGetVersion is not available.");
  }
  int version = 0;
  musaError_t result = functions_->runtime_get_version(&version);
  if (result != musaSuccess) {
    return ToStatus(result, "musaRuntimeGetVersion", ErrorString(result));
  }
  return version;
}

const char* MusaRuntime::ErrorString(musaError_t result) const {
  if (functions_->get_error_string == nullptr) {
    return "";
  }
  const char* error = functions_->get_error_string(result);
  return error == nullptr ? "" : error;
}

}  // namespace stream_executor::musa
