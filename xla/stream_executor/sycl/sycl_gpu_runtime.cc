/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::sycl {

namespace {

absl::Mutex async_error_mu(absl::kConstInit);
auto* async_errors ABSL_GUARDED_BY(async_error_mu) =
    new absl::flat_hash_map<int, std::vector<absl::Status>>();

absl::Status MergeStatuses(absl::Status primary, absl::Status secondary) {
  if (primary.ok()) {
    return secondary;
  }
  if (secondary.ok()) {
    return primary;
  }
  return absl::Status(primary.code(),
                      absl::StrCat(primary.message(), "; additionally: ",
                                   secondary.message()));
}

absl::Status SyclExceptionStatus(absl::string_view source,
                                 const ::sycl::exception& e) {
  return absl::InternalError(absl::StrCat(
      source, ": SYCL exception: ", e.what(), " (code ", e.code().value(),
      ", category ", e.code().category().name(), ")"));
}

void RecordAsyncError(int device_ordinal, absl::Status status) {
  if (status.ok()) {
    return;
  }
  LOG(ERROR) << status;
  absl::MutexLock lock(&async_error_mu);
  (*async_errors)[device_ordinal].push_back(std::move(status));
}

void RecordAsyncException(int device_ordinal, absl::string_view source,
                          const ::sycl::exception& e) {
  RecordAsyncError(
      device_ordinal,
      absl::InternalError(absl::StrCat(
          source, ": async SYCL exception on device ordinal ", device_ordinal,
          ": ", e.what(), " (code ", e.code().value(), ", category ",
          e.code().category().name(), ")")));
}

::sycl::async_handler MakeSyclAsyncHandler(int device_ordinal,
                                           absl::string_view source) {
  std::string source_string(source);
  return [device_ordinal, source_string](::sycl::exception_list ex_list) {
    for (const auto& e : ex_list) {
      try {
        std::rethrow_exception(e);
      } catch (const ::sycl::exception& e) {
        RecordAsyncException(device_ordinal, source_string, e);
      } catch (const std::exception& e) {
        RecordAsyncError(
            device_ordinal,
            absl::InternalError(absl::StrCat(
                source_string, ": async non-SYCL exception on device ordinal ",
                device_ordinal, ": ", e.what())));
      } catch (...) {
        RecordAsyncError(device_ordinal,
                         absl::InternalError(absl::StrCat(
                             source_string,
                             ": async unknown exception on device ordinal ",
                             device_ordinal)));
      }
    }
  };
}

absl::Status DrainAsyncErrors(int device_ordinal, absl::string_view consumer) {
  std::vector<absl::Status> errors;
  {
    absl::MutexLock lock(&async_error_mu);
    auto it = async_errors->find(device_ordinal);
    if (it == async_errors->end() || it->second.empty()) {
      return absl::OkStatus();
    }
    errors = std::move(it->second);
    async_errors->erase(it);
  }

  std::string message =
      absl::StrCat(consumer, ": observed ", errors.size(),
                   " async SYCL error", errors.size() == 1 ? "" : "s",
                   " for device ordinal ", device_ordinal, ": ",
                   errors.front().message());
  if (errors.size() > 1) {
    absl::StrAppend(&message, " (", errors.size() - 1,
                    " additional async SYCL error",
                    errors.size() == 2 ? "" : "s", " omitted)");
  }
  return absl::Status(errors.front().code(), message);
}

absl::StatusOr<int> GetDeviceOrdinalForQueue(::sycl::queue* stream_handle,
                                             absl::string_view function_name) {
  try {
    return SyclDevicePool::GetDeviceOrdinal(stream_handle->get_device());
  } catch (const ::sycl::exception& e) {
    return SyclExceptionStatus(function_name, e);
  }
}

absl::Status WaitForQueueAndDrain(::sycl::queue* stream_handle,
                                  absl::string_view source) {
  ASSIGN_OR_RETURN(int device_ordinal,
                   GetDeviceOrdinalForQueue(stream_handle, source));
  absl::Status wait_status = absl::OkStatus();
  try {
    stream_handle->wait_and_throw();
  } catch (const ::sycl::exception& e) {
    wait_status = SyclExceptionStatus(source, e);
  }
  return MergeStatuses(wait_status, DrainAsyncErrors(device_ordinal, source));
}

const char* UsmAllocName(::sycl::usm::alloc alloc_type) {
  switch (alloc_type) {
    case ::sycl::usm::alloc::host:
      return "host";
    case ::sycl::usm::alloc::device:
      return "device";
    case ::sycl::usm::alloc::shared:
      return "shared";
    case ::sycl::usm::alloc::unknown:
      return "unknown";
  }
}

absl::StatusOr<::sycl::usm::alloc> GetPointerTypeInContext(
    const void* ptr, const ::sycl::context& context, absl::string_view source) {
  try {
    return ::sycl::get_pointer_type(ptr, context);
  } catch (const ::sycl::exception& e) {
    return SyclExceptionStatus(source, e);
  }
}

absl::StatusOr<MemorySpace> MemorySpaceFromUsmAlloc(
    ::sycl::usm::alloc alloc_type, absl::string_view source) {
  switch (alloc_type) {
    case ::sycl::usm::alloc::device:
      return MemorySpace::kDevice;
    case ::sycl::usm::alloc::shared:
      return MemorySpace::kUnified;
    case ::sycl::usm::alloc::host:
      return MemorySpace::kHost;
    case ::sycl::usm::alloc::unknown:
      return absl::InvalidArgumentError(
          absl::StrCat(source, ": pointer is not a USM allocation in this "
                               "SYCL context"));
  }
}

struct UsmAllocationOwner {
  int device_ordinal;
  ::sycl::usm::alloc alloc_type;
};

absl::StatusOr<std::optional<UsmAllocationOwner>> FindUsmAllocationOwner(
    const void* ptr, absl::string_view source) {
  ASSIGN_OR_RETURN(int device_count, SyclDevicePool::GetDeviceCount());
  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    ASSIGN_OR_RETURN(::sycl::context context,
                     SyclDevicePool::GetDeviceContext(ordinal));
    ASSIGN_OR_RETURN(::sycl::usm::alloc alloc_type,
                     GetPointerTypeInContext(ptr, context, source));
    if (alloc_type != ::sycl::usm::alloc::unknown) {
      return UsmAllocationOwner{ordinal, alloc_type};
    }
  }
  return std::nullopt;
}

absl::Status UnknownUsmPointerError(const void* ptr, absl::string_view arg_name,
                                    absl::string_view source) {
  ASSIGN_OR_RETURN(std::optional<UsmAllocationOwner> owner,
                   FindUsmAllocationOwner(ptr, source));
  if (owner.has_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        source, ": ", arg_name, " pointer ", absl::StrFormat("%p", ptr),
        " is ", UsmAllocName(owner->alloc_type),
        " USM allocated in device ordinal ", owner->device_ordinal,
        "'s SYCL context, but this copy is submitted to a different "
        "per-ordinal context"));
  }
  return absl::InvalidArgumentError(absl::StrCat(
      source, ": ", arg_name, " pointer ", absl::StrFormat("%p", ptr),
      " is not a known SYCL USM allocation"));
}

absl::Status ValidateQueueDevicePointer(::sycl::queue* stream_handle,
                                        const void* ptr,
                                        absl::string_view arg_name,
                                        absl::string_view source) {
  ASSIGN_OR_RETURN(::sycl::usm::alloc alloc_type,
                   GetPointerTypeInContext(ptr, stream_handle->get_context(),
                                           source));
  if (alloc_type == ::sycl::usm::alloc::unknown) {
    return UnknownUsmPointerError(ptr, arg_name, source);
  }
  if (alloc_type == ::sycl::usm::alloc::host) {
    return absl::InvalidArgumentError(absl::StrCat(
        source, ": ", arg_name, " pointer ", absl::StrFormat("%p", ptr),
        " is host USM, but device or shared USM is required"));
  }

  try {
    ::sycl::device queue_device = stream_handle->get_device();
    ::sycl::device pointer_device =
        ::sycl::get_pointer_device(ptr, stream_handle->get_context());
    if (pointer_device == queue_device) {
      return absl::OkStatus();
    }
    if (queue_device.ext_oneapi_can_access_peer(
            pointer_device,
            ::sycl::ext::oneapi::peer_access::access_supported)) {
      return absl::OkStatus();
    }
    ASSIGN_OR_RETURN(int queue_ordinal,
                     SyclDevicePool::GetDeviceOrdinal(queue_device));
    ASSIGN_OR_RETURN(int pointer_ordinal,
                     SyclDevicePool::GetDeviceOrdinal(pointer_device));
    return absl::FailedPreconditionError(absl::StrCat(
        source, ": ", arg_name, " pointer ", absl::StrFormat("%p", ptr),
        " belongs to device ordinal ", pointer_ordinal,
        ", but queue device ordinal ", queue_ordinal,
        " cannot access it as peer memory"));
  } catch (const ::sycl::exception& e) {
    return SyclExceptionStatus(source, e);
  }
}

absl::StatusOr<bool> IsHostAccessibleInQueueContext(
    ::sycl::queue* stream_handle, const void* ptr, absl::string_view source) {
  ASSIGN_OR_RETURN(::sycl::usm::alloc alloc_type,
                   GetPointerTypeInContext(ptr, stream_handle->get_context(),
                                           source));
  switch (alloc_type) {
    case ::sycl::usm::alloc::host:
    case ::sycl::usm::alloc::shared:
      return true;
    case ::sycl::usm::alloc::unknown:
      return false;
    case ::sycl::usm::alloc::device:
      return absl::InvalidArgumentError(absl::StrCat(
          source, ": host pointer argument is device USM"));
  }
}

absl::Status EnqueueStagedHostToDevice(::sycl::queue* stream_handle,
                                       void* dst_device,
                                       const void* src_host,
                                       size_t byte_count) {
  void* staging = nullptr;
  try {
    staging = ::sycl::aligned_alloc_host(/*alignment=*/64, byte_count,
                                         *stream_handle);
    if (staging == nullptr) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "MemcpyHostToDevice: failed to allocate ", byte_count,
          " bytes of host USM staging memory"));
    }
    std::memcpy(staging, src_host, byte_count);
    stream_handle->memcpy(dst_device, staging, byte_count);
    ::sycl::context context = stream_handle->get_context();
    stream_handle->submit([staging, context](::sycl::handler& cgh) {
      cgh.host_task([staging, context]() { ::sycl::free(staging, context); });
    });
    return absl::OkStatus();
  } catch (const ::sycl::exception& e) {
    if (staging != nullptr) {
      ::sycl::free(staging, stream_handle->get_context());
    }
    return SyclExceptionStatus("MemcpyHostToDevice", e);
  }
}

absl::Status EnqueueStagedDeviceToHost(::sycl::queue* stream_handle,
                                       void* dst_host,
                                       const void* src_device,
                                       size_t byte_count) {
  void* staging = nullptr;
  try {
    staging = ::sycl::aligned_alloc_host(/*alignment=*/64, byte_count,
                                         *stream_handle);
    if (staging == nullptr) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "MemcpyDeviceToHost: failed to allocate ", byte_count,
          " bytes of host USM staging memory"));
    }
    stream_handle->memcpy(staging, src_device, byte_count);
    ::sycl::context context = stream_handle->get_context();
    stream_handle->submit(
        [staging, dst_host, byte_count, context](::sycl::handler& cgh) {
          cgh.host_task([staging, dst_host, byte_count, context]() {
            std::memcpy(dst_host, staging, byte_count);
            ::sycl::free(staging, context);
          });
        });
    return absl::OkStatus();
  } catch (const ::sycl::exception& e) {
    if (staging != nullptr) {
      ::sycl::free(staging, stream_handle->get_context());
    }
    return SyclExceptionStatus("MemcpyDeviceToHost", e);
  }
}

absl::Status IsValidDeviceOrdinal(int device_ordinal,
                                  const absl::string_view& function_name) {
  ASSIGN_OR_RETURN(int device_count, SyclDevicePool::GetDeviceCount());
  if (device_ordinal >= 0 && device_ordinal < device_count) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      function_name, ": Invalid device ordinal: ", device_ordinal));
}

absl::Status MemcpyDeviceToHost(::sycl::queue* stream_handle, void* dst_host,
                                const void* src_device, size_t byte_count,
                                bool async = false) {
  RETURN_IF_ERROR(ValidateQueueDevicePointer(
      stream_handle, src_device, "source", "MemcpyDeviceToHost"));
  if (async) {
    ASSIGN_OR_RETURN(bool dst_is_host_usm,
                     IsHostAccessibleInQueueContext(
                         stream_handle, dst_host, "MemcpyDeviceToHost"));
    if (!dst_is_host_usm) {
      return EnqueueStagedDeviceToHost(stream_handle, dst_host, src_device,
                                       byte_count);
    }
  }
  try {
    ::sycl::event event =
        stream_handle->memcpy(dst_host, src_device, byte_count);
    if (!async) {
      ASSIGN_OR_RETURN(int device_ordinal,
                       GetDeviceOrdinalForQueue(stream_handle,
                                                "MemcpyDeviceToHost"));
      RETURN_IF_ERROR(SyclEventSynchronize(event, device_ordinal,
                                           "MemcpyDeviceToHost"));
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        "MemcpyDeviceToHost failed: " + std::string(e.what()) +
        ", file = " + __FILE__ + ", line = " + std::to_string(__LINE__) + ".");
  }
  return absl::OkStatus();
}

absl::Status MemcpyHostToDevice(::sycl::queue* stream_handle, void* dst_device,
                                const void* src_host, size_t byte_count,
                                bool async = false) {
  RETURN_IF_ERROR(ValidateQueueDevicePointer(
      stream_handle, dst_device, "destination", "MemcpyHostToDevice"));
  if (async) {
    ASSIGN_OR_RETURN(bool src_is_host_usm,
                     IsHostAccessibleInQueueContext(
                         stream_handle, src_host, "MemcpyHostToDevice"));
    if (!src_is_host_usm) {
      return EnqueueStagedHostToDevice(stream_handle, dst_device, src_host,
                                       byte_count);
    }
  }
  try {
    ::sycl::event event =
        stream_handle->memcpy(dst_device, src_host, byte_count);
    if (!async) {
      ASSIGN_OR_RETURN(int device_ordinal,
                       GetDeviceOrdinalForQueue(stream_handle,
                                                "MemcpyHostToDevice"));
      RETURN_IF_ERROR(SyclEventSynchronize(event, device_ordinal,
                                           "MemcpyHostToDevice"));
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        "MemcpyHostToDevice failed: " + std::string(e.what()) +
        ", file = " + __FILE__ + ", line = " + std::to_string(__LINE__) + ".");
  }
  return absl::OkStatus();
}

absl::Status MemcpyDeviceToDevice(::sycl::queue* stream_handle,
                                  void* dst_device, const void* src_device,
                                  size_t byte_count, bool async = false) {
  RETURN_IF_ERROR(ValidateQueueDevicePointer(
      stream_handle, dst_device, "destination", "MemcpyDeviceToDevice"));
  RETURN_IF_ERROR(ValidateQueueDevicePointer(
      stream_handle, src_device, "source", "MemcpyDeviceToDevice"));
  try {
    ::sycl::event event =
        stream_handle->memcpy(dst_device, src_device, byte_count);
    if (!async) {
      ASSIGN_OR_RETURN(int device_ordinal,
                       GetDeviceOrdinalForQueue(stream_handle,
                                                "MemcpyDeviceToDevice"));
      RETURN_IF_ERROR(SyclEventSynchronize(event, device_ordinal,
                                           "MemcpyDeviceToDevice"));
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        "MemcpyDeviceToDevice failed: " + std::string(e.what()) +
        ", file = " + __FILE__ + ", line = " + std::to_string(__LINE__) + ".");
  }
  return absl::OkStatus();
}

absl::Status MemsetDevice(::sycl::queue* stream_handle, void* dst_device,
                          unsigned char value, size_t count,
                          bool async = false) {
  try {
    ::sycl::event event =
        stream_handle->memset(dst_device, value, count * sizeof(uint8_t));
    if (!async) {
      ASSIGN_OR_RETURN(int device_ordinal,
                       GetDeviceOrdinalForQueue(stream_handle,
                                                "MemsetDevice"));
      RETURN_IF_ERROR(
          SyclEventSynchronize(event, device_ordinal, "MemsetDevice"));
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError("MemsetDevice failed: " + std::string(e.what()) +
                               ", file = " + __FILE__ +
                               ", line = " + std::to_string(__LINE__) + ".");
  }
  return absl::OkStatus();
}

absl::Status MemfillDevice(::sycl::queue* stream_handle, void* dst_device,
                           uint32_t value, size_t count, bool async = false) {
  try {
    ::sycl::event event = stream_handle->fill(dst_device, value, count);
    if (!async) {
      ASSIGN_OR_RETURN(int device_ordinal,
                       GetDeviceOrdinalForQueue(stream_handle,
                                                "MemfillDevice"));
      RETURN_IF_ERROR(
          SyclEventSynchronize(event, device_ordinal, "MemfillDevice"));
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        "MemfillDevice failed: " + std::string(e.what()) +
        ", file = " + __FILE__ + ", line = " + std::to_string(__LINE__) + ".");
  }
  return absl::OkStatus();
}

}  // namespace

DevicePool SyclDevicePool::device_pool_;

absl::Status SyclDevicePool::InitDevicePool() {
  static absl::once_flag device_init_flag;
  static absl::Status init_status = absl::OkStatus();
  absl::call_once(device_init_flag, []() {
    DevicePool devices;
    std::vector<::sycl::platform> platform_list =
        ::sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      std::string platform_name =
          platform.get_info<::sycl::info::platform::name>();
      // Add all Level-Zero backend GPUs to the device pool so that it can be
      // used by the SYCL runtime.
      if (platform_name.find("Level-Zero") != std::string::npos) {
        LOG(INFO) << "Selected platform: " << platform_name;
        std::vector<::sycl::device> device_list = platform.get_devices();
        for (const auto& device : device_list) {
          if (device.is_gpu()) {
            devices.push_back(device);
          }
        }
      }
    }
    if (devices.empty()) {
      init_status = absl::InternalError(
          "SyclDevicePool::InitDevicePool: No SYCL devices found with "
          "Level-Zero "
          "backend. Check oneAPI installation and environment variables.");
      return;
    }
    device_pool_ = std::move(devices);
  });
  return init_status;
}

absl::StatusOr<::sycl::context> SyclDevicePool::GetDeviceContext(
    int device_ordinal) {
  RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclDevicePool::GetDeviceContext"));

  static absl::Mutex contexts_mu(absl::kConstInit);
  static auto* contexts =
      new absl::flat_hash_map<int, std::unique_ptr<::sycl::context>>();

  absl::MutexLock lock(&contexts_mu);
  auto it = contexts->find(device_ordinal);
  if (it == contexts->end()) {
    // Keep USM allocations scoped to one physical device. A multi-device
    // Level Zero context can make independent GPU BFC arenas compete in the
    // same context-level allocation budget.
    auto context =
        std::make_unique<::sycl::context>(
            device_pool_[device_ordinal],
            MakeSyclAsyncHandler(device_ordinal, "SYCL context"));
    it = contexts->emplace(device_ordinal, std::move(context)).first;
  }
  return *it->second;
}

absl::StatusOr<int> SyclDevicePool::GetDeviceCount() {
  RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  // Cast to int since device_ordinal is usually an int.
  return static_cast<int>(device_pool_.size());
}

absl::StatusOr<int> SyclDevicePool::GetDeviceOrdinal(
    const ::sycl::device& device) {
  RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  auto it = std::find(device_pool_.begin(), device_pool_.end(), device);
  if (it != device_pool_.end()) {
    return static_cast<int>(it - device_pool_.begin());
  }
  return absl::InternalError(
      "SyclDevicePool::GetDeviceOrdinal failed, got invalid device");
}

absl::StatusOr<::sycl::device> SyclDevicePool::GetDevice(int device_ordinal) {
  RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclDevicePool::GetDevice"));
  return device_pool_[device_ordinal];
}

StreamPoolMap SyclStreamPool::stream_pool_map_;
absl::Mutex SyclStreamPool::stream_pool_mu_(absl::kConstInit);

absl::StatusOr<StreamPool*> SyclStreamPool::InitStreamPool(int device_ordinal) {
  {
    absl::ReaderMutexLock read_lock(&stream_pool_mu_);
    auto it = stream_pool_map_.find(device_ordinal);
    // Returns the existing non-empty stream pool for this device, if available.
    // The pool may be empty if DestroyStream was called on the last stream.
    if (it != stream_pool_map_.end() && !it->second.empty()) {
      VLOG(2) << "Check 1: Returning existing stream pool for device ordinal "
              << device_ordinal << " whose size is " << it->second.size();
      return &(it->second);
    }
  }
  // Creates a new stream pool for this device using the device and context.
  ::sycl::property_list prop_list{::sycl::property::queue::enable_profiling(),
                                  ::sycl::property::queue::in_order()};
  ASSIGN_OR_RETURN(::sycl::device sycl_device,
                   SyclDevicePool::GetDevice(device_ordinal));
  ASSIGN_OR_RETURN(::sycl::context sycl_context,
                   SyclDevicePool::GetDeviceContext(device_ordinal));

  VLOG(2) << "Creating new stream pool for device ordinal " << device_ordinal;
  absl::MutexLock write_lock(&stream_pool_mu_);
  auto it = stream_pool_map_.find(device_ordinal);
  // Double-checks that another thread has not already created the pool.
  if (it != stream_pool_map_.end() && !it->second.empty()) {
    VLOG(2) << "Check 2: Returning existing stream pool for device ordinal "
            << device_ordinal << " whose size is " << it->second.size();
    return &(it->second);
  }

  StreamPool stream_pool = {std::make_shared<::sycl::queue>(
      sycl_context, sycl_device,
      MakeSyclAsyncHandler(device_ordinal, "SYCL queue"), prop_list)};

  // Use assignment (not insert) to update the stream pool if it was
  // previously destroyed.
  stream_pool_map_[device_ordinal] = std::move(stream_pool);

  return &(stream_pool_map_[device_ordinal]);
}

absl::StatusOr<StreamPtr> SyclStreamPool::GetDefaultStream(int device_ordinal) {
  RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclStreamPool::GetDefaultStream"));
  ASSIGN_OR_RETURN(StreamPool * stream_pool,
                   SyclStreamPool::InitStreamPool(device_ordinal));
  // InitStreamPool always returns a valid pointer, so no null check is needed.
  absl::ReaderMutexLock read_lock(&stream_pool_mu_);
  if (stream_pool->empty()) {
    return absl::InternalError(
        absl::StrCat("SyclStreamPool::GetDefaultStream: Stream pool is empty "
                     "for device ordinal ",
                     device_ordinal,
                     ". The pool may have been destroyed by another thread."));
  }
  return stream_pool->front();
}

absl::StatusOr<StreamPtr> SyclStreamPool::GetOrCreateStream(
    int device_ordinal, bool enable_multiple_streams) {
  VLOG(2) << "SyclStreamPool::GetOrCreateStream called for device ordinal "
          << device_ordinal
          << ", enable_multiple_streams: " << enable_multiple_streams;
  if (!enable_multiple_streams) {
    return SyclStreamPool::GetDefaultStream(device_ordinal);
  }
  RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal,
                                       "SyclStreamPool::GetOrCreateStream"));
  ASSIGN_OR_RETURN(StreamPool * stream_pool,
                   SyclStreamPool::InitStreamPool(device_ordinal));
  // If multiple streams are enabled, create a new stream and add it
  // to the pool, unless the pool has reached kMaxStreamsPerDevice.
  absl::MutexLock write_lock(&stream_pool_mu_);
  if (stream_pool->size() >= kMaxStreamsPerDevice) {
    VLOG(2) << "Stream pool size for device ordinal " << device_ordinal
            << " exceeds the maximum limit of " << kMaxStreamsPerDevice;
    return absl::ResourceExhaustedError(
        absl::StrCat("SyclStreamPool::GetOrCreateStream: Maximum number of "
                     "streams reached for device ordinal ",
                     device_ordinal, "."));
  }
  VLOG(2) << "Stream pool size for device ordinal " << device_ordinal << ": "
          << stream_pool->size();
  ::sycl::property_list prop_list{::sycl::property::queue::enable_profiling(),
                                  ::sycl::property::queue::in_order()};
  ASSIGN_OR_RETURN(::sycl::device sycl_device,
                   SyclDevicePool::GetDevice(device_ordinal));
  ASSIGN_OR_RETURN(::sycl::context sycl_context,
                   SyclDevicePool::GetDeviceContext(device_ordinal));
  stream_pool->push_back(std::make_shared<::sycl::queue>(
      sycl_context, sycl_device,
      MakeSyclAsyncHandler(device_ordinal, "SYCL queue"), prop_list));
  return stream_pool->back();
}

absl::Status SyclStreamPool::SynchronizeStreamPool(int device_ordinal) {
  RETURN_IF_ERROR(IsValidDeviceOrdinal(
      device_ordinal, "SyclStreamPool::SynchronizeStreamPool"));
  ASSIGN_OR_RETURN(StreamPool * stream_pool,
                   SyclStreamPool::InitStreamPool(device_ordinal));
  absl::ReaderMutexLock read_lock(&stream_pool_mu_);
  if (stream_pool->empty()) {
    return absl::InternalError(
        absl::StrCat("SyclStreamPool::SynchronizeStreamPool: Stream pool is "
                     "empty for device ordinal ",
                     device_ordinal,
                     ". The pool may have been destroyed by another thread."));
  }
  absl::Status status = absl::OkStatus();
  for (auto& stream : *stream_pool) {
    status = MergeStatuses(std::move(status),
                           WaitForQueueAndDrain(
                               stream.get(),
                               "SyclStreamPool::SynchronizeStreamPool"));
  }
  return status;
}

absl::Status SyclStreamPool::DestroyStream(int device_ordinal,
                                           StreamPtr& stream_handle) {
  if (stream_handle == nullptr) {
    return absl::InvalidArgumentError(
        "SyclStreamPool::DestroyStream: Attempting to destroy a null stream "
        "handle.");
  }
  RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclStreamPool::DestroyStream"));
  ASSIGN_OR_RETURN(StreamPool * stream_pool,
                   SyclStreamPool::InitStreamPool(device_ordinal));
  absl::MutexLock write_lock(&stream_pool_mu_);
  if (stream_pool->empty()) {
    return absl::InternalError(
        absl::StrCat("SyclStreamPool::DestroyStream: Stream pool is empty for "
                     "device ordinal ",
                     device_ordinal,
                     ". The pool may have been destroyed by another thread."));
  }
  auto it = std::find(stream_pool->begin(), stream_pool->end(), stream_handle);
  if (it == stream_pool->end()) {
    return absl::NotFoundError(absl::StrCat(
        "SyclStreamPool::DestroyStream: Stream handle for device ordinal ",
        device_ordinal, " not found in the pool."));
  }
  RETURN_IF_ERROR(WaitForQueueAndDrain(stream_handle.get(),
                                       "SyclStreamPool::DestroyStream"));
  // Remove the stream from the pool and reset the handle.
  // The stream pool remains, but may become empty.
  stream_pool->erase(it);
  stream_handle.reset();
  VLOG(2) << "Successfully destroyed stream for device ordinal "
          << device_ordinal << ", stream pool size is " << stream_pool->size();
  return absl::OkStatus();
}

void SyclStreamPool::Reset() {
  absl::MutexLock write_lock(&stream_pool_mu_);
  for (auto& [device_ordinal, stream_pool] : stream_pool_map_) {
    for (auto& stream_handle : stream_pool) {
      if (stream_handle) {
        absl::Status status =
            WaitForQueueAndDrain(stream_handle.get(), "SyclStreamPool::Reset");
        if (!status.ok()) {
          LOG(FATAL) << status;
        }
        stream_handle.reset();
      }
    }
    stream_pool.clear();
  }
  stream_pool_map_.clear();
}

absl::StatusOr<SyclTimerProperties> SyclGetTimerProperties(int device_ordinal) {
  RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclGetTimerProperties"));
  ASSIGN_OR_RETURN(::sycl::device device,
                   SyclDevicePool::GetDevice(device_ordinal));
  ze_device_handle_t lz_device_handle =
      ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(device);
  ze_device_properties_t lz_device_props{
      // timerResolution will be in cycles/sec (Hz) with this structure type.
      ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2,
  };
  ze_result_t status =
      zeDeviceGetProperties(lz_device_handle, &lz_device_props);
  if (status != ZE_RESULT_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("SyclGetTimerProperties: zeDeviceGetProperties failed for "
                     "device ordinal ",
                     device_ordinal, " with return code: ", status));
  }
  uint64_t timer_freq_hz = lz_device_props.timerResolution;
  uint32_t kernel_ts_valid_bits = lz_device_props.kernelTimestampValidBits;
  uint64_t timestamp_mask = 0;
  if (kernel_ts_valid_bits == 0 || kernel_ts_valid_bits > 64) {
    return absl::InternalError(absl::StrCat(
        "SyclGetTimerProperties: Invalid kernel timestamp valid bits (",
        kernel_ts_valid_bits, ") for device ordinal ", device_ordinal));
  } else if (kernel_ts_valid_bits < 64) {
    timestamp_mask = (1ull << kernel_ts_valid_bits) - 1ull;
  } else {
    // Prevent overflow when shifting by 64.
    timestamp_mask = ~0ull;
  }
  return SyclTimerProperties{timer_freq_hz, timestamp_mask};
}

absl::Status SyclStreamSynchronize(::sycl::queue* stream_handle) {
  return WaitForQueueAndDrain(stream_handle, "SyclStreamSynchronize");
}

absl::Status SyclEventSynchronize(::sycl::event event, int device_ordinal,
                                  absl::string_view source) {
  absl::Status wait_status = absl::OkStatus();
  try {
    event.wait_and_throw();
  } catch (const ::sycl::exception& e) {
    wait_status = SyclExceptionStatus(source, e);
  }
  return MergeStatuses(wait_status, DrainAsyncErrors(device_ordinal, source));
}

void SyclRecordAsyncErrorForTesting(int device_ordinal, absl::Status status) {
  RecordAsyncError(device_ordinal, std::move(status));
}

absl::StatusOr<MemorySpace> SyclGetPointerMemorySpace(int device_ordinal,
                                                      const void* ptr) {
  if (ptr == nullptr) {
    return absl::InvalidArgumentError(
        "SyclGetPointerMemorySpace: pointer is null");
  }
  RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclGetPointerMemorySpace"));
  ASSIGN_OR_RETURN(::sycl::context context,
                   SyclDevicePool::GetDeviceContext(device_ordinal));
  ASSIGN_OR_RETURN(
      ::sycl::usm::alloc alloc_type,
      GetPointerTypeInContext(ptr, context, "SyclGetPointerMemorySpace"));
  return MemorySpaceFromUsmAlloc(alloc_type, "SyclGetPointerMemorySpace");
}

absl::StatusOr<::sycl::event> SyclSubmitBarrierEvent(
    ::sycl::queue* stream_handle) {
  try {
    // Record a fresh marker event. A barrier is an explicit SYCL command whose
    // event denotes completion of all prior work in this in-order queue.
    return stream_handle->ext_oneapi_submit_barrier();
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(absl::StrCat(
        "SyclSubmitBarrierEvent: Failed to submit barrier event: ", e.what(),
        ", file = ", __FILE__, ", line = ", __LINE__));
  }
}

absl::Status SyclMemcpyAsync(::sycl::queue* stream_handle, void* dst,
                             const void* src, size_t byte_count,
                             SyclMemcpyKind kind) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMemcpyAsync: Attempting to copy zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst == nullptr || src == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyAsync: Null pointer provided for destination or source.");
  }
  switch (kind) {
    case SyclMemcpyKind::kSyclMemcpyDeviceToHost:
      return MemcpyDeviceToHost(stream_handle, dst, src, byte_count,
                                /*async=*/true);
    case SyclMemcpyKind::kSyclMemcpyHostToDevice:
      return MemcpyHostToDevice(stream_handle, dst, src, byte_count,
                                /*async=*/true);
    case SyclMemcpyKind::kSyclMemcpyDeviceToDevice:
      return MemcpyDeviceToDevice(stream_handle, dst, src, byte_count,
                                  /*async=*/true);
    default:
      return absl::InvalidArgumentError(
          "SyclMemcpyAsync: Invalid SyclMemcpyKind provided.");
  }
}

absl::Status SyclMemcpyDeviceToHost(int device_ordinal, void* dst_host,
                                    const void* src_device, size_t byte_count) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMemcpyDeviceToHost: Attempting to copy zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_host == nullptr || src_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyDeviceToHost: Null pointer provided for destination or "
        "source.");
  }
  RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclMemcpyDeviceToHost"));
  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetDefaultStream(device_ordinal));
  return MemcpyDeviceToHost(stream_handle.get(), dst_host, src_device,
                            byte_count);
}

absl::Status SyclMemcpyHostToDevice(int device_ordinal, void* dst_device,
                                    const void* src_host, size_t byte_count) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMemcpyHostToDevice: Attempting to copy zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_device == nullptr || src_host == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyHostToDevice: Null pointer provided for destination or "
        "source.");
  }
  RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclMemcpyHostToDevice"));
  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetDefaultStream(device_ordinal));
  return MemcpyHostToDevice(stream_handle.get(), dst_device, src_host,
                            byte_count);
}

absl::Status SyclMemcpyDeviceToDevice(int device_ordinal, void* dst_device,
                                      const void* src_device,
                                      size_t byte_count) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMemcpyDeviceToDevice: Attempting to copy zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_device == nullptr || src_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyDeviceToDevice: Null pointer provided for destination or "
        "source.");
  }
  RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclMemcpyDeviceToDevice"));
  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetDefaultStream(device_ordinal));
  return MemcpyDeviceToDevice(stream_handle.get(), dst_device, src_device,
                              byte_count);
}

absl::Status SyclMemcpyDeviceToHostAsync(::sycl::queue* stream_handle,
                                         void* dst_host, const void* src_device,
                                         size_t byte_count) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMemcpyDeviceToHostAsync: Attempting to copy zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_host == nullptr || src_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyDeviceToHostAsync: Null pointer provided for destination or "
        "source.");
  }
  ::sycl::usm::alloc dst_alloc_type =
      ::sycl::get_pointer_type(dst_host, stream_handle->get_context());
  bool async = (dst_alloc_type == ::sycl::usm::alloc::host);
  return MemcpyDeviceToHost(stream_handle, dst_host, src_device, byte_count,
                            async);
}

absl::Status SyclMemcpyHostToDeviceAsync(::sycl::queue* stream_handle,
                                         void* dst_device, const void* src_host,
                                         size_t byte_count) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMemcpyHostToDeviceAsync: Attempting to copy zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_device == nullptr || src_host == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyHostToDeviceAsync: Null pointer provided for destination or "
        "source.");
  }
  ::sycl::usm::alloc src_alloc_type =
      ::sycl::get_pointer_type(src_host, stream_handle->get_context());
  bool async = (src_alloc_type == ::sycl::usm::alloc::host);
  return MemcpyHostToDevice(stream_handle, dst_device, src_host, byte_count,
                            async);
}

absl::Status SyclMemcpyDeviceToDeviceAsync(::sycl::queue* stream_handle,
                                           void* dst_device,
                                           const void* src_device,
                                           size_t byte_count) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMemcpyDeviceToDeviceAsync: Attempting to copy zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_device == nullptr || src_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemcpyDeviceToDeviceAsync: Null pointer provided for destination "
        "or source.");
  }
  return MemcpyDeviceToDevice(stream_handle, dst_device, src_device, byte_count,
                              /*async=*/true);
}

absl::Status SyclMemsetDevice(int device_ordinal, void* dst_device,
                              unsigned char value, size_t count) {
  if (count == 0) {
    VLOG(2) << "SyclMemsetDevice: Attempting to set zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemsetDevice: Null pointer provided for destination.");
  }
  RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMemsetDevice"));
  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetDefaultStream(device_ordinal));
  return MemsetDevice(stream_handle.get(), dst_device, value, count);
}

absl::Status SyclMemsetDeviceAsync(::sycl::queue* stream_handle,
                                   void* dst_device, unsigned char value,
                                   size_t count) {
  if (count == 0) {
    VLOG(2) << "SyclMemsetDeviceAsync: Attempting to set zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemsetDeviceAsync: Null pointer provided for destination handle.");
  }
  return MemsetDevice(stream_handle, dst_device, value, count, /*async=*/true);
}

absl::Status SyclMemfillDevice(int device_ordinal, void* dst_device,
                               uint32_t value, size_t count) {
  if (count == 0) {
    VLOG(2) << "SyclMemfillDevice: Attempting to fill zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemfillDevice: Null pointer provided for destination.");
  }
  RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMemfillDevice"));
  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetDefaultStream(device_ordinal));
  return MemfillDevice(stream_handle.get(), dst_device, value, count);
}

absl::Status SyclMemfillDeviceAsync(::sycl::queue* stream_handle,
                                    void* dst_device, uint32_t value,
                                    size_t count) {
  if (count == 0) {
    VLOG(2) << "SyclMemfillDeviceAsync: Attempting to fill zero bytes, "
               "skipping operation.";
    return absl::OkStatus();
  }
  if (dst_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemfillDeviceAsync: Null pointer provided for destination handle");
  }
  return MemfillDevice(stream_handle, dst_device, value, count, /*async=*/true);
}

absl::StatusOr<void*> SyclMallocDevice(int device_ordinal, size_t byte_count) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMallocDevice: Attempting to allocate zero bytes, "
               "returning nullptr.";
    return nullptr;
  }
  RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMallocDevice"));
  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetDefaultStream(device_ordinal));
  try {
    // Use the default stream to allocate memory
    void* ptr = ::sycl::aligned_alloc_device(/*alignment=*/64, byte_count,
                                             *stream_handle);
    if (ptr == nullptr) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "SyclMallocDevice: Failed to allocate ", byte_count,
          " bytes of device memory for device ordinal ", device_ordinal,
          ": SYCL returned nullptr."));
    }
    return ptr;
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat(
        "SyclMallocDevice: Failed to allocate device memory: ", e.what(),
        ", file = ", __FILE__, ", line = ", __LINE__));
  }
}

absl::StatusOr<void*> SyclMallocHost(int device_ordinal, size_t byte_count) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMallocHost: Attempting to allocate zero bytes, "
               "returning nullptr.";
    return nullptr;
  }
  RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMallocHost"));
  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetDefaultStream(device_ordinal));
  try {
    // Use the default stream to allocate memory
    void* ptr = ::sycl::aligned_alloc_host(/*alignment=*/64, byte_count,
                                           *stream_handle);
    if (ptr == nullptr) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "SyclMallocHost: Failed to allocate ", byte_count,
          " bytes of host memory for device ordinal ", device_ordinal,
          ": SYCL returned nullptr."));
    }
    return ptr;
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat(
        "SyclMallocHost: Failed to allocate host memory: ", e.what(),
        ", file = ", __FILE__, ", line = ", __LINE__));
  }
}

absl::StatusOr<void*> SyclMallocShared(int device_ordinal, size_t byte_count) {
  if (byte_count == 0) {
    VLOG(2) << "SyclMallocShared: Attempting to allocate zero bytes, "
               "returning nullptr.";
    return nullptr;
  }
  RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMallocShared"));
  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetDefaultStream(device_ordinal));
  try {
    // Use the default stream to allocate memory
    void* ptr = ::sycl::aligned_alloc_shared(/*alignment=*/64, byte_count,
                                             *stream_handle);
    if (ptr == nullptr) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "SyclMallocShared: Failed to allocate ", byte_count,
          " bytes of shared memory for device ordinal ", device_ordinal,
          ": SYCL returned nullptr."));
    }
    return ptr;
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat(
        "SyclMallocShared: Failed to allocate shared memory: ", e.what(),
        ", file = ", __FILE__, ", line = ", __LINE__));
  }
}

absl::Status SyclFree(int device_ordinal, void*& ptr) {
  if (ptr == nullptr) {
    return absl::InvalidArgumentError(
        "SyclFree: Attempting to free a null pointer.");
  }
  RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclFree"));
  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetDefaultStream(device_ordinal));
  try {
    // Use the default stream to free memory
    ::sycl::free(ptr, *stream_handle);
    ptr = nullptr;
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("SyclFree: Failed to free memory: ", e.what(),
                     ", file = ", __FILE__, ", line = ", __LINE__));
  }
  return absl::OkStatus();
}

}  // namespace stream_executor::sycl
