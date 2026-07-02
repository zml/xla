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

#include "xla/stream_executor/sycl/sycl_stream.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::sycl {

namespace {

struct HostCallbackTask {
  absl::AnyInvocable<absl::Status() &&> callback;
  absl::AnyInvocable<void(absl::Status) &&> error_cb;
};

void RunHostCallbackTask(std::shared_ptr<HostCallbackTask> task) {
  absl::Status status = std::move(task->callback)();
  if (status.ok()) {
    return;
  }

  if (task->error_cb) {
    std::move(task->error_cb)(status);
  } else {
    LOG(WARNING) << "Host callback failed: " << status;
  }
}

absl::Status LaunchSyclKernel(
    StreamExecutor* executor, absl::string_view kernel_name,
    ::sycl::kernel* function, unsigned int thread_dim_x,
    unsigned int thread_dim_y, unsigned int thread_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    ::sycl::queue* stream_handle, void** kernel_params, void** extra) {
  VLOG(2) << "Launching kernel '" << kernel_name << "' using ThreadDim{"
          << thread_dim_x << ", " << thread_dim_y << ", " << thread_dim_z
          << "}, BlockDim{" << block_dim_x << ", " << block_dim_y << ", "
          << block_dim_z << "}; function: " << (const void*)function;

  if (function == nullptr) {
    return absl::InvalidArgumentError(
        "LaunchSyclKernel: No kernel function provided.");
  }

  if (stream_handle == nullptr) {
    return absl::InvalidArgumentError(
        "LaunchSyclKernel: No stream handle provided.");
  }

  ::sycl::range<3> global_range(block_dim_z * thread_dim_z,
                                block_dim_y * thread_dim_y,
                                block_dim_x * thread_dim_x);
  ::sycl::range<3> local_range(thread_dim_z, thread_dim_y, thread_dim_x);
  ::sycl::nd_range<3> nd_range(global_range, local_range);

  void** arg_ptrs = nullptr;
  size_t num_args = 0;
  if (kernel_params != nullptr) {
    // kernel_params is expected to be an array of two pointers:
    // kernel_params[0]: pointer to array of kernel argument pointers
    // kernel_params[1]: pointer to number of kernel arguments
    if (kernel_params[0] == nullptr || kernel_params[1] == nullptr) {
      return absl::InvalidArgumentError(
          "LaunchSyclKernel: kernel_params[0] or kernel_params[1] is null, "
          "cannot set kernel arguments.");
    }

    arg_ptrs = static_cast<void**>(kernel_params[0]);
    size_t* num_args_ptr = static_cast<size_t*>(kernel_params[1]);
    num_args = num_args_ptr ? *num_args_ptr : 0;
  }

  size_t reflected_total_args =
      num_args + (shared_mem_bytes > 0 ? size_t{1} : size_t{0});
  try {
    reflected_total_args =
        function->get_info<::sycl::info::kernel::num_args>();
    size_t required_args =
        num_args + (shared_mem_bytes > 0 ? size_t{1} : size_t{0});
    if (shared_mem_bytes > 0) {
      if (reflected_total_args < required_args) {
        return absl::InternalError(absl::StrCat(
            "LaunchSyclKernel: kernel '", kernel_name,
            "' reports ", reflected_total_args,
            " reflected arguments but requires at least ", required_args,
            " for ", shared_mem_bytes, " bytes of shared memory."));
      }
    } else {
      required_args = num_args;
    }
    if (required_args > reflected_total_args) {
      return absl::InvalidArgumentError(
          absl::StrCat("LaunchSyclKernel: kernel '", kernel_name, "' has ",
                       num_args,
                       " packed arguments, but reflected SYCL arity is only ",
                       reflected_total_args, "."));
    }
    size_t max_packed_args =
        reflected_total_args - (shared_mem_bytes > 0 ? size_t{1} : size_t{0});
    if (num_args > max_packed_args) {
      VLOG(1) << "LaunchSyclKernel: kernel '" << kernel_name << "' has "
              << num_args << " packed regular arguments, but reflected SYCL "
              << "arity can accept " << max_packed_args
              << "; ignoring trailing packed arguments.";
      num_args = max_packed_args;
    }
  } catch (const ::sycl::exception& e) {
    VLOG(1) << "LaunchSyclKernel: failed to reflect argument count for kernel '"
            << kernel_name << "': " << e.what()
            << "; using packed argument count " << num_args;
  }

  try {
    stream_handle->submit([&](::sycl::handler& cgh) {
      for (size_t arg_index = 0; arg_index < num_args; ++arg_index) {
        if (arg_ptrs[arg_index] == nullptr) {
          LOG(ERROR) << "LaunchSyclKernel: kernel argument " << arg_index
                     << " is null, cannot set kernel argument.";
          return;
        }
        VLOG(2) << "Setting kernel argument " << arg_index
                << " at address: " << arg_ptrs[arg_index];
        cgh.set_arg(arg_index, arg_ptrs[arg_index]);
      }

      if (num_args == 0) {
        VLOG(2) << "LaunchSyclKernel: No kernel arguments to set; launching "
                   "kernel.";
      }
      const size_t local_arg_index = shared_mem_bytes > 0
                                         ? reflected_total_args - 1
                                         : reflected_total_args;
      for (size_t arg_index = num_args; arg_index < local_arg_index;
           ++arg_index) {
        void* scratch_arg = nullptr;
        VLOG(2) << "Setting trailing scratch argument " << arg_index
                << " to null.";
        cgh.set_arg(arg_index, scratch_arg);
      }
      if (shared_mem_bytes > 0) {
        using shared_mem_t = ::sycl::local_accessor<int8_t, 1>;
        shared_mem_t local_buffer(shared_mem_bytes, cgh);
        VLOG(2) << "Setting shared memory argument " << local_arg_index
                << " with " << shared_mem_bytes << " bytes.";
        cgh.set_arg(local_arg_index, local_buffer);
      }
      cgh.parallel_for(nd_range, *function);
    });
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("LaunchSyclKernel: failed to submit kernel '",
                     kernel_name, "': ", e.what()));
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("LaunchSyclKernel: failed to submit kernel '",
                     kernel_name, "': ", e.what()));
  } catch (...) {
    return absl::InternalError(
        absl::StrCat("LaunchSyclKernel: failed to submit kernel '",
                     kernel_name, "': unknown exception"));
  }
  return absl::OkStatus();
}

absl::Status LaunchSyclKernel(
    StreamExecutor* executor, absl::string_view kernel_name,
    ::sycl::kernel* function, unsigned int cluster_dim_x,
    unsigned int cluster_dim_y, unsigned int cluster_dim_z,
    unsigned int thread_dim_x, unsigned int thread_dim_y,
    unsigned int thread_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, ::sycl::queue* stream_handle,
    void** kernel_params, void** extra) {
  if (cluster_dim_x != 1 || cluster_dim_y != 1 || cluster_dim_z != 1)
    return absl::UnimplementedError(
        "LaunchSyclKernel: Non-default cluster dimensions are not supported.");
  return LaunchSyclKernel(executor, kernel_name, function, thread_dim_x,
                          thread_dim_y, thread_dim_z, block_dim_x, block_dim_y,
                          block_dim_z, shared_mem_bytes, stream_handle,
                          kernel_params, extra);
}

}  // namespace

absl::Status SyclStream::WaitFor(Stream* other) {
  SyclStream* other_stream = static_cast<SyclStream*>(other);
  RETURN_IF_ERROR(other_stream->RecordCompletedEvent());
  return SyclEvent::WaitStreamOnEvent(executor_, stream_handle_.get(),
                                      other_stream->completed_event_);
}

absl::Status SyclStream::RecordEvent(Event* event) {
  ASSIGN_OR_RETURN(::sycl::event barrier_event,
                   SyclSubmitBarrierEvent(stream_handle_.get()));
  // Update the event to a barrier that marks completion of prior stream work.
  return static_cast<SyclEvent*>(event)->SetRecordedEvent(
      barrier_event, stream_handle_.get(), /*is_barrier_marker=*/true);
}

absl::Status SyclStream::WaitFor(Event* event) {
  return SyclEvent::WaitStreamOnEvent(executor_, stream_handle_.get(),
                                      *static_cast<SyclEvent*>(event));
}

absl::Status SyclStream::Memset32(DeviceAddressBase* location, uint32_t pattern,
                                  uint64_t size) {
  VLOG(2) << "Enqueuing memset32 operation onto stream " << stream_handle_.get()
          << " at location " << reinterpret_cast<const void*>(location)
          << " with size " << size << " and pattern 0x" << std::hex << pattern
          << std::dec;
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) != 0) {
    return absl::InvalidArgumentError("Location must be 4 byte aligned");
  }
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("Size must be a multiple of 4 bytes.");
  }
  RETURN_IF_ERROR(SyclMemfillDeviceAsync(stream_handle_.get(),
                                         const_cast<void*>(location->opaque()),
                                         pattern, size / sizeof(uint32_t)));
  VLOG(2) << "Successfully enqueued async memset32 of "
          << size / sizeof(uint32_t) << " uint32s at " << location
          << " with value 0x" << std::hex << pattern << std::dec
          << " on stream " << stream_handle_;
  return absl::OkStatus();
}

absl::Status SyclStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return SyclStream::Memset32(location, 0x0, size);
  }
  RETURN_IF_ERROR(SyclMemsetDeviceAsync(
      stream_handle_.get(), const_cast<void*>(location->opaque()), 0x0, size));
  VLOG(2) << "Successfully enqueued async memset8 of " << size << " bytes at "
          << location << " with value 0x0 on stream " << stream_handle_;
  return absl::OkStatus();
}

absl::Status SyclStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const void* host_src, uint64_t size) {
  RETURN_IF_ERROR(SyclMemcpyHostToDeviceAsync(
      stream_handle_.get(), const_cast<void*>(gpu_dst->opaque()), host_src,
      size));
  VLOG(2) << "Successfully enqueued async memcpy H2D of " << size << " bytes"
          << " from " << host_src << " to " << gpu_dst << " on stream "
          << stream_handle_;
  return absl::OkStatus();
}

absl::Status SyclStream::Memcpy(void* host_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  RETURN_IF_ERROR(
      SyclMemcpyDeviceToHostAsync(stream_handle_.get(), host_dst,
                                  const_cast<void*>(gpu_src.opaque()), size));
  VLOG(2) << "Successfully enqueued async memcpy D2H of " << size << " bytes"
          << " from " << gpu_src.opaque() << " to " << host_dst << " on stream "
          << stream_handle_;
  return absl::OkStatus();
}

absl::Status SyclStream::Memcpy(DeviceAddressBase* gpu_dst,
                                const DeviceAddressBase& gpu_src,
                                uint64_t size) {
  RETURN_IF_ERROR(SyclMemcpyDeviceToDeviceAsync(
      stream_handle_.get(), const_cast<void*>(gpu_dst->opaque()),
      const_cast<void*>(gpu_src.opaque()), size));
  VLOG(2) << "Successfully enqueued async memcpy D2D of " << size << " bytes"
          << " from " << gpu_src.opaque() << " to " << gpu_dst << " on stream "
          << stream_handle_;
  return absl::OkStatus();
}

absl::Status SyclStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  return DoHostCallbackWithStatus(std::move(callback), /*error_cb=*/nullptr);
}

absl::Status SyclStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_cb) {
  auto task = std::make_shared<HostCallbackTask>();
  task->callback = std::move(callback);
  task->error_cb = std::move(error_cb);
  try {
    stream_handle_->submit([task](::sycl::handler& cgh) {
      cgh.host_task([task]() { RunHostCallbackTask(task); });
    });
  } catch (const ::sycl::exception& e) {
    absl::Status status = absl::InternalError(absl::StrCat(
        "DoHostCallbackWithStatus: failed to submit host task: ", e.what()));
    if (task->error_cb) {
      std::move(task->error_cb)(status);
    }
    return status;
  } catch (const std::exception& e) {
    absl::Status status = absl::InternalError(absl::StrCat(
        "DoHostCallbackWithStatus: failed to submit host task: ", e.what()));
    if (task->error_cb) {
      std::move(task->error_cb)(status);
    }
    return status;
  } catch (...) {
    absl::Status status = absl::InternalError(
        "DoHostCallbackWithStatus: failed to submit host task: unknown "
        "exception");
    if (task->error_cb) {
      std::move(task->error_cb)(status);
    }
    return status;
  }
  return absl::OkStatus();
}

absl::Status SyclStream::BlockHostUntilDone() {
  return SyclStreamSynchronize(stream_handle_.get());
}

absl::StatusOr<std::unique_ptr<SyclStream>> SyclStream::Create(
    StreamExecutor* executor, bool enable_multiple_streams,
    std::optional<std::variant<StreamPriority, int>> priority) {
  // Determine stream priority.
  int stream_priority = 0;
  if (priority.has_value()) {
    if (std::holds_alternative<int>(priority.value())) {
      stream_priority = std::get<int>(priority.value());
    } else if (std::get<StreamPriority>(priority.value()) ==
               StreamPriority::Default) {
      stream_priority = 0;
    } else {
      return absl::UnimplementedError(
          "SyclStream::Create: SYCL does not support non-default (non-zero) "
          "stream priority.");
    }
  }

  VLOG(0) << "Creating stream for device ordinal " << executor->device_ordinal()
          << (enable_multiple_streams ? " with" : " without")
          << " multiple streams enabled";

  ASSIGN_OR_RETURN(StreamPtr stream_handle,
                   SyclStreamPool::GetOrCreateStream(executor->device_ordinal(),
                                                     enable_multiple_streams));

  ASSIGN_OR_RETURN(SyclEvent completed_event, SyclEvent::Create(executor));

  return std::unique_ptr<SyclStream>(new SyclStream(
      executor, std::move(completed_event), priority, stream_handle));
}

SyclStream::~SyclStream() {
  // Wait for all pending operations to complete before destroying the stream.
  absl::Status block_status = BlockHostUntilDone();
  if (!block_status.ok()) {
    LOG(FATAL) << "Failed to synchronize stream " << stream_handle_.get()
               << " for device " << executor_->device_ordinal()
               << " before destruction: " << block_status;
  }

  // Remove this stream from the executor's list of allocated streams.
  executor_->DeallocateStream(this);

  // Destroy the underlying SYCL stream.
  absl::Status destroy_status = SyclStreamPool::DestroyStream(
      executor_->device_ordinal(), stream_handle_);
  if (!destroy_status.ok()) {
    LOG(ERROR) << "Failed to destroy stream " << stream_handle_.get()
               << " for device " << executor_->device_ordinal() << ", got "
               << destroy_status;
  } else {
    VLOG(2) << "Successfully destroyed stream " << stream_handle_.get()
            << " for device " << executor_->device_ordinal();
    stream_handle_ = nullptr;
  }
}

absl::Status SyclStream::RecordCompletedEvent() {
  return RecordEvent(&completed_event_);
}

absl::Status SyclStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes, bool use_pdl) {
  if (cluster_dims.has_value()) {
    return LaunchSyclKernel(
        executor_, name, static_cast<::sycl::kernel*>(function),
        cluster_dims->x, cluster_dims->y, cluster_dims->z, thread_dims.x,
        thread_dims.y, thread_dims.z, block_dims.x, block_dims.y, block_dims.z,
        shmem_bytes, stream_handle_.get(), args, /*extra=*/nullptr);
  }
  return LaunchSyclKernel(
      executor_, name, static_cast<::sycl::kernel*>(function), thread_dims.x,
      thread_dims.y, thread_dims.z, block_dims.x, block_dims.y, block_dims.z,
      shmem_bytes, stream_handle_.get(), args, /*extra=*/nullptr);
}

}  // namespace stream_executor::sycl
