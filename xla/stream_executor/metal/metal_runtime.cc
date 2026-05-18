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

#include "xla/stream_executor/metal/metal_runtime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::metal {
namespace {

template <typename T>
T Obj(void* object) {
  return (__bridge T)object;
}

void* RetainObj(id object) { return (__bridge_retained void*)object; }

std::string NSStringToString(NSString* string) {
  if (string == nil) return "";
  const char* utf8 = [string UTF8String];
  return utf8 == nullptr ? "" : std::string(utf8);
}

absl::StatusOr<NSArray<id<MTLDevice>>*> CopyDevices() {
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
  if ([devices count] == 0) {
    id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
    if (default_device == nil) {
      return absl::NotFoundError("No Metal devices are visible.");
    }
    devices = @[ default_device ];
  }
  return devices;
}

absl::StatusOr<id<MTLDevice>> DeviceAtOrdinal(int ordinal) {
  if (ordinal < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Metal device ordinal must be non-negative, got ",
                     ordinal));
  }
  TF_ASSIGN_OR_RETURN(NSArray<id<MTLDevice>>* devices, CopyDevices());
  if (ordinal >= [devices count]) {
    return absl::NotFoundError(absl::StrCat(
        "Metal device ordinal ", ordinal, " is outside visible device count ",
        static_cast<int>([devices count])));
  }
  return [devices objectAtIndex:ordinal];
}

std::string ErrorMessage(NSError* error) {
  if (error == nil) return "unknown Metal error";
  return NSStringToString([error localizedDescription]);
}

}  // namespace

int GetDeviceCount() {
  auto devices = CopyDevices();
  if (!devices.ok()) return 0;
  return static_cast<int>([*devices count]);
}

absl::StatusOr<MetalDeviceInfo> GetDeviceInfo(int ordinal) {
  TF_ASSIGN_OR_RETURN(id<MTLDevice> device, DeviceAtOrdinal(ordinal));
  MetalDeviceInfo info;
  info.name = NSStringToString([device name]);
  info.registry_id = absl::StrCat([device registryID]);
  if (@available(macOS 10.15, *)) {
    info.recommended_max_working_set_size =
        [device recommendedMaxWorkingSetSize];
    info.has_unified_memory = [device hasUnifiedMemory];
  }
  info.max_buffer_length = [device maxBufferLength];
  info.max_threads_per_threadgroup =
      [device maxThreadsPerThreadgroup].width *
      [device maxThreadsPerThreadgroup].height *
      [device maxThreadsPerThreadgroup].depth;
  return info;
}

absl::StatusOr<void*> RetainDevice(int ordinal) {
  TF_ASSIGN_OR_RETURN(id<MTLDevice> device, DeviceAtOrdinal(ordinal));
  return RetainObj(device);
}

absl::StatusOr<void*> NewCommandQueue(void* device) {
  id<MTLCommandQueue> queue = [Obj<id<MTLDevice>>(device) newCommandQueue];
  if (queue == nil) {
    return absl::InternalError("Failed to create Metal command queue.");
  }
  return RetainObj(queue);
}

void* RetainObject(void* object) {
  if (object != nullptr) {
    CFRetain(object);
  }
  return object;
}

void ReleaseObject(void* object) {
  if (object != nullptr) {
    CFRelease(object);
  }
}

absl::StatusOr<void*> NewSharedBuffer(void* device, uint64_t size,
                                      void** contents) {
  if (contents == nullptr) {
    return absl::InvalidArgumentError("contents must not be null");
  }
  if (size == 0) {
    *contents = nullptr;
    return nullptr;
  }
  MTLResourceOptions options = MTLResourceStorageModeShared;
  id<MTLBuffer> buffer = [Obj<id<MTLDevice>>(device)
      newBufferWithLength:size
                  options:options];
  if (buffer == nil) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Failed to allocate Metal shared buffer of ", size,
                     " bytes."));
  }
  *contents = [buffer contents];
  return RetainObj(buffer);
}

absl::StatusOr<void*> CompileLibrary(void* device, absl::string_view source) {
  NSString* msl =
      [[NSString alloc] initWithBytes:source.data()
                               length:source.size()
                             encoding:NSUTF8StringEncoding];
  if (msl == nil) {
    return absl::InvalidArgumentError("MSL source is not valid UTF-8.");
  }

  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  NSError* error = nil;
  id<MTLLibrary> library =
      [Obj<id<MTLDevice>>(device) newLibraryWithSource:msl
                                               options:options
                                                 error:&error];
  if (library == nil) {
    return absl::InternalError(
        absl::StrCat("Metal failed to compile MSL source: ",
                     ErrorMessage(error)));
  }
  return RetainObj(library);
}

absl::StatusOr<void*> NewFunction(void* library, absl::string_view name) {
  NSString* function_name =
      [[NSString alloc] initWithBytes:name.data()
                               length:name.size()
                             encoding:NSUTF8StringEncoding];
  if (function_name == nil) {
    return absl::InvalidArgumentError("Metal function name is not valid UTF-8.");
  }
  id<MTLFunction> function =
      [Obj<id<MTLLibrary>>(library) newFunctionWithName:function_name];
  if (function == nil) {
    return absl::NotFoundError(
        absl::StrCat("Metal library does not contain kernel '", name, "'."));
  }
  return RetainObj(function);
}

absl::StatusOr<void*> NewComputePipeline(void* device, void* function) {
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline =
      [Obj<id<MTLDevice>>(device) newComputePipelineStateWithFunction:
                                  Obj<id<MTLFunction>>(function)
                                                               error:&error];
  if (pipeline == nil) {
    return absl::InternalError(
        absl::StrCat("Metal failed to create compute pipeline: ",
                     ErrorMessage(error)));
  }
  return RetainObj(pipeline);
}

absl::StatusOr<void*> Launch(void* command_queue, void* pipeline,
                             void* function, bool use_argument_buffer,
                             absl::Span<const MetalKernelArgument> arguments,
                             const ThreadDim& thread_dims,
                             const BlockDim& block_dims,
                             int64_t shmem_bytes) {
  if (shmem_bytes != 0) {
    return absl::UnimplementedError(
        "Dynamic threadgroup memory is not implemented for Metal yet.");
  }

  id<MTLCommandBuffer> command_buffer =
      [Obj<id<MTLCommandQueue>>(command_queue) commandBuffer];
  if (command_buffer == nil) {
    return absl::InternalError("Failed to create Metal command buffer.");
  }

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer computeCommandEncoder];
  if (encoder == nil) {
    return absl::InternalError("Failed to create Metal compute encoder.");
  }
  [encoder setComputePipelineState:Obj<id<MTLComputePipelineState>>(pipeline)];
  if (use_argument_buffer) {
    if (function == nullptr) {
      return absl::InvalidArgumentError(
          "Metal argument-buffer launch requires a function object.");
    }
    id<MTLArgumentEncoder> argument_encoder =
        [Obj<id<MTLFunction>>(function) newArgumentEncoderWithBufferIndex:0];
    if (argument_encoder == nil) {
      return absl::InternalError("Failed to create Metal argument encoder.");
    }
    id<MTLBuffer> argument_buffer = [[Obj<id<MTLCommandQueue>>(command_queue)
        device] newBufferWithLength:[argument_encoder encodedLength]
                            options:MTLResourceStorageModeShared];
    if (argument_buffer == nil) {
      return absl::ResourceExhaustedError(
          "Failed to allocate Metal argument buffer.");
    }
    [argument_encoder setArgumentBuffer:argument_buffer offset:0];
    for (NSUInteger i = 0; i < arguments.size(); ++i) {
      const MetalKernelArgument& arg = arguments[i];
      if (arg.buffer != nullptr) {
        [argument_encoder setBuffer:Obj<id<MTLBuffer>>(arg.buffer)
                              offset:arg.offset
                             atIndex:i];
        [encoder useResource:Obj<id<MTLBuffer>>(arg.buffer)
                       usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      } else {
        void* dst = [argument_encoder constantDataAtIndex:i];
        if (dst == nullptr) {
          return absl::InternalError(
              absl::StrCat("Metal argument buffer has no constant slot ", i,
                           "."));
        }
        std::memcpy(dst, arg.bytes, arg.bytes_size);
      }
    }
    [encoder setBuffer:argument_buffer offset:0 atIndex:0];
  } else {
    for (NSUInteger i = 0; i < arguments.size(); ++i) {
      const MetalKernelArgument& arg = arguments[i];
      if (arg.buffer != nullptr) {
        [encoder setBuffer:Obj<id<MTLBuffer>>(arg.buffer)
                    offset:arg.offset
                   atIndex:i];
      } else {
        [encoder setBytes:arg.bytes length:arg.bytes_size atIndex:i];
      }
    }
  }

  MTLSize threads_per_threadgroup =
      MTLSizeMake(thread_dims.x, thread_dims.y, thread_dims.z);
  MTLSize threadgroups =
      MTLSizeMake(block_dims.x, block_dims.y, block_dims.z);
  [encoder dispatchThreadgroups:threadgroups
          threadsPerThreadgroup:threads_per_threadgroup];
  [encoder endEncoding];
  [command_buffer commit];
  return RetainObj(command_buffer);
}

absl::Status WaitUntilCompleted(void* command_buffer) {
  if (command_buffer == nullptr) return absl::OkStatus();
  id<MTLCommandBuffer> buffer = Obj<id<MTLCommandBuffer>>(command_buffer);
  [buffer waitUntilCompleted];
  if ([buffer status] == MTLCommandBufferStatusError) {
    return absl::InternalError(
        absl::StrCat("Metal command buffer failed: ",
                     ErrorMessage([buffer error])));
  }
  return absl::OkStatus();
}

absl::Status SynchronizeCommandQueue(void* command_queue) {
  id<MTLCommandBuffer> command_buffer =
      [Obj<id<MTLCommandQueue>>(command_queue) commandBuffer];
  if (command_buffer == nil) {
    return absl::InternalError(
        "Failed to create Metal synchronization command buffer.");
  }
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if ([command_buffer status] == MTLCommandBufferStatusError) {
    return absl::InternalError(
        absl::StrCat("Metal command queue synchronization failed: ",
                     ErrorMessage([command_buffer error])));
  }
  return absl::OkStatus();
}

Event::Status PollCommandBufferStatus(void* command_buffer) {
  if (command_buffer == nullptr) return Event::Status::kComplete;
  switch ([Obj<id<MTLCommandBuffer>>(command_buffer) status]) {
    case MTLCommandBufferStatusNotEnqueued:
    case MTLCommandBufferStatusEnqueued:
    case MTLCommandBufferStatusCommitted:
    case MTLCommandBufferStatusScheduled:
      return Event::Status::kPending;
    case MTLCommandBufferStatusCompleted:
      return Event::Status::kComplete;
    case MTLCommandBufferStatusError:
      return Event::Status::kError;
  }
}

}  // namespace stream_executor::metal
