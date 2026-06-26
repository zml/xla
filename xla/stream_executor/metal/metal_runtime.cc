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

#import <dispatch/dispatch.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
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

// === Metal GPU profiling state (xprof device tracer) =======================
// Per-op GPU timing via an MTLCounterSampleBuffer sampled at compute-encoder
// boundaries. All mutable state is guarded by g_prof_mu except g_prof_enabled
// (an atomic fast-path flag read once per dispatch when profiling is off). The
// sample buffer is reused each step: decode is sequential (encode -> commit ->
// wait -> resolve -> reset), so no two steps' samples are ever live at once.
//
// LOAD-BEARING ASSUMPTIONS (true for the ZML `--profile` flow; this state is a
// single global, not per-stream/per-cb, so it is only correct under them):
//   1. One profiled stream at a time. The reserved sample-index space and the
//      "resolve the whole [0,used) range at commit" logic are shared globally;
//      two streams encoding+committing concurrently would interleave indices
//      and resolve each other's pending samples. Multi-stream profiling would
//      need per-command-buffer index ranges + per-cb pending lists.
//   2. The executor is quiescent when MetalProfilingStop()/Drain() run (ZML
//      stops the session after generation finishes), so no commit races Stop.
//   3. A single GPU: g_prof_buf is created once against the first device seen
//      and reused (intentional process-lifetime singleton).
// Device max counter-sample-buffer length is 32768 B (== 4096 timestamps) on
// Apple Silicon. 2 samples/op => up to 2048 ops/step; far above one execution's
// ~tens of dispatches (used resets after each step's commit+resolve).
constexpr uint32_t kProfMaxSamples = 4096;

std::atomic<bool> g_prof_enabled{false};
std::mutex g_prof_mu;
id<MTLCounterSampleBuffer> g_prof_buf = nil;
bool g_prof_unsupported = false;
uint32_t g_prof_cap = 0;
uint32_t g_prof_used = 0;
bool g_prof_have_anchor = false;
uint64_t g_prof_gpu0 = 0;   // GPU timestamp (== mach ns) at the anchor instant
uint64_t g_prof_wall0 = 0;  // host wall-clock ns (UNIX epoch) at the same instant

struct ProfPending {
  std::string name;
  std::string details;
  uint64_t bytes;
  uint32_t start_idx;  // start sample; the end sample is start_idx + 1
};
std::vector<ProfPending> g_prof_pending;
std::vector<MetalProfileEvent> g_prof_events;
uint64_t g_prof_dropped = 0;

// Lazily creates the timestamp counter sample buffer for `device`. Returns false
// (and latches g_prof_unsupported) if this GPU can't sample at encoder
// boundaries. Caller holds g_prof_mu.
bool ProfEnsureBuffer(id<MTLDevice> device) {
  if (g_prof_buf != nil) return true;
  if (g_prof_unsupported) return false;
  if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
    g_prof_unsupported = true;
    LOG(WARNING) << "Metal profiling: GPU lacks stage-boundary counter "
                    "sampling; no per-op GPU timing.";
    return false;
  }
  id<MTLCounterSet> timestamp_set = nil;
  for (id<MTLCounterSet> cs in device.counterSets) {
    if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamp_set = cs;
  }
  if (timestamp_set == nil) {
    g_prof_unsupported = true;
    LOG(WARNING) << "Metal profiling: no 'timestamp' counter set on device.";
    return false;
  }
  MTLCounterSampleBufferDescriptor* desc =
      [[MTLCounterSampleBufferDescriptor alloc] init];
  desc.counterSet = timestamp_set;
  desc.storageMode = MTLStorageModeShared;
  desc.sampleCount = kProfMaxSamples;
  NSError* error = nil;
  g_prof_buf = [device newCounterSampleBufferWithDescriptor:desc error:&error];
  if (g_prof_buf == nil) {
    g_prof_unsupported = true;
    LOG(WARNING) << "Metal profiling: newCounterSampleBuffer failed: "
                 << (error ? error.localizedDescription.UTF8String : "unknown");
    return false;
  }
  g_prof_cap = kProfMaxSamples;
  return true;
}

// Result of reserving a timestamp sample pair for one compute encoder.
struct ProfBegin {
  id<MTLCounterSampleBuffer> buffer;
  uint32_t start;
  uint32_t end;
};

// If profiling is on, reserves a (start,end) timestamp sample pair for the next
// encoder and records its label/bytes. Returns false when off / buffer full /
// device unsupported (the caller then encodes without sampling). Captures the
// GPU<->wall clock anchor on the first sampled encoder of a session.
bool ProfBeginEncoder(id<MTLCommandBuffer> command_buffer, absl::string_view name,
                      absl::string_view details, uint64_t bytes,
                      ProfBegin* out) {
  if (!g_prof_enabled.load(std::memory_order_relaxed)) return false;
  std::lock_guard<std::mutex> lock(g_prof_mu);
  if (!g_prof_enabled.load(std::memory_order_relaxed)) return false;
  id<MTLDevice> device = [command_buffer device];
  if (!ProfEnsureBuffer(device)) return false;
  if (!g_prof_have_anchor) {
    MTLTimestamp cpu_ts = 0, gpu_ts = 0;
    [device sampleTimestamps:&cpu_ts gpuTimestamp:&gpu_ts];
    g_prof_gpu0 = gpu_ts;
    g_prof_wall0 = static_cast<uint64_t>(absl::GetCurrentTimeNanos());
    g_prof_have_anchor = true;
    LOG(INFO) << "Metal GPU profiling: recording (clock anchor captured).";
  }
  if (g_prof_used + 2 > g_prof_cap) {
    ++g_prof_dropped;
    return false;
  }
  uint32_t s = g_prof_used;
  g_prof_used += 2;
  g_prof_pending.push_back(
      {std::string(name), std::string(details), bytes, s});
  out->buffer = g_prof_buf;
  out->start = s;
  out->end = s + 1;
  return true;
}

// Releases the most recent ProfBeginEncoder reservation. Called when the encoder
// fails to create after we reserved its sample slots, so the (never-written)
// slots don't accumulate toward the cap. Pops the last pending entry — correct
// under the single-stream assumption (the last reservation is ours).
void ProfRollback() {
  std::lock_guard<std::mutex> lock(g_prof_mu);
  if (!g_prof_pending.empty() && g_prof_used >= 2) {
    g_prof_pending.pop_back();
    g_prof_used -= 2;
  }
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
  info.max_threadgroup_memory_length = [device maxThreadgroupMemoryLength];
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

absl::StatusOr<void*> LoadLibraryFromData(void* device,
                                          absl::Span<const uint8_t> data) {
  if (data.empty()) {
    return absl::InvalidArgumentError("Metal library data is empty.");
  }
  void* bytes = std::malloc(data.size());
  if (bytes == nullptr) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Failed to allocate ", data.size(),
                     " bytes for Metal library data."));
  }
  std::memcpy(bytes, data.data(), data.size());

  dispatch_data_t library_data =
      dispatch_data_create(bytes, data.size(),
                           dispatch_get_global_queue(
                               QOS_CLASS_USER_INITIATED, /*flags=*/0),
                           DISPATCH_DATA_DESTRUCTOR_FREE);
  if (library_data == nil) {
    std::free(bytes);
    return absl::InternalError("Failed to create Metal library data object.");
  }

  NSError* error = nil;
  id<MTLLibrary> library =
      [Obj<id<MTLDevice>>(device) newLibraryWithData:library_data
                                               error:&error];
  if (library == nil) {
    return absl::InternalError(
        absl::StrCat("Metal failed to load library data: ",
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

absl::StatusOr<void*> NewFunctionWithConstants(
    void* library, absl::string_view name,
    absl::Span<const MetalFunctionConstant> constants) {
  @autoreleasepool {
    NSString* function_name =
        [[NSString alloc] initWithBytes:name.data()
                                 length:name.size()
                               encoding:NSUTF8StringEncoding];
    if (function_name == nil) {
      return absl::InvalidArgumentError(
          "Metal function name is not valid UTF-8.");
    }
    MTLFunctionConstantValues* values =
        [[MTLFunctionConstantValues alloc] init];
    for (const MetalFunctionConstant& c : constants) {
      if (c.kind == MetalFunctionConstant::Kind::kBool) {
        bool b = c.value != 0;
        [values setConstantValue:&b
                            type:MTLDataTypeBool
                         atIndex:c.index];
      } else {
        int v = static_cast<int>(c.value);
        [values setConstantValue:&v
                            type:MTLDataTypeInt
                         atIndex:c.index];
      }
    }
    NSError* error = nil;
    id<MTLFunction> function =
        [Obj<id<MTLLibrary>>(library) newFunctionWithName:function_name
                                          constantValues:values
                                                   error:&error];
    if (function == nil) {
      return absl::NotFoundError(absl::StrCat(
          "Metal newFunctionWithName:constantValues: failed for '", name,
          "': ", ErrorMessage(error)));
    }
    return RetainObj(function);
  }
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

// One-encoder batching (the ggml-metal model). Instead of one compute encoder
// PER dispatch, keep ONE serial-dispatch encoder open per command buffer and let
// Metal's AUTOMATIC IN-ENCODER hazard tracking order dependent dispatches. The
// auto hazard that has a SIZE-DEPENDENT cost (a sharp step once a buffer passes
// ~256 MB) is the CROSS-ENCODER one — the per-layer KV-cache write(DUS)->read
// (attention) dependency, an encoder boundary every layer, made decode lose ~9%
// once the KV grew past ~350 MB (the large-seqlen "cliff"). A single open encoder
// pulls that hazard INSIDE the encoder, where the auto barrier is size-independent
// AND precise: independent dispatches (Q/K/V, gate/up) are not ordered against
// each other, so they overlap — faster than an explicit per-dispatch
// memoryBarrierWithScope:Buffers (which is a full drain). Verified byte-identical
// + flat ~68.7 tok/s to seqlen 16384. Remaining cross-encoder hazards (a blit, an
// execute boundary) still rely on Metal auto-tracking, but they are on the small
// per-step activations, not the big KV.
static std::mutex g_enc_mu;
static std::unordered_map<void*, id<MTLComputeCommandEncoder>> g_open_enc;

// Returns the command buffer's open serial-dispatch encoder, creating it on first
// use. No explicit barrier: a serial-dispatch encoder auto-hazard-tracks its
// dispatches (orders only true dependencies; independent ones overlap).
static id<MTLComputeCommandEncoder> GetConcurrentEncoder(
    id<MTLCommandBuffer> command_buffer) {
  std::lock_guard<std::mutex> lk(g_enc_mu);
  void* key = (__bridge void*)command_buffer;
  auto it = g_open_enc.find(key);
  if (it != g_open_enc.end() && it->second != nil) return it->second;
  id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
  if (enc != nil) {
    g_open_enc[key] = enc;
  }
  return enc;
}

// Ends and forgets the open concurrent encoder for `command_buffer`, if any.
// MUST be called before committing the buffer or encoding into it by any other
// means (blit, a profiled per-dispatch encoder) — Metal forbids two open
// encoders on one command buffer.
static void FlushConcurrentEncoder(id<MTLCommandBuffer> command_buffer) {
  if (command_buffer == nil) return;
  std::lock_guard<std::mutex> lk(g_enc_mu);
  void* key = (__bridge void*)command_buffer;
  auto it = g_open_enc.find(key);
  if (it != g_open_enc.end()) {
    if (it->second != nil) [it->second endEncoding];
    g_open_enc.erase(it);
  }
}

// Encodes one compute dispatch into `command_buffer`. By default it appends to
// the buffer's shared concurrent encoder (above); under profiling it uses a
// dedicated per-dispatch encoder for accurate GPU timestamps. Does NOT commit —
// the caller decides when (and FlushConcurrentEncoder runs before commit).
static absl::Status EncodeComputeInto(
    id<MTLCommandBuffer> command_buffer, void* pipeline, void* function,
    bool use_argument_buffer, absl::Span<const MetalKernelArgument> arguments,
    absl::string_view name, const ThreadDim& thread_dims,
    const BlockDim& block_dims, int64_t shmem_bytes = 0) {
  // Defensive launch guard. maxTotalThreadsPerThreadgroup is PER-PIPELINE (a
  // register-heavy kernel can report < the device's nominal 1024) and
  // maxThreadgroupMemoryLength is the device's threadgroup-memory budget;
  // exceeding either is a hard dispatch failure / undefined behavior on Metal.
  // Both limits hold on every shipping Apple GPU (M1..M4), so this never fires
  // today — it converts a mis-sized launch (bad tile math, or a future
  // smaller-smem / lower-occupancy part) into a clear error HERE instead of an
  // opaque GPU fault or uninitialized threadgroup memory. This is the one funnel
  // every AIR + custom-kernel dispatch flows through, so the check is universal.
  {
    id<MTLComputePipelineState> pso = Obj<id<MTLComputePipelineState>>(pipeline);
    const uint64_t threads_per_tg = static_cast<uint64_t>(thread_dims.x) *
                                    thread_dims.y * thread_dims.z;
    const uint64_t pso_max = [pso maxTotalThreadsPerThreadgroup];
    if (pso_max != 0 && threads_per_tg > pso_max) {
      return absl::InternalError(absl::StrCat(
          "Metal launch '", name, "': threadgroup size ", threads_per_tg,
          " (", thread_dims.x, "x", thread_dims.y, "x", thread_dims.z,
          ") exceeds this pipeline's maxTotalThreadsPerThreadgroup ", pso_max,
          "."));
    }
    if (shmem_bytes > 0) {
      const uint64_t tg_mem_limit =
          [[command_buffer device] maxThreadgroupMemoryLength];
      if (tg_mem_limit != 0 &&
          static_cast<uint64_t>(shmem_bytes) > tg_mem_limit) {
        return absl::InternalError(absl::StrCat(
            "Metal launch '", name, "': threadgroup memory ", shmem_bytes,
            " bytes exceeds device maxThreadgroupMemoryLength ", tg_mem_limit,
            " bytes."));
      }
    }
  }
  // Profiling (off unless a profile session is active): reserve GPU-timestamp
  // samples at this encoder's boundaries so the xprof Metal tracer can time the
  // op on the GPU clock. See MetalProfiling* in metal_runtime.h.
  id<MTLComputeCommandEncoder> encoder = nil;
  ProfBegin prof_begin;
  bool prof_on = false;
  if (MetalProfilingEnabled()) {
    uint64_t bytes = 0;
    for (const auto& arg : arguments) {
      if (arg.buffer != nullptr) bytes += [Obj<id<MTLBuffer>>(arg.buffer) length];
    }
    std::string label = absl::StrCat(name, " grid=", block_dims.x, "x",
                                     block_dims.y, "x", block_dims.z);
    std::string details = absl::StrCat(
        "regs:0 static_shared:0 dynamic_shared:0 grid:", block_dims.x, ",",
        block_dims.y, ",", block_dims.z, " block:", thread_dims.x, ",",
        thread_dims.y, ",", thread_dims.z, " occ_pct:0");
    prof_on =
        ProfBeginEncoder(command_buffer, label, details, bytes, &prof_begin);
  }
  bool shared_encoder = false;
  if (prof_on) {
    // Per-op GPU timing needs one encoder per dispatch; end any open batch
    // encoder so this profiled encoder is the buffer's only live one.
    FlushConcurrentEncoder(command_buffer);
    MTLComputePassDescriptor* pass =
        [MTLComputePassDescriptor computePassDescriptor];
    MTLComputePassSampleBufferAttachmentDescriptor* sample =
        pass.sampleBufferAttachments[0];
    sample.sampleBuffer = prof_begin.buffer;
    sample.startOfEncoderSampleIndex = prof_begin.start;
    sample.endOfEncoderSampleIndex = prof_begin.end;
    encoder = [command_buffer computeCommandEncoderWithDescriptor:pass];
  } else {
    encoder = GetConcurrentEncoder(command_buffer);
    shared_encoder = true;  // owned by the command buffer; do not endEncoding here
  }
  if (encoder == nil) {
    if (prof_on) ProfRollback();  // release the reserved-but-unused samples
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
    id<MTLBuffer> argument_buffer = [[command_buffer device]
        newBufferWithLength:[argument_encoder encodedLength]
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

  // Dynamic threadgroup memory (one slab at threadgroup index 0). Kernels with
  // STATIC threadgroup arrays (the emitter's reductions) pass shmem_bytes=0 and
  // are unaffected; flash-attention / fused kernels that declare a sized
  // `threadgroup T* x [[threadgroup(0)]]` need the host to set its length here.
  if (shmem_bytes > 0) {
    [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(shmem_bytes)
                               atIndex:0];
  }
  MTLSize threads_per_threadgroup =
      MTLSizeMake(thread_dims.x, thread_dims.y, thread_dims.z);
  MTLSize threadgroups =
      MTLSizeMake(block_dims.x, block_dims.y, block_dims.z);
  [encoder dispatchThreadgroups:threadgroups
          threadsPerThreadgroup:threads_per_threadgroup];
  if (!shared_encoder) [encoder endEncoding];
  return absl::OkStatus();
}

void* NewBatchCommandBuffer(void* command_queue) {
  @autoreleasepool {
    MPSCommandBuffer* cb = [MPSCommandBuffer
        commandBufferFromCommandQueue:Obj<id<MTLCommandQueue>>(command_queue)];
    return RetainObj(cb);  // +1; released when the stream finishes the batch
  }
}

absl::Status EncodeKernel(void* batch_command_buffer, void* pipeline,
                          void* function, bool use_argument_buffer,
                          absl::Span<const MetalKernelArgument> arguments,
                          absl::string_view name, const ThreadDim& thread_dims,
                          const BlockDim& block_dims, int64_t shmem_bytes) {
  if (batch_command_buffer == nullptr) {
    return absl::InternalError("Metal EncodeKernel: null batch command buffer.");
  }
  // Encode into the MPSCommandBuffer's current underlying MTLCommandBuffer,
  // alongside the step's other compute encoders in the same buffer. No commit.
  id<MTLCommandBuffer> cb =
      Obj<MPSCommandBuffer*>(batch_command_buffer).commandBuffer;
  return EncodeComputeInto(cb, pipeline, function, use_argument_buffer,
                           arguments, name, thread_dims, block_dims,
                           shmem_bytes);
}

absl::Status EncodeBlitCopy(void* batch_command_buffer, void* dst_buffer,
                            uint64_t dst_offset, void* src_buffer,
                            uint64_t src_offset, uint64_t size) {
  if (batch_command_buffer == nullptr || dst_buffer == nullptr ||
      src_buffer == nullptr) {
    return absl::InternalError("Metal EncodeBlitCopy: null buffer.");
  }
  if (size == 0) return absl::OkStatus();
  @autoreleasepool {
    id<MTLCommandBuffer> cb =
        Obj<MPSCommandBuffer*>(batch_command_buffer).commandBuffer;
    FlushConcurrentEncoder(cb);  // end open compute encoder pre-blit
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:Obj<id<MTLBuffer>>(src_buffer)
            sourceOffset:src_offset
                toBuffer:Obj<id<MTLBuffer>>(dst_buffer)
       destinationOffset:dst_offset
                    size:size];
    [blit endEncoding];
  }
  return absl::OkStatus();
}

// Env-gated standalone KPROF reporter (defined below; forward-declared so the
// commit funnels can drive it without the xprof tracer being attached).
void MetalKprofMaybeStart();
void MetalKprofReport();

void* CommitBatchCommandBuffer(void* batch_command_buffer) {
  if (batch_command_buffer == nullptr) return nullptr;
  MetalKprofMaybeStart();  // env-gated: start a profiling session on first use
  @autoreleasepool {
    MPSCommandBuffer* mpscb = Obj<MPSCommandBuffer*>(batch_command_buffer);
    FlushConcurrentEncoder(mpscb.commandBuffer);
    [mpscb commit];
    // The final committed underlying cb; waiting it drains all prior segments
    // (an MPSCommandBuffer may split via commitAndContinue) by queue order.
    id<MTLCommandBuffer> committed = mpscb.commandBuffer;
    // Profiling: this is the one funnel every step's commit flows through
    // (BlockHostUntilDone AND RecordEvent). GPU timestamp samples are only
    // readable once the buffer completes, so when profiling is on we wait here
    // and resolve — serializing the step (acceptable: profiling is opt-in, and
    // per-op GPU durations stay accurate). No wait on the normal path.
    if (MetalProfilingEnabled()) {
      [committed waitUntilCompleted];
      MetalProfilingResolveStep(nullptr);
      MetalKprofReport();  // env-gated: aggregate + periodic histogram
    }
    return RetainObj(committed);
  }
}

void CommitBatchCommandBufferWithCompletion(
    void* batch_command_buffer, absl::AnyInvocable<void() &&> on_complete) {
  if (batch_command_buffer == nullptr) {
    // No command buffer was opened, so there is no GPU work to wait on: run the
    // callback right away.
    std::move(on_complete)();
    return;
  }
  MetalKprofMaybeStart();  // env-gated: start a profiling session on first use
  @autoreleasepool {
    MPSCommandBuffer* mpscb = Obj<MPSCommandBuffer*>(batch_command_buffer);
    FlushConcurrentEncoder(mpscb.commandBuffer);
    // Heap-own the move-only callback so the ObjC completion block (which Metal
    // copies onto an internal queue) can invoke it exactly once and free it.
    auto* cb = new absl::AnyInvocable<void() &&>(std::move(on_complete));
    // addCompletedHandler MUST be registered before commit; the handler fires on
    // a Metal-internal thread once the GPU finishes this command buffer.
    [mpscb.commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
      std::move (*cb)();
      delete cb;
    }];
    [mpscb commit];
    // Profiling resolve funnel: this completion variant is the commit path the
    // BlockHostUntilDone batch drain uses (metal_stream.cc), and with adaptive_k
    // batching it is the DOMINANT prefill commit — so resolve here too, else the
    // GPU-timestamp samples attached to this buffer's encoders are never read
    // back. Opt-in only (MetalProfilingEnabled() == false on the normal fast
    // path), so the serializing wait never touches production runs. The
    // completion handler still fires for the event/host-callback bookkeeping.
    if (MetalProfilingEnabled()) {
      [mpscb.commandBuffer waitUntilCompleted];
      MetalProfilingResolveStep(nullptr);
      MetalKprofReport();  // env-gated: aggregate + periodic histogram
    }
  }
}

void* NewSharedEvent(void* device) {
  @autoreleasepool {
    id<MTLSharedEvent> ev = [Obj<id<MTLDevice>>(device) newSharedEvent];
    return RetainObj(ev);
  }
}

void EncodeSignalSharedEvent(void* batch_command_buffer, void* shared_event,
                             uint64_t value) {
  if (batch_command_buffer == nullptr || shared_event == nullptr) return;
  @autoreleasepool {
    id<MTLCommandBuffer> cb =
        Obj<MPSCommandBuffer*>(batch_command_buffer).commandBuffer;
    FlushConcurrentEncoder(cb);  // cb-level op: no open encoder
    [cb encodeSignalEvent:Obj<id<MTLSharedEvent>>(shared_event) value:value];
  }
}

void EncodeWaitForSharedEvent(void* batch_command_buffer, void* shared_event,
                              uint64_t value) {
  if (batch_command_buffer == nullptr || shared_event == nullptr) return;
  @autoreleasepool {
    id<MTLCommandBuffer> cb =
        Obj<MPSCommandBuffer*>(batch_command_buffer).commandBuffer;
    FlushConcurrentEncoder(cb);  // cb-level op: no open encoder
    [cb encodeWaitForEvent:Obj<id<MTLSharedEvent>>(shared_event) value:value];
  }
}

uint64_t SharedEventSignaledValue(void* shared_event) {
  if (shared_event == nullptr) return 0;
  @autoreleasepool {
    return Obj<id<MTLSharedEvent>>(shared_event).signaledValue;
  }
}

absl::Status WaitUntilSignaledValue(void* shared_event, uint64_t value) {
  if (shared_event == nullptr || value == 0) return absl::OkStatus();
  @autoreleasepool {
    id<MTLSharedEvent> ev = Obj<id<MTLSharedEvent>>(shared_event);
    // Fast path: already signaled.
    if (ev.signaledValue >= value) return absl::OkStatus();
    // Block until the GPU signals >= value. UINT64_MAX ms is effectively
    // "wait forever", matching the old [commandBuffer waitUntilCompleted]; the
    // signaling command buffer is committed before any consumer waits, so this
    // resolves promptly and never deadlocks on the happy path.
    bool ok = [ev waitUntilSignaledValue:value timeoutMS:UINT64_MAX];
    if (!ok) {
      return absl::InternalError(
          absl::StrCat("Metal shared-event wait timed out at value ", value));
    }
    return absl::OkStatus();
  }
}

void* NewSharedEventListener() {
  @autoreleasepool {
    dispatch_queue_t queue = dispatch_queue_create(
        "xla.metal.shared_event_listener", DISPATCH_QUEUE_SERIAL);
    MTLSharedEventListener* listener =
        [[MTLSharedEventListener alloc] initWithDispatchQueue:queue];
    return RetainObj(listener);  // +1; released via ReleaseObject in the dtor.
  }
}

void NotifySharedEventListener(void* listener, void* shared_event,
                               uint64_t value,
                               absl::AnyInvocable<void() &&> callback) {
  if (listener == nullptr || shared_event == nullptr) return;
  // Heap-own the move-only callback so the ObjC block (copied onto the
  // listener's queue) invokes it exactly once and frees it — same pattern as
  // CommitBatchCommandBufferWithCompletion.
  auto* held = new absl::AnyInvocable<void() &&>(std::move(callback));
  @autoreleasepool {
    [Obj<id<MTLSharedEvent>>(shared_event)
        notifyListener:Obj<MTLSharedEventListener*>(listener)
               atValue:value
                 block:^(id<MTLSharedEvent> _Nonnull, uint64_t) {
                   std::move (*held)();
                   delete held;
                 }];
  }
}

// === Metal GPU profiling — public API (see metal_runtime.h) ================

void MetalProfilingStart() {
  std::lock_guard<std::mutex> lock(g_prof_mu);
  g_prof_used = 0;
  g_prof_pending.clear();
  g_prof_events.clear();
  g_prof_dropped = 0;
  g_prof_have_anchor = false;
  g_prof_enabled.store(true, std::memory_order_relaxed);
  LOG(INFO) << "Metal GPU profiling: session started.";
}

void MetalProfilingStop() {
  g_prof_enabled.store(false, std::memory_order_relaxed);
}

bool MetalProfilingEnabled() {
  return g_prof_enabled.load(std::memory_order_relaxed);
}

void MetalProfilingResolveStep(void* /*device*/) {
  std::lock_guard<std::mutex> lock(g_prof_mu);
  if (g_prof_used == 0 || g_prof_buf == nil) return;
  // The step's command buffer has completed, so its samples are written. GPU
  // timestamps are nanoseconds in the mach domain; shift them onto the host
  // wall clock (UNIX epoch) via the anchor captured at the first sampled
  // encoder. (Verified: GPU ticks == cpu mach ns, so the map is a pure offset.)
  NSData* data = [g_prof_buf resolveCounterRange:NSMakeRange(0, g_prof_used)];
  const int64_t offset =
      static_cast<int64_t>(g_prof_wall0) - static_cast<int64_t>(g_prof_gpu0);
  if (data != nil &&
      data.length >= g_prof_used * sizeof(MTLCounterResultTimestamp)) {
    const auto* r = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
    for (const auto& p : g_prof_pending) {
      uint64_t st = r[p.start_idx].timestamp;
      uint64_t en = r[p.start_idx + 1].timestamp;
      if (st == MTLCounterErrorValue || en == MTLCounterErrorValue || en <= st) {
        ++g_prof_dropped;
        continue;
      }
      MetalProfileEvent ev;
      ev.name = p.name;
      ev.details = p.details;
      ev.bytes = p.bytes;
      ev.start_ns = static_cast<uint64_t>(static_cast<int64_t>(st) + offset);
      ev.end_ns = static_cast<uint64_t>(static_cast<int64_t>(en) + offset);
      g_prof_events.push_back(std::move(ev));
    }
  } else {
    g_prof_dropped += g_prof_pending.size();
  }
  g_prof_pending.clear();
  g_prof_used = 0;
}

std::vector<MetalProfileEvent> MetalProfilingDrain() {
  std::lock_guard<std::mutex> lock(g_prof_mu);
  std::vector<MetalProfileEvent> out = std::move(g_prof_events);
  g_prof_events.clear();
  return out;
}

uint64_t MetalProfilingDroppedCount() {
  std::lock_guard<std::mutex> lock(g_prof_mu);
  return g_prof_dropped;
}

// === Env-gated per-kernel GPU-time reporter (METAL_KPROF=1) =================
// Lightweight standalone decode profiler (re-adds the removed KPROF tool): with
// METAL_KPROF set, lazily starts a profiling session and prints, every
// METAL_KPROF_EVERY commits (default 400 ~= 100 decode tokens), a histogram of
// GPU time aggregated by kernel name. Profiling forces a per-dispatch encoder
// plus a per-commit waitUntilCompleted, so it SERIALIZES and slows the run --
// this is a measurement mode, never the production fast path.
static bool MetalKprofWanted() {
  static const bool on = [] {
    const char* e = std::getenv("METAL_KPROF");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

void MetalKprofMaybeStart() {
  if (!MetalKprofWanted()) return;
  static std::once_flag once;
  std::call_once(once, [] { MetalProfilingStart(); });
}

void MetalKprofReport() {
  if (!MetalKprofWanted()) return;
  static const int report_every = [] {
    const char* e = std::getenv("METAL_KPROF_EVERY");
    int v = e ? std::atoi(e) : 400;
    return v > 0 ? v : 400;
  }();
  struct Agg {
    uint64_t ns = 0;
    uint64_t count = 0;
  };
  static std::mutex mu;
  static std::unordered_map<std::string, Agg> agg;
  static int commits = 0;
  static uint64_t total_ns = 0;

  std::vector<MetalProfileEvent> evs = MetalProfilingDrain();
  std::lock_guard<std::mutex> lock(mu);
  for (const auto& ev : evs) {
    std::string key = ev.name;
    auto pos = key.find(" grid=");
    if (pos != std::string::npos) key.resize(pos);
    Agg& a = agg[key];
    const uint64_t d = ev.end_ns - ev.start_ns;
    a.ns += d;
    a.count += 1;
    total_ns += d;
  }
  if (++commits < report_every) return;
  std::vector<std::pair<std::string, Agg>> rows(agg.begin(), agg.end());
  std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
  LOG(INFO) << "=== METAL_KPROF: GPU time by kernel over " << commits
            << " commits (total GPU " << (total_ns / 1.0e6) << " ms) ===";
  for (const auto& [name, a] : rows) {
    LOG(INFO) << "  " << (a.ns / 1.0e6) << " ms  "
              << (total_ns ? 100.0 * a.ns / total_ns : 0.0) << "%  n=" << a.count
              << "  " << name;
  }
  agg.clear();
  commits = 0;
  total_ns = 0;
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
