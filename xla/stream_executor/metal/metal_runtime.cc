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
#import <IOKit/IOKitLib.h>
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
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"

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

constexpr uint32_t kProfMaxSamples = 4096;

static bool MetalBlitStatsEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("METAL_BLIT_STATS");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

struct MetalBlitStats {
  std::atomic<uint64_t> copies{0}, copy_bytes{0}, fills{0}, fill_bytes{0};
  std::atomic<uint64_t> last_report_ns{0};
};
static MetalBlitStats g_blit_stats;

static void MetalBlitStatsRecord(bool is_fill, uint64_t size) {
  if (!MetalBlitStatsEnabled()) return;
  if (is_fill) {
    g_blit_stats.fills.fetch_add(1, std::memory_order_relaxed);
    g_blit_stats.fill_bytes.fetch_add(size, std::memory_order_relaxed);
  } else {
    g_blit_stats.copies.fetch_add(1, std::memory_order_relaxed);
    g_blit_stats.copy_bytes.fetch_add(size, std::memory_order_relaxed);
  }
  const uint64_t now = absl::GetCurrentTimeNanos();
  uint64_t last = g_blit_stats.last_report_ns.load(std::memory_order_relaxed);
  if (now - last < 1000000000ull) return;
  if (!g_blit_stats.last_report_ns.compare_exchange_strong(last, now)) return;
  fprintf(stderr,
          "[metal-blit] copies=%llu (%.2f GB)  fills=%llu (%.2f GB)\n",
          (unsigned long long)g_blit_stats.copies.load(),
          g_blit_stats.copy_bytes.load() / 1e9,
          (unsigned long long)g_blit_stats.fills.load(),
          g_blit_stats.fill_bytes.load() / 1e9);
}

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

struct ProfBegin {
  id<MTLCounterSampleBuffer> buffer;
  uint32_t start;
  uint32_t end;
};

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

static uint64_t QueryGpuCoreCount() {
  io_iterator_t iter = IO_OBJECT_NULL;
  if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                   IOServiceMatching("AGXAccelerator"),
                                   &iter) != KERN_SUCCESS) {
    return 0;
  }
  uint64_t cores = 0;
  io_object_t service = IO_OBJECT_NULL;
  while (cores == 0 && (service = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
    CFTypeRef prop = IORegistryEntrySearchCFProperty(
        service, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault,
        kIORegistryIterateRecursively | kIORegistryIterateParents);
    if (prop != nullptr) {
      if (CFGetTypeID(prop) == CFNumberGetTypeID()) {
        int value = 0;
        if (CFNumberGetValue(static_cast<CFNumberRef>(prop), kCFNumberIntType,
                             &value) &&
            value > 0) {
          cores = static_cast<uint64_t>(value);
        }
      }
      CFRelease(prop);
    }
    IOObjectRelease(service);
  }
  IOObjectRelease(iter);
  return cores;
}

absl::StatusOr<MetalDeviceInfo> GetDeviceInfo(int ordinal) {
  TF_ASSIGN_OR_RETURN(id<MTLDevice> device, DeviceAtOrdinal(ordinal));
  MetalDeviceInfo info;
  info.name = NSStringToString([device name]);
  if (@available(macOS 14.0, *)) {
    info.architecture = NSStringToString([[device architecture] name]);
  }
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
  info.gpu_core_count = QueryGpuCoreCount();
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

absl::StatusOr<void*> NewResidencySet(void* device) {
  id<MTLDevice> dev = Obj<id<MTLDevice>>(device);
  if (![dev supportsFamily:MTLGPUFamilyMetal3]) {
    return nullptr;
  }
  if (@available(macOS 15.0, *)) {
    MTLResidencySetDescriptor* desc = [[MTLResidencySetDescriptor alloc] init];
    desc.label = @"XLA device allocations";
    desc.initialCapacity = 256;
    NSError* error = nil;
    id<MTLResidencySet> set = [dev newResidencySetWithDescriptor:desc
                                                           error:&error];
    if (set == nil) {
      LOG(WARNING) << "Metal: newResidencySetWithDescriptor failed ("
                   << ErrorMessage(error)
                   << "); weight buffers may be compressed by the OS, which "
                      "shows up as a slow first inference.";
      return nullptr;
    }
    return RetainObj(set);
  }
  return nullptr;
}

void ResidencySetAddAllocation(void* residency_set, void* buffer) {
  if (residency_set == nullptr || buffer == nullptr) return;
  if (@available(macOS 15.0, *)) {
    [Obj<id<MTLResidencySet>>(residency_set)
        addAllocation:Obj<id<MTLBuffer>>(buffer)];
  }
}

void ResidencySetRemoveAllocation(void* residency_set, void* buffer) {
  if (residency_set == nullptr || buffer == nullptr) return;
  if (@available(macOS 15.0, *)) {
    [Obj<id<MTLResidencySet>>(residency_set)
        removeAllocation:Obj<id<MTLBuffer>>(buffer)];
  }
}

void ResidencySetCommit(void* residency_set) {
  if (residency_set == nullptr) return;
  if (@available(macOS 15.0, *)) {
    [Obj<id<MTLResidencySet>>(residency_set) commit];
  }
}

void ResidencySetRequestResidency(void* residency_set) {
  if (residency_set == nullptr) return;
  if (@available(macOS 15.0, *)) {
    [Obj<id<MTLResidencySet>>(residency_set) requestResidency];
  }
}

void ResidencySetEndResidency(void* residency_set) {
  if (residency_set == nullptr) return;
  if (@available(macOS 15.0, *)) {
    [Obj<id<MTLResidencySet>>(residency_set) endResidency];
  }
}

uint64_t ResidencySetAllocatedSize(void* residency_set) {
  if (residency_set == nullptr) return 0;
  if (@available(macOS 15.0, *)) {
    return [Obj<id<MTLResidencySet>>(residency_set) allocatedSize];
  }
  return 0;
}

uint64_t BufferAllocatedSize(void* buffer) {
  if (buffer == nullptr) return 0;
  return [Obj<id<MTLBuffer>>(buffer) allocatedSize];
}

void CommandQueueAddResidencySet(void* queue, void* residency_set) {
  if (queue == nullptr || residency_set == nullptr) return;
  if (@available(macOS 15.0, *)) {
    [Obj<id<MTLCommandQueue>>(queue)
        addResidencySet:Obj<id<MTLResidencySet>>(residency_set)];
  }
}

uint64_t RecommendedMaxWorkingSetSize(void* device) {
  if (device == nullptr) return 0;
  return [Obj<id<MTLDevice>>(device) recommendedMaxWorkingSetSize];
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

static std::mutex g_enc_mu;
static std::unordered_map<void*, id<MTLComputeCommandEncoder>> g_open_enc;

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

static absl::Status EncodeComputeInto(
    id<MTLCommandBuffer> command_buffer, void* pipeline, void* function,
    bool use_argument_buffer, absl::Span<const MetalKernelArgument> arguments,
    absl::string_view name, const ThreadDim& thread_dims,
    const BlockDim& block_dims, int64_t shmem_bytes = 0) {
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
      if (arg.kind == MetalKernelArgument::Kind::kBuffer) {
        [argument_encoder setBuffer:Obj<id<MTLBuffer>>(arg.buffer)
                             offset:arg.offset
                            atIndex:i];
        if (arg.buffer != nullptr) {
          [encoder useResource:Obj<id<MTLBuffer>>(arg.buffer)
                         usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        }
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
      if (arg.kind == MetalKernelArgument::Kind::kBuffer) {
        [encoder setBuffer:Obj<id<MTLBuffer>>(arg.buffer)
                    offset:arg.offset
                   atIndex:i];
      } else {
        [encoder setBytes:arg.bytes length:arg.bytes_size atIndex:i];
      }
    }
  }

  if (shmem_bytes > 0) {
    [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(shmem_bytes)
                               atIndex:0];
  }
  MTLSize threads_per_threadgroup =
      MTLSizeMake(thread_dims.x, thread_dims.y, thread_dims.z);
  MTLSize threadgroups = MTLSizeMake(block_dims.x, block_dims.y, block_dims.z);
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
  MetalBlitStatsRecord(/*is_fill=*/false, size);
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

absl::Status EncodeBlitFill(void* batch_command_buffer, void* buffer,
                            uint64_t offset, uint64_t size, uint8_t value) {
  if (batch_command_buffer == nullptr || buffer == nullptr) {
    return absl::InternalError("Metal EncodeBlitFill: null buffer.");
  }
  if (size == 0) return absl::OkStatus();
  MetalBlitStatsRecord(/*is_fill=*/true, size);
  @autoreleasepool {
    id<MTLCommandBuffer> cb =
        Obj<MPSCommandBuffer*>(batch_command_buffer).commandBuffer;
    FlushConcurrentEncoder(cb);  // end open compute encoder pre-blit
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:Obj<id<MTLBuffer>>(buffer)
               range:NSMakeRange(offset, size)
               value:value];
    [blit endEncoding];
  }
  return absl::OkStatus();
}

void MetalKprofMaybeStart();
void MetalKprofReport();

void* CommitBatchCommandBuffer(void* batch_command_buffer) {
  if (batch_command_buffer == nullptr) return nullptr;
  MetalKprofMaybeStart();  // env-gated: start a profiling session on first use
  @autoreleasepool {
    MPSCommandBuffer* mpscb = Obj<MPSCommandBuffer*>(batch_command_buffer);
    FlushConcurrentEncoder(mpscb.commandBuffer);
    [mpscb commit];
    id<MTLCommandBuffer> committed = mpscb.commandBuffer;
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
    std::move(on_complete)();
    return;
  }
  MetalKprofMaybeStart();  // env-gated: start a profiling session on first use
  @autoreleasepool {
    MPSCommandBuffer* mpscb = Obj<MPSCommandBuffer*>(batch_command_buffer);
    FlushConcurrentEncoder(mpscb.commandBuffer);
    auto* cb = new absl::AnyInvocable<void() &&>(std::move(on_complete));
    [mpscb.commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
      std::move (*cb)();
      delete cb;
    }];
    [mpscb commit];
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
    if (ev.signaledValue >= value) return absl::OkStatus();
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
  NSData* data = [g_prof_buf resolveCounterRange:NSMakeRange(0, g_prof_used)];
  const int64_t offset =
      static_cast<int64_t>(g_prof_wall0) - static_cast<int64_t>(g_prof_gpu0);
  if (data != nil &&
      data.length >= g_prof_used * sizeof(MTLCounterResultTimestamp)) {
    const auto* r = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
    for (const auto& p : g_prof_pending) {
      uint64_t st = r[p.start_idx].timestamp;
      uint64_t en = r[p.start_idx + 1].timestamp;
      if (st == MTLCounterErrorValue || en == MTLCounterErrorValue) {
        ++g_prof_dropped;
        continue;
      }
      if (en < st) en = st;  // clamp; 0-duration event still counted
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

static bool MetalKprofKeepGrid() {
  static const bool on = [] {
    const char* e = std::getenv("METAL_DFLASH_KPROF");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

static bool MetalKprofWanted() {
  static const bool on = [] {
    const char* e = std::getenv("METAL_KPROF");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on || MetalKprofKeepGrid();
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
  static std::vector<std::pair<uint64_t, uint64_t>> intervals;  // (start,end) ns
  static int commits = 0;
  static uint64_t total_ns = 0;

  std::vector<MetalProfileEvent> evs = MetalProfilingDrain();
  std::lock_guard<std::mutex> lock(mu);
  for (const auto& ev : evs) {
    std::string key = ev.name;
    auto pos = key.find(" grid=");
    if (pos != std::string::npos && !MetalKprofKeepGrid()) key.resize(pos);
    Agg& a = agg[key];
    const uint64_t d = ev.end_ns - ev.start_ns;
    a.ns += d;
    a.count += 1;
    total_ns += d;
    intervals.push_back({ev.start_ns, ev.end_ns});  // for the timeline analysis
  }
  if (++commits < report_every) return;
  double span_ms = 0, busy_ms = 0, concurrency = 0;
  if (!intervals.empty()) {
    std::sort(intervals.begin(), intervals.end());
    uint64_t min_s = intervals.front().first, max_e = 0, busy = 0;
    uint64_t cur_s = intervals.front().first, cur_e = intervals.front().second;
    for (const auto& iv : intervals) {
      max_e = std::max(max_e, iv.second);
      if (iv.first <= cur_e) {
        cur_e = std::max(cur_e, iv.second);
      } else {
        busy += cur_e - cur_s;
        cur_s = iv.first;
        cur_e = iv.second;
      }
    }
    busy += cur_e - cur_s;
    span_ms = (max_e - min_s) / 1.0e6;
    busy_ms = busy / 1.0e6;
    concurrency = busy ? static_cast<double>(total_ns) / busy : 0.0;
  }
  std::vector<std::pair<std::string, Agg>> rows(agg.begin(), agg.end());
  std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
  uint64_t total_count = 0;
  for (const auto& [name, a] : rows) total_count += a.count;
  LOG(INFO) << "=== METAL_KPROF: GPU time by kernel over " << commits
            << " commits (total GPU " << (total_ns / 1.0e6) << " ms, "
            << total_count << " dispatches, " << MetalProfilingDroppedCount()
            << " dropped) ===";
  LOG(INFO) << "  TIMELINE: span=" << span_ms << " ms  busy-union=" << busy_ms
            << " ms (" << (span_ms ? 100.0 * busy_ms / span_ms : 0.0)
            << "% of span)  sum_dur=" << (total_ns / 1.0e6)
            << " ms  concurrency=" << concurrency
            << "x (1.0=fully serial; >1=overlap). critical path >= busy-union.";
  for (const auto& [name, a] : rows) {
    LOG(INFO) << "  " << (a.ns / 1.0e6) << " ms  "
              << (total_ns ? 100.0 * a.ns / total_ns : 0.0) << "%  n=" << a.count
              << "  " << name;
  }
  agg.clear();
  intervals.clear();
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

}  // namespace stream_executor::metal
