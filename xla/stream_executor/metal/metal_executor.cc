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

#include "xla/stream_executor/metal/metal_executor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/metal/metal_event.h"
#include "xla/stream_executor/metal/metal_kernel.h"
#include "xla/stream_executor/metal/metal_runtime.h"
#include "xla/stream_executor/metal/metal_stream.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::metal {
namespace {

bool Contains(const MetalExecutor::Allocation& allocation, const void* ptr) {
  auto base = reinterpret_cast<uintptr_t>(allocation.contents);
  auto value = reinterpret_cast<uintptr_t>(ptr);
  return value >= base && value < base + allocation.size;
}

bool IsMetallib(absl::Span<const uint8_t> payload) {
  constexpr char kMetallibMagic[] = "MTLB";
  return payload.size() >= 4 &&
         std::memcmp(payload.data(), kMetallibMagic, 4) == 0;
}

absl::StatusOr<void*> LoadMetalLibrary(void* device,
                                       absl::Span<const uint8_t> payload) {
  if (IsMetallib(payload)) {
    return LoadLibraryFromData(device, payload);
  }
  std::string source(reinterpret_cast<const char*>(payload.data()),
                     payload.size());
  return CompileLibrary(device, source);
}

}  // namespace

MetalExecutor::~MetalExecutor() {
  {
    absl::MutexLock lock(allocations_mu_);
    for (const auto& [_, allocation] : allocations_) {
      ReleaseObject(allocation.buffer);
    }
  }
  {
    absl::MutexLock lock(modules_mu_);
    for (const auto& [_, module] : loaded_modules_) {
      ReleaseObject(module);
    }
  }
  ReleaseObject(shared_event_listener_);
  ReleaseObject(shared_event_);
  ReleaseObject(command_queue_);
  ReleaseObject(device_);
}

absl::Status MetalExecutor::Init() {
  TF_ASSIGN_OR_RETURN(device_, RetainDevice(device_ordinal()));
  TF_ASSIGN_OR_RETURN(command_queue_, NewCommandQueue(device_));
  // One shared event per device orders dependent executes on the GPU (see
  // metal_runtime.h / MetalStream). nil only if the driver refuses it — treat as
  // fatal-soft: a null shared_event_ makes the event ops no-op and the stream
  // falls back to host syncs (correct, just unpipelined).
  shared_event_ = NewSharedEvent(device_);
  if (shared_event_ == nullptr) {
    LOG(WARNING) << "Metal: newSharedEvent failed; falling back to host-side "
                    "execute ordering (no GPU pipelining).";
  } else {
    // One listener per device drives host callbacks (SetStateConcrete) off
    // shared-event values without a per-execute commit (see metal_runtime.h).
    shared_event_listener_ = NewSharedEventListener();
    if (shared_event_listener_ == nullptr) {
      LOG(WARNING) << "Metal: MTLSharedEventListener creation failed; host "
                      "callbacks fall back to per-buffer completion handlers.";
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> MetalExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  if (!spec.has_metal_library_in_memory()) {
    return absl::UnimplementedError(
        "Metal only loads in-memory metallib data.");
  }
  auto payload = spec.metal_library_in_memory();
  if (!payload.has_value()) {
    return absl::InvalidArgumentError("Metal kernel payload is empty.");
  }
  TF_ASSIGN_OR_RETURN(auto kernel,
                      LoadKernelFromLibraryPayload(payload->metallib_bytes,
                                                   spec.kernel_name(),
                                                   spec.arity()));
  if (std::holds_alternative<KernelLoaderSpec::KernelArgsPackingFunc>(
          spec.kernel_args_packing())) {
    kernel->set_args_packing(
        std::get<KernelLoaderSpec::KernelArgsPackingFunc>(
            spec.kernel_args_packing()));
  } else {
    const auto& packing_spec =
        std::get<KernelArgsPackingSpec>(spec.kernel_args_packing());
    kernel->set_args_packing(
        [packing_spec](const Kernel& kernel, const KernelArgs& args) {
          const auto& mem_args = Cast<KernelArgsDeviceAddressArray>(&args);
          return packing_spec.BuildArguments(mem_args->packed_args(),
                                             args.number_of_shared_bytes());
        });
  }
  return kernel;
}

absl::StatusOr<std::unique_ptr<Kernel>>
MetalExecutor::LoadKernelFromLibraryPayload(
    absl::Span<const uint8_t> payload, const std::string& kernel_name,
    size_t arity) {
  TF_ASSIGN_OR_RETURN(void* library, LoadMetalLibrary(device_, payload));
  TF_ASSIGN_OR_RETURN(void* function, NewFunction(library, kernel_name));
  TF_ASSIGN_OR_RETURN(void* pipeline, NewComputePipeline(device_, function));

  auto kernel = std::make_unique<MetalKernel>(this);
  kernel->set_name(kernel_name);
  kernel->set_arity(arity);
  kernel->set_library(library);
  kernel->set_function(function);
  kernel->set_pipeline(pipeline);
  kernel->set_uses_argument_buffer(arity > 31);
  return kernel;
}

absl::StatusOr<std::unique_ptr<Kernel>> MetalExecutor::LoadKernelWithConstants(
    absl::Span<const uint8_t> metallib, const std::string& kernel_name,
    size_t arity, absl::Span<const MetalFunctionConstant> constants) {
  TF_ASSIGN_OR_RETURN(void* library, LoadMetalLibrary(device_, metallib));
  TF_ASSIGN_OR_RETURN(
      void* function,
      NewFunctionWithConstants(library, kernel_name, constants));
  TF_ASSIGN_OR_RETURN(void* pipeline, NewComputePipeline(device_, function));

  auto kernel = std::make_unique<MetalKernel>(this);
  kernel->set_name(kernel_name);
  kernel->set_arity(arity);
  kernel->set_library(library);
  kernel->set_function(function);
  kernel->set_pipeline(pipeline);
  kernel->set_uses_argument_buffer(arity > 31);
  return kernel;
}

absl::StatusOr<ModuleHandle> MetalExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  if (!spec.has_metal_library_in_memory()) {
    return absl::UnimplementedError(
        "Metal only loads in-memory metallib data.");
  }
  absl::Span<const uint8_t> payload = spec.metal_library_in_memory();
  TF_ASSIGN_OR_RETURN(void* library, LoadMetalLibrary(device_, payload));
  ModuleHandle handle(library);
  absl::MutexLock lock(modules_mu_);
  loaded_modules_.emplace(handle.id(), library);
  return handle;
}

bool MetalExecutor::UnloadModule(ModuleHandle module_handle) {
  absl::MutexLock lock(modules_mu_);
  auto it = loaded_modules_.find(module_handle.id());
  if (it == loaded_modules_.end()) {
    return false;
  }
  ReleaseObject(it->second);
  loaded_modules_.erase(it);
  return true;
}

DeviceAddressBase MetalExecutor::Allocate(uint64_t size, int64_t memory_space) {
  auto memory_space_kind = static_cast<MemorySpace>(memory_space);
  if (memory_space_kind == MemorySpace::kCollective) {
    LOG(ERROR) << "Metal does not support collective memory yet.";
    return DeviceAddressBase(nullptr, 0);
  }
  void* contents = nullptr;
  auto buffer = NewSharedBuffer(device_, size, &contents);
  if (!buffer.ok()) {
    LOG(ERROR) << buffer.status();
    return DeviceAddressBase(nullptr, 0);
  }
  if (size != 0) {
    absl::MutexLock lock(allocations_mu_);
    allocations_.emplace(reinterpret_cast<uintptr_t>(contents),
                         Allocation{*buffer, contents, size});
  }
  return DeviceAddressBase(contents, size);
}

void MetalExecutor::Deallocate(DeviceAddressBase* mem) {
  if (mem == nullptr || mem->opaque() == nullptr) {
    return;
  }
  absl::MutexLock lock(allocations_mu_);
  auto it = allocations_.find(reinterpret_cast<uintptr_t>(mem->opaque()));
  if (it == allocations_.end()) {
    LOG(ERROR) << "Attempted to deallocate unknown Metal allocation "
               << mem->opaque();
    *mem = DeviceAddressBase(nullptr, 0);
    return;
  }
  ReleaseObject(it->second.buffer);
  allocations_.erase(it);
  *mem = DeviceAddressBase(nullptr, 0);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
MetalExecutor::CreateMemoryAllocator(MemorySpace memory_space) {
  if (memory_space == MemorySpace::kCollective) {
    return absl::UnimplementedError(
        "Metal collective memory space is not implemented.");
  }
  return std::make_unique<GenericMemoryAllocator>(
      [this](uint64_t size)
          -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
        DeviceAddressBase address = Allocate(size, 0);
        if (address.is_null() && size != 0) {
          return absl::ResourceExhaustedError(
              absl::StrCat("Failed to allocate ", size,
                           " bytes of Metal shared memory."));
        }
        return std::make_unique<GenericMemoryAllocation>(
            address.opaque(), size, [this](void* ptr, uint64_t size) {
              DeviceAddressBase mem(ptr, size);
              Deallocate(&mem);
            });
      });
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
MetalExecutor::HostMemoryAllocate(uint64_t size) {
  void* ptr = ::operator new(size);
  return std::make_unique<GenericMemoryAllocation>(
      ptr, size, [](void* ptr, uint64_t) { ::operator delete(ptr); });
}

absl::Status MetalExecutor::SynchronousMemcpy(DeviceAddressBase* device_dst,
                                              const void* host_src,
                                              uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (device_dst == nullptr || device_dst->opaque() == nullptr) {
    return absl::InvalidArgumentError("Metal H2D destination is null.");
  }
  std::memcpy(device_dst->opaque(), host_src, size);
  return absl::OkStatus();
}

absl::Status MetalExecutor::SynchronousMemcpy(
    void* host_dst, const DeviceAddressBase& device_src, uint64_t size) {
  if (size == 0) return absl::OkStatus();
  if (host_dst == nullptr || device_src.opaque() == nullptr) {
    return absl::InvalidArgumentError("Metal D2H source or destination is null.");
  }
  std::memcpy(host_dst, device_src.opaque(), size);
  return absl::OkStatus();
}

bool MetalExecutor::SynchronizeAllActivity() {
  return SynchronizeCommandQueue(command_queue_).ok();
}

absl::StatusOr<std::unique_ptr<Stream>> MetalExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  return std::make_unique<MetalStream>(this, priority);
}

absl::StatusOr<std::unique_ptr<Event>> MetalExecutor::CreateEvent() {
  return std::make_unique<MetalEvent>();
}

void MetalExecutor::RegisterStream(MetalStream* stream) {
  absl::MutexLock lock(streams_mu_);
  streams_.push_back(stream);
}

void MetalExecutor::UnregisterStream(MetalStream* stream) {
  absl::MutexLock lock(streams_mu_);
  streams_.erase(std::remove(streams_.begin(), streams_.end(), stream),
                 streams_.end());
}

void MetalExecutor::CommitOpenBufferThrough(uint64_t value) {
  if (value == 0) return;
  absl::MutexLock lock(streams_mu_);
  for (MetalStream* stream : streams_) {
    stream->FlushOpenBufferIfCarrying(value);
  }
}

absl::Status MetalExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  return absl::UnimplementedError("Metal peer access is not implemented.");
}

bool MetalExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  auto info = GetDeviceInfo(device_ordinal());
  if (!info.ok()) return false;
  if (total != nullptr) {
    *total = info->recommended_max_working_set_size;
  }
  if (free != nullptr) {
    *free = -1;
  }
  return true;
}

namespace {

// Published LPDDR peak bandwidth per Apple Silicon part.
//
// Nothing on the system reports DRAM bandwidth: Metal has no query, the
// IORegistry AGXAccelerator node carries only "gpu-core-count", the SoC nodes
// expose "dramcfg-data" as an undocumented blob, and sysctl has nothing. So it
// is looked up from the two things the device does report -- the GPU
// generation, parsed out of [MTLDevice architecture] ("applegpu_g16s" -> 16),
// and the IORegistry core count, which together name the part exactly. Within a
// generation, core count identifies the die tier (base / Pro / Max / Ultra) and
// Apple sizes the memory bus by that same tier.
//
// Rows are ascending by core budget within a generation and the first match
// wins, i.e. the smallest row whose budget covers `cores`. Binned-down GPUs
// therefore land on their own die's row: a 7-core M1, a 14-core M1 Pro and an
// 8-core M4 resolve to the base part, and the Max bins that genuinely differ
// (M3 30 vs 40, M4 32 vs 40, M5 32 vs 40) split at the right boundary.
//
// Values are bus width x transfer rate, not Apple's rounded marketing figures:
// 409.6 rather than "400". The distinction matters here because these numbers
// are also the denominator when reading a measured GB/s back as a fraction of
// peak, and rounding down manufactures >100%-of-peak artifacts (this backend's
// NVFP4 GEMVs measure 533-558 GB/s on a 40-core M4 Max, against 546.1 real and
// "546" advertised).
int64_t AppleGpuMemoryBandwidth(const MetalComputeCapability& cc,
                                uint64_t core_count) {
  struct Part {
    int gen;
    int64_t max_cores;
    int64_t bytes_per_second;
  };
  static constexpr Part kParts[] = {
      {13, 8, 68'250'000'000},    // M1            128-bit LPDDR4X-4266
      {13, 16, 204'800'000'000},  // M1 Pro        256-bit LPDDR5-6400
      {13, 32, 409'600'000'000},  // M1 Max        512-bit
      {13, 64, 819'200'000'000},  // M1 Ultra     1024-bit
      {14, 10, 102'400'000'000},  // M2            128-bit LPDDR5-6400
      {14, 19, 204'800'000'000},  // M2 Pro        256-bit
      {14, 38, 409'600'000'000},  // M2 Max        512-bit
      {14, 76, 819'200'000'000},  // M2 Ultra     1024-bit
      {15, 10, 102'400'000'000},  // M3            128-bit LPDDR5-6400
      {15, 18, 153'600'000'000},  // M3 Pro        192-bit
      {15, 30, 307'200'000'000},  // M3 Max        384-bit (30-core bin)
      {15, 40, 409'600'000'000},  // M3 Max        512-bit (40-core bin)
      {15, 80, 819'200'000'000},  // M3 Ultra     1024-bit
      {16, 10, 120'000'000'000},  // M4            128-bit LPDDR5X-7500
      {16, 20, 273'100'000'000},  // M4 Pro        256-bit LPDDR5X-8533
      {16, 32, 409'600'000'000},  // M4 Max        384-bit (32-core bin)
      {16, 40, 546'100'000'000},  // M4 Max        512-bit (40-core bin)
      {17, 10, 153'600'000'000},  // M5            128-bit LPDDR5X-9600
      {17, 20, 307'200'000'000},  // M5 Pro        256-bit
      {17, 32, 460'800'000'000},  // M5 Max        384-bit (32-core bin)
      {17, 40, 614'400'000'000},  // M5 Max        512-bit (40-core bin)
  };
  // Complete for shipping silicon: the Ultra line last updated at M3, and
  // neither an M4 nor an M5 Ultra exists.
  //
  // TODO: the M5 rows are unverified on hardware -- they assume M5 reports
  // generation 17, i.e. that [MTLDevice architecture] reads "applegpu_g17*".
  // Corroborated but not proven: MLX gates its neural-accelerator path on
  // `gen >= (arch == 'p' ? 18 : 17)` (mlx/backend/metal/device.cpp,
  // is_nax_available), and that path is M5-and-up, so MLX makes the same
  // assumption. If Apple's numbering skips, these four rows never match and the
  // per-core fallback below covers M5 instead -- degraded, not broken.

  const int gen = cc.architecture_gen();
  const int64_t cores = static_cast<int64_t>(core_count);
  if (cores > 0) {
    for (const Part& part : kParts) {
      if (part.gen == gen && cores <= part.max_cores) {
        return part.bytes_per_second;
      }
    }
  }

  // Unknown silicon (a generation past the table, an Ultra tier that does not
  // exist yet, or a failed architecture/core-count query). Apple sizes the bus
  // by die tier and scales GPU cores with the same tier, so bandwidth per core
  // stays in a narrow band across the whole table -- 8.5 GB/s on M1, rising to
  // 15.4 on M5 -- which is what makes a single per-core constant a usable
  // stand-in for a missing row. It tracks the recent end of that band rather
  // than its middle, because the only parts that can land here are newer than
  // the table and the trend has been monotonically upward.
  //
  // Clamped so a garbage core count degrades to a plausible bandwidth rather
  // than to zero, which is the degenerate case this whole field exists to
  // avoid.
  constexpr int64_t kFallbackBytesPerSecondPerCore = 15'000'000'000;
  const int64_t assumed_cores = cores > 0 ? cores : 8;
  return std::clamp<int64_t>(assumed_cores * kFallbackBytesPerSecondPerCore,
                             60'000'000'000, 3'000'000'000'000);
}

}  // namespace

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MetalExecutor::CreateDeviceDescription() const {
  TF_ASSIGN_OR_RETURN(MetalDeviceInfo info, GetDeviceInfo(device_ordinal()));
  auto desc = std::make_unique<DeviceDescription>();
  desc->set_device_vendor("Apple");
  desc->set_platform_version("Metal");
  // Metal is a first-class GpuComputeCapability alternative (no longer a faked
  // CUDA cap). With a MetalComputeCapability in the variant, cuda_compute_capability()
  // returns nullptr, so GpuCompiler's force-upcast (gpu_compiler.cc:705, gated on
  // a non-null sub-Volta CUDA cap) and the Ampere-gated Triton/GemmFusion passes
  // all self-skip — the exact behavior the old {7,5} masquerade hacked toward,
  // now honest. IsNvidiaGpu() is correctly false for Metal.
  desc->set_gpu_compute_capability(
      GpuComputeCapability(MetalComputeCapability(info.architecture)));
  desc->set_driver_version(SemanticVersion{0, 0, 0});
  desc->set_runtime_version(SemanticVersion{0, 0, 0});
  desc->set_compile_time_toolkit_version(SemanticVersion{0, 0, 0});
  desc->set_name(info.name);
  desc->set_model_str(info.name);
  desc->set_pci_bus_id(info.registry_id);
  desc->set_numa_node(0);
  desc->set_thread_dim_limit(ThreadDim(1024, 1024, 64));
  desc->set_block_dim_limit(BlockDim(2147483647, 65535, 65535));
  desc->set_threads_per_block_limit(
      info.max_threads_per_threadgroup == 0
          ? 1024
          : static_cast<int64_t>(info.max_threads_per_threadgroup));
  desc->set_threads_per_core_limit(1024);
  desc->set_threads_per_warp(32);
  desc->set_registers_per_core_limit(0);
  desc->set_registers_per_block_limit(0);
  desc->set_device_address_bits(64);
  desc->set_device_memory_size(info.recommended_max_working_set_size);
  desc->set_l2_cache_size(0);
  // Non-zero, order-of-magnitude-correct DRAM bandwidth. This field feeds ONLY
  // the fusion cost model (GpuPerformanceModelBase::Read/WriteTime and
  // GpuIndexingPerformanceModel), never codegen — see the clock_rate comment
  // below for why that distinction is what makes this safe.
  //
  // At 0, `WriteTime = absl::Seconds(bytes / 0)` = +InfiniteDuration for EVERY
  // kernel, so PriorityFusion's producer priority
  // (`time_unfused - time_fused`) is +Inf - +Inf. absl::Duration saturates
  // rather than producing NaN, but the result is meaningless and the cost model
  // can no longer rank anything: on gemma-4-26B-A4B decode this left 51
  // dispatches per layer, ~12 of them kernels that exist only to broadcast a
  // scalar or materialize a small constant into their single consumer.
  //
  // Look the part up by (GPU generation, core count) -- see
  // AppleGpuMemoryBandwidth. Both inputs are reported by the device; bandwidth
  // itself is not, by Metal or the IORegistry or sysctl.
  //
  // Getting this per-part matters because of what the model does with it. A
  // uniform bandwidth error cancels out of PriorityFusion's
  // `time_unfused - time_fused` (both sides scale together), so the absolute
  // value is not what counts -- the flops-per-byte ratio is, because that is
  // what decides whether a kernel is modelled as compute- or memory-bound.
  // ComputeTime already scales with the real core count, so a bandwidth
  // CONSTANT made that ratio a function of the part: at a flat 400 GB/s a
  // 40-core M4 Max modelled 17.9 flop/byte against a true ~26, while a 10-core
  // base M4 modelled 4.5 against a true ~15 -- three times more memory-bound
  // than the silicon is.
  desc->set_memory_bandwidth(AppleGpuMemoryBandwidth(
      MetalComputeCapability(info.architecture), info.gpu_core_count));
  desc->set_pcie_bandwidth(0);
  // Threadgroup-memory budget from the device, not a hardcoded 32 KB. Every
  // shipping Apple GPU (M1..M4) reports 32768, so this is byte-identical today,
  // but it makes the scheduler's resource model track the real device limit
  // instead of an assumption. Fall back to the historical 32 KB if the query is
  // unavailable (info == 0).
  const int64_t tg_mem = info.max_threadgroup_memory_length == 0
                             ? 32 * 1024
                             : static_cast<int64_t>(
                                   info.max_threadgroup_memory_length);
  desc->set_shared_memory_per_core(tg_mem);
  desc->set_shared_memory_per_block(tg_mem);
  desc->set_shared_memory_per_block_optin(tg_mem);
  // Apple GPU shader clock (~1.3-1.6 GHz across M1..M4) and ALUs per GPU core
  // (128, an Apple GPU architectural constant). Like memory_bandwidth these are
  // cost-model-only inputs: clock_rate_ghz and fpus_per_core are read solely by
  // GpuPerformanceModelBase (CalculateEffectiveFlopsPerNs) and the autotune
  // cache key. At 0/1 the effective FLOP rate is 0, so ComputeTime is +Inf for
  // every kernel and the model is degenerate in the same way the zero bandwidth
  // made it.
  //
  // Unlike bandwidth this one cannot be derived, and the search is recorded so
  // it is not repeated: Metal exposes no clock, the IORegistry AGXAccelerator
  // node carries no frequency or performance-state property (the `frequency-*`
  // keys elsewhere in the tree are PLL tolerances for a clock source, not the
  // GPU), and sysctl has nothing. powermetrics can read it but needs root and a
  // sampling window. The spread across shipping parts is ~1.2x and it enters
  // the model only through the same flop/byte ratio that the derived bandwidth
  // now fixes, so a constant is honest here in a way it was not for bandwidth,
  // which was off by the part's tier.
  desc->set_clock_rate_ghz(1.4);
  // Real GPU core count, from the IORegistry (Metal has no such query). At the
  // historical value of 1 the cost model modelled a single-core GPU, so
  // CalculateEffectiveFlopsPerNs clamped n_active_core to 1 and overestimated
  // every kernel's compute time by ~core_count, biasing PriorityFusion against
  // fusing. Measured on gemma-4-26B-A4B decode (M4 Max, 40 cores): 1344 -> 1194
  // dispatches/token and 73.88 -> 77.49 tok/s at bs=1, bit-identical output.
  //
  // Unlike memory_bandwidth/clock_rate_ghz this field is NOT cost-model-only —
  // the reduction emitter (min_desired_blocks), the split-K rewriter and the
  // transpose emitter also key their blocking off it, so it can move reduction
  // arithmetic order. Verified on this model that it does not (golden sha
  // unchanged across the 1 -> 40 flip), but re-check the golden when enabling a
  // new model/shape rather than assuming.
  //
  // Fall back to 1 (the historical value) if the registry lookup fails, so a
  // future OS layout change degrades to today's behavior rather than to a
  // fabricated core count.
  desc->set_core_count(info.gpu_core_count == 0
                           ? 1
                           : static_cast<int>(info.gpu_core_count));
  desc->set_fpus_per_core(128);
  desc->set_ecc_enabled(false);
  return desc;
}

absl::StatusOr<DeviceAddressBase> MetalExecutor::GetMemoryRange(
    const DeviceAddressBase& location) const {
  TF_ASSIGN_OR_RETURN(Allocation allocation,
                      ResolveAllocation(location.opaque()));
  return DeviceAddressBase(allocation.contents, allocation.size);
}

absl::StatusOr<MemorySpace> MetalExecutor::GetPointerMemorySpace(
    const void* ptr) {
  TF_RETURN_IF_ERROR(ResolveAllocation(ptr).status());
  return MemorySpace::kDevice;
}

absl::StatusOr<MetalExecutor::Allocation> MetalExecutor::ResolveAllocation(
    const void* ptr) const {
  if (ptr == nullptr) {
    return absl::InvalidArgumentError("Cannot resolve null Metal pointer.");
  }
  absl::MutexLock lock(allocations_mu_);
  // Allocations are keyed by base address and ordered, so the only allocation
  // that can contain an interior pointer is the one with the greatest base
  // <= ptr (the predecessor of upper_bound). The ranges are non-overlapping
  // (distinct MTLBuffers), so that predecessor is the unique candidate — but it
  // still needs the containment check, since ptr may fall past its end.
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = allocations_.upper_bound(addr);
  if (it != allocations_.begin()) {
    --it;
    if (Contains(it->second, ptr)) {
      return it->second;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("No Metal allocation contains pointer %p", ptr));
}

}  // namespace stream_executor::metal
