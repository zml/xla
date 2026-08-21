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

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rocblas/rocblas.h"
#include "xla/stream_executor/rocm/rocm_host_overhead_benchmark_kernels.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"

namespace stream_executor::gpu {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int kBatchSize = 256;

double Seconds(Clock::duration duration) {
  return std::chrono::duration<double>(duration).count();
}

bool CheckHip(benchmark::State& state, hipError_t result,
              const char* operation) {
  if (result == hipSuccess) return true;
  std::string message = std::string(operation) + ": " +
                        hipGetErrorName(result) + " (" +
                        hipGetErrorString(result) + ")";
  state.SkipWithError(message.c_str());
  return false;
}

bool CheckRocblas(benchmark::State& state, rocblas_status result,
                  const char* operation) {
  if (result == rocblas_status_success) return true;
  std::string message = std::string(operation) + ": rocBLAS status " +
                        std::to_string(static_cast<int>(result));
  state.SkipWithError(message.c_str());
  return false;
}

class HipSubAllocator : public tsl::SubAllocator {
 public:
  HipSubAllocator() : SubAllocator({}, {}) {}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    (void)alignment;
    void* pointer = nullptr;
    if (hipMalloc(&pointer, num_bytes) != hipSuccess) return nullptr;
    *bytes_received = num_bytes;
    return pointer;
  }

  void Free(void* pointer, size_t num_bytes) override {
    (void)num_bytes;
    (void)hipFree(pointer);
  }

  bool SupportsCoalescing() const override { return false; }
  tsl::AllocatorMemoryType GetMemoryType() const override {
    return tsl::AllocatorMemoryType::kDevice;
  }
};

void SetBatchCounters(benchmark::State& state, double enqueue_seconds,
                      double sync_seconds, int64_t operations) {
  state.counters["enqueue_ns/op"] = enqueue_seconds * 1e9 / operations;
  state.counters["sync_ns/batch"] =
      sync_seconds * 1e9 / state.iterations();
  state.counters["total_ns/op"] =
      (enqueue_seconds + sync_seconds) * 1e9 / operations;
  state.SetItemsProcessed(operations);
}

void BM_ModuleLaunchKernel(benchmark::State& state, int arity,
                           bool packed_buffer, unsigned int stream_flags) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  hipFunction_t function = nullptr;
  if (!CheckHip(state, GetRocmHostOverheadBenchmarkKernel(arity, &function),
                "GetRocmHostOverheadBenchmarkKernel")) {
    return;
  }

  hipStream_t stream = nullptr;
  if (!CheckHip(state, hipStreamCreateWithFlags(&stream, stream_flags),
                "hipStreamCreateWithFlags")) {
    return;
  }

  std::array<uint64_t, 64> values{};
  std::array<void*, 64> parameter_addresses{};
  for (int i = 0; i < arity; ++i) {
    parameter_addresses[i] = &values[i];
  }

  size_t packed_size = arity * sizeof(uint64_t);
  void* packed_config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, values.data(),
                           HIP_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                           HIP_LAUNCH_PARAM_END};
  void** kernel_params = arity == 0 ? nullptr : parameter_addresses.data();
  void** extra = packed_buffer ? packed_config : nullptr;
  if (packed_buffer) kernel_params = nullptr;

  if (!CheckHip(state,
                hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream,
                                      kernel_params, extra),
                "warm-up hipModuleLaunchKernel") ||
      !CheckHip(state, hipStreamSynchronize(stream),
                "warm-up hipStreamSynchronize")) {
    (void)hipStreamDestroy(stream);
    return;
  }

  double enqueue_seconds = 0;
  double sync_seconds = 0;
  int64_t operations = 0;
  for (auto _ : state) {
    auto enqueue_start = Clock::now();
    for (int i = 0; i < kBatchSize; ++i) {
      if (!CheckHip(state,
                    hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0,
                                          stream, kernel_params, extra),
                    "hipModuleLaunchKernel")) {
        break;
      }
    }
    auto enqueue_end = Clock::now();
    if (!CheckHip(state, hipStreamSynchronize(stream),
                  "hipStreamSynchronize")) {
      break;
    }
    auto sync_end = Clock::now();
    enqueue_seconds += Seconds(enqueue_end - enqueue_start);
    sync_seconds += Seconds(sync_end - enqueue_end);
    operations += kBatchSize;
    state.SetIterationTime(Seconds(sync_end - enqueue_start));
  }

  SetBatchCounters(state, enqueue_seconds, sync_seconds, operations);
  (void)hipStreamDestroy(stream);
}

enum class EventVisibility { kSystem, kDevice, kTiming, kTimingNoSystemFence };

unsigned int EventFlags(EventVisibility visibility) {
  switch (visibility) {
    case EventVisibility::kSystem:
      return hipEventDisableTiming | hipEventReleaseToSystem;
    case EventVisibility::kDevice:
      return hipEventDisableTiming | hipEventReleaseToDevice;
    case EventVisibility::kTiming:
      return hipEventDefault;
    case EventVisibility::kTimingNoSystemFence:
      return hipEventDisableSystemFence;
  }
}

void BM_EventRecordWait(benchmark::State& state,
                        EventVisibility visibility) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  hipStream_t producer = nullptr;
  hipStream_t consumer = nullptr;
  if (!CheckHip(state, hipStreamCreateWithFlags(&producer, hipStreamNonBlocking),
                "hipStreamCreateWithFlags(producer)") ||
      !CheckHip(state, hipStreamCreateWithFlags(&consumer, hipStreamNonBlocking),
                "hipStreamCreateWithFlags(consumer)")) {
    if (producer != nullptr) (void)hipStreamDestroy(producer);
    return;
  }

  std::array<hipEvent_t, kBatchSize> events{};
  for (hipEvent_t& event : events) {
    if (!CheckHip(state, hipEventCreateWithFlags(&event, EventFlags(visibility)),
                  "hipEventCreateWithFlags")) {
      for (hipEvent_t created : events) {
        if (created != nullptr) (void)hipEventDestroy(created);
      }
      (void)hipStreamDestroy(consumer);
      (void)hipStreamDestroy(producer);
      return;
    }
  }

  const bool wait_on_consumer =
      visibility == EventVisibility::kSystem ||
      visibility == EventVisibility::kDevice;
  double enqueue_seconds = 0;
  double sync_seconds = 0;
  int64_t operations = 0;
  for (auto _ : state) {
    auto enqueue_start = Clock::now();
    for (hipEvent_t event : events) {
      if (!CheckHip(state, hipEventRecord(event, producer), "hipEventRecord")) {
        break;
      }
      if (wait_on_consumer &&
          !CheckHip(state, hipStreamWaitEvent(consumer, event, 0),
                    "hipStreamWaitEvent")) {
        break;
      }
    }
    auto enqueue_end = Clock::now();
    if (!CheckHip(state, hipStreamSynchronize(producer),
                  "hipStreamSynchronize(producer)") ||
        (wait_on_consumer &&
         !CheckHip(state, hipStreamSynchronize(consumer),
                   "hipStreamSynchronize(consumer)"))) {
      break;
    }
    auto sync_end = Clock::now();
    enqueue_seconds += Seconds(enqueue_end - enqueue_start);
    sync_seconds += Seconds(sync_end - enqueue_end);
    operations += kBatchSize;
    state.SetIterationTime(Seconds(sync_end - enqueue_start));
  }

  SetBatchCounters(state, enqueue_seconds, sync_seconds, operations);
  for (hipEvent_t event : events) (void)hipEventDestroy(event);
  (void)hipStreamDestroy(consumer);
  (void)hipStreamDestroy(producer);
}

enum class StatePattern { kSame, kAlternate, kCachedSame };

void BM_RocblasSetStream(benchmark::State& state, StatePattern pattern) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  hipStream_t streams[2] = {nullptr, nullptr};
  if (!CheckHip(state,
                hipStreamCreateWithFlags(&streams[0], hipStreamNonBlocking),
                "hipStreamCreateWithFlags(0)") ||
      !CheckHip(state,
                hipStreamCreateWithFlags(&streams[1], hipStreamNonBlocking),
                "hipStreamCreateWithFlags(1)")) {
    if (streams[0] != nullptr) (void)hipStreamDestroy(streams[0]);
    return;
  }

  rocblas_handle handle = nullptr;
  if (!CheckRocblas(state, rocblas_create_handle(&handle),
                    "rocblas_create_handle")) {
    (void)hipStreamDestroy(streams[1]);
    (void)hipStreamDestroy(streams[0]);
    return;
  }
  if (!CheckRocblas(state, rocblas_set_stream(handle, streams[0]),
                    "warm-up rocblas_set_stream")) {
    (void)rocblas_destroy_handle(handle);
    (void)hipStreamDestroy(streams[1]);
    (void)hipStreamDestroy(streams[0]);
    return;
  }

  constexpr int kStateBatchSize = 16384;
  hipStream_t cached_stream = streams[0];
  double elapsed_seconds = 0;
  int64_t operations = 0;
  int64_t api_calls = 0;
  for (auto _ : state) {
    auto start = Clock::now();
    for (int i = 0; i < kStateBatchSize; ++i) {
      hipStream_t next = pattern == StatePattern::kAlternate ? streams[i & 1]
                                                             : streams[0];
      if (pattern == StatePattern::kCachedSame && next == cached_stream) {
        benchmark::DoNotOptimize(cached_stream);
        continue;
      }
      if (!CheckRocblas(state, rocblas_set_stream(handle, next),
                        "rocblas_set_stream")) {
        break;
      }
      cached_stream = next;
      ++api_calls;
    }
    auto end = Clock::now();
    elapsed_seconds += Seconds(end - start);
    operations += kStateBatchSize;
    state.SetIterationTime(Seconds(end - start));
  }

  state.counters["api_call_fraction"] =
      static_cast<double>(api_calls) / operations;
  state.counters["ns/op"] = elapsed_seconds * 1e9 / operations;
  state.SetItemsProcessed(operations);
  (void)rocblas_destroy_handle(handle);
  (void)hipStreamDestroy(streams[1]);
  (void)hipStreamDestroy(streams[0]);
}

void BM_RocblasSetAtomicsMode(benchmark::State& state, StatePattern pattern) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  rocblas_handle handle = nullptr;
  if (!CheckRocblas(state, rocblas_create_handle(&handle),
                    "rocblas_create_handle")) {
    return;
  }

  constexpr int kStateBatchSize = 16384;
  rocblas_atomics_mode cached_mode = rocblas_atomics_allowed;
  double elapsed_seconds = 0;
  int64_t operations = 0;
  int64_t api_calls = 0;
  for (auto _ : state) {
    auto start = Clock::now();
    for (int i = 0; i < kStateBatchSize; ++i) {
      rocblas_atomics_mode next =
          pattern == StatePattern::kAlternate
              ? (i & 1 ? rocblas_atomics_not_allowed : rocblas_atomics_allowed)
              : rocblas_atomics_allowed;
      if (pattern == StatePattern::kCachedSame && next == cached_mode) {
        benchmark::DoNotOptimize(cached_mode);
        continue;
      }
      if (!CheckRocblas(state, rocblas_set_atomics_mode(handle, next),
                        "rocblas_set_atomics_mode")) {
        break;
      }
      cached_mode = next;
      ++api_calls;
    }
    auto end = Clock::now();
    elapsed_seconds += Seconds(end - start);
    operations += kStateBatchSize;
    state.SetIterationTime(Seconds(end - start));
  }

  state.counters["api_call_fraction"] =
      static_cast<double>(api_calls) / operations;
  state.counters["ns/op"] = elapsed_seconds * 1e9 / operations;
  state.SetItemsProcessed(operations);
  (void)rocblas_destroy_handle(handle);
}

struct GraphData {
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  std::vector<hipGraphNode_t> nodes;
  uint64_t argument = 0;
  void* argument_address = &argument;
  hipKernelNodeParams params{};
};

void DestroyGraph(GraphData& data) {
  if (data.exec != nullptr) (void)hipGraphExecDestroy(data.exec);
  if (data.graph != nullptr) (void)hipGraphDestroy(data.graph);
  data.exec = nullptr;
  data.graph = nullptr;
}

bool CreateGraph(benchmark::State& state, hipFunction_t function,
                 int node_count, bool instantiate, GraphData& data) {
  if (!CheckHip(state, hipGraphCreate(&data.graph, 0), "hipGraphCreate")) {
    return false;
  }

  data.params.func = function;
  data.params.gridDim = dim3(1, 1, 1);
  data.params.blockDim = dim3(1, 1, 1);
  data.params.sharedMemBytes = 0;
  data.params.kernelParams = &data.argument_address;
  data.params.extra = nullptr;
  data.nodes.reserve(node_count);

  hipGraphNode_t dependency = nullptr;
  for (int i = 0; i < node_count; ++i) {
    hipGraphNode_t node = nullptr;
    const hipGraphNode_t* dependencies = i == 0 ? nullptr : &dependency;
    size_t dependency_count = i == 0 ? 0 : 1;
    if (!CheckHip(state,
                  hipGraphAddKernelNode(&node, data.graph, dependencies,
                                        dependency_count, &data.params),
                  "hipGraphAddKernelNode")) {
      DestroyGraph(data);
      return false;
    }
    data.nodes.push_back(node);
    dependency = node;
  }

  if (instantiate &&
      !CheckHip(state,
                hipGraphInstantiate(&data.exec, data.graph, nullptr, nullptr, 0),
                "hipGraphInstantiate")) {
    DestroyGraph(data);
    return false;
  }
  return true;
}

void BM_GraphNodeUpdates(benchmark::State& state, int node_count) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  hipFunction_t function = nullptr;
  if (!CheckHip(state, GetRocmHostOverheadBenchmarkKernel(1, &function),
                "GetRocmHostOverheadBenchmarkKernel")) {
    return;
  }

  GraphData data;
  if (!CreateGraph(state, function, node_count, true, data)) return;

  double elapsed_seconds = 0;
  int64_t updates = 0;
  for (auto _ : state) {
    data.argument ^= 1;
    auto start = Clock::now();
    for (hipGraphNode_t node : data.nodes) {
      if (!CheckHip(state,
                    hipGraphExecKernelNodeSetParams(data.exec, node,
                                                    &data.params),
                    "hipGraphExecKernelNodeSetParams")) {
        break;
      }
      ++updates;
    }
    auto end = Clock::now();
    elapsed_seconds += Seconds(end - start);
    state.SetIterationTime(Seconds(end - start));
  }

  state.counters["ns/node"] = elapsed_seconds * 1e9 / updates;
  state.SetItemsProcessed(updates);
  DestroyGraph(data);
}

void BM_WholeGraphExecUpdate(benchmark::State& state, int node_count) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  hipFunction_t function = nullptr;
  if (!CheckHip(state, GetRocmHostOverheadBenchmarkKernel(1, &function),
                "GetRocmHostOverheadBenchmarkKernel")) {
    return;
  }

  GraphData first;
  GraphData second;
  first.argument = 0;
  second.argument = 1;
  if (!CreateGraph(state, function, node_count, true, first) ||
      !CreateGraph(state, function, node_count, false, second)) {
    DestroyGraph(second);
    DestroyGraph(first);
    return;
  }

  double elapsed_seconds = 0;
  int64_t updates = 0;
  for (auto _ : state) {
    GraphData& update = updates & 1 ? first : second;
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
    auto start = Clock::now();
    hipError_t result = hipGraphExecUpdate(first.exec, update.graph, &error_node,
                                           &update_result);
    auto end = Clock::now();
    if (!CheckHip(state, result, "hipGraphExecUpdate")) break;
    if (update_result != hipGraphExecUpdateSuccess) {
      state.SkipWithError("hipGraphExecUpdate did not accept matching graph");
      break;
    }
    elapsed_seconds += Seconds(end - start);
    ++updates;
    state.SetIterationTime(Seconds(end - start));
  }

  state.counters["ns/graph"] = elapsed_seconds * 1e9 / updates;
  state.counters["ns/node"] = elapsed_seconds * 1e9 / updates / node_count;
  state.SetItemsProcessed(updates * node_count);
  DestroyGraph(second);
  DestroyGraph(first);
}

void BM_GraphReplay(benchmark::State& state, int node_count, bool upload,
                    bool first_replay) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  hipFunction_t function = nullptr;
  if (!CheckHip(state, GetRocmHostOverheadBenchmarkKernel(1, &function),
                "GetRocmHostOverheadBenchmarkKernel")) {
    return;
  }

  hipStream_t stream = nullptr;
  if (!CheckHip(state, hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                "hipStreamCreateWithFlags")) {
    return;
  }

  double enqueue_seconds = 0;
  double sync_seconds = 0;
  double upload_enqueue_seconds = 0;
  double upload_sync_seconds = 0;
  int64_t launches = 0;

  GraphData steady_graph;
  if (!first_replay &&
      !CreateGraph(state, function, node_count, true, steady_graph)) {
    (void)hipStreamDestroy(stream);
    return;
  }

  if (!first_replay) {
    if (upload) {
      auto upload_start = Clock::now();
      if (!CheckHip(state, hipGraphUpload(steady_graph.exec, stream),
                    "hipGraphUpload")) {
        DestroyGraph(steady_graph);
        (void)hipStreamDestroy(stream);
        return;
      }
      auto upload_enqueue_end = Clock::now();
      if (!CheckHip(state, hipStreamSynchronize(stream),
                    "hipGraphUpload synchronization")) {
        DestroyGraph(steady_graph);
        (void)hipStreamDestroy(stream);
        return;
      }
      auto upload_end = Clock::now();
      upload_enqueue_seconds = Seconds(upload_enqueue_end - upload_start);
      upload_sync_seconds = Seconds(upload_end - upload_enqueue_end);
    }
    if (!CheckHip(state, hipGraphLaunch(steady_graph.exec, stream),
                  "warm-up hipGraphLaunch") ||
        !CheckHip(state, hipStreamSynchronize(stream),
                  "warm-up hipStreamSynchronize")) {
      DestroyGraph(steady_graph);
      (void)hipStreamDestroy(stream);
      return;
    }
  }

  const int batch_size = first_replay ? 1 : 64;
  for (auto _ : state) {
    GraphData first_graph;
    GraphData* graph = &steady_graph;
    if (first_replay) {
      if (!CreateGraph(state, function, node_count, true, first_graph)) break;
      graph = &first_graph;
      if (upload) {
        auto upload_start = Clock::now();
        if (!CheckHip(state, hipGraphUpload(graph->exec, stream),
                      "hipGraphUpload")) {
          DestroyGraph(first_graph);
          break;
        }
        auto upload_enqueue_end = Clock::now();
        if (!CheckHip(state, hipStreamSynchronize(stream),
                      "hipGraphUpload synchronization")) {
          DestroyGraph(first_graph);
          break;
        }
        auto upload_end = Clock::now();
        upload_enqueue_seconds += Seconds(upload_enqueue_end - upload_start);
        upload_sync_seconds += Seconds(upload_end - upload_enqueue_end);
      }
    }

    auto enqueue_start = Clock::now();
    for (int i = 0; i < batch_size; ++i) {
      if (!CheckHip(state, hipGraphLaunch(graph->exec, stream),
                    "hipGraphLaunch")) {
        break;
      }
    }
    auto enqueue_end = Clock::now();
    if (!CheckHip(state, hipStreamSynchronize(stream),
                  "hipStreamSynchronize")) {
      if (first_replay) DestroyGraph(first_graph);
      break;
    }
    auto sync_end = Clock::now();
    enqueue_seconds += Seconds(enqueue_end - enqueue_start);
    sync_seconds += Seconds(sync_end - enqueue_end);
    launches += batch_size;
    state.SetIterationTime(Seconds(sync_end - enqueue_start));
    if (first_replay) DestroyGraph(first_graph);
  }

  SetBatchCounters(state, enqueue_seconds, sync_seconds, launches);
  if (upload) {
    int64_t upload_count = first_replay ? state.iterations() : 1;
    state.counters["upload_enqueue_us"] =
        upload_enqueue_seconds * 1e6 / upload_count;
    state.counters["upload_sync_us"] =
        upload_sync_seconds * 1e6 / upload_count;
  }
  DestroyGraph(steady_graph);
  (void)hipStreamDestroy(stream);
}

void BM_HipMallocFree(benchmark::State& state, size_t allocation_size) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  void* warmup = nullptr;
  if (!CheckHip(state, hipMalloc(&warmup, allocation_size),
                "warm-up hipMalloc") ||
      !CheckHip(state, hipFree(warmup), "warm-up hipFree")) {
    return;
  }

  constexpr int kAllocationBatchSize = 64;
  double elapsed_seconds = 0;
  int64_t operations = 0;
  for (auto _ : state) {
    auto start = Clock::now();
    for (int i = 0; i < kAllocationBatchSize; ++i) {
      void* pointer = nullptr;
      if (!CheckHip(state, hipMalloc(&pointer, allocation_size), "hipMalloc") ||
          !CheckHip(state, hipFree(pointer), "hipFree")) {
        break;
      }
    }
    auto end = Clock::now();
    elapsed_seconds += Seconds(end - start);
    operations += kAllocationBatchSize;
    state.SetIterationTime(Seconds(end - start));
  }

  state.counters["ns/alloc_free"] = elapsed_seconds * 1e9 / operations;
  state.SetItemsProcessed(operations);
}

void BM_HipMallocAsyncFree(benchmark::State& state, size_t allocation_size) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  hipStream_t stream = nullptr;
  if (!CheckHip(state, hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                "hipStreamCreateWithFlags")) {
    return;
  }

  void* warmup = nullptr;
  if (!CheckHip(state, hipMallocAsync(&warmup, allocation_size, stream),
                "warm-up hipMallocAsync") ||
      !CheckHip(state, hipFreeAsync(warmup, stream),
                "warm-up hipFreeAsync") ||
      !CheckHip(state, hipStreamSynchronize(stream),
                "warm-up hipStreamSynchronize")) {
    (void)hipStreamDestroy(stream);
    return;
  }

  constexpr int kAllocationBatchSize = 256;
  std::array<void*, kAllocationBatchSize> pointers{};
  double enqueue_seconds = 0;
  double sync_seconds = 0;
  int64_t operations = 0;
  for (auto _ : state) {
    auto enqueue_start = Clock::now();
    for (void*& pointer : pointers) {
      if (!CheckHip(state,
                    hipMallocAsync(&pointer, allocation_size, stream),
                    "hipMallocAsync")) {
        break;
      }
    }
    for (void* pointer : pointers) {
      if (!CheckHip(state, hipFreeAsync(pointer, stream), "hipFreeAsync")) {
        break;
      }
    }
    auto enqueue_end = Clock::now();
    if (!CheckHip(state, hipStreamSynchronize(stream),
                  "hipStreamSynchronize")) {
      break;
    }
    auto sync_end = Clock::now();
    enqueue_seconds += Seconds(enqueue_end - enqueue_start);
    sync_seconds += Seconds(sync_end - enqueue_end);
    operations += kAllocationBatchSize;
    state.SetIterationTime(Seconds(sync_end - enqueue_start));
  }

  SetBatchCounters(state, enqueue_seconds, sync_seconds, operations);
  (void)hipStreamDestroy(stream);
}

void BM_BfcReuse(benchmark::State& state, size_t allocation_size) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  tsl::BFCAllocator::Options options;
  options.allow_growth = true;
  tsl::BFCAllocator allocator(std::make_unique<HipSubAllocator>(),
                              /*total_memory=*/64ULL << 20,
                              "rocm_host_overhead_benchmark", options);

  void* warmup = allocator.AllocateRaw(/*alignment=*/256, allocation_size);
  if (warmup == nullptr) {
    state.SkipWithError("BFC warm-up allocation failed");
    return;
  }
  allocator.DeallocateRaw(warmup);

  constexpr int kAllocationBatchSize = 16384;
  double elapsed_seconds = 0;
  int64_t operations = 0;
  for (auto _ : state) {
    auto start = Clock::now();
    for (int i = 0; i < kAllocationBatchSize; ++i) {
      void* pointer =
          allocator.AllocateRaw(/*alignment=*/256, allocation_size);
      benchmark::DoNotOptimize(pointer);
      allocator.DeallocateRaw(pointer);
    }
    auto end = Clock::now();
    elapsed_seconds += Seconds(end - start);
    operations += kAllocationBatchSize;
    state.SetIterationTime(Seconds(end - start));
  }

  state.counters["ns/alloc_free"] = elapsed_seconds * 1e9 / operations;
  state.SetItemsProcessed(operations);
}

void BM_StreamWriteValue64(benchmark::State& state, int logical_teardowns,
                           bool coalesced) {
  if (!CheckHip(state, hipSetDevice(0), "hipSetDevice")) return;

  hipStream_t stream = nullptr;
  uint64_t* marker = nullptr;
  if (!CheckHip(state, hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                "hipStreamCreateWithFlags") ||
      !CheckHip(state, hipMalloc(reinterpret_cast<void**>(&marker),
                                sizeof(uint64_t)),
                "hipMalloc(marker)")) {
    if (stream != nullptr) (void)hipStreamDestroy(stream);
    return;
  }

  if (!CheckHip(state,
                hipStreamWriteValue64(stream, marker, 0,
                                      hipStreamWriteValueDefault),
                "warm-up hipStreamWriteValue64") ||
      !CheckHip(state, hipStreamSynchronize(stream),
                "warm-up hipStreamSynchronize")) {
    (void)hipFree(marker);
    (void)hipStreamDestroy(stream);
    return;
  }

  double enqueue_seconds = 0;
  double sync_seconds = 0;
  int64_t operations = 0;
  uint64_t value = 0;
  for (auto _ : state) {
    auto enqueue_start = Clock::now();
    int marker_count = coalesced ? 1 : logical_teardowns;
    for (int i = 0; i < marker_count; ++i) {
      if (!CheckHip(state,
                    hipStreamWriteValue64(stream, marker, ++value,
                                          hipStreamWriteValueDefault),
                    "hipStreamWriteValue64")) {
        break;
      }
    }
    auto enqueue_end = Clock::now();
    if (!CheckHip(state, hipStreamSynchronize(stream),
                  "hipStreamSynchronize")) {
      break;
    }
    auto sync_end = Clock::now();
    enqueue_seconds += Seconds(enqueue_end - enqueue_start);
    sync_seconds += Seconds(sync_end - enqueue_end);
    operations += logical_teardowns;
    state.SetIterationTime(Seconds(sync_end - enqueue_start));
  }

  SetBatchCounters(state, enqueue_seconds, sync_seconds, operations);
  state.counters["marker_calls/batch"] = coalesced ? 1 : logical_teardowns;
  (void)hipFree(marker);
  (void)hipStreamDestroy(stream);
}

#define REGISTER_LAUNCH(ARITY, ARG_PATH, PACKED, STREAM_NAME, STREAM_FLAGS) \
  BENCHMARK_CAPTURE(BM_ModuleLaunchKernel,                              \
                    args##ARITY##_##ARG_PATH##_##STREAM_NAME, ARITY,    \
                    PACKED, STREAM_FLAGS)                              \
      ->UseManualTime()

#define REGISTER_LAUNCH_ARITY(ARITY)                                  \
  REGISTER_LAUNCH(ARITY, pointers, false, default_stream,              \
                  hipStreamDefault);                                  \
  REGISTER_LAUNCH(ARITY, packed, true, default_stream,                 \
                  hipStreamDefault);                                  \
  REGISTER_LAUNCH(ARITY, pointers, false, nonblocking_stream,          \
                  hipStreamNonBlocking);                              \
  REGISTER_LAUNCH(ARITY, packed, true, nonblocking_stream,             \
                  hipStreamNonBlocking)

REGISTER_LAUNCH_ARITY(0);
REGISTER_LAUNCH_ARITY(1);
REGISTER_LAUNCH_ARITY(4);
REGISTER_LAUNCH_ARITY(16);
REGISTER_LAUNCH_ARITY(64);

BENCHMARK_CAPTURE(BM_EventRecordWait, system_scope,
                  EventVisibility::kSystem)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_EventRecordWait, device_scope,
                  EventVisibility::kDevice)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_EventRecordWait, timing,
                  EventVisibility::kTiming)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_EventRecordWait, timing_no_system_fence,
                  EventVisibility::kTimingNoSystemFence)
    ->UseManualTime();

BENCHMARK_CAPTURE(BM_RocblasSetStream, same_stream, StatePattern::kSame)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_RocblasSetStream, alternating_streams,
                  StatePattern::kAlternate)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_RocblasSetStream, cached_same_stream,
                  StatePattern::kCachedSame)
    ->UseManualTime();

BENCHMARK_CAPTURE(BM_RocblasSetAtomicsMode, same_mode, StatePattern::kSame)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_RocblasSetAtomicsMode, alternating_modes,
                  StatePattern::kAlternate)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_RocblasSetAtomicsMode, cached_same_mode,
                  StatePattern::kCachedSame)
    ->UseManualTime();

#define REGISTER_GRAPH_SIZE(NODES)                                  \
  BENCHMARK_CAPTURE(BM_GraphNodeUpdates, nodes_##NODES, NODES)       \
      ->UseManualTime();                                             \
  BENCHMARK_CAPTURE(BM_WholeGraphExecUpdate, nodes_##NODES, NODES)   \
      ->UseManualTime()

REGISTER_GRAPH_SIZE(1);
REGISTER_GRAPH_SIZE(8);
REGISTER_GRAPH_SIZE(32);
REGISTER_GRAPH_SIZE(128);
REGISTER_GRAPH_SIZE(512);

BENCHMARK_CAPTURE(BM_GraphReplay, first_no_upload, 128, false, true)
    ->UseManualTime()
    ->Iterations(20);
BENCHMARK_CAPTURE(BM_GraphReplay, first_after_upload, 128, true, true)
    ->UseManualTime()
    ->Iterations(20);
BENCHMARK_CAPTURE(BM_GraphReplay, steady_no_upload, 128, false, false)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_GraphReplay, steady_after_upload, 128, true, false)
    ->UseManualTime();

#define REGISTER_ALLOCATION_SIZE(NAME, SIZE)                     \
  BENCHMARK_CAPTURE(BM_HipMallocFree, NAME, SIZE)->UseManualTime(); \
  BENCHMARK_CAPTURE(BM_HipMallocAsyncFree, NAME, SIZE)              \
      ->UseManualTime();                                             \
  BENCHMARK_CAPTURE(BM_BfcReuse, NAME, SIZE)->UseManualTime()

REGISTER_ALLOCATION_SIZE(bytes_4k, 4 << 10);
REGISTER_ALLOCATION_SIZE(bytes_1m, 1 << 20);

BENCHMARK_CAPTURE(BM_StreamWriteValue64, markers_128, 128, false)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_StreamWriteValue64, coalesced_128, 128, true)
    ->UseManualTime();

#undef REGISTER_ALLOCATION_SIZE
#undef REGISTER_GRAPH_SIZE
#undef REGISTER_LAUNCH_ARITY
#undef REGISTER_LAUNCH

}  // namespace
}  // namespace stream_executor::gpu
