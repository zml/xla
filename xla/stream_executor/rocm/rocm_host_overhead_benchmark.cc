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
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rocblas/rocblas.h"
#include "xla/stream_executor/rocm/rocm_host_overhead_benchmark_kernels.h"

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

#undef REGISTER_LAUNCH_ARITY
#undef REGISTER_LAUNCH

}  // namespace
}  // namespace stream_executor::gpu
