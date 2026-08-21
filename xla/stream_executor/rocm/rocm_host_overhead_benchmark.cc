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

#undef REGISTER_LAUNCH_ARITY
#undef REGISTER_LAUNCH

}  // namespace
}  // namespace stream_executor::gpu
