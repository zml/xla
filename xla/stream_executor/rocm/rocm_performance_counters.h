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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_PERFORMANCE_COUNTERS_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_PERFORMANCE_COUNTERS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/optimization.h"
#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

// Host API calls that are useful when deciding whether a ROCm performance
// optimization is exercised by a workload. Keep this list at API granularity:
// it intentionally counts attempted calls, including calls that return an
// error, rather than higher-level operations that may avoid the API entirely.
enum class RocmPerformanceCounter : size_t {
  kKernelLaunch,
  kGraphCreate,
  kGraphInstantiate,
  kGraphUpload,
  kGraphLaunch,
  kGraphTrace,
  kGraphNodeCreateKernel,
  kGraphNodeCreateMemcpy,
  kGraphNodeCreateMemset,
  kGraphNodeCreateChild,
  kGraphNodeCreateEmpty,
  kGraphNodeUpdateKernel,
  kGraphNodeUpdateMemcpy,
  kGraphNodeUpdateMemset,
  kGraphNodeUpdateChild,
  kEventCreateTiming,
  kEventCreateSystem,
  kEventRecord,
  kEventWait,
  kEventSynchronize,
  kRocblasSetStream,
  kRocblasSetAtomicsMode,
  kMiopenSetStream,
  kFftSetStream,
  kFftSetWorkArea,
  kVmmMap,
  kVmmUnmap,
  kVmmTimelineWrite,
  kPeerAccessQuery,
  kDeviceCountQuery,
  kStreamPriorityRangeQuery,
  kDevicePropertiesQuery,
  kCount,
};

inline constexpr size_t kRocmPerformanceCounterCount =
    static_cast<size_t>(RocmPerformanceCounter::kCount);
using RocmPerformanceCounterSnapshot =
    std::array<uint64_t, kRocmPerformanceCounterCount>;

// Initialized once from XLA_ROCM_PERFORMANCE_COUNTERS when the ROCm plugin is
// loaded. The hot path is a single predictable-not-taken load and branch when
// diagnostics are disabled. Values accepted as true are 1, true, t, yes, and y
// (case-insensitive).
extern const bool kRocmPerformanceCountersEnabled;

// Slow path called only when counters are enabled.
void IncrementRocmPerformanceCounterSlow(RocmPerformanceCounter counter,
                                         uint64_t amount);

inline void IncrementRocmPerformanceCounter(
    RocmPerformanceCounter counter, uint64_t amount = 1) {
  if (ABSL_PREDICT_FALSE(kRocmPerformanceCountersEnabled)) {
    IncrementRocmPerformanceCounterSlow(counter, amount);
  }
}

RocmPerformanceCounterSnapshot GetRocmPerformanceCounterSnapshot();
RocmPerformanceCounterSnapshot SubtractRocmPerformanceCounterSnapshots(
    const RocmPerformanceCounterSnapshot& end,
    const RocmPerformanceCounterSnapshot& begin);
std::string FormatRocmPerformanceCounterSnapshot(
    const RocmPerformanceCounterSnapshot& snapshot);
void LogRocmPerformanceCounterSnapshot(absl::string_view label,
                                       const RocmPerformanceCounterSnapshot&);

// Attributes calls made synchronously on the current host thread to one GPU
// executable invocation. Calls made by auxiliary threads remain visible in the
// process totals but are intentionally not guessed into an execution range.
// Nested ranges contribute to both their own report and the parent report.
class ScopedRocmPerformanceCounterRange {
 public:
  explicit ScopedRocmPerformanceCounterRange(absl::string_view label);
  ~ScopedRocmPerformanceCounterRange();

  ScopedRocmPerformanceCounterRange(
      const ScopedRocmPerformanceCounterRange&) = delete;
  ScopedRocmPerformanceCounterRange& operator=(
      const ScopedRocmPerformanceCounterRange&) = delete;

 private:
  bool enabled_ = false;
  std::string label_;
  RocmPerformanceCounterSnapshot counters_{};
  RocmPerformanceCounterSnapshot* parent_ = nullptr;
};

// Test-only helpers deliberately bypass the environment-variable gate.
void ResetRocmPerformanceCountersForTest();
void IncrementRocmPerformanceCounterForTest(RocmPerformanceCounter counter,
                                            uint64_t amount = 1);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_PERFORMANCE_COUNTERS_H_
