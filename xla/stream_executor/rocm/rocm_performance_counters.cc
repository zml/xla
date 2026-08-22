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

#include "xla/stream_executor/rocm/rocm_performance_counters.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace stream_executor::gpu {
namespace {

using AtomicCounters =
    std::array<std::atomic<uint64_t>, kRocmPerformanceCounterCount>;

thread_local RocmPerformanceCounterSnapshot* current_execution_counters =
    nullptr;

AtomicCounters& Counters() {
  static auto* counters = new AtomicCounters{};
  return *counters;
}

struct ExecutableCounters {
  std::mutex mutex;
  std::unordered_map<std::string, RocmPerformanceCounterSnapshot> counters;
};

ExecutableCounters& PerExecutableCounters() {
  static auto* counters = new ExecutableCounters;
  return *counters;
}

constexpr std::array<absl::string_view, kRocmPerformanceCounterCount>
    kCounterNames = {
        "kernel_launch",
        "graph_create",
        "graph_instantiate",
        "graph_upload",
        "graph_launch",
        "graph_trace",
        "graph_node_create_kernel",
        "graph_node_create_memcpy",
        "graph_node_create_memset",
        "graph_node_create_child",
        "graph_node_create_empty",
        "graph_node_update_kernel",
        "graph_node_update_memcpy",
        "graph_node_update_memset",
        "graph_node_update_child",
        "event_create_timing",
        "event_create_system",
        "event_record",
        "event_wait",
        "event_synchronize",
        "rocblas_set_stream",
        "miopen_set_stream",
        "fft_set_stream",
        "fft_set_work_area",
        "vmm_map",
        "vmm_unmap",
        "vmm_timeline_write",
        "peer_access_query",
        "device_count_query",
        "stream_priority_range_query",
        "device_properties_query",
    };

bool ReadEnabledFromEnvironment() {
  const char* value = std::getenv("XLA_ROCM_PERFORMANCE_COUNTERS");
  if (value == nullptr) return false;
  std::string normalized = absl::AsciiStrToLower(value);
  return normalized == "1" || normalized == "true" || normalized == "t" ||
         normalized == "yes" || normalized == "y";
}

void LogTotalsAtExit() {
  // The logging runtime can already be partially torn down when a plugin's
  // atexit handler runs. Write directly to stderr so the final diagnostic is
  // not lost because of global-destruction order.
  std::string formatted = FormatRocmPerformanceCounterSnapshot(
      GetRocmPerformanceCounterSnapshot());
  std::fprintf(stderr, "XLA_ROCM_PERFORMANCE_COUNTERS process_total %s\n",
               formatted.c_str());

  std::vector<std::pair<std::string, RocmPerformanceCounterSnapshot>>
      per_executable;
  {
    ExecutableCounters& state = PerExecutableCounters();
    std::lock_guard<std::mutex> lock(state.mutex);
    per_executable.reserve(state.counters.size());
    for (const auto& entry : state.counters) per_executable.push_back(entry);
  }
  std::sort(per_executable.begin(), per_executable.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.first < rhs.first;
            });
  for (const auto& [label, counters] : per_executable) {
    formatted = FormatRocmPerformanceCounterSnapshot(counters);
    std::fprintf(stderr, "XLA_ROCM_PERFORMANCE_COUNTERS executable=%s %s\n",
                 label.c_str(), formatted.c_str());
  }
  std::fflush(stderr);
}

void RegisterExitLogger() {
  static absl::once_flag once;
  absl::call_once(once, [] { std::atexit(LogTotalsAtExit); });
}

}  // namespace

const bool kRocmPerformanceCountersEnabled = ReadEnabledFromEnvironment();

void IncrementRocmPerformanceCounterSlow(RocmPerformanceCounter counter,
                                         uint64_t amount) {
  RegisterExitLogger();
  size_t index = static_cast<size_t>(counter);
  Counters()[index].fetch_add(amount, std::memory_order_relaxed);
  if (current_execution_counters != nullptr) {
    (*current_execution_counters)[index] += amount;
  }
}

RocmPerformanceCounterSnapshot GetRocmPerformanceCounterSnapshot() {
  RocmPerformanceCounterSnapshot snapshot;
  for (size_t i = 0; i < snapshot.size(); ++i) {
    snapshot[i] = Counters()[i].load(std::memory_order_relaxed);
  }
  return snapshot;
}

RocmPerformanceCounterSnapshot SubtractRocmPerformanceCounterSnapshots(
    const RocmPerformanceCounterSnapshot& end,
    const RocmPerformanceCounterSnapshot& begin) {
  RocmPerformanceCounterSnapshot delta;
  for (size_t i = 0; i < delta.size(); ++i) delta[i] = end[i] - begin[i];
  return delta;
}

std::string FormatRocmPerformanceCounterSnapshot(
    const RocmPerformanceCounterSnapshot& snapshot) {
  std::string out = "{";
  bool first = true;
  for (size_t i = 0; i < snapshot.size(); ++i) {
    if (snapshot[i] == 0) continue;
    absl::StrAppend(&out, first ? "" : ",", "\"", kCounterNames[i],
                    "\":", snapshot[i]);
    first = false;
  }
  absl::StrAppend(&out, "}");
  return out;
}

void LogRocmPerformanceCounterSnapshot(
    absl::string_view label,
    const RocmPerformanceCounterSnapshot& snapshot) {
  LOG(INFO) << "XLA_ROCM_PERFORMANCE_COUNTERS " << label << " "
            << FormatRocmPerformanceCounterSnapshot(snapshot);
}

ScopedRocmPerformanceCounterRange::ScopedRocmPerformanceCounterRange(
    absl::string_view label)
    : enabled_(kRocmPerformanceCountersEnabled) {
  if (!enabled_) return;
  label_ = label;
  parent_ = current_execution_counters;
  current_execution_counters = &counters_;
}

ScopedRocmPerformanceCounterRange::~ScopedRocmPerformanceCounterRange() {
  if (!enabled_) return;
  current_execution_counters = parent_;

  bool any = false;
  for (size_t i = 0; i < counters_.size(); ++i) {
    if (parent_ != nullptr) (*parent_)[i] += counters_[i];
    any |= counters_[i] != 0;
  }
  if (!any) return;

  ExecutableCounters& state = PerExecutableCounters();
  std::lock_guard<std::mutex> lock(state.mutex);
  RocmPerformanceCounterSnapshot& total = state.counters[label_];
  for (size_t i = 0; i < total.size(); ++i) total[i] += counters_[i];
}

void ResetRocmPerformanceCountersForTest() {
  for (auto& counter : Counters()) {
    counter.store(0, std::memory_order_relaxed);
  }
}

void IncrementRocmPerformanceCounterForTest(RocmPerformanceCounter counter,
                                            uint64_t amount) {
  Counters()[static_cast<size_t>(counter)].fetch_add(amount,
                                                      std::memory_order_relaxed);
}

}  // namespace stream_executor::gpu
