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

#include <string>

#include "gtest/gtest.h"

namespace stream_executor::gpu {
namespace {

TEST(RocmPerformanceCountersTest, SnapshotsAndFormatsNonzeroCounters) {
  ResetRocmPerformanceCountersForTest();
  RocmPerformanceCounterSnapshot begin =
      GetRocmPerformanceCounterSnapshot();
  if (kRocmPerformanceCountersEnabled) {
    IncrementRocmPerformanceCounter(RocmPerformanceCounter::kKernelLaunch, 7);
  } else {
    IncrementRocmPerformanceCounterForTest(
        RocmPerformanceCounter::kKernelLaunch, 7);
  }
  IncrementRocmPerformanceCounterForTest(
      RocmPerformanceCounter::kGraphNodeUpdateKernel, 3);

  RocmPerformanceCounterSnapshot delta =
      SubtractRocmPerformanceCounterSnapshots(
          GetRocmPerformanceCounterSnapshot(), begin);
  EXPECT_EQ(delta[static_cast<size_t>(RocmPerformanceCounter::kKernelLaunch)],
            7);
  EXPECT_EQ(delta[static_cast<size_t>(
                RocmPerformanceCounter::kGraphNodeUpdateKernel)],
            3);
  EXPECT_EQ(FormatRocmPerformanceCounterSnapshot(delta),
            "{\"kernel_launch\":7,\"graph_node_update_kernel\":3}");
}

}  // namespace
}  // namespace stream_executor::gpu
