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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_performance_counters.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

uint64_t RocblasSetStreamCalls() {
  return GetRocmPerformanceCounterSnapshot()[static_cast<size_t>(
      RocmPerformanceCounter::kRocblasSetStream)];
}

bool StreamCacheDisabled() {
  const char* value = std::getenv("XLA_ROCM_DISABLE_ROCBLAS_STREAM_CACHE");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

class RocmBlasTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        Platform * platform,
        PlatformManager::PlatformWithId(rocm::kROCmPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    blas_ = executor_->AsBlas();
    ASSERT_NE(blas_, nullptr);
  }

  bool Scale(Stream* stream, DeviceAddress<float>* values) {
    return blas_->DoBlasScal(stream, /*elem_count=*/4, /*alpha=*/2.0f, values,
                             /*incx=*/1);
  }

  StreamExecutor* executor_ = nullptr;
  blas::BlasSupport* blas_ = nullptr;
};

TEST_F(RocmBlasTest, CachesSameStreamAndRebindsOnSwitch) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream_a,
                          executor_->CreateStream(std::nullopt));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream_b,
                          executor_->CreateStream(std::nullopt));
  DeviceAddress<float> values = executor_->AllocateArray<float>(4, 0);
  ASSERT_NE(values.opaque(), nullptr);

  ResetRocmPerformanceCountersForTest();
  EXPECT_TRUE(Scale(stream_a.get(), &values));
  EXPECT_TRUE(Scale(stream_a.get(), &values));
  EXPECT_EQ(RocblasSetStreamCalls(), StreamCacheDisabled() ? 4 : 1);

  EXPECT_TRUE(Scale(stream_b.get(), &values));
  EXPECT_TRUE(Scale(stream_a.get(), &values));
  EXPECT_EQ(RocblasSetStreamCalls(), StreamCacheDisabled() ? 8 : 3);

  EXPECT_THAT(stream_a->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(stream_b->BlockHostUntilDone(), absl_testing::IsOk());
  executor_->Deallocate(&values);
}

TEST_F(RocmBlasTest, RebindsAfterStreamWrapperLifetimeEnds) {
  DeviceAddress<float> values = executor_->AllocateArray<float>(4, 0);
  ASSERT_NE(values.opaque(), nullptr);
  void* first_handle = nullptr;

  ResetRocmPerformanceCountersForTest();
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                            executor_->CreateStream(std::nullopt));
    first_handle = stream->platform_specific_handle().stream;
    EXPECT_TRUE(Scale(stream.get(), &values));
    EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> reused_stream,
                          executor_->CreateStream(std::nullopt));
  EXPECT_EQ(reused_stream->platform_specific_handle().stream, first_handle);
  EXPECT_TRUE(Scale(reused_stream.get(), &values));
  EXPECT_EQ(RocblasSetStreamCalls(), StreamCacheDisabled() ? 4 : 2);

  EXPECT_THAT(reused_stream->BlockHostUntilDone(), absl_testing::IsOk());
  executor_->Deallocate(&values);
}

}  // namespace
}  // namespace stream_executor::gpu
