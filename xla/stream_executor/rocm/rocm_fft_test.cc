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

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_performance_counters.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

uint64_t Counter(RocmPerformanceCounter counter) {
  return GetRocmPerformanceCounterSnapshot()[static_cast<size_t>(counter)];
}

class StableScratchAllocator : public ScratchAllocator {
 public:
  explicit StableScratchAllocator(StreamExecutor* executor)
      : executor_(executor) {}

  ~StableScratchAllocator() override {
    if (storage_ != nullptr) executor_->Deallocate(&storage_);
  }

  int64_t GetMemoryLimitInBytes() override { return -1; }

  absl::StatusOr<DeviceAddress<uint8_t>> AllocateBytes(
      int64_t byte_size) override {
    if (byte_size == 0) return DeviceAddress<uint8_t>();
    if (storage_ == nullptr) storage_ = executor_->Allocate(byte_size, 0);
    if (storage_ == nullptr || storage_.size() < byte_size) {
      return absl::ResourceExhaustedError("failed to allocate FFT scratch");
    }
    return DeviceAddress<uint8_t>(storage_);
  }

 private:
  StreamExecutor* executor_;
  DeviceAddressBase storage_;
};

class RocmFftTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        Platform * platform,
        PlatformManager::PlatformWithId(rocm::kROCmPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    fft_ = executor_->AsFft();
    ASSERT_NE(fft_, nullptr);
  }

  std::unique_ptr<fft::Plan> CreatePlan(Stream* stream,
                                        ScratchAllocator* scratch = nullptr) {
    uint64_t element_count[] = {8};
    return fft_->CreateBatchedPlanWithScratchAllocator(
        stream, /*rank=*/1, element_count, /*input_embed=*/nullptr,
        /*input_stride=*/0, /*input_distance=*/0,
        /*output_embed=*/nullptr, /*output_stride=*/0,
        /*output_distance=*/0, fft::Type::kC2CForward,
        /*in_place_fft=*/false, /*batch_count=*/1, scratch);
  }

  bool Execute(Stream* stream, fft::Plan* plan,
               DeviceAddress<std::complex<float>>* input,
               DeviceAddress<std::complex<float>>* output) {
    return fft_->DoFft(stream, plan, *input, output);
  }

  StreamExecutor* executor_ = nullptr;
  fft::FftSupport* fft_ = nullptr;
};

TEST_F(RocmFftTest, CachesStreamAndRebindsAfterRawHandleReuse) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor_->CreateStream(std::nullopt));
  std::unique_ptr<fft::Plan> plan = CreatePlan(stream.get());
  ASSERT_NE(plan, nullptr);

  auto input = executor_->AllocateArray<std::complex<float>>(8, 0);
  auto output = executor_->AllocateArray<std::complex<float>>(8, 0);
  ASSERT_NE(input.opaque(), nullptr);
  ASSERT_NE(output.opaque(), nullptr);

  ResetRocmPerformanceCountersForTest();
  EXPECT_TRUE(Execute(stream.get(), plan.get(), &input, &output));
  EXPECT_TRUE(Execute(stream.get(), plan.get(), &input, &output));
  EXPECT_EQ(Counter(RocmPerformanceCounter::kFftSetStream), 1);
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  void* first_handle = stream->platform_specific_handle().stream;
  stream.reset();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> reused_stream,
                          executor_->CreateStream(std::nullopt));
  EXPECT_EQ(reused_stream->platform_specific_handle().stream, first_handle);

  ResetRocmPerformanceCountersForTest();
  EXPECT_TRUE(Execute(reused_stream.get(), plan.get(), &input, &output));
  EXPECT_EQ(Counter(RocmPerformanceCounter::kFftSetStream), 1);
  EXPECT_THAT(reused_stream->BlockHostUntilDone(), absl_testing::IsOk());

  executor_->Deallocate(&input);
  executor_->Deallocate(&output);
}

TEST_F(RocmFftTest, SkipsUnchangedWorkArea) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor_->CreateStream(std::nullopt));
  StableScratchAllocator scratch(executor_);

  ResetRocmPerformanceCountersForTest();
  std::unique_ptr<fft::Plan> plan = CreatePlan(stream.get(), &scratch);
  ASSERT_NE(plan, nullptr);
  EXPECT_EQ(Counter(RocmPerformanceCounter::kFftSetWorkArea), 1);

  fft_->UpdatePlanWithScratchAllocator(stream.get(), plan.get(), &scratch);
  fft_->UpdatePlanWithScratchAllocator(stream.get(), plan.get(), &scratch);
  EXPECT_EQ(Counter(RocmPerformanceCounter::kFftSetWorkArea), 1);
}

}  // namespace
}  // namespace stream_executor::gpu
