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

#include "xla/stream_executor/device_address_vmm_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace {

using ::absl_testing::StatusIs;
using ::testing::NiceMock;
using ::testing::Return;

constexpr uint64_t kGranularity = 64;

uint64_t RoundUpTestSize(uint64_t size) {
  return ((size + kGranularity - 1) / kGranularity) * kGranularity;
}

class TestMemoryAllocation final : public MemoryAllocation {
 public:
  explicit TestMemoryAllocation(uint64_t size)
      : storage_(std::make_unique<uint8_t[]>(size)), size_(size) {}

  DeviceAddressBase address() const override {
    return DeviceAddressBase(storage_.get(), size_);
  }

 private:
  std::unique_ptr<uint8_t[]> storage_;
  uint64_t size_;
};

class TestMemoryReservation final : public MemoryReservation {
 public:
  explicit TestMemoryReservation(uint64_t size)
      : storage_(std::make_unique<uint8_t[]>(size)), size_(size) {}

  DeviceAddressBase address() const override {
    return DeviceAddressBase(storage_.get(), size_);
  }

  int active_mapping_count() const { return active_mapping_count_; }

 private:
  absl::Status Map(size_t reservation_offset, size_t allocation_offset,
                   size_t size, MemoryAllocation& allocation) override {
    if (reservation_offset > size_ || size > size_ - reservation_offset ||
        allocation_offset > allocation.address().size() ||
        size > allocation.address().size() - allocation_offset) {
      return absl::InvalidArgumentError("mapping range is out of bounds");
    }
    ++active_mapping_count_;
    return absl::OkStatus();
  }

  absl::Status SetAccess(uint64_t /*reservation_offset*/,
                         size_t /*size*/) override {
    return absl::OkStatus();
  }

  absl::Status UnMap(size_t /*reservation_offset*/, size_t /*size*/) override {
    if (active_mapping_count_ == 0) {
      return absl::FailedPreconditionError("reservation is not mapped");
    }
    --active_mapping_count_;
    return absl::OkStatus();
  }

  std::unique_ptr<uint8_t[]> storage_;
  uint64_t size_;
  int active_mapping_count_ = 0;
};

class TestDeviceAddressVmmAllocator final : public DeviceAddressVmmAllocator {
 public:
  static absl::StatusOr<std::unique_ptr<TestDeviceAddressVmmAllocator>> Create(
      const Platform* platform, absl::Span<const DeviceConfig> devices,
      uint64_t physical_size_padding = 0) {
    auto allocator = std::unique_ptr<TestDeviceAddressVmmAllocator>(
        new TestDeviceAddressVmmAllocator(platform, physical_size_padding));
    absl::Status status = PopulateDevices(allocator.get(), devices);
    if (!status.ok()) {
      return status;
    }
    return allocator;
  }

  int allocation_count() const { return allocation_count_; }

 protected:
  absl::Status InitializeDeviceState(PerDeviceState& state) override {
    state.allocation_granularity = kGranularity;
    auto* timeline = new uint64_t(0);
    state.pinned_timeline = timeline;
    state.destroy_fn = [timeline] { delete timeline; };
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> CreateAllocation(
      StreamExecutor* /*executor*/, uint64_t size) override {
    ++allocation_count_;
    return std::make_unique<TestMemoryAllocation>(RoundUpTestSize(size) +
                                                  physical_size_padding_);
  }

  absl::StatusOr<std::unique_ptr<MemoryReservation>> CreateReservation(
      StreamExecutor* /*executor*/, uint64_t size) override {
    return std::make_unique<TestMemoryReservation>(RoundUpTestSize(size) +
                                                   physical_size_padding_);
  }

  absl::Status EnqueueDeferredDeallocation(PerDeviceState& state,
                                           uint64_t seqno) override {
    __atomic_store_n(state.pinned_timeline, seqno, __ATOMIC_RELEASE);
    return absl::OkStatus();
  }

 private:
  TestDeviceAddressVmmAllocator(const Platform* platform,
                                uint64_t physical_size_padding)
      : DeviceAddressVmmAllocator(platform),
        physical_size_padding_(physical_size_padding) {}

  uint64_t physical_size_padding_;
  int allocation_count_ = 0;
};

class TestDelegatingAllocator final : public DeviceAddressAllocator {
 public:
  TestDelegatingAllocator(const Platform* platform, Stream* stream)
      : DeviceAddressAllocator(platform), stream_(stream) {}

  absl::StatusOr<ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool /*retry_on_failure*/,
      int64_t memory_space) override {
    ++allocation_count_;
    allocated_memory_spaces_.push_back(memory_space);
    auto* storage = new uint8_t[size];
    return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(storage, size),
                                        device_ordinal, this);
  }

  absl::Status Deallocate(int /*device_ordinal*/,
                          DeviceAddressBase mem) override {
    ++deallocation_count_;
    delete[] static_cast<uint8_t*>(mem.opaque());
    return absl::OkStatus();
  }

  absl::StatusOr<Stream*> GetStream(int /*device_ordinal*/) override {
    return stream_;
  }

  int allocation_count() const { return allocation_count_; }
  int deallocation_count() const { return deallocation_count_; }
  const std::vector<int64_t>& allocated_memory_spaces() const {
    return allocated_memory_spaces_;
  }

 private:
  Stream* stream_;
  int allocation_count_ = 0;
  int deallocation_count_ = 0;
  std::vector<int64_t> allocated_memory_spaces_;
};

class DeviceAddressVmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ON_CALL(executor_, device_ordinal()).WillByDefault(Return(0));
    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
  }

  DeviceAddressVmmAllocator::DeviceConfig Config(uint64_t pa_budget) {
    return {&executor_, &stream_, pa_budget};
  }

  NiceMock<MockPlatform> platform_;
  NiceMock<MockStreamExecutor> executor_;
  NiceMock<MockStream> stream_;
};

TEST_F(DeviceAddressVmmAllocatorTest,
       DelegatesSeveralMemorySpacesToOneAllocator) {
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {Config(UINT64_MAX)}));
  auto delegate =
      std::make_unique<TestDelegatingAllocator>(&platform_, &stream_);
  TestDelegatingAllocator* delegate_ptr = delegate.get();
  constexpr int64_t kDelegatedMemorySpaces[] = {0, 1, 5};
  allocator->SetAllocatorForMemorySpaces(kDelegatedMemorySpaces,
                                         std::move(delegate));

  ASSERT_OK_AND_ASSIGN(auto default_buffer,
                       allocator->Allocate(/*device_ordinal=*/0, 16,
                                           /*retry_on_failure=*/true,
                                           /*memory_space=*/0));
  ASSERT_OK_AND_ASSIGN(auto collective_buffer,
                       allocator->Allocate(/*device_ordinal=*/0, 16,
                                           /*retry_on_failure=*/true,
                                           /*memory_space=*/1));
  ASSERT_OK_AND_ASSIGN(auto host_buffer,
                       allocator->Allocate(/*device_ordinal=*/0, 16,
                                           /*retry_on_failure=*/true,
                                           /*memory_space=*/5));
  ASSERT_OK_AND_ASSIGN(auto temp_buffer,
                       allocator->Allocate(/*device_ordinal=*/0, 16,
                                           /*retry_on_failure=*/true,
                                           /*memory_space=*/2));

  EXPECT_EQ(delegate_ptr->allocation_count(), 3);
  EXPECT_THAT(delegate_ptr->allocated_memory_spaces(),
              ::testing::ElementsAre(0, 1, 5));
  EXPECT_EQ(allocator->allocation_count(), 1);

  ASSERT_THAT(default_buffer.Free(), absl_testing::IsOk());
  ASSERT_THAT(collective_buffer.Free(), absl_testing::IsOk());
  ASSERT_THAT(host_buffer.Free(), absl_testing::IsOk());
  ASSERT_THAT(temp_buffer.Free(), absl_testing::IsOk());
  EXPECT_EQ(delegate_ptr->deallocation_count(), 3);
}

TEST_F(DeviceAddressVmmAllocatorTest, RetryFlagDoesNotDisablePendingReclaim) {
  const DeviceAddressVmmAllocator::DeviceConfig config =
      Config(2 * kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {config}));

  ASSERT_OK_AND_ASSIGN(
      auto first,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/true, /*memory_space=*/0));
  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, first.Release()),
              absl_testing::IsOk());

  ASSERT_OK_AND_ASSIGN(
      auto retried,
      allocator->Allocate(/*device_ordinal=*/0, 2 * kGranularity,
                          /*retry_on_failure=*/false, /*memory_space=*/0));
  EXPECT_EQ(allocator->allocation_count(), 2);
}

TEST_F(DeviceAddressVmmAllocatorTest,
       RetryDisabledStillReusesCompatiblePendingAllocation) {
  const DeviceAddressVmmAllocator::DeviceConfig config = Config(kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {config}));

  ASSERT_OK_AND_ASSIGN(
      auto first,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/true, /*memory_space=*/0));
  void* first_address = first->opaque();
  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, first.Release()),
              absl_testing::IsOk());

  ASSERT_OK_AND_ASSIGN(
      auto reused,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/false, /*memory_space=*/0));
  EXPECT_EQ(reused->opaque(), first_address);
  EXPECT_EQ(allocator->allocation_count(), 1);
}

TEST_F(DeviceAddressVmmAllocatorTest,
       RetryFlagDoesNotDisableMappedPendingReclaim) {
  auto reservation = std::make_unique<TestMemoryReservation>(2 * kGranularity);
  const DeviceAddressVmmAllocator::DeviceConfig config =
      Config(2 * kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator, TestDeviceAddressVmmAllocator::Create(
                                           &platform_, {config}));

  ASSERT_OK_AND_ASSIGN(
      auto first,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/true, /*memory_space=*/0));
  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, first.Release()),
              absl_testing::IsOk());

  ASSERT_OK_AND_ASSIGN(
      auto retried,
      allocator->Allocate(
          /*device_ordinal=*/0, /*allocation_size=*/2 * kGranularity,
          /*retry_on_failure=*/false, /*memory_space=*/0, reservation.get(),
          /*reservation_offset=*/0, /*mapping_size=*/2 * kGranularity,
          /*return_reservation_address=*/true));
  EXPECT_EQ(reservation->active_mapping_count(), 1);
  EXPECT_EQ(allocator->allocation_count(), 2);
}

TEST_F(DeviceAddressVmmAllocatorTest,
       PhysicalAllocationSizeControlsBudgetAccounting) {
  const DeviceAddressVmmAllocator::DeviceConfig config =
      Config(2 * kGranularity);
  ASSERT_OK_AND_ASSIGN(auto allocator,
                       TestDeviceAddressVmmAllocator::Create(
                           &platform_, {config},
                           /*physical_size_padding=*/kGranularity));

  ASSERT_OK_AND_ASSIGN(
      auto first,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/true, /*memory_space=*/0));
  ASSERT_NE(allocator->GetRawAllocation(/*device_ordinal=*/0, first.cref()),
            nullptr);
  EXPECT_EQ(allocator->GetRawAllocation(/*device_ordinal=*/0, first.cref())
                ->address()
                .size(),
            2 * kGranularity);
  // The first allocation consumes the full budget based on the physical size,
  // even though its requested size was one granularity unit.
  EXPECT_THAT(allocator->Allocate(/*device_ordinal=*/0, 2 * kGranularity,
                                  /*retry_on_failure=*/false,
                                  /*memory_space=*/0),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(allocator->allocation_count(), 1);

  ASSERT_THAT(allocator->Deallocate(/*device_ordinal=*/0, first.Release()),
              absl_testing::IsOk());
  ASSERT_THAT(allocator->SynchronizePendingOperations(/*device_ordinal=*/0),
              absl_testing::IsOk());
  ASSERT_OK_AND_ASSIGN(
      auto second,
      allocator->Allocate(/*device_ordinal=*/0, kGranularity,
                          /*retry_on_failure=*/false, /*memory_space=*/0));
  EXPECT_EQ(allocator->allocation_count(), 2);
}

}  // namespace
}  // namespace stream_executor
