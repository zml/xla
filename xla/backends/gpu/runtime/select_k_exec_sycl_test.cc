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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace xla::gpu {
namespace {

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  CHECK(platform != nullptr);
  CHECK_OK(platform->ExecutorForDevice(0));
  return platform->ExecutorForDevice(0).value();
}

template <typename T>
std::vector<std::uint32_t> ExpectedTopK(absl::Span<const T> row,
                                        std::uint32_t k) {
  std::vector<std::uint32_t> order(row.size());
  std::iota(order.begin(), order.end(), 0);
  auto better = [&](std::uint32_t lhs, std::uint32_t rhs) {
    const float lhs_value = static_cast<float>(row[lhs]);
    const float rhs_value = static_cast<float>(row[rhs]);
    if (lhs_value == rhs_value) {
      return lhs < rhs;
    }
    return lhs_value > rhs_value;
  };
  std::partial_sort(order.begin(), order.begin() + k, order.end(), better);
  order.resize(k);
  return order;
}

template <typename T>
std::vector<std::uint32_t> ExpectedTopKWithPayload(
    absl::Span<const T> row, absl::Span<const std::uint32_t> payload,
    std::uint32_t k) {
  std::vector<std::uint32_t> order(row.size());
  std::iota(order.begin(), order.end(), 0);
  auto better = [&](std::uint32_t lhs, std::uint32_t rhs) {
    const float lhs_value = static_cast<float>(row[lhs]);
    const float rhs_value = static_cast<float>(row[rhs]);
    if (lhs_value == rhs_value) {
      return payload[lhs] < payload[rhs];
    }
    return lhs_value > rhs_value;
  };
  std::partial_sort(order.begin(), order.begin() + k, order.end(), better);
  std::vector<std::uint32_t> result;
  result.reserve(k);
  for (std::uint32_t i = 0; i < k; ++i) {
    result.push_back(payload[order[i]]);
  }
  return result;
}

template <typename T>
void RunSelectKTest(std::uint32_t batch, std::uint32_t n, std::uint32_t k,
                    bool force_ties) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);

  std::vector<T> h_data_in(static_cast<std::uint64_t>(batch) * n);
  for (std::uint32_t row = 0; row < batch; ++row) {
    for (std::uint32_t col = 0; col < n; ++col) {
      h_data_in[static_cast<std::uint64_t>(row) * n + col] =
          static_cast<T>(static_cast<float>(col % 257) / 1024.0f);
    }
    for (std::uint32_t i = 0; i < k; ++i) {
      const std::uint32_t index =
          force_ties ? (7 + i * 11) : ((row * 9973 + i * 7919 + 17) % n);
      h_data_in[static_cast<std::uint64_t>(row) * n + index] =
          static_cast<T>(force_ties ? 32.0f : 32.0f - static_cast<float>(i));
    }
  }

  se::DeviceAddress<T> d_data_in =
      stream_executor->AllocateArray<T>(batch * n, 0);
  se::DeviceAddress<T> d_data_out =
      stream_executor->AllocateArray<T>(batch * k, 0);
  se::DeviceAddress<std::uint32_t> d_indices_out =
      stream_executor->AllocateArray<std::uint32_t>(batch * k, 0);

  TF_ASSERT_OK(stream->MemcpyH2D(absl::MakeConstSpan(h_data_in), &d_data_in));
  TF_ASSERT_OK(select_k_exec<T>(
      stream_executor->device_ordinal(), &allocator, stream.get(), d_data_in,
      d_data_out, d_indices_out, batch, n, k));

  std::vector<T> h_data_out(static_cast<std::uint64_t>(batch) * k);
  std::vector<std::uint32_t> h_indices_out(
      static_cast<std::uint64_t>(batch) * k);
  TF_ASSERT_OK(stream->MemcpyD2H(d_data_out, absl::MakeSpan(h_data_out)));
  TF_ASSERT_OK(
      stream->MemcpyD2H(d_indices_out, absl::MakeSpan(h_indices_out)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  for (std::uint32_t row = 0; row < batch; ++row) {
    auto input_row = absl::MakeConstSpan(h_data_in).subspan(
        static_cast<std::uint64_t>(row) * n, n);
    const std::vector<std::uint32_t> expected = ExpectedTopK(input_row, k);
    for (std::uint32_t i = 0; i < k; ++i) {
      const std::uint64_t out = static_cast<std::uint64_t>(row) * k + i;
      EXPECT_EQ(h_indices_out[out], expected[i])
          << "row=" << row << " i=" << i;
      EXPECT_EQ(h_data_out[out], input_row[expected[i]])
          << "row=" << row << " i=" << i;
    }
  }

  stream_executor->Deallocate(&d_indices_out);
  stream_executor->Deallocate(&d_data_out);
  stream_executor->Deallocate(&d_data_in);
}

template <typename T>
void RunSelectKPayloadTest(std::uint32_t batch, std::uint32_t n,
                           std::uint32_t k, bool force_ties) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);

  std::vector<T> h_data_in(static_cast<std::uint64_t>(batch) * n);
  std::vector<std::uint32_t> h_indices_in(
      static_cast<std::uint64_t>(batch) * n);
  for (std::uint32_t row = 0; row < batch; ++row) {
    for (std::uint32_t col = 0; col < n; ++col) {
      const std::uint64_t offset = static_cast<std::uint64_t>(row) * n + col;
      h_data_in[offset] =
          static_cast<T>(static_cast<float>((col * 13 + row) % 257) / 1024.0f);
      h_indices_in[offset] = row * 10000 + (n - 1 - col);
    }
    for (std::uint32_t i = 0; i < k; ++i) {
      const std::uint32_t index =
          force_ties ? (3 + i * 17) : ((row * 997 + i * 101 + 11) % n);
      const std::uint64_t offset = static_cast<std::uint64_t>(row) * n + index;
      h_data_in[offset] =
          static_cast<T>(force_ties ? 64.0f : 64.0f - static_cast<float>(i));
      h_indices_in[offset] =
          force_ties ? row * 10000 + i : row * 10000 + index;
    }
  }

  se::DeviceAddress<T> d_data_in =
      stream_executor->AllocateArray<T>(batch * n, 0);
  se::DeviceAddress<std::uint32_t> d_indices_in =
      stream_executor->AllocateArray<std::uint32_t>(batch * n, 0);
  se::DeviceAddress<T> d_data_out =
      stream_executor->AllocateArray<T>(batch * k, 0);
  se::DeviceAddress<std::uint32_t> d_indices_out =
      stream_executor->AllocateArray<std::uint32_t>(batch * k, 0);

  TF_ASSERT_OK(stream->MemcpyH2D(absl::MakeConstSpan(h_data_in), &d_data_in));
  TF_ASSERT_OK(
      stream->MemcpyH2D(absl::MakeConstSpan(h_indices_in), &d_indices_in));
  TF_ASSERT_OK(select_k_payload_exec<T>(
      stream_executor->device_ordinal(), &allocator, stream.get(), d_data_in,
      d_indices_in, d_data_out, d_indices_out, batch, n, k));

  std::vector<T> h_data_out(static_cast<std::uint64_t>(batch) * k);
  std::vector<std::uint32_t> h_indices_out(
      static_cast<std::uint64_t>(batch) * k);
  TF_ASSERT_OK(stream->MemcpyD2H(d_data_out, absl::MakeSpan(h_data_out)));
  TF_ASSERT_OK(
      stream->MemcpyD2H(d_indices_out, absl::MakeSpan(h_indices_out)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  for (std::uint32_t row = 0; row < batch; ++row) {
    auto input_row = absl::MakeConstSpan(h_data_in).subspan(
        static_cast<std::uint64_t>(row) * n, n);
    auto payload_row = absl::MakeConstSpan(h_indices_in).subspan(
        static_cast<std::uint64_t>(row) * n, n);
    const std::vector<std::uint32_t> expected =
        ExpectedTopKWithPayload(input_row, payload_row, k);
    for (std::uint32_t i = 0; i < k; ++i) {
      const std::uint64_t out = static_cast<std::uint64_t>(row) * k + i;
      EXPECT_EQ(h_indices_out[out], expected[i])
          << "row=" << row << " i=" << i;
    }
  }

  stream_executor->Deallocate(&d_indices_out);
  stream_executor->Deallocate(&d_data_out);
  stream_executor->Deallocate(&d_indices_in);
  stream_executor->Deallocate(&d_data_in);
}

bool RunLargeRegression() {
  const char* value = std::getenv("XLA_RUN_ONEAPI_LARGE_SELECT_K_TEST");
  return value != nullptr && std::string(value) == "1";
}

TEST(SyclSelectKExecTest, SelectKFloat) {
  RunSelectKTest<float>(/*batch=*/4, /*n=*/4096, /*k=*/4,
                        /*force_ties=*/false);
}

TEST(SyclSelectKExecTest, SelectKBFloat16) {
  RunSelectKTest<::xla::bfloat16>(/*batch=*/4, /*n=*/4096, /*k=*/4,
                                  /*force_ties=*/false);
}

TEST(SyclSelectKExecTest, StableTieBreakUsesLowerIndex) {
  RunSelectKTest<float>(/*batch=*/2, /*n=*/512, /*k=*/4,
                        /*force_ties=*/true);
}

TEST(SyclSelectKExecTest, PayloadSelectKFloat) {
  RunSelectKPayloadTest<float>(/*batch=*/4, /*n=*/512, /*k=*/4,
                               /*force_ties=*/false);
}

TEST(SyclSelectKExecTest, PayloadSelectKBFloat16) {
  RunSelectKPayloadTest<::xla::bfloat16>(/*batch=*/4, /*n=*/512, /*k=*/4,
                                         /*force_ties=*/false);
}

TEST(SyclSelectKExecTest, PayloadTieBreakUsesLowerPayloadIndex) {
  RunSelectKPayloadTest<float>(/*batch=*/2, /*n=*/512, /*k=*/4,
                               /*force_ties=*/true);
}

TEST(SyclSelectKExecTest, LmHeadShapeBFloat16) {
  if (!RunLargeRegression()) {
    GTEST_SKIP() << "Set XLA_RUN_ONEAPI_LARGE_SELECT_K_TEST=1 to run the "
                    "BF16 [2048, 128256], k=4 regression.";
  }
  RunSelectKTest<::xla::bfloat16>(/*batch=*/2048, /*n=*/128256, /*k=*/4,
                                  /*force_ties=*/false);
}

}  // namespace
}  // namespace xla::gpu
