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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/service/gpu/metal_kernels/fp8_gemv_pc.h"
#include "xla/service/gpu/metal_kernels/mlx_kernels.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace se = ::stream_executor;

constexpr uint8_t kE4m3One = 0x38;
constexpr uint8_t kE4m3Two = 0x40;

uint16_t Bfloat16Bits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  __builtin_memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float Bfloat16ToFloat(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float value;
  __builtin_memcpy(&value, &widened, sizeof(value));
  return value;
}

class MetalFp8KernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (std::getenv("METAL_TOOLCHAIN") == nullptr) {
      GTEST_SKIP() << "METAL_TOOLCHAIN is required to compile Metal sources.";
    }
    TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                            se::PlatformManager::PlatformWithName("METAL"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    metal_executor_ = static_cast<se::metal::MetalExecutor*>(executor_);
    TF_ASSERT_OK_AND_ASSIGN(
        qmm_metallib_, CompileMetalSourceToMetallibCached(get_mlx_steel_qgemm()));
    TF_ASSERT_OK_AND_ASSIGN(
        gemv_metallib_, CompileMetalSourceToMetallibCached(get_fp8_gemv_pc()));
  }

  void TearDown() override {
    for (se::DeviceAddressBase& allocation : allocations_) {
      executor_->Deallocate(&allocation);
    }
  }

  se::DeviceAddressBase AllocateAndCopy(const void* data, uint64_t bytes) {
    se::DeviceAddressBase allocation = executor_->Allocate(bytes, 0);
    EXPECT_NE(allocation.opaque(), nullptr);
    if (allocation.opaque() != nullptr) {
      EXPECT_TRUE(executor_->SynchronousMemcpy(&allocation, data, bytes).ok());
      allocations_.push_back(allocation);
    }
    return allocation;
  }

  void RunPerChannelCase(int32_t b, int32_t k, int32_t n) {
    ASSERT_EQ(k % 32, 0) << "the per-channel kernels have no contraction tail";

    std::vector<uint16_t> x(static_cast<size_t>(b) * k);
    for (int32_t row = 0; row < b; ++row) {
      const uint16_t value = Bfloat16Bits(static_cast<float>(1 + (row % 3)));
      for (int32_t col = 0; col < k; ++col) {
        x[static_cast<size_t>(row) * k + col] = value;
      }
    }
    std::vector<uint8_t> w(static_cast<size_t>(n) * k);
    for (int32_t col = 0; col < n; ++col) {
      const uint8_t value = (col % 2 == 0) ? kE4m3One : kE4m3Two;
      for (int32_t depth = 0; depth < k; ++depth) {
        w[static_cast<size_t>(col) * k + depth] = value;
      }
    }
    std::vector<uint16_t> scale(static_cast<size_t>(n));
    for (int32_t col = 0; col < n; ++col) {
      scale[col] = Bfloat16Bits(static_cast<float>(1 + (col % 4)));
    }
    std::vector<uint16_t> out(static_cast<size_t>(b) * n, 0x7fc1);

    const int32_t dims[4] = {b, k, n, k / 128};
    se::DeviceAddressBase x_device =
        AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
    se::DeviceAddressBase w_device =
        AllocateAndCopy(w.data(), w.size() * sizeof(w[0]));
    se::DeviceAddressBase scale_device =
        AllocateAndCopy(scale.data(), scale.size() * sizeof(scale[0]));
    se::DeviceAddressBase out_device =
        AllocateAndCopy(out.data(), out.size() * sizeof(out[0]));
    ASSERT_NE(x_device.opaque(), nullptr);
    ASSERT_NE(w_device.opaque(), nullptr);
    ASSERT_NE(scale_device.opaque(), nullptr);
    ASSERT_NE(out_device.opaque(), nullptr);

    const bool decode = (b == 1);
    const bool large_m = b > 16;
    const std::string kernel_name =
        decode ? "fp8_gemv_pc"
               : (large_m ? "fp8_qmm_t_pc_bm64" : "fp8_qmm_t_pc");
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> kernel,
        metal_executor_->LoadKernelWithConstants(
            decode ? gemv_metallib_ : qmm_metallib_, kernel_name,
            /*arity=*/5, {}));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                            executor_->CreateStream());

    se::KernelArgsPackedArray args(/*num_args=*/5);
    args.add_argument(x_device);
    args.add_argument(w_device);
    args.add_argument(scale_device);
    args.add_argument(out_device);
    args.add_argument(dims);

    if (decode) {
      TF_ASSERT_OK(kernel->Launch(se::ThreadDim(256, 1, 1),
                                  se::BlockDim(static_cast<uint64_t>(n), 1, 1),
                                  stream.get(), args));
    } else {
      constexpr int64_t kPcBN = 64;
      const int64_t pc_bm = large_m ? 64 : 16;
      TF_ASSERT_OK(kernel->Launch(
          se::ThreadDim(32, 2, 2),
          se::BlockDim(static_cast<uint64_t>((n + kPcBN - 1) / kPcBN),
                       static_cast<uint64_t>((b + pc_bm - 1) / pc_bm), 1),
          stream.get(), args));
    }
    TF_ASSERT_OK(stream->BlockHostUntilDone());
    TF_ASSERT_OK(executor_->SynchronousMemcpyD2H(
        out_device, out.size() * sizeof(out[0]), out.data()));

    for (int32_t row = 0; row < b; ++row) {
      for (int32_t col = 0; col < n; ++col) {
        const float expected = static_cast<float>(1 + (row % 3)) *
                               static_cast<float>(k) *
                               static_cast<float>((col % 2 == 0) ? 1 : 2) *
                               static_cast<float>(1 + (col % 4));
        const float actual =
            Bfloat16ToFloat(out[static_cast<size_t>(row) * n + col]);
        EXPECT_NEAR(actual, expected, expected * 0.005f)
            << "at row " << row << ", column " << col;
      }
    }
  }

  se::StreamExecutor* executor_ = nullptr;
  se::metal::MetalExecutor* metal_executor_ = nullptr;
  std::vector<uint8_t> qmm_metallib_;
  std::vector<uint8_t> gemv_metallib_;
  std::vector<se::DeviceAddressBase> allocations_;
};

TEST_F(MetalFp8KernelTest, PerChannelGemvMatchesGolden) {
  RunPerChannelCase(/*b=*/1, /*k=*/256, /*n=*/128);
}

TEST_F(MetalFp8KernelTest, PerChannelGemvHandlesNonBlockAlignedK) {
  RunPerChannelCase(/*b=*/1, /*k=*/160, /*n=*/64);
}

TEST_F(MetalFp8KernelTest, PerChannelGemvHandlesPartialN) {
  RunPerChannelCase(/*b=*/1, /*k=*/128, /*n=*/70);
}

TEST_F(MetalFp8KernelTest, PerChannelQmmMatchesGolden) {
  RunPerChannelCase(/*b=*/8, /*k=*/256, /*n=*/128);
}

TEST_F(MetalFp8KernelTest, PerChannelQmmHandlesPartialMAndN) {
  RunPerChannelCase(/*b=*/5, /*k=*/128, /*n=*/70);
}

TEST_F(MetalFp8KernelTest, PerChannelQmmHandlesNTailWiderThanBk) {
  RunPerChannelCase(/*b=*/5, /*k=*/128, /*n=*/100);
}

TEST_F(MetalFp8KernelTest, PerChannelQmmBm64MatchesGolden) {
  RunPerChannelCase(/*b=*/32, /*k=*/256, /*n=*/128);
}

TEST_F(MetalFp8KernelTest, PerChannelQmmBm64HandlesPartialMAndN) {
  RunPerChannelCase(/*b=*/20, /*k=*/160, /*n=*/70);
}

TEST_F(MetalFp8KernelTest, PerChannelQmmBm64HandlesNTailWiderThanBk) {
  RunPerChannelCase(/*b=*/20, /*k=*/160, /*n=*/100);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
