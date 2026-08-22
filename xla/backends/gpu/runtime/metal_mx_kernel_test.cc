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
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/mlx_kernels.h"
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

constexpr int32_t kGroup = 32;

constexpr uint8_t kE4m3One = 0x38;
constexpr uint8_t kE4m3Two = 0x40;
constexpr uint8_t kE2m1One = 0x2;
constexpr uint8_t kE2m1Two = 0x4;
constexpr uint8_t kE8m0Bias = 127;

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

float ActivationValue(int32_t row, int32_t group) {
  return static_cast<float>((1 + (row % 3)) * (1 + (group % 2)));
}
float WeightValue(int32_t col, int32_t group) {
  return ((col + group) % 2 == 0) ? 1.0f : 2.0f;
}
int32_t ScaleExponent(int32_t col, int32_t group) {
  return (group % 2) + (col % 2);
}

class MetalMxKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (std::getenv("METAL_TOOLCHAIN") == nullptr) {
      GTEST_SKIP() << "METAL_TOOLCHAIN is required to compile Metal sources.";
    }
    TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                            se::PlatformManager::PlatformWithName("METAL"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    metal_executor_ = static_cast<se::metal::MetalExecutor*>(executor_);
    TF_ASSERT_OK_AND_ASSIGN(qmv_metallib_, CompileMetalSourceToMetallibCached(
                                               get_mlx_mxfp_qmv()));
    TF_ASSERT_OK_AND_ASSIGN(qmm_metallib_, CompileMetalSourceToMetallibCached(
                                               get_mlx_steel_qgemm()));
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

  void RunMxCase(int32_t bits, int32_t m, int32_t k, int32_t n) {
    ASSERT_EQ(k % kGroup, 0) << "MX is group-32; K must be a whole number of "
                                "groups or the scale grid is not expressible";
    const int32_t groups = k / kGroup;

    std::vector<uint16_t> x(static_cast<size_t>(m) * k);
    for (int32_t row = 0; row < m; ++row) {
      for (int32_t col = 0; col < k; ++col) {
        x[static_cast<size_t>(row) * k + col] =
            Bfloat16Bits(ActivationValue(row, col / kGroup));
      }
    }

    const size_t w_bytes =
        static_cast<size_t>(n) * k / (bits == 8 ? 1 : 2);
    std::vector<uint8_t> w(w_bytes, 0);
    for (int32_t col = 0; col < n; ++col) {
      for (int32_t depth = 0; depth < k; ++depth) {
        const float value = WeightValue(col, depth / kGroup);
        if (bits == 8) {
          w[static_cast<size_t>(col) * k + depth] =
              (value == 1.0f) ? kE4m3One : kE4m3Two;
        } else {
          const uint8_t nibble = (value == 1.0f) ? kE2m1One : kE2m1Two;
          const size_t index = (static_cast<size_t>(col) * k + depth) / 2;
          w[index] |= static_cast<uint8_t>(nibble << ((depth % 2) * 4));
        }
      }
    }

    std::vector<uint8_t> scales(static_cast<size_t>(n) * groups);
    for (int32_t col = 0; col < n; ++col) {
      for (int32_t g = 0; g < groups; ++g) {
        scales[static_cast<size_t>(col) * groups + g] =
            static_cast<uint8_t>(kE8m0Bias + ScaleExponent(col, g));
      }
    }

    std::vector<uint16_t> out(static_cast<size_t>(m) * n, 0x7fc1);

    const int32_t dims[4] = {m, k, n, groups};
    se::DeviceAddressBase x_device =
        AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
    se::DeviceAddressBase w_device = AllocateAndCopy(w.data(), w.size());
    se::DeviceAddressBase scale_device =
        AllocateAndCopy(scales.data(), scales.size());
    se::DeviceAddressBase out_device =
        AllocateAndCopy(out.data(), out.size() * sizeof(out[0]));
    ASSERT_NE(x_device.opaque(), nullptr);
    ASSERT_NE(w_device.opaque(), nullptr);
    ASSERT_NE(scale_device.opaque(), nullptr);
    ASSERT_NE(out_device.opaque(), nullptr);

    constexpr int64_t kQmvMaxBatch = 8;
    const bool decode = m < kQmvMaxBatch;
    const int32_t pack_factor = 32 / bits;
    const int32_t fast_block = pack_factor * 2 * 32;
    const bool aligned = (n % 8 == 0) && (k % fast_block == 0);
    const std::string prefix = (bits == 8) ? "mxfp8" : "mxfp4";
    const std::string kernel_name =
        decode ? (prefix + (aligned ? "_qmv_fast" : "_qmv"))
               : (prefix + "_qmm_t");

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> kernel,
        metal_executor_->LoadKernelWithConstants(
            decode ? qmv_metallib_ : qmm_metallib_, kernel_name,
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
      TF_ASSERT_OK(kernel->Launch(
          se::ThreadDim(32, 2, 1),
          se::BlockDim(static_cast<uint64_t>(m),
                       static_cast<uint64_t>((n + 7) / 8), 1),
          stream.get(), args));
    } else {
      constexpr int64_t kQmmBM = 16, kQmmBN = 64;
      TF_ASSERT_OK(kernel->Launch(
          se::ThreadDim(32, 2, 2),
          se::BlockDim(static_cast<uint64_t>((n + kQmmBN - 1) / kQmmBN),
                       static_cast<uint64_t>((m + kQmmBM - 1) / kQmmBM), 1),
          stream.get(), args));
    }
    TF_ASSERT_OK(stream->BlockHostUntilDone());
    TF_ASSERT_OK(executor_->SynchronousMemcpyD2H(
        out_device, out.size() * sizeof(out[0]), out.data()));

    for (int32_t row = 0; row < m; ++row) {
      for (int32_t col = 0; col < n; ++col) {
        float expected = 0.0f;
        for (int32_t g = 0; g < groups; ++g) {
          const float scale =
              static_cast<float>(1 << ScaleExponent(col, g));
          expected += static_cast<float>(kGroup) * ActivationValue(row, g) *
                      WeightValue(col, g) * scale;
        }
        const float actual =
            Bfloat16ToFloat(out[static_cast<size_t>(row) * n + col]);
        EXPECT_NEAR(actual, expected, expected * 0.005f)
            << "bits=" << bits << " at row " << row << ", column " << col;
      }
    }
  }

  se::StreamExecutor* executor_ = nullptr;
  se::metal::MetalExecutor* metal_executor_ = nullptr;
  std::vector<uint8_t> qmv_metallib_;
  std::vector<uint8_t> qmm_metallib_;
  std::vector<se::DeviceAddressBase> allocations_;
};

TEST_F(MetalMxKernelTest, Mxfp8QmvFastMatchesGolden) {
  RunMxCase(/*bits=*/8, /*m=*/1, /*k=*/256, /*n=*/128);
}

TEST_F(MetalMxKernelTest, Mxfp8QmvGuardedHandlesUnalignedK) {
  RunMxCase(/*bits=*/8, /*m=*/1, /*k=*/160, /*n=*/64);
}

TEST_F(MetalMxKernelTest, Mxfp8QmvHandlesPartialN) {
  RunMxCase(/*bits=*/8, /*m=*/1, /*k=*/128, /*n=*/70);
}

TEST_F(MetalMxKernelTest, Mxfp8QmvHandlesThinBatch) {
  RunMxCase(/*bits=*/8, /*m=*/4, /*k=*/256, /*n=*/64);
}

TEST_F(MetalMxKernelTest, Mxfp8QmmMatchesGolden) {
  RunMxCase(/*bits=*/8, /*m=*/16, /*k=*/256, /*n=*/64);
}

TEST_F(MetalMxKernelTest, Mxfp8QmmHandlesPartialTiles) {
  RunMxCase(/*bits=*/8, /*m=*/20, /*k=*/128, /*n=*/72);
}

TEST_F(MetalMxKernelTest, Mxfp4QmvFastMatchesGolden) {
  RunMxCase(/*bits=*/4, /*m=*/1, /*k=*/512, /*n=*/128);
}

TEST_F(MetalMxKernelTest, Mxfp4QmvGuardedHandlesUnalignedK) {
  RunMxCase(/*bits=*/4, /*m=*/1, /*k=*/160, /*n=*/64);
}

TEST_F(MetalMxKernelTest, Mxfp4QmvHandlesPartialN) {
  RunMxCase(/*bits=*/4, /*m=*/1, /*k=*/128, /*n=*/71);
}

TEST_F(MetalMxKernelTest, Mxfp4QmmMatchesGolden) {
  RunMxCase(/*bits=*/4, /*m=*/16, /*k=*/512, /*n=*/64);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
