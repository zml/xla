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

// Golden-value tests for the per-channel FP8 dense kernels behind
// zml$scaled_matmul (MetalScaledMatmulScheme::kFp8PerChannel): the decode GEMV
// and the two thin-M/prefill qmm tiles.
//
// These carry every FP8 projection of a mixed-precision NVFP4 checkpoint --
// Qwen3.6-27B-NVFP4 puts 233 of them through this path -- and had no numeric
// coverage at all. The values below are chosen so the exact answer is an
// integer: e4m3 and bf16 both represent small integers exactly, so
// out[b][n] = scale[n] * sum_k x[b][k] * w[n][k] holds with no rounding.
//
// A per-channel scale is constant along K, so a kernel that indexed it by
// anything other than the output column would still pass an all-ones test.
// Every case here varies the scale per column to pin that down.

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

// e4m3 code 0x38 is exactly 1.0; 0x40 is exactly 2.0.
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

  // Runs one per-channel case and checks every output element against the exact
  // product. `b == 1` selects the decode GEMV, otherwise the qmm tile whose BM
  // matches `b` the way MetalFp8GemvThunk picks it.
  void RunPerChannelCase(int32_t b, int32_t k, int32_t n) {
    ASSERT_EQ(k % 32, 0) << "the per-channel kernels have no contraction tail";

    // Every one of the three indices gets its own signature, so a kernel that
    // confuses them fails rather than coincidentally agreeing: x varies down M,
    // w varies down N (constant along K, so a transposed read changes the sum),
    // and the scale varies along N independently of w. All values are exact in
    // e4m3/bf16 and the accumulator is fp32, so the golden below is exact.
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
    // 0x7fc1 is a NaN, so an element the kernel never writes fails loudly.
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

    // Mirrors MetalFp8GemvThunk::ExecuteOnStream exactly; a divergence here
    // would test a launch geometry nothing in production uses.
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
        // bf16 keeps 8 significand bits, so allow a relative ulp either way.
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

// Decode: one row, the shape every token after the prompt takes.
TEST_F(MetalFp8KernelTest, PerChannelGemvMatchesGolden) {
  RunPerChannelCase(/*b=*/1, /*k=*/256, /*n=*/128);
}

// K is a multiple of 32 but not of 128, which the block-128 scheme could never
// produce and the classifier now explicitly admits for per-channel.
TEST_F(MetalFp8KernelTest, PerChannelGemvHandlesNonBlockAlignedK) {
  RunPerChannelCase(/*b=*/1, /*k=*/160, /*n=*/64);
}

// N is not a multiple of the BN=64 tile, so the last threadgroup is partial.
TEST_F(MetalFp8KernelTest, PerChannelGemvHandlesPartialN) {
  RunPerChannelCase(/*b=*/1, /*k=*/128, /*n=*/70);
}

// Thin-M qmm (BM=16): 1 < B <= 16.
TEST_F(MetalFp8KernelTest, PerChannelQmmMatchesGolden) {
  RunPerChannelCase(/*b=*/8, /*k=*/256, /*n=*/128);
}

// A partial M tile and a partial N tile at once.
TEST_F(MetalFp8KernelTest, PerChannelQmmHandlesPartialMAndN) {
  RunPerChannelCase(/*b=*/5, /*k=*/128, /*n=*/70);
}

// An N tail *wider than BK*, which is the only shape that reaches the row guard
// in PerChannelFp8BlockLoader::load_safe. N=100 leaves the second tile with
// num_outs=36 against BK=32, so a guard comparing the row index `bi` to the
// column extent zeroes rows 32..35 -- columns 96..99 come back 0. The n=70 cases
// above cannot catch it: their tail of 6 is below BK, so nothing is wrongly
// zeroed. Both BM tiles, because the guard is in the shared loader.
TEST_F(MetalFp8KernelTest, PerChannelQmmHandlesNTailWiderThanBk) {
  RunPerChannelCase(/*b=*/5, /*k=*/128, /*n=*/100);
}

// Prefill qmm (BM=64): B > 16.
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
