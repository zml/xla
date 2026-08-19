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

// Bfloat16Bits above truncates, which is exact for every value the cases below
// feed in -- they are all small integers -- but wrong for a computed result:
// the hardware rounds to nearest-even on the store. Use this whenever the
// expected value is a product rather than an input.
uint16_t Bfloat16BitsRne(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = ((bits >> 16) & 1) + 0x7fff;
  return static_cast<uint16_t>((bits + rounding) >> 16);
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
  //
  // `per_tensor` sends a one-element scale buffer with the negative dims.w
  // sentinel, i.e. stride 0: every output column must then read scale[0]. A
  // kernel that ignores the stride walks off the end of a 2-byte allocation --
  // which Metal will not fault on, the pages being there -- so the golden
  // below, not a crash, is what catches it.
  // `f32_scale` sends the scale as f32 through the _f32 entry points. That is a
  // wiring check: an f32 buffer bound to a `device const bfloat*` is read as
  // pairs of bf16 and silently produces garbage. It does NOT prove the extra
  // precision -- see PerTensorGemvKeepsAnF32ScaleExact for that.
  void RunPerChannelCase(int32_t b, int32_t k, int32_t n,
                         bool per_tensor = false, bool f32_scale = false) {
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
    auto scale_of = [&](int32_t col) {
      return static_cast<float>(per_tensor ? 3 : 1 + (col % 4));
    };
    const size_t scale_count = per_tensor ? 1 : static_cast<size_t>(n);
    std::vector<uint16_t> scale_bf16(f32_scale ? 0 : scale_count);
    std::vector<float> scale_f32(f32_scale ? scale_count : 0);
    for (size_t i = 0; i < scale_count; ++i) {
      const float v = scale_of(static_cast<int32_t>(i));
      if (f32_scale) {
        scale_f32[i] = v;
      } else {
        scale_bf16[i] = Bfloat16Bits(v);
      }
    }
    const void* scale_data =
        f32_scale ? static_cast<const void*>(scale_f32.data())
                  : static_cast<const void*>(scale_bf16.data());
    const uint64_t scale_bytes = scale_count * (f32_scale ? 4 : 2);
    // 0x7fc1 is a NaN, so an element the kernel never writes fails loudly.
    std::vector<uint16_t> out(static_cast<size_t>(b) * n, 0x7fc1);

    // dims.w is the scale's row stride for these kernels: 1 for [N,1], and a
    // negative sentinel for a [1,1] whole-tensor scale. It is not k/128 here --
    // only the block-128 GEMV reads it that way.
    const int32_t dims[4] = {b, k, n, per_tensor ? -1 : 1};
    se::DeviceAddressBase x_device =
        AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
    se::DeviceAddressBase w_device =
        AllocateAndCopy(w.data(), w.size() * sizeof(w[0]));
    se::DeviceAddressBase scale_device = AllocateAndCopy(scale_data, scale_bytes);
    se::DeviceAddressBase out_device =
        AllocateAndCopy(out.data(), out.size() * sizeof(out[0]));
    ASSERT_NE(x_device.opaque(), nullptr);
    ASSERT_NE(w_device.opaque(), nullptr);
    ASSERT_NE(scale_device.opaque(), nullptr);
    ASSERT_NE(out_device.opaque(), nullptr);

    const bool decode = (b == 1);
    const bool large_m = b > 16;
    std::string kernel_name = decode ? "fp8_gemv_pc"
                                    : (large_m ? "fp8_qmm_t_pc_bm64"
                                               : "fp8_qmm_t_pc");
    if (f32_scale) kernel_name += "_f32";
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
                               scale_of(col);
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

// A whole-tensor scale is the same kernel at stride 0: one element, read by
// every output column. All three entries, since each indexes the scale itself.
TEST_F(MetalFp8KernelTest, PerTensorGemvReadsTheSingleScale) {
  RunPerChannelCase(/*b=*/1, /*k=*/256, /*n=*/128, /*per_tensor=*/true);
}

TEST_F(MetalFp8KernelTest, PerTensorQmmReadsTheSingleScale) {
  RunPerChannelCase(/*b=*/8, /*k=*/256, /*n=*/128, /*per_tensor=*/true);
}

TEST_F(MetalFp8KernelTest, PerTensorQmmBm64ReadsTheSingleScale) {
  RunPerChannelCase(/*b=*/32, /*k=*/256, /*n=*/128, /*per_tensor=*/true);
}

// Stride 0 must survive a partial N tile too: the tail path indexes the scale
// through the same base, so a stride the loader ignores shows up here as well.
TEST_F(MetalFp8KernelTest, PerTensorQmmHandlesPartialN) {
  RunPerChannelCase(/*b=*/5, /*k=*/128, /*n=*/100, /*per_tensor=*/true);
}

// The _f32 entries, through all three tiles and both scale grids. Wiring only:
// every value here is bf16-exact, so an f32 and a bf16 scale agree bit for bit.
TEST_F(MetalFp8KernelTest, PerChannelGemvAcceptsAnF32Scale) {
  RunPerChannelCase(/*b=*/1, /*k=*/256, /*n=*/128, /*per_tensor=*/false,
                    /*f32_scale=*/true);
}

TEST_F(MetalFp8KernelTest, PerChannelQmmAcceptsAnF32Scale) {
  RunPerChannelCase(/*b=*/8, /*k=*/256, /*n=*/128, /*per_tensor=*/false,
                    /*f32_scale=*/true);
}

TEST_F(MetalFp8KernelTest, PerChannelQmmBm64AcceptsAnF32Scale) {
  RunPerChannelCase(/*b=*/32, /*k=*/256, /*n=*/128, /*per_tensor=*/false,
                    /*f32_scale=*/true);
}

TEST_F(MetalFp8KernelTest, PerTensorGemvAcceptsAnF32Scale) {
  RunPerChannelCase(/*b=*/1, /*k=*/256, /*n=*/128, /*per_tensor=*/true,
                    /*f32_scale=*/true);
}

// The one case that actually proves the f32 scale buys precision, which the
// harness above cannot: its generators only produce bf16-exact values and
// powers of two, and bf16 rounding commutes with those, so an f32-scale kernel
// and a bf16-scale kernel are bit-identical on every case there.
//
// This uses a real ModelOpt scale -- layer 11's q_proj -- which bf16 cannot
// hold, and asserts the output bit pattern. The decode GEMV multiplies the f32
// scale into an f32 accumulator and rounds once on the store, so the answer is
// exactly bf16(scale * sum). Rounding the scale first lands elsewhere, and the
// test pins that the two differ so it cannot pass vacuously.
TEST_F(MetalFp8KernelTest, PerTensorGemvKeepsAnF32ScaleExact) {
  // K must make the running sum a non-power-of-two: bf16 rounding commutes
  // with multiplication by a power of two, so at K=256 the two spellings of the
  // scale agree bit for bit and the test would pass vacuously. 320 = 2^6 * 5
  // does not commute away, and the ASSERT_NE below keeps that honest.
  constexpr int32_t kK = 320;
  constexpr int32_t kN = 64;
  constexpr float kScale = 0.00111607f;  // not representable in bf16

  std::vector<uint16_t> x(kK, Bfloat16Bits(1.0f));
  std::vector<uint8_t> w(static_cast<size_t>(kN) * kK, kE4m3One);
  std::vector<float> scale(1, kScale);
  std::vector<uint16_t> out(kN, 0x7fc1);
  const int32_t dims[4] = {1, kK, kN, -1};

  se::DeviceAddressBase x_device =
      AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
  se::DeviceAddressBase w_device =
      AllocateAndCopy(w.data(), w.size() * sizeof(w[0]));
  se::DeviceAddressBase scale_device =
      AllocateAndCopy(scale.data(), scale.size() * sizeof(scale[0]));
  se::DeviceAddressBase out_device =
      AllocateAndCopy(out.data(), out.size() * sizeof(out[0]));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Kernel> kernel,
                          metal_executor_->LoadKernelWithConstants(
                              gemv_metallib_, "fp8_gemv_pc_f32",
                              /*arity=*/5, {}));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor_->CreateStream());
  se::KernelArgsPackedArray args(/*num_args=*/5);
  args.add_argument(x_device);
  args.add_argument(w_device);
  args.add_argument(scale_device);
  args.add_argument(out_device);
  args.add_argument(dims);
  TF_ASSERT_OK(kernel->Launch(se::ThreadDim(256, 1, 1),
                              se::BlockDim(static_cast<uint64_t>(kN), 1, 1),
                              stream.get(), args));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  TF_ASSERT_OK(executor_->SynchronousMemcpyD2H(
      out_device, out.size() * sizeof(out[0]), out.data()));

  const float sum = static_cast<float>(kK);  // 1.0 * 1.0, kK times
  const uint16_t expected = Bfloat16BitsRne(kScale * sum);
  const uint16_t if_scale_were_rounded =
      Bfloat16BitsRne(Bfloat16ToFloat(Bfloat16BitsRne(kScale)) * sum);
  ASSERT_NE(expected, if_scale_were_rounded)
      << "pick a scale where the two spellings actually differ";
  for (int32_t col = 0; col < kN; ++col) {
    EXPECT_EQ(out[col], expected) << "at column " << col;
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
