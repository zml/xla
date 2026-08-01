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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/metal_nvfp4_dispatch.h"
#include "xla/service/gpu/metal_air_toolchain.h"
#include "xla/service/gpu/metal_kernels/mlx_kernels.h"
#include "xla/service/gpu/metal_kernels/moe_argsort.h"
#include "xla/service/gpu/metal_kernels/permute_rows.h"
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

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

uint16_t Bfloat16Bits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

uint16_t RoundedBfloat16Bits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t retained_lsb = (bits >> 16) & 1;
  bits += 0x7fff + retained_lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float Bfloat16Value(uint16_t bits) {
  uint32_t value_bits = static_cast<uint32_t>(bits) << 16;
  float value;
  static_assert(sizeof(value_bits) == sizeof(value));
  std::memcpy(&value, &value_bits, sizeof(value));
  return value;
}

float DecodeFp4E2m1(uint8_t bits) {
  constexpr float kMagnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                    2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = kMagnitudes[bits & 0x7];
  return (bits & 0x8) != 0 ? -magnitude : magnitude;
}

float DecodeFp8E4m3(uint8_t bits) {
  const int exponent = (bits >> 3) & 0xf;
  const int mantissa = bits & 0x7;
  const float magnitude =
      exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                    : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                                 exponent - 7);
  return (bits & 0x80) != 0 ? -magnitude : magnitude;
}

class MetalNvfp4KernelTest : public ::testing::Test {
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
        metallib_, CompileMetalSourceToMetallibCached(get_mlx_steel_qgemm()));
    TF_ASSERT_OK_AND_ASSIGN(
        qmv_metallib_, CompileMetalSourceToMetallibCached(get_mlx_fp4_qmv()));
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

  se::DeviceAddressBase Allocate(uint64_t bytes) {
    se::DeviceAddressBase allocation = executor_->Allocate(bytes, 0);
    EXPECT_NE(allocation.opaque(), nullptr);
    if (allocation.opaque() != nullptr) allocations_.push_back(allocation);
    return allocation;
  }

  void TearDown() override {
    for (se::DeviceAddressBase& allocation : allocations_) {
      executor_->Deallocate(&allocation);
    }
  }

  void RunDenseCase(int32_t m, int32_t k, int32_t n,
                    const std::string& kernel_name) {
    ASSERT_EQ(k % kNvfp4GroupSize, 0);

    // f4e2m1 code 0x2 is 1.0; e4m3 code 0x38 is 1.0. Therefore an
    // all-ones activation and weight matrix must produce exactly float(k).
    std::vector<uint16_t> x(static_cast<size_t>(m) * k, Bfloat16Bits(1.0f));
    std::vector<uint8_t> w(static_cast<size_t>(n) * k / 2, 0x22);
    std::vector<uint8_t> scales(static_cast<size_t>(n) * k / 16, 0x38);
    std::vector<uint16_t> output(static_cast<size_t>(m) * n, 0x7fc1);
    const int32_t dims[4] = {m, k, n, k / 16};

    se::DeviceAddressBase x_device =
        AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
    se::DeviceAddressBase w_device =
        AllocateAndCopy(w.data(), w.size() * sizeof(w[0]));
    se::DeviceAddressBase scales_device =
        AllocateAndCopy(scales.data(), scales.size() * sizeof(scales[0]));
    se::DeviceAddressBase output_device =
        Allocate(output.size() * sizeof(output[0]));
    ASSERT_NE(x_device.opaque(), nullptr);
    ASSERT_NE(w_device.opaque(), nullptr);
    ASSERT_NE(scales_device.opaque(), nullptr);
    ASSERT_NE(output_device.opaque(), nullptr);
    ASSERT_TRUE(executor_
                    ->SynchronousMemcpy(&output_device, output.data(),
                                        output.size() * sizeof(output[0]))
                    .ok());

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> kernel,
        metal_executor_->LoadKernelWithConstants(metallib_, kernel_name,
                                                 /*arity=*/5, {}));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                            executor_->CreateStream());

    se::KernelArgsPackedArray args(/*num_args=*/5);
    args.add_argument(x_device);
    args.add_argument(w_device);
    args.add_argument(scales_device);
    args.add_argument(output_device);
    args.add_argument(dims);
    TF_ASSERT_OK(kernel->Launch(
        se::ThreadDim(32, 2, 2),
        se::BlockDim(static_cast<uint64_t>((n + kNvfp4QmmBN - 1) / kNvfp4QmmBN),
                     static_cast<uint64_t>((m + kNvfp4QmmBM - 1) / kNvfp4QmmBM),
                     1),
        stream.get(), args));
    TF_ASSERT_OK(stream->Memcpy(output.data(), output_device,
                                output.size() * sizeof(output[0])));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    const uint16_t expected = Bfloat16Bits(static_cast<float>(k));
    for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_EQ(output[i], expected)
          << "m=" << m << " k=" << k << " n=" << n << " output[" << i << "]";
    }
  }

  void RunQmvCase(int32_t m, int32_t k, int32_t n,
                  const std::string& kernel_name, int32_t vecs_per_tg) {
    ASSERT_EQ(k % kNvfp4GroupSize, 0);

    std::vector<uint16_t> x(static_cast<size_t>(m) * k, Bfloat16Bits(1.0f));
    std::vector<uint8_t> w(static_cast<size_t>(n) * k / 2, 0x22);
    std::vector<uint8_t> scales(static_cast<size_t>(n) * k / 16, 0x38);
    std::vector<uint16_t> output(static_cast<size_t>(m) * n, 0x7fc1);
    const int32_t dims[4] = {m, k, n, k / 16};

    se::DeviceAddressBase w_device =
        AllocateAndCopy(w.data(), w.size() * sizeof(w[0]));
    se::DeviceAddressBase scales_device =
        AllocateAndCopy(scales.data(), scales.size() * sizeof(scales[0]));
    se::DeviceAddressBase x_device =
        AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
    se::DeviceAddressBase output_device =
        AllocateAndCopy(output.data(), output.size() * sizeof(output[0]));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> kernel,
        metal_executor_->LoadKernelWithConstants(qmv_metallib_, kernel_name,
                                                 /*arity=*/5, {}));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                            executor_->CreateStream());

    se::KernelArgsPackedArray args(/*num_args=*/5);
    args.add_argument(w_device);
    args.add_argument(scales_device);
    args.add_argument(x_device);
    args.add_argument(output_device);
    args.add_argument(dims);
    const bool is_wide = vecs_per_tg > 1;
    TF_ASSERT_OK(kernel->Launch(
        se::ThreadDim(32, 2, 1),
        se::BlockDim(static_cast<uint64_t>((m + vecs_per_tg - 1) / vecs_per_tg),
                     static_cast<uint64_t>((n + (is_wide ? 4 : 8) - 1) /
                                           (is_wide ? 4 : 8)),
                     1),
        stream.get(), args));
    TF_ASSERT_OK(stream->Memcpy(output.data(), output_device,
                                output.size() * sizeof(output[0])));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    const uint16_t expected = Bfloat16Bits(static_cast<float>(k));
    for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_EQ(output[i], expected)
          << kernel_name << " m=" << m << " k=" << k << " n=" << n
          << " output[" << i << "]";
    }
  }

  void RunQmvFastGoldenCase() {
    constexpr int32_t kM = 1;
    constexpr int32_t kK = 1024;
    constexpr int32_t kN = 16;
    constexpr size_t kCanaryElements = 4;
    constexpr uint16_t kCanary = 0x7fc1;
    static_assert(kK % 512 == 0);
    static_assert(kN % 8 == 0);

    // Every group varies activations, nibble positions, signs, and its E4M3
    // scale. Block one is deliberately asymmetric with block zero so a broken
    // 512-value weight/scale/input step cannot replay the first block and pass.
    constexpr float kActivationPattern[16] = {
        1.0f,   -0.5f, 2.0f,  -1.0f, 0.25f, -2.0f, 0.5f,  0.0f,
        -0.25f, 1.5f,  -1.5f, 4.0f,  -4.0f, 3.0f,  -3.0f, 0.75f};
    constexpr uint8_t kWeightPattern[16] = {0x1, 0xa, 0x3, 0xc, 0x5, 0xe,
                                            0x7, 0x9, 0x2, 0xb, 0x4, 0xd,
                                            0x6, 0xf, 0x0, 0x8};
    constexpr uint8_t kScalePattern[5] = {0x18, 0x20, 0x28, 0x30, 0x38};

    std::vector<uint16_t> x(kK);
    std::vector<uint8_t> w(static_cast<size_t>(kN) * kK / 2, 0);
    std::vector<uint8_t> scales(static_cast<size_t>(kN) * kK / 16);
    for (int32_t k = 0; k < kK; ++k) {
      const int32_t block = k / 512;
      const int32_t group_in_block = (k % 512) / 16;
      const int32_t lane = k % 16;
      const int32_t activation_index =
          (lane + 3 * group_in_block + 5 * block) % 16;
      x[k] = Bfloat16Bits(kActivationPattern[activation_index]);
      for (int32_t n = 0; n < kN; ++n) {
        const int32_t weight_index =
            (lane + 3 * n + 5 * group_in_block + 7 * block) % 16;
        const uint8_t code = kWeightPattern[weight_index];
        const size_t byte_index = static_cast<size_t>(n) * kK / 2 + k / 2;
        w[byte_index] |= code << (4 * (k & 1));
      }
    }
    for (int32_t n = 0; n < kN; ++n) {
      for (int32_t group = 0; group < kK / 16; ++group) {
        const int32_t block = group / 32;
        const int32_t group_in_block = group % 32;
        scales[static_cast<size_t>(n) * (kK / 16) + group] =
            kScalePattern[(n + 2 * group_in_block + 3 * block) % 5];
      }
    }

    std::array<uint16_t, kN> golden;
    for (int32_t n = 0; n < kN; ++n) {
      float sum = 0.0f;
      for (int32_t k = 0; k < kK; ++k) {
        const uint8_t packed = w[static_cast<size_t>(n) * kK / 2 + k / 2];
        const uint8_t weight_bits = (packed >> (4 * (k & 1))) & 0xf;
        const uint8_t scale_bits =
            scales[static_cast<size_t>(n) * (kK / 16) + k / 16];
        sum += Bfloat16Value(x[k]) * DecodeFp4E2m1(weight_bits) *
               DecodeFp8E4m3(scale_bits);
      }
      golden[n] = RoundedBfloat16Bits(sum);
    }

    std::vector<uint16_t> output(kN + kCanaryElements, kCanary);
    const int32_t dims[4] = {kM, kK, kN, kK / 16};
    se::DeviceAddressBase w_device =
        AllocateAndCopy(w.data(), w.size() * sizeof(w[0]));
    se::DeviceAddressBase scales_device =
        AllocateAndCopy(scales.data(), scales.size() * sizeof(scales[0]));
    se::DeviceAddressBase x_device =
        AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
    se::DeviceAddressBase output_device =
        AllocateAndCopy(output.data(), output.size() * sizeof(output[0]));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> kernel,
        metal_executor_->LoadKernelWithConstants(
            qmv_metallib_, "nvfp4_qmv_fast", /*arity=*/5, {}));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                            executor_->CreateStream());
    se::KernelArgsPackedArray args(/*num_args=*/5);
    args.add_argument(w_device);
    args.add_argument(scales_device);
    args.add_argument(x_device);
    args.add_argument(output_device);
    args.add_argument(dims);
    TF_ASSERT_OK(kernel->Launch(se::ThreadDim(32, 2, 1),
                                se::BlockDim(kM, kN / 8, 1), stream.get(),
                                args));
    TF_ASSERT_OK(stream->Memcpy(output.data(), output_device,
                                output.size() * sizeof(output[0])));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    for (int32_t n = 0; n < kN; ++n) {
      EXPECT_EQ(output[n], golden[n]) << "output[" << n << "]";
    }
    for (size_t i = kN; i < output.size(); ++i) {
      EXPECT_EQ(output[i], kCanary) << "output canary[" << i << "]";
    }
  }

  void RunSplitKCase(int32_t m, int32_t k, int32_t n, int32_t split_k) {
    ASSERT_EQ(k % split_k, 0);
    ASSERT_EQ((k / split_k) % kNvfp4GroupSize, 0);
    const uint32_t rows = static_cast<uint32_t>(m);
    ASSERT_GT(rows, 0);
    constexpr size_t kCanaryElements = 8;
    constexpr uint16_t kCanary = 0x7fc1;

    std::vector<uint16_t> x(static_cast<size_t>(m) * k, Bfloat16Bits(1.0f));
    std::vector<uint8_t> w(static_cast<size_t>(n) * k / 2, 0x22);
    std::vector<uint8_t> scales(static_cast<size_t>(n) * k / 16, 0x38);
    std::vector<uint16_t> intermediate(
        static_cast<size_t>(split_k) * rows * n + kCanaryElements, kCanary);
    std::vector<uint16_t> output(static_cast<size_t>(m) * n + kCanaryElements,
                                 kCanary);
    const int32_t dims[4] = {m, k, n, k / 16};
    const uint32_t control[4] = {rows, static_cast<uint32_t>(split_k),
                                 static_cast<uint32_t>(k / split_k),
                                 rows * n};

    se::DeviceAddressBase x_device =
        AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
    se::DeviceAddressBase w_device =
        AllocateAndCopy(w.data(), w.size() * sizeof(w[0]));
    se::DeviceAddressBase scales_device =
        AllocateAndCopy(scales.data(), scales.size() * sizeof(scales[0]));
    se::DeviceAddressBase intermediate_device = AllocateAndCopy(
        intermediate.data(), intermediate.size() * sizeof(intermediate[0]));
    se::DeviceAddressBase output_device =
        AllocateAndCopy(output.data(), output.size() * sizeof(output[0]));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> split_kernel,
        metal_executor_->LoadKernelWithConstants(metallib_,
                                                 n % kNvfp4SplitkBN == 0
                                                     ? "nvfp4_qmm_t_splitk_alN"
                                                     : "nvfp4_qmm_t_splitk",
                                                 /*arity=*/6, {}));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> sum_kernel,
        metal_executor_->LoadKernelWithConstants(metallib_, "nvfp4_splitk_sum",
                                                 /*arity=*/3, {}));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                            executor_->CreateStream());

    se::KernelArgsPackedArray split_args(/*num_args=*/6);
    split_args.add_argument(x_device);
    split_args.add_argument(w_device);
    split_args.add_argument(scales_device);
    split_args.add_argument(intermediate_device);
    split_args.add_argument(dims);
    split_args.add_argument(control);
    TF_ASSERT_OK(split_kernel->Launch(
        se::ThreadDim(32, 2, 2),
        se::BlockDim(
            static_cast<uint64_t>((n + kNvfp4SplitkBN - 1) / kNvfp4SplitkBN),
            static_cast<uint64_t>((rows + kNvfp4SplitkBM - 1) /
                                  kNvfp4SplitkBM),
            static_cast<uint64_t>(split_k)),
        stream.get(), split_args));

    se::KernelArgsPackedArray sum_args(/*num_args=*/3);
    sum_args.add_argument(intermediate_device);
    sum_args.add_argument(output_device);
    sum_args.add_argument(control);
    const uint64_t mn = static_cast<uint64_t>(rows) * n;
    TF_ASSERT_OK(sum_kernel->Launch(se::ThreadDim(256, 1, 1),
                                    se::BlockDim((mn + 255) / 256, 1, 1),
                                    stream.get(), sum_args));
    TF_ASSERT_OK(stream->Memcpy(intermediate.data(), intermediate_device,
                                intermediate.size() * sizeof(intermediate[0])));
    TF_ASSERT_OK(stream->Memcpy(output.data(), output_device,
                                output.size() * sizeof(output[0])));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    const uint16_t partial_expected =
        Bfloat16Bits(static_cast<float>(k / split_k));
    const size_t intermediate_elements =
        static_cast<size_t>(split_k) * rows * n;
    for (size_t i = 0; i < intermediate_elements; ++i) {
      EXPECT_EQ(intermediate[i], partial_expected) << "intermediate[" << i
                                                   << "]";
    }
    for (size_t i = intermediate_elements; i < intermediate.size(); ++i) {
      EXPECT_EQ(intermediate[i], kCanary) << "intermediate canary[" << i << "]";
    }
    const uint16_t expected = Bfloat16Bits(static_cast<float>(k));
    const size_t output_elements = static_cast<size_t>(m) * n;
    for (size_t i = 0; i < output_elements; ++i) {
      EXPECT_EQ(output[i], expected) << "output[" << i << "]";
    }
    for (size_t i = output_elements; i < output.size(); ++i) {
      EXPECT_EQ(output[i], kCanary) << "output canary[" << i << "]";
    }
  }

  void RunSplitKPartition176GoldenCase() {
    constexpr int32_t kM = 2;
    constexpr int32_t kK = 704;
    constexpr int32_t kN = 2816;
    constexpr int32_t kSplitK = 4;
    constexpr int32_t kPartition = kK / kSplitK;
    constexpr size_t kCanaryElements = 8;
    constexpr uint16_t kCanary = 0x7fc1;
    static_assert(kPartition == 176);
    ASSERT_EQ(ComputeNvfp4QmmSplitK(16, kN, kK), kSplitK);

    constexpr float kXPattern[8] = {0.5f, -1.0f, 2.0f, -0.5f,
                                    1.0f, -2.0f, 4.0f, -4.0f};
    constexpr uint8_t kWeightPattern[16] = {0x1, 0xa, 0x3, 0xc, 0x5, 0xe,
                                            0x7, 0x9, 0x2, 0xb, 0x4, 0xd,
                                            0x6, 0xf, 0x0, 0x8};
    constexpr uint8_t kScalePattern[4] = {0x28, 0x30, 0x38, 0x40};

    std::vector<uint16_t> x(static_cast<size_t>(kM) * kK);
    for (int32_t m = 0; m < kM; ++m) {
      for (int32_t k = 0; k < kK; ++k) {
        x[static_cast<size_t>(m) * kK + k] =
            Bfloat16Bits(kXPattern[(k + 3 * m + 5 * (k / kPartition)) % 8]);
      }
    }

    std::vector<uint8_t> w(static_cast<size_t>(kN) * kK / 2, 0);
    for (int32_t n = 0; n < kN; ++n) {
      for (int32_t k = 0; k < kK; ++k) {
        const int32_t partition = k / kPartition;
        const int32_t group = k / kNvfp4GroupSize;
        const uint8_t code =
            kWeightPattern[(k + 3 * n + 5 * group + 7 * partition) % 16];
        const size_t byte = static_cast<size_t>(n) * kK / 2 + k / 2;
        w[byte] |= code << (4 * (k & 1));
      }
    }

    std::vector<uint8_t> scales(static_cast<size_t>(kN) * kK / 16);
    for (int32_t n = 0; n < kN; ++n) {
      for (int32_t group = 0; group < kK / 16; ++group) {
        const int32_t partition = group / (kPartition / 16);
        scales[static_cast<size_t>(n) * (kK / 16) + group] =
            kScalePattern[(n + group + 2 * partition) % 4];
      }
    }

    const size_t plane_elements = static_cast<size_t>(kM) * kN;
    std::vector<uint16_t> partial_golden(static_cast<size_t>(kSplitK) *
                                         plane_elements);
    std::vector<uint16_t> output_golden(plane_elements);
    for (int32_t m = 0; m < kM; ++m) {
      for (int32_t n = 0; n < kN; ++n) {
        float final_sum = 0.0f;
        for (int32_t part = 0; part < kSplitK; ++part) {
          float partial = 0.0f;
          for (int32_t k = part * kPartition; k < (part + 1) * kPartition;
               ++k) {
            const uint8_t packed = w[static_cast<size_t>(n) * kK / 2 + k / 2];
            const uint8_t weight_bits = (packed >> (4 * (k & 1))) & 0xf;
            const uint8_t scale_bits =
                scales[static_cast<size_t>(n) * (kK / 16) + k / 16];
            partial += Bfloat16Value(x[static_cast<size_t>(m) * kK + k]) *
                       DecodeFp4E2m1(weight_bits) * DecodeFp8E4m3(scale_bits);
          }
          const uint16_t partial_bits = RoundedBfloat16Bits(partial);
          partial_golden[static_cast<size_t>(part) * plane_elements +
                         static_cast<size_t>(m) * kN + n] = partial_bits;
          final_sum += Bfloat16Value(partial_bits);
        }
        output_golden[static_cast<size_t>(m) * kN + n] =
            RoundedBfloat16Bits(final_sum);
      }
    }

    std::vector<uint16_t> intermediate(partial_golden.size() + kCanaryElements,
                                       kCanary);
    std::vector<uint16_t> output(output_golden.size() + kCanaryElements,
                                 kCanary);
    const int32_t dims[4] = {kM, kK, kN, kK / 16};
    const uint32_t control[4] = {kM, kSplitK, kPartition,
                                 static_cast<uint32_t>(plane_elements)};

    se::DeviceAddressBase x_device =
        AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
    se::DeviceAddressBase w_device =
        AllocateAndCopy(w.data(), w.size() * sizeof(w[0]));
    se::DeviceAddressBase scales_device =
        AllocateAndCopy(scales.data(), scales.size() * sizeof(scales[0]));
    se::DeviceAddressBase intermediate_device = AllocateAndCopy(
        intermediate.data(), intermediate.size() * sizeof(intermediate[0]));
    se::DeviceAddressBase output_device =
        AllocateAndCopy(output.data(), output.size() * sizeof(output[0]));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> split_kernel,
        metal_executor_->LoadKernelWithConstants(
            metallib_, "nvfp4_qmm_t_splitk_alN", /*arity=*/6, {}));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> sum_kernel,
        metal_executor_->LoadKernelWithConstants(metallib_, "nvfp4_splitk_sum",
                                                 /*arity=*/3, {}));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                            executor_->CreateStream());

    se::KernelArgsPackedArray split_args(/*num_args=*/6);
    split_args.add_argument(x_device);
    split_args.add_argument(w_device);
    split_args.add_argument(scales_device);
    split_args.add_argument(intermediate_device);
    split_args.add_argument(dims);
    split_args.add_argument(control);
    TF_ASSERT_OK(split_kernel->Launch(
        se::ThreadDim(32, 2, 2),
        se::BlockDim((kN + kNvfp4SplitkBN - 1) / kNvfp4SplitkBN,
                     (kM + kNvfp4SplitkBM - 1) / kNvfp4SplitkBM, kSplitK),
        stream.get(), split_args));

    se::KernelArgsPackedArray sum_args(/*num_args=*/3);
    sum_args.add_argument(intermediate_device);
    sum_args.add_argument(output_device);
    sum_args.add_argument(control);
    TF_ASSERT_OK(
        sum_kernel->Launch(se::ThreadDim(256, 1, 1),
                           se::BlockDim((plane_elements + 255) / 256, 1, 1),
                           stream.get(), sum_args));
    TF_ASSERT_OK(stream->Memcpy(intermediate.data(), intermediate_device,
                                intermediate.size() * sizeof(intermediate[0])));
    TF_ASSERT_OK(stream->Memcpy(output.data(), output_device,
                                output.size() * sizeof(output[0])));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    for (size_t i = 0; i < partial_golden.size(); ++i) {
      EXPECT_EQ(intermediate[i], partial_golden[i])
          << "partition=" << i / plane_elements
          << " plane_element=" << i % plane_elements;
    }
    for (size_t i = partial_golden.size(); i < intermediate.size(); ++i) {
      EXPECT_EQ(intermediate[i], kCanary) << "staging canary[" << i << "]";
    }
    for (size_t i = 0; i < output_golden.size(); ++i) {
      EXPECT_EQ(output[i], output_golden[i]) << "output[" << i << "]";
    }
    for (size_t i = output_golden.size(); i < output.size(); ++i) {
      EXPECT_EQ(output[i], kCanary) << "output canary[" << i << "]";
    }
  }

  void RunGatherCase(int32_t r, int32_t e, int32_t k, int32_t n) {
    ASSERT_EQ(k % kNvfp4GroupSize, 0);
    ASSERT_EQ(r % e, 0);
    constexpr size_t kCanaryElements = 4;
    constexpr uint16_t kCanary = 0x7fc1;

    std::vector<uint16_t> x(static_cast<size_t>(r) * k, Bfloat16Bits(1.0f));
    std::vector<uint8_t> w(static_cast<size_t>(e) * n * k / 2, 0x22);
    std::vector<uint8_t> scales(static_cast<size_t>(e) * n * k / 16, 0x38);
    std::vector<uint32_t> expert_ids(r);
    for (int32_t i = 0; i < r; ++i) expert_ids[i] = i / (r / e);
    const size_t output_elements = static_cast<size_t>(r) * n;
    std::vector<uint16_t> output(output_elements + kCanaryElements, kCanary);
    const int32_t dims[4] = {r, n, k, 1};

    se::DeviceAddressBase x_device =
        AllocateAndCopy(x.data(), x.size() * sizeof(x[0]));
    se::DeviceAddressBase w_device =
        AllocateAndCopy(w.data(), w.size() * sizeof(w[0]));
    se::DeviceAddressBase scales_device =
        AllocateAndCopy(scales.data(), scales.size() * sizeof(scales[0]));
    se::DeviceAddressBase ids_device = AllocateAndCopy(
        expert_ids.data(), expert_ids.size() * sizeof(expert_ids[0]));
    se::DeviceAddressBase output_device =
        AllocateAndCopy(output.data(), output.size() * sizeof(output[0]));
    ASSERT_NE(x_device.opaque(), nullptr);
    ASSERT_NE(w_device.opaque(), nullptr);
    ASSERT_NE(scales_device.opaque(), nullptr);
    ASSERT_NE(ids_device.opaque(), nullptr);
    ASSERT_NE(output_device.opaque(), nullptr);

    using FC = se::metal::MetalFunctionConstant;
    const FC constants[] = {
        {200, FC::Kind::kBool, 0},
        {201, FC::Kind::kBool, n % 32 == 0},
        {202, FC::Kind::kBool, k % 32 == 0},
        // No trailing per-expert global scale: six operands, no buffer(6).
        {440, FC::Kind::kBool, 0},
    };
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> kernel,
        metal_executor_->LoadKernelWithConstants(
            metallib_, "nvfp4_gather_qmm_rhs", /*arity=*/6, constants));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                            executor_->CreateStream());

    se::KernelArgsPackedArray args(/*num_args=*/6);
    args.add_argument(x_device);
    args.add_argument(w_device);
    args.add_argument(scales_device);
    args.add_argument(ids_device);
    args.add_argument(output_device);
    args.add_argument(dims);
    TF_ASSERT_OK(
        kernel->Launch(se::ThreadDim(32, 2, 1),
                       se::BlockDim(static_cast<uint64_t>((n + 31) / 32),
                                    static_cast<uint64_t>((r + 15) / 16), 1),
                       stream.get(), args));
    TF_ASSERT_OK(stream->Memcpy(output.data(), output_device,
                                output.size() * sizeof(output[0])));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    const uint16_t expected = Bfloat16Bits(static_cast<float>(k));
    for (size_t i = 0; i < output_elements; ++i) {
      EXPECT_EQ(output[i], expected) << "r=" << r << " e=" << e << " k=" << k
                                     << " n=" << n << " output[" << i << "]";
    }
    for (size_t i = output_elements; i < output.size(); ++i) {
      EXPECT_EQ(output[i], kCanary) << "output canary[" << i << "]";
    }
  }

  void RunUnsupportedArgsortExpertCountCase() {
    constexpr int32_t kR = 4;
    constexpr int32_t kE = 257;
    constexpr int32_t kSentinel = -7777777;
    const std::vector<int32_t> expert_ids = {0, 256, -1, 1};
    std::vector<int32_t> order(kR, kSentinel);
    std::vector<int32_t> sorted_ids(kR, kSentinel);

    se::DeviceAddressBase ids_device = AllocateAndCopy(
        expert_ids.data(), expert_ids.size() * sizeof(expert_ids[0]));
    se::DeviceAddressBase order_device =
        AllocateAndCopy(order.data(), order.size() * sizeof(order[0]));
    se::DeviceAddressBase sorted_ids_device = AllocateAndCopy(
        sorted_ids.data(), sorted_ids.size() * sizeof(sorted_ids[0]));

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<uint8_t> argsort_metallib,
        CompileMetalSourceToMetallibCached(get_moe_argsort()));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> argsort_kernel,
        metal_executor_->LoadKernelWithConstants(
            argsort_metallib, "moe_argsort", /*arity=*/4, {}));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                            executor_->CreateStream());

    const int32_t argsort_dims[4] = {kR, kE, 0, 0};
    se::KernelArgsPackedArray argsort_args(/*num_args=*/4);
    argsort_args.add_argument(ids_device);
    argsort_args.add_argument(order_device);
    argsort_args.add_argument(sorted_ids_device);
    argsort_args.add_argument(argsort_dims);
    TF_ASSERT_OK(argsort_kernel->Launch(se::ThreadDim(256, 1, 1),
                                        se::BlockDim(1, 1, 1), stream.get(),
                                        argsort_args));

    TF_ASSERT_OK(stream->Memcpy(order.data(), order_device,
                                order.size() * sizeof(order[0])));
    TF_ASSERT_OK(stream->Memcpy(sorted_ids.data(), sorted_ids_device,
                                sorted_ids.size() * sizeof(sorted_ids[0])));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    EXPECT_TRUE(std::all_of(order.begin(), order.end(),
                            [](int32_t value) { return value == kSentinel; }));
    EXPECT_TRUE(std::all_of(sorted_ids.begin(), sorted_ids.end(),
                            [](int32_t value) { return value == kSentinel; }));
  }

  void RunOddWidthPermuteCase() {
    constexpr int32_t kR = 3, kW = 7, kE = 2;
    constexpr uint16_t kSentinel = 0x7fc1;
    std::vector<uint16_t> src(static_cast<size_t>(kR) * kW);
    for (size_t i = 0; i < src.size(); ++i) {
      src[i] = Bfloat16Bits(static_cast<float>(i + 1));
    }
    // Row two carries an out-of-range order entry: gather must zero it instead
    // of dereferencing it, and scatter must not write through it.
    const std::array<int32_t, kR> order = {1, 0, -777};
    const std::array<int32_t, kR> experts = {0, 1, 0};
    std::vector<uint16_t> gathered(src.size() + 4, kSentinel);
    std::vector<uint16_t> scattered(src.size() + 4, kSentinel);

    se::DeviceAddressBase src_device =
        AllocateAndCopy(src.data(), src.size() * sizeof(src[0]));
    se::DeviceAddressBase order_device =
        AllocateAndCopy(order.data(), sizeof(order));
    se::DeviceAddressBase experts_device =
        AllocateAndCopy(experts.data(), sizeof(experts));
    se::DeviceAddressBase gathered_device =
        AllocateAndCopy(gathered.data(), gathered.size() * sizeof(gathered[0]));
    se::DeviceAddressBase scattered_device = AllocateAndCopy(
        scattered.data(), scattered.size() * sizeof(scattered[0]));

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<uint8_t> lib,
        CompileMetalSourceToMetallibCached(get_permute_rows()));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> gather,
        metal_executor_->LoadKernelWithConstants(lib, "gather_rows",
                                                 /*arity=*/5, {}));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::Kernel> scatter,
        metal_executor_->LoadKernelWithConstants(lib, "scatter_rows",
                                                 /*arity=*/5, {}));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                            executor_->CreateStream());
    const int32_t dims[4] = {kR, kW, kE, 0};
    constexpr uint64_t kWordGroups = (kW + 3) / 4;

    se::KernelArgsPackedArray gather_args(/*num_args=*/5);
    gather_args.add_argument(src_device);
    gather_args.add_argument(order_device);
    gather_args.add_argument(gathered_device);
    gather_args.add_argument(dims);
    gather_args.add_argument(experts_device);
    TF_ASSERT_OK(gather->Launch(se::ThreadDim(64, 1, 1),
                                se::BlockDim((kWordGroups + 63) / 64, kR, 1),
                                stream.get(), gather_args));

    se::KernelArgsPackedArray scatter_args(/*num_args=*/5);
    scatter_args.add_argument(gathered_device);
    scatter_args.add_argument(order_device);
    scatter_args.add_argument(scattered_device);
    scatter_args.add_argument(dims);
    scatter_args.add_argument(experts_device);
    TF_ASSERT_OK(scatter->Launch(se::ThreadDim(64, 1, 1),
                                 se::BlockDim((kWordGroups + 63) / 64, kR, 1),
                                 stream.get(), scatter_args));
    TF_ASSERT_OK(stream->Memcpy(gathered.data(), gathered_device,
                                gathered.size() * sizeof(gathered[0])));
    TF_ASSERT_OK(stream->Memcpy(scattered.data(), scattered_device,
                                scattered.size() * sizeof(scattered[0])));
    TF_ASSERT_OK(stream->BlockHostUntilDone());

    for (int32_t w = 0; w < kW; ++w) {
      EXPECT_EQ(gathered[w], src[kW + w]);
      EXPECT_EQ(gathered[kW + w], src[w]);
      // The invalid order entry gathers a zero row.
      EXPECT_EQ(gathered[2 * kW + w], 0);
      EXPECT_EQ(scattered[w], src[w]);
      EXPECT_EQ(scattered[kW + w], src[kW + w]);
      // ... and never scatters through the invalid destination.
      EXPECT_EQ(scattered[2 * kW + w], kSentinel);
    }
    for (size_t i = static_cast<size_t>(kR) * kW; i < gathered.size(); ++i) {
      EXPECT_EQ(gathered[i], kSentinel);
      EXPECT_EQ(scattered[i], kSentinel);
    }
  }

  se::StreamExecutor* executor_ = nullptr;
  se::metal::MetalExecutor* metal_executor_ = nullptr;
  std::vector<uint8_t> metallib_;
  std::vector<uint8_t> qmv_metallib_;
  std::vector<se::DeviceAddressBase> allocations_;
};

TEST_F(MetalNvfp4KernelTest, DenseQmmHandlesPartialMAndN) {
  RunDenseCase(/*m=*/17, /*k=*/32, /*n=*/63, "nvfp4_qmm_t");
}

TEST_F(MetalNvfp4KernelTest, DenseQmmHandlesGroupAlignedKTail) {
  RunDenseCase(/*m=*/17, /*k=*/48, /*n=*/63, "nvfp4_qmm_t");
}

TEST_F(MetalNvfp4KernelTest, DenseQmmAlignedNHandlesKTail) {
  RunDenseCase(/*m=*/16, /*k=*/48, /*n=*/64, "nvfp4_qmm_t_alN");
}

TEST_F(MetalNvfp4KernelTest, DenseQmvHandlesPartialN) {
  RunQmvCase(/*m=*/1, /*k=*/32, /*n=*/9, "nvfp4_qmv", /*vecs_per_tg=*/1);
}

TEST_F(MetalNvfp4KernelTest, DenseQmvFastMatchesNonuniformGolden) {
  RunQmvFastGoldenCase();
}

TEST_F(MetalNvfp4KernelTest, DenseQmvWideHandlesPartialVectorTile) {
  RunQmvCase(/*m=*/5, /*k=*/32, /*n=*/13, "nvfp4_qmv_wide_5",
             /*vecs_per_tg=*/5);
}

TEST_F(MetalNvfp4KernelTest, DenseSplitKSumsPartialPlanes) {
  RunSplitKCase(/*m=*/17, /*k=*/64, /*n=*/63, /*split_k=*/2);
}

TEST_F(MetalNvfp4KernelTest, DenseSplitKHandlesPartition176PackingAndScales) {
  RunSplitKPartition176GoldenCase();
}

TEST_F(MetalNvfp4KernelTest, GatherQmmHandlesPartialN) {
  RunGatherCase(/*r=*/16, /*e=*/2, /*k=*/32, /*n=*/35);
}

TEST_F(MetalNvfp4KernelTest, GatherQmmHandlesGroupAlignedKTail) {
  RunGatherCase(/*r=*/16, /*e=*/2, /*k=*/16, /*n=*/64);
}

TEST_F(MetalNvfp4KernelTest, GatherQmmHandlesPartialKAndN) {
  RunGatherCase(/*r=*/16, /*e=*/2, /*k=*/48, /*n=*/37);
}

TEST_F(MetalNvfp4KernelTest, ArgsortRejectsMoreThan256ExpertsSafely) {
  RunUnsupportedArgsortExpertCountCase();
}

TEST_F(MetalNvfp4KernelTest, MoePermuteHandlesOddWidthTail) {
  RunOddWidthPermuteCase();
}

}  // namespace
}  // namespace xla::gpu
