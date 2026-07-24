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

#include "xla/stream_executor/musa/mublas_shim/mublas_shim_abi.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <gtest/gtest.h>
#include "xla/stream_executor/musa/mublas_shim/mublas_shim_float16.h"

namespace {

using ::stream_executor::musa::mublas_shim_internal::Float32ToFloat16Bits;

float Float32FromBits(uint32_t bits) {
  float value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static_assert(sizeof(XlaMusaMuBlasStatus) == 4);
static_assert(sizeof(XlaMusaMuBlasDataType) == 4);
static_assert(sizeof(XlaMusaMuBlasOperation) == 4);
static_assert(sizeof(XlaMusaMuBlasCapabilities) == 8);
static_assert(std::is_standard_layout_v<XlaMusaMuBlasApiV1>);

TEST(MuBlasShimAbiTest, HasStableElf64Layout) {
  static_assert(sizeof(void*) == 8, "the qualified MUSA host ABI is ELF64");
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV1, struct_size), 0);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV1, abi_version), 4);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV1, capabilities), 8);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV1, create), 16);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV1, destroy), 24);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV1, set_stream), 32);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV1, get_version), 40);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV1, gemm), 48);
  EXPECT_EQ(sizeof(XlaMusaMuBlasApiV1), 56);
  EXPECT_EQ(XLA_MUSA_MUBLAS_API_V1_MIN_STRUCT_SIZE, sizeof(XlaMusaMuBlasApiV1));
}

TEST(MuBlasShimAbiTest, CapabilityBitsAreAppendOnlyAndDisjoint) {
  constexpr uint64_t kRequiredV1Capabilities =
      XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F16 |
      XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F32 | XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F64;
  constexpr uint64_t kAllKnownCapabilities =
      XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F16 |
      XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F32 |
      XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F64 |
      XLA_MUSA_MUBLAS_CAPABILITY_GEMM_BF16;
  EXPECT_EQ(kAllKnownCapabilities, UINT64_C(0xf));
  EXPECT_EQ(XLA_MUSA_MUBLAS_CAPABILITIES_V1, kRequiredV1Capabilities);
  EXPECT_EQ(XLA_MUSA_MUBLAS_CAPABILITIES_V1, UINT64_C(0x7));
  EXPECT_EQ(
      XLA_MUSA_MUBLAS_CAPABILITIES_V1 & XLA_MUSA_MUBLAS_CAPABILITY_GEMM_BF16,
      0);
  EXPECT_EQ(XLA_MUSA_MUBLAS_ABI_VERSION_1, UINT32_C(1));
}

TEST(MuBlasShimAbiTest, ConvertsF32ScalarsToF16WithIeeeRounding) {
  EXPECT_EQ(Float32ToFloat16Bits(1.0f), 0x3c00);
  EXPECT_EQ(Float32ToFloat16Bits(-2.0f), 0xc000);
  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0xb7f84001)), 0x81f1);

  // Exact halfway values select the result with an even low significand bit.
  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0x3f801000)), 0x3c00);
  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0x3f803000)), 0x3c02);
  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0x33000000)), 0x0000);
  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0x33000001)), 0x0001);
  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0x33c00000)), 0x0002);

  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0x7f800000)), 0x7c00);
  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0xff800000)), 0xfc00);
  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0x7fc12345)), 0x7e09);
  EXPECT_EQ(Float32ToFloat16Bits(Float32FromBits(0x7f800001)), 0x7c01);
}

}  // namespace
