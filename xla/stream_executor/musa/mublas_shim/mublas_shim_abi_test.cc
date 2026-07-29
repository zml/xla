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
static_assert(sizeof(XlaMusaMuBlasAlgorithm) == 4);
static_assert(sizeof(XlaMusaMuBlasAdvancedCapabilities) == 8);
static_assert(sizeof(XlaMusaMuBlasScalType) == 4);
static_assert(sizeof(XlaMusaMuBlasTrsmType) == 4);
static_assert(sizeof(XlaMusaMuBlasSide) == 4);
static_assert(sizeof(XlaMusaMuBlasFill) == 4);
static_assert(sizeof(XlaMusaMuBlasDiagonal) == 4);
static_assert(std::is_standard_layout_v<XlaMusaMuBlasApiV1>);
static_assert(std::is_standard_layout_v<XlaMusaMuBlasApiV2>);

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

TEST(MuBlasShimAbiTest, V2AppendsOptionalTailsWithoutChangingRequiredPrefix) {
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, struct_size), 0);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, abi_version), 4);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, capabilities), 8);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, advanced_capabilities), 16);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, create), 24);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, destroy), 32);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, set_stream), 40);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, get_version), 48);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, gemm), 56);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, set_atomics_mode), 64);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, gemm_with_algorithm), 72);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, gemm_batched), 80);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, gemm_strided_batched), 88);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, scal), 96);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, trsm), 104);
  EXPECT_EQ(offsetof(XlaMusaMuBlasApiV2, trsm_batched), 112);
  EXPECT_EQ(sizeof(XlaMusaMuBlasApiV2), 120);
  EXPECT_EQ(XLA_MUSA_MUBLAS_API_V2_MIN_STRUCT_SIZE, 96);
  EXPECT_EQ(XLA_MUSA_MUBLAS_API_V2_SCAL_STRUCT_SIZE, 104);
  EXPECT_EQ(XLA_MUSA_MUBLAS_API_V2_TRSM_STRUCT_SIZE, 112);
  EXPECT_EQ(XLA_MUSA_MUBLAS_API_V2_TRSM_BATCHED_STRUCT_SIZE,
            sizeof(XlaMusaMuBlasApiV2));
  EXPECT_EQ(sizeof(XlaMusaMuBlasApiV1), 56);
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
  EXPECT_EQ(XLA_MUSA_MUBLAS_ABI_VERSION_2, UINT32_C(2));
}

TEST(MuBlasShimAbiTest, V2AlgorithmsAndCapabilitiesAreNormalized) {
  EXPECT_EQ(XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT, UINT32_C(0));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_SET_ATOMICS_MODE, UINT64_C(1));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_GEMM_WITH_ALGORITHM,
            UINT64_C(2));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_GEMM_BATCHED, UINT64_C(4));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_GEMM_STRIDED_BATCHED,
            UINT64_C(8));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_ZERO_EXTERNAL_WORKSPACE,
            UINT64_C(16));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_TENSOR_OP_F32, UINT64_C(32));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_SCAL, UINT64_C(64));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_TRSM, UINT64_C(128));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_TRSM_BATCHED, UINT64_C(256));
  EXPECT_EQ(XLA_MUSA_MUBLAS_ADVANCED_CAPABILITIES_V2, UINT64_C(63));
}

TEST(MuBlasShimAbiTest, ScalRoutesAreNormalizedAndStable) {
  EXPECT_EQ(XLA_MUSA_MUBLAS_SCAL_TYPE_F32, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUBLAS_SCAL_TYPE_F64, UINT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUBLAS_SCAL_TYPE_C64, UINT32_C(3));
  EXPECT_EQ(XLA_MUSA_MUBLAS_SCAL_TYPE_C128, UINT32_C(4));
  EXPECT_EQ(XLA_MUSA_MUBLAS_SCAL_TYPE_C64_F32, UINT32_C(5));
  EXPECT_EQ(XLA_MUSA_MUBLAS_SCAL_TYPE_C128_F64, UINT32_C(6));
}

TEST(MuBlasShimAbiTest, TrsmRoutesAndOptionsAreNormalizedAndStable) {
  EXPECT_EQ(XLA_MUSA_MUBLAS_TRSM_TYPE_F32, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUBLAS_TRSM_TYPE_F64, UINT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUBLAS_TRSM_TYPE_C64, UINT32_C(3));
  EXPECT_EQ(XLA_MUSA_MUBLAS_TRSM_TYPE_C128, UINT32_C(4));
  EXPECT_EQ(XLA_MUSA_MUBLAS_SIDE_LEFT, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUBLAS_SIDE_RIGHT, UINT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUBLAS_FILL_UPPER, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUBLAS_FILL_LOWER, UINT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUBLAS_DIAGONAL_NON_UNIT, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUBLAS_DIAGONAL_UNIT, UINT32_C(2));
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
