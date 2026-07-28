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

#include "xla/stream_executor/musa/mufft_shim/mufft_shim_abi.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

static_assert(sizeof(XlaMusaMuFftStatus) == 4);
static_assert(sizeof(XlaMusaMuFftType) == 4);
static_assert(sizeof(XlaMusaMuFftDirection) == 4);
static_assert(sizeof(XlaMusaMuFftCapabilities) == 8);
static_assert(sizeof(XlaMusaMuFftVersion) == 12);
static_assert(std::is_standard_layout_v<XlaMusaMuFftVersion>);
static_assert(std::is_standard_layout_v<XlaMusaMuFftApiV1>);

TEST(MuFftShimAbiTest, HasStableElf64Layout) {
  static_assert(sizeof(void*) == 8, "the qualified MUSA host ABI is ELF64");
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, struct_size), 0);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, abi_version), 4);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, capabilities), 8);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, get_version), 16);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, create), 24);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, destroy), 32);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, make_plan_many), 40);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, set_work_area), 48);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, set_stream), 56);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, exec_c2c), 64);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, exec_r2c), 72);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, exec_c2r), 80);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, exec_z2z), 88);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, exec_d2z), 96);
  EXPECT_EQ(offsetof(XlaMusaMuFftApiV1, exec_z2d), 104);
  EXPECT_EQ(sizeof(XlaMusaMuFftApiV1), 112);
  EXPECT_EQ(XLA_MUSA_MUFFT_API_V1_MIN_STRUCT_SIZE, sizeof(XlaMusaMuFftApiV1));
}

TEST(MuFftShimAbiTest, CapabilityBitsAreAppendOnlyAndComplete) {
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITY_C2C, UINT64_C(1));
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITY_R2C, UINT64_C(2));
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITY_C2R, UINT64_C(4));
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITY_Z2Z, UINT64_C(8));
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITY_D2Z, UINT64_C(16));
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITY_Z2D, UINT64_C(32));
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITY_RANK_1_TO_3, UINT64_C(64));
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITY_EXTERNAL_WORKSPACE, UINT64_C(128));
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITY_STREAM_BINDING, UINT64_C(256));
  EXPECT_EQ(XLA_MUSA_MUFFT_CAPABILITIES_V1, UINT64_C(511));
}

TEST(MuFftShimAbiTest, TypesDirectionsAndStatusesAreNormalized) {
  EXPECT_EQ(XLA_MUSA_MUFFT_ABI_VERSION_1, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUFFT_TYPE_C2C, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUFFT_TYPE_R2C, UINT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUFFT_TYPE_C2R, UINT32_C(3));
  EXPECT_EQ(XLA_MUSA_MUFFT_TYPE_Z2Z, UINT32_C(4));
  EXPECT_EQ(XLA_MUSA_MUFFT_TYPE_D2Z, UINT32_C(5));
  EXPECT_EQ(XLA_MUSA_MUFFT_TYPE_Z2D, UINT32_C(6));
  EXPECT_EQ(XLA_MUSA_MUFFT_DIRECTION_FORWARD, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUFFT_DIRECTION_INVERSE, UINT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUFFT_STATUS_SUCCESS, INT32_C(0));
  EXPECT_EQ(XLA_MUSA_MUFFT_STATUS_VENDOR_ERROR, INT32_C(6));
}

}  // namespace
