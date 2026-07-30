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

#include "xla/stream_executor/musa/mudnn_shim/mudnn_shim_abi.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

static_assert(sizeof(XlaMusaMuDnnStatus) == 4);
static_assert(sizeof(XlaMusaMuDnnDataType) == 4);
static_assert(sizeof(XlaMusaMuDnnTensorFormat) == 4);
static_assert(sizeof(XlaMusaMuDnnConvolutionKind) == 4);
static_assert(sizeof(XlaMusaMuDnnAlgorithm) == 4);
static_assert(sizeof(XlaMusaMuDnnCapabilities) == 8);
static_assert(sizeof(XlaMusaMuDnnVersion) == 12);
static_assert(std::is_standard_layout_v<XlaMusaMuDnnVersion>);
static_assert(std::is_standard_layout_v<XlaMusaMuDnnWorkspaceAllocator>);
static_assert(std::is_standard_layout_v<XlaMusaMuDnnApiV1>);
static_assert(std::is_trivially_copyable_v<XlaMusaMuDnnVersion>);
static_assert(std::is_trivially_copyable_v<XlaMusaMuDnnWorkspaceAllocator>);
static_assert(std::is_trivially_copyable_v<XlaMusaMuDnnApiV1>);

TEST(MuDnnShimAbiTest, HasStableElf64Layout) {
  static_assert(sizeof(void*) == 8, "the qualified MUSA host ABI is ELF64");
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, struct_size), 0);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, abi_version), 4);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, capabilities), 8);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, get_version), 16);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, create_handle), 24);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, destroy_handle), 32);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, set_stream), 40);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, set_allow_tf32), 48);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, create_tensor), 56);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, destroy_tensor), 64);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, configure_tensor), 72);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, set_tensor_address), 80);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, create_convolution), 88);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, destroy_convolution), 96);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, configure_convolution), 104);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, get_recommended_algorithm), 112);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, get_workspace_size), 120);
  EXPECT_EQ(offsetof(XlaMusaMuDnnApiV1, convolve), 128);
  EXPECT_EQ(sizeof(XlaMusaMuDnnApiV1), 136);
  EXPECT_EQ(XLA_MUSA_MUDNN_API_V1_MIN_STRUCT_SIZE, sizeof(XlaMusaMuDnnApiV1));
}

TEST(MuDnnShimAbiTest, WorkspaceAllocatorHasStablePrefix) {
  EXPECT_EQ(offsetof(XlaMusaMuDnnWorkspaceAllocator, struct_size), 0);
  EXPECT_EQ(offsetof(XlaMusaMuDnnWorkspaceAllocator, reserved), 4);
  EXPECT_EQ(offsetof(XlaMusaMuDnnWorkspaceAllocator, user_data), 8);
  EXPECT_EQ(offsetof(XlaMusaMuDnnWorkspaceAllocator, allocate), 16);
  EXPECT_EQ(offsetof(XlaMusaMuDnnWorkspaceAllocator, release), 24);
  EXPECT_EQ(sizeof(XlaMusaMuDnnWorkspaceAllocator), 32);
  EXPECT_EQ(XLA_MUSA_MUDNN_WORKSPACE_ALLOCATOR_MIN_STRUCT_SIZE,
            sizeof(XlaMusaMuDnnWorkspaceAllocator));
}

TEST(MuDnnShimAbiTest, CapabilityBitsAreAppendOnlyAndComplete) {
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_F16, UINT64_C(1));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_BF16, UINT64_C(2));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_F32, UINT64_C(4));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_RANK_3_TO_5, UINT64_C(8));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_EXPLICIT_STRIDES, UINT64_C(16));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_FORWARD, UINT64_C(32));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_BACKWARD_DATA, UINT64_C(64));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_BACKWARD_FILTER,
            UINT64_C(128));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_GROUPED_CONVOLUTION, UINT64_C(256));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_PAD_STRIDE_DILATION, UINT64_C(512));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_RECOMMENDED_ALGORITHM, UINT64_C(1024));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_EXPLICIT_ALGORITHM, UINT64_C(2048));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_EXTERNAL_WORKSPACE, UINT64_C(4096));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_STREAM_BINDING, UINT64_C(8192));
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITY_ALLOW_TF32, UINT64_C(16384));
  // The BF16 type value remains reserved, but muDNN 2.8.0 convolution on the
  // qualified S80 stack does not advertise executable BF16 support.
  EXPECT_EQ(XLA_MUSA_MUDNN_CAPABILITIES_V1, UINT64_C(0x7ffd));
}

TEST(MuDnnShimAbiTest, TypesFormatsAndKindsAreNormalized) {
  EXPECT_EQ(XLA_MUSA_MUDNN_ABI_VERSION_1, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUDNN_DATA_TYPE_F16, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUDNN_DATA_TYPE_BF16, UINT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUDNN_DATA_TYPE_F32, UINT32_C(3));
  EXPECT_EQ(XLA_MUSA_MUDNN_TENSOR_FORMAT_UNKNOWN, UINT32_C(0));
  EXPECT_EQ(XLA_MUSA_MUDNN_TENSOR_FORMAT_NCW, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUDNN_TENSOR_FORMAT_NWC, UINT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW, UINT32_C(3));
  EXPECT_EQ(XLA_MUSA_MUDNN_TENSOR_FORMAT_NHWC, UINT32_C(4));
  EXPECT_EQ(XLA_MUSA_MUDNN_TENSOR_FORMAT_HWCN, UINT32_C(5));
  EXPECT_EQ(XLA_MUSA_MUDNN_TENSOR_FORMAT_NCDHW, UINT32_C(6));
  EXPECT_EQ(XLA_MUSA_MUDNN_TENSOR_FORMAT_NDHWC, UINT32_C(7));
  EXPECT_EQ(XLA_MUSA_MUDNN_TENSOR_FORMAT_DHWCN, UINT32_C(8));
  EXPECT_EQ(XLA_MUSA_MUDNN_CONVOLUTION_FORWARD, UINT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA, UINT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER, UINT32_C(3));
}

TEST(MuDnnShimAbiTest, AlgorithmsAndStatusesAreNormalized) {
  EXPECT_EQ(XLA_MUSA_MUDNN_ALGORITHM_RECOMMENDED, -1);
  EXPECT_EQ(XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM, 0);
  EXPECT_EQ(XLA_MUSA_MUDNN_ALGORITHM_DIRECT, 1);
  EXPECT_EQ(XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED, 2);
  EXPECT_EQ(XLA_MUSA_MUDNN_ALGORITHM_GEMM, 3);
  EXPECT_EQ(XLA_MUSA_MUDNN_STATUS_SUCCESS, INT32_C(0));
  EXPECT_EQ(XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT, INT32_C(1));
  EXPECT_EQ(XLA_MUSA_MUDNN_STATUS_NOT_SUPPORTED, INT32_C(2));
  EXPECT_EQ(XLA_MUSA_MUDNN_STATUS_OUT_OF_RANGE, INT32_C(3));
  EXPECT_EQ(XLA_MUSA_MUDNN_STATUS_RESOURCE_EXHAUSTED, INT32_C(4));
  EXPECT_EQ(XLA_MUSA_MUDNN_STATUS_FAILED_PRECONDITION, INT32_C(5));
  EXPECT_EQ(XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR, INT32_C(6));
}

}  // namespace
