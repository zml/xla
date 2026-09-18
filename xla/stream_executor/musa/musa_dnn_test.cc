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

#include "xla/stream_executor/musa/musa_dnn.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/musa/mudnn_shim/mudnn_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_mudnn_api.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;

struct TensorCall {
  void* tensor = nullptr;
  XlaMusaMuDnnDataType data_type = 0;
  XlaMusaMuDnnTensorFormat format = 0;
  std::vector<int64_t> dimensions;
  std::vector<int64_t> strides;
};

struct TensorAddressCall {
  void* tensor = nullptr;
  void* address = nullptr;
};

struct ConvolutionConfigCall {
  int32_t spatial_rank = 0;
  std::vector<int64_t> padding;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  int64_t group_count = 0;
};

struct WorkspaceCall {
  XlaMusaMuDnnConvolutionKind kind = 0;
  void* output = nullptr;
  const void* data = nullptr;
  const void* filter = nullptr;
  XlaMusaMuDnnAlgorithm algorithm = 0;
};

struct ConvolveCall : WorkspaceCall {
  uint32_t allocator_struct_size = 0;
  uint32_t allocator_reserved = 0;
  uint64_t allocation_size = 0;
  void* allocation = nullptr;
  XlaMusaMuDnnStatus allocation_status = XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR;
};

struct FakeState {
  int create_handles = 0;
  int destroy_handles = 0;
  int create_tensors = 0;
  int destroy_tensors = 0;
  int create_convolutions = 0;
  int destroy_convolutions = 0;
  int32_t device_ordinal = -1;
  std::vector<void*> streams;
  std::vector<uint32_t> allow_tf32;
  std::vector<TensorCall> tensors;
  std::vector<TensorAddressCall> tensor_addresses;
  std::vector<ConvolutionConfigCall> convolution_configs;
  std::vector<WorkspaceCall> workspace_calls;
  std::vector<ConvolveCall> convolve_calls;
  uint64_t workspace_base = 1024;
  uint64_t convolve_allocation_size = 513;
};

FakeState* g_state = nullptr;
const XlaMusaMuDnnApiV1* g_api = nullptr;

class FixedTimer final : public EventBasedTimer {
 public:
  absl::StatusOr<absl::Duration> GetElapsedDuration() override {
    return absl::Microseconds(1250);
  }
};

XlaMusaMuDnnStatus FakeGetVersion(XlaMusaMuDnnVersion* version) {
  *version = {/*major=*/2, /*minor=*/8, /*patch=*/0};
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeCreateHandle(int32_t device_ordinal, void** handle) {
  ++g_state->create_handles;
  g_state->device_ordinal = device_ordinal;
  *handle = reinterpret_cast<void*>(0x1000);
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeDestroyHandle(void*) {
  ++g_state->destroy_handles;
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeSetStream(void*, void* stream) {
  g_state->streams.push_back(stream);
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeSetAllowTf32(void*, uint32_t allow_tf32) {
  g_state->allow_tf32.push_back(allow_tf32);
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeCreateTensor(void** tensor) {
  ++g_state->create_tensors;
  *tensor = reinterpret_cast<void*>(
      static_cast<uintptr_t>(0x2000 + g_state->create_tensors * 0x100));
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeDestroyTensor(void*) {
  ++g_state->destroy_tensors;
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeConfigureTensor(void* tensor,
                                       XlaMusaMuDnnDataType data_type,
                                       XlaMusaMuDnnTensorFormat format,
                                       int32_t rank, const int64_t* dimensions,
                                       const int64_t* strides) {
  g_state->tensors.push_back(
      TensorCall{tensor, data_type, format,
                 std::vector<int64_t>(dimensions, dimensions + rank),
                 std::vector<int64_t>(strides, strides + rank)});
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeSetTensorAddress(void* tensor, void* address) {
  g_state->tensor_addresses.push_back({tensor, address});
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeCreateConvolution(void** convolution) {
  ++g_state->create_convolutions;
  *convolution = reinterpret_cast<void*>(
      static_cast<uintptr_t>(0x6000 + g_state->create_convolutions * 0x100));
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeDestroyConvolution(void*) {
  ++g_state->destroy_convolutions;
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeConfigureConvolution(void*, int32_t spatial_rank,
                                            const int64_t* padding,
                                            const int64_t* strides,
                                            const int64_t* dilations,
                                            int64_t group_count) {
  g_state->convolution_configs.push_back(ConvolutionConfigCall{
      spatial_rank, std::vector<int64_t>(padding, padding + spatial_rank),
      std::vector<int64_t>(strides, strides + spatial_rank),
      std::vector<int64_t>(dilations, dilations + spatial_rank), group_count});
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeGetRecommendedAlgorithm(
    void*, void*, XlaMusaMuDnnConvolutionKind, void*, const void*, const void*,
    XlaMusaMuDnnAlgorithm* algorithm) {
  *algorithm = XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM;
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeGetWorkspaceSize(void*, void*,
                                        XlaMusaMuDnnConvolutionKind kind,
                                        void* output, const void* data,
                                        const void* filter,
                                        XlaMusaMuDnnAlgorithm algorithm,
                                        uint64_t* workspace_size_bytes) {
  g_state->workspace_calls.push_back({kind, output, data, filter, algorithm});
  *workspace_size_bytes =
      g_state->workspace_base + static_cast<uint64_t>(algorithm) * 256;
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

XlaMusaMuDnnStatus FakeConvolve(
    void*, void*, XlaMusaMuDnnConvolutionKind kind, void* output,
    const void* data, const void* filter, XlaMusaMuDnnAlgorithm algorithm,
    const XlaMusaMuDnnWorkspaceAllocator* allocator) {
  ConvolveCall call;
  call.kind = kind;
  call.output = output;
  call.data = data;
  call.filter = filter;
  call.algorithm = algorithm;
  call.allocator_struct_size = allocator->struct_size;
  call.allocator_reserved = allocator->reserved;
  call.allocation_size = g_state->convolve_allocation_size;
  call.allocation_status = allocator->allocate(
      allocator->user_data, call.allocation_size, &call.allocation);
  if (call.allocation_status == XLA_MUSA_MUDNN_STATUS_SUCCESS) {
    allocator->release(allocator->user_data, call.allocation,
                       call.allocation_size);
  }
  g_state->convolve_calls.push_back(call);
  return call.allocation_status;
}

const XlaMusaMuDnnApiV1* FakeGetter() { return g_api; }

XlaMusaMuDnnApiV1 CompleteApi() {
  XlaMusaMuDnnApiV1 api = {};
  api.struct_size = sizeof(api);
  api.abi_version = XLA_MUSA_MUDNN_ABI_VERSION_1;
  api.capabilities = XLA_MUSA_MUDNN_CAPABILITIES_V1;
  api.get_version = FakeGetVersion;
  api.create_handle = FakeCreateHandle;
  api.destroy_handle = FakeDestroyHandle;
  api.set_stream = FakeSetStream;
  api.set_allow_tf32 = FakeSetAllowTf32;
  api.create_tensor = FakeCreateTensor;
  api.destroy_tensor = FakeDestroyTensor;
  api.configure_tensor = FakeConfigureTensor;
  api.set_tensor_address = FakeSetTensorAddress;
  api.create_convolution = FakeCreateConvolution;
  api.destroy_convolution = FakeDestroyConvolution;
  api.configure_convolution = FakeConfigureConvolution;
  api.get_recommended_algorithm = FakeGetRecommendedAlgorithm;
  api.get_workspace_size = FakeGetWorkspaceSize;
  api.convolve = FakeConvolve;
  return api;
}

class FakeLoader final : public internal::MusaSymbolLoader {
 public:
  absl::Status Load() override { return absl::OkStatus(); }

  absl::StatusOr<void*> Resolve(absl::string_view symbol) const override {
    if (symbol == "xla_musa_mudnn_get_api_v1") {
      return reinterpret_cast<void*>(&FakeGetter);
    }
    return absl::NotFoundError("unexpected fake muDNN symbol");
  }

  absl::string_view loaded_path() const override {
    return "fake-libxla_musa_mudnn_shim.so.1";
  }
};

dnn::BatchDescriptor MakeBatch2D(dnn::DataLayout layout, int64_t count,
                                 int64_t channels, int64_t height,
                                 int64_t width) {
  dnn::BatchDescriptor descriptor(/*ndims=*/2);
  descriptor.set_count(count)
      .set_feature_map_count(channels)
      .set_height(height)
      .set_width(width)
      .set_layout(layout);
  return descriptor;
}

dnn::FilterDescriptor MakeFilter2D(dnn::FilterLayout layout, int64_t output,
                                   int64_t input, int64_t height,
                                   int64_t width) {
  dnn::FilterDescriptor descriptor(/*ndims=*/2);
  descriptor.set_output_feature_map_count(output)
      .set_input_feature_map_count(input)
      .set_input_filter_height(height)
      .set_input_filter_width(width)
      .set_layout(layout);
  return descriptor;
}

dnn::ConvolutionDescriptor MakeConvolution2D() {
  dnn::ConvolutionDescriptor descriptor(/*ndims=*/2);
  descriptor.set_zero_padding_height(1)
      .set_zero_padding_width(2)
      .set_vertical_filter_stride(3)
      .set_horizontal_filter_stride(4)
      .set_vertical_dilation_rate(5)
      .set_horizontal_dilation_rate(6)
      .set_group_count(2);
  return descriptor;
}

class MusaDnnTest : public ::testing::Test {
 protected:
  void SetUp() override {
    state_ = {};
    api_table_ = CompleteApi();
    g_state = &state_;
    g_api = &api_table_;
    ON_CALL(executor_, Activate()).WillByDefault([] {
      return std::make_unique<ActivateContext>();
    });
    ON_CALL(executor_, device_ordinal()).WillByDefault(Return(7));
    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
    ON_CALL(stream_, platform_specific_handle())
        .WillByDefault(Return(
            Stream::PlatformSpecificHandle{reinterpret_cast<void*>(0x7000)}));
    ON_CALL(stream_, CreateEventBasedTimer)
        .WillByDefault(
            [](bool) -> absl::StatusOr<std::unique_ptr<EventBasedTimer>> {
              return std::make_unique<FixedTimer>();
            });
  }

  void TearDown() override {
    g_state = nullptr;
    g_api = nullptr;
  }

  std::unique_ptr<MusaMuDnnApi> CreateApi() {
    return MusaMuDnnApi::CreateForTesting(std::make_unique<FakeLoader>());
  }

  std::unique_ptr<MusaDnn> CreateInitializedDnn(MusaMuDnnApi* api) {
    auto dnn_support = std::make_unique<MusaDnn>(&executor_, api);
    EXPECT_TRUE(dnn_support->Init().ok());
    return dnn_support;
  }

  absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> MakeRunner(
      MusaDnn* dnn_support, dnn::ConvolutionKind kind,
      const dnn::BatchDescriptor& input, const dnn::FilterDescriptor& filter,
      const dnn::BatchDescriptor& output,
      const dnn::ConvolutionDescriptor& convolution,
      dnn::AlgorithmDesc algorithm = dnn::AlgorithmDesc(
          XLA_MUSA_MUDNN_ALGORITHM_GEMM, /*use_tensor_ops=*/true,
          /*workspace_size=*/513)) {
    return dnn_support->ConvolveRunnerFromDesc(
        &stream_, algorithm, kind, dnn::DataType::kFloat, dnn::DataType::kFloat,
        input, filter, output, convolution);
  }

  FakeState state_;
  XlaMusaMuDnnApiV1 api_table_ = {};
  NiceMock<MockStreamExecutor> executor_;
  NiceMock<MockStream> stream_;
};

TEST_F(MusaDnnTest, ForwardsAsymmetricNchwAndOihwDescriptors) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi();
  std::unique_ptr<MusaDnn> dnn_support = CreateInitializedDnn(api.get());
  dnn::BatchDescriptor input =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 3, 5, 7);
  dnn::FilterDescriptor filter =
      MakeFilter2D(dnn::FilterLayout::kOutputInputYX, 11, 3, 2, 4);
  dnn::BatchDescriptor output =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 11, 4, 3);
  dnn::ConvolutionDescriptor convolution = MakeConvolution2D();

  absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> runner =
      MakeRunner(dnn_support.get(), dnn::ConvolutionKind::FORWARD, input,
                 filter, output, convolution);
  ASSERT_TRUE(runner.ok()) << runner.status();
  ASSERT_EQ(state_.tensors.size(), 3);
  EXPECT_EQ(state_.tensors[0].format, XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW);
  EXPECT_THAT(state_.tensors[0].dimensions, ElementsAre(2, 3, 5, 7));
  EXPECT_THAT(state_.tensors[0].strides, ElementsAre(105, 35, 7, 1));
  EXPECT_EQ(state_.tensors[1].format, XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW);
  EXPECT_THAT(state_.tensors[1].dimensions, ElementsAre(11, 3, 2, 4));
  EXPECT_THAT(state_.tensors[1].strides, ElementsAre(24, 8, 4, 1));
  EXPECT_EQ(state_.tensors[2].format, XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW);
  EXPECT_THAT(state_.tensors[2].dimensions, ElementsAre(2, 11, 4, 3));
  EXPECT_THAT(state_.tensors[2].strides, ElementsAre(132, 12, 3, 1));

  ASSERT_EQ(state_.convolution_configs.size(), 1);
  const ConvolutionConfigCall& config = state_.convolution_configs.front();
  EXPECT_EQ(config.spatial_rank, 2);
  EXPECT_THAT(config.padding, ElementsAre(1, 2));
  EXPECT_THAT(config.strides, ElementsAre(3, 4));
  EXPECT_THAT(config.dilations, ElementsAre(5, 6));
  EXPECT_EQ(config.group_count, 2);
}

TEST_F(MusaDnnTest, RejectsNonContiguousNhwcFilterLayout) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi();
  std::unique_ptr<MusaDnn> dnn_support = CreateInitializedDnn(api.get());
  dnn::BatchDescriptor input =
      MakeBatch2D(dnn::DataLayout::kBatchYXDepth, 2, 3, 5, 7);
  dnn::FilterDescriptor filter =
      MakeFilter2D(dnn::FilterLayout::kOutputYXInput, 11, 3, 2, 4);
  dnn::BatchDescriptor output =
      MakeBatch2D(dnn::DataLayout::kBatchYXDepth, 2, 11, 4, 3);

  absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> runner =
      MakeRunner(dnn_support.get(), dnn::ConvolutionKind::FORWARD, input,
                 filter, output, MakeConvolution2D());
  EXPECT_THAT(runner,
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("requires contiguous filters")));
  EXPECT_TRUE(state_.tensors.empty());
}

TEST_F(MusaDnnTest, RejectsNonContiguousNdhwcFilterLayout) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi();
  std::unique_ptr<MusaDnn> dnn_support = CreateInitializedDnn(api.get());
  dnn::BatchDescriptor input(/*ndims=*/3);
  input.set_count(2)
      .set_feature_map_count(4)
      .set_spatial_dim(dnn::DimIndex::Z, 3)
      .set_spatial_dim(dnn::DimIndex::Y, 5)
      .set_spatial_dim(dnn::DimIndex::X, 7)
      .set_layout(dnn::DataLayout::kBatchYXDepth);
  dnn::FilterDescriptor filter(/*ndims=*/3);
  filter.set_output_feature_map_count(6)
      .set_input_feature_map_count(4)
      .set_spatial_dim(dnn::DimIndex::Z, 2)
      .set_spatial_dim(dnn::DimIndex::Y, 3)
      .set_spatial_dim(dnn::DimIndex::X, 5)
      .set_layout(dnn::FilterLayout::kOutputYXInput);
  dnn::BatchDescriptor output(/*ndims=*/3);
  output.set_count(2)
      .set_feature_map_count(6)
      .set_spatial_dim(dnn::DimIndex::Z, 2)
      .set_spatial_dim(dnn::DimIndex::Y, 3)
      .set_spatial_dim(dnn::DimIndex::X, 4)
      .set_layout(dnn::DataLayout::kBatchYXDepth);
  dnn::ConvolutionDescriptor convolution(/*ndims=*/3);

  absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> runner =
      MakeRunner(dnn_support.get(), dnn::ConvolutionKind::FORWARD, input,
                 filter, output, convolution);
  EXPECT_THAT(runner,
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("requires contiguous filters")));
  EXPECT_TRUE(state_.tensors.empty());
}

TEST_F(MusaDnnTest, MapsForwardAndBackwardRolesAndRoundTripsAlgorithm) {
  struct RoleCase {
    dnn::ConvolutionKind kind;
    XlaMusaMuDnnConvolutionKind mudnn_kind;
    int output_index;
    int data_index;
    int filter_index;
  };
  const RoleCase role_cases[] = {
      {dnn::ConvolutionKind::FORWARD, XLA_MUSA_MUDNN_CONVOLUTION_FORWARD,
       /*output_index=*/2, /*data_index=*/0, /*filter_index=*/1},
      {dnn::ConvolutionKind::BACKWARD_DATA,
       XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA,
       /*output_index=*/0, /*data_index=*/2, /*filter_index=*/1},
      {dnn::ConvolutionKind::BACKWARD_FILTER,
       XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER,
       /*output_index=*/1, /*data_index=*/0, /*filter_index=*/2},
  };
  std::unique_ptr<MusaMuDnnApi> api = CreateApi();
  std::unique_ptr<MusaDnn> dnn_support = CreateInitializedDnn(api.get());
  dnn::BatchDescriptor input =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 3, 5, 7);
  dnn::FilterDescriptor filter =
      MakeFilter2D(dnn::FilterLayout::kOutputInputYX, 11, 3, 2, 4);
  dnn::BatchDescriptor output =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 11, 4, 3);
  dnn::ConvolutionDescriptor convolution = MakeConvolution2D();
  DeviceAddressBase input_data(reinterpret_cast<void*>(0x8100), 256);
  DeviceAddressBase filter_data(reinterpret_cast<void*>(0x8200), 256);
  DeviceAddressBase output_data(reinterpret_cast<void*>(0x8300), 256);
  DeviceAddressBase scratch(reinterpret_cast<void*>(0x9000), 1024);

  for (const RoleCase& role_case : role_cases) {
    const size_t tensor_offset = state_.tensors.size();
    const size_t address_offset = state_.tensor_addresses.size();
    const size_t convolve_offset = state_.convolve_calls.size();
    absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> runner = MakeRunner(
        dnn_support.get(), role_case.kind, input, filter, output, convolution);
    ASSERT_TRUE(runner.ok()) << runner.status();
    ASSERT_GE(state_.tensors.size(), tensor_offset + 3);
    const void* handles[] = {state_.tensors[tensor_offset + 0].tensor,
                             state_.tensors[tensor_offset + 1].tensor,
                             state_.tensors[tensor_offset + 2].tensor};

    EXPECT_EQ((*runner)->GetWorkspaceSize(), 513);
    absl::StatusOr<dnn::AlgorithmDesc> serialized =
        (*runner)->ToAlgorithmDesc();
    ASSERT_TRUE(serialized.ok()) << serialized.status();
    EXPECT_EQ(serialized->algo_id(), XLA_MUSA_MUDNN_ALGORITHM_GEMM);
    EXPECT_TRUE(serialized->tensor_ops_enabled());
    EXPECT_EQ(serialized->workspace_size(), 513);

    dnn::ProfileResult profile;
    EXPECT_TRUE((**runner)(&stream_, &profile, scratch, input_data, filter_data,
                           output_data)
                    .ok());
    ASSERT_EQ(state_.tensor_addresses.size(), address_offset + 3);
    EXPECT_EQ(state_.tensor_addresses[address_offset + 0].tensor, handles[0]);
    EXPECT_EQ(state_.tensor_addresses[address_offset + 0].address,
              input_data.opaque());
    EXPECT_EQ(state_.tensor_addresses[address_offset + 1].tensor, handles[1]);
    EXPECT_EQ(state_.tensor_addresses[address_offset + 1].address,
              filter_data.opaque());
    EXPECT_EQ(state_.tensor_addresses[address_offset + 2].tensor, handles[2]);
    EXPECT_EQ(state_.tensor_addresses[address_offset + 2].address,
              output_data.opaque());

    ASSERT_EQ(state_.convolve_calls.size(), convolve_offset + 1);
    const ConvolveCall& call = state_.convolve_calls.back();
    EXPECT_EQ(call.kind, role_case.mudnn_kind);
    EXPECT_EQ(call.output, handles[role_case.output_index]);
    EXPECT_EQ(call.data, handles[role_case.data_index]);
    EXPECT_EQ(call.filter, handles[role_case.filter_index]);
    EXPECT_EQ(call.algorithm, XLA_MUSA_MUDNN_ALGORITHM_GEMM);
    EXPECT_EQ(call.allocator_struct_size,
              sizeof(XlaMusaMuDnnWorkspaceAllocator));
    EXPECT_EQ(call.allocator_reserved, 0);
    EXPECT_EQ(call.allocation_size, 513);
    EXPECT_EQ(call.allocation, scratch.opaque());
    EXPECT_EQ(call.allocation_status, XLA_MUSA_MUDNN_STATUS_SUCCESS);
    EXPECT_EQ(profile.algorithm().algo_id(), XLA_MUSA_MUDNN_ALGORITHM_GEMM);
    EXPECT_EQ(profile.algorithm().workspace_size(), 513);
    EXPECT_FLOAT_EQ(profile.elapsed_time_in_ms(), 1.25f);
  }
  EXPECT_THAT(state_.streams, ElementsAre(reinterpret_cast<void*>(0x7000),
                                          reinterpret_cast<void*>(0x7000),
                                          reinterpret_cast<void*>(0x7000)));
  EXPECT_THAT(state_.allow_tf32, ElementsAre(1, 1, 1));
}

TEST_F(MusaDnnTest, DiscoversQualifiedAlgorithmsAndWorkspaceSizes) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi();
  std::unique_ptr<MusaDnn> dnn_support = CreateInitializedDnn(api.get());
  dnn::BatchDescriptor input =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 3, 5, 7);
  dnn::FilterDescriptor filter =
      MakeFilter2D(dnn::FilterLayout::kOutputInputYX, 11, 3, 2, 4);
  dnn::BatchDescriptor output =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 11, 4, 3);
  DeviceAddressBase input_data(reinterpret_cast<void*>(0x8100), 256);
  DeviceAddressBase filter_data(reinterpret_cast<void*>(0x8200), 256);
  DeviceAddressBase output_data(reinterpret_cast<void*>(0x8300), 256);
  std::vector<std::unique_ptr<const dnn::ConvRunner>> runners;

  EXPECT_TRUE(dnn_support
                  ->GetConvolveRunners(
                      dnn::ConvolutionKind::FORWARD, dnn::DataType::kFloat,
                      dnn::DataType::kFloat, &stream_, input, input_data,
                      filter, filter_data, output, output_data,
                      MakeConvolution2D(), /*use_fallback=*/false,
                      /*scratch_allocator=*/nullptr,
                      EngineOptions(/*require_determinism=*/false,
                                    /*allow_tf32=*/false,
                                    /*require_command_buffer=*/false),
                      &runners)
                  .ok());
  constexpr int64_t kAlgorithms[] = {
      XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM,
      XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED,
      XLA_MUSA_MUDNN_ALGORITHM_GEMM};
  ASSERT_EQ(runners.size(), 3);
  ASSERT_EQ(state_.workspace_calls.size(), 3);
  for (int index = 0; index < 3; ++index) {
    const int64_t algorithm = kAlgorithms[index];
    absl::StatusOr<dnn::AlgorithmDesc> desc =
        runners[index]->ToAlgorithmDesc();
    ASSERT_TRUE(desc.ok()) << desc.status();
    EXPECT_EQ(desc->algo_id(), algorithm);
    EXPECT_FALSE(desc->tensor_ops_enabled());
    EXPECT_EQ(desc->workspace_size(), 1024 + algorithm * 256);
    EXPECT_EQ(runners[index]->GetWorkspaceSize(), 1024 + algorithm * 256);
    EXPECT_EQ(state_.workspace_calls[index].kind,
              XLA_MUSA_MUDNN_CONVOLUTION_FORWARD);
    EXPECT_EQ(state_.workspace_calls[index].algorithm, algorithm);
  }
  EXPECT_THAT(state_.allow_tf32, ElementsAre(0, 0, 0));
}

TEST_F(MusaDnnTest, RejectsBf16AndUnqualifiedDirectAlgorithm) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi();
  std::unique_ptr<MusaDnn> dnn_support = CreateInitializedDnn(api.get());
  dnn::BatchDescriptor input =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 1, 8, 4, 4);
  dnn::FilterDescriptor filter =
      MakeFilter2D(dnn::FilterLayout::kOutputInputYX, 8, 8, 3, 3);
  dnn::BatchDescriptor output =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 1, 8, 4, 4);

  EXPECT_THAT(
      dnn_support->ConvolveRunnerFromDesc(
          &stream_,
          dnn::AlgorithmDesc(XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED,
                             /*use_tensor_ops=*/false,
                             /*workspace_size=*/0),
          dnn::ConvolutionKind::FORWARD, dnn::DataType::kBF16,
          dnn::DataType::kBF16, input, filter, output, MakeConvolution2D()),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("supports only f16 and f32")));
  EXPECT_THAT(
      dnn_support->ConvolveRunnerFromDesc(
          &stream_,
          dnn::AlgorithmDesc(XLA_MUSA_MUDNN_ALGORITHM_DIRECT,
                             /*use_tensor_ops=*/false,
                             /*workspace_size=*/0),
          dnn::ConvolutionKind::FORWARD, dnn::DataType::kFloat,
          dnn::DataType::kFloat, input, filter, output, MakeConvolution2D()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("expected one of 0, 2, or 3")));
  EXPECT_TRUE(state_.tensors.empty());
}

TEST_F(MusaDnnTest, DeterminismAndCommandBuffersFailClosed) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi();
  std::unique_ptr<MusaDnn> dnn_support = CreateInitializedDnn(api.get());
  dnn::BatchDescriptor input =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 3, 5, 7);
  dnn::FilterDescriptor filter =
      MakeFilter2D(dnn::FilterLayout::kOutputInputYX, 11, 3, 2, 4);
  dnn::BatchDescriptor output =
      MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 11, 4, 3);
  DeviceAddressBase input_data(reinterpret_cast<void*>(0x8100), 256);
  DeviceAddressBase filter_data(reinterpret_cast<void*>(0x8200), 256);
  DeviceAddressBase output_data(reinterpret_cast<void*>(0x8300), 256);
  std::vector<std::unique_ptr<const dnn::ConvRunner>> runners;

  EXPECT_THAT(
      dnn_support->GetConvolveRunners(
          dnn::ConvolutionKind::FORWARD, dnn::DataType::kFloat,
          dnn::DataType::kFloat, &stream_, input, input_data, filter,
          filter_data, output, output_data, MakeConvolution2D(),
          /*use_fallback=*/false, /*scratch_allocator=*/nullptr,
          EngineOptions(/*require_determinism=*/true, /*allow_tf32=*/true,
                        /*require_command_buffer=*/false),
          &runners),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("deterministic convolution fails closed")));
  EXPECT_THAT(
      dnn_support->GetConvolveRunners(
          dnn::ConvolutionKind::FORWARD, dnn::DataType::kFloat,
          dnn::DataType::kFloat, &stream_, input, input_data, filter,
          filter_data, output, output_data, MakeConvolution2D(),
          /*use_fallback=*/false, /*scratch_allocator=*/nullptr,
          EngineOptions(/*require_determinism=*/false, /*allow_tf32=*/true,
                        /*require_command_buffer=*/true),
          &runners),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("explicit command buffers")));
  EXPECT_TRUE(state_.tensors.empty());
  EXPECT_TRUE(state_.workspace_calls.empty());
  EXPECT_TRUE(state_.convolve_calls.empty());
}

TEST_F(MusaDnnTest, RejectsInsufficientScratchBeforeVendorDispatch) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi();
  std::unique_ptr<MusaDnn> dnn_support = CreateInitializedDnn(api.get());
  absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> runner =
      MakeRunner(dnn_support.get(), dnn::ConvolutionKind::FORWARD,
                 MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 3, 5, 7),
                 MakeFilter2D(dnn::FilterLayout::kOutputInputYX, 11, 3, 2, 4),
                 MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 11, 4, 3),
                 MakeConvolution2D());
  ASSERT_TRUE(runner.ok()) << runner.status();

  EXPECT_THAT(
      (**runner)(&stream_, /*output_profile_result=*/nullptr,
                 DeviceAddressBase(reinterpret_cast<void*>(0x9000), 512),
                 DeviceAddressBase(reinterpret_cast<void*>(0x8100), 256),
                 DeviceAddressBase(reinterpret_cast<void*>(0x8200), 256),
                 DeviceAddressBase(reinterpret_cast<void*>(0x8300), 256)),
      StatusIs(absl::StatusCode::kResourceExhausted,
               HasSubstr("requires 513 scratch bytes")));
  EXPECT_TRUE(state_.convolve_calls.empty());
}

TEST_F(MusaDnnTest, RejectsRunnerWithUnknownWorkspaceSize) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi();
  std::unique_ptr<MusaDnn> dnn_support = CreateInitializedDnn(api.get());
  EXPECT_THAT(
      MakeRunner(dnn_support.get(), dnn::ConvolutionKind::FORWARD,
                 MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 3, 5, 7),
                 MakeFilter2D(dnn::FilterLayout::kOutputInputYX, 11, 3, 2, 4),
                 MakeBatch2D(dnn::DataLayout::kBatchDepthYX, 2, 11, 4, 3),
                 MakeConvolution2D(),
                 dnn::AlgorithmDesc(XLA_MUSA_MUDNN_ALGORITHM_GEMM,
                                    /*use_tensor_ops=*/false)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("explicit workspace size")));
  EXPECT_TRUE(state_.tensors.empty());
}

}  // namespace
}  // namespace stream_executor::musa
