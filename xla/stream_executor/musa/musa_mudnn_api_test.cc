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

#include "xla/stream_executor/musa/musa_mudnn_api.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/stream_executor/musa/mudnn_shim/mudnn_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

struct FakeCalls {
  int get_version = 0;
  int create_handle = 0;
  int destroy_handle = 0;
  int set_stream = 0;
  int set_allow_tf32 = 0;
  int create_tensor = 0;
  int destroy_tensor = 0;
  int configure_tensor = 0;
  int set_tensor_address = 0;
  int create_convolution = 0;
  int destroy_convolution = 0;
  int configure_convolution = 0;
  int get_recommended_algorithm = 0;
  int get_workspace_size = 0;
  int convolve = 0;

  int32_t device_ordinal = -1;
  void* stream = nullptr;
  uint32_t allow_tf32 = 0;
  XlaMusaMuDnnDataType data_type = 0;
  XlaMusaMuDnnTensorFormat format = 0;
  int32_t rank = 0;
  int64_t dimensions[5] = {};
  int64_t tensor_strides[5] = {};
  void* tensor_address = nullptr;
  int32_t spatial_rank = 0;
  int64_t padding[3] = {};
  int64_t convolution_strides[3] = {};
  int64_t dilations[3] = {};
  int64_t group_count = 0;
  XlaMusaMuDnnConvolutionKind kind = 0;
  XlaMusaMuDnnAlgorithm algorithm = 0;
  const XlaMusaMuDnnWorkspaceAllocator* workspace_allocator = nullptr;
};

FakeCalls* g_calls = nullptr;
XlaMusaMuDnnApiV1* g_api = nullptr;
XlaMusaMuDnnStatus g_next_status = XLA_MUSA_MUDNN_STATUS_SUCCESS;
XlaMusaMuDnnVersion g_version = {/*major=*/2, /*minor=*/8, /*patch=*/0};
XlaMusaMuDnnAlgorithm g_recommended_algorithm =
    XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM;

XlaMusaMuDnnStatus FakeGetVersion(XlaMusaMuDnnVersion* version) {
  ++g_calls->get_version;
  if (g_next_status == XLA_MUSA_MUDNN_STATUS_SUCCESS) *version = g_version;
  return g_next_status;
}

XlaMusaMuDnnStatus FakeCreateHandle(int32_t device_ordinal, void** handle) {
  ++g_calls->create_handle;
  g_calls->device_ordinal = device_ordinal;
  if (g_next_status == XLA_MUSA_MUDNN_STATUS_SUCCESS) {
    *handle = reinterpret_cast<void*>(0x1000);
  }
  return g_next_status;
}

XlaMusaMuDnnStatus FakeDestroyHandle(void*) {
  ++g_calls->destroy_handle;
  return g_next_status;
}

XlaMusaMuDnnStatus FakeSetStream(void*, void* stream) {
  ++g_calls->set_stream;
  g_calls->stream = stream;
  return g_next_status;
}

XlaMusaMuDnnStatus FakeSetAllowTf32(void*, uint32_t allow_tf32) {
  ++g_calls->set_allow_tf32;
  g_calls->allow_tf32 = allow_tf32;
  return g_next_status;
}

XlaMusaMuDnnStatus FakeCreateTensor(void** tensor) {
  ++g_calls->create_tensor;
  if (g_next_status == XLA_MUSA_MUDNN_STATUS_SUCCESS) {
    *tensor = reinterpret_cast<void*>(0x2000);
  }
  return g_next_status;
}

XlaMusaMuDnnStatus FakeDestroyTensor(void*) {
  ++g_calls->destroy_tensor;
  return g_next_status;
}

XlaMusaMuDnnStatus FakeConfigureTensor(void*, XlaMusaMuDnnDataType data_type,
                                       XlaMusaMuDnnTensorFormat format,
                                       int32_t rank, const int64_t* dimensions,
                                       const int64_t* strides) {
  ++g_calls->configure_tensor;
  g_calls->data_type = data_type;
  g_calls->format = format;
  g_calls->rank = rank;
  for (int32_t index = 0; index < rank; ++index) {
    g_calls->dimensions[index] = dimensions[index];
    g_calls->tensor_strides[index] = strides[index];
  }
  return g_next_status;
}

XlaMusaMuDnnStatus FakeSetTensorAddress(void*, void* address) {
  ++g_calls->set_tensor_address;
  g_calls->tensor_address = address;
  return g_next_status;
}

XlaMusaMuDnnStatus FakeCreateConvolution(void** convolution) {
  ++g_calls->create_convolution;
  if (g_next_status == XLA_MUSA_MUDNN_STATUS_SUCCESS) {
    *convolution = reinterpret_cast<void*>(0x3000);
  }
  return g_next_status;
}

XlaMusaMuDnnStatus FakeDestroyConvolution(void*) {
  ++g_calls->destroy_convolution;
  return g_next_status;
}

XlaMusaMuDnnStatus FakeConfigureConvolution(void*, int32_t spatial_rank,
                                            const int64_t* padding,
                                            const int64_t* strides,
                                            const int64_t* dilations,
                                            int64_t group_count) {
  ++g_calls->configure_convolution;
  g_calls->spatial_rank = spatial_rank;
  for (int32_t index = 0; index < spatial_rank; ++index) {
    g_calls->padding[index] = padding[index];
    g_calls->convolution_strides[index] = strides[index];
    g_calls->dilations[index] = dilations[index];
  }
  g_calls->group_count = group_count;
  return g_next_status;
}

void RecordConvolutionCall(XlaMusaMuDnnConvolutionKind kind,
                           XlaMusaMuDnnAlgorithm algorithm) {
  g_calls->kind = kind;
  g_calls->algorithm = algorithm;
}

XlaMusaMuDnnStatus FakeGetRecommendedAlgorithm(
    void*, void*, XlaMusaMuDnnConvolutionKind kind, void*, const void*,
    const void*, XlaMusaMuDnnAlgorithm* algorithm) {
  ++g_calls->get_recommended_algorithm;
  RecordConvolutionCall(kind, XLA_MUSA_MUDNN_ALGORITHM_RECOMMENDED);
  if (g_next_status == XLA_MUSA_MUDNN_STATUS_SUCCESS) {
    *algorithm = g_recommended_algorithm;
  }
  return g_next_status;
}

XlaMusaMuDnnStatus FakeGetWorkspaceSize(void*, void*,
                                        XlaMusaMuDnnConvolutionKind kind, void*,
                                        const void*, const void*,
                                        XlaMusaMuDnnAlgorithm algorithm,
                                        uint64_t* workspace_size_bytes) {
  ++g_calls->get_workspace_size;
  RecordConvolutionCall(kind, algorithm);
  if (g_next_status == XLA_MUSA_MUDNN_STATUS_SUCCESS) {
    *workspace_size_bytes = 8192;
  }
  return g_next_status;
}

XlaMusaMuDnnStatus FakeConvolve(
    void*, void*, XlaMusaMuDnnConvolutionKind kind, void*, const void*,
    const void*, XlaMusaMuDnnAlgorithm algorithm,
    const XlaMusaMuDnnWorkspaceAllocator* workspace_allocator) {
  ++g_calls->convolve;
  RecordConvolutionCall(kind, algorithm);
  g_calls->workspace_allocator = workspace_allocator;
  return g_next_status;
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

class FakeSymbolLoader final : public internal::MusaSymbolLoader {
 public:
  explicit FakeSymbolLoader(
      void* getter, absl::Status load_status = absl::OkStatus(),
      absl::Status resolve_status = absl::NotFoundError("missing getter"))
      : getter_(getter),
        load_status_(std::move(load_status)),
        resolve_status_(std::move(resolve_status)) {}

  absl::Status Load() override {
    ++load_calls_;
    return load_status_;
  }

  absl::StatusOr<void*> Resolve(absl::string_view symbol) const override {
    resolved_.push_back(std::string(symbol));
    if (getter_ != nullptr && symbol == "xla_musa_mudnn_get_api_v1") {
      return getter_;
    }
    return resolve_status_;
  }

  absl::string_view loaded_path() const override {
    return "fake-libxla_musa_mudnn_shim.so.1";
  }

  int load_calls() const { return load_calls_; }
  const std::vector<std::string>& resolved() const { return resolved_; }

 private:
  void* getter_;
  absl::Status load_status_;
  absl::Status resolve_status_;
  int load_calls_ = 0;
  mutable std::vector<std::string> resolved_;
};

XlaMusaMuDnnStatus FakeWorkspaceAllocate(void*, uint64_t, void** address) {
  *address = reinterpret_cast<void*>(0x9000);
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

void FakeWorkspaceRelease(void*, void*, uint64_t) {}

class MusaMuDnnApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    calls_ = {};
    api_table_ = CompleteApi();
    g_calls = &calls_;
    g_api = &api_table_;
    g_next_status = XLA_MUSA_MUDNN_STATUS_SUCCESS;
    g_version = {/*major=*/2, /*minor=*/8, /*patch=*/0};
    g_recommended_algorithm = XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM;
  }

  void TearDown() override {
    g_calls = nullptr;
    g_api = nullptr;
  }

  std::unique_ptr<MusaMuDnnApi> CreateApi(
      std::unique_ptr<internal::MusaSymbolLoader> loader) {
    return MusaMuDnnApi::CreateForTesting(std::move(loader));
  }

  FakeCalls calls_;
  XlaMusaMuDnnApiV1 api_table_ = {};
};

TEST(MusaMuDnnShimCandidatesTest, UsesExplicitAbsolutePathOnly) {
  absl::StatusOr<std::vector<std::string>> candidates =
      internal::GetMusaMuDnnShimCandidates(
          std::optional<absl::string_view>("/opt/musa/lib/custom-mudnn.so"),
          "/plugins/libpjrt_musa.so");
  ASSERT_TRUE(candidates.ok());
  EXPECT_THAT(*candidates, ElementsAre("/opt/musa/lib/custom-mudnn.so"));
}

TEST(MusaMuDnnShimCandidatesTest, UsesAdjacentThenVersionedSoname) {
  absl::StatusOr<std::vector<std::string>> candidates =
      internal::GetMusaMuDnnShimCandidates(std::nullopt,
                                           "/plugins/libpjrt_musa.so");
  ASSERT_TRUE(candidates.ok());
  EXPECT_THAT(*candidates, ElementsAre("/plugins/libxla_musa_mudnn_shim.so.1",
                                       "libxla_musa_mudnn_shim.so.1"));
}

TEST(MusaMuDnnShimCandidatesTest, RejectsInvalidConfiguredPaths) {
  EXPECT_THAT(
      internal::GetMusaMuDnnShimCandidates(std::optional<absl::string_view>(""),
                                           "/plugins/libpjrt_musa.so"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("must not be")));
  EXPECT_THAT(
      internal::GetMusaMuDnnShimCandidates(
          std::optional<absl::string_view>("relative/libmudnn.so"),
          "/plugins/libpjrt_musa.so"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("absolute")));
  const std::string oversized_path(4097, 'x');
  EXPECT_THAT(internal::GetMusaMuDnnShimCandidates(
                  std::optional<absl::string_view>(oversized_path),
                  "/plugins/libpjrt_musa.so"),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("4096")));
}

TEST_F(MusaMuDnnApiTest, LoadsOnceAndDispatchesEveryTypedOperation) {
  auto loader =
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter));
  FakeSymbolLoader* loader_ptr = loader.get();
  std::unique_ptr<MusaMuDnnApi> api = CreateApi(std::move(loader));

  ASSERT_TRUE(api->Init().ok());
  EXPECT_TRUE(api->Init().ok());
  EXPECT_EQ(loader_ptr->load_calls(), 1);
  EXPECT_THAT(loader_ptr->resolved(), ElementsAre("xla_musa_mudnn_get_api_v1"));
  EXPECT_EQ(api->abi_version(), XLA_MUSA_MUDNN_ABI_VERSION_1);
  EXPECT_EQ(api->capabilities(), XLA_MUSA_MUDNN_CAPABILITIES_V1);
  EXPECT_EQ(api->loaded_path(), "fake-libxla_musa_mudnn_shim.so.1");

  XlaMusaMuDnnVersion version = {};
  ASSERT_TRUE(api->GetVersion(&version).ok());
  EXPECT_EQ(version.major, 2);
  EXPECT_EQ(version.minor, 8);
  EXPECT_EQ(version.patch, 0);

  void* handle = nullptr;
  ASSERT_TRUE(api->CreateHandle(3, &handle).ok());
  ASSERT_NE(handle, nullptr);
  void* stream = reinterpret_cast<void*>(0x1100);
  EXPECT_TRUE(api->SetStream(handle, stream).ok());
  EXPECT_TRUE(api->SetAllowTf32(handle, true).ok());

  void* tensor = nullptr;
  ASSERT_TRUE(api->CreateTensor(&tensor).ok());
  ASSERT_NE(tensor, nullptr);
  const int64_t dimensions[] = {2, 3, 5, 7};
  const int64_t strides[] = {105, 35, 7, 1};
  EXPECT_TRUE(api->ConfigureTensor(tensor, XLA_MUSA_MUDNN_DATA_TYPE_F16,
                                   XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW, 4,
                                   dimensions, strides)
                  .ok());
  void* tensor_address = reinterpret_cast<void*>(0x2200);
  EXPECT_TRUE(api->SetTensorAddress(tensor, tensor_address).ok());

  void* convolution = nullptr;
  ASSERT_TRUE(api->CreateConvolution(&convolution).ok());
  ASSERT_NE(convolution, nullptr);
  const int64_t padding[] = {1, 2};
  const int64_t convolution_strides[] = {2, 3};
  const int64_t dilations[] = {1, 2};
  EXPECT_TRUE(api->ConfigureConvolution(convolution, 2, padding,
                                        convolution_strides, dilations, 4)
                  .ok());

  void* output = reinterpret_cast<void*>(0x4000);
  void* data = reinterpret_cast<void*>(0x5000);
  void* filter = reinterpret_cast<void*>(0x6000);
  XlaMusaMuDnnAlgorithm algorithm = XLA_MUSA_MUDNN_ALGORITHM_RECOMMENDED;
  EXPECT_TRUE(api->GetRecommendedAlgorithm(handle, convolution,
                                           XLA_MUSA_MUDNN_CONVOLUTION_FORWARD,
                                           output, data, filter, &algorithm)
                  .ok());
  EXPECT_EQ(algorithm, XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM);

  uint64_t workspace_size_bytes = 0;
  EXPECT_TRUE(api->GetWorkspaceSize(handle, convolution,
                                    XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA,
                                    output, data, filter,
                                    XLA_MUSA_MUDNN_ALGORITHM_DIRECT,
                                    &workspace_size_bytes)
                  .ok());
  EXPECT_EQ(workspace_size_bytes, 8192);
  XlaMusaMuDnnWorkspaceAllocator workspace_allocator = {
      /*struct_size=*/sizeof(XlaMusaMuDnnWorkspaceAllocator),
      /*reserved=*/0,
      /*user_data=*/nullptr,
      /*allocate=*/FakeWorkspaceAllocate,
      /*release=*/FakeWorkspaceRelease,
  };
  EXPECT_TRUE(api->Convolve(handle, convolution,
                            XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER, output,
                            data, filter, XLA_MUSA_MUDNN_ALGORITHM_GEMM,
                            &workspace_allocator)
                  .ok());

  EXPECT_TRUE(api->DestroyConvolution(convolution).ok());
  EXPECT_TRUE(api->DestroyConvolution(nullptr).ok());
  EXPECT_TRUE(api->DestroyTensor(tensor).ok());
  EXPECT_TRUE(api->DestroyTensor(nullptr).ok());
  EXPECT_TRUE(api->DestroyHandle(handle).ok());
  EXPECT_TRUE(api->DestroyHandle(nullptr).ok());

  EXPECT_EQ(calls_.get_version, 1);
  EXPECT_EQ(calls_.create_handle, 1);
  EXPECT_EQ(calls_.device_ordinal, 3);
  EXPECT_EQ(calls_.set_stream, 1);
  EXPECT_EQ(calls_.stream, stream);
  EXPECT_EQ(calls_.set_allow_tf32, 1);
  EXPECT_EQ(calls_.allow_tf32, 1);
  EXPECT_EQ(calls_.create_tensor, 1);
  EXPECT_EQ(calls_.configure_tensor, 1);
  EXPECT_EQ(calls_.data_type, XLA_MUSA_MUDNN_DATA_TYPE_F16);
  EXPECT_EQ(calls_.format, XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW);
  EXPECT_EQ(calls_.rank, 4);
  EXPECT_EQ(calls_.dimensions[3], 7);
  EXPECT_EQ(calls_.tensor_strides[2], 7);
  EXPECT_EQ(calls_.set_tensor_address, 1);
  EXPECT_EQ(calls_.tensor_address, tensor_address);
  EXPECT_EQ(calls_.create_convolution, 1);
  EXPECT_EQ(calls_.configure_convolution, 1);
  EXPECT_EQ(calls_.spatial_rank, 2);
  EXPECT_EQ(calls_.padding[1], 2);
  EXPECT_EQ(calls_.convolution_strides[1], 3);
  EXPECT_EQ(calls_.dilations[1], 2);
  EXPECT_EQ(calls_.group_count, 4);
  EXPECT_EQ(calls_.get_recommended_algorithm, 1);
  EXPECT_EQ(calls_.get_workspace_size, 1);
  EXPECT_EQ(calls_.convolve, 1);
  EXPECT_EQ(calls_.kind, XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER);
  EXPECT_EQ(calls_.algorithm, XLA_MUSA_MUDNN_ALGORITHM_GEMM);
  EXPECT_EQ(calls_.workspace_allocator, &workspace_allocator);
  EXPECT_EQ(calls_.destroy_convolution, 1);
  EXPECT_EQ(calls_.destroy_tensor, 1);
  EXPECT_EQ(calls_.destroy_handle, 1);
}

TEST_F(MusaMuDnnApiTest, ConcurrentInitializationIsCallOnce) {
  auto loader =
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter));
  FakeSymbolLoader* loader_ptr = loader.get();
  std::unique_ptr<MusaMuDnnApi> api = CreateApi(std::move(loader));
  std::vector<absl::Status> statuses(16);
  std::vector<std::thread> threads;
  threads.reserve(statuses.size());
  for (int index = 0; index < statuses.size(); ++index) {
    threads.emplace_back([&, index] { statuses[index] = api->Init(); });
  }
  for (std::thread& thread : threads) thread.join();

  for (const absl::Status& status : statuses) EXPECT_TRUE(status.ok());
  EXPECT_EQ(loader_ptr->load_calls(), 1);
  EXPECT_EQ(calls_.get_version, 1);
}

TEST_F(MusaMuDnnApiTest, MissingShimRemainsAnOptionalNotFoundState) {
  std::unique_ptr<MusaMuDnnApi> api =
      CreateApi(std::make_unique<FakeSymbolLoader>(
          nullptr, absl::NotFoundError("optional muDNN shim absent")));
  EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kNotFound,
                                    HasSubstr("optional muDNN shim absent")));
  EXPECT_FALSE(api->IsLoaded());
  EXPECT_EQ(api->capabilities(), 0);
  EXPECT_EQ(api->abi_version(), 0);
}

TEST_F(MusaMuDnnApiTest, PresentShimWithoutGetterFailsClosed) {
  std::unique_ptr<MusaMuDnnApi> api =
      CreateApi(std::make_unique<FakeSymbolLoader>(nullptr));
  EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                    HasSubstr("does not export a usable")));
}

TEST_F(MusaMuDnnApiTest, RejectsMalformedApiTables) {
  {
    g_api = nullptr;
    std::unique_ptr<MusaMuDnnApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("null v1 API table")));
  }
  g_api = &api_table_;
  {
    api_table_.struct_size = XLA_MUSA_MUDNN_API_V1_MIN_STRUCT_SIZE - 1;
    std::unique_ptr<MusaMuDnnApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("too small")));
    api_table_ = CompleteApi();
  }
  {
    api_table_.abi_version = 99;
    std::unique_ptr<MusaMuDnnApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("ABI version 99")));
    api_table_ = CompleteApi();
  }
  {
    api_table_.capabilities &=
        ~XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_BACKWARD_FILTER;
    std::unique_ptr<MusaMuDnnApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("capabilities are incomplete")));
    api_table_ = CompleteApi();
  }
  {
    api_table_.convolve = nullptr;
    std::unique_ptr<MusaMuDnnApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("null required function")));
  }
}

TEST_F(MusaMuDnnApiTest, RejectsUnqualifiedVendorVersion) {
  g_version = {/*major=*/2, /*minor=*/8, /*patch=*/1};
  std::unique_ptr<MusaMuDnnApi> api = CreateApi(
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter)));
  EXPECT_THAT(api->Init(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("unqualified muDNN version 2.8.1")));
}

TEST_F(MusaMuDnnApiTest, RejectsInvalidRecommendedAlgorithmFromShim) {
  g_recommended_algorithm = XLA_MUSA_MUDNN_ALGORITHM_RECOMMENDED;
  std::unique_ptr<MusaMuDnnApi> api = CreateApi(
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter)));
  ASSERT_TRUE(api->Init().ok());
  XlaMusaMuDnnAlgorithm algorithm = XLA_MUSA_MUDNN_ALGORITHM_RECOMMENDED;
  EXPECT_THAT(
      api->GetRecommendedAlgorithm(
          reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000),
          XLA_MUSA_MUDNN_CONVOLUTION_FORWARD, reinterpret_cast<void*>(0x3000),
          reinterpret_cast<void*>(0x4000), reinterpret_cast<void*>(0x5000),
          &algorithm),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("invalid recommended algorithm")));
}

TEST_F(MusaMuDnnApiTest, NormalizesShimStatuses) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi(
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter)));
  ASSERT_TRUE(api->Init().ok());

  void* handle = nullptr;
  g_next_status = XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  EXPECT_THAT(api->CreateHandle(0, &handle),
              StatusIs(absl::StatusCode::kInvalidArgument));
  g_next_status = XLA_MUSA_MUDNN_STATUS_NOT_SUPPORTED;
  EXPECT_THAT(api->CreateHandle(0, &handle),
              StatusIs(absl::StatusCode::kUnimplemented));
  g_next_status = XLA_MUSA_MUDNN_STATUS_OUT_OF_RANGE;
  EXPECT_THAT(api->CreateHandle(0, &handle),
              StatusIs(absl::StatusCode::kOutOfRange));
  g_next_status = XLA_MUSA_MUDNN_STATUS_RESOURCE_EXHAUSTED;
  EXPECT_THAT(api->CreateHandle(0, &handle),
              StatusIs(absl::StatusCode::kResourceExhausted));
  g_next_status = XLA_MUSA_MUDNN_STATUS_FAILED_PRECONDITION;
  EXPECT_THAT(api->CreateHandle(0, &handle),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  g_next_status = XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR;
  EXPECT_THAT(api->CreateHandle(0, &handle),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(MusaMuDnnApiTest, ValidatesArgumentsBeforeDispatch) {
  std::unique_ptr<MusaMuDnnApi> api = CreateApi(
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter)));
  ASSERT_TRUE(api->Init().ok());

  EXPECT_THAT(api->GetVersion(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  void* handle = reinterpret_cast<void*>(0x1000);
  EXPECT_THAT(api->CreateHandle(-1, &handle),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(api->CreateHandle(0, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  handle = reinterpret_cast<void*>(0x1000);
  EXPECT_THAT(api->SetStream(nullptr, nullptr),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(api->SetAllowTf32(nullptr, false),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  EXPECT_THAT(api->CreateTensor(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  void* tensor = reinterpret_cast<void*>(0x2000);
  const int64_t dimensions[] = {2, 3, 5, 7};
  const int64_t strides[] = {105, 35, 7, 1};
  EXPECT_THAT(api->ConfigureTensor(nullptr, XLA_MUSA_MUDNN_DATA_TYPE_F32,
                                   XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW, 4,
                                   dimensions, strides),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(api->ConfigureTensor(
                  tensor, static_cast<XlaMusaMuDnnDataType>(99),
                  XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW, 4, dimensions, strides),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unknown normalized muDNN data type")));
  EXPECT_THAT(
      api->ConfigureTensor(tensor, XLA_MUSA_MUDNN_DATA_TYPE_F32,
                           XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW, 2, dimensions,
                           strides),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("ranks 3")));
  EXPECT_THAT(api->ConfigureTensor(tensor, XLA_MUSA_MUDNN_DATA_TYPE_F32,
                                   XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW, 4,
                                   dimensions, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(api->SetTensorAddress(tensor, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(api->CreateConvolution(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  void* convolution = reinterpret_cast<void*>(0x3000);
  const int64_t padding[] = {1, 1};
  const int64_t convolution_strides[] = {1, 1};
  const int64_t dilations[] = {1, 1};
  EXPECT_THAT(api->ConfigureConvolution(nullptr, 2, padding,
                                        convolution_strides, dilations, 1),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(api->ConfigureConvolution(convolution, 4, padding,
                                        convolution_strides, dilations, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(api->ConfigureConvolution(convolution, 2, padding,
                                        convolution_strides, dilations, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));

  void* output = reinterpret_cast<void*>(0x4000);
  void* data = reinterpret_cast<void*>(0x5000);
  void* filter = reinterpret_cast<void*>(0x6000);
  uint64_t workspace_size_bytes = 0;
  EXPECT_THAT(
      api->GetWorkspaceSize(
          handle, convolution, static_cast<XlaMusaMuDnnConvolutionKind>(99),
          output, data, filter, XLA_MUSA_MUDNN_ALGORITHM_DIRECT,
          &workspace_size_bytes),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("unknown normalized muDNN convolution kind")));
  EXPECT_THAT(api->GetWorkspaceSize(
                  handle, convolution, XLA_MUSA_MUDNN_CONVOLUTION_FORWARD,
                  output, data, filter, static_cast<XlaMusaMuDnnAlgorithm>(99),
                  &workspace_size_bytes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unknown normalized muDNN algorithm")));
  EXPECT_THAT(
      api->GetWorkspaceSize(handle, convolution,
                            XLA_MUSA_MUDNN_CONVOLUTION_FORWARD, output, data,
                            filter, XLA_MUSA_MUDNN_ALGORITHM_DIRECT, nullptr),
      StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(
      api->Convolve(handle, convolution, XLA_MUSA_MUDNN_CONVOLUTION_FORWARD,
                    output, data, filter, XLA_MUSA_MUDNN_ALGORITHM_DIRECT,
                    nullptr),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("non-null")));

  XlaMusaMuDnnWorkspaceAllocator short_allocator = {
      /*struct_size=*/
      XLA_MUSA_MUDNN_WORKSPACE_ALLOCATOR_MIN_STRUCT_SIZE - 1,
      /*reserved=*/0,
      /*user_data=*/nullptr,
      /*allocate=*/FakeWorkspaceAllocate,
      /*release=*/FakeWorkspaceRelease,
  };
  EXPECT_THAT(
      api->Convolve(handle, convolution, XLA_MUSA_MUDNN_CONVOLUTION_FORWARD,
                    output, data, filter, XLA_MUSA_MUDNN_ALGORITHM_DIRECT,
                    &short_allocator),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("too small")));
  XlaMusaMuDnnWorkspaceAllocator reserved_allocator = {
      /*struct_size=*/sizeof(XlaMusaMuDnnWorkspaceAllocator),
      /*reserved=*/1,
      /*user_data=*/nullptr,
      /*allocate=*/FakeWorkspaceAllocate,
      /*release=*/FakeWorkspaceRelease,
  };
  EXPECT_THAT(
      api->Convolve(handle, convolution, XLA_MUSA_MUDNN_CONVOLUTION_FORWARD,
                    output, data, filter, XLA_MUSA_MUDNN_ALGORITHM_DIRECT,
                    &reserved_allocator),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("reserved")));
  XlaMusaMuDnnWorkspaceAllocator null_callback_allocator = {
      /*struct_size=*/sizeof(XlaMusaMuDnnWorkspaceAllocator),
      /*reserved=*/0,
      /*user_data=*/nullptr,
      /*allocate=*/nullptr,
      /*release=*/FakeWorkspaceRelease,
  };
  EXPECT_THAT(
      api->Convolve(handle, convolution, XLA_MUSA_MUDNN_CONVOLUTION_FORWARD,
                    output, data, filter, XLA_MUSA_MUDNN_ALGORITHM_DIRECT,
                    &null_callback_allocator),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("callbacks")));

  EXPECT_EQ(calls_.create_handle, 0);
  EXPECT_EQ(calls_.set_stream, 0);
  EXPECT_EQ(calls_.set_allow_tf32, 0);
  EXPECT_EQ(calls_.configure_tensor, 0);
  EXPECT_EQ(calls_.set_tensor_address, 0);
  EXPECT_EQ(calls_.configure_convolution, 0);
  EXPECT_EQ(calls_.get_workspace_size, 0);
  EXPECT_EQ(calls_.convolve, 0);
}

TEST(MusaMuDnnAbiIdentityTest, IsCanonicalAndExact) {
  EXPECT_STREQ(kMusaMuDnnLibraryAbiName, "mudnn");
  EXPECT_STREQ(kMusaMuDnnLibraryAbiVersion, "1");
  EXPECT_STREQ(
      kMusaMuDnnAbiContractV1,
      "xla-musa-mudnn;abi=1;capabilities=32765;types=f16,f32;abi-rank=3-5;"
      "convolution-rank=4-5;layout=contiguous-channels-first;"
      "workspace=callback;stream=bound;tf32=controlled;"
      "algorithms=implicit-gemm,winograd-nonfused,gemm;"
      "convolution=forward,backward-data,backward-filter;mudnn=2.8.0");
  EXPECT_STREQ(
      kMusaMuDnnAbiFingerprintV1,
      "dfaab657ef752c2f591b8dd38c1310c39c0d6eecf785341f141942fa439a719a");
  EXPECT_EQ(std::string(kMusaMuDnnAbiFingerprintV1).size(), 64);
}

}  // namespace
}  // namespace stream_executor::musa
