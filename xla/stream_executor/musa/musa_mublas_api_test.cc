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

#include "xla/stream_executor/musa/musa_mublas_api.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/stream_executor/musa/mublas_shim/mublas_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/tsl/platform/env.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

struct FakeCalls {
  int create = 0;
  int destroy = 0;
  int set_stream = 0;
  int get_version = 0;
  int gemm = 0;
  int set_atomics_mode = 0;
  int gemm_with_algorithm = 0;
  int gemm_batched = 0;
  int gemm_strided_batched = 0;
  void* stream = nullptr;
  bool allow_atomics = false;
  XlaMusaMuBlasDataType input_type = 0;
  XlaMusaMuBlasDataType output_type = 0;
  XlaMusaMuBlasComputeType compute_type = 0;
  XlaMusaMuBlasAlgorithm algorithm = 0;
  int64_t batch_count = 0;
  int64_t stride_a = 0;
  int64_t stride_b = 0;
  int64_t stride_c = 0;
};

FakeCalls* g_calls = nullptr;
XlaMusaMuBlasApiV1* g_api_v1 = nullptr;
XlaMusaMuBlasApiV2* g_api_v2 = nullptr;

XlaMusaMuBlasStatus FakeCreate(void** handle) {
  ++g_calls->create;
  *handle = reinterpret_cast<void*>(0x1234);
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus FakeDestroy(void*) {
  ++g_calls->destroy;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus FakeSetStream(void*, void* stream) {
  ++g_calls->set_stream;
  g_calls->stream = stream;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus FakeGetVersion(void*, int32_t* version) {
  ++g_calls->get_version;
  *version = 40100;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus FakeGemm(void*, XlaMusaMuBlasDataType input_type,
                             XlaMusaMuBlasDataType output_type,
                             XlaMusaMuBlasComputeType compute_type,
                             XlaMusaMuBlasOperation, XlaMusaMuBlasOperation,
                             int64_t, int64_t, int64_t, const void*,
                             const void*, int64_t, const void*, int64_t,
                             const void*, void*, int64_t) {
  ++g_calls->gemm;
  g_calls->input_type = input_type;
  g_calls->output_type = output_type;
  g_calls->compute_type = compute_type;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus FakeSetAtomicsMode(void*, uint32_t allow_atomics) {
  ++g_calls->set_atomics_mode;
  g_calls->allow_atomics = allow_atomics != 0;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus FakeGemmWithAlgorithm(
    void*, XlaMusaMuBlasDataType input_type, XlaMusaMuBlasDataType output_type,
    XlaMusaMuBlasComputeType compute_type, XlaMusaMuBlasOperation,
    XlaMusaMuBlasOperation, int64_t, int64_t, int64_t, const void*, const void*,
    int64_t, const void*, int64_t, const void*, void*, int64_t,
    XlaMusaMuBlasAlgorithm algorithm) {
  ++g_calls->gemm_with_algorithm;
  g_calls->input_type = input_type;
  g_calls->output_type = output_type;
  g_calls->compute_type = compute_type;
  g_calls->algorithm = algorithm;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus FakeGemmBatched(
    void*, XlaMusaMuBlasDataType, XlaMusaMuBlasDataType,
    XlaMusaMuBlasComputeType, XlaMusaMuBlasOperation, XlaMusaMuBlasOperation,
    int64_t, int64_t, int64_t, const void*, const void* const*, int64_t,
    const void* const*, int64_t, const void*, void* const*, int64_t,
    int64_t batch_count, XlaMusaMuBlasAlgorithm algorithm) {
  ++g_calls->gemm_batched;
  g_calls->batch_count = batch_count;
  g_calls->algorithm = algorithm;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

XlaMusaMuBlasStatus FakeGemmStridedBatched(
    void*, XlaMusaMuBlasDataType, XlaMusaMuBlasDataType,
    XlaMusaMuBlasComputeType, XlaMusaMuBlasOperation, XlaMusaMuBlasOperation,
    int64_t, int64_t, int64_t, const void*, const void*, int64_t,
    int64_t stride_a, const void*, int64_t, int64_t stride_b, const void*,
    void*, int64_t, int64_t stride_c, int64_t batch_count,
    XlaMusaMuBlasAlgorithm algorithm) {
  ++g_calls->gemm_strided_batched;
  g_calls->stride_a = stride_a;
  g_calls->stride_b = stride_b;
  g_calls->stride_c = stride_c;
  g_calls->batch_count = batch_count;
  g_calls->algorithm = algorithm;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}

const XlaMusaMuBlasApiV1* FakeGetterV1() { return g_api_v1; }
const XlaMusaMuBlasApiV2* FakeGetterV2() { return g_api_v2; }

XlaMusaMuBlasApiV1 CompleteApi() {
  XlaMusaMuBlasApiV1 api = {};
  api.struct_size = sizeof(api);
  api.abi_version = XLA_MUSA_MUBLAS_ABI_VERSION_1;
  api.capabilities = XLA_MUSA_MUBLAS_CAPABILITIES_V1;
  api.create = FakeCreate;
  api.destroy = FakeDestroy;
  api.set_stream = FakeSetStream;
  api.get_version = FakeGetVersion;
  api.gemm = FakeGemm;
  return api;
}

XlaMusaMuBlasApiV2 CompleteApiV2() {
  XlaMusaMuBlasApiV2 api = {};
  api.struct_size = sizeof(api);
  api.abi_version = XLA_MUSA_MUBLAS_ABI_VERSION_2;
  api.capabilities = XLA_MUSA_MUBLAS_CAPABILITIES_V1;
  api.advanced_capabilities = XLA_MUSA_MUBLAS_ADVANCED_CAPABILITIES_V2;
  api.create = FakeCreate;
  api.destroy = FakeDestroy;
  api.set_stream = FakeSetStream;
  api.get_version = FakeGetVersion;
  api.gemm = FakeGemm;
  api.set_atomics_mode = FakeSetAtomicsMode;
  api.gemm_with_algorithm = FakeGemmWithAlgorithm;
  api.gemm_batched = FakeGemmBatched;
  api.gemm_strided_batched = FakeGemmStridedBatched;
  return api;
}

class FakeSymbolLoader final : public internal::MusaSymbolLoader {
 public:
  explicit FakeSymbolLoader(void* v1_getter,
                            absl::Status load_status = absl::OkStatus())
      : v1_getter_(v1_getter), load_status_(std::move(load_status)) {}

  FakeSymbolLoader(void* v1_getter, void* v2_getter,
                   absl::Status load_status = absl::OkStatus())
      : v1_getter_(v1_getter),
        v2_getter_(v2_getter),
        load_status_(std::move(load_status)) {}

  absl::Status Load() override {
    ++load_calls_;
    return load_status_;
  }

  absl::StatusOr<void*> Resolve(absl::string_view symbol) const override {
    resolved_.push_back(std::string(symbol));
    if (symbol == "xla_musa_mublas_get_api_v2" && v2_getter_ != nullptr) {
      return v2_getter_;
    }
    if (symbol == "xla_musa_mublas_get_api_v1" && v1_getter_ != nullptr) {
      return v1_getter_;
    }
    return absl::NotFoundError(std::string(symbol));
  }

  absl::string_view loaded_path() const override {
    return "fake-libxla_musa_mublas_shim.so.1";
  }

  int load_calls() const { return load_calls_; }
  const std::vector<std::string>& resolved() const { return resolved_; }

 private:
  void* v1_getter_;
  void* v2_getter_ = nullptr;
  absl::Status load_status_;
  int load_calls_ = 0;
  mutable std::vector<std::string> resolved_;
};

class MusaMuBlasApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    calls_ = {};
    api_ = CompleteApi();
    api_v2_ = CompleteApiV2();
    g_calls = &calls_;
    g_api_v1 = &api_;
    g_api_v2 = &api_v2_;
  }

  void TearDown() override {
    g_calls = nullptr;
    g_api_v1 = nullptr;
    g_api_v2 = nullptr;
  }

  FakeCalls calls_;
  XlaMusaMuBlasApiV1 api_ = {};
  XlaMusaMuBlasApiV2 api_v2_ = {};
};

TEST_F(MusaMuBlasApiTest, LoadsOnlyVersionedGetterAndDispatchesTypedCalls) {
  auto loader = std::make_unique<FakeSymbolLoader>(
      reinterpret_cast<void*>(&FakeGetterV1));
  FakeSymbolLoader* loader_ptr = loader.get();
  std::unique_ptr<MusaMuBlasApi> api =
      MusaMuBlasApi::CreateForTesting(std::move(loader));

  ASSERT_EQ(api_.capabilities, UINT64_C(0x7));
  EXPECT_TRUE(api->Init().ok());
  EXPECT_TRUE(api->Init().ok());
  EXPECT_EQ(loader_ptr->load_calls(), 1);
  EXPECT_THAT(
      loader_ptr->resolved(),
      ElementsAre("xla_musa_mublas_get_api_v2", "xla_musa_mublas_get_api_v1"));
  EXPECT_EQ(api->capabilities(), api_.capabilities);

  void* handle = nullptr;
  EXPECT_TRUE(api->Create(&handle).ok());
  ASSERT_NE(handle, nullptr);
  EXPECT_TRUE(api->SetStream(handle, reinterpret_cast<void*>(0x5678)).ok());
  int32_t version = 0;
  EXPECT_TRUE(api->GetVersion(handle, &version).ok());
  EXPECT_EQ(version, 40100);
  float alpha = 1.0f;
  float beta = 0.0f;
  EXPECT_TRUE(api->Gemm(handle, XLA_MUSA_MUBLAS_DATA_TYPE_F16,
                        XLA_MUSA_MUBLAS_DATA_TYPE_F16,
                        XLA_MUSA_MUBLAS_COMPUTE_TYPE_F16,
                        XLA_MUSA_MUBLAS_OPERATION_NONE,
                        XLA_MUSA_MUBLAS_OPERATION_TRANSPOSE, 2, 3, 4, &alpha,
                        reinterpret_cast<void*>(0x1000), 2,
                        reinterpret_cast<void*>(0x2000), 3, &beta,
                        reinterpret_cast<void*>(0x3000), 2)
                  .ok());
  EXPECT_TRUE(api->Destroy(handle).ok());

  EXPECT_EQ(calls_.create, 1);
  EXPECT_EQ(calls_.destroy, 1);
  EXPECT_EQ(calls_.set_stream, 1);
  EXPECT_EQ(calls_.stream, reinterpret_cast<void*>(0x5678));
  EXPECT_EQ(calls_.get_version, 1);
  EXPECT_EQ(calls_.gemm, 1);
  EXPECT_EQ(calls_.input_type, XLA_MUSA_MUBLAS_DATA_TYPE_F16);
  EXPECT_EQ(calls_.output_type, XLA_MUSA_MUBLAS_DATA_TYPE_F16);
  EXPECT_EQ(calls_.compute_type, XLA_MUSA_MUBLAS_COMPUTE_TYPE_F16);
}

TEST_F(MusaMuBlasApiTest, PrefersV2AndDispatchesAdvancedOperations) {
  auto loader = std::make_unique<FakeSymbolLoader>(
      reinterpret_cast<void*>(&FakeGetterV1),
      reinterpret_cast<void*>(&FakeGetterV2));
  FakeSymbolLoader* loader_ptr = loader.get();
  std::unique_ptr<MusaMuBlasApi> api =
      MusaMuBlasApi::CreateForTesting(std::move(loader));

  ASSERT_TRUE(api->Init().ok());
  EXPECT_THAT(loader_ptr->resolved(),
              ElementsAre("xla_musa_mublas_get_api_v2"));
  EXPECT_EQ(api->abi_version(), XLA_MUSA_MUBLAS_ABI_VERSION_2);
  EXPECT_EQ(api->advanced_capabilities(),
            XLA_MUSA_MUBLAS_ADVANCED_CAPABILITIES_V2);
  EXPECT_TRUE(api->SupportsSetAtomicsMode());
  EXPECT_TRUE(api->SupportsGemmWithAlgorithm());
  EXPECT_TRUE(api->SupportsGemmBatched());
  EXPECT_TRUE(api->SupportsGemmStridedBatched());
  EXPECT_TRUE(api->SupportsTensorOpF32());
  EXPECT_TRUE(api->UsesZeroExternalWorkspace());
  EXPECT_EQ(api->advanced_abi_fingerprint(),
            kMusaMuBlasAdvancedAbiFingerprintV2);

  void* handle = nullptr;
  ASSERT_TRUE(api->Create(&handle).ok());
  ASSERT_TRUE(api->SetAtomicsMode(handle, false).ok());
  EXPECT_FALSE(calls_.allow_atomics);

  float scalar = 1.0f;
  ASSERT_TRUE(api->GemmWithAlgorithm(handle, XLA_MUSA_MUBLAS_DATA_TYPE_F32,
                                     XLA_MUSA_MUBLAS_DATA_TYPE_F32,
                                     XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32,
                                     XLA_MUSA_MUBLAS_OPERATION_NONE,
                                     XLA_MUSA_MUBLAS_OPERATION_NONE, 2, 3, 4,
                                     &scalar, reinterpret_cast<void*>(0x1000),
                                     2, reinterpret_cast<void*>(0x2000), 4,
                                     &scalar, reinterpret_cast<void*>(0x3000),
                                     2, XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP)
                  .ok());
  EXPECT_EQ(calls_.algorithm, XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP);

  auto* a = reinterpret_cast<const void* const*>(0x4000);
  auto* b = reinterpret_cast<const void* const*>(0x5000);
  auto* c = reinterpret_cast<void* const*>(0x6000);
  ASSERT_TRUE(api->GemmBatched(handle, XLA_MUSA_MUBLAS_DATA_TYPE_F32,
                               XLA_MUSA_MUBLAS_DATA_TYPE_F32,
                               XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32,
                               XLA_MUSA_MUBLAS_OPERATION_NONE,
                               XLA_MUSA_MUBLAS_OPERATION_NONE, 2, 3, 4, &scalar,
                               a, 2, b, 4, &scalar, c, 2, 7,
                               XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT)
                  .ok());
  EXPECT_EQ(calls_.batch_count, 7);

  ASSERT_TRUE(
      api->GemmStridedBatched(
             handle, XLA_MUSA_MUBLAS_DATA_TYPE_F32,
             XLA_MUSA_MUBLAS_DATA_TYPE_F32, XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32,
             XLA_MUSA_MUBLAS_OPERATION_NONE, XLA_MUSA_MUBLAS_OPERATION_NONE, 2,
             3, 4, &scalar, reinterpret_cast<void*>(0x1000), 2, 8,
             reinterpret_cast<void*>(0x2000), 4, 12, &scalar,
             reinterpret_cast<void*>(0x3000), 2, 6, 7,
             XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP)
          .ok());
  EXPECT_EQ(calls_.stride_a, 8);
  EXPECT_EQ(calls_.stride_b, 12);
  EXPECT_EQ(calls_.stride_c, 6);
  EXPECT_EQ(calls_.batch_count, 7);

  EXPECT_THAT(
      api->GemmWithAlgorithm(
          handle, XLA_MUSA_MUBLAS_DATA_TYPE_F32, XLA_MUSA_MUBLAS_DATA_TYPE_F32,
          XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32, XLA_MUSA_MUBLAS_OPERATION_NONE,
          XLA_MUSA_MUBLAS_OPERATION_NONE, 1, 1, 1, &scalar, &scalar, 1, &scalar,
          1, &scalar, &scalar, 1, static_cast<XlaMusaMuBlasAlgorithm>(99)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("unknown normalized")));
  EXPECT_THAT(
      api->GemmWithAlgorithm(
          handle, XLA_MUSA_MUBLAS_DATA_TYPE_F16, XLA_MUSA_MUBLAS_DATA_TYPE_F16,
          XLA_MUSA_MUBLAS_COMPUTE_TYPE_F16, XLA_MUSA_MUBLAS_OPERATION_NONE,
          XLA_MUSA_MUBLAS_OPERATION_NONE, 1, 1, 1, &scalar, &scalar, 1, &scalar,
          1, &scalar, &scalar, 1, XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("only for homogeneous F32")));
  EXPECT_EQ(calls_.gemm_with_algorithm, 1);
}

TEST_F(MusaMuBlasApiTest, V1FallbackRejectsAdvancedOperations) {
  auto loader = std::make_unique<FakeSymbolLoader>(
      reinterpret_cast<void*>(&FakeGetterV1));
  std::unique_ptr<MusaMuBlasApi> api =
      MusaMuBlasApi::CreateForTesting(std::move(loader));
  void* handle = nullptr;
  ASSERT_TRUE(api->Create(&handle).ok());
  EXPECT_EQ(api->abi_version(), XLA_MUSA_MUBLAS_ABI_VERSION_1);
  EXPECT_EQ(api->advanced_capabilities(), 0);
  EXPECT_TRUE(api->advanced_abi_fingerprint().empty());
  EXPECT_THAT(api->SetAtomicsMode(handle, false),
              StatusIs(absl::StatusCode::kUnimplemented));
  float scalar = 1.0f;
  EXPECT_THAT(
      api->GemmWithAlgorithm(
          handle, XLA_MUSA_MUBLAS_DATA_TYPE_F32, XLA_MUSA_MUBLAS_DATA_TYPE_F32,
          XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32, XLA_MUSA_MUBLAS_OPERATION_NONE,
          XLA_MUSA_MUBLAS_OPERATION_NONE, 1, 1, 1, &scalar, &scalar, 1, &scalar,
          1, &scalar, &scalar, 1, XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("only the default")));
}

TEST_F(MusaMuBlasApiTest, MissingShimRemainsNotFoundAndCached) {
  auto loader = std::make_unique<FakeSymbolLoader>(
      nullptr, absl::NotFoundError("shim absent"));
  FakeSymbolLoader* loader_ptr = loader.get();
  std::unique_ptr<MusaMuBlasApi> api =
      MusaMuBlasApi::CreateForTesting(std::move(loader));

  EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(loader_ptr->load_calls(), 1);
  EXPECT_TRUE(loader_ptr->resolved().empty());
}

TEST_F(MusaMuBlasApiTest, RejectsMissingGetterAndMalformedTables) {
  {
    auto loader = std::make_unique<FakeSymbolLoader>(nullptr);
    std::unique_ptr<MusaMuBlasApi> api =
        MusaMuBlasApi::CreateForTesting(std::move(loader));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("xla_musa_mublas_get_api_v1")));
  }
  {
    api_v2_.advanced_capabilities &=
        ~XLA_MUSA_MUBLAS_ADVANCED_CAPABILITY_GEMM_BATCHED;
    auto loader = std::make_unique<FakeSymbolLoader>(
        reinterpret_cast<void*>(&FakeGetterV1),
        reinterpret_cast<void*>(&FakeGetterV2));
    std::unique_ptr<MusaMuBlasApi> api =
        MusaMuBlasApi::CreateForTesting(std::move(loader));
    EXPECT_THAT(api->Init(),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("advanced capabilities are incomplete")));
    api_v2_ = CompleteApiV2();
  }
  {
    api_.abi_version = 2;
    auto loader = std::make_unique<FakeSymbolLoader>(
        reinterpret_cast<void*>(&FakeGetterV1));
    std::unique_ptr<MusaMuBlasApi> api =
        MusaMuBlasApi::CreateForTesting(std::move(loader));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("ABI version 2")));
  }
  {
    api_ = CompleteApi();
    api_.gemm = nullptr;
    auto loader = std::make_unique<FakeSymbolLoader>(
        reinterpret_cast<void*>(&FakeGetterV1));
    std::unique_ptr<MusaMuBlasApi> api =
        MusaMuBlasApi::CreateForTesting(std::move(loader));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("null required function")));
  }
  {
    api_ = CompleteApi();
    api_.capabilities &= ~XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F64;
    auto loader = std::make_unique<FakeSymbolLoader>(
        reinterpret_cast<void*>(&FakeGetterV1));
    std::unique_ptr<MusaMuBlasApi> api =
        MusaMuBlasApi::CreateForTesting(std::move(loader));
    EXPECT_THAT(api->Init(),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("capabilities are incomplete")));
  }
}

TEST_F(MusaMuBlasApiTest, ConvertsVendorNeutralStatusCodes) {
  api_.gemm = [](void*, XlaMusaMuBlasDataType, XlaMusaMuBlasDataType,
                 XlaMusaMuBlasComputeType, XlaMusaMuBlasOperation,
                 XlaMusaMuBlasOperation, int64_t, int64_t, int64_t, const void*,
                 const void*, int64_t, const void*, int64_t, const void*, void*,
                 int64_t) -> XlaMusaMuBlasStatus {
    return XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED;
  };
  auto loader = std::make_unique<FakeSymbolLoader>(
      reinterpret_cast<void*>(&FakeGetterV1));
  std::unique_ptr<MusaMuBlasApi> api =
      MusaMuBlasApi::CreateForTesting(std::move(loader));
  void* handle = nullptr;
  ASSERT_TRUE(api->Create(&handle).ok());
  float scalar = 1.0f;
  EXPECT_THAT(
      api->Gemm(handle, XLA_MUSA_MUBLAS_DATA_TYPE_F32,
                XLA_MUSA_MUBLAS_DATA_TYPE_F32, XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32,
                XLA_MUSA_MUBLAS_OPERATION_NONE, XLA_MUSA_MUBLAS_OPERATION_NONE,
                1, 1, 1, &scalar, &scalar, 1, &scalar, 1, &scalar, &scalar, 1),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("status 2")));
}

TEST(MusaMuBlasShimCandidatesTest, ExplicitPathIsAbsoluteAndFailClosed) {
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> candidates,
      internal::GetMusaMuBlasShimCandidates(
          std::optional<absl::string_view>("/qualified/libmublas-shim.so.1"),
          "/plugin/libpjrt_musa.so"));
  EXPECT_THAT(candidates, ElementsAre("/qualified/libmublas-shim.so.1"));

  EXPECT_THAT(internal::GetMusaMuBlasShimCandidates(
                  std::optional<absl::string_view>("relative/shim.so"),
                  "/plugin/libpjrt_musa.so"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be an absolute path")));
  std::string oversized(4097, 'x');
  oversized.front() = '/';
  EXPECT_THAT(internal::GetMusaMuBlasShimCandidates(
                  std::optional<absl::string_view>(oversized),
                  "/plugin/libpjrt_musa.so"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds 4096 bytes")));
}

TEST(MusaMuBlasShimCandidatesTest, DefaultsToAdjacentThenSoname) {
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> candidates,
                       internal::GetMusaMuBlasShimCandidates(
                           std::nullopt, "/plugin/libpjrt_musa.so"));
  EXPECT_THAT(candidates, ElementsAre("/plugin/libxla_musa_mublas_shim.so.1",
                                      "libxla_musa_mublas_shim.so.1"));
}

TEST(MusaMuBlasShimCandidatesTest, ExplicitMissingOrUnloadableShimFailsClosed) {
  tsl::Env* env = tsl::Env::Default();
  std::string invalid_dso;
  ASSERT_TRUE(env->LocalTempFilename(&invalid_dso));

  std::unique_ptr<MusaMuBlasApi> missing_api =
      MusaMuBlasApi::CreateForTesting(internal::CreateMusaMuBlasShimLoader(
          std::optional<absl::string_view>(invalid_dso),
          "/plugin/libpjrt_musa.so"));
  EXPECT_THAT(missing_api->Init(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Could not load a MUSA shared library")));

  ASSERT_TRUE(tsl::WriteStringToFile(env, invalid_dso, "not an ELF DSO").ok());

  std::unique_ptr<MusaMuBlasApi> api =
      MusaMuBlasApi::CreateForTesting(internal::CreateMusaMuBlasShimLoader(
          std::optional<absl::string_view>(invalid_dso),
          "/plugin/libpjrt_musa.so"));
  EXPECT_THAT(api->Init(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Could not load a MUSA shared library")));
  EXPECT_TRUE(env->DeleteFile(invalid_dso).ok());
}

}  // namespace
}  // namespace stream_executor::musa
