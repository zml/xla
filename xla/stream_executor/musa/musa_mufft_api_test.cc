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

#include "xla/stream_executor/musa/musa_mufft_api.h"

#include <cstdint>
#include <cstdlib>
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
#include "xla/stream_executor/musa/mufft_shim/mufft_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

struct FakeCalls {
  int get_version = 0;
  int create = 0;
  int destroy = 0;
  int make_plan_many = 0;
  int set_work_area = 0;
  int set_stream = 0;
  int exec_c2c = 0;
  int exec_r2c = 0;
  int exec_c2r = 0;
  int exec_z2z = 0;
  int exec_d2z = 0;
  int exec_z2d = 0;
  int32_t rank = 0;
  uint64_t element_count[3] = {};
  uint64_t input_embed[3] = {};
  uint64_t output_embed[3] = {};
  uint64_t input_stride = 0;
  uint64_t input_distance = 0;
  uint64_t output_stride = 0;
  uint64_t output_distance = 0;
  uint64_t batch_count = 0;
  XlaMusaMuFftType type = 0;
  XlaMusaMuFftDirection direction = 0;
  void* workspace = nullptr;
  void* stream = nullptr;
};

FakeCalls* g_calls = nullptr;
XlaMusaMuFftApiV1* g_api = nullptr;
XlaMusaMuFftStatus g_next_status = XLA_MUSA_MUFFT_STATUS_SUCCESS;
XlaMusaMuFftVersion g_version = {/*major=*/1, /*minor=*/6, /*patch=*/0};

XlaMusaMuFftStatus FakeGetVersion(XlaMusaMuFftVersion* version) {
  ++g_calls->get_version;
  if (g_next_status == XLA_MUSA_MUFFT_STATUS_SUCCESS) *version = g_version;
  return g_next_status;
}

XlaMusaMuFftStatus FakeCreate(void** plan) {
  ++g_calls->create;
  if (g_next_status == XLA_MUSA_MUFFT_STATUS_SUCCESS) {
    *plan = reinterpret_cast<void*>(0x1234);
  }
  return g_next_status;
}

XlaMusaMuFftStatus FakeDestroy(void*) {
  ++g_calls->destroy;
  return g_next_status;
}

XlaMusaMuFftStatus FakeMakePlanMany(
    void*, int32_t rank, const uint64_t* element_count,
    const uint64_t* input_embed, uint64_t input_stride, uint64_t input_distance,
    const uint64_t* output_embed, uint64_t output_stride,
    uint64_t output_distance, XlaMusaMuFftType type, uint64_t batch_count,
    uint64_t* workspace_size_bytes) {
  ++g_calls->make_plan_many;
  g_calls->rank = rank;
  for (int32_t index = 0; index < rank; ++index) {
    g_calls->element_count[index] = element_count[index];
    if (input_embed != nullptr) {
      g_calls->input_embed[index] = input_embed[index];
    }
    if (output_embed != nullptr) {
      g_calls->output_embed[index] = output_embed[index];
    }
  }
  g_calls->input_stride = input_stride;
  g_calls->input_distance = input_distance;
  g_calls->output_stride = output_stride;
  g_calls->output_distance = output_distance;
  g_calls->batch_count = batch_count;
  g_calls->type = type;
  if (g_next_status == XLA_MUSA_MUFFT_STATUS_SUCCESS) {
    *workspace_size_bytes = 4096;
  }
  return g_next_status;
}

XlaMusaMuFftStatus FakeSetWorkArea(void*, void* workspace) {
  ++g_calls->set_work_area;
  g_calls->workspace = workspace;
  return g_next_status;
}

XlaMusaMuFftStatus FakeSetStream(void*, void* stream) {
  ++g_calls->set_stream;
  g_calls->stream = stream;
  return g_next_status;
}

XlaMusaMuFftStatus FakeExecC2C(void*, void*, void*,
                               XlaMusaMuFftDirection direction) {
  ++g_calls->exec_c2c;
  g_calls->direction = direction;
  return g_next_status;
}

XlaMusaMuFftStatus FakeExecR2C(void*, void*, void*) {
  ++g_calls->exec_r2c;
  return g_next_status;
}

XlaMusaMuFftStatus FakeExecC2R(void*, void*, void*) {
  ++g_calls->exec_c2r;
  return g_next_status;
}

XlaMusaMuFftStatus FakeExecZ2Z(void*, void*, void*,
                               XlaMusaMuFftDirection direction) {
  ++g_calls->exec_z2z;
  g_calls->direction = direction;
  return g_next_status;
}

XlaMusaMuFftStatus FakeExecD2Z(void*, void*, void*) {
  ++g_calls->exec_d2z;
  return g_next_status;
}

XlaMusaMuFftStatus FakeExecZ2D(void*, void*, void*) {
  ++g_calls->exec_z2d;
  return g_next_status;
}

const XlaMusaMuFftApiV1* FakeGetter() { return g_api; }

XlaMusaMuFftApiV1 CompleteApi() {
  XlaMusaMuFftApiV1 api = {};
  api.struct_size = sizeof(api);
  api.abi_version = XLA_MUSA_MUFFT_ABI_VERSION_1;
  api.capabilities = XLA_MUSA_MUFFT_CAPABILITIES_V1;
  api.get_version = FakeGetVersion;
  api.create = FakeCreate;
  api.destroy = FakeDestroy;
  api.make_plan_many = FakeMakePlanMany;
  api.set_work_area = FakeSetWorkArea;
  api.set_stream = FakeSetStream;
  api.exec_c2c = FakeExecC2C;
  api.exec_r2c = FakeExecR2C;
  api.exec_c2r = FakeExecC2R;
  api.exec_z2z = FakeExecZ2Z;
  api.exec_d2z = FakeExecD2Z;
  api.exec_z2d = FakeExecZ2D;
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
    if (getter_ != nullptr && symbol == "xla_musa_mufft_get_api_v1") {
      return getter_;
    }
    return resolve_status_;
  }

  absl::string_view loaded_path() const override {
    return "fake-libxla_musa_mufft_shim.so.1";
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

class MusaMuFftApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    calls_ = {};
    api_table_ = CompleteApi();
    g_calls = &calls_;
    g_api = &api_table_;
    g_next_status = XLA_MUSA_MUFFT_STATUS_SUCCESS;
    g_version = {/*major=*/1, /*minor=*/6, /*patch=*/0};
  }

  void TearDown() override {
    g_calls = nullptr;
    g_api = nullptr;
  }

  std::unique_ptr<MusaMuFftApi> CreateApi(
      std::unique_ptr<internal::MusaSymbolLoader> loader) {
    return MusaMuFftApi::CreateForTesting(std::move(loader));
  }

  FakeCalls calls_;
  XlaMusaMuFftApiV1 api_table_ = {};
};

TEST(MusaMuFftShimCandidatesTest, UsesExplicitAbsolutePathOnly) {
  absl::StatusOr<std::vector<std::string>> candidates =
      internal::GetMusaMuFftShimCandidates(
          std::optional<absl::string_view>("/opt/musa/lib/custom-mufft.so"),
          "/plugins/libpjrt_musa.so");
  ASSERT_TRUE(candidates.ok());
  EXPECT_THAT(*candidates, ElementsAre("/opt/musa/lib/custom-mufft.so"));
}

TEST(MusaMuFftShimCandidatesTest, UsesAdjacentThenVersionedSoname) {
  absl::StatusOr<std::vector<std::string>> candidates =
      internal::GetMusaMuFftShimCandidates(std::nullopt,
                                           "/plugins/libpjrt_musa.so");
  ASSERT_TRUE(candidates.ok());
  EXPECT_THAT(*candidates, ElementsAre("/plugins/libxla_musa_mufft_shim.so.1",
                                       "libxla_musa_mufft_shim.so.1"));
}

TEST(MusaMuFftShimCandidatesTest,
     AdjacentClosurePathsAreAbsoluteAndDependencyOrdered) {
  absl::StatusOr<std::vector<std::string>> paths =
      internal::GetMusaMuFftAdjacentClosurePaths(
          "/plugins/libxla_musa_mufft_shim.so.1");
  ASSERT_TRUE(paths.ok());
  EXPECT_THAT(*paths, ElementsAre("/plugins/libmusart.so.1.5",
                                  "/plugins/libmtfft-device-1.so",
                                  "/plugins/libmtfft-device-2.so",
                                  "/plugins/libmtfft-device-3.so",
                                  "/plugins/libmtfft-device-0.so"));
  EXPECT_THAT(
      internal::GetMusaMuFftAdjacentClosurePaths("libxla_musa_mufft_shim.so.1"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("must be absolute")));
}

TEST(MusaMuFftLiveClosureTest, LoadsExplicitCopiedClosureWhenRequested) {
  const char* shim_path = std::getenv("XLA_MUSA_MUFFT_TEST_SHIM_PATH");
  if (shim_path == nullptr || shim_path[0] == '\0') {
    GTEST_SKIP() << "set XLA_MUSA_MUFFT_TEST_SHIM_PATH for live closure test";
  }
  std::unique_ptr<MusaMuFftApi> api =
      MusaMuFftApi::CreateForTesting(internal::CreateMusaMuFftShimLoader(
          std::optional<absl::string_view>(shim_path), /*plugin_path=*/""));
  ASSERT_TRUE(api->Init().ok()) << api->Init();
  EXPECT_EQ(api->abi_version(), XLA_MUSA_MUFFT_ABI_VERSION_1);
  EXPECT_EQ(api->capabilities(), XLA_MUSA_MUFFT_CAPABILITIES_V1);
  XlaMusaMuFftVersion version = {};
  ASSERT_TRUE(api->GetVersion(&version).ok());
  EXPECT_EQ(version.major, 1);
  EXPECT_EQ(version.minor, 6);
  EXPECT_EQ(version.patch, 0);
}

TEST(MusaMuFftShimCandidatesTest, RejectsInvalidConfiguredPaths) {
  EXPECT_THAT(
      internal::GetMusaMuFftShimCandidates(std::optional<absl::string_view>(""),
                                           "/plugins/libpjrt_musa.so"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("must not be")));
  EXPECT_THAT(
      internal::GetMusaMuFftShimCandidates(
          std::optional<absl::string_view>("relative/libmufft.so"),
          "/plugins/libpjrt_musa.so"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("absolute")));
  const std::string oversized_path(4097, 'x');
  EXPECT_THAT(internal::GetMusaMuFftShimCandidates(
                  std::optional<absl::string_view>(oversized_path),
                  "/plugins/libpjrt_musa.so"),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("4096")));
}

TEST_F(MusaMuFftApiTest, LoadsOnceAndDispatchesEveryTypedOperation) {
  auto loader =
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter));
  FakeSymbolLoader* loader_ptr = loader.get();
  std::unique_ptr<MusaMuFftApi> api = CreateApi(std::move(loader));

  ASSERT_TRUE(api->Init().ok());
  EXPECT_TRUE(api->Init().ok());
  EXPECT_EQ(loader_ptr->load_calls(), 1);
  EXPECT_THAT(loader_ptr->resolved(), ElementsAre("xla_musa_mufft_get_api_v1"));
  EXPECT_EQ(api->abi_version(), XLA_MUSA_MUFFT_ABI_VERSION_1);
  EXPECT_EQ(api->capabilities(), XLA_MUSA_MUFFT_CAPABILITIES_V1);
  EXPECT_EQ(api->loaded_path(), "fake-libxla_musa_mufft_shim.so.1");

  XlaMusaMuFftVersion version = {};
  ASSERT_TRUE(api->GetVersion(&version).ok());
  EXPECT_EQ(version.major, 1);
  EXPECT_EQ(version.minor, 6);
  EXPECT_EQ(version.patch, 0);

  void* plan = nullptr;
  ASSERT_TRUE(api->Create(&plan).ok());
  ASSERT_NE(plan, nullptr);
  const uint64_t element_count[] = {8, 16};
  const uint64_t input_embed[] = {10, 20};
  const uint64_t output_embed[] = {12, 24};
  uint64_t workspace_size_bytes = 0;
  ASSERT_TRUE(api->MakePlanMany(plan, 2, element_count, input_embed, 2, 160,
                                output_embed, 3, 288, XLA_MUSA_MUFFT_TYPE_C2C,
                                7, &workspace_size_bytes)
                  .ok());
  EXPECT_EQ(workspace_size_bytes, 4096);
  EXPECT_EQ(calls_.rank, 2);
  EXPECT_EQ(calls_.element_count[0], 8);
  EXPECT_EQ(calls_.element_count[1], 16);
  EXPECT_EQ(calls_.input_embed[1], 20);
  EXPECT_EQ(calls_.output_embed[1], 24);
  EXPECT_EQ(calls_.input_stride, 2);
  EXPECT_EQ(calls_.input_distance, 160);
  EXPECT_EQ(calls_.output_stride, 3);
  EXPECT_EQ(calls_.output_distance, 288);
  EXPECT_EQ(calls_.batch_count, 7);
  EXPECT_EQ(calls_.type, XLA_MUSA_MUFFT_TYPE_C2C);

  void* workspace = reinterpret_cast<void*>(0x4000);
  void* stream = reinterpret_cast<void*>(0x5000);
  void* input = reinterpret_cast<void*>(0x6000);
  void* output = reinterpret_cast<void*>(0x7000);
  EXPECT_TRUE(api->SetWorkArea(plan, workspace).ok());
  EXPECT_TRUE(api->SetStream(plan, stream).ok());
  EXPECT_TRUE(
      api->ExecC2C(plan, input, output, XLA_MUSA_MUFFT_DIRECTION_FORWARD).ok());
  EXPECT_TRUE(api->ExecR2C(plan, input, output).ok());
  EXPECT_TRUE(api->ExecC2R(plan, input, output).ok());
  EXPECT_TRUE(
      api->ExecZ2Z(plan, input, output, XLA_MUSA_MUFFT_DIRECTION_INVERSE).ok());
  EXPECT_TRUE(api->ExecD2Z(plan, input, output).ok());
  EXPECT_TRUE(api->ExecZ2D(plan, input, output).ok());
  EXPECT_TRUE(api->Destroy(plan).ok());
  EXPECT_TRUE(api->Destroy(nullptr).ok());

  EXPECT_EQ(calls_.get_version, 1);
  EXPECT_EQ(calls_.create, 1);
  EXPECT_EQ(calls_.destroy, 1);
  EXPECT_EQ(calls_.make_plan_many, 1);
  EXPECT_EQ(calls_.set_work_area, 1);
  EXPECT_EQ(calls_.workspace, workspace);
  EXPECT_EQ(calls_.set_stream, 1);
  EXPECT_EQ(calls_.stream, stream);
  EXPECT_EQ(calls_.exec_c2c, 1);
  EXPECT_EQ(calls_.exec_r2c, 1);
  EXPECT_EQ(calls_.exec_c2r, 1);
  EXPECT_EQ(calls_.exec_z2z, 1);
  EXPECT_EQ(calls_.exec_d2z, 1);
  EXPECT_EQ(calls_.exec_z2d, 1);
  EXPECT_EQ(calls_.direction, XLA_MUSA_MUFFT_DIRECTION_INVERSE);
}

TEST_F(MusaMuFftApiTest, ConcurrentInitializationIsCallOnce) {
  auto loader =
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter));
  FakeSymbolLoader* loader_ptr = loader.get();
  std::unique_ptr<MusaMuFftApi> api = CreateApi(std::move(loader));
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

TEST_F(MusaMuFftApiTest, MissingShimRemainsAnOptionalNotFoundState) {
  std::unique_ptr<MusaMuFftApi> api =
      CreateApi(std::make_unique<FakeSymbolLoader>(
          nullptr, absl::NotFoundError("optional muFFT shim absent")));
  EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kNotFound,
                                    HasSubstr("optional muFFT shim absent")));
  EXPECT_FALSE(api->IsLoaded());
  EXPECT_EQ(api->capabilities(), 0);
  EXPECT_EQ(api->abi_version(), 0);
}

TEST_F(MusaMuFftApiTest, PresentShimWithoutGetterFailsClosed) {
  std::unique_ptr<MusaMuFftApi> api =
      CreateApi(std::make_unique<FakeSymbolLoader>(nullptr));
  EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                    HasSubstr("does not export a usable")));
}

TEST_F(MusaMuFftApiTest, RejectsMalformedApiTables) {
  {
    g_api = nullptr;
    std::unique_ptr<MusaMuFftApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("null v1 API table")));
  }
  g_api = &api_table_;
  {
    api_table_.struct_size = XLA_MUSA_MUFFT_API_V1_MIN_STRUCT_SIZE - 1;
    std::unique_ptr<MusaMuFftApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("too small")));
    api_table_ = CompleteApi();
  }
  {
    api_table_.abi_version = 99;
    std::unique_ptr<MusaMuFftApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("ABI version 99")));
    api_table_ = CompleteApi();
  }
  {
    api_table_.capabilities &= ~XLA_MUSA_MUFFT_CAPABILITY_Z2D;
    std::unique_ptr<MusaMuFftApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("capabilities are incomplete")));
    api_table_ = CompleteApi();
  }
  {
    api_table_.exec_d2z = nullptr;
    std::unique_ptr<MusaMuFftApi> api =
        CreateApi(std::make_unique<FakeSymbolLoader>(
            reinterpret_cast<void*>(&FakeGetter)));
    EXPECT_THAT(api->Init(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("null required function")));
  }
}

TEST_F(MusaMuFftApiTest, RejectsUnqualifiedVendorVersion) {
  g_version = {/*major=*/1, /*minor=*/7, /*patch=*/0};
  std::unique_ptr<MusaMuFftApi> api = CreateApi(
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter)));
  EXPECT_THAT(api->Init(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("unqualified muFFT version 1.7.0")));
}

TEST_F(MusaMuFftApiTest, NormalizesShimStatuses) {
  std::unique_ptr<MusaMuFftApi> api = CreateApi(
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter)));
  ASSERT_TRUE(api->Init().ok());

  void* plan = nullptr;
  g_next_status = XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  EXPECT_THAT(api->Create(&plan), StatusIs(absl::StatusCode::kInvalidArgument));
  g_next_status = XLA_MUSA_MUFFT_STATUS_NOT_SUPPORTED;
  EXPECT_THAT(api->Create(&plan), StatusIs(absl::StatusCode::kUnimplemented));
  g_next_status = XLA_MUSA_MUFFT_STATUS_OUT_OF_RANGE;
  EXPECT_THAT(api->Create(&plan), StatusIs(absl::StatusCode::kOutOfRange));
  g_next_status = XLA_MUSA_MUFFT_STATUS_RESOURCE_EXHAUSTED;
  EXPECT_THAT(api->Create(&plan),
              StatusIs(absl::StatusCode::kResourceExhausted));
  g_next_status = XLA_MUSA_MUFFT_STATUS_FAILED_PRECONDITION;
  EXPECT_THAT(api->Create(&plan),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  g_next_status = XLA_MUSA_MUFFT_STATUS_VENDOR_ERROR;
  EXPECT_THAT(api->Create(&plan), StatusIs(absl::StatusCode::kInternal));
}

TEST_F(MusaMuFftApiTest, ValidatesArgumentsBeforeDispatch) {
  std::unique_ptr<MusaMuFftApi> api = CreateApi(
      std::make_unique<FakeSymbolLoader>(reinterpret_cast<void*>(&FakeGetter)));
  ASSERT_TRUE(api->Init().ok());

  EXPECT_THAT(api->GetVersion(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(api->Create(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  const uint64_t element_count[] = {8};
  uint64_t workspace_size_bytes = 123;
  EXPECT_THAT(
      api->MakePlanMany(nullptr, 1, element_count, nullptr, 1, 8, nullptr, 1, 8,
                        XLA_MUSA_MUFFT_TYPE_C2C, 1, &workspace_size_bytes),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  void* plan = reinterpret_cast<void*>(0x1000);
  EXPECT_THAT(
      api->MakePlanMany(plan, 0, element_count, nullptr, 1, 8, nullptr, 1, 8,
                        XLA_MUSA_MUFFT_TYPE_C2C, 1, &workspace_size_bytes),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      api->MakePlanMany(plan, 4, element_count, nullptr, 1, 8, nullptr, 1, 8,
                        XLA_MUSA_MUFFT_TYPE_C2C, 1, &workspace_size_bytes),
      StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(api->MakePlanMany(plan, 1, element_count, nullptr, 1, 8, nullptr,
                                1, 8, static_cast<XlaMusaMuFftType>(99), 1,
                                &workspace_size_bytes),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unknown normalized muFFT type")));
  EXPECT_THAT(api->SetWorkArea(nullptr, nullptr),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(api->SetStream(nullptr, nullptr),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  void* input = reinterpret_cast<void*>(0x2000);
  void* output = reinterpret_cast<void*>(0x3000);
  EXPECT_THAT(
      api->ExecC2C(plan, input, output, static_cast<XlaMusaMuFftDirection>(99)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("unknown normalized muFFT direction")));
  EXPECT_THAT(api->ExecR2C(plan, nullptr, output),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_EQ(calls_.make_plan_many, 0);
  EXPECT_EQ(calls_.set_work_area, 0);
  EXPECT_EQ(calls_.set_stream, 0);
  EXPECT_EQ(calls_.exec_c2c, 0);
  EXPECT_EQ(calls_.exec_r2c, 0);
}

TEST(MusaMuFftAbiIdentityTest, IsCanonicalAndExact) {
  EXPECT_STREQ(kMusaMuFftLibraryAbiName, "mufft");
  EXPECT_STREQ(kMusaMuFftLibraryAbiVersion, "1");
  EXPECT_STREQ(
      kMusaMuFftAbiContractV1,
      "xla-musa-mufft;abi=1;capabilities=511;rank=1-3;layout=i32;"
      "workspace=external;stream=bound;inverse=unnormalized;mufft=1.6.0");
  EXPECT_STREQ(
      kMusaMuFftAbiFingerprintV1,
      "51a66971dfa8bc0259ac291a5c7f3d1674822f453ed3bf405aa63a28bdc6df27");
  EXPECT_EQ(std::string(kMusaMuFftAbiFingerprintV1).size(), 64);
}

}  // namespace
}  // namespace stream_executor::musa
