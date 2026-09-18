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

#include "xla/stream_executor/musa/musa_fft.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
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
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/musa/mufft_shim/mufft_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_mufft_api.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::A;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;

enum class FftRoute {
  kC2C,
  kR2C,
  kC2R,
  kZ2Z,
  kD2Z,
  kZ2D,
};

struct PlanCall {
  void* plan = nullptr;
  int32_t rank = 0;
  std::vector<uint64_t> element_count;
  std::optional<std::vector<uint64_t>> input_embed;
  uint64_t input_stride = 0;
  uint64_t input_distance = 0;
  std::optional<std::vector<uint64_t>> output_embed;
  uint64_t output_stride = 0;
  uint64_t output_distance = 0;
  XlaMusaMuFftType type = 0;
  uint64_t batch_count = 0;
};

struct ExecCall {
  FftRoute route;
  void* plan = nullptr;
  void* input = nullptr;
  void* output = nullptr;
  XlaMusaMuFftDirection direction = 0;
};

// The C ABI has no user-data slot, so the fake table uses one process-local
// call recorder, following the other MUSA optional-library adapter tests.
// Its entry points are synchronized to make the API concurrency test valid.
struct FakeCalls {
  std::mutex mu;
  int get_versions = 0;
  int creates = 0;
  std::vector<void*> created_handles;
  std::vector<void*> destroyed_handles;
  std::vector<PlanCall> plans;
  std::vector<void*> work_areas;
  std::vector<void*> streams;
  std::vector<ExecCall> executions;

  uint64_t workspace_size_bytes = 256;
  XlaMusaMuFftStatus get_version_status = XLA_MUSA_MUFFT_STATUS_SUCCESS;
  XlaMusaMuFftStatus create_status = XLA_MUSA_MUFFT_STATUS_SUCCESS;
  XlaMusaMuFftStatus destroy_status = XLA_MUSA_MUFFT_STATUS_SUCCESS;
  XlaMusaMuFftStatus make_plan_status = XLA_MUSA_MUFFT_STATUS_SUCCESS;
  XlaMusaMuFftStatus set_work_area_status = XLA_MUSA_MUFFT_STATUS_SUCCESS;
  XlaMusaMuFftStatus set_stream_status = XLA_MUSA_MUFFT_STATUS_SUCCESS;
  XlaMusaMuFftStatus execution_status = XLA_MUSA_MUFFT_STATUS_SUCCESS;
};

FakeCalls* g_calls = nullptr;
const XlaMusaMuFftApiV1* g_api = nullptr;

XlaMusaMuFftStatus FakeGetVersion(XlaMusaMuFftVersion* version) {
  std::lock_guard<std::mutex> lock(g_calls->mu);
  ++g_calls->get_versions;
  if (g_calls->get_version_status == XLA_MUSA_MUFFT_STATUS_SUCCESS) {
    *version = {/*major=*/1, /*minor=*/6, /*patch=*/0};
  }
  return g_calls->get_version_status;
}

XlaMusaMuFftStatus FakeCreate(void** plan) {
  std::lock_guard<std::mutex> lock(g_calls->mu);
  ++g_calls->creates;
  if (g_calls->create_status == XLA_MUSA_MUFFT_STATUS_SUCCESS) {
    *plan = reinterpret_cast<void*>(
        static_cast<uintptr_t>(0x1000 + g_calls->creates * 0x100));
    g_calls->created_handles.push_back(*plan);
  }
  return g_calls->create_status;
}

XlaMusaMuFftStatus FakeDestroy(void* plan) {
  std::lock_guard<std::mutex> lock(g_calls->mu);
  g_calls->destroyed_handles.push_back(plan);
  return g_calls->destroy_status;
}

XlaMusaMuFftStatus FakeMakePlanMany(
    void* plan, int32_t rank, const uint64_t* element_count,
    const uint64_t* input_embed, uint64_t input_stride, uint64_t input_distance,
    const uint64_t* output_embed, uint64_t output_stride,
    uint64_t output_distance, XlaMusaMuFftType type, uint64_t batch_count,
    uint64_t* workspace_size_bytes) {
  std::lock_guard<std::mutex> lock(g_calls->mu);
  PlanCall call;
  call.plan = plan;
  call.rank = rank;
  call.element_count.assign(element_count, element_count + rank);
  if (input_embed != nullptr) {
    call.input_embed.emplace(input_embed, input_embed + rank);
  }
  call.input_stride = input_stride;
  call.input_distance = input_distance;
  if (output_embed != nullptr) {
    call.output_embed.emplace(output_embed, output_embed + rank);
  }
  call.output_stride = output_stride;
  call.output_distance = output_distance;
  call.type = type;
  call.batch_count = batch_count;
  g_calls->plans.push_back(std::move(call));
  if (g_calls->make_plan_status == XLA_MUSA_MUFFT_STATUS_SUCCESS) {
    *workspace_size_bytes = g_calls->workspace_size_bytes;
  }
  return g_calls->make_plan_status;
}

XlaMusaMuFftStatus FakeSetWorkArea(void*, void* workspace) {
  std::lock_guard<std::mutex> lock(g_calls->mu);
  g_calls->work_areas.push_back(workspace);
  return g_calls->set_work_area_status;
}

XlaMusaMuFftStatus FakeSetStream(void*, void* stream) {
  std::lock_guard<std::mutex> lock(g_calls->mu);
  g_calls->streams.push_back(stream);
  return g_calls->set_stream_status;
}

XlaMusaMuFftStatus RecordExecution(FftRoute route, void* plan, void* input,
                                   void* output,
                                   XlaMusaMuFftDirection direction = 0) {
  std::lock_guard<std::mutex> lock(g_calls->mu);
  g_calls->executions.push_back(
      ExecCall{route, plan, input, output, direction});
  return g_calls->execution_status;
}

XlaMusaMuFftStatus FakeExecC2C(void* plan, void* input, void* output,
                               XlaMusaMuFftDirection direction) {
  return RecordExecution(FftRoute::kC2C, plan, input, output, direction);
}

XlaMusaMuFftStatus FakeExecR2C(void* plan, void* input, void* output) {
  return RecordExecution(FftRoute::kR2C, plan, input, output);
}

XlaMusaMuFftStatus FakeExecC2R(void* plan, void* input, void* output) {
  return RecordExecution(FftRoute::kC2R, plan, input, output);
}

XlaMusaMuFftStatus FakeExecZ2Z(void* plan, void* input, void* output,
                               XlaMusaMuFftDirection direction) {
  return RecordExecution(FftRoute::kZ2Z, plan, input, output, direction);
}

XlaMusaMuFftStatus FakeExecD2Z(void* plan, void* input, void* output) {
  return RecordExecution(FftRoute::kD2Z, plan, input, output);
}

XlaMusaMuFftStatus FakeExecZ2D(void* plan, void* input, void* output) {
  return RecordExecution(FftRoute::kZ2D, plan, input, output);
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

class FakeLoader final : public internal::MusaSymbolLoader {
 public:
  explicit FakeLoader(
      void* getter = reinterpret_cast<void*>(&FakeGetter),
      absl::Status load_status = absl::OkStatus(),
      absl::Status resolve_status = absl::NotFoundError("missing getter"))
      : getter_(getter),
        load_status_(std::move(load_status)),
        resolve_status_(std::move(resolve_status)) {}

  absl::Status Load() override {
    ++load_calls_;
    return load_status_;
  }

  absl::StatusOr<void*> Resolve(absl::string_view symbol) const override {
    ++resolve_calls_;
    if (getter_ != nullptr && symbol == "xla_musa_mufft_get_api_v1") {
      return getter_;
    }
    return resolve_status_;
  }

  absl::string_view loaded_path() const override {
    return "fake-libxla_musa_mufft_shim.so.1";
  }

  int load_calls() const { return load_calls_.load(); }
  int resolve_calls() const { return resolve_calls_.load(); }

 private:
  void* getter_;
  absl::Status load_status_;
  absl::Status resolve_status_;
  std::atomic<int> load_calls_ = 0;
  mutable std::atomic<int> resolve_calls_ = 0;
};

class FakeScratchAllocator final : public ScratchAllocator {
 public:
  int64_t GetMemoryLimitInBytes() override { return memory_limit_; }

  absl::StatusOr<DeviceAddress<uint8_t>> AllocateBytes(
      int64_t byte_size) override {
    std::lock_guard<std::mutex> lock(mu_);
    sizes_.push_back(byte_size);
    if (next_failure_.has_value()) {
      absl::Status failure = *std::move(next_failure_);
      next_failure_.reset();
      return failure;
    }
    if (return_null_next_) {
      return_null_next_ = false;
      return DeviceAddress<uint8_t>();
    }
    void* address = reinterpret_cast<void*>(
        base_ + static_cast<uintptr_t>((sizes_.size() + 1) * 0x100));
    addresses_.push_back(address);
    return DeviceAddress<uint8_t>::MakeFromByteSize(address, byte_size);
  }

  void FailNext(absl::Status status) {
    std::lock_guard<std::mutex> lock(mu_);
    next_failure_ = std::move(status);
  }

  void ReturnNullNext() {
    std::lock_guard<std::mutex> lock(mu_);
    return_null_next_ = true;
  }

  std::vector<int64_t> sizes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return sizes_;
  }

  std::vector<void*> addresses() const {
    std::lock_guard<std::mutex> lock(mu_);
    return addresses_;
  }

 private:
  mutable std::mutex mu_;
  int64_t memory_limit_ = -1;
  uintptr_t base_ = 0x8000;
  std::optional<absl::Status> next_failure_;
  bool return_null_next_ = false;
  std::vector<int64_t> sizes_;
  std::vector<void*> addresses_;
};

class ForeignPlan final : public fft::Plan {};

template <typename T>
DeviceAddress<T> Address(uintptr_t address, uint64_t size) {
  return DeviceAddress<T>::MakeFromByteSize(reinterpret_cast<void*>(address),
                                            size);
}

class MusaFftTest : public ::testing::Test {
 protected:
  void SetUp() override {
    calls_ = std::make_unique<FakeCalls>();
    api_table_ = CompleteApi();
    g_calls = calls_.get();
    g_api = &api_table_;
  }

  void TearDown() override {
    g_calls = nullptr;
    g_api = nullptr;
  }

  void Configure(NiceMock<MockStreamExecutor>* executor,
                 NiceMock<MockStream>* stream,
                 void* native_stream = reinterpret_cast<void*>(0x2222)) {
    ON_CALL(*executor, Activate()).WillByDefault([] {
      return std::make_unique<ActivateContext>();
    });
    ON_CALL(*stream, parent()).WillByDefault(Return(executor));
    ON_CALL(*stream, platform_specific_handle())
        .WillByDefault(Return(Stream::PlatformSpecificHandle{native_stream}));
    ON_CALL(*stream, Memcpy(A<DeviceAddressBase*>(),
                            A<const DeviceAddressBase&>(), ::testing::_))
        .WillByDefault(Return(absl::OkStatus()));
  }

  std::unique_ptr<MusaMuFftApi> CreateApi() {
    return MusaMuFftApi::CreateForTesting(std::make_unique<FakeLoader>());
  }

  std::unique_ptr<fft::Plan> CreatePlan(MusaFft* fft_support, Stream* stream,
                                        fft::Type type,
                                        ScratchAllocator* scratch_allocator,
                                        int rank = 1, int batch_count = 1) {
    std::array<uint64_t, 3> element_count = {8, 4, 2};
    return fft_support->CreateBatchedPlanWithScratchAllocator(
        stream, rank, element_count.data(), /*input_embed=*/nullptr,
        /*input_stride=*/1, /*input_distance=*/8,
        /*output_embed=*/nullptr, /*output_stride=*/1,
        /*output_distance=*/8, type, /*in_place_fft=*/false, batch_count,
        scratch_allocator);
  }

  std::unique_ptr<FakeCalls> calls_;
  XlaMusaMuFftApiV1 api_table_ = {};
};

TEST_F(MusaFftTest, ForwardsPlanShapeLayoutBatchAndWorkspace) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 4096;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;

  std::array<uint64_t, 2> element_count = {8, 16};
  std::array<uint64_t, 2> input_embed = {10, 20};
  std::array<uint64_t, 2> output_embed = {12, 24};
  std::unique_ptr<fft::Plan> plan =
      fft_support.CreateBatchedPlanWithScratchAllocator(
          &stream, /*rank=*/2, element_count.data(), input_embed.data(),
          /*input_stride=*/2, /*input_distance=*/160, output_embed.data(),
          /*output_stride=*/3, /*output_distance=*/288, fft::Type::kR2C,
          /*in_place_fft=*/false, /*batch_count=*/7, &scratch);
  ASSERT_NE(plan, nullptr);
  auto* musa_plan = dynamic_cast<MusaFftPlan*>(plan.get());
  ASSERT_NE(musa_plan, nullptr);
  EXPECT_EQ(musa_plan->type(), fft::Type::kR2C);
  EXPECT_EQ(musa_plan->scratch_allocator(), &scratch);
  EXPECT_TRUE(musa_plan->status().ok());

  ASSERT_EQ(calls_->plans.size(), 1);
  const PlanCall& call = calls_->plans.front();
  EXPECT_EQ(call.rank, 2);
  EXPECT_THAT(call.element_count, ElementsAre(8, 16));
  ASSERT_TRUE(call.input_embed.has_value());
  EXPECT_THAT(*call.input_embed, ElementsAre(10, 20));
  EXPECT_EQ(call.input_stride, 2);
  EXPECT_EQ(call.input_distance, 160);
  ASSERT_TRUE(call.output_embed.has_value());
  EXPECT_THAT(*call.output_embed, ElementsAre(12, 24));
  EXPECT_EQ(call.output_stride, 3);
  EXPECT_EQ(call.output_distance, 288);
  EXPECT_EQ(call.type, XLA_MUSA_MUFFT_TYPE_R2C);
  EXPECT_EQ(call.batch_count, 7);
  EXPECT_THAT(scratch.sizes(), ElementsAre(4096));
  ASSERT_EQ(scratch.addresses().size(), 1);
  EXPECT_THAT(calls_->work_areas, ElementsAre(scratch.addresses().front()));
  EXPECT_TRUE(calls_->streams.empty());
}

TEST_F(MusaFftTest, SupportsRanksOneThroughThreeAndRejectsInvalidPlans) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 0;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;

  for (int rank = 1; rank <= 3; ++rank) {
    std::unique_ptr<fft::Plan> plan = CreatePlan(
        &fft_support, &stream, fft::Type::kC2CForward, &scratch, rank);
    ASSERT_NE(plan, nullptr);
  }
  ASSERT_EQ(calls_->plans.size(), 3);
  EXPECT_EQ(calls_->plans[0].rank, 1);
  EXPECT_EQ(calls_->plans[1].rank, 2);
  EXPECT_EQ(calls_->plans[2].rank, 3);
  for (const PlanCall& call : calls_->plans) {
    EXPECT_FALSE(call.input_embed.has_value());
    EXPECT_FALSE(call.output_embed.has_value());
  }

  std::array<uint64_t, 4> dimensions = {8, 4, 2, 1};
  EXPECT_EQ(fft_support.CreateBatchedPlanWithScratchAllocator(
                &stream, /*rank=*/0, dimensions.data(), nullptr, 1, 8, nullptr,
                1, 8, fft::Type::kC2CForward, false, 1, &scratch),
            nullptr);
  EXPECT_EQ(fft_support.CreateBatchedPlanWithScratchAllocator(
                &stream, /*rank=*/4, dimensions.data(), nullptr, 1, 8, nullptr,
                1, 8, fft::Type::kC2CForward, false, 1, &scratch),
            nullptr);
  EXPECT_EQ(CreatePlan(&fft_support, &stream, fft::Type::kC2CForward, &scratch,
                       /*rank=*/1, /*batch_count=*/0),
            nullptr);
  EXPECT_EQ(CreatePlan(&fft_support, &stream, fft::Type::kInvalid, &scratch),
            nullptr);
  EXPECT_EQ(fft_support.CreateBatchedPlanWithScratchAllocator(
                &stream, 1, dimensions.data(), nullptr, 1, 8, nullptr, 1, 8,
                fft::Type::kC2CForward, /*in_place_fft=*/true, 1, &scratch),
            nullptr);
  EXPECT_EQ(CreatePlan(&fft_support, &stream, fft::Type::kC2CForward,
                       /*scratch_allocator=*/nullptr),
            nullptr);
  EXPECT_EQ(calls_->creates, 3);
  EXPECT_EQ(calls_->destroyed_handles.size(), 3);
}

TEST_F(MusaFftTest, DispatchesAllSixRoutesAndBothComplexDirections) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 0;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;

  DeviceAddress<std::complex<float>> c64_input =
      Address<std::complex<float>>(0x3000, 64);
  DeviceAddress<std::complex<float>> c64_output =
      Address<std::complex<float>>(0x4000, 64);
  DeviceAddress<float> f32_input = Address<float>(0x5000, 32);
  DeviceAddress<float> f32_output = Address<float>(0x6000, 32);
  DeviceAddress<std::complex<double>> c128_input =
      Address<std::complex<double>>(0x7000, 128);
  DeviceAddress<std::complex<double>> c128_output =
      Address<std::complex<double>>(0x9000, 128);
  DeviceAddress<double> f64_input = Address<double>(0xa000, 64);
  DeviceAddress<double> f64_output = Address<double>(0xb000, 64);

  {
    auto plan =
        CreatePlan(&fft_support, &stream, fft::Type::kC2CForward, &scratch);
    ASSERT_NE(plan, nullptr);
    EXPECT_TRUE(fft_support.DoFft(&stream, plan.get(), c64_input, &c64_output));
  }
  {
    auto plan =
        CreatePlan(&fft_support, &stream, fft::Type::kC2CInverse, &scratch);
    ASSERT_NE(plan, nullptr);
    EXPECT_TRUE(fft_support.DoFft(&stream, plan.get(), c64_input, &c64_output));
  }
  {
    auto plan = CreatePlan(&fft_support, &stream, fft::Type::kR2C, &scratch);
    ASSERT_NE(plan, nullptr);
    EXPECT_TRUE(fft_support.DoFft(&stream, plan.get(), f32_input, &c64_output));
  }
  {
    auto plan = CreatePlan(&fft_support, &stream, fft::Type::kC2R, &scratch);
    ASSERT_NE(plan, nullptr);
    EXPECT_TRUE(fft_support.DoFft(&stream, plan.get(), c64_input, &f32_output));
  }
  {
    auto plan =
        CreatePlan(&fft_support, &stream, fft::Type::kZ2ZForward, &scratch);
    ASSERT_NE(plan, nullptr);
    EXPECT_TRUE(
        fft_support.DoFft(&stream, plan.get(), c128_input, &c128_output));
  }
  {
    auto plan =
        CreatePlan(&fft_support, &stream, fft::Type::kZ2ZInverse, &scratch);
    ASSERT_NE(plan, nullptr);
    EXPECT_TRUE(
        fft_support.DoFft(&stream, plan.get(), c128_input, &c128_output));
  }
  {
    auto plan = CreatePlan(&fft_support, &stream, fft::Type::kD2Z, &scratch);
    ASSERT_NE(plan, nullptr);
    EXPECT_TRUE(
        fft_support.DoFft(&stream, plan.get(), f64_input, &c128_output));
  }
  {
    auto plan = CreatePlan(&fft_support, &stream, fft::Type::kZ2D, &scratch);
    ASSERT_NE(plan, nullptr);
    EXPECT_TRUE(
        fft_support.DoFft(&stream, plan.get(), c128_input, &f64_output));
  }

  ASSERT_EQ(calls_->executions.size(), 8);
  EXPECT_EQ(calls_->executions[0].route, FftRoute::kC2C);
  EXPECT_EQ(calls_->executions[0].direction, XLA_MUSA_MUFFT_DIRECTION_FORWARD);
  EXPECT_EQ(calls_->executions[1].route, FftRoute::kC2C);
  EXPECT_EQ(calls_->executions[1].direction, XLA_MUSA_MUFFT_DIRECTION_INVERSE);
  EXPECT_EQ(calls_->executions[2].route, FftRoute::kR2C);
  EXPECT_EQ(calls_->executions[3].route, FftRoute::kC2R);
  EXPECT_EQ(calls_->executions[4].route, FftRoute::kZ2Z);
  EXPECT_EQ(calls_->executions[4].direction, XLA_MUSA_MUFFT_DIRECTION_FORWARD);
  EXPECT_EQ(calls_->executions[5].route, FftRoute::kZ2Z);
  EXPECT_EQ(calls_->executions[5].direction, XLA_MUSA_MUFFT_DIRECTION_INVERSE);
  EXPECT_EQ(calls_->executions[6].route, FftRoute::kD2Z);
  EXPECT_EQ(calls_->executions[7].route, FftRoute::kZ2D);
  EXPECT_EQ(calls_->streams.size(), calls_->executions.size());
  EXPECT_TRUE(std::all_of(
      calls_->streams.begin(), calls_->streams.end(),
      [](void* stream) { return stream == reinterpret_cast<void*>(0x2222); }));
  EXPECT_THAT(scratch.sizes(), ElementsAre(32, 64, 64, 128));
}

TEST_F(MusaFftTest, RebindsTheNativeStreamBeforeEveryDispatch) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> first_stream;
  NiceMock<MockStream> second_stream;
  Configure(&executor, &first_stream, reinterpret_cast<void*>(0x2222));
  Configure(&executor, &second_stream, reinterpret_cast<void*>(0x3333));
  calls_->workspace_size_bytes = 0;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;
  auto plan =
      CreatePlan(&fft_support, &first_stream, fft::Type::kC2CForward, &scratch);
  ASSERT_NE(plan, nullptr);
  DeviceAddress<std::complex<float>> input =
      Address<std::complex<float>>(0x4000, 64);
  DeviceAddress<std::complex<float>> output =
      Address<std::complex<float>>(0x5000, 64);

  EXPECT_TRUE(fft_support.DoFft(&first_stream, plan.get(), input, &output));
  EXPECT_TRUE(fft_support.DoFft(&second_stream, plan.get(), input, &output));

  EXPECT_THAT(calls_->streams, ElementsAre(reinterpret_cast<void*>(0x2222),
                                           reinterpret_cast<void*>(0x3333)));
  EXPECT_EQ(calls_->executions.size(), 2);
}

TEST_F(MusaFftTest, ReplacesWorkspaceAndUsesTheNewScratchAllocator) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 512;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator first_scratch;
  FakeScratchAllocator second_scratch;
  auto plan =
      CreatePlan(&fft_support, &stream, fft::Type::kR2C, &first_scratch);
  ASSERT_NE(plan, nullptr);
  auto* musa_plan = dynamic_cast<MusaFftPlan*>(plan.get());
  ASSERT_NE(musa_plan, nullptr);

  fft_support.UpdatePlanWithScratchAllocator(&stream, plan.get(),
                                             &second_scratch);
  EXPECT_TRUE(musa_plan->status().ok());
  EXPECT_EQ(musa_plan->scratch_allocator(), &second_scratch);
  EXPECT_THAT(first_scratch.sizes(), ElementsAre(512));
  EXPECT_THAT(second_scratch.sizes(), ElementsAre(512));
  ASSERT_EQ(first_scratch.addresses().size(), 1);
  ASSERT_EQ(second_scratch.addresses().size(), 1);
  EXPECT_THAT(calls_->work_areas,
              ElementsAre(first_scratch.addresses().front(),
                          second_scratch.addresses().front()));

  DeviceAddress<float> input = Address<float>(0x3000, 32);
  DeviceAddress<std::complex<float>> output =
      Address<std::complex<float>>(0x4000, 64);
  EXPECT_TRUE(fft_support.DoFft(&stream, plan.get(), input, &output));
  EXPECT_THAT(second_scratch.sizes(), ElementsAre(512, 32));
  EXPECT_THAT(first_scratch.sizes(), ElementsAre(512));
}

TEST_F(MusaFftTest, ZeroWorkspaceBindsNullWithoutAllocating) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 0;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;

  auto plan =
      CreatePlan(&fft_support, &stream, fft::Type::kC2CForward, &scratch);
  ASSERT_NE(plan, nullptr);
  EXPECT_TRUE(scratch.sizes().empty());
  EXPECT_THAT(calls_->work_areas, ElementsAre(nullptr));
}

TEST_F(MusaFftTest, RejectsFailedAndNullWorkspaceAllocations) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 256;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());

  FakeScratchAllocator failing_scratch;
  failing_scratch.FailNext(absl::ResourceExhaustedError("no workspace"));
  EXPECT_EQ(CreatePlan(&fft_support, &stream, fft::Type::kC2CForward,
                       &failing_scratch),
            nullptr);

  FakeScratchAllocator null_scratch;
  null_scratch.ReturnNullNext();
  EXPECT_EQ(
      CreatePlan(&fft_support, &stream, fft::Type::kC2CForward, &null_scratch),
      nullptr);

  EXPECT_EQ(calls_->creates, 2);
  EXPECT_TRUE(calls_->work_areas.empty());
  ASSERT_EQ(calls_->created_handles.size(), 2);
  ASSERT_EQ(calls_->destroyed_handles.size(), 2);
  EXPECT_THAT(
      calls_->destroyed_handles,
      ElementsAre(calls_->created_handles[0], calls_->created_handles[1]));
}

TEST_F(MusaFftTest, FailedWorkspaceUpdatePoisonsPlanWithoutDispatch) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 256;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator initial_scratch;
  auto plan =
      CreatePlan(&fft_support, &stream, fft::Type::kR2C, &initial_scratch);
  ASSERT_NE(plan, nullptr);
  auto* musa_plan = dynamic_cast<MusaFftPlan*>(plan.get());
  ASSERT_NE(musa_plan, nullptr);

  FakeScratchAllocator failing_scratch;
  failing_scratch.FailNext(absl::ResourceExhaustedError("update failed"));
  fft_support.UpdatePlanWithScratchAllocator(&stream, plan.get(),
                                             &failing_scratch);
  EXPECT_THAT(musa_plan->status(),
              StatusIs(absl::StatusCode::kResourceExhausted,
                       HasSubstr("update failed")));

  DeviceAddress<float> input = Address<float>(0x3000, 32);
  DeviceAddress<std::complex<float>> output =
      Address<std::complex<float>>(0x4000, 64);
  EXPECT_FALSE(fft_support.DoFft(&stream, plan.get(), input, &output));
  EXPECT_TRUE(calls_->streams.empty());
  EXPECT_TRUE(calls_->executions.empty());
}

TEST_F(MusaFftTest, RejectsForeignWrongTypeAndNullExecutionArguments) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 0;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;
  auto plan = CreatePlan(&fft_support, &stream, fft::Type::kR2C, &scratch);
  ASSERT_NE(plan, nullptr);

  DeviceAddress<float> input = Address<float>(0x3000, 32);
  DeviceAddress<std::complex<float>> complex_output =
      Address<std::complex<float>>(0x4000, 64);
  DeviceAddress<std::complex<float>> complex_input =
      Address<std::complex<float>>(0x5000, 64);
  DeviceAddress<float> real_output = Address<float>(0x6000, 32);
  ForeignPlan foreign_plan;
  EXPECT_FALSE(
      fft_support.DoFft(&stream, &foreign_plan, input, &complex_output));
  EXPECT_FALSE(
      fft_support.DoFft(&stream, plan.get(), complex_input, &complex_output));

  DeviceAddress<float> null_input;
  EXPECT_FALSE(
      fft_support.DoFft(&stream, plan.get(), null_input, &complex_output));
  EXPECT_FALSE(fft_support.DoFft(
      &stream, plan.get(), input,
      static_cast<DeviceAddress<std::complex<float>>*>(nullptr)));
  DeviceAddress<std::complex<float>> null_output;
  EXPECT_FALSE(fft_support.DoFft(&stream, plan.get(), input, &null_output));
  EXPECT_FALSE(fft_support.DoFft(nullptr, plan.get(), input, &complex_output));

  // A C2R-typed overload must also reject the R2C plan before dispatch.
  EXPECT_FALSE(
      fft_support.DoFft(&stream, plan.get(), complex_input, &real_output));
  EXPECT_TRUE(calls_->streams.empty());
  EXPECT_TRUE(calls_->executions.empty());
}

TEST_F(MusaFftTest, PreservesRealInputWithScratchCopyBeforeExecution) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 0;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;
  auto plan = CreatePlan(&fft_support, &stream, fft::Type::kR2C, &scratch);
  ASSERT_NE(plan, nullptr);

  DeviceAddress<float> input = Address<float>(0x3000, 64);
  DeviceAddress<std::complex<float>> output =
      Address<std::complex<float>>(0x4000, 80);
  void* copied_input = nullptr;
  EXPECT_CALL(stream, Memcpy(A<DeviceAddressBase*>(),
                             A<const DeviceAddressBase&>(), input.size()))
      .WillOnce(Invoke([&](DeviceAddressBase* destination,
                           const DeviceAddressBase& source, uint64_t size) {
        copied_input = destination->opaque();
        EXPECT_EQ(source.opaque(), input.opaque());
        EXPECT_EQ(source.size(), input.size());
        EXPECT_EQ(size, input.size());
        return absl::OkStatus();
      }));

  EXPECT_TRUE(fft_support.DoFft(&stream, plan.get(), input, &output));

  EXPECT_EQ(input.opaque(), reinterpret_cast<void*>(0x3000));
  EXPECT_THAT(scratch.sizes(), ElementsAre(64));
  ASSERT_EQ(scratch.addresses().size(), 1);
  EXPECT_EQ(copied_input, scratch.addresses().front());
  ASSERT_EQ(calls_->executions.size(), 1);
  EXPECT_EQ(calls_->executions.front().route, FftRoute::kR2C);
  EXPECT_EQ(calls_->executions.front().input, copied_input);
  EXPECT_NE(calls_->executions.front().input, input.opaque());
  EXPECT_EQ(calls_->executions.front().output, output.opaque());
}

TEST_F(MusaFftTest, ComplexTransformDoesNotCopyInput) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 0;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;
  auto plan =
      CreatePlan(&fft_support, &stream, fft::Type::kC2CForward, &scratch);
  ASSERT_NE(plan, nullptr);
  DeviceAddress<std::complex<float>> input =
      Address<std::complex<float>>(0x3000, 64);
  DeviceAddress<std::complex<float>> output =
      Address<std::complex<float>>(0x4000, 64);
  EXPECT_CALL(stream, Memcpy(A<DeviceAddressBase*>(),
                             A<const DeviceAddressBase&>(), ::testing::_))
      .Times(0);

  EXPECT_TRUE(fft_support.DoFft(&stream, plan.get(), input, &output));
  ASSERT_EQ(calls_->executions.size(), 1);
  EXPECT_EQ(calls_->executions.front().input, input.opaque());
  EXPECT_TRUE(scratch.sizes().empty());
}

TEST_F(MusaFftTest, DestroysEachCreatedHandleExactlyOnce) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  calls_->workspace_size_bytes = 0;
  auto api = CreateApi();
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;

  std::unique_ptr<fft::Plan> plan =
      CreatePlan(&fft_support, &stream, fft::Type::kC2CForward, &scratch);
  ASSERT_NE(plan, nullptr);
  void* successful_handle = dynamic_cast<MusaFftPlan*>(plan.get())->handle();
  std::unique_ptr<fft::Plan> moved_plan = std::move(plan);
  EXPECT_TRUE(calls_->destroyed_handles.empty());
  moved_plan.reset();
  EXPECT_EQ(std::count(calls_->destroyed_handles.begin(),
                       calls_->destroyed_handles.end(), successful_handle),
            1);

  calls_->make_plan_status = XLA_MUSA_MUFFT_STATUS_NOT_SUPPORTED;
  EXPECT_EQ(CreatePlan(&fft_support, &stream, fft::Type::kC2CForward, &scratch),
            nullptr);
  ASSERT_EQ(calls_->created_handles.size(), 2);
  void* failed_handle = calls_->created_handles.back();
  EXPECT_EQ(std::count(calls_->destroyed_handles.begin(),
                       calls_->destroyed_handles.end(), failed_handle),
            1);
  EXPECT_EQ(calls_->destroyed_handles.size(), 2);
}

TEST_F(MusaFftTest, MissingShimFailsPlanCreationAndStaysCached) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  Configure(&executor, &stream);
  auto loader = std::make_unique<FakeLoader>(
      /*getter=*/nullptr, absl::NotFoundError("optional muFFT shim absent"));
  FakeLoader* loader_ptr = loader.get();
  auto api = MusaMuFftApi::CreateForTesting(std::move(loader));
  MusaFft fft_support(&executor, api.get());
  FakeScratchAllocator scratch;
  std::array<uint64_t, 1> element_count = {8};

  MusaFftPlan direct_plan(&executor, api.get());
  EXPECT_THAT(direct_plan.Initialize(&stream, 1, element_count.data(), nullptr,
                                     1, 8, nullptr, 1, 8,
                                     fft::Type::kC2CForward, 1, &scratch),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("optional muFFT shim absent")));
  EXPECT_EQ(CreatePlan(&fft_support, &stream, fft::Type::kC2CForward, &scratch),
            nullptr);

  EXPECT_EQ(loader_ptr->load_calls(), 1);
  EXPECT_EQ(loader_ptr->resolve_calls(), 0);
  EXPECT_EQ(calls_->get_versions, 0);
  EXPECT_EQ(calls_->creates, 0);
}

TEST_F(MusaFftTest, ApiInitializationAndIndependentPlansAreConcurrentSafe) {
  auto loader = std::make_unique<FakeLoader>();
  FakeLoader* loader_ptr = loader.get();
  auto api = MusaMuFftApi::CreateForTesting(std::move(loader));
  constexpr int kThreadCount = 16;
  std::array<absl::Status, kThreadCount> statuses;
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (int index = 0; index < kThreadCount; ++index) {
    threads.emplace_back([&, index] { statuses[index] = api->Init(); });
  }
  for (std::thread& thread : threads) thread.join();
  for (const absl::Status& status : statuses) EXPECT_TRUE(status.ok());
  EXPECT_EQ(loader_ptr->load_calls(), 1);
  EXPECT_EQ(loader_ptr->resolve_calls(), 1);
  EXPECT_EQ(calls_->get_versions, 1);

  threads.clear();
  for (int index = 0; index < kThreadCount; ++index) {
    threads.emplace_back([&, index] {
      void* plan = nullptr;
      statuses[index] = api->Create(&plan);
      if (statuses[index].ok()) statuses[index] = api->Destroy(plan);
    });
  }
  for (std::thread& thread : threads) thread.join();
  for (const absl::Status& status : statuses) EXPECT_TRUE(status.ok());
  EXPECT_EQ(calls_->creates, kThreadCount);
  EXPECT_EQ(calls_->created_handles.size(), kThreadCount);
  EXPECT_EQ(calls_->destroyed_handles.size(), kThreadCount);
  for (void* handle : calls_->created_handles) {
    EXPECT_EQ(std::count(calls_->destroyed_handles.begin(),
                         calls_->destroyed_handles.end(), handle),
              1);
  }
}

}  // namespace
}  // namespace stream_executor::musa
