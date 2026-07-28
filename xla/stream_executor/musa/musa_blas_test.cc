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

#include "xla/stream_executor/musa/musa_blas.h"

#include <array>
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
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/musa/mublas_shim/mublas_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_mublas_api.h"
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

struct GemmCall {
  int creates = 0;
  int destroys = 0;
  int set_streams = 0;
  int gemms = 0;
  int algorithm_gemms = 0;
  int batched_gemms = 0;
  int strided_batched_gemms = 0;
  int atomics_calls = 0;
  bool allow_atomics = true;
  int64_t batch_count = 0;
  int64_t stride_a = 0;
  int64_t stride_b = 0;
  int64_t stride_c = 0;
  XlaMusaMuBlasAlgorithm algorithm = XLA_MUSA_MUBLAS_ALGORITHM_DEFAULT;
  void* stream = nullptr;
  XlaMusaMuBlasDataType input_type = 0;
  XlaMusaMuBlasDataType output_type = 0;
  XlaMusaMuBlasComputeType compute_type = 0;
  XlaMusaMuBlasOperation trans_a = 0;
  XlaMusaMuBlasOperation trans_b = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

GemmCall* g_call = nullptr;
XlaMusaMuBlasApiV1* g_table = nullptr;
XlaMusaMuBlasApiV2* g_table_v2 = nullptr;

XlaMusaMuBlasStatus Create(void** handle) {
  ++g_call->creates;
  *handle = reinterpret_cast<void*>(0x1100 + g_call->creates);
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}
XlaMusaMuBlasStatus SetAtomicsMode(void*, uint32_t allow_atomics) {
  ++g_call->atomics_calls;
  g_call->allow_atomics = allow_atomics != 0;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}
XlaMusaMuBlasStatus Destroy(void*) {
  ++g_call->destroys;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}
XlaMusaMuBlasStatus SetStream(void*, void* stream) {
  ++g_call->set_streams;
  g_call->stream = stream;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}
XlaMusaMuBlasStatus GetVersion(void*, int32_t* version) {
  *version = 40100;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}
XlaMusaMuBlasStatus Gemm(void*, XlaMusaMuBlasDataType input_type,
                         XlaMusaMuBlasDataType output_type,
                         XlaMusaMuBlasComputeType compute_type,
                         XlaMusaMuBlasOperation trans_a,
                         XlaMusaMuBlasOperation trans_b, int64_t m, int64_t n,
                         int64_t k, const void*, const void*, int64_t,
                         const void*, int64_t, const void*, void*, int64_t) {
  ++g_call->gemms;
  g_call->input_type = input_type;
  g_call->output_type = output_type;
  g_call->compute_type = compute_type;
  g_call->trans_a = trans_a;
  g_call->trans_b = trans_b;
  g_call->m = m;
  g_call->n = n;
  g_call->k = k;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}
XlaMusaMuBlasStatus GemmWithAlgorithm(
    void* handle, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
    XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* a, int64_t lda,
    const void* b, int64_t ldb, const void* beta, void* c, int64_t ldc,
    XlaMusaMuBlasAlgorithm algorithm) {
  ++g_call->algorithm_gemms;
  g_call->algorithm = algorithm;
  return Gemm(handle, input_type, output_type, compute_type, trans_a, trans_b,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
XlaMusaMuBlasStatus GemmBatched(void*, XlaMusaMuBlasDataType,
                                XlaMusaMuBlasDataType, XlaMusaMuBlasComputeType,
                                XlaMusaMuBlasOperation, XlaMusaMuBlasOperation,
                                int64_t, int64_t, int64_t, const void*,
                                const void* const*, int64_t, const void* const*,
                                int64_t, const void*, void* const*, int64_t,
                                int64_t batch_count,
                                XlaMusaMuBlasAlgorithm algorithm) {
  ++g_call->batched_gemms;
  g_call->batch_count = batch_count;
  g_call->algorithm = algorithm;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}
XlaMusaMuBlasStatus GemmStridedBatched(
    void*, XlaMusaMuBlasDataType, XlaMusaMuBlasDataType,
    XlaMusaMuBlasComputeType, XlaMusaMuBlasOperation, XlaMusaMuBlasOperation,
    int64_t, int64_t, int64_t, const void*, const void*, int64_t,
    int64_t stride_a, const void*, int64_t, int64_t stride_b, const void*,
    void*, int64_t, int64_t stride_c, int64_t batch_count,
    XlaMusaMuBlasAlgorithm algorithm) {
  ++g_call->strided_batched_gemms;
  g_call->stride_a = stride_a;
  g_call->stride_b = stride_b;
  g_call->stride_c = stride_c;
  g_call->batch_count = batch_count;
  g_call->algorithm = algorithm;
  return XLA_MUSA_MUBLAS_STATUS_SUCCESS;
}
const XlaMusaMuBlasApiV1* Getter() { return g_table; }
const XlaMusaMuBlasApiV2* GetterV2() { return g_table_v2; }

class FakeLoader final : public internal::MusaSymbolLoader {
 public:
  explicit FakeLoader(bool expose_v2 = true) : expose_v2_(expose_v2) {}
  absl::Status Load() override { return absl::OkStatus(); }
  absl::StatusOr<void*> Resolve(absl::string_view symbol) const override {
    if (symbol == "xla_musa_mublas_get_api_v2" && expose_v2_) {
      return reinterpret_cast<void*>(&GetterV2);
    }
    if (symbol == "xla_musa_mublas_get_api_v1") {
      return reinterpret_cast<void*>(&Getter);
    }
    return absl::NotFoundError(std::string(symbol));
  }
  absl::string_view loaded_path() const override { return "fake-mublas-shim"; }

 private:
  bool expose_v2_;
};

XlaMusaMuBlasApiV1 Table(uint64_t capabilities) {
  XlaMusaMuBlasApiV1 table = {};
  table.struct_size = sizeof(table);
  table.abi_version = XLA_MUSA_MUBLAS_ABI_VERSION_1;
  table.capabilities = capabilities;
  table.create = Create;
  table.destroy = Destroy;
  table.set_stream = SetStream;
  table.get_version = GetVersion;
  table.gemm = Gemm;
  return table;
}

XlaMusaMuBlasApiV2 TableV2() {
  XlaMusaMuBlasApiV2 table = {};
  table.struct_size = sizeof(table);
  table.abi_version = XLA_MUSA_MUBLAS_ABI_VERSION_2;
  table.capabilities = XLA_MUSA_MUBLAS_CAPABILITIES_V1;
  table.advanced_capabilities = XLA_MUSA_MUBLAS_ADVANCED_CAPABILITIES_V2;
  table.create = Create;
  table.destroy = Destroy;
  table.set_stream = SetStream;
  table.get_version = GetVersion;
  table.gemm = Gemm;
  table.set_atomics_mode = SetAtomicsMode;
  table.gemm_with_algorithm = GemmWithAlgorithm;
  table.gemm_batched = GemmBatched;
  table.gemm_strided_batched = GemmStridedBatched;
  return table;
}

class FixedTimer final : public EventBasedTimer {
 public:
  absl::StatusOr<absl::Duration> GetElapsedDuration() override {
    return absl::Microseconds(1250);
  }
};

class TrackingHostMemoryAllocation final : public MemoryAllocation {
 public:
  TrackingHostMemoryAllocation(uint64_t size, std::shared_ptr<bool> alive)
      : storage_(size), alive_(std::move(alive)) {
    *alive_ = true;
  }
  ~TrackingHostMemoryAllocation() override { *alive_ = false; }

  DeviceAddressBase address() const override {
    return DeviceAddressBase(const_cast<uint8_t*>(storage_.data()),
                             storage_.size());
  }

 private:
  std::vector<uint8_t> storage_;
  std::shared_ptr<bool> alive_;
};

void ConfigureHostMemoryAllocator(MockStreamExecutor* executor,
                                  std::shared_ptr<bool> alive) {
  ON_CALL(*executor, HostMemoryAllocate)
      .WillByDefault(
          Invoke([alive = std::move(alive)](uint64_t size)
                     -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
            return std::make_unique<TrackingHostMemoryAllocation>(size, alive);
          }));
}

class FakeScratchAllocator final : public ScratchAllocator {
 public:
  int64_t GetMemoryLimitInBytes() override { return memory_limit; }
  absl::StatusOr<DeviceAddress<uint8_t>> AllocateBytes(
      int64_t byte_size) override {
    sizes.push_back(byte_size);
    return DeviceAddress<uint8_t>::MakeFromByteSize(
        reinterpret_cast<void*>(0x8000 + sizes.size() * 0x100), byte_size);
  }

  int64_t memory_limit = -1;
  std::vector<int64_t> sizes;
};

class MusaBlasTest : public ::testing::Test {
 protected:
  void SetUp() override {
    call_ = {};
    table_ = Table(XLA_MUSA_MUBLAS_CAPABILITIES_V1);
    table_v2_ = TableV2();
    g_call = &call_;
    g_table = &table_;
    g_table_v2 = &table_v2_;
  }
  void TearDown() override {
    g_call = nullptr;
    g_table = nullptr;
    g_table_v2 = nullptr;
  }

  GemmCall call_;
  XlaMusaMuBlasApiV1 table_ = {};
  XlaMusaMuBlasApiV2 table_v2_ = {};
};

struct NativeGemmCase {
  blas::DataType data_type;
  blas::ComputationType computation_type;
  XlaMusaMuBlasDataType shim_data_type;
  XlaMusaMuBlasComputeType shim_compute_type;
};

class MusaBlasNativeGemmTest
    : public MusaBlasTest,
      public ::testing::WithParamInterface<NativeGemmCase> {};

TEST_P(MusaBlasNativeGemmTest, DispatchesNativeGemmTypes) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  ON_CALL(executor, Activate()).WillByDefault([] {
    return std::make_unique<ActivateContext>();
  });
  ON_CALL(stream, parent()).WillByDefault(Return(&executor));
  ON_CALL(stream, platform_specific_handle())
      .WillByDefault(Return(
          Stream::PlatformSpecificHandle{reinterpret_cast<void*>(0x2222)}));
  ON_CALL(stream, CreateEventBasedTimer)
      .WillByDefault(
          [](bool) -> absl::StatusOr<std::unique_ptr<EventBasedTimer>> {
            return std::make_unique<FixedTimer>();
          });

  auto api = MusaMuBlasApi::CreateForTesting(std::make_unique<FakeLoader>());
  MusaBlas blas(&executor, api.get());
  ASSERT_TRUE(blas.Init());

  float alpha_f32 = 1.0f;
  float beta_f32 = 0.0f;
  double alpha_f64 = 1.0;
  double beta_f64 = 0.0;
  const bool is_f64 = GetParam().data_type == blas::DataType::kDouble;
  const void* alpha = is_f64 ? static_cast<const void*>(&alpha_f64)
                             : static_cast<const void*>(&alpha_f32);
  const void* beta = is_f64 ? static_cast<const void*>(&beta_f64)
                            : static_cast<const void*>(&beta_f32);
  DeviceAddressBase a(reinterpret_cast<void*>(0x3000), 256);
  DeviceAddressBase b(reinterpret_cast<void*>(0x4000), 256);
  DeviceAddressBase c(reinterpret_cast<void*>(0x5000), 256);
  blas::ProfileResult profile;
  profile.set_is_valid(true);
  EXPECT_TRUE(blas.DoBlasGemmWithAlgorithm(
                      &stream, blas::Transpose::kNoTranspose,
                      blas::Transpose::kTranspose, 2, 3, 4, alpha, a,
                      GetParam().data_type, 4, b, GetParam().data_type, 4, beta,
                      &c, GetParam().data_type, 2, GetParam().computation_type,
                      blas::kDefaultAlgorithm, EngineOptions{}, &profile,
                      blas::CallContext::kNone)
                  .ok());

  EXPECT_TRUE(profile.is_valid());
  EXPECT_EQ(profile.algorithm(), blas::kDefaultAlgorithm);
  EXPECT_FLOAT_EQ(profile.elapsed_time_in_ms(), 1.25f);
  EXPECT_EQ(call_.gemms, 1);
  EXPECT_EQ(call_.algorithm_gemms, 1);
  EXPECT_EQ(call_.input_type, GetParam().shim_data_type);
  EXPECT_EQ(call_.output_type, GetParam().shim_data_type);
  EXPECT_EQ(call_.compute_type, GetParam().shim_compute_type);
  EXPECT_EQ(call_.trans_a, XLA_MUSA_MUBLAS_OPERATION_NONE);
  EXPECT_EQ(call_.trans_b, XLA_MUSA_MUBLAS_OPERATION_TRANSPOSE);
  EXPECT_EQ(call_.m, 2);
  EXPECT_EQ(call_.n, 3);
  EXPECT_EQ(call_.k, 4);
}

INSTANTIATE_TEST_SUITE_P(
    NativeTypes, MusaBlasNativeGemmTest,
    ::testing::Values(NativeGemmCase{blas::DataType::kFloat,
                                     blas::ComputationType::kF32,
                                     XLA_MUSA_MUBLAS_DATA_TYPE_F32,
                                     XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32},
                      NativeGemmCase{blas::DataType::kDouble,
                                     blas::ComputationType::kF64,
                                     XLA_MUSA_MUBLAS_DATA_TYPE_F64,
                                     XLA_MUSA_MUBLAS_COMPUTE_TYPE_F64}));

TEST_F(MusaBlasTest, ActivatesContextAndDispatchesF16Gemm) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  ON_CALL(executor, Activate()).WillByDefault([] {
    return std::make_unique<ActivateContext>();
  });
  ON_CALL(stream, parent()).WillByDefault(Return(&executor));
  ON_CALL(stream, platform_specific_handle())
      .WillByDefault(Return(
          Stream::PlatformSpecificHandle{reinterpret_cast<void*>(0x2222)}));

  auto api = MusaMuBlasApi::CreateForTesting(std::make_unique<FakeLoader>());
  {
    MusaBlas blas(&executor, api.get());
    ASSERT_TRUE(blas.Init());
    ASSERT_TRUE(blas.IsMainStreamSet().ok());
    EXPECT_TRUE(*blas.IsMainStreamSet());

    float alpha = 1.0f;
    float beta = 0.0f;
    DeviceAddressBase a(reinterpret_cast<void*>(0x3000), 256);
    DeviceAddressBase b(reinterpret_cast<void*>(0x4000), 256);
    DeviceAddressBase c(reinterpret_cast<void*>(0x5000), 256);
    EXPECT_TRUE(blas.DoBlasGemmWithAlgorithm(
                        &stream, blas::Transpose::kTranspose,
                        blas::Transpose::kNoTranspose, 2, 3, 4, &alpha, a,
                        blas::DataType::kHalf, 4, b, blas::DataType::kHalf, 4,
                        &beta, &c, blas::DataType::kHalf, 2,
                        blas::ComputationType::kF16, blas::kDefaultAlgorithm,
                        EngineOptions{}, nullptr, blas::CallContext::kNone)
                    .ok());
    EXPECT_FALSE(*blas.IsMainStreamSet());
    std::string version;
    EXPECT_TRUE(blas.GetVersion(&version).ok());
    EXPECT_EQ(version, "40100");
  }

  EXPECT_EQ(call_.creates, 2);
  EXPECT_EQ(call_.destroys, 2);
  EXPECT_EQ(call_.set_streams, 2);
  EXPECT_EQ(call_.stream, reinterpret_cast<void*>(0x2222));
  EXPECT_EQ(call_.gemms, 1);
  EXPECT_EQ(call_.input_type, XLA_MUSA_MUBLAS_DATA_TYPE_F16);
  EXPECT_EQ(call_.output_type, XLA_MUSA_MUBLAS_DATA_TYPE_F16);
  EXPECT_EQ(call_.compute_type, XLA_MUSA_MUBLAS_COMPUTE_TYPE_F16);
  EXPECT_EQ(call_.trans_a, XLA_MUSA_MUBLAS_OPERATION_TRANSPOSE);
  EXPECT_EQ(call_.trans_b, XLA_MUSA_MUBLAS_OPERATION_NONE);
  EXPECT_EQ(call_.m, 2);
  EXPECT_EQ(call_.n, 3);
  EXPECT_EQ(call_.k, 4);
}

TEST_F(MusaBlasTest, RejectsUnsupportedTypeAndCommandBufferClearly) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  ON_CALL(executor, Activate()).WillByDefault([] {
    return std::make_unique<ActivateContext>();
  });
  ON_CALL(stream, parent()).WillByDefault(Return(&executor));
  ON_CALL(stream, platform_specific_handle())
      .WillByDefault(Return(Stream::PlatformSpecificHandle{nullptr}));
  auto api = MusaMuBlasApi::CreateForTesting(std::make_unique<FakeLoader>());
  MusaBlas blas(&executor, api.get());
  ASSERT_TRUE(blas.Init());

  float scalar = 1.0f;
  DeviceAddressBase matrix(reinterpret_cast<void*>(0x3000), 64);
  EXPECT_TRUE(blas.DoBlasGemm(&stream, blas::Transpose::kNoTranspose,
                              blas::Transpose::kNoTranspose, 0, 1, 1,
                              blas::DataType::kFloat, &scalar, matrix, 1,
                              matrix, 1, &scalar, &matrix, 1, EngineOptions{},
                              blas::CallContext::kNone)
                  .ok());
  EXPECT_THAT(blas.DoBlasGemm(&stream, blas::Transpose::kNoTranspose,
                              blas::Transpose::kNoTranspose, 1, 1, 1,
                              blas::DataType::kComplexFloat, &scalar, matrix, 1,
                              matrix, 1, &scalar, &matrix, 1, EngineOptions{},
                              blas::CallContext::kNone),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("does not support data type")));
  EXPECT_THAT(blas.DoBlasGemm(&stream, blas::Transpose::kNoTranspose,
                              blas::Transpose::kNoTranspose, 1, 1, 1,
                              blas::DataType::kHalf, &scalar, matrix, 1, matrix,
                              1, &scalar, &matrix, 1, EngineOptions{},
                              blas::CallContext::kNone),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("f16 GEMM requires f32 computation")));

  EngineOptions command_buffer;
  command_buffer.require_command_buffer = true;
  EXPECT_THAT(blas.DoBlasGemm(&stream, blas::Transpose::kNoTranspose,
                              blas::Transpose::kNoTranspose, 1, 1, 1,
                              blas::DataType::kFloat, &scalar, matrix, 1,
                              matrix, 1, &scalar, &matrix, 1, command_buffer,
                              blas::CallContext::kNone),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("command-buffer GEMM")));
  EngineOptions deterministic;
  deterministic.require_determinism = true;
  EXPECT_TRUE(blas.DoBlasGemm(&stream, blas::Transpose::kNoTranspose,
                              blas::Transpose::kNoTranspose, 1, 1, 1,
                              blas::DataType::kFloat, &scalar, matrix, 1,
                              matrix, 1, &scalar, &matrix, 1, deterministic,
                              blas::CallContext::kNone)
                  .ok());
  EXPECT_FALSE(call_.allow_atomics);
  EXPECT_THAT(
      blas.DoBlasGemmWithAlgorithm(
          &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose,
          1, 1, 1, &scalar, matrix, blas::DataType::kHalf, 1, matrix,
          blas::DataType::kHalf, 1, &scalar, &matrix, blas::DataType::kFloat, 1,
          blas::ComputationType::kF32, blas::kDefaultAlgorithm, EngineOptions{},
          nullptr, blas::CallContext::kNone),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("does not support f16 inputs, f32 output")));
  EXPECT_THAT(
      blas.DoBlasGemmWithAlgorithm(
          &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose,
          1, 1, 1, &scalar, matrix, blas::DataType::kBF16, 1, matrix,
          blas::DataType::kBF16, 1, &scalar, &matrix, blas::DataType::kBF16, 1,
          blas::ComputationType::kF32, blas::kDefaultAlgorithm, EngineOptions{},
          nullptr, blas::CallContext::kNone),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("does not support bf16 inputs")));
  EXPECT_THAT(
      blas.DoBlasGemmWithAlgorithm(
          &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose,
          1, 1, 1, &scalar, matrix, blas::DataType::kHalf, 1, matrix,
          blas::DataType::kFloat, 1, &scalar, &matrix, blas::DataType::kFloat,
          1, blas::ComputationType::kF32, blas::kDefaultAlgorithm,
          EngineOptions{}, nullptr, blas::CallContext::kNone),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("requires identical input types")));
  EXPECT_EQ(call_.gemms, 1);
}

TEST_F(MusaBlasTest, EnumeratesStableAlgorithmsAndRestrictsTensorOpToF32) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  ON_CALL(executor, Activate()).WillByDefault([] {
    return std::make_unique<ActivateContext>();
  });
  ON_CALL(stream, parent()).WillByDefault(Return(&executor));
  ON_CALL(stream, platform_specific_handle())
      .WillByDefault(Return(
          Stream::PlatformSpecificHandle{reinterpret_cast<void*>(0x2222)}));
  ON_CALL(stream, CreateEventBasedTimer)
      .WillByDefault(
          [](bool) -> absl::StatusOr<std::unique_ptr<EventBasedTimer>> {
            return std::make_unique<FixedTimer>();
          });
  auto api = MusaMuBlasApi::CreateForTesting(std::make_unique<FakeLoader>());
  MusaBlas blas(&executor, api.get());
  ASSERT_TRUE(blas.Init());
  EXPECT_EQ(blas.GetBlasLt(), nullptr);

  gpu::MatrixDescriptor a_desc;
  gpu::MatrixDescriptor b_desc;
  gpu::MatrixDescriptor c_base;
  a_desc.type = b_desc.type = c_base.type = blas::DataType::kFloat;
  gpu::OutputMatrixDescriptor c_desc(std::move(c_base));
  c_desc.compute_type = blas::ComputationType::kF32;
  float alpha = 1.0f;
  float beta = 0.0f;
  std::vector<blas::AlgorithmType> algorithms;
  ASSERT_TRUE(blas.GetBlasGemmAlgorithms(&stream, a_desc, b_desc, &c_desc,
                                         &alpha, &beta, &algorithms));
  EXPECT_THAT(algorithms, ElementsAre(kMusaBlasDefaultAlgorithm,
                                      kMusaBlasTensorOpAlgorithm));

  DeviceAddressBase matrix(reinterpret_cast<void*>(0x3000), 64);
  blas::ProfileResult profile;
  ASSERT_TRUE(blas.DoBlasGemmWithAlgorithm(
                      &stream, blas::Transpose::kNoTranspose,
                      blas::Transpose::kNoTranspose, 2, 2, 2, &alpha, matrix,
                      blas::DataType::kFloat, 2, matrix, blas::DataType::kFloat,
                      2, &beta, &matrix, blas::DataType::kFloat, 2,
                      blas::ComputationType::kF32, kMusaBlasTensorOpAlgorithm,
                      EngineOptions{}, &profile, blas::CallContext::kNone)
                  .ok());
  EXPECT_TRUE(profile.is_valid());
  EXPECT_FLOAT_EQ(profile.elapsed_time_in_ms(), 1.25f);
  EXPECT_EQ(call_.algorithm, XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP);
  EXPECT_TRUE(call_.allow_atomics);

  EngineOptions deterministic;
  deterministic.require_determinism = true;
  EXPECT_THAT(
      blas.DoBlasGemmWithAlgorithm(
          &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose,
          2, 2, 2, &alpha, matrix, blas::DataType::kFloat, 2, matrix,
          blas::DataType::kFloat, 2, &beta, &matrix, blas::DataType::kFloat, 2,
          blas::ComputationType::kF32, kMusaBlasTensorOpAlgorithm,
          deterministic, nullptr, blas::CallContext::kNone),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("supports only algorithm 0")));
}

TEST_F(MusaBlasTest, DispatchesAndValidatesStridedBatchedGemm) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  ON_CALL(executor, Activate()).WillByDefault([] {
    return std::make_unique<ActivateContext>();
  });
  ON_CALL(stream, parent()).WillByDefault(Return(&executor));
  ON_CALL(stream, platform_specific_handle())
      .WillByDefault(Return(
          Stream::PlatformSpecificHandle{reinterpret_cast<void*>(0x2222)}));
  ON_CALL(stream, CreateEventBasedTimer)
      .WillByDefault(
          [](bool) -> absl::StatusOr<std::unique_ptr<EventBasedTimer>> {
            return std::make_unique<FixedTimer>();
          });
  auto api = MusaMuBlasApi::CreateForTesting(std::make_unique<FakeLoader>());
  MusaBlas blas(&executor, api.get());
  ASSERT_TRUE(blas.Init());

  float alpha = 1.0f;
  float beta = 0.0f;
  DeviceAddressBase a(reinterpret_cast<void*>(0x3000), 32);
  DeviceAddressBase b(reinterpret_cast<void*>(0x4000), 32);
  DeviceAddressBase c(reinterpret_cast<void*>(0x5000), 32);
  blas::ProfileResult profile;
  ASSERT_TRUE(blas.DoBlasGemmStridedBatchedWithAlgorithm(
                      &stream, blas::Transpose::kNoTranspose,
                      blas::Transpose::kNoTranspose, 2, 2, 2, &alpha, a,
                      blas::DataType::kFloat, 2, 4, b, blas::DataType::kFloat,
                      2, 4, &beta, &c, blas::DataType::kFloat, 2, 4, 2,
                      blas::ComputationType::kF32, kMusaBlasTensorOpAlgorithm,
                      EngineOptions{}, &profile, blas::CallContext::kNone)
                  .ok());
  EXPECT_EQ(call_.strided_batched_gemms, 1);
  EXPECT_EQ(call_.batch_count, 2);
  EXPECT_EQ(call_.stride_a, 4);
  EXPECT_EQ(call_.algorithm, XLA_MUSA_MUBLAS_ALGORITHM_TENSOR_OP);
  EXPECT_TRUE(profile.is_valid());

  EXPECT_THAT(
      blas.DoBlasGemmStridedBatched(
          &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose,
          2, 2, 2, blas::DataType::kFloat, &alpha, a, 2, -1, b, 2, 4, &beta, &c,
          2, 4, 2, EngineOptions{}, blas::CallContext::kNone),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("strides must be non-negative")));
  EXPECT_THAT(
      blas.DoBlasGemmStridedBatched(
          &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose,
          2, 2, 2, blas::DataType::kFloat, &alpha, a, 2, 4, b, 2, 4, &beta, &c,
          2, 0, 2, EngineOptions{}, blas::CallContext::kNone),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("output batch stride must be positive")));
  EXPECT_THAT(
      blas.DoBlasGemmStridedBatched(
          &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose,
          2, 2, 2, blas::DataType::kFloat, &alpha, a, 2, 4, b, 2, 4, &beta, &c,
          2, 3, 2, EngineOptions{}, blas::CallContext::kNone),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("overlaps a matrix extent of 4 elements")));
}

TEST_F(MusaBlasTest, StagesAndValidatesPointerArrayBatches) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  ConfigureHostMemoryAllocator(&executor, std::make_shared<bool>(false));
  ON_CALL(executor, Activate()).WillByDefault([] {
    return std::make_unique<ActivateContext>();
  });
  ON_CALL(stream, parent()).WillByDefault(Return(&executor));
  ON_CALL(stream, platform_specific_handle())
      .WillByDefault(Return(
          Stream::PlatformSpecificHandle{reinterpret_cast<void*>(0x2222)}));
  auto api = MusaMuBlasApi::CreateForTesting(std::make_unique<FakeLoader>());
  MusaBlas blas(&executor, api.get());
  ASSERT_TRUE(blas.Init());

  DeviceAddress<float> a0 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x3000), 16);
  DeviceAddress<float> a1 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x3100), 16);
  DeviceAddress<float> b0 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x4000), 16);
  DeviceAddress<float> b1 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x4100), 16);
  DeviceAddress<float> c0 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x5000), 16);
  DeviceAddress<float> c1 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x5100), 16);
  std::array<DeviceAddress<float>*, 2> a = {&a0, &a1};
  std::array<DeviceAddress<float>*, 2> b = {&b0, &b1};
  std::array<DeviceAddress<float>*, 2> c = {&c0, &c1};
  FakeScratchAllocator scratch;
  ASSERT_TRUE(blas.DoBlasGemmBatched(
      &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose, 2,
      2, 2, 1.0f, a, 2, b, 2, 0.0f, c, 2, 2, EngineOptions{}, &scratch,
      blas::CallContext::kNone));
  EXPECT_EQ(call_.batched_gemms, 1);
  EXPECT_EQ(call_.batch_count, 2);
  EXPECT_THAT(scratch.sizes, ElementsAre(16, 16, 16));

  absl::Span<DeviceAddress<float>* const> short_a(a.data(), 1);
  EXPECT_FALSE(blas.DoBlasGemmBatched(
      &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose, 2,
      2, 2, 1.0f, short_a, 2, b, 2, 0.0f, c, 2, 2, EngineOptions{}, &scratch,
      blas::CallContext::kNone));
  EXPECT_EQ(call_.batched_gemms, 1);
}

TEST_F(MusaBlasTest, PointerBatchCallbackFailureDrainsBeforeHostRelease) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> stream;
  auto host_alive = std::make_shared<bool>(false);
  ConfigureHostMemoryAllocator(&executor, host_alive);
  ON_CALL(executor, Activate()).WillByDefault([] {
    return std::make_unique<ActivateContext>();
  });
  ON_CALL(stream, parent()).WillByDefault(Return(&executor));
  ON_CALL(stream, platform_specific_handle())
      .WillByDefault(Return(
          Stream::PlatformSpecificHandle{reinterpret_cast<void*>(0x2222)}));

  std::vector<const void*> copy_sources;
  EXPECT_CALL(stream, Memcpy(A<DeviceAddressBase*>(), A<const void*>(),
                             2 * sizeof(void*)))
      .Times(3)
      .WillRepeatedly(
          Invoke([&copy_sources](DeviceAddressBase*, const void* host_source,
                                 uint64_t) {
            copy_sources.push_back(host_source);
            return absl::OkStatus();
          }));
  EXPECT_CALL(stream, DoHostCallbackWithStatus)
      .WillOnce(Return(absl::InternalError("callback enqueue failed")));

  DeviceAddress<float> a0 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x3000), 16);
  DeviceAddress<float> a1 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x3100), 16);
  DeviceAddress<float> b0 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x4000), 16);
  DeviceAddress<float> b1 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x4100), 16);
  DeviceAddress<float> c0 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x5000), 16);
  DeviceAddress<float> c1 = DeviceAddress<float>::MakeFromByteSize(
      reinterpret_cast<void*>(0x5100), 16);
  std::array<DeviceAddress<float>*, 2> a = {&a0, &a1};
  std::array<DeviceAddress<float>*, 2> b = {&b0, &b1};
  std::array<DeviceAddress<float>*, 2> c = {&c0, &c1};

  EXPECT_CALL(stream, BlockHostUntilDone).WillOnce(Invoke([&] {
    EXPECT_TRUE(*host_alive);
    EXPECT_EQ(copy_sources.size(), 3);
    if (copy_sources.size() != 3) {
      return absl::InternalError("missing staged pointer-array copies");
    }
    const auto* staged_a = static_cast<void* const*>(copy_sources[0]);
    const auto* staged_b = static_cast<void* const*>(copy_sources[1]);
    const auto* staged_c = static_cast<void* const*>(copy_sources[2]);
    EXPECT_THAT(absl::MakeConstSpan(staged_a, 2),
                ElementsAre(a0.opaque(), a1.opaque()));
    EXPECT_THAT(absl::MakeConstSpan(staged_b, 2),
                ElementsAre(b0.opaque(), b1.opaque()));
    EXPECT_THAT(absl::MakeConstSpan(staged_c, 2),
                ElementsAre(c0.opaque(), c1.opaque()));
    return absl::OkStatus();
  }));

  auto api = MusaMuBlasApi::CreateForTesting(std::make_unique<FakeLoader>());
  MusaBlas blas(&executor, api.get());
  ASSERT_TRUE(blas.Init());
  FakeScratchAllocator scratch;
  EXPECT_FALSE(blas.DoBlasGemmBatched(
      &stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose, 2,
      2, 2, 1.0f, a, 2, b, 2, 0.0f, c, 2, 2, EngineOptions{}, &scratch,
      blas::CallContext::kNone));
  EXPECT_FALSE(*host_alive);
  EXPECT_EQ(call_.batched_gemms, 0);
}

TEST_F(MusaBlasTest, ReleasesHandleBeforeStreamDestructionAndRawHandleReuse) {
  NiceMock<MockStreamExecutor> executor;
  NiceMock<MockStream> first_stream;
  NiceMock<MockStream> reused_stream;
  ON_CALL(executor, Activate()).WillByDefault([] {
    return std::make_unique<ActivateContext>();
  });
  for (MockStream* stream : {&first_stream, &reused_stream}) {
    ON_CALL(*stream, parent()).WillByDefault(Return(&executor));
    ON_CALL(*stream, platform_specific_handle())
        .WillByDefault(Return(
            Stream::PlatformSpecificHandle{reinterpret_cast<void*>(0x2222)}));
  }

  auto api = MusaMuBlasApi::CreateForTesting(std::make_unique<FakeLoader>());
  {
    MusaBlas blas(&executor, api.get());
    ASSERT_TRUE(blas.Init());
    float alpha = 1.0f;
    float beta = 0.0f;
    DeviceAddressBase a(reinterpret_cast<void*>(0x3000), 256);
    DeviceAddressBase b(reinterpret_cast<void*>(0x4000), 256);
    DeviceAddressBase c(reinterpret_cast<void*>(0x5000), 256);
    auto run_gemm = [&](Stream* stream) {
      return blas.DoBlasGemmWithAlgorithm(
          stream, blas::Transpose::kNoTranspose, blas::Transpose::kNoTranspose,
          2, 2, 2, &alpha, a, blas::DataType::kFloat, 2, b,
          blas::DataType::kFloat, 2, &beta, &c, blas::DataType::kFloat, 2,
          blas::ComputationType::kF32, blas::kDefaultAlgorithm, EngineOptions{},
          nullptr, blas::CallContext::kNone);
    };

    EXPECT_TRUE(run_gemm(&first_stream).ok());
    EXPECT_EQ(call_.creates, 2);
    EXPECT_EQ(call_.destroys, 0);
    blas.NotifyStreamDestroyed(&first_stream);
    EXPECT_EQ(call_.destroys, 1);

    EXPECT_TRUE(run_gemm(&reused_stream).ok());
    EXPECT_EQ(call_.creates, 3);
  }
  EXPECT_EQ(call_.destroys, 3);
}

}  // namespace
}  // namespace stream_executor::musa
