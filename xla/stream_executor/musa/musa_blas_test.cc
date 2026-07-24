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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/musa/mublas_shim/mublas_shim_abi.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_mublas_api.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;

struct GemmCall {
  int creates = 0;
  int destroys = 0;
  int set_streams = 0;
  int gemms = 0;
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

XlaMusaMuBlasStatus Create(void** handle) {
  ++g_call->creates;
  *handle = reinterpret_cast<void*>(0x1111);
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
const XlaMusaMuBlasApiV1* Getter() { return g_table; }

class FakeLoader final : public internal::MusaSymbolLoader {
 public:
  absl::Status Load() override { return absl::OkStatus(); }
  absl::StatusOr<void*> Resolve(absl::string_view symbol) const override {
    if (symbol != "xla_musa_mublas_get_api_v1") {
      return absl::NotFoundError(std::string(symbol));
    }
    return reinterpret_cast<void*>(&Getter);
  }
  absl::string_view loaded_path() const override { return "fake-mublas-shim"; }
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

class MusaBlasTest : public ::testing::Test {
 protected:
  void SetUp() override {
    call_ = {};
    table_ = Table(XLA_MUSA_MUBLAS_CAPABILITIES_V1);
    g_call = &call_;
    g_table = &table_;
  }
  void TearDown() override {
    g_call = nullptr;
    g_table = nullptr;
  }

  GemmCall call_;
  XlaMusaMuBlasApiV1 table_ = {};
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

  EXPECT_FALSE(profile.is_valid());
  EXPECT_EQ(profile.algorithm(), blas::kDefaultAlgorithm);
  EXPECT_EQ(call_.gemms, 1);
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

  EXPECT_EQ(call_.creates, 1);
  EXPECT_EQ(call_.destroys, 1);
  EXPECT_EQ(call_.set_streams, 1);
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
  EXPECT_THAT(blas.DoBlasGemm(&stream, blas::Transpose::kNoTranspose,
                              blas::Transpose::kNoTranspose, 1, 1, 1,
                              blas::DataType::kFloat, &scalar, matrix, 1,
                              matrix, 1, &scalar, &matrix, 1, deterministic,
                              blas::CallContext::kNone),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("deterministic GEMM")));
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
  EXPECT_EQ(call_.gemms, 0);
}

}  // namespace
}  // namespace stream_executor::musa
