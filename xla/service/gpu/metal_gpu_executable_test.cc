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

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/array2d.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

constexpr int64_t kM = 16;
constexpr int64_t kN = 32;
constexpr int64_t kK = 8;

Literal MakeLhs() {
  Array2D<float> values(kM, kK);
  for (int64_t row = 0; row < kM; ++row) {
    for (int64_t col = 0; col < kK; ++col) {
      const int64_t i = row * kK + col;
      values(row, col) = static_cast<float>((i % 17) - 8) * 0.125f;
    }
  }
  return LiteralUtil::CreateR2FromArray2D(values);
}

Literal MakeRhs() {
  Array2D<float> values(kK, kN);
  for (int64_t row = 0; row < kK; ++row) {
    for (int64_t col = 0; col < kN; ++col) {
      const int64_t i = row * kN + col;
      values(row, col) = static_cast<float>((i % 23) - 11) * 0.0625f;
    }
  }
  return LiteralUtil::CreateR2FromArray2D(values);
}

float Reference(const Literal& lhs, const Literal& rhs, int64_t row,
                int64_t col, bool relu) {
  float acc = 0.0f;
  for (int64_t kk = 0; kk < kK; ++kk) {
    acc += lhs.Get<float>({row, kk}) * rhs.Get<float>({kk, col});
  }
  return relu ? std::max(acc, 0.0f) : acc;
}

absl::StatusOr<Literal> ExecuteMetalMatmul(bool relu) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("metal"));
  if (platform->VisibleDeviceCount() == 0) {
    return absl::FailedPreconditionError("No visible Metal devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));

  XlaBuilder builder(relu ? "metal_dot_relu" : "metal_dot");
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {kM, kK});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {kK, kN});
  XlaOp lhs = Parameter(&builder, 0, lhs_shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, rhs_shape, "rhs");
  XlaOp dot = Dot(lhs, rhs);
  if (relu) {
    Max(dot, Broadcast(ConstantR0<float>(&builder, 0.0f), {kM, kN}));
  }
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal lhs_literal = MakeLhs();
  Literal rhs_literal = MakeRhs();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> lhs_data,
                      client->TransferToServer(lhs_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> rhs_data,
                      client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

void ExpectMatchesReference(const Literal& actual, const Literal& lhs,
                            const Literal& rhs, bool relu) {
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F32, {kM, kN})));
  for (int64_t row = 0; row < kM; ++row) {
    for (int64_t col = 0; col < kN; ++col) {
      EXPECT_NEAR(actual.Get<float>({row, col}),
                  Reference(lhs, rhs, row, col, relu), 2.0e-3)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, Dot) {
  auto result = ExecuteMetalMatmul(/*relu=*/false);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesReference(actual, MakeLhs(), MakeRhs(), /*relu=*/false);
}

TEST(MetalGpuExecutableTest, DotRelu) {
  auto result = ExecuteMetalMatmul(/*relu=*/true);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesReference(actual, MakeLhs(), MakeRhs(), /*relu=*/true);
}

}  // namespace
}  // namespace xla
