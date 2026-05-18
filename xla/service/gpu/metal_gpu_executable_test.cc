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
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/array2d.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/hlo/builder/lib/math.h"
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
constexpr int64_t kElementCount = 257;
constexpr int64_t kGatherInputSize = 16;
constexpr int64_t kGatherOutputSize = 4;

absl::StatusOr<LocalClient*> GetMetalClient() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("metal"));
  if (platform->VisibleDeviceCount() == 0) {
    return absl::FailedPreconditionError("No visible Metal devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

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

Literal MakeElementwiseLhs() {
  std::vector<float> values(kElementCount);
  for (int64_t i = 0; i < kElementCount; ++i) {
    values[i] = static_cast<float>((i % 19) - 9) * 0.25f;
  }
  return LiteralUtil::CreateR1<float>(values);
}

Literal MakeElementwiseRhs() {
  std::vector<float> values(kElementCount);
  for (int64_t i = 0; i < kElementCount; ++i) {
    values[i] = static_cast<float>((i % 13) - 6) * 0.5f;
  }
  return LiteralUtil::CreateR1<float>(values);
}

Literal MakeElementwiseS32() {
  std::vector<int32_t> values(kElementCount);
  for (int64_t i = 0; i < kElementCount; ++i) {
    values[i] = static_cast<int32_t>((i % 31) - 15);
  }
  return LiteralUtil::CreateR1<int32_t>(values);
}

Literal MakeElementwisePred() {
  Literal literal =
      Literal::CreateFromShape(ShapeUtil::MakeShape(PRED, {kElementCount}));
  for (int64_t i = 0; i < kElementCount; ++i) {
    literal.Set<bool>({i}, (i % 3) == 0);
  }
  return literal;
}

Literal MakeGatherInput() {
  std::vector<float> values(kGatherInputSize);
  for (int64_t i = 0; i < kGatherInputSize; ++i) {
    values[i] = static_cast<float>(i) * 0.25f - 2.0f;
  }
  return LiteralUtil::CreateR1<float>(values);
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
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

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

template <typename BuildFn>
absl::StatusOr<Literal> ExecuteMetalElementwiseBinary(const char* name,
                                                      BuildFn build) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder(name);
  Shape shape = ShapeUtil::MakeShape(F32, {kElementCount});
  XlaOp lhs = Parameter(&builder, 0, shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, shape, "rhs");
  build(&builder, lhs, rhs);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal lhs_literal = MakeElementwiseLhs();
  Literal rhs_literal = MakeElementwiseRhs();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> lhs_data,
                      client->TransferToServer(lhs_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> rhs_data,
                      client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

template <typename BuildFn>
absl::StatusOr<Literal> ExecuteMetalElementwiseUnary(const char* name,
                                                     BuildFn build) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder(name);
  Shape shape = ShapeUtil::MakeShape(F32, {kElementCount});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  build(&builder, input);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwiseLhs();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

template <typename BuildFn>
absl::StatusOr<Literal> ExecuteMetalPredUnary(const char* name, BuildFn build) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder(name);
  Shape shape = ShapeUtil::MakeShape(PRED, {kElementCount});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  build(&builder, input);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwisePred();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalElementwiseScalarMaximum() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_elementwise_scalar_maximum");
  Shape shape = ShapeUtil::MakeShape(F32, {kElementCount});
  XlaOp lhs = Parameter(&builder, 0, shape, "lhs");
  Max(lhs, Broadcast(ConstantR0<float>(&builder, 1.0f), {kElementCount}));
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal lhs_literal = MakeElementwiseLhs();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> lhs_data,
                      client->TransferToServer(lhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalElementwiseWhereCall() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape shape = ShapeUtil::MakeShape(F32, {kElementCount});
  Shape pred_shape = ShapeUtil::MakeShape(PRED, {kElementCount});

  XlaBuilder where_builder("_where");
  XlaOp pred = Parameter(&where_builder, 0, pred_shape, "pred");
  XlaOp on_true = Parameter(&where_builder, 1, shape, "on_true");
  XlaOp on_false = Parameter(&where_builder, 2, shape, "on_false");
  Select(pred, on_true, on_false);
  TF_ASSIGN_OR_RETURN(XlaComputation where, where_builder.Build());

  XlaBuilder builder("metal_elementwise_where_call");
  XlaOp input = Parameter(&builder, 0, shape, "input");
  XlaOp zero = Broadcast(ConstantR0<float>(&builder, 0.0f), {kElementCount});
  Call(&builder, where, {Gt(input, zero), input, Neg(input)});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwiseLhs();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalPadCall() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape input_shape = ShapeUtil::MakeShape(F32, {kElementCount});
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});

  PaddingConfig padding;
  PaddingConfig::PaddingConfigDimension* dimension =
      padding.add_dimensions();
  dimension->set_edge_padding_low(2);
  dimension->set_edge_padding_high(3);
  dimension->set_interior_padding(0);

  XlaBuilder pad_builder("_pad");
  XlaOp callee_input = Parameter(&pad_builder, 0, input_shape, "input");
  XlaOp callee_padding =
      Parameter(&pad_builder, 1, scalar_shape, "padding");
  Pad(callee_input, callee_padding, padding);
  TF_ASSIGN_OR_RETURN(XlaComputation pad, pad_builder.Build());

  XlaBuilder builder("metal_elementwise_pad_call");
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp padding_value = ConstantR0<float>(&builder, 1.5f);
  Call(&builder, pad, {input, padding_value});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwiseLhs();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalGatherSelect() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_elementwise_gather_select");
  Shape input_shape = ShapeUtil::MakeShape(F32, {kGatherInputSize});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");

  Array2D<int32_t> indices(kGatherOutputSize, 1);
  indices(0, 0) = 0;
  indices(1, 0) = 2;
  indices(2, 0) = 4;
  indices(3, 0) = 8;
  XlaOp start_indices = ConstantR2FromArray2D<int32_t>(&builder, indices);

  GatherDimensionNumbers dnums;
  dnums.add_collapsed_slice_dims(0);
  dnums.add_start_index_map(0);
  dnums.set_index_vector_dim(1);
  XlaOp gather = Gather(input, start_indices, dnums, {1},
                        /*indices_are_sorted=*/true);
  XlaOp pred =
      Broadcast(ConstantR0<bool>(&builder, true), {kGatherOutputSize});
  XlaOp fallback = Broadcast(
      ConstantR0<float>(&builder, std::numeric_limits<float>::quiet_NaN()),
      {kGatherOutputSize});
  Select(pred, gather, fallback);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeGatherInput();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalIota() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_iota");
  Add(Iota(&builder, F32, 16), Broadcast(ConstantR0<float>(&builder, 1.0f),
                                        {16}));
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());
  std::vector<GlobalData*> arguments;
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalS32Iota() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_s32_iota");
  Add(Iota(&builder, S32, 16), Broadcast(ConstantR0<int32_t>(&builder, 1),
                                        {16}));
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());
  std::vector<GlobalData*> arguments;
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalS32Add() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_s32_add");
  Shape shape = ShapeUtil::MakeShape(S32, {kElementCount});
  XlaOp lhs = Parameter(&builder, 0, shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, shape, "rhs");
  Add(lhs, rhs);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal lhs_literal = MakeElementwiseS32();
  Literal rhs_literal = MakeElementwiseS32();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> lhs_data,
                      client->TransferToServer(lhs_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> rhs_data,
                      client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

template <typename BuildFn>
absl::StatusOr<Literal> ExecuteMetalS32Unary(const char* name, BuildFn build) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder(name);
  Shape shape = ShapeUtil::MakeShape(S32, {kElementCount});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  build(&builder, input);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwiseS32();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalConvert(PrimitiveType input_type,
                                            PrimitiveType output_type) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_convert");
  Shape input_shape = ShapeUtil::MakeShape(input_type, {kElementCount});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  ConvertElementType(input, output_type);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal =
      input_type == F32 ? MakeElementwiseLhs() : MakeElementwiseS32();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalReduction(HloOpcode opcode,
                                              float init_value) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder reducer_builder("metal_reducer");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  switch (opcode) {
    case HloOpcode::kAdd:
      Add(lhs, rhs);
      break;
    case HloOpcode::kMultiply:
      Mul(lhs, rhs);
      break;
    case HloOpcode::kMaximum:
      Max(lhs, rhs);
      break;
    case HloOpcode::kMinimum:
      Min(lhs, rhs);
      break;
    default:
      return absl::InvalidArgumentError("Unexpected reduction opcode.");
  }
  TF_ASSIGN_OR_RETURN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder("metal_reduction");
  Shape input_shape = ShapeUtil::MakeShape(F32, {kElementCount});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  Reduce(input, ConstantR0<float>(&builder, init_value), reducer, {0});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwiseLhs();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
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

template <typename ReferenceFn>
void ExpectMatchesElementwiseReference(const Literal& actual,
                                       ReferenceFn reference,
                                       float tolerance = 1.0e-6f) {
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F32, {kElementCount})));
  for (int64_t i = 0; i < kElementCount; ++i) {
    EXPECT_NEAR(actual.Get<float>({i}), reference(i), tolerance) << "at " << i;
  }
}

template <typename ReferenceFn>
void ExpectMatchesPredReference(const Literal& actual, ReferenceFn reference) {
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(PRED, {kElementCount})));
  for (int64_t i = 0; i < kElementCount; ++i) {
    EXPECT_EQ(actual.Get<bool>({i}), reference(i)) << "at " << i;
  }
}

template <typename ReferenceFn>
void ExpectMatchesS32Reference(const Literal& actual, ReferenceFn reference) {
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(S32, {kElementCount})));
  for (int64_t i = 0; i < kElementCount; ++i) {
    EXPECT_EQ(actual.Get<int32_t>({i}), reference(i)) << "at " << i;
  }
}

void ExpectMatchesSliceReference(const Literal& actual, const Literal& input,
                                 int64_t start, int64_t size) {
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(F32, {size})));
  for (int64_t i = 0; i < size; ++i) {
    EXPECT_EQ(actual.Get<float>({i}), input.Get<float>({start + i}))
        << "at " << i;
  }
}

void ExpectMatchesPadReference(const Literal& actual, const Literal& input,
                               int64_t low_padding, int64_t high_padding,
                               float pad_value) {
  const int64_t input_elements = ShapeUtil::ElementsIn(input.shape());
  const int64_t output_elements =
      input_elements + low_padding + high_padding;
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F32, {output_elements})));
  for (int64_t i = 0; i < output_elements; ++i) {
    const bool in_input =
        i >= low_padding && i < low_padding + input_elements;
    const float expected =
        in_input ? input.Get<float>({i - low_padding}) : pad_value;
    EXPECT_EQ(actual.Get<float>({i}), expected) << "at " << i;
  }
}

void ExpectMatchesGatherReference(const Literal& actual, const Literal& input) {
  constexpr int64_t kIndices[kGatherOutputSize] = {0, 2, 4, 8};
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F32, {kGatherOutputSize})));
  for (int64_t i = 0; i < kGatherOutputSize; ++i) {
    EXPECT_EQ(actual.Get<float>({i}), input.Get<float>({kIndices[i]}))
        << "at " << i;
  }
}

void ExpectMatchesReshapeSliceReference(const Literal& actual,
                                        const Literal& input) {
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F32, {16, 16})));
  for (int64_t row = 0; row < 16; ++row) {
    for (int64_t col = 0; col < 16; ++col) {
      const int64_t index = row * 16 + col;
      EXPECT_EQ(actual.Get<float>({row, col}), input.Get<float>({index}))
          << "at (" << row << ", " << col << ")";
    }
  }
}

void ExpectMatchesTransposeSliceReference(const Literal& actual,
                                          const Literal& input) {
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F32, {16, 16})));
  for (int64_t row = 0; row < 16; ++row) {
    for (int64_t col = 0; col < 16; ++col) {
      const int64_t index = col * 16 + row;
      EXPECT_EQ(actual.Get<float>({row, col}), input.Get<float>({index}))
          << "at (" << row << ", " << col << ")";
    }
  }
}

void ExpectScalarNear(const Literal& actual, float expected,
                      float tolerance = 1.0e-6f) {
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {})));
  EXPECT_NEAR(actual.Get<float>({}), expected, tolerance);
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

TEST(MetalGpuExecutableTest, ElementwiseAdd) {
  auto result = ExecuteMetalElementwiseBinary(
      "metal_elementwise_add",
      [](XlaBuilder*, XlaOp lhs, XlaOp rhs) { Add(lhs, rhs); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal lhs = MakeElementwiseLhs();
  Literal rhs = MakeElementwiseRhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return lhs.Get<float>({i}) + rhs.Get<float>({i}); });
}

TEST(MetalGpuExecutableTest, ElementwiseMultiply) {
  auto result = ExecuteMetalElementwiseBinary(
      "metal_elementwise_multiply",
      [](XlaBuilder*, XlaOp lhs, XlaOp rhs) { Mul(lhs, rhs); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal lhs = MakeElementwiseLhs();
  Literal rhs = MakeElementwiseRhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return lhs.Get<float>({i}) * rhs.Get<float>({i}); });
}

TEST(MetalGpuExecutableTest, ElementwisePower) {
  auto result = ExecuteMetalElementwiseBinary(
      "metal_elementwise_power", [](XlaBuilder* builder, XlaOp lhs, XlaOp rhs) {
        XlaOp base =
            Add(lhs, Broadcast(ConstantR0<float>(builder, 3.0f),
                               {kElementCount}));
        XlaOp exponent =
            Add(Mul(rhs, Broadcast(ConstantR0<float>(builder, 0.25f),
                                   {kElementCount})),
                Broadcast(ConstantR0<float>(builder, 1.25f),
                          {kElementCount}));
        Pow(base, exponent);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal lhs = MakeElementwiseLhs();
  Literal rhs = MakeElementwiseRhs();
  ExpectMatchesElementwiseReference(
      actual,
      [&](int64_t i) {
        const float base = lhs.Get<float>({i}) + 3.0f;
        const float exponent = rhs.Get<float>({i}) * 0.25f + 1.25f;
        return std::pow(base, exponent);
      },
      2.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseRemainder) {
  auto result = ExecuteMetalElementwiseBinary(
      "metal_elementwise_remainder",
      [](XlaBuilder* builder, XlaOp lhs, XlaOp rhs) {
        XlaOp divisor =
            Add(Abs(rhs), Broadcast(ConstantR0<float>(builder, 1.0f),
                                    {kElementCount}));
        Rem(lhs, divisor);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal lhs = MakeElementwiseLhs();
  Literal rhs = MakeElementwiseRhs();
  ExpectMatchesElementwiseReference(
      actual,
      [&](int64_t i) {
        return std::fmod(lhs.Get<float>({i}),
                         std::abs(rhs.Get<float>({i})) + 1.0f);
      },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseAtan2) {
  auto result = ExecuteMetalElementwiseBinary(
      "metal_elementwise_atan2",
      [](XlaBuilder*, XlaOp lhs, XlaOp rhs) { Atan2(lhs, rhs); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal lhs = MakeElementwiseLhs();
  Literal rhs = MakeElementwiseRhs();
  ExpectMatchesElementwiseReference(
      actual,
      [&](int64_t i) {
        return std::atan2(lhs.Get<float>({i}), rhs.Get<float>({i}));
      },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseScalarMaximum) {
  auto result = ExecuteMetalElementwiseScalarMaximum();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal lhs = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::max(lhs.Get<float>({i}), 1.0f); });
}

TEST(MetalGpuExecutableTest, ElementwiseExp) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_exp",
      [](XlaBuilder*, XlaOp input) { Exp(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::exp(input.Get<float>({i})); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseSin) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_sin",
      [](XlaBuilder*, XlaOp input) { Sin(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::sin(input.Get<float>({i})); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseSqrt) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_sqrt",
      [](XlaBuilder* builder, XlaOp input) {
        Sqrt(Add(input,
                 Broadcast(ConstantR0<float>(builder, 3.0f),
                           {kElementCount})));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::sqrt(input.Get<float>({i}) + 3.0f); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseTanh) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_tanh",
      [](XlaBuilder*, XlaOp input) { Tanh(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::tanh(input.Get<float>({i})); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseCos) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_cos",
      [](XlaBuilder*, XlaOp input) { Cos(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::cos(input.Get<float>({i})); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseTan) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_tan",
      [](XlaBuilder*, XlaOp input) { Tan(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::tan(input.Get<float>({i})); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseAsin) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_asin",
      [](XlaBuilder* builder, XlaOp input) {
        XlaOp scaled = Mul(input, Broadcast(ConstantR0<float>(builder, 0.25f),
                                           {kElementCount}));
        Asin(scaled, /*result_accuracy=*/std::nullopt, /*expand=*/false);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual,
      [&](int64_t i) { return std::asin(input.Get<float>({i}) * 0.25f); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseAcos) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_acos",
      [](XlaBuilder* builder, XlaOp input) {
        XlaOp scaled = Mul(input, Broadcast(ConstantR0<float>(builder, 0.25f),
                                           {kElementCount}));
        Acos(scaled, /*result_accuracy=*/std::nullopt, /*expand=*/false);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual,
      [&](int64_t i) { return std::acos(input.Get<float>({i}) * 0.25f); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseSinh) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_sinh",
      [](XlaBuilder*, XlaOp input) {
        Sinh(input, /*result_accuracy=*/std::nullopt, /*expand=*/false);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::sinh(input.Get<float>({i})); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseCosh) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_cosh",
      [](XlaBuilder*, XlaOp input) {
        Cosh(input, /*result_accuracy=*/std::nullopt, /*expand=*/false);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::cosh(input.Get<float>({i})); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseAsinh) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_asinh",
      [](XlaBuilder*, XlaOp input) {
        Asinh(input, /*result_accuracy=*/std::nullopt, /*expand=*/false);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::asinh(input.Get<float>({i})); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseAcosh) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_acosh",
      [](XlaBuilder* builder, XlaOp input) {
        XlaOp shifted = Add(input, Broadcast(ConstantR0<float>(builder, 3.5f),
                                            {kElementCount}));
        Acosh(shifted, /*result_accuracy=*/std::nullopt, /*expand=*/false);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::acosh(input.Get<float>({i}) + 3.5f); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseAtanh) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_atanh",
      [](XlaBuilder* builder, XlaOp input) {
        XlaOp scaled = Mul(input, Broadcast(ConstantR0<float>(builder, 0.25f),
                                           {kElementCount}));
        Atanh(scaled, /*result_accuracy=*/std::nullopt, /*expand=*/false);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual,
      [&](int64_t i) { return std::atanh(input.Get<float>({i}) * 0.25f); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseLog) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_log",
      [](XlaBuilder* builder, XlaOp input) {
        Log(Add(input,
                Broadcast(ConstantR0<float>(builder, 3.0f), {kElementCount})));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::log(input.Get<float>({i}) + 3.0f); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseLog1p) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_log1p",
      [](XlaBuilder* builder, XlaOp input) {
        Log1p(Add(input,
                  Broadcast(ConstantR0<float>(builder, 3.0f),
                            {kElementCount})));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual,
      [&](int64_t i) { return std::log1p(input.Get<float>({i}) + 3.0f); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseExpm1) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_expm1",
      [](XlaBuilder*, XlaOp input) { Expm1(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::expm1(input.Get<float>({i})); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseRsqrt) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_rsqrt",
      [](XlaBuilder* builder, XlaOp input) {
        Rsqrt(Add(input,
                  Broadcast(ConstantR0<float>(builder, 3.0f),
                            {kElementCount})));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual,
      [&](int64_t i) { return 1.0f / std::sqrt(input.Get<float>({i}) + 3.0f); },
      1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseFloor) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_floor",
      [](XlaBuilder*, XlaOp input) { Floor(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::floor(input.Get<float>({i})); });
}

TEST(MetalGpuExecutableTest, ElementwiseCeil) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_ceil",
      [](XlaBuilder*, XlaOp input) { Ceil(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::ceil(input.Get<float>({i})); });
}

TEST(MetalGpuExecutableTest, ElementwiseRoundNearestEven) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_round_nearest_even",
      [](XlaBuilder*, XlaOp input) { RoundNearestEven(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::rint(input.Get<float>({i})); });
}

TEST(MetalGpuExecutableTest, ElementwiseSign) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_sign",
      [](XlaBuilder*, XlaOp input) { Sign(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(actual, [&](int64_t i) {
    const float value = input.Get<float>({i});
    return value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
  });
}

TEST(MetalGpuExecutableTest, ElementwiseErf) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_erf", [](XlaBuilder*, XlaOp input) { Erf(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return std::erf(input.Get<float>({i})); },
      1.5e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseLogistic) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_logistic",
      [](XlaBuilder*, XlaOp input) { Logistic(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(actual, [&](int64_t i) {
    const float value = input.Get<float>({i});
    return 1.0f / (1.0f + std::exp(-value));
  }, 1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseSelectCompare) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_select_compare",
      [](XlaBuilder* builder, XlaOp input) {
        XlaOp zero =
            Broadcast(ConstantR0<float>(builder, 0.0f), {kElementCount});
        Select(Gt(input, zero), input, Neg(input));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(actual, [&](int64_t i) {
    const float value = input.Get<float>({i});
    return value > 0.0f ? value : -value;
  });
}

TEST(MetalGpuExecutableTest, ElementwiseCompareRoot) {
  auto result = ExecuteMetalElementwiseBinary(
      "metal_elementwise_compare_root",
      [](XlaBuilder*, XlaOp lhs, XlaOp rhs) { Gt(lhs, rhs); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal lhs = MakeElementwiseLhs();
  Literal rhs = MakeElementwiseRhs();
  ExpectMatchesPredReference(
      actual, [&](int64_t i) { return lhs.Get<float>({i}) > rhs.Get<float>({i}); });
}

TEST(MetalGpuExecutableTest, ElementwisePredicateCompare) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_predicate_compare",
      [](XlaBuilder* builder, XlaOp input) {
        XlaOp zero =
            Broadcast(ConstantR0<float>(builder, 0.0f), {kElementCount});
        Ne(Gt(input, zero), Lt(input, zero));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesPredReference(actual, [&](int64_t i) {
    const float value = input.Get<float>({i});
    return (value > 0.0f) != (value < 0.0f);
  });
}

TEST(MetalGpuExecutableTest, ElementwiseConvertCompareToF32) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_convert_compare_to_f32",
      [](XlaBuilder* builder, XlaOp input) {
        XlaOp zero =
            Broadcast(ConstantR0<float>(builder, 0.0f), {kElementCount});
        ConvertElementType(Gt(input, zero), F32);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(actual, [&](int64_t i) {
    return input.Get<float>({i}) > 0.0f ? 1.0f : 0.0f;
  });
}

TEST(MetalGpuExecutableTest, ElementwiseWhereCall) {
  auto result = ExecuteMetalElementwiseWhereCall();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(actual, [&](int64_t i) {
    const float value = input.Get<float>({i});
    return value > 0.0f ? value : -value;
  });
}

TEST(MetalGpuExecutableTest, ElementwisePadCall) {
  auto result = ExecuteMetalPadCall();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesPadReference(actual, MakeElementwiseLhs(), /*low_padding=*/2,
                            /*high_padding=*/3, /*pad_value=*/1.5f);
}

TEST(MetalGpuExecutableTest, ElementwiseGatherSelect) {
  auto result = ExecuteMetalGatherSelect();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesGatherReference(actual, MakeGatherInput());
}

TEST(MetalGpuExecutableTest, ElementwiseSlice) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_slice",
      [](XlaBuilder*, XlaOp input) { Slice(input, {3}, {200}, {1}); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesSliceReference(actual, MakeElementwiseLhs(), /*start=*/3,
                              /*size=*/197);
}

TEST(MetalGpuExecutableTest, ElementwiseReshapeSlice) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_reshape_slice",
      [](XlaBuilder*, XlaOp input) {
        Reshape(Slice(input, {0}, {256}, {1}), {16, 16});
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesReshapeSliceReference(actual, MakeElementwiseLhs());
}

TEST(MetalGpuExecutableTest, ElementwiseTransposeSlice) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_transpose_slice",
      [](XlaBuilder*, XlaOp input) {
        Transpose(Reshape(Slice(input, {0}, {256}, {1}), {16, 16}), {1, 0});
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesTransposeSliceReference(actual, MakeElementwiseLhs());
}

TEST(MetalGpuExecutableTest, ElementwiseConcatenateSlices) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_concatenate_slices",
      [](XlaBuilder*, XlaOp input) {
        ConcatInDim(input.builder(),
                    {Slice(input, {0}, {128}, {1}),
                     Slice(input, {128}, {256}, {1})},
                    0);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesSliceReference(actual, MakeElementwiseLhs(), /*start=*/0,
                              /*size=*/256);
}

TEST(MetalGpuExecutableTest, ElementwiseScalarSliceBroadcast) {
  auto result = ExecuteMetalElementwiseUnary(
      "metal_elementwise_scalar_slice_broadcast",
      [](XlaBuilder* builder, XlaOp input) {
        XlaOp scalar = Reshape(Slice(input, {0}, {1}, {1}), {});
        Add(Broadcast(scalar, {kElementCount}),
            Broadcast(ConstantR0<float>(builder, 1.0f), {kElementCount}));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t) { return input.Get<float>({0}) + 1.0f; });
}

TEST(MetalGpuExecutableTest, ElementwiseIota) {
  auto result = ExecuteMetalIota();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {16})));
  for (int64_t i = 0; i < 16; ++i) {
    EXPECT_EQ(actual.Get<float>({i}), static_cast<float>(i + 1));
  }
}

TEST(MetalGpuExecutableTest, ElementwiseS32Iota) {
  auto result = ExecuteMetalS32Iota();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(S32, {16})));
  for (int64_t i = 0; i < 16; ++i) {
    EXPECT_EQ(actual.Get<int32_t>({i}), static_cast<int32_t>(i + 1));
  }
}

TEST(MetalGpuExecutableTest, ElementwiseS32Add) {
  auto result = ExecuteMetalS32Add();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseS32();
  ExpectMatchesS32Reference(
      actual, [&](int64_t i) { return input.Get<int32_t>({i}) * 2; });
}

TEST(MetalGpuExecutableTest, ElementwiseS32Remainder) {
  auto result = ExecuteMetalS32Unary(
      "metal_s32_remainder", [](XlaBuilder* builder, XlaOp input) {
        Rem(input, Broadcast(ConstantR0<int32_t>(builder, 7), {kElementCount}));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseS32();
  ExpectMatchesS32Reference(
      actual, [&](int64_t i) { return input.Get<int32_t>({i}) % 7; });
}

TEST(MetalGpuExecutableTest, ElementwiseS32Bitwise) {
  auto result = ExecuteMetalS32Unary(
      "metal_s32_bitwise", [](XlaBuilder* builder, XlaOp input) {
        XlaOp mask = Broadcast(ConstantR0<int32_t>(builder, 7), {kElementCount});
        XlaOp high =
            Broadcast(ConstantR0<int32_t>(builder, 16), {kElementCount});
        XlaOp toggle =
            Broadcast(ConstantR0<int32_t>(builder, 3), {kElementCount});
        Xor(Or(And(input, mask), high), toggle);
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseS32();
  ExpectMatchesS32Reference(actual, [&](int64_t i) {
    return ((input.Get<int32_t>({i}) & 7) | 16) ^ 3;
  });
}

TEST(MetalGpuExecutableTest, ElementwiseS32Shifts) {
  auto result = ExecuteMetalS32Unary(
      "metal_s32_shifts", [](XlaBuilder* builder, XlaOp input) {
        XlaOp amount =
            And(input,
                Broadcast(ConstantR0<int32_t>(builder, 3), {kElementCount}));
        XlaOp positive_lhs =
            And(input,
                Broadcast(ConstantR0<int32_t>(builder, 255), {kElementCount}));
        XlaOp left = ShiftLeft(positive_lhs, amount);
        XlaOp arithmetic = ShiftRightArithmetic(input, amount);
        XlaOp logical = ShiftRightLogical(input, amount);
        Xor(left, Xor(arithmetic, logical));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseS32();
  ExpectMatchesS32Reference(actual, [&](int64_t i) {
    const int32_t value = input.Get<int32_t>({i});
    const int32_t amount = value & 3;
    const int32_t left = (value & 255) << amount;
    const int32_t arithmetic = value >> amount;
    const int32_t logical =
        static_cast<int32_t>(static_cast<uint32_t>(value) >> amount);
    return left ^ (arithmetic ^ logical);
  });
}

TEST(MetalGpuExecutableTest, ElementwiseS32UnaryOps) {
  auto result = ExecuteMetalS32Unary(
      "metal_s32_unary_ops", [](XlaBuilder*, XlaOp input) {
        XlaOp sign = Sign(input);
        XlaOp abs = Abs(input);
        XlaOp neg = Neg(input);
        Add(sign, Add(abs, neg));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseS32();
  ExpectMatchesS32Reference(actual, [&](int64_t i) {
    const int32_t value = input.Get<int32_t>({i});
    const int32_t sign = (value > 0) ? 1 : ((value < 0) ? -1 : 0);
    return sign + std::abs(value) - value;
  });
}

TEST(MetalGpuExecutableTest, ElementwiseS32NotPopcount) {
  auto result = ExecuteMetalS32Unary(
      "metal_s32_not_popcount", [](XlaBuilder*, XlaOp input) {
        Add(Not(input), PopulationCount(input));
      });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseS32();
  ExpectMatchesS32Reference(actual, [&](int64_t i) {
    const int32_t value = input.Get<int32_t>({i});
    uint32_t bits = static_cast<uint32_t>(value);
    int32_t count = 0;
    while (bits != 0) {
      count += static_cast<int32_t>(bits & 1u);
      bits >>= 1;
    }
    return ~value + count;
  });
}

TEST(MetalGpuExecutableTest, ElementwisePredParameterNot) {
  auto result = ExecuteMetalPredUnary(
      "metal_pred_parameter_not",
      [](XlaBuilder*, XlaOp input) { Not(input); });
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwisePred();
  ExpectMatchesPredReference(
      actual, [&](int64_t i) { return !input.Get<bool>({i}); });
}

TEST(MetalGpuExecutableTest, ElementwiseS32CompareRoot) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_s32_compare_root");
  Shape shape = ShapeUtil::MakeShape(S32, {kElementCount});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  Gt(input, Broadcast(ConstantR0<int32_t>(&builder, 0), {kElementCount}));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwiseS32();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesPredReference(actual, [&](int64_t i) {
    return input_literal.Get<int32_t>({i}) > 0;
  });
}

TEST(MetalGpuExecutableTest, ConvertF32ToS32) {
  auto result = ExecuteMetalConvert(F32, S32);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesS32Reference(actual, [&](int64_t i) {
    return static_cast<int32_t>(input.Get<float>({i}));
  });
}

TEST(MetalGpuExecutableTest, ConvertS32ToF32) {
  auto result = ExecuteMetalConvert(S32, F32);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseS32();
  ExpectMatchesElementwiseReference(actual, [&](int64_t i) {
    return static_cast<float>(input.Get<int32_t>({i}));
  });
}

TEST(MetalGpuExecutableTest, ReductionSum) {
  auto result = ExecuteMetalReduction(HloOpcode::kAdd, 0.0f);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  float expected = 0.0f;
  for (int64_t i = 0; i < kElementCount; ++i) {
    expected += input.Get<float>({i});
  }
  ExpectScalarNear(actual, expected);
}

TEST(MetalGpuExecutableTest, ReductionMaximum) {
  auto result = ExecuteMetalReduction(
      HloOpcode::kMaximum, -std::numeric_limits<float>::infinity());
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  float expected = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < kElementCount; ++i) {
    expected = std::max(expected, input.Get<float>({i}));
  }
  ExpectScalarNear(actual, expected);
}

TEST(MetalGpuExecutableTest, ReductionMinimum) {
  auto result = ExecuteMetalReduction(
      HloOpcode::kMinimum, std::numeric_limits<float>::infinity());
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  float expected = std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < kElementCount; ++i) {
    expected = std::min(expected, input.Get<float>({i}));
  }
  ExpectScalarNear(actual, expected);
}

TEST(MetalGpuExecutableTest, ReductionProduct) {
  auto result = ExecuteMetalReduction(HloOpcode::kMultiply, 1.0f);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  float expected = 1.0f;
  for (int64_t i = 0; i < kElementCount; ++i) {
    expected *= input.Get<float>({i});
  }
  ExpectScalarNear(actual, expected);
}

TEST(MetalGpuExecutableTest, ReductionMean) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder reducer_builder("metal_mean_reducer");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  Add(lhs, rhs);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder("metal_reduction_mean");
  Shape input_shape = ShapeUtil::MakeShape(F32, {kElementCount});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp sum = Reduce(input, ConstantR0<float>(&builder, 0.0f), reducer, {0});
  Div(sum, ConstantR0<float>(&builder, static_cast<float>(kElementCount)));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwiseLhs();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  float expected = 0.0f;
  for (int64_t i = 0; i < kElementCount; ++i) {
    expected += input_literal.Get<float>({i});
  }
  expected /= static_cast<float>(kElementCount);
  ExpectScalarNear(actual, expected);
}

}  // namespace
}  // namespace xla
