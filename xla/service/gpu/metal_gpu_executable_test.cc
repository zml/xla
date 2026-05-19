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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

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

absl::StatusOr<Literal> ExecuteMetalBatchedDot() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_batched_dot");
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 2, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  XlaOp lhs = Parameter(&builder, 0, lhs_shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, rhs_shape, "rhs");
  DotDimensionNumbers dnums;
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  DotGeneral(lhs, rhs, dnums);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array3D<float> lhs_values(2, 2, 3);
  Array3D<float> rhs_values(2, 3, 4);
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t row = 0; row < 2; ++row) {
      for (int64_t col = 0; col < 3; ++col) {
        lhs_values(b, row, col) =
            static_cast<float>(b * 6 + row * 3 + col) * 0.1f;
      }
    }
    for (int64_t row = 0; row < 3; ++row) {
      for (int64_t col = 0; col < 4; ++col) {
        rhs_values(b, row, col) =
            static_cast<float>(b * 12 + row * 4 + col) / 7.0f;
      }
    }
  }
  Literal lhs_literal = LiteralUtil::CreateR3FromArray3D(lhs_values);
  Literal rhs_literal = LiteralUtil::CreateR3FromArray3D(rhs_values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> lhs_data,
                      client->TransferToServer(lhs_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> rhs_data,
                      client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalBatchedDotBf16() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_batched_dot_bf16");
  Shape lhs_shape = ShapeUtil::MakeShape(BF16, {2, 2, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(BF16, {2, 3, 4});
  XlaOp lhs = Parameter(&builder, 0, lhs_shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, rhs_shape, "rhs");
  DotDimensionNumbers dnums;
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  DotGeneral(lhs, rhs, dnums);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal lhs_literal = Literal::CreateFromShape(lhs_shape);
  Literal rhs_literal = Literal::CreateFromShape(rhs_shape);
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t row = 0; row < 2; ++row) {
      for (int64_t col = 0; col < 3; ++col) {
        lhs_literal.Set<bfloat16>(
            {b, row, col},
            bfloat16(static_cast<float>(b * 6 + row * 3 + col) * 0.125f));
      }
    }
    for (int64_t row = 0; row < 3; ++row) {
      for (int64_t col = 0; col < 4; ++col) {
        rhs_literal.Set<bfloat16>(
            {b, row, col},
            bfloat16(static_cast<float>(b * 12 + row * 4 + col) * 0.0625f));
      }
    }
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> lhs_data,
                      client->TransferToServer(lhs_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> rhs_data,
                      client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalBatchedDotF16Rank4() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_batched_dot_f16_rank4");
  Shape lhs_shape = ShapeUtil::MakeShape(F16, {1, 2, 2, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F16, {1, 2, 3, 1});
  XlaOp lhs = Parameter(&builder, 0, lhs_shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, rhs_shape, "rhs");
  DotDimensionNumbers dnums;
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_lhs_batch_dimensions(1);
  dnums.add_rhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(1);
  dnums.add_lhs_contracting_dimensions(3);
  dnums.add_rhs_contracting_dimensions(2);
  DotGeneral(lhs, rhs, dnums);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal lhs_literal = Literal::CreateFromShape(lhs_shape);
  Literal rhs_literal = Literal::CreateFromShape(rhs_shape);
  for (int64_t b1 = 0; b1 < 2; ++b1) {
    for (int64_t row = 0; row < 2; ++row) {
      for (int64_t col = 0; col < 3; ++col) {
        lhs_literal.Set<half>(
            {0, b1, row, col},
            half(static_cast<float>(b1 * 6 + row * 3 + col) * 0.125f));
      }
    }
    for (int64_t row = 0; row < 3; ++row) {
      rhs_literal.Set<half>(
          {0, b1, row, 0},
          half(static_cast<float>(b1 * 3 + row) * 0.25f - 0.5f));
    }
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> lhs_data,
                      client->TransferToServer(lhs_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> rhs_data,
                      client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalConv1D() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_conv_1d");
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 5, 1});
  Shape kernel_shape = ShapeUtil::MakeShape(F32, {3, 1, 1});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp kernel = Parameter(&builder, 1, kernel_shape, "kernel");
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.set_input_feature_dimension(2);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.set_kernel_output_feature_dimension(2);
  dnums.set_output_batch_dimension(0);
  dnums.add_output_spatial_dimensions(1);
  dnums.set_output_feature_dimension(2);
  ConvWithGeneralDimensions(input, kernel, {1}, Padding::kValid, dnums);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array3D<float> input_values(1, 5, 1);
  for (int64_t i = 0; i < 5; ++i) {
    input_values(0, i, 0) = static_cast<float>(i);
  }
  Array3D<float> kernel_values(3, 1, 1);
  kernel_values(0, 0, 0) = 1.0f;
  kernel_values(1, 0, 0) = 2.0f;
  kernel_values(2, 0, 0) = -1.0f;
  Literal input_literal = LiteralUtil::CreateR3FromArray3D(input_values);
  Literal kernel_literal = LiteralUtil::CreateR3FromArray3D(kernel_values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> kernel_data,
                      client->TransferToServer(kernel_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), kernel_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalConv0DIsDot() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_conv_0d_is_dot");
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {10, 5});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {5, 7});
  XlaOp lhs = Parameter(&builder, 0, lhs_shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, rhs_shape, "rhs");
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.set_output_batch_dimension(0);
  dnums.set_output_feature_dimension(1);
  ConvGeneralDilated(lhs, rhs, {}, {}, {}, {}, dnums);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> lhs_values(10, 5);
  for (int64_t row = 0; row < 10; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      lhs_values(row, col) = static_cast<float>((row * 5 + col) % 9 - 4);
    }
  }
  Array2D<float> rhs_values(5, 7);
  for (int64_t row = 0; row < 5; ++row) {
    for (int64_t col = 0; col < 7; ++col) {
      rhs_values(row, col) =
          static_cast<float>((row * 7 + col) % 11 - 5) * 0.25f;
    }
  }
  Literal lhs_literal = LiteralUtil::CreateR2FromArray2D(lhs_values);
  Literal rhs_literal = LiteralUtil::CreateR2FromArray2D(rhs_values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> lhs_data,
                      client->TransferToServer(lhs_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> rhs_data,
                      client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalConv2D() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_conv_2d");
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 1});
  Shape kernel_shape = ShapeUtil::MakeShape(F32, {2, 2, 1, 1});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp kernel = Parameter(&builder, 1, kernel_shape, "kernel");
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);
  dnums.set_output_batch_dimension(0);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_output_feature_dimension(3);
  ConvWithGeneralDimensions(input, kernel, {1, 1}, Padding::kValid, dnums);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array4D<float> input_values(1, 4, 4, 1);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      input_values(0, row, col, 0) = static_cast<float>(row * 4 + col) / 4.0f;
    }
  }
  Array4D<float> kernel_values(2, 2, 1, 1);
  kernel_values(0, 0, 0, 0) = 1.0f;
  kernel_values(0, 1, 0, 0) = 1.0f;
  kernel_values(1, 0, 0, 0) = 1.0f;
  kernel_values(1, 1, 0, 0) = 1.0f;
  Literal input_literal = LiteralUtil::CreateR4FromArray4D(input_values);
  Literal kernel_literal = LiteralUtil::CreateR4FromArray4D(kernel_values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> kernel_data,
                      client->TransferToServer(kernel_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), kernel_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalConv2DNchwF16SameLower() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_conv_2d_nchw_f16_same_lower");
  Shape input_shape = ShapeUtil::MakeShape(F16, {2, 2, 9, 10});
  Shape kernel_shape = ShapeUtil::MakeShape(F16, {3, 2, 4, 5});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp kernel = Parameter(&builder, 1, kernel_shape, "kernel");
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.set_output_batch_dimension(0);
  dnums.set_output_feature_dimension(1);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(3);
  std::vector<std::pair<int64_t, int64_t>> padding = {{2, 1}, {2, 2}};
  ConvGeneralDilated(input, kernel, {2, 1}, padding, {}, {}, dnums);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = Literal::CreateFromShape(input_shape);
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t c = 0; c < 2; ++c) {
      for (int64_t h = 0; h < 9; ++h) {
        for (int64_t w = 0; w < 10; ++w) {
          const float value =
              static_cast<float>((b * 180 + c * 90 + h * 10 + w) % 17 - 8) /
              8.0f;
          input_literal.Set<half>({b, c, h, w}, half(value));
        }
      }
    }
  }
  Literal kernel_literal = Literal::CreateFromShape(kernel_shape);
  for (int64_t o = 0; o < 3; ++o) {
    for (int64_t c = 0; c < 2; ++c) {
      for (int64_t h = 0; h < 4; ++h) {
        for (int64_t w = 0; w < 5; ++w) {
          const float value =
              static_cast<float>((o * 40 + c * 20 + h * 5 + w) % 11 - 5) /
              16.0f;
          kernel_literal.Set<half>({o, c, h, w}, half(value));
        }
      }
    }
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> kernel_data,
                      client->TransferToServer(kernel_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), kernel_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalRfft() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_rfft");
  Shape input_shape = ShapeUtil::MakeShape(F32, {8});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  Fft(input, FftType::RFFT, {8});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal =
      LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f, 7.0f});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalComplexTranspose() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_complex_transpose");
  Shape input_shape = ShapeUtil::MakeShape(C64, {2, 3});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  Transpose(input, {1, 0});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = Literal::CreateFromShape(input_shape);
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      input_literal.Set<complex64>(
          {row, col}, complex64(static_cast<float>(row * 3 + col),
                                static_cast<float>(row - col)));
    }
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
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

absl::StatusOr<Literal> ExecuteMetalPadRank2() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_elementwise_pad_rank2");
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  PaddingConfig padding;
  PaddingConfig::PaddingConfigDimension* rows = padding.add_dimensions();
  rows->set_edge_padding_low(1);
  rows->set_edge_padding_high(1);
  rows->set_interior_padding(0);
  PaddingConfig::PaddingConfigDimension* cols = padding.add_dimensions();
  cols->set_edge_padding_low(2);
  cols->set_edge_padding_high(1);
  cols->set_interior_padding(0);
  Pad(input, ConstantR0<float>(&builder, 5.0f), padding);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> values(2, 3);
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      values(row, col) = static_cast<float>(row * 3 + col);
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
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

absl::StatusOr<Literal> ExecuteMetalGatherRank2(bool gather_rows) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_gather_rank2");
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  Shape index_shape = ShapeUtil::MakeShape(S32, {2, 1});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp indices = Parameter(&builder, 1, index_shape, "indices");

  GatherDimensionNumbers dnums;
  dnums.set_index_vector_dim(1);
  if (gather_rows) {
    dnums.add_offset_dims(1);
    dnums.add_collapsed_slice_dims(0);
    dnums.add_start_index_map(0);
    Gather(input, indices, dnums, {1, 5}, /*indices_are_sorted=*/false);
  } else {
    dnums.add_offset_dims(0);
    dnums.add_collapsed_slice_dims(1);
    dnums.add_start_index_map(1);
    Gather(input, indices, dnums, {4, 1}, /*indices_are_sorted=*/false);
  }
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> values(4, 5);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      values(row, col) = static_cast<float>(row * 5 + col);
    }
  }
  Array2D<int32_t> index_values(2, 1);
  index_values(0, 0) = gather_rows ? 0 : 1;
  index_values(1, 0) = gather_rows ? 2 : 3;
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  Literal index_literal = LiteralUtil::CreateR2FromArray2D(index_values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> index_data,
                      client->TransferToServer(index_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), index_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalSortRank1() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaBuilder comparator_builder("metal_sort_comparator");
  XlaOp lhs = Parameter(&comparator_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&comparator_builder, 1, scalar_shape, "rhs");
  Lt(lhs, rhs);
  TF_ASSIGN_OR_RETURN(XlaComputation comparator, comparator_builder.Build());

  XlaBuilder builder("metal_sort_rank1");
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  Sort({input}, comparator, 0, /*is_stable=*/true);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR1<float>(
      {1.5f, -2.0f, 0.25f, 4.0f, 0.25f, -1.0f, 3.0f, 2.0f});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalSortU16Rank1() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape scalar_shape = ShapeUtil::MakeShape(U16, {});
  XlaBuilder comparator_builder("metal_sort_u16_comparator");
  XlaOp lhs = Parameter(&comparator_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&comparator_builder, 1, scalar_shape, "rhs");
  Lt(lhs, rhs);
  TF_ASSIGN_OR_RETURN(XlaComputation comparator, comparator_builder.Build());

  XlaBuilder builder("metal_sort_u16_rank1");
  Shape shape = ShapeUtil::MakeShape(U16, {8});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  Sort({input}, comparator, 0, /*is_stable=*/true);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR1<uint16_t>(
      {9, 1, 65535, 3, 1, 42, 0, 17});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalSortRank2() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaBuilder comparator_builder("metal_sort_rank2_comparator");
  XlaOp lhs = Parameter(&comparator_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&comparator_builder, 1, scalar_shape, "rhs");
  Lt(lhs, rhs);
  TF_ASSIGN_OR_RETURN(XlaComputation comparator, comparator_builder.Build());

  XlaBuilder builder("metal_sort_rank2");
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  Sort({input}, comparator, 1, /*is_stable=*/true);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> values(2, 4);
  values(0, 0) = 3.0f;
  values(0, 1) = 1.0f;
  values(0, 2) = 2.0f;
  values(0, 3) = 1.0f;
  values(1, 0) = 0.0f;
  values(1, 1) = 5.0f;
  values(1, 2) = 4.0f;
  values(1, 3) = -1.0f;
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalTopKValuesRank1() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_topk_values_rank1");
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  GetTupleElement(TopK(input, 3, /*largest=*/true), 0);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR1<float>(
      {1.5f, -2.0f, 0.25f, 4.0f, 0.25f, -1.0f, 3.0f, 2.0f});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalTopKIndicesRank1() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_topk_indices_rank1");
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  GetTupleElement(TopK(input, 3, /*largest=*/true), 1);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR1<float>(
      {1.5f, -2.0f, 0.25f, 4.0f, 0.25f, -1.0f, 3.0f, 2.0f});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalTopKRank2(bool indices) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder(indices ? "metal_topk_indices_rank2"
                            : "metal_topk_values_rank2");
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  GetTupleElement(TopK(input, 2, /*largest=*/true), indices ? 1 : 0);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> values(2, 4);
  values(0, 0) = 1.0f;
  values(0, 1) = 4.0f;
  values(0, 2) = 3.0f;
  values(0, 3) = 4.0f;
  values(1, 0) = 6.0f;
  values(1, 1) = 2.0f;
  values(1, 2) = 5.0f;
  values(1, 3) = 0.0f;
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalTopKTupleRank1() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_topk_tuple_rank1");
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  TopK(input, 3, /*largest=*/true);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR1<float>(
      {1.5f, -2.0f, 0.25f, 4.0f, 0.25f, -1.0f, 3.0f, 2.0f});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalTopKTupleS32Rank1() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_topk_tuple_s32_rank1");
  Shape shape = ShapeUtil::MakeShape(S32, {8});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  XlaOp topk = TopK(input, 3, /*largest=*/true);
  Tuple(&builder, {GetTupleElement(topk, 0), GetTupleElement(topk, 1)});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal =
      LiteralUtil::CreateR1<int32_t>({2, -4, 7, 7, 1, 9, -1, 3});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalTopKTupleRank2() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_topk_tuple_rank2");
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  TopK(input, 2, /*largest=*/true);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> values(2, 4);
  values(0, 0) = 1.0f;
  values(0, 1) = 4.0f;
  values(0, 2) = 3.0f;
  values(0, 3) = 4.0f;
  values(1, 0) = 6.0f;
  values(1, 1) = 2.0f;
  values(1, 2) = 5.0f;
  values(1, 3) = 0.0f;
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalArgmaxRank1() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_argmax_rank1");
  Shape shape = ShapeUtil::MakeShape(F32, {5});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  ArgMax(input, S32, 0);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal =
      LiteralUtil::CreateR1<float>({1.0f, 7.0f, 3.0f, 7.0f, 6.0f});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalSortDescendingExpression() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaBuilder comparator_builder("metal_desc_sort_comparator");
  XlaOp lhs = Parameter(&comparator_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&comparator_builder, 1, scalar_shape, "rhs");
  Lt(lhs, rhs);
  TF_ASSIGN_OR_RETURN(XlaComputation comparator, comparator_builder.Build());

  XlaBuilder builder("metal_sort_descending_expression");
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  Neg(Sort({Neg(input)}, comparator, 0, /*is_stable=*/true));
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR1<float>(
      {1.5f, -2.0f, 0.25f, 4.0f, 0.25f, -1.0f, 3.0f, 2.0f});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalClamp() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_clamp");
  Shape shape = ShapeUtil::MakeShape(F32, {kElementCount});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  Clamp(Broadcast(ConstantR0<float>(&builder, -0.5f), {kElementCount}), input,
        Broadcast(ConstantR0<float>(&builder, 0.75f), {kElementCount}));
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwiseLhs();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalClampS16ScalarBounds() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_clamp_s16_scalar_bounds");
  Shape scalar_shape = ShapeUtil::MakeShape(S16, {});
  Shape input_shape = ShapeUtil::MakeShape(S16, {2, 3});
  XlaOp min = Parameter(&builder, 0, scalar_shape, "min");
  XlaOp input = Parameter(&builder, 1, input_shape, "input");
  XlaOp max = Parameter(&builder, 2, scalar_shape, "max");
  Clamp(min, input, max);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal min_literal = LiteralUtil::CreateR0<int16_t>(-2);
  Literal input_literal = Literal::CreateFromShape(input_shape);
  int16_t value = -4;
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      input_literal.Set<int16_t>({row, col}, value++);
    }
  }
  Literal max_literal = LiteralUtil::CreateR0<int16_t>(1);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> min_data,
                      client->TransferToServer(min_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> max_data,
                      client->TransferToServer(max_literal));
  std::vector<GlobalData*> arguments = {min_data.get(), input_data.get(),
                                        max_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalReverseRank2() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_reverse_rank2");
  Shape shape = ShapeUtil::MakeShape(F32, {3, 4});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  Rev(input, {0, 1});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> values(3, 4);
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      values(row, col) = static_cast<float>(row * 4 + col);
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalDynamicSliceDynamicStart() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_dynamic_slice_dynamic_start");
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp row_start = Parameter(&builder, 1, scalar_shape, "row_start");
  DynamicSlice(input, {row_start, ConstantR0<int32_t>(&builder, 2)}, {2, 2});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> values(4, 5);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      values(row, col) = static_cast<float>(row * 5 + col);
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  Literal row_start_literal = LiteralUtil::CreateR0<int32_t>(1);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> row_start_data,
                      client->TransferToServer(row_start_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), row_start_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalS32DynamicSliceDynamicStart() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_s32_dynamic_slice_dynamic_start");
  Shape input_shape = ShapeUtil::MakeShape(S32, {4, 5});
  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp row_start = Parameter(&builder, 1, scalar_shape, "row_start");
  DynamicSlice(input, {row_start, ConstantR0<int32_t>(&builder, 2)}, {2, 2});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<int32_t> values(4, 5);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      values(row, col) = static_cast<int32_t>(row * 5 + col);
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  Literal row_start_literal = LiteralUtil::CreateR0<int32_t>(1);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> row_start_data,
                      client->TransferToServer(row_start_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), row_start_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalDynamicSliceRank3() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_dynamic_slice_rank3");
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp dim0 = Parameter(&builder, 1, scalar_shape, "dim0");
  XlaOp dim1 = Parameter(&builder, 2, scalar_shape, "dim1");
  XlaOp dim2 = Parameter(&builder, 3, scalar_shape, "dim2");
  DynamicSlice(input, {dim0, dim1, dim2}, {1, 2, 3});
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array3D<float> values(2, 3, 4);
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        values(i, j, k) = static_cast<float>(i * 12 + j * 4 + k);
      }
    }
  }
  Literal input_literal = LiteralUtil::CreateR3FromArray3D(values);
  Literal dim0_literal = LiteralUtil::CreateR0<int32_t>(1);
  Literal dim1_literal = LiteralUtil::CreateR0<int32_t>(1);
  Literal dim2_literal = LiteralUtil::CreateR0<int32_t>(1);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> dim0_data,
                      client->TransferToServer(dim0_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> dim1_data,
                      client->TransferToServer(dim1_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> dim2_data,
                      client->TransferToServer(dim2_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), dim0_data.get(),
                                        dim1_data.get(), dim2_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalConcatenateRank2(int64_t dim) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_concatenate_rank2");
  Shape lhs_shape =
      dim == 0 ? ShapeUtil::MakeShape(F32, {2, 3})
               : ShapeUtil::MakeShape(F32, {3, 2});
  Shape rhs_shape =
      dim == 0 ? ShapeUtil::MakeShape(F32, {1, 3})
               : ShapeUtil::MakeShape(F32, {3, 1});
  XlaOp lhs = Parameter(&builder, 0, lhs_shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, rhs_shape, "rhs");
  ConcatInDim(&builder, {lhs, rhs}, dim);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> lhs_values(lhs_shape.dimensions(0), lhs_shape.dimensions(1));
  for (int64_t row = 0; row < lhs_shape.dimensions(0); ++row) {
    for (int64_t col = 0; col < lhs_shape.dimensions(1); ++col) {
      lhs_values(row, col) = static_cast<float>(row * lhs_shape.dimensions(1) +
                                                col);
    }
  }
  Array2D<float> rhs_values(rhs_shape.dimensions(0), rhs_shape.dimensions(1));
  for (int64_t row = 0; row < rhs_shape.dimensions(0); ++row) {
    for (int64_t col = 0; col < rhs_shape.dimensions(1); ++col) {
      rhs_values(row, col) =
          100.0f + static_cast<float>(row * rhs_shape.dimensions(1) + col);
    }
  }
  Literal lhs_literal = LiteralUtil::CreateR2FromArray2D(lhs_values);
  Literal rhs_literal = LiteralUtil::CreateR2FromArray2D(rhs_values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> lhs_data,
                      client->TransferToServer(lhs_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> rhs_data,
                      client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalConcatenateRank3(int64_t dim) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_concatenate_rank3");
  std::vector<int64_t> operand_sizes = {3, 1, 4};
  std::vector<Shape> operand_shapes;
  std::vector<XlaOp> operands;
  for (int64_t operand_size : operand_sizes) {
    std::vector<int64_t> dimensions = {2, 3, 4};
    dimensions[dim] = operand_size;
    operand_shapes.push_back(ShapeUtil::MakeShape(S32, dimensions));
    operands.push_back(
        Parameter(&builder, operands.size(), operand_shapes.back(), "input"));
  }
  ConcatInDim(&builder, operands, dim);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  std::vector<Literal> literals;
  for (int64_t operand_index = 0; operand_index < operand_shapes.size();
       ++operand_index) {
    const Shape& shape = operand_shapes[operand_index];
    Literal literal = Literal::CreateFromShape(shape);
    for (int64_t i = 0; i < shape.dimensions(0); ++i) {
      for (int64_t j = 0; j < shape.dimensions(1); ++j) {
        for (int64_t k = 0; k < shape.dimensions(2); ++k) {
          int32_t value = static_cast<int32_t>(
              operand_index * 1000 + (i * shape.dimensions(1) + j) *
                                         shape.dimensions(2) +
              k);
          literal.Set<int32_t>({i, j, k}, value);
        }
      }
    }
    literals.push_back(std::move(literal));
  }

  std::vector<std::unique_ptr<GlobalData>> data;
  std::vector<GlobalData*> arguments;
  for (const Literal& literal : literals) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                        client->TransferToServer(literal));
    arguments.push_back(input_data.get());
    data.push_back(std::move(input_data));
  }
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalScatter(bool add) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaBuilder reducer_builder(add ? "metal_scatter_add_reducer"
                                 : "metal_scatter_set_reducer");
  XlaOp current = Parameter(&reducer_builder, 0, scalar_shape, "current");
  XlaOp update = Parameter(&reducer_builder, 1, scalar_shape, "update");
  if (add) {
    Add(current, update);
  }
  TF_ASSIGN_OR_RETURN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder(add ? "metal_scatter_add" : "metal_scatter_set");
  Shape input_shape = ShapeUtil::MakeShape(F32, {8});
  Shape update_shape = ShapeUtil::MakeShape(F32, {3});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp updates = Parameter(&builder, 1, update_shape, "updates");
  Array2D<int32_t> index_values(3, 1);
  index_values(0, 0) = 1;
  index_values(1, 0) = 3;
  index_values(2, 0) = 6;
  XlaOp indices = ConstantR2FromArray2D<int32_t>(&builder, index_values);
  ScatterDimensionNumbers dnums;
  dnums.add_inserted_window_dims(0);
  dnums.add_scatter_dims_to_operand_dims(0);
  dnums.set_index_vector_dim(1);
  Scatter(input, indices, updates, reducer, dnums,
          /*indices_are_sorted=*/true, /*unique_indices=*/true);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal =
      LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f, 7.0f});
  Literal update_literal = LiteralUtil::CreateR1<float>({-1.0f, -2.0f, -3.0f});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> update_data,
                      client->TransferToServer(update_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), update_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalScatterRank2(bool add) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaBuilder reducer_builder(add ? "metal_scatter_rank2_add_reducer"
                                 : "metal_scatter_rank2_set_reducer");
  XlaOp current = Parameter(&reducer_builder, 0, scalar_shape, "current");
  XlaOp update = Parameter(&reducer_builder, 1, scalar_shape, "update");
  if (add) {
    Add(current, update);
  }
  TF_ASSIGN_OR_RETURN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder(add ? "metal_scatter_rank2_add"
                         : "metal_scatter_rank2_set");
  Shape input_shape = ShapeUtil::MakeShape(F32, {3, 4});
  Shape index_shape = ShapeUtil::MakeShape(S32, {3, 2});
  Shape update_shape = ShapeUtil::MakeShape(F32, {3});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp indices = Parameter(&builder, 1, index_shape, "indices");
  XlaOp updates = Parameter(&builder, 2, update_shape, "updates");
  ScatterDimensionNumbers dnums;
  dnums.add_inserted_window_dims(0);
  dnums.add_inserted_window_dims(1);
  dnums.add_scatter_dims_to_operand_dims(0);
  dnums.add_scatter_dims_to_operand_dims(1);
  dnums.set_index_vector_dim(1);
  Scatter(input, indices, updates, reducer, dnums,
          /*indices_are_sorted=*/false, /*unique_indices=*/true);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> input_values(3, 4);
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      input_values(row, col) = static_cast<float>(row * 4 + col);
    }
  }
  Array2D<int32_t> index_values(3, 2);
  index_values(0, 0) = 0;
  index_values(0, 1) = 1;
  index_values(1, 0) = 2;
  index_values(1, 1) = 3;
  index_values(2, 0) = 1;
  index_values(2, 1) = 2;
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_values);
  Literal index_literal = LiteralUtil::CreateR2FromArray2D(index_values);
  Literal update_literal = LiteralUtil::CreateR1<float>({-5.0f, 7.0f, 4.0f});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> index_data,
                      client->TransferToServer(index_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> update_data,
                      client->TransferToServer(update_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), index_data.get(),
                                        update_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalReduceWindow(HloOpcode opcode,
                                                 float init_value) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder reducer_builder("metal_reduce_window_reducer");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  switch (opcode) {
    case HloOpcode::kAdd:
      Add(lhs, rhs);
      break;
    case HloOpcode::kMaximum:
      Max(lhs, rhs);
      break;
    default:
      return absl::InvalidArgumentError("Unexpected reduce-window opcode.");
  }
  TF_ASSIGN_OR_RETURN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder("metal_reduce_window");
  Shape input_shape = ShapeUtil::MakeShape(F32, {16});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  ReduceWindow(input, ConstantR0<float>(&builder, init_value), reducer, {3},
               {1}, Padding::kValid);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  std::vector<float> values(16);
  for (int64_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i) * 0.25f - 2.0f;
  }
  Literal input_literal = LiteralUtil::CreateR1<float>(values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalReduceWindowRank2(HloOpcode opcode,
                                                     float init_value) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder reducer_builder("metal_reduce_window_rank2_reducer");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  switch (opcode) {
    case HloOpcode::kAdd:
      Add(lhs, rhs);
      break;
    case HloOpcode::kMaximum:
      Max(lhs, rhs);
      break;
    default:
      return absl::InvalidArgumentError("Unexpected reduce-window opcode.");
  }
  TF_ASSIGN_OR_RETURN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder("metal_reduce_window_rank2");
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  ReduceWindow(input, ConstantR0<float>(&builder, init_value), reducer, {2, 2},
               {1, 1}, Padding::kValid);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Array2D<float> values(4, 5);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      values(row, col) = static_cast<float>(row * 5 + col);
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalConditional() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape shape = ShapeUtil::MakeShape(F32, {kElementCount});
  XlaBuilder true_builder("metal_cond_true");
  XlaOp true_input = Parameter(&true_builder, 0, shape, "input");
  Add(true_input, Broadcast(ConstantR0<float>(&true_builder, 1.0f),
                            {kElementCount}));
  TF_ASSIGN_OR_RETURN(XlaComputation true_computation, true_builder.Build());

  XlaBuilder false_builder("metal_cond_false");
  XlaOp false_input = Parameter(&false_builder, 0, shape, "input");
  Sub(false_input, Broadcast(ConstantR0<float>(&false_builder, 1.0f),
                             {kElementCount}));
  TF_ASSIGN_OR_RETURN(XlaComputation false_computation, false_builder.Build());

  XlaBuilder builder("metal_conditional");
  XlaOp input = Parameter(&builder, 0, shape, "input");
  XlaOp pred = Parameter(&builder, 1, ShapeUtil::MakeShape(PRED, {}), "pred");
  Conditional(pred, input, true_computation, input, false_computation);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = MakeElementwiseLhs();
  Literal pred_literal = LiteralUtil::CreateR0<bool>(true);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> pred_data,
                      client->TransferToServer(pred_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), pred_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalWhileAccumulator() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape index_shape = ShapeUtil::MakeShape(S32, {});
  Shape acc_shape = ShapeUtil::MakeShape(F32, {});
  Shape input_shape = ShapeUtil::MakeShape(F32, {16});
  Shape state_shape = ShapeUtil::MakeTupleShape(
      {index_shape, acc_shape, input_shape});

  XlaBuilder cond_builder("metal_while_cond");
  XlaOp cond_state = Parameter(&cond_builder, 0, state_shape, "state");
  Lt(GetTupleElement(cond_state, 0), ConstantR0<int32_t>(&cond_builder, 4));
  TF_ASSIGN_OR_RETURN(XlaComputation condition, cond_builder.Build());

  XlaBuilder body_builder("metal_while_body");
  XlaOp body_state = Parameter(&body_builder, 0, state_shape, "state");
  XlaOp i = GetTupleElement(body_state, 0);
  XlaOp acc = GetTupleElement(body_state, 1);
  XlaOp input = GetTupleElement(body_state, 2);
  XlaOp next_i = Add(i, ConstantR0<int32_t>(&body_builder, 1));
  XlaOp slice = DynamicSlice(input, {i}, {1});
  XlaOp value = Reshape(slice, {});
  Tuple(&body_builder, {next_i, Add(acc, value), input});
  TF_ASSIGN_OR_RETURN(XlaComputation body, body_builder.Build());

  XlaBuilder builder("metal_while_accumulator");
  XlaOp entry_input = Parameter(&builder, 0, input_shape, "input");
  XlaOp init = Tuple(&builder, {ConstantR0<int32_t>(&builder, 0),
                               ConstantR0<float>(&builder, 0.0f),
                               entry_input});
  XlaOp result = While(condition, body, init);
  GetTupleElement(result, 1);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  std::vector<float> values(16);
  for (int64_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i) * 0.25f - 2.0f;
  }
  Literal input_literal = LiteralUtil::CreateR1<float>(values);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalWhileVectorAccumulator() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  Shape index_shape = ShapeUtil::MakeShape(S32, {});
  Shape vector_shape = ShapeUtil::MakeShape(F32, {3});
  Shape state_shape =
      ShapeUtil::MakeTupleShape({index_shape, vector_shape, vector_shape});

  XlaBuilder cond_builder("metal_while_vector_cond");
  XlaOp cond_state = Parameter(&cond_builder, 0, state_shape, "state");
  Lt(GetTupleElement(cond_state, 0), ConstantR0<int32_t>(&cond_builder, 3));
  TF_ASSIGN_OR_RETURN(XlaComputation condition, cond_builder.Build());

  XlaBuilder body_builder("metal_while_vector_body");
  XlaOp body_state = Parameter(&body_builder, 0, state_shape, "state");
  XlaOp i = GetTupleElement(body_state, 0);
  XlaOp acc = GetTupleElement(body_state, 1);
  XlaOp input = GetTupleElement(body_state, 2);
  XlaOp next_i = Add(i, ConstantR0<int32_t>(&body_builder, 1));
  Tuple(&body_builder, {next_i, Add(acc, input), input});
  TF_ASSIGN_OR_RETURN(XlaComputation while_body, body_builder.Build());

  XlaBuilder builder("metal_while_vector_accumulator");
  XlaOp entry_input = Parameter(&builder, 0, vector_shape, "input");
  XlaOp zero_vector =
      Broadcast(ConstantR0<float>(&builder, 0.0f), {3});
  XlaOp init =
      Tuple(&builder, {ConstantR0<int32_t>(&builder, 0), zero_vector,
                       entry_input});
  XlaOp result = While(condition, while_body, init);
  GetTupleElement(result, 1);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
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

absl::StatusOr<Literal> ExecuteMetalIotaRank2(int64_t iota_dimension) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_iota_rank2");
  Shape shape = ShapeUtil::MakeShape(F32, {3, 4});
  Iota(&builder, shape, iota_dimension);
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

absl::StatusOr<Literal> ExecuteMetalBitcastConvertU16ToF16() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_bitcast_convert_u16_to_f16");
  Shape input_shape = ShapeUtil::MakeShape(U16, {4});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  BitcastConvertType(input, F16);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal =
      LiteralUtil::CreateR1<uint16_t>({0x0000, 0x3c00, 0xc000, 0x7bff});
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalBitcastConvertS4ToF16() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_bitcast_convert_s4_to_f16");
  Shape input_shape = ShapeUtil::MakeShape(S4, {4});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  BitcastConvertType(input, F16);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal =
      LiteralUtil::CreateR1<s4>({s4{1}, s4{-2}, s4{-3}, s4{4}});
  input_literal.mutable_shape_do_not_use()
      ->mutable_layout()
      ->set_element_size_in_bits(4);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalBitcastConvertF32ToU4() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_bitcast_convert_f32_to_u4");
  Shape input_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  BitcastConvertType(input, U4);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR0<float>(1.0f);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalBitcastConvertU32ToS8() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_bitcast_convert_u32_to_s8");
  Shape input_shape = ShapeUtil::MakeShape(U32, {});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  BitcastConvertType(input, S8);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR0<uint32_t>(0x1234fedc);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                      client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  return client->ExecuteAndTransfer(computation, arguments);
}

absl::StatusOr<Literal> ExecuteMetalBitcastConvertF32ToBF16() {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetMetalClient());

  XlaBuilder builder("metal_bitcast_convert_f32_to_bf16");
  Shape input_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  BitcastConvertType(input, BF16);
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR0<float>(1.0f);
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

void ExpectScalarPred(const Literal& actual, bool expected) {
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(PRED, {})));
  EXPECT_EQ(actual.Get<bool>({}), expected);
}

uint16_t HalfBits(half value) {
  uint16_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint16_t Bfloat16Bits(bfloat16 value) {
  uint16_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void ExpectScalarS32(const Literal& actual, int32_t expected) {
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(S32, {})));
  EXPECT_EQ(actual.Get<int32_t>({}), expected);
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

TEST(MetalGpuExecutableTest, SmallDotUsesScalarAirFallback) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_small_dot");
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {8});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {8});
  XlaOp lhs = Parameter(&builder, 0, lhs_shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, rhs_shape, "rhs");
  Dot(lhs, rhs);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Literal lhs_literal = LiteralUtil::CreateR1<float>(
      {-1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f});
  Literal rhs_literal = LiteralUtil::CreateR1<float>(
      {0.5f, -1.0f, 1.5f, -2.0f, 2.5f, -3.0f, 3.5f, -4.0f});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> lhs_data,
                          client->TransferToServer(lhs_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> rhs_data,
                          client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  float expected = 0.0f;
  for (int64_t i = 0; i < 8; ++i) {
    expected += lhs_literal.Get<float>({i}) * rhs_literal.Get<float>({i});
  }
  ExpectScalarNear(actual, expected, 1.0e-5f);
}

TEST(MetalGpuExecutableTest, SmallMatmulUsesScalarAirFallback) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  constexpr int64_t kSmallM = 4;
  constexpr int64_t kSmallK = 6;
  constexpr int64_t kSmallN = 3;
  XlaBuilder builder("metal_small_matmul");
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {kSmallM, kSmallK});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {kSmallK, kSmallN});
  XlaOp lhs = Parameter(&builder, 0, lhs_shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, rhs_shape, "rhs");
  XlaOp dot = Dot(lhs, rhs);
  Max(dot, Broadcast(ConstantR0<float>(&builder, 0.0f), {kSmallM, kSmallN}));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Array2D<float> lhs_values(kSmallM, kSmallK);
  for (int64_t row = 0; row < kSmallM; ++row) {
    for (int64_t col = 0; col < kSmallK; ++col) {
      lhs_values(row, col) =
          static_cast<float>(((row * kSmallK + col) % 11) - 5) / 7.0f;
    }
  }
  Array2D<float> rhs_values(kSmallK, kSmallN);
  for (int64_t row = 0; row < kSmallK; ++row) {
    for (int64_t col = 0; col < kSmallN; ++col) {
      rhs_values(row, col) =
          static_cast<float>(((row * kSmallN + col) % 7) - 3) / 9.0f;
    }
  }
  Literal lhs_literal = LiteralUtil::CreateR2FromArray2D(lhs_values);
  Literal rhs_literal = LiteralUtil::CreateR2FromArray2D(rhs_values);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> lhs_data,
                          client->TransferToServer(lhs_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> rhs_data,
                          client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F32, {kSmallM, kSmallN})));
  for (int64_t row = 0; row < kSmallM; ++row) {
    for (int64_t col = 0; col < kSmallN; ++col) {
      float expected = 0.0f;
      for (int64_t kk = 0; kk < kSmallK; ++kk) {
        expected += lhs_values(row, kk) * rhs_values(kk, col);
      }
      expected = std::max(expected, 0.0f);
      EXPECT_NEAR(actual.Get<float>({row, col}), expected, 1.0e-5f)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, BatchedDotUsesScalarAirFallback) {
  auto result = ExecuteMetalBatchedDot();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {2, 2, 4})));
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t row = 0; row < 2; ++row) {
      for (int64_t col = 0; col < 4; ++col) {
        float expected = 0.0f;
        for (int64_t kk = 0; kk < 3; ++kk) {
          const float lhs =
              static_cast<float>(b * 6 + row * 3 + kk) * 0.1f;
          const float rhs =
              static_cast<float>(b * 12 + kk * 4 + col) / 7.0f;
          expected += lhs * rhs;
        }
        EXPECT_NEAR(actual.Get<float>({b, row, col}), expected, 1.0e-5f)
            << "at (" << b << ", " << row << ", " << col << ")";
      }
    }
  }
}

TEST(MetalGpuExecutableTest, BatchedDotBf16UsesScalarAirFallback) {
  auto result = ExecuteMetalBatchedDotBf16();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(BF16, {2, 2, 4})));
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t row = 0; row < 2; ++row) {
      for (int64_t col = 0; col < 4; ++col) {
        float expected = 0.0f;
        for (int64_t kk = 0; kk < 3; ++kk) {
          const float lhs = static_cast<float>(bfloat16(
              static_cast<float>(b * 6 + row * 3 + kk) * 0.125f));
          const float rhs = static_cast<float>(bfloat16(
              static_cast<float>(b * 12 + kk * 4 + col) * 0.0625f));
          expected += lhs * rhs;
        }
        EXPECT_EQ(actual.Get<bfloat16>({b, row, col}), bfloat16(expected))
            << "at (" << b << ", " << row << ", " << col << ")";
      }
    }
  }
}

TEST(MetalGpuExecutableTest, BatchedDotF16Rank4UsesScalarAirFallback) {
  auto result = ExecuteMetalBatchedDotF16Rank4();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F16, {1, 2, 2, 1})));
  for (int64_t b1 = 0; b1 < 2; ++b1) {
    for (int64_t row = 0; row < 2; ++row) {
      float expected = 0.0f;
      for (int64_t kk = 0; kk < 3; ++kk) {
        const float lhs = static_cast<float>(
            half(static_cast<float>(b1 * 6 + row * 3 + kk) * 0.125f));
        const float rhs = static_cast<float>(
            half(static_cast<float>(b1 * 3 + kk) * 0.25f - 0.5f));
        expected += lhs * rhs;
      }
      EXPECT_EQ(actual.Get<half>({0, b1, row, 0}), half(expected))
          << "at (0, " << b1 << ", " << row << ", 0)";
    }
  }
}

TEST(MetalGpuExecutableTest, Conv1DUsesScalarAirFallback) {
  auto result = ExecuteMetalConv1D();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {1, 3, 1})));
  std::vector<float> expected = {0.0f, 2.0f, 4.0f};
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<float>({0, i, 0}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, Conv0DIsDotUsesScalarAirFallback) {
  auto result = ExecuteMetalConv0DIsDot();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {10, 7})));
  for (int64_t row = 0; row < 10; ++row) {
    for (int64_t col = 0; col < 7; ++col) {
      float expected = 0.0f;
      for (int64_t k = 0; k < 5; ++k) {
        const float lhs = static_cast<float>((row * 5 + k) % 9 - 4);
        const float rhs = static_cast<float>((k * 7 + col) % 11 - 5) * 0.25f;
        expected += lhs * rhs;
      }
      EXPECT_EQ(actual.Get<float>({row, col}), expected)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, Conv2DUsesScalarAirFallback) {
  auto result = ExecuteMetalConv2D();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F32, {1, 3, 3, 1})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      const float expected =
          (static_cast<float>(row * 4 + col) +
           static_cast<float>(row * 4 + col + 1) +
           static_cast<float>((row + 1) * 4 + col) +
           static_cast<float>((row + 1) * 4 + col + 1)) /
          4.0f;
      EXPECT_EQ(actual.Get<float>({0, row, col, 0}), expected)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, Conv2DNchwF16SameLowerUsesScalarAirFallback) {
  auto result = ExecuteMetalConv2DNchwF16SameLower();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(F16, {2, 3, 5, 10})));

  auto input_value = [](int64_t b, int64_t c, int64_t h, int64_t w) {
    return static_cast<float>(
        half(static_cast<float>((b * 180 + c * 90 + h * 10 + w) % 17 - 8) /
             8.0f));
  };
  auto kernel_value = [](int64_t o, int64_t c, int64_t h, int64_t w) {
    return static_cast<float>(
        half(static_cast<float>((o * 40 + c * 20 + h * 5 + w) % 11 - 5) /
             16.0f));
  };

  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t o = 0; o < 3; ++o) {
      for (int64_t oh = 0; oh < 5; ++oh) {
        for (int64_t ow = 0; ow < 10; ++ow) {
          float expected = 0.0f;
          for (int64_t kh = 0; kh < 4; ++kh) {
            const int64_t ih = oh * 2 + kh - 2;
            if (ih < 0 || ih >= 9) {
              continue;
            }
            for (int64_t kw = 0; kw < 5; ++kw) {
              const int64_t iw = ow + kw - 2;
              if (iw < 0 || iw >= 10) {
                continue;
              }
              for (int64_t c = 0; c < 2; ++c) {
                expected += input_value(b, c, ih, iw) *
                            kernel_value(o, c, kh, kw);
              }
            }
          }
          EXPECT_NEAR(static_cast<float>(actual.Get<half>({b, o, oh, ow})),
                      static_cast<float>(half(expected)), 1.0e-3f)
              << "at (" << b << ", " << o << ", " << oh << ", " << ow << ")";
        }
      }
    }
  }
}

TEST(MetalGpuExecutableTest, RfftUsesScalarAirFallback) {
  auto result = ExecuteMetalRfft();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(C64, {5})));
  const double pi = std::acos(-1.0);
  for (int64_t k = 0; k < 5; ++k) {
    float expected_real = 0.0f;
    float expected_imag = 0.0f;
    for (int64_t n = 0; n < 8; ++n) {
      const double angle = -2.0 * pi * static_cast<double>(k) *
                           static_cast<double>(n) / 8.0;
      expected_real += static_cast<float>(n) *
                       static_cast<float>(std::cos(angle));
      expected_imag += static_cast<float>(n) *
                       static_cast<float>(std::sin(angle));
    }
    const complex64 value = actual.Get<complex64>({k});
    EXPECT_NEAR(value.real(), expected_real, 1.0e-4f) << "real at " << k;
    EXPECT_NEAR(value.imag(), expected_imag, 1.0e-4f) << "imag at " << k;
  }
}

TEST(MetalGpuExecutableTest, ComplexTranspose) {
  auto result = ExecuteMetalComplexTranspose();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(C64, {3, 2})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 2; ++col) {
      const complex64 expected(static_cast<float>(col * 3 + row),
                               static_cast<float>(col - row));
      const complex64 value = actual.Get<complex64>({row, col});
      EXPECT_EQ(value.real(), expected.real()) << "real at (" << row << ", "
                                               << col << ")";
      EXPECT_EQ(value.imag(), expected.imag()) << "imag at (" << row << ", "
                                               << col << ")";
    }
  }
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

TEST(MetalGpuExecutableTest, ElementwisePadRank2) {
  auto result = ExecuteMetalPadRank2();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {4, 6})));
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 6; ++col) {
      const bool in_input = row >= 1 && row < 3 && col >= 2 && col < 5;
      const float expected =
          in_input ? static_cast<float>((row - 1) * 3 + (col - 2)) : 5.0f;
      EXPECT_EQ(actual.Get<float>({row, col}), expected)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseGatherSelect) {
  auto result = ExecuteMetalGatherSelect();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectMatchesGatherReference(actual, MakeGatherInput());
}

TEST(MetalGpuExecutableTest, ElementwiseGatherRank2Rows) {
  auto result = ExecuteMetalGatherRank2(/*gather_rows=*/true);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {2, 5})));
  for (int64_t row = 0; row < 2; ++row) {
    const int64_t source_row = row == 0 ? 0 : 2;
    for (int64_t col = 0; col < 5; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}),
                static_cast<float>(source_row * 5 + col))
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseGatherRank2Cols) {
  auto result = ExecuteMetalGatherRank2(/*gather_rows=*/false);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {4, 2})));
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 2; ++col) {
      const int64_t source_col = col == 0 ? 1 : 3;
      EXPECT_EQ(actual.Get<float>({row, col}),
                static_cast<float>(row * 5 + source_col))
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseSortRank1) {
  auto result = ExecuteMetalSortRank1();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<float> expected = {-2.0f, -1.0f, 0.25f, 0.25f,
                                 1.5f,  2.0f,  3.0f,  4.0f};
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {8})));
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<float>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseSortU16Rank1) {
  auto result = ExecuteMetalSortU16Rank1();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<uint16_t> expected = {0, 1, 1, 3, 9, 17, 42, 65535};
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(U16, {8})));
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<uint16_t>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseSortRank2) {
  auto result = ExecuteMetalSortRank2();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  float expected[2][4] = {{1.0f, 1.0f, 2.0f, 3.0f},
                          {-1.0f, 0.0f, 4.0f, 5.0f}};
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(F32, {2, 4})));
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}), expected[row][col])
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseTopKValuesRank1) {
  auto result = ExecuteMetalTopKValuesRank1();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<float> expected = {4.0f, 3.0f, 2.0f};
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3})));
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<float>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseTopKIndicesRank1) {
  auto result = ExecuteMetalTopKIndicesRank1();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<int32_t> expected = {3, 6, 7};
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(S32, {3})));
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<int32_t>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseTopKValuesRank2) {
  auto result = ExecuteMetalTopKRank2(/*indices=*/false);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  float expected[2][2] = {{4.0f, 4.0f}, {6.0f, 5.0f}};
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(F32, {2, 2})));
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 2; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}), expected[row][col])
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseTopKIndicesRank2) {
  auto result = ExecuteMetalTopKRank2(/*indices=*/true);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  int32_t expected[2][2] = {{1, 3}, {0, 2}};
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(S32, {2, 2})));
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 2; ++col) {
      EXPECT_EQ(actual.Get<int32_t>({row, col}), expected[row][col])
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseTopKTupleRank1) {
  auto result = ExecuteMetalTopKTupleRank1();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Shape expected_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3}), ShapeUtil::MakeShape(S32, {3})});
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(), expected_shape));

  LiteralSlice values(actual, {0});
  LiteralSlice indices(actual, {1});
  std::vector<float> expected_values = {4.0f, 3.0f, 2.0f};
  std::vector<int32_t> expected_indices = {3, 6, 7};
  for (int64_t i = 0; i < expected_values.size(); ++i) {
    EXPECT_EQ(values.Get<float>({i}), expected_values[i]) << "value at " << i;
    EXPECT_EQ(indices.Get<int32_t>({i}), expected_indices[i])
        << "index at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseTopKTupleS32Rank1) {
  auto result = ExecuteMetalTopKTupleS32Rank1();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Shape expected_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {3}), ShapeUtil::MakeShape(S32, {3})});
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(), expected_shape));

  LiteralSlice values(actual, {0});
  LiteralSlice indices(actual, {1});
  std::vector<int32_t> expected_values = {9, 7, 7};
  std::vector<int32_t> expected_indices = {5, 2, 3};
  for (int64_t i = 0; i < expected_values.size(); ++i) {
    EXPECT_EQ(values.Get<int32_t>({i}), expected_values[i])
        << "value at " << i;
    EXPECT_EQ(indices.Get<int32_t>({i}), expected_indices[i])
        << "index at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseTopKTupleRank2) {
  auto result = ExecuteMetalTopKTupleRank2();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Shape expected_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 2}), ShapeUtil::MakeShape(S32, {2, 2})});
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(), expected_shape));

  LiteralSlice values(actual, {0});
  LiteralSlice indices(actual, {1});
  float expected_values[2][2] = {{4.0f, 4.0f}, {6.0f, 5.0f}};
  int32_t expected_indices[2][2] = {{1, 3}, {0, 2}};
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 2; ++col) {
      EXPECT_EQ(values.Get<float>({row, col}), expected_values[row][col])
          << "value at (" << row << ", " << col << ")";
      EXPECT_EQ(indices.Get<int32_t>({row, col}), expected_indices[row][col])
          << "index at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseArgmaxRank1) {
  auto result = ExecuteMetalArgmaxRank1();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectScalarS32(actual, 1);
}

TEST(MetalGpuExecutableTest, ElementwiseSortDescendingExpression) {
  auto result = ExecuteMetalSortDescendingExpression();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<float> expected = {4.0f,  3.0f,  2.0f,  1.5f,
                                 0.25f, 0.25f, -1.0f, -2.0f};
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {8})));
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<float>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseClamp) {
  auto result = ExecuteMetalClamp();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(actual, [&](int64_t i) {
    return std::min(std::max(input.Get<float>({i}), -0.5f), 0.75f);
  });
}

TEST(MetalGpuExecutableTest, ElementwiseClampS16ScalarBounds) {
  auto result = ExecuteMetalClampS16ScalarBounds();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(S16, {2, 3})));
  int16_t value = -4;
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      const int16_t expected =
          std::min<int16_t>(std::max<int16_t>(value, -2), 1);
      EXPECT_EQ(actual.Get<int16_t>({row, col}), expected)
          << "at [" << row << ", " << col << "]";
      ++value;
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseReverseRank2) {
  auto result = ExecuteMetalReverseRank2();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3, 4})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}),
                static_cast<float>((2 - row) * 4 + (3 - col)))
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseDynamicSliceDynamicStart) {
  auto result = ExecuteMetalDynamicSliceDynamicStart();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {2, 2})));
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 2; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}),
                static_cast<float>((row + 1) * 5 + (col + 2)))
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseS32DynamicSliceDynamicStart) {
  auto result = ExecuteMetalS32DynamicSliceDynamicStart();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(S32, {2, 2})));
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t col = 0; col < 2; ++col) {
      EXPECT_EQ(actual.Get<int32_t>({row, col}),
                static_cast<int32_t>((row + 1) * 5 + (col + 2)))
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseDynamicSliceRank3) {
  auto result = ExecuteMetalDynamicSliceRank3();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(F32, {1, 2, 3})));
  for (int64_t i = 0; i < 1; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      for (int64_t k = 0; k < 3; ++k) {
        EXPECT_EQ(actual.Get<float>({i, j, k}),
                  static_cast<float>((i + 1) * 12 + (j + 1) * 4 + (k + 1)))
            << "at (" << i << ", " << j << ", " << k << ")";
      }
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseScatterSet) {
  auto result = ExecuteMetalScatter(/*add=*/false);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<float> expected = {0.0f, -1.0f, 2.0f, -2.0f,
                                 4.0f, 5.0f,  -3.0f, 7.0f};
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {8})));
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<float>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseScatterAdd) {
  auto result = ExecuteMetalScatter(/*add=*/true);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<float> expected = {0.0f, 0.0f, 2.0f, 1.0f,
                                 4.0f, 5.0f, 3.0f, 7.0f};
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {8})));
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<float>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseScatterRank2Set) {
  auto result = ExecuteMetalScatterRank2(/*add=*/false);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<float> expected = {0.0f, -5.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f,  4.0f, 7.0f,
                                 8.0f, 9.0f,  10.0f, 7.0f};
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(F32, {3, 4})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      const int64_t index = row * 4 + col;
      EXPECT_EQ(actual.Get<float>({row, col}), expected[index])
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseScatterRank2Add) {
  auto result = ExecuteMetalScatterRank2(/*add=*/true);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<float> expected = {0.0f, -4.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f,  10.0f, 7.0f,
                                 8.0f, 9.0f,  10.0f, 18.0f};
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(F32, {3, 4})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      const int64_t index = row * 4 + col;
      EXPECT_EQ(actual.Get<float>({row, col}), expected[index])
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseReduceWindowSum) {
  auto result = ExecuteMetalReduceWindow(HloOpcode::kAdd, 0.0f);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {14})));
  for (int64_t i = 0; i < 14; ++i) {
    const float x0 = static_cast<float>(i) * 0.25f - 2.0f;
    const float x1 = static_cast<float>(i + 1) * 0.25f - 2.0f;
    const float x2 = static_cast<float>(i + 2) * 0.25f - 2.0f;
    EXPECT_EQ(actual.Get<float>({i}), x0 + x1 + x2) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseReduceWindowMax) {
  auto result = ExecuteMetalReduceWindow(
      HloOpcode::kMaximum, -std::numeric_limits<float>::infinity());
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {14})));
  for (int64_t i = 0; i < 14; ++i) {
    EXPECT_EQ(actual.Get<float>({i}),
              static_cast<float>(i + 2) * 0.25f - 2.0f)
        << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, ElementwiseReduceWindowRank2Sum) {
  auto result = ExecuteMetalReduceWindowRank2(HloOpcode::kAdd, 0.0f);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3, 4})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      const float expected =
          static_cast<float>(row * 5 + col) +
          static_cast<float>(row * 5 + col + 1) +
          static_cast<float>((row + 1) * 5 + col) +
          static_cast<float>((row + 1) * 5 + col + 1);
      EXPECT_EQ(actual.Get<float>({row, col}), expected)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseReduceWindowRank2Max) {
  auto result = ExecuteMetalReduceWindowRank2(
      HloOpcode::kMaximum, -std::numeric_limits<float>::infinity());
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3, 4})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}),
                static_cast<float>((row + 1) * 5 + col + 1))
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseConditional) {
  auto result = ExecuteMetalConditional();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  Literal input = MakeElementwiseLhs();
  ExpectMatchesElementwiseReference(
      actual, [&](int64_t i) { return input.Get<float>({i}) + 1.0f; });
}

TEST(MetalGpuExecutableTest, ElementwiseWhileAccumulator) {
  auto result = ExecuteMetalWhileAccumulator();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  float expected = 0.0f;
  for (int64_t i = 0; i < 4; ++i) {
    expected += static_cast<float>(i) * 0.25f - 2.0f;
  }
  ExpectScalarNear(actual, expected, 1.0e-5f);
}

TEST(MetalGpuExecutableTest, ElementwiseWhileVectorAccumulator) {
  auto result = ExecuteMetalWhileVectorAccumulator();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  std::vector<float> expected = {3.0f, 6.0f, 9.0f};
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3})));
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<float>({i}), expected[i]) << "at " << i;
  }
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

TEST(MetalGpuExecutableTest, ElementwiseRank2Slice) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_elementwise_rank2_slice");
  Shape input_shape = ShapeUtil::MakeShape(F32, {5, 6});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp sliced = Slice(input, {1, 2}, {4, 5}, {1, 1});
  Add(sliced, Broadcast(ConstantR0<float>(&builder, 1.0f), {3, 3}));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Array2D<float> values(5, 6);
  for (int64_t row = 0; row < 5; ++row) {
    for (int64_t col = 0; col < 6; ++col) {
      values(row, col) = static_cast<float>(((row * 6 + col) % 13) - 6) / 4.0f;
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3, 3})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}), values(row + 1, col + 2) + 1.0f)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseDynamicUpdateSlice) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_elementwise_dynamic_update_slice");
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  Shape update_shape = ShapeUtil::MakeShape(F32, {2, 2});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp update = Parameter(&builder, 1, update_shape, "update");
  DynamicUpdateSlice(input, update,
                     {ConstantR0<int32_t>(&builder, 1),
                      ConstantR0<int32_t>(&builder, 2)});
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Array2D<float> input_values(4, 5);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      input_values(row, col) = static_cast<float>(row * 5 + col);
    }
  }
  Array2D<float> update_values(2, 2);
  update_values(0, 0) = -1.0f;
  update_values(0, 1) = -2.0f;
  update_values(1, 0) = -3.0f;
  update_values(1, 1) = -4.0f;
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_values);
  Literal update_literal = LiteralUtil::CreateR2FromArray2D(update_values);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> update_data,
                          client->TransferToServer(update_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), update_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(), input_shape));
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      float expected = input_values(row, col);
      if (row >= 1 && row < 3 && col >= 2 && col < 4) {
        expected = update_values(row - 1, col - 2);
      }
      EXPECT_EQ(actual.Get<float>({row, col}), expected)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseDynamicUpdateSliceDynamicStart) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_elementwise_dynamic_update_slice_dynamic_start");
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  Shape update_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp update = Parameter(&builder, 1, update_shape, "update");
  XlaOp row_start = Parameter(&builder, 2, scalar_shape, "row_start");
  XlaOp col_start = Parameter(&builder, 3, scalar_shape, "col_start");
  DynamicUpdateSlice(input, update, {row_start, col_start});
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Array2D<float> input_values(4, 5);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      input_values(row, col) = static_cast<float>(row * 5 + col);
    }
  }
  Array2D<float> update_values(2, 2);
  update_values(0, 0) = -1.0f;
  update_values(0, 1) = -2.0f;
  update_values(1, 0) = -3.0f;
  update_values(1, 1) = -4.0f;
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_values);
  Literal update_literal = LiteralUtil::CreateR2FromArray2D(update_values);
  Literal row_start_literal = LiteralUtil::CreateR0<int32_t>(1);
  Literal col_start_literal = LiteralUtil::CreateR0<int32_t>(2);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> update_data,
                          client->TransferToServer(update_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> row_start_data,
                          client->TransferToServer(row_start_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> col_start_data,
                          client->TransferToServer(col_start_literal));
  std::vector<GlobalData*> arguments = {
      input_data.get(), update_data.get(), row_start_data.get(),
      col_start_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(), input_shape));
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      float expected = input_values(row, col);
      if (row >= 1 && row < 3 && col >= 2 && col < 4) {
        expected = update_values(row - 1, col - 2);
      }
      EXPECT_EQ(actual.Get<float>({row, col}), expected)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseDynamicUpdateSliceRank3) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_elementwise_dynamic_update_slice_rank3");
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Shape update_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp update = Parameter(&builder, 1, update_shape, "update");
  XlaOp dim0 = Parameter(&builder, 2, scalar_shape, "dim0");
  XlaOp dim1 = Parameter(&builder, 3, scalar_shape, "dim1");
  XlaOp dim2 = Parameter(&builder, 4, scalar_shape, "dim2");
  DynamicUpdateSlice(input, update, {dim0, dim1, dim2});
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Array3D<float> input_values(2, 3, 4);
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        input_values(i, j, k) = static_cast<float>(i * 12 + j * 4 + k);
      }
    }
  }
  Array3D<float> update_values(1, 2, 2);
  for (int64_t j = 0; j < 2; ++j) {
    for (int64_t k = 0; k < 2; ++k) {
      update_values(0, j, k) = -1.0f - static_cast<float>(j * 2 + k);
    }
  }
  Literal input_literal = LiteralUtil::CreateR3FromArray3D(input_values);
  Literal update_literal = LiteralUtil::CreateR3FromArray3D(update_values);
  Literal dim0_literal = LiteralUtil::CreateR0<int32_t>(1);
  Literal dim1_literal = LiteralUtil::CreateR0<int32_t>(1);
  Literal dim2_literal = LiteralUtil::CreateR0<int32_t>(1);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> update_data,
                          client->TransferToServer(update_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> dim0_data,
                          client->TransferToServer(dim0_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> dim1_data,
                          client->TransferToServer(dim1_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> dim2_data,
                          client->TransferToServer(dim2_literal));
  std::vector<GlobalData*> arguments = {input_data.get(), update_data.get(),
                                        dim0_data.get(), dim1_data.get(),
                                        dim2_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(), input_shape));
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        float expected = input_values(i, j, k);
        if (i == 1 && j >= 1 && j < 3 && k >= 1 && k < 3) {
          expected = update_values(0, j - 1, k - 1);
        }
        EXPECT_EQ(actual.Get<float>({i, j, k}), expected)
            << "at (" << i << ", " << j << ", " << k << ")";
      }
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseReshapeParameter) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_elementwise_reshape_parameter");
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  Reshape(input, {20});
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Array2D<float> values(4, 5);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      values(row, col) = static_cast<float>(row * 5 + col) * 0.25f;
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(F32, {20})));
  for (int64_t i = 0; i < 20; ++i) {
    EXPECT_EQ(actual.Get<float>({i}), static_cast<float>(i) * 0.25f);
  }
}

TEST(MetalGpuExecutableTest, ElementwiseVectorBroadcast) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_elementwise_vector_broadcast");
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp offsets = Iota(&builder, F32, 5);
  Add(input, BroadcastInDim(offsets, {4, 5}, {1}));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Array2D<float> values(4, 5);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      values(row, col) = static_cast<float>(row * 5 + col) * 0.25f;
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(),
                                    ShapeUtil::MakeShape(F32, {4, 5})));
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}), values(row, col) + col)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseRank2ToRank4Broadcast) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_elementwise_rank2_to_rank4_broadcast");
  Shape input_shape = ShapeUtil::MakeShape(S32, {2, 3});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  BroadcastInDim(input, {1, 2, 2, 3}, {2, 3});
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR2<int32_t>({{1, 2, 3},
                                                          {4, 5, 6}});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(ShapeUtil::Compatible(
      actual.shape(), ShapeUtil::MakeShape(S32, {1, 2, 2, 3})));
  for (int64_t b0 = 0; b0 < 1; ++b0) {
    for (int64_t b1 = 0; b1 < 2; ++b1) {
      for (int64_t row = 0; row < 2; ++row) {
        for (int64_t col = 0; col < 3; ++col) {
          EXPECT_EQ(actual.Get<int32_t>({b0, b1, row, col}),
                    input_literal.Get<int32_t>({row, col}))
              << "at (" << b0 << ", " << b1 << ", " << row << ", " << col
              << ")";
        }
      }
    }
  }
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

TEST(MetalGpuExecutableTest, ElementwiseConcatenateTwoParameters) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder builder("metal_elementwise_concatenate_two_parameters");
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  XlaOp lhs = Parameter(&builder, 0, shape, "lhs");
  XlaOp rhs = Parameter(&builder, 1, shape, "rhs");
  ConcatInDim(&builder, {lhs, rhs}, 0);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Literal lhs_literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal rhs_literal =
      LiteralUtil::CreateR1<float>({-1.0f, -2.0f, -3.0f, -4.0f});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> lhs_data,
                          client->TransferToServer(lhs_literal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> rhs_data,
                          client->TransferToServer(rhs_literal));
  std::vector<GlobalData*> arguments = {lhs_data.get(), rhs_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {8})));
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual.Get<float>({i}), lhs_literal.Get<float>({i}));
    EXPECT_EQ(actual.Get<float>({i + 4}), rhs_literal.Get<float>({i}));
  }
}

TEST(MetalGpuExecutableTest, ElementwiseConcatenateRank2Dim0) {
  auto result = ExecuteMetalConcatenateRank2(/*dim=*/0);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3, 3})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      float expected = row < 2 ? static_cast<float>(row * 3 + col)
                               : 100.0f + static_cast<float>(col);
      EXPECT_EQ(actual.Get<float>({row, col}), expected)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseConcatenateRank2Dim1) {
  auto result = ExecuteMetalConcatenateRank2(/*dim=*/1);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3, 3})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 3; ++col) {
      float expected = col < 2 ? static_cast<float>(row * 2 + col)
                               : 100.0f + static_cast<float>(row);
      EXPECT_EQ(actual.Get<float>({row, col}), expected)
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseConcatenateRank3) {
  std::vector<int64_t> operand_sizes = {3, 1, 4};
  for (int64_t dim = 0; dim < 3; ++dim) {
    auto result = ExecuteMetalConcatenateRank3(dim);
    if (absl::IsFailedPrecondition(result.status())) {
      GTEST_SKIP() << result.status();
    }
    TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
    std::vector<int64_t> result_dimensions = {2, 3, 4};
    result_dimensions[dim] = 8;
    ASSERT_TRUE(ShapeUtil::Compatible(
        actual.shape(), ShapeUtil::MakeShape(S32, result_dimensions)));

    for (int64_t i = 0; i < result_dimensions[0]; ++i) {
      for (int64_t j = 0; j < result_dimensions[1]; ++j) {
        for (int64_t k = 0; k < result_dimensions[2]; ++k) {
          std::vector<int64_t> coords = {i, j, k};
          int64_t offset = 0;
          int64_t operand_index = 0;
          for (; operand_index < operand_sizes.size(); ++operand_index) {
            if (coords[dim] < offset + operand_sizes[operand_index]) {
              break;
            }
            offset += operand_sizes[operand_index];
          }
          std::vector<int64_t> local_coords = coords;
          local_coords[dim] -= offset;
          std::vector<int64_t> operand_dimensions = {2, 3, 4};
          operand_dimensions[dim] = operand_sizes[operand_index];
          int32_t expected = static_cast<int32_t>(
              operand_index * 1000 +
              (local_coords[0] * operand_dimensions[1] + local_coords[1]) *
                  operand_dimensions[2] +
              local_coords[2]);
          EXPECT_EQ(actual.Get<int32_t>({i, j, k}), expected)
              << "for dim " << dim << " at (" << i << ", " << j << ", " << k
              << ")";
        }
      }
    }
  }
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

TEST(MetalGpuExecutableTest, ElementwiseIotaRank2Dim0) {
  auto result = ExecuteMetalIotaRank2(/*iota_dimension=*/0);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3, 4})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}), static_cast<float>(row))
          << "at (" << row << ", " << col << ")";
    }
  }
}

TEST(MetalGpuExecutableTest, ElementwiseIotaRank2Dim1) {
  auto result = ExecuteMetalIotaRank2(/*iota_dimension=*/1);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F32, {3, 4})));
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t col = 0; col < 4; ++col) {
      EXPECT_EQ(actual.Get<float>({row, col}), static_cast<float>(col))
          << "at (" << row << ", " << col << ")";
    }
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

TEST(MetalGpuExecutableTest, BitcastConvertU16ToF16) {
  auto result = ExecuteMetalBitcastConvertU16ToF16();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F16, {4})));
  std::vector<half> expected = {half(0.0f), half(1.0f), half(-2.0f),
                                std::numeric_limits<half>::max()};
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<half>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, BitcastConvertS4ToF16) {
  auto result = ExecuteMetalBitcastConvertS4ToF16();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(F16, {})));
  EXPECT_EQ(HalfBits(actual.Get<half>({})), 0x4de1);
}

TEST(MetalGpuExecutableTest, BitcastConvertF32ToU4) {
  auto result = ExecuteMetalBitcastConvertF32ToU4();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(U4, {8})));
  const std::vector<u4> expected = {u4{0}, u4{0}, u4{0},  u4{0},
                                    u4{0}, u4{8}, u4{15}, u4{3}};
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<u4>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, BitcastConvertU32ToS8) {
  auto result = ExecuteMetalBitcastConvertU32ToS8();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(S8, {4})));
  const std::vector<int8_t> expected = {
      static_cast<int8_t>(0xdc), static_cast<int8_t>(0xfe),
      static_cast<int8_t>(0x34), static_cast<int8_t>(0x12)};
  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual.Get<int8_t>({i}), expected[i]) << "at " << i;
  }
}

TEST(MetalGpuExecutableTest, BitcastConvertF32ToBF16) {
  auto result = ExecuteMetalBitcastConvertF32ToBF16();
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(BF16, {2})));
  EXPECT_EQ(Bfloat16Bits(actual.Get<bfloat16>({0})), 0x0000);
  EXPECT_EQ(Bfloat16Bits(actual.Get<bfloat16>({1})), 0x3f80);
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

TEST(MetalGpuExecutableTest, ReductionProductElementwiseInput) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder reducer_builder("metal_product_expr_reducer");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  Mul(lhs, rhs);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder("metal_product_expr_reduction");
  Shape input_shape = ShapeUtil::MakeShape(F32, {kElementCount});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp sliced = Slice(input, {0}, {8}, {1});
  XlaOp shifted = Add(sliced, Broadcast(ConstantR0<float>(&builder, 2.0f), {8}));
  Reduce(shifted, ConstantR0<float>(&builder, 1.0f), reducer, {0});
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
  float expected = 1.0f;
  for (int64_t i = 0; i < 8; ++i) {
    expected *= input_literal.Get<float>({i}) + 2.0f;
  }
  ExpectScalarNear(actual, expected);
}

TEST(MetalGpuExecutableTest, ReductionPredAllAny) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  auto build_reducer = [](HloOpcode opcode) -> absl::StatusOr<XlaComputation> {
    XlaBuilder reducer_builder("metal_pred_reducer");
    Shape scalar_shape = ShapeUtil::MakeShape(PRED, {});
    XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
    XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
    if (opcode == HloOpcode::kAnd) {
      And(lhs, rhs);
    } else {
      Or(lhs, rhs);
    }
    return reducer_builder.Build();
  };

  auto execute = [&](HloOpcode opcode,
                     bool init_value) -> absl::StatusOr<Literal> {
    TF_ASSIGN_OR_RETURN(XlaComputation reducer, build_reducer(opcode));
    XlaBuilder builder("metal_pred_reduction");
    Shape input_shape = ShapeUtil::MakeShape(PRED, {kElementCount});
    XlaOp input = Parameter(&builder, 0, input_shape, "input");
    Reduce(input, ConstantR0<bool>(&builder, init_value), reducer, {0});
    TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());
    Literal input_literal = MakeElementwisePred();
    TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                        client->TransferToServer(input_literal));
    std::vector<GlobalData*> arguments = {input_data.get()};
    return client->ExecuteAndTransfer(computation, arguments);
  };

  auto all_result = execute(HloOpcode::kAnd, true);
  if (absl::IsFailedPrecondition(all_result.status())) {
    GTEST_SKIP() << all_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal all_actual, std::move(all_result));
  ExpectScalarPred(all_actual, false);

  auto any_result = execute(HloOpcode::kOr, false);
  if (absl::IsFailedPrecondition(any_result.status())) {
    GTEST_SKIP() << any_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal any_actual, std::move(any_result));
  ExpectScalarPred(any_actual, true);
}

TEST(MetalGpuExecutableTest, ReductionF32Rank2ToRank1) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder reducer_builder("metal_rank2_sum_reducer");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  Add(lhs, rhs);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation reducer, reducer_builder.Build());

  Array2D<float> values(4, 5);
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      values(row, col) = static_cast<float>(((row * 5 + col) % 11) - 5) / 3.0f;
    }
  }
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(values);

  auto execute = [&](int64_t dimension) -> absl::StatusOr<Literal> {
    XlaBuilder builder("metal_rank2_sum");
    Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5});
    XlaOp input = Parameter(&builder, 0, input_shape, "input");
    Reduce(input, ConstantR0<float>(&builder, 0.0f), reducer, {dimension});
    TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());
    TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                        client->TransferToServer(input_literal));
    std::vector<GlobalData*> arguments = {input_data.get()};
    return client->ExecuteAndTransfer(computation, arguments);
  };

  auto axis0_result = execute(0);
  if (absl::IsFailedPrecondition(axis0_result.status())) {
    GTEST_SKIP() << axis0_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal axis0_actual, std::move(axis0_result));
  ASSERT_TRUE(ShapeUtil::Compatible(axis0_actual.shape(),
                                    ShapeUtil::MakeShape(F32, {5})));
  for (int64_t col = 0; col < 5; ++col) {
    float expected = 0.0f;
    for (int64_t row = 0; row < 4; ++row) {
      expected += values(row, col);
    }
    EXPECT_NEAR(axis0_actual.Get<float>({col}), expected, 1.0e-6f)
        << "at col " << col;
  }

  auto axis1_result = execute(1);
  if (absl::IsFailedPrecondition(axis1_result.status())) {
    GTEST_SKIP() << axis1_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal axis1_actual, std::move(axis1_result));
  ASSERT_TRUE(ShapeUtil::Compatible(axis1_actual.shape(),
                                    ShapeUtil::MakeShape(F32, {4})));
  for (int64_t row = 0; row < 4; ++row) {
    float expected = 0.0f;
    for (int64_t col = 0; col < 5; ++col) {
      expected += values(row, col);
    }
    EXPECT_NEAR(axis1_actual.Get<float>({row}), expected, 1.0e-6f)
        << "at row " << row;
  }
}

TEST(MetalGpuExecutableTest, ReductionF32Rank3ToScalar) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder reducer_builder("metal_rank3_sum_reducer");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  Add(lhs, rhs);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder("metal_rank3_sum_to_scalar");
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  Reduce(input, ConstantR0<float>(&builder, 0.0f), reducer, {0, 1, 2});
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Array3D<float> values(2, 3, 4);
  float expected = 0.0f;
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        values(i, j, k) = static_cast<float>(i * 12 + j * 4 + k) * 0.25f;
        expected += values(i, j, k);
      }
    }
  }
  Literal input_literal = LiteralUtil::CreateR3FromArray3D(values);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ExpectScalarNear(actual, expected);
}

TEST(MetalGpuExecutableTest, ReductionF32Rank3ToRank1) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder reducer_builder("metal_rank3_to_rank1_sum_reducer");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  Add(lhs, rhs);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation reducer, reducer_builder.Build());

  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Array3D<float> values(2, 3, 4);
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        values(i, j, k) = static_cast<float>(i * 12 + j * 4 + k) * 0.5f;
      }
    }
  }
  Literal input_literal = LiteralUtil::CreateR3FromArray3D(values);

  auto execute =
      [&](std::vector<int64_t> reduction_dims) -> absl::StatusOr<Literal> {
    XlaBuilder builder("metal_rank3_sum_to_rank1");
    XlaOp input = Parameter(&builder, 0, input_shape, "input");
    Reduce(input, ConstantR0<float>(&builder, 0.0f), reducer, reduction_dims);
    TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());
    TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                        client->TransferToServer(input_literal));
    std::vector<GlobalData*> arguments = {input_data.get()};
    return client->ExecuteAndTransfer(computation, arguments);
  };

  for (const std::vector<int64_t>& reduction_dims :
       {std::vector<int64_t>{1, 2}, std::vector<int64_t>{0, 2},
        std::vector<int64_t>{0, 1}}) {
    auto result = execute(reduction_dims);
    if (absl::IsFailedPrecondition(result.status())) {
      GTEST_SKIP() << result.status();
    }
    TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));

    std::vector<bool> reduced(3, false);
    for (int64_t dim : reduction_dims) {
      reduced[dim] = true;
    }
    int64_t kept_dim = 0;
    for (int64_t dim = 0; dim < 3; ++dim) {
      if (!reduced[dim]) {
        kept_dim = dim;
        break;
      }
    }
    const int64_t output_size = input_shape.dimensions(kept_dim);
    ASSERT_TRUE(ShapeUtil::Compatible(
        actual.shape(), ShapeUtil::MakeShape(F32, {output_size})));

    for (int64_t out = 0; out < output_size; ++out) {
      float expected = 0.0f;
      for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
          for (int64_t k = 0; k < 4; ++k) {
            if ((kept_dim == 0 && i == out) ||
                (kept_dim == 1 && j == out) ||
                (kept_dim == 2 && k == out)) {
              expected += values(i, j, k);
            }
          }
        }
      }
      EXPECT_NEAR(actual.Get<float>({out}), expected, 1.0e-6f)
          << "for reduction dims " << absl::StrJoin(reduction_dims, ",")
          << " at output index " << out;
    }
  }
}

TEST(MetalGpuExecutableTest, ReductionPredRank2AllAxis1) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder reducer_builder("metal_rank2_pred_reducer");
  Shape scalar_shape = ShapeUtil::MakeShape(PRED, {});
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  And(lhs, rhs);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder("metal_rank2_all");
  Shape input_shape = ShapeUtil::MakeShape(PRED, {4, 5});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  Reduce(input, ConstantR0<bool>(&builder, true), reducer, {1});
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Literal input_literal =
      Literal::CreateFromShape(ShapeUtil::MakeShape(PRED, {4, 5}));
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t col = 0; col < 5; ++col) {
      input_literal.Set<bool>({row, col}, row != 2 && col != 3);
    }
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  auto result = client->ExecuteAndTransfer(computation, arguments);
  if (absl::IsFailedPrecondition(result.status())) {
    GTEST_SKIP() << result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, std::move(result));
  ASSERT_TRUE(
      ShapeUtil::Compatible(actual.shape(), ShapeUtil::MakeShape(PRED, {4})));
  for (int64_t row = 0; row < 4; ++row) {
    bool expected = true;
    for (int64_t col = 0; col < 5; ++col) {
      expected = expected && input_literal.Get<bool>({row, col});
    }
    EXPECT_EQ(actual.Get<bool>({row}), expected) << "at row " << row;
  }
}

TEST(MetalGpuExecutableTest, ReductionS32SumMaximum) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  auto build_reducer = [](HloOpcode opcode) -> absl::StatusOr<XlaComputation> {
    XlaBuilder reducer_builder("metal_s32_reducer");
    Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
    XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
    XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
    if (opcode == HloOpcode::kAdd) {
      Add(lhs, rhs);
    } else {
      Max(lhs, rhs);
    }
    return reducer_builder.Build();
  };

  auto execute = [&](HloOpcode opcode,
                     int32_t init_value) -> absl::StatusOr<Literal> {
    TF_ASSIGN_OR_RETURN(XlaComputation reducer, build_reducer(opcode));
    XlaBuilder builder("metal_s32_reduction");
    Shape input_shape = ShapeUtil::MakeShape(S32, {kElementCount});
    XlaOp input = Parameter(&builder, 0, input_shape, "input");
    Reduce(input, ConstantR0<int32_t>(&builder, init_value), reducer, {0});
    TF_ASSIGN_OR_RETURN(XlaComputation computation, builder.Build());
    Literal input_literal = MakeElementwiseS32();
    TF_ASSIGN_OR_RETURN(std::unique_ptr<GlobalData> input_data,
                        client->TransferToServer(input_literal));
    std::vector<GlobalData*> arguments = {input_data.get()};
    return client->ExecuteAndTransfer(computation, arguments);
  };

  auto sum_result = execute(HloOpcode::kAdd, 0);
  if (absl::IsFailedPrecondition(sum_result.status())) {
    GTEST_SKIP() << sum_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal sum_actual, std::move(sum_result));
  Literal input = MakeElementwiseS32();
  int32_t expected_sum = 0;
  int32_t expected_max = std::numeric_limits<int32_t>::min();
  for (int64_t i = 0; i < kElementCount; ++i) {
    const int32_t value = input.Get<int32_t>({i});
    expected_sum += value;
    expected_max = std::max(expected_max, value);
  }
  ExpectScalarS32(sum_actual, expected_sum);

  auto max_result =
      execute(HloOpcode::kMaximum, std::numeric_limits<int32_t>::min());
  if (absl::IsFailedPrecondition(max_result.status())) {
    GTEST_SKIP() << max_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(Literal max_actual, std::move(max_result));
  ExpectScalarS32(max_actual, expected_max);
}

TEST(MetalGpuExecutableTest, ReductionS32ProductCallInput) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XlaBuilder remainder_builder("metal_s32_remainder");
  Shape input_shape = ShapeUtil::MakeShape(S32, {kElementCount});
  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  XlaOp callee_input = Parameter(&remainder_builder, 0, input_shape, "input");
  XlaOp divisor = Parameter(&remainder_builder, 1, scalar_shape, "divisor");
  Rem(callee_input, Broadcast(divisor, {kElementCount}));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation remainder,
                          remainder_builder.Build());

  XlaBuilder reducer_builder("metal_s32_product_reducer");
  XlaOp lhs = Parameter(&reducer_builder, 0, scalar_shape, "lhs");
  XlaOp rhs = Parameter(&reducer_builder, 1, scalar_shape, "rhs");
  Mul(lhs, rhs);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation reducer, reducer_builder.Build());

  XlaBuilder builder("metal_s32_product_call_input");
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp rem = Call(&builder, remainder, {input, ConstantR0<int32_t>(&builder, 5)});
  XlaOp shifted =
      Add(rem, Broadcast(ConstantR0<int32_t>(&builder, 1), {kElementCount}));
  Reduce(shifted, ConstantR0<int32_t>(&builder, 1), reducer, {0});
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
  int32_t expected = 1;
  for (int64_t i = 0; i < kElementCount; ++i) {
    expected *= (input_literal.Get<int32_t>({i}) % 5) + 1;
  }
  ExpectScalarS32(actual, expected);
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
