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

#include "xla/backends/gpu/autotuner/fly.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOk;
using ::testing::IsEmpty;

constexpr char kF16DotHlo[] = R"(
HloModule fly_f16_dot

gemm {
  lhs = f16[256,1024]{1,0} parameter(0)
  rhs = f16[1024,256]{0,1} parameter(1)
  ROOT dot = f16[256,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f16[256,1024]{1,0} parameter(0)
  rhs = f16[1024,256]{0,1} parameter(1)
  ROOT fusion = f16[256,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kF32DotHlo[] = R"(
HloModule fly_f32_dot

gemm {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT fusion = f32[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kF16GemvHlo[] = R"(
HloModule fly_f16_gemv

gemv {
  lhs = f16[1,1024]{1,0} parameter(0)
  rhs = f16[1024,256]{0,1} parameter(1)
  ROOT dot = f16[1,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f16[1,1024]{1,0} parameter(0)
  rhs = f16[1024,256]{0,1} parameter(1)
  ROOT fusion = f16[1,256]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kF16BatchedGemmHlo[] = R"(
HloModule fly_f16_batched_gemm

gemm {
  lhs = f16[4,256,128]{2,1,0} parameter(0)
  rhs = f16[4,192,128]{2,1,0} parameter(1)
  ROOT dot = f16[4,256,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  lhs = f16[4,256,128]{2,1,0} parameter(0)
  rhs = f16[4,192,128]{2,1,0} parameter(1)
  ROOT fusion = f16[4,256,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kF16BatchedGemvHlo[] = R"(
HloModule fly_f16_batched_gemv

gemv {
  lhs = f16[4,1,256]{2,1,0} parameter(0)
  rhs = f16[4,256,192]{2,1,0} parameter(1)
  ROOT dot = f16[4,1,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f16[4,1,256]{2,1,0} parameter(0)
  rhs = f16[4,256,192]{2,1,0} parameter(1)
  ROOT fusion = f16[4,1,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kF16BatchedMatrixVectorHlo[] = R"(
HloModule fly_f16_batched_matrix_vector

gemv {
  lhs = f16[4,256,1024]{2,1,0} parameter(0)
  rhs = f16[4,1024,1]{1,2,0} parameter(1)
  ROOT dot = f16[4,256,1]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f16[4,256,1024]{2,1,0} parameter(0)
  rhs = f16[4,1024,1]{1,2,0} parameter(1)
  ROOT fusion = f16[4,256,1]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kF16BatchedEpilogueHlo[] = R"(
HloModule fly_f16_batched_epilogue

gemm {
  lhs = f16[4,256,128]{2,1,0} parameter(0)
  rhs = f16[4,128,192]{2,1,0} parameter(1)
  column_bias = f16[192]{0} parameter(2)
  batch_scale = f16[4]{0} parameter(3)
  row_bias = f16[256]{0} parameter(4)
  dot = f16[4,256,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
  column_broadcast = f16[4,256,192]{2,1,0} broadcast(column_bias),
      dimensions={2}
  add_column = f16[4,256,192]{2,1,0} add(dot, column_broadcast)
  batch_broadcast = f16[4,256,192]{2,1,0} broadcast(batch_scale),
      dimensions={0}
  multiply_batch = f16[4,256,192]{2,1,0} multiply(add_column,
      batch_broadcast)
  row_broadcast = f16[4,256,192]{2,1,0} broadcast(row_bias), dimensions={1}
  ROOT add_row = f16[4,256,192]{2,1,0} add(multiply_batch, row_broadcast)
}

ENTRY main {
  lhs = f16[4,256,128]{2,1,0} parameter(0)
  rhs = f16[4,128,192]{2,1,0} parameter(1)
  column_bias = f16[192]{0} parameter(2)
  batch_scale = f16[4]{0} parameter(3)
  row_bias = f16[256]{0} parameter(4)
  ROOT fusion = f16[4,256,192]{2,1,0} fusion(lhs, rhs, column_bias,
      batch_scale, row_bias), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kAttentionValueOutputTransposeHlo[] = R"(
HloModule fly_attention_value_output_transpose

gemm {
  lhs = bf16[32,64,128]{2,1,0} parameter(0)
  rhs = bf16[32,128,128]{2,1,0} parameter(1)
  dot = bf16[32,64,128]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
  bitcast = bf16[2,16,64,128]{3,2,1,0} bitcast(dot)
  ROOT transpose = bf16[2,128,16,64]{3,2,1,0} transpose(bitcast),
      dimensions={0,3,1,2}
}

ENTRY main {
  lhs = bf16[32,64,128]{2,1,0} parameter(0)
  rhs = bf16[32,128,128]{2,1,0} parameter(1)
  ROOT fusion = bf16[2,128,16,64]{3,2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kScaledDotHlo[] = R"(
HloModule fly_scaled_dot

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = bf16[1024,1024]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kFnuzFp8ScaledDotHlo[] = R"(
HloModule fly_fnuz_fp8_scaled_dot

gemm {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e5m2fnuz[1024,1024]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = f32[1024,1024]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e5m2fnuz[1024,1024]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = f32[1024,1024]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kBatchedFnuzFp8ScaledDotHlo[] = R"(
HloModule fly_batched_fnuz_fp8_scaled_dot

gemm {
  lhs = f8e4m3fnuz[3,128,128]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,128,192]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT dot = f32[3,128,192]{2,1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f8e4m3fnuz[3,128,128]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[3,128,192]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT fusion = f32[3,128,192]{2,1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kHomogeneousFnuzFp8ScaledDotHlo[] = R"(
HloModule fly_homogeneous_fnuz_fp8_scaled_dot

gemm {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e4m3fnuz[1024,1024]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = f32[1024,1024]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e4m3fnuz[1024,1024]{0,1} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = f32[1024,1024]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kNonuniformScaleHlo[] = R"(
HloModule fly_scaled_dot_nonuniform_scale

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs_scale = f32[1024,8]{1,0} parameter(2)
  rhs_scale = f32[8,1024]{1,0} parameter(3)
  ROOT dot = bf16[1024,1024]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs_scale = f32[1024,8]{1,0} parameter(2)
  rhs_scale = f32[8,1024]{1,0} parameter(3)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kFnuzFp8NonuniformScaleHlo[] = R"(
HloModule fly_fnuz_fp8_scaled_dot_nonuniform_scale

gemm {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e4m3fnuz[1024,1024]{0,1} parameter(1)
  lhs_scale = f8e8m0fnu[1024,32]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[32,1024]{1,0} parameter(3)
  ROOT dot = f32[1024,1024]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e4m3fnuz[1024,1024]{0,1} parameter(1)
  lhs_scale = f8e8m0fnu[1024,32]{1,0} parameter(2)
  rhs_scale = f8e8m0fnu[32,1024]{1,0} parameter(3)
  ROOT fusion = f32[1024,1024]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kNarrowingEpilogueHlo[] = R"(
HloModule fly_narrowing_epilogue

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kNarrowingShortKEpilogueHlo[] = R"(
HloModule fly_narrowing_short_k_epilogue

gemm {
  lhs = bf16[512,512]{1,0} parameter(0)
  rhs = bf16[512,512]{0,1} parameter(1)
  dot = f32[512,512]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[512,512]{1,0} convert(dot)
}

ENTRY main {
  lhs = bf16[512,512]{1,0} parameter(0)
  rhs = bf16[512,512]{0,1} parameter(1)
  ROOT fusion = bf16[512,512]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kScalarEpilogueHlo[] = R"(
HloModule fly_scalar_epilogue

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  alpha = f32[] parameter(2)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  broadcast = f32[1024,1024]{1,0} broadcast(alpha), dimensions={}
  multiply = f32[1024,1024]{1,0} multiply(dot, broadcast)
  ROOT convert = bf16[1024,1024]{1,0} convert(multiply)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  alpha = f32[] parameter(2)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs, alpha), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kVectorEpilogueHlo[] = R"(
HloModule fly_vector_epilogue

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  bias = f32[1024]{0} parameter(2)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  broadcast = f32[1024,1024]{1,0} broadcast(bias), dimensions={1}
  add = f32[1024,1024]{1,0} add(dot, broadcast)
  ROOT convert = bf16[1024,1024]{1,0} convert(add)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  bias = f32[1024]{0} parameter(2)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs, bias), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kEpilogueChainHlo[] = R"(
HloModule fly_epilogue_chain

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  scale = f32[1024]{0} parameter(2)
  bias = f32[] parameter(3)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  scale_broadcast = f32[1024,1024]{1,0} broadcast(scale), dimensions={1}
  multiply = f32[1024,1024]{1,0} multiply(dot, scale_broadcast)
  bias_broadcast = f32[1024,1024]{1,0} broadcast(bias), dimensions={}
  add = f32[1024,1024]{1,0} add(multiply, bias_broadcast)
  negate = f32[1024,1024]{1,0} negate(add)
  ROOT convert = bf16[1024,1024]{1,0} convert(negate)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  scale = f32[1024]{0} parameter(2)
  bias = f32[] parameter(3)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs, scale, bias),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kBiasReluEpilogueHlo[] = R"(
HloModule fly_bias_relu_epilogue

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{1,0} parameter(1)
  bias = f32[1024]{0} parameter(2)
  zero = f32[] constant(0)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bias_broadcast = f32[1024,1024]{1,0} broadcast(bias), dimensions={1}
  add = f32[1024,1024]{1,0} add(dot, bias_broadcast)
  zero_broadcast = f32[1024,1024]{1,0} broadcast(zero), dimensions={}
  maximum = f32[1024,1024]{1,0} maximum(add, zero_broadcast)
  ROOT convert = bf16[1024,1024]{1,0} convert(maximum)
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{1,0} parameter(1)
  bias = f32[1024]{0} parameter(2)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs, bias),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedInputsHlo[] = R"(
HloModule fly_converted_inputs

gemm {
  lhs_f32 = f32[1024,1024]{1,0} parameter(0)
  rhs_f32 = f32[1024,1024]{0,1} parameter(1)
  lhs = bf16[1024,1024]{1,0} convert(lhs_f32)
  rhs = bf16[1024,1024]{0,1} convert(rhs_f32)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[1024,1024]{1,0} parameter(0)
  rhs = f32[1024,1024]{0,1} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedGemvInputsHlo[] = R"(
HloModule fly_converted_gemv_inputs

gemv {
  lhs_f32 = f32[256,256]{1,0} parameter(0)
  rhs_f32 = f32[256,1]{1,0} parameter(1)
  lhs = bf16[256,256]{1,0} convert(lhs_f32)
  rhs = bf16[256,1]{1,0} convert(rhs_f32)
  ROOT dot = bf16[256,1]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f32[256,256]{1,0} parameter(0)
  rhs = f32[256,1]{1,0} parameter(1)
  ROOT fusion = bf16[256,1]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedBitcastInputsHlo[] = R"(
HloModule fly_converted_bitcast_inputs

gemm {
  lhs_f32_physical = f32[512,2048]{1,0} parameter(0)
  rhs_f32_physical = f32[512,2048]{1,0} parameter(1)
  lhs_f32 = f32[1024,1024]{1,0} bitcast(lhs_f32_physical)
  lhs = bf16[1024,1024]{1,0} convert(lhs_f32)
  rhs_bf16_physical = bf16[512,2048]{1,0} convert(rhs_f32_physical)
  rhs = bf16[1024,1024]{0,1} bitcast(rhs_bf16_physical)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[512,2048]{1,0} parameter(0)
  rhs = f32[512,2048]{1,0} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedSliceInputsHlo[] = R"(
HloModule fly_converted_slice_inputs

gemm {
  lhs_f32_physical = f32[1056,1088]{1,0} parameter(0)
  rhs_f32_physical = f32[1088,1056]{0,1} parameter(1)
  lhs_f32 = f32[1024,1024]{1,0} slice(lhs_f32_physical),
      slice={[16:1040], [32:1056]}
  rhs_f32 = f32[1024,1024]{0,1} slice(rhs_f32_physical),
      slice={[32:1056], [16:1040]}
  lhs = bf16[1024,1024]{1,0} convert(lhs_f32)
  rhs = bf16[1024,1024]{0,1} convert(rhs_f32)
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[1056,1088]{1,0} parameter(0)
  rhs = f32[1088,1056]{0,1} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kStridedSliceInputHlo[] = R"(
HloModule fly_strided_slice_input

gemm {
  lhs_physical = bf16[1024,2048]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  lhs = bf16[1024,1024]{1,0} slice(lhs_physical),
      slice={[0:1024], [0:2048:2]}
  ROOT dot = bf16[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,2048]{1,0} parameter(0)
  rhs = bf16[1024,1024]{0,1} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kDynamicSliceInputsHlo[] = R"(
HloModule fly_dynamic_slice_inputs

gemm {
  lhs_physical = bf16[1056,1088]{1,0} parameter(0)
  rhs_physical = bf16[1088,1056]{0,1} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  lhs = bf16[1024,1024]{1,0} dynamic-slice(
      lhs_physical, lhs_start_m, lhs_start_k),
      dynamic_slice_sizes={1024,1024}
  rhs = bf16[1024,1024]{0,1} dynamic-slice(
      rhs_physical, rhs_start_k, rhs_start_n),
      dynamic_slice_sizes={1024,1024}
  ROOT dot = bf16[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1056,1088]{1,0} parameter(0)
  rhs = bf16[1088,1056]{0,1} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(
      lhs, rhs, lhs_start_m, lhs_start_k, rhs_start_k, rhs_start_n),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kFnuzFp8InputsHlo[] = R"(
HloModule fly_fnuz_fp8_inputs

gemm {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e5m2fnuz[1024,1024]{0,1} parameter(1)
  ROOT dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e5m2fnuz[1024,1024]{0,1} parameter(1)
  ROOT fusion = f32[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kHomogeneousFnuzFp8InputsHlo[] = R"(
HloModule fly_homogeneous_fnuz_fp8_inputs

gemm {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e4m3fnuz[1024,1024]{0,1} parameter(1)
  ROOT dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[1024,1024]{1,0} parameter(0)
  rhs = f8e4m3fnuz[1024,1024]{0,1} parameter(1)
  ROOT fusion = f32[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kFnuzFp8GemvHlo[] = R"(
HloModule fly_fnuz_fp8_gemv

gemv {
  lhs = f8e4m3fnuz[1,4096]{1,0} parameter(0)
  rhs = f8e5m2fnuz[4096,4096]{1,0} parameter(1)
  ROOT dot = f32[1,4096]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[1,4096]{1,0} parameter(0)
  rhs = f8e5m2fnuz[4096,4096]{1,0} parameter(1)
  ROOT fusion = f32[1,4096]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kUniformScaledFnuzFp8BatchedGemvHlo[] = R"(
HloModule fly_uniform_scaled_fnuz_fp8_batched_gemv

gemv {
  lhs = f8e4m3fnuz[4,1,256]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[4,256,192]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT dot = f32[4,1,192]{2,1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f8e4m3fnuz[4,1,256]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[4,256,192]{2,1,0} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT fusion = f32[4,1,192]{2,1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kFnuzFp8BatchedGemmHlo[] = R"(
HloModule fly_fnuz_fp8_batched_gemm

gemm {
  lhs = f8e4m3fnuz[4,256,128]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[4,128,192]{2,1,0} parameter(1)
  ROOT dot = f32[4,256,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f8e4m3fnuz[4,256,128]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[4,128,192]{2,1,0} parameter(1)
  ROOT fusion = f32[4,256,192]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kFnuzFp8BatchedGemvHlo[] = R"(
HloModule fly_fnuz_fp8_batched_gemv

gemv {
  lhs = f8e4m3fnuz[4,1,256]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[4,256,192]{2,1,0} parameter(1)
  ROOT dot = f32[4,1,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f8e4m3fnuz[4,1,256]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[4,256,192]{2,1,0} parameter(1)
  ROOT fusion = f32[4,1,192]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kS4DequantizedRhsHlo[] = R"(
HloModule fly_s4_dequantized_rhs

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs.s4 = s4[1024,1024]{1,0:E(4)} parameter(1)
  rhs.s8 = s8[1024,1024]{1,0} convert(rhs.s4)
  rhs.bf16 = bf16[1024,1024]{1,0} convert(rhs.s8)
  ROOT dot = f32[1024,1024]{1,0} dot(lhs, rhs.bf16),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = s4[1024,1024]{1,0:E(4)} parameter(1)
  ROOT fusion = f32[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__triton_gemm"}}
})";

constexpr char kS4DequantizedKContiguousRhsHlo[] = R"(
HloModule fly_s4_dequantized_k_contiguous_rhs

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs.s4 = s4[1024,1024]{0,1:E(4)} parameter(1)
  rhs.s8 = s8[1024,1024]{0,1} convert(rhs.s4)
  rhs.bf16 = bf16[1024,1024]{0,1} convert(rhs.s8)
  ROOT dot = f32[1024,1024]{1,0} dot(lhs, rhs.bf16),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = s4[1024,1024]{0,1:E(4)} parameter(1)
  ROOT fusion = f32[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__triton_gemm"}}
})";

constexpr char kS4ChannelScaledBatchedLhsHlo[] = R"(
HloModule fly_s4_channel_scaled_batched_lhs

gemm {
  weights.s4 = s4[4,64,64]{1,2,0:E(4)} parameter(0)
  weights.s8 = s8[4,64,64]{1,2,0} convert(weights.s4)
  weights.bf16 = bf16[4,64,64]{1,2,0} convert(weights.s8)
  weights.transpose = bf16[4,64,64]{1,2,0}
      transpose(weights.bf16), dimensions={0,2,1}
  scales = bf16[4,64]{1,0} parameter(1)
  scales.broadcast = bf16[4,64,64]{1,2,0}
      broadcast(scales), dimensions={0,2}
  weights.scaled = bf16[4,64,64]{1,2,0}
      multiply(weights.transpose, scales.broadcast)
  activations = bf16[4,64,128]{2,1,0} parameter(2)
  ROOT dot = f32[4,64,128]{2,1,0} dot(weights.scaled, activations),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  weights = s4[4,64,64]{1,2,0:E(4)} parameter(0)
  scales = bf16[4,64]{1,0} parameter(1)
  activations = bf16[4,64,128]{2,1,0} parameter(2)
  ROOT fusion = f32[4,64,128]{2,1,0}
      fusion(weights, scales, activations), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kS4SubchannelScaledBatchedLhsHlo[] = R"(
HloModule fly_s4_subchannel_scaled_batched_lhs

gemm {
  weights.s4 = s4[2,2048,32]{2,1,0:E(4)} parameter(0)
  weights.s8 = s8[2,2048,32]{2,1,0} convert(weights.s4)
  weights.groups.s8 = s8[2,8,256,32]{3,2,1,0} bitcast(weights.s8)
  weights.groups = bf16[2,8,256,32]{3,2,1,0}
      convert(weights.groups.s8)
  scales = bf16[2,8,1,32]{3,2,1,0} parameter(1)
  scales.view = bf16[2,8,32]{2,1,0} bitcast(scales)
  scales.broadcast = bf16[2,8,256,32]{3,2,1,0}
      broadcast(scales.view), dimensions={0,1,3}
  weights.scaled.groups = bf16[2,8,256,32]{3,2,1,0}
      multiply(weights.groups, scales.broadcast)
  weights.scaled = bf16[2,2048,32]{2,1,0}
      bitcast(weights.scaled.groups)
  activations = bf16[2,2,1,2048]{3,2,1,0} parameter(2)
  activations.view = bf16[2,2,2048]{2,1,0} bitcast(activations)
  ROOT dot = f32[2,32,2]{2,1,0} dot(weights.scaled, activations.view),
      lhs_batch_dims={0}, lhs_contracting_dims={1},
      rhs_batch_dims={1}, rhs_contracting_dims={2}
}

ENTRY main {
  weights = s4[2,2048,32]{2,1,0:E(4)} parameter(0)
  scales = bf16[2,8,1,32]{3,2,1,0} parameter(1)
  activations = bf16[2,2,1,2048]{3,2,1,0} parameter(2)
  ROOT fusion = f32[2,32,2]{2,1,0}
      fusion(weights, scales, activations), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kS4ChannelScaledMultiBatchLhsHlo[] = R"(
HloModule fly_s4_channel_scaled_multi_batch_lhs

gemm {
  weights.s4 = s4[3,2,32,16]{3,2,1,0:E(4)} parameter(0)
  weights.s8 = s8[3,2,32,16]{3,2,1,0} convert(weights.s4)
  weights.bf16 = bf16[3,2,32,16]{3,2,1,0} convert(weights.s8)
  scales = bf16[3,16]{1,0} parameter(1)
  scales.broadcast = bf16[3,2,32,16]{3,2,1,0}
      broadcast(scales), dimensions={0,3}
  weights.scaled = bf16[3,2,32,16]{3,2,1,0}
      multiply(weights.bf16, scales.broadcast)
  activations = bf16[3,2,32,16]{3,2,1,0} parameter(2)
  ROOT dot = f32[2,3,16,16]{3,2,1,0} dot(weights.scaled, activations),
      lhs_batch_dims={1,0}, lhs_contracting_dims={2},
      rhs_batch_dims={1,0}, rhs_contracting_dims={2}
}

ENTRY main {
  weights = s4[3,2,32,16]{3,2,1,0:E(4)} parameter(0)
  scales = bf16[3,16]{1,0} parameter(1)
  activations = bf16[3,2,32,16]{3,2,1,0} parameter(2)
  ROOT fusion = f32[2,3,16,16]{3,2,1,0}
      fusion(weights, scales, activations), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kConvertedConcatInputsHlo[] = R"(
HloModule fly_converted_concat_inputs

gemm {
  lhs0_f32 = f32[512,1024]{1,0} parameter(0)
  lhs1_f32 = f32[512,1024]{1,0} parameter(1)
  rhs0_f32 = f32[1024,512]{0,1} parameter(2)
  rhs1_f32 = f32[1024,512]{0,1} parameter(3)
  lhs0 = bf16[512,1024]{1,0} convert(lhs0_f32)
  lhs1 = bf16[512,1024]{1,0} convert(lhs1_f32)
  rhs0 = bf16[1024,512]{0,1} convert(rhs0_f32)
  rhs1 = bf16[1024,512]{0,1} convert(rhs1_f32)
  lhs = bf16[1024,1024]{1,0} concatenate(lhs0, lhs1), dimensions={0}
  rhs = bf16[1024,1024]{0,1} concatenate(rhs0, rhs1), dimensions={1}
  dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[1024,1024]{1,0} convert(dot)
}

ENTRY main {
  lhs0 = f32[512,1024]{1,0} parameter(0)
  lhs1 = f32[512,1024]{1,0} parameter(1)
  rhs0 = f32[1024,512]{0,1} parameter(2)
  rhs1 = f32[1024,512]{0,1} parameter(3)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs0, lhs1, rhs0, rhs1),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kContractingConcatInputHlo[] = R"(
HloModule fly_contracting_concat_input

gemm {
  lhs0 = bf16[1024,512]{1,0} parameter(0)
  lhs1 = bf16[1024,512]{1,0} parameter(1)
  rhs = bf16[1024,1024]{0,1} parameter(2)
  lhs = bf16[1024,1024]{1,0} concatenate(lhs0, lhs1), dimensions={1}
  ROOT dot = bf16[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs0 = bf16[1024,512]{1,0} parameter(0)
  lhs1 = bf16[1024,512]{1,0} parameter(1)
  rhs = bf16[1024,1024]{0,1} parameter(2)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs0, lhs1, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kTransposedUnevenConcatInputHlo[] = R"(
HloModule fly_transposed_uneven_concat_input

gemm {
  lhs = bf16[256,1024]{1,0} parameter(0)
  rhs0 = bf16[1024,1024]{1,0} parameter(1)
  rhs1 = bf16[1024,512]{1,0} parameter(2)
  concatenated = bf16[1024,1536]{1,0}
      concatenate(rhs0, rhs1), dimensions={1}
  rhs = bf16[1536,1024]{0,1}
      transpose(concatenated), dimensions={1,0}
  ROOT dot = bf16[256,1536]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[256,1024]{1,0} parameter(0)
  rhs0 = bf16[1024,1024]{1,0} parameter(1)
  rhs1 = bf16[1024,512]{1,0} parameter(2)
  ROOT fusion = bf16[256,1536]{1,0} fusion(lhs, rhs0, rhs1),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kBatchedGemmHlo[] = R"(
HloModule fly_batched_gemm

gemm {
  lhs = bf16[4,256,128]{2,1,0} parameter(0)
  rhs = bf16[4,192,128]{2,1,0} parameter(1)
  ROOT dot = bf16[4,256,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  lhs = bf16[4,256,128]{2,1,0} parameter(0)
  rhs = bf16[4,192,128]{2,1,0} parameter(1)
  ROOT fusion = bf16[4,256,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kBatchedGemvHlo[] = R"(
HloModule fly_batched_gemv

gemv {
  lhs = bf16[4,1,256]{2,1,0} parameter(0)
  rhs = bf16[4,256,192]{2,1,0} parameter(1)
  ROOT dot = bf16[4,1,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[4,1,256]{2,1,0} parameter(0)
  rhs = bf16[4,256,192]{2,1,0} parameter(1)
  ROOT fusion = bf16[4,1,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kBatchedGemvKContiguousHlo[] = R"(
HloModule fly_batched_gemv_k_contiguous

gemv {
  lhs = bf16[4,1,256]{2,1,0} parameter(0)
  rhs = bf16[4,256,192]{1,2,0} parameter(1)
  ROOT dot = bf16[4,1,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[4,1,256]{2,1,0} parameter(0)
  rhs = bf16[4,256,192]{1,2,0} parameter(1)
  ROOT fusion = bf16[4,1,192]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kF32OutputBf16GemmHlo[] = R"(
HloModule fly_f32_output_bf16_gemm

gemm {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{1,0} parameter(1)
  ROOT dot = f32[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,1024]{1,0} parameter(0)
  rhs = bf16[1024,1024]{1,0} parameter(1)
  ROOT fusion = f32[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

constexpr char kOddOutputTailBf16GemmHlo[] = R"(
HloModule fly_odd_output_tail_bf16_gemm

gemm {
  lhs = bf16[79,48]{1,0} parameter(0)
  rhs = bf16[48,111]{0,1} parameter(1)
  ROOT dot = bf16[79,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[79,48]{1,0} parameter(0)
  rhs = bf16[48,111]{0,1} parameter(1)
  ROOT fusion = bf16[79,111]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";

class FlyBackendTest : public HloHardwareIndependentTestBase {
 protected:
  FlyBackendTest()
      : platform_(PlatformUtil::GetDefaultPlatform().value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        target_config_(stream_executor_),
        compiler_(Compiler::GetForPlatform(platform_->id()).value()),
        backend_(&debug_options_, compiler_.get(), &target_config_) {
    debug_options_.set_xla_gpu_enable_flydsl_gemm(true);
  }

  DebugOptions debug_options_;
  se::Platform* platform_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  std::unique_ptr<Compiler> compiler_;
  FlyBackend backend_;
};

class FlyFusionBackendTest : public HloHardwareIndependentTestBase {
 protected:
  FlyFusionBackendTest()
      : platform_(PlatformUtil::GetDefaultPlatform().value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        target_config_(stream_executor_),
        compiler_(Compiler::GetForPlatform(platform_->id()).value()),
        backend_(&debug_options_, compiler_.get(), &target_config_) {
    debug_options_.set_xla_gpu_enable_flydsl_fusion(true);
    debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(1);
  }

  DebugOptions debug_options_;
  se::Platform* platform_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  std::unique_ptr<Compiler> compiler_;
  FlyFusionBackend backend_;
};

TEST_F(FlyBackendTest, BoundsDefaultGemmSearchAndKeepsExhaustiveOptIn) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kF16DotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> defaults,
                       backend_.GetSupportedConfigs(*fusion));
  EXPECT_LT(defaults.size(), 256);
  EXPECT_TRUE(absl::c_any_of(defaults, [](const auto& config) {
    const FlyGemmConfig& fly = config->fly();
    return fly.block_m() == 32 && fly.block_n() == 16 &&
           fly.block_k() == 256 && fly.num_warps() == 2 &&
           fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16;
  }));

  debug_options_.set_xla_gpu_exhaustive_tiling_search(true);
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> exhaustive,
                       backend_.GetSupportedConfigs(*fusion));
  EXPECT_GT(exhaustive.size(), defaults.size() * 4);
}

TEST_F(FlyFusionBackendTest, RejectsGenericFusionWithOpaqueCustomCall) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_opaque_custom_call

body {
  p0 = f32[8]{0} parameter(0)
  ROOT unsupported_call = f32[8]{0} custom-call(p0),
      custom_call_target="__opaque$unsupported"
}

ENTRY main {
  p0 = f32[8]{0} parameter(0)
  ROOT fusion = f32[8]{0} fusion(p0), kind=kCustom, calls=body,
      backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["8"]}],"num_warps":"1","num_ctas":"1","num_stages":"1"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  EXPECT_TRUE(configs.empty());
}

TEST_F(FlyFusionBackendTest, LeavesLargeOrdinaryIndexedDagOnNativeEmitter) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_large_ordinary_indexed_dag

body {
  input = f32[16,256]{1,0} parameter(0)
  slice0 = f32[1,256]{1,0} slice(input), slice={[0:1], [0:256]}
  slice1 = f32[1,256]{1,0} slice(input), slice={[1:2], [0:256]}
  slice2 = f32[1,256]{1,0} slice(input), slice={[2:3], [0:256]}
  slice3 = f32[1,256]{1,0} slice(input), slice={[3:4], [0:256]}
  slice4 = f32[1,256]{1,0} slice(input), slice={[4:5], [0:256]}
  slice5 = f32[1,256]{1,0} slice(input), slice={[5:6], [0:256]}
  slice6 = f32[1,256]{1,0} slice(input), slice={[6:7], [0:256]}
  slice7 = f32[1,256]{1,0} slice(input), slice={[7:8], [0:256]}
  slice8 = f32[1,256]{1,0} slice(input), slice={[8:9], [0:256]}
  slice9 = f32[1,256]{1,0} slice(input), slice={[9:10], [0:256]}
  slice10 = f32[1,256]{1,0} slice(input), slice={[10:11], [0:256]}
  slice11 = f32[1,256]{1,0} slice(input), slice={[11:12], [0:256]}
  slice12 = f32[1,256]{1,0} slice(input), slice={[12:13], [0:256]}
  slice13 = f32[1,256]{1,0} slice(input), slice={[13:14], [0:256]}
  slice14 = f32[1,256]{1,0} slice(input), slice={[14:15], [0:256]}
  slice15 = f32[1,256]{1,0} slice(input), slice={[15:16], [0:256]}
  ROOT result = f32[16,256]{1,0} concatenate(
      slice0, slice1, slice2, slice3, slice4, slice5, slice6, slice7,
      slice8, slice9, slice10, slice11, slice12, slice13, slice14, slice15),
      dimensions={0}
}

ENTRY main {
  input = f32[16,256]{1,0} parameter(0)
  ROOT fusion = f32[16,256]{1,0} fusion(input), kind=kLoop, calls=body
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsEmpty());
}

TEST_F(FlyFusionBackendTest,
       StrictReplacementLeavesOrdinaryInputReductionOnNativeEmitter) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ordinary_input_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

body {
  input = bf16[2,128,512]{2,1,0} parameter(0)
  flattened = bf16[256,512]{1,0} bitcast(input)
  converted = f32[256,512]{1,0} convert(flattened)
  zero_bf16 = bf16[] constant(0)
  zero = f32[] convert(zero_bf16)
  reduced = f32[512]{0} reduce(converted, zero), dimensions={0},
    to_apply=add
  ROOT result = bf16[512]{0} convert(reduced)
}

ENTRY main {
  input = bf16[2,128,512]{2,1,0} parameter(0)
  ROOT fusion = bf16[512]{0} fusion(input), kind=kInput, calls=body
}
)";
  debug_options_.set_xla_gpu_flydsl_replace_triton(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsEmpty());
}

TEST_F(FlyFusionBackendTest,
       StrictReplacementAcceptsNativeMixedOutputLeadingReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_non_native_mixed_output_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

body {
  residual = bf16[2,17,259]{2,1,0} parameter(0)
  partials = f32[2,34,259]{2,1,0} parameter(1)
  residual_f32 = f32[2,17,259]{2,1,0} convert(residual)
  zero = f32[] constant(0)
  projected = f32[34,259]{1,0} reduce(partials, zero), dimensions={0},
    to_apply=add
  projected_bf16 = bf16[34,259]{1,0} convert(projected)
  ROOT tuple = (f32[2,17,259]{2,1,0}, bf16[34,259]{1,0})
    tuple(residual_f32, projected_bf16)
}

ENTRY main {
  residual = bf16[2,17,259]{2,1,0} parameter(0)
  partials = f32[2,34,259]{2,1,0} parameter(1)
  ROOT fusion = (f32[2,17,259]{2,1,0}, bf16[34,259]{1,0})
    fusion(residual, partials), kind=kLoop, calls=body
}
)";
  debug_options_.set_xla_gpu_flydsl_replace_triton(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  ASSERT_FALSE(configs.empty());
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, TunesNativeScanWavePackingAndOccupancy) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_scan_autotune

scan_computation {
  input = bf16[4096,257]{1,0} parameter(0)
  ROOT scan = bf16[4096,257]{1,0} custom-call(input),
    custom_call_target="__fly$scan",
    backend_config={"vector_length":1,"row_length":257,
      "column_length":4096,"kind":1,"is_reverse":false}
}

ENTRY main {
  input = bf16[4096,257]{1,0} parameter(0)
  ROOT fusion = bf16[4096,257]{1,0} fusion(input), kind=kCustom,
    calls=scan_computation,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1","1"]}],
      "num_warps":"4","num_ctas":"1","num_stages":"1",
      "waves_per_eu":"2"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<std::pair<int64_t, int64_t>, 8> kExpected = {{
      {8, 0},
      {4, 0},
      {16, 0},
      {2, 0},
      {8, 2},
      {4, 2},
      {1, 0},
      {8, 4},
  }};
  for (int64_t i = 0; i < configs.size(); ++i) {
    EXPECT_EQ(configs[i]->block_level().num_warps(), kExpected[i].first);
    EXPECT_EQ(configs[i]->block_level().waves_per_eu(), kExpected[i].second);
  }
}

TEST_F(FlyBackendTest, SupportsOddOutputTailsAndMinimumKTile) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kOddOutputTailBf16GemmHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 64 && fly.block_n() == 64 &&
               fly.block_k() == 16 && fly.num_warps() == 4 &&
               !fly.stage_rhs() && !fly.stage_output();
      }));
}

TEST_F(FlyBackendTest, SupportsPredicatedSmallMGemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_predicated_small_m_gemm

gemm {
  lhs = bf16[4,256]{1,0} parameter(0)
  rhs = bf16[256,192]{1,0} parameter(1)
  ROOT dot = bf16[4,192]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[4,256]{1,0} parameter(0)
  rhs = bf16[256,192]{1,0} parameter(1)
  ROOT fusion = bf16[4,192]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  auto gemv_config =
      std::find_if(configs.begin(), configs.end(), [](const auto& c) {
        const FlyGemmConfig& fly = c->fly();
        return fly.block_m() == 4 && fly.block_n() == 64 &&
               fly.gemv_outputs_per_wave() == 4 && !fly.gemv_split_k();
      });
  ASSERT_NE(gemv_config, configs.end());
  auto staged_config =
      std::find_if(configs.begin(), configs.end(), [](const auto& c) {
        const FlyGemmConfig& fly = c->fly();
        return fly.block_m() == 16 && fly.block_n() == 64 &&
               fly.block_k() == 64 && fly.num_warps() == 4 &&
               fly.stage_rhs();
      });
  ASSERT_NE(staged_config, configs.end());
  auto generic_config =
      std::find_if(configs.begin(), configs.end(), [](const auto& c) {
        const FlyGemmConfig& fly = c->fly();
        return fly.block_m() == 16 && fly.block_n() == 64 &&
               fly.block_k() == 32 && !fly.stage_rhs();
      });
  ASSERT_NE(generic_config, configs.end());
  ASSERT_OK(backend_.ApplyConfig(*fusion, **gemv_config));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, OffersPipelinedMfma4DecoderProjection) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_mfma4_decoder_projection

gemm {
  lhs = bf16[4,256]{1,0} parameter(0)
  rhs = bf16[256,256]{1,0} parameter(1)
  ROOT dot = bf16[4,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[4,256]{1,0} parameter(0)
  rhs = bf16[256,256]{1,0} parameter(1)
  ROOT fusion = bf16[4,256]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  auto mfma4_config =
      std::find_if(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 4 && fly.block_n() == 128 &&
               fly.block_k() == 128 && fly.num_warps() == 8 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_4X4X4_BF16 &&
               fly.gemv_outputs_per_wave() == 1 &&
               fly.gemv_k_vector_width() == 1 && !fly.gemv_split_k();
      });
  ASSERT_NE(mfma4_config, configs.end());
  ASSERT_OK(backend_.ApplyConfig(*fusion, **mfma4_config));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, OffersBatchedPipelinedMfma4DecoderProjection) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_batched_mfma4_decoder_projection

gemm {
  lhs = bf16[4,3,224]{2,1,0} parameter(0)
  rhs = bf16[3,224,128]{2,1,0} parameter(1)
  ROOT dot = f32[3,4,128]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[4,3,224]{2,1,0} parameter(0)
  rhs = bf16[3,224,128]{2,1,0} parameter(1)
  ROOT fusion = f32[3,4,128]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  auto mfma4_config =
      std::find_if(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 4 && fly.block_n() == 64 &&
               fly.block_k() == 128 && fly.num_warps() == 2 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_4X4X4_BF16 &&
               fly.gemv_outputs_per_wave() == 1 &&
               fly.gemv_k_vector_width() == 1 && !fly.gemv_split_k();
      });
  ASSERT_NE(mfma4_config, configs.end());
  ASSERT_OK(backend_.ApplyConfig(*fusion, **mfma4_config));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, FusesRoundedContractingScaleIntoSmallMProjection) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_small_m_rounded_contracting_scale

gemm {
  data = bf16[4,256]{1,0} parameter(0)
  scale = bf16[256]{0} parameter(1)
  matrix = bf16[256,256]{1,0} parameter(2)
  data_f32 = f32[4,256]{1,0} convert(data)
  scale_broadcast = bf16[4,256]{1,0} broadcast(scale), dimensions={1}
  scale_f32 = f32[4,256]{1,0} convert(scale_broadcast)
  product = f32[4,256]{1,0} multiply(data_f32, scale_f32)
  rounded = bf16[4,256]{1,0} convert(product)
  ROOT dot = bf16[4,256]{1,0} dot(rounded, matrix),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  data = bf16[4,256]{1,0} parameter(0)
  scale = bf16[256]{0} parameter(1)
  matrix = bf16[256,256]{1,0} parameter(2)
  ROOT fusion = bf16[4,256]{1,0} fusion(data, scale, matrix),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  auto gemv_config =
      std::find_if(configs.begin(), configs.end(), [](const auto& c) {
        return c->fly().block_m() == 4 &&
               c->fly().gemv_outputs_per_wave() == 4 &&
               !c->fly().stage_rhs();
      });
  ASSERT_NE(gemv_config, configs.end());
  auto staged_config =
      std::find_if(configs.begin(), configs.end(), [](const auto& c) {
        return c->fly().block_m() == 16 && c->fly().block_n() == 64 &&
               c->fly().block_k() == 64 && c->fly().num_warps() == 4 &&
               c->fly().stage_rhs();
      });
  ASSERT_NE(staged_config, configs.end());
  auto repository_small_m_config =
      std::find_if(configs.begin(), configs.end(), [](const auto& c) {
        return c->fly().block_m() == 16 && c->fly().block_n() == 64 &&
               c->fly().block_k() == 128 && c->fly().num_warps() == 4 &&
               c->fly().stage_rhs() && c->fly().preload_lds_fragments();
      });
  ASSERT_NE(repository_small_m_config, configs.end());
  auto mfma4_config =
      std::find_if(configs.begin(), configs.end(), [](const auto& c) {
        return c->fly().block_m() == 4 && c->fly().block_n() == 128 &&
               c->fly().block_k() == 128 && c->fly().num_warps() == 8 &&
               c->fly().mfma_atom() ==
                   FlyGemmConfig::FLY_MFMA_4X4X4_BF16 &&
               c->fly().gemv_outputs_per_wave() == 1 &&
               c->fly().gemv_k_vector_width() == 1;
      });
  ASSERT_NE(mfma4_config, configs.end());
  ASSERT_OK(backend_.ApplyConfig(*fusion, **mfma4_config));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, SupportsSmallMRank3DecoderContraction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_small_m_rank3_decoder_contraction

gemm {
  lhs = bf16[4,3,128]{2,1,0} parameter(0)
  rhs = bf16[3,128,192]{2,1,0} parameter(1)
  ROOT dot = f32[3,4,192]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[4,3,128]{2,1,0} parameter(0)
  rhs = bf16[3,128,192]{2,1,0} parameter(1)
  ROOT fusion = f32[3,4,192]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  auto config =
      std::find_if(configs.begin(), configs.end(), [](const auto& c) {
        const FlyGemmConfig& fly = c->fly();
        return fly.block_m() == 4 && fly.block_n() == 64 &&
               fly.gemv_outputs_per_wave() == 4;
      });
  ASSERT_NE(config, configs.end());
  ASSERT_OK(backend_.ApplyConfig(*fusion, **config));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, SupportsNonAlignedBf16KWithMaskedAtom) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_non_aligned_bf16_k

gemm {
  lhs = bf16[79,47]{1,0} parameter(0)
  rhs = bf16[47,111]{0,1} parameter(1)
  ROOT dot = bf16[79,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[79,47]{1,0} parameter(0)
  rhs = bf16[47,111]{0,1} parameter(1)
  ROOT fusion = bf16[79,111]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 64 && fly.block_n() == 64 &&
               fly.block_k() == 64 && fly.num_warps() == 4 && !fly.stage_rhs();
      }));
  EXPECT_TRUE(std::all_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().block_k() <= 64; }));
}

TEST_F(FlyBackendTest, OffersPreloadedRhsForLongMaskedKTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_long_masked_bf16_k

gemm {
  lhs = bf16[1024,1023]{1,0} parameter(0)
  rhs = bf16[1023,1024]{0,1} parameter(1)
  ROOT dot = bf16[1024,1024]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1024,1023]{1,0} parameter(0)
  rhs = bf16[1023,1024]{0,1} parameter(1)
  ROOT fusion = bf16[1024,1024]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  auto config = std::find_if(configs.begin(), configs.end(), [](const auto& c) {
    const FlyGemmConfig& fly = c->fly();
    return fly.block_m() == 64 && fly.block_n() == 64 && fly.block_k() == 64 &&
           fly.num_warps() == 4 && fly.stage_rhs() &&
           fly.preload_lds_fragments() && !fly.stage_output();
  });
  ASSERT_NE(config, configs.end());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      backend_.Compile(*module->entry_computation()->root_instruction(),
                       **config));
  EXPECT_THAT(executable->module_stats(), IsEmpty());
}

TEST_F(FlyBackendTest, OffersDirectAndStagedRowMajorMfma32Output) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_row_major_mfma32_output

gemm {
  lhs = bf16[256,1024]{1,0} parameter(0)
  rhs = bf16[1024,256]{1,0} parameter(1)
  ROOT dot = bf16[256,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[256,1024]{1,0} parameter(0)
  rhs = bf16[1024,256]{1,0} parameter(1)
  ROOT fusion = bf16[256,256]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  auto is_row_major_mfma32 = [](const auto& config) {
    const FlyGemmConfig& fly = config->fly();
    return fly.block_m() == 128 && fly.block_n() == 128 &&
           fly.block_k() == 64 && fly.num_warps() == 4 &&
           fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 &&
           fly.stage_rhs() && fly.preload_lds_fragments() &&
           !fly.single_buffer_lds();
  };
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [&](const auto& config) {
        return is_row_major_mfma32(config) && !config->fly().stage_output() &&
               config->fly().schedule_instructions();
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [&](const auto& config) {
        return is_row_major_mfma32(config) && config->fly().stage_output() &&
               !config->fly().schedule_instructions();
      }));
}

TEST_F(FlyBackendTest, OffersCompactVectorRefillForShallowBf16Projection) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_shallow_bf16_projection

gemm {
  lhs = bf16[128,1024]{1,0} parameter(0)
  rhs = bf16[1024,256]{0,1} parameter(1)
  ROOT dot = bf16[128,256]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[128,1024]{1,0} parameter(0)
  rhs = bf16[1024,256]{0,1} parameter(1)
  ROOT fusion = bf16[128,256]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    const FlyGemmConfig& fly = c->fly();
    return fly.block_m() == 32 && fly.block_n() == 16 &&
           fly.block_k() == 256 && fly.num_warps() == 2 && fly.stage_rhs() &&
           fly.single_buffer_lds() && !fly.preload_lds_fragments() &&
           !fly.schedule_instructions();
  }));
}

TEST_F(FlyBackendTest, OffersK32PipelineForRaggedBatchedRowMajorRhs) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ragged_batched_row_major_rhs

gemm {
  lhs = bf16[2,129,1025]{2,1,0} parameter(0)
  rhs = bf16[2,1025,257]{2,1,0} parameter(1)
  ROOT dot = bf16[2,129,257]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[2,129,1025]{2,1,0} parameter(0)
  rhs = bf16[2,1025,257]{2,1,0} parameter(1)
  ROOT fusion = bf16[2,129,257]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 128 && fly.block_n() == 128 &&
               fly.block_k() == 32 && fly.num_warps() == 4 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               !fly.single_buffer_lds() && !fly.stage_output() &&
               fly.schedule_instructions();
      }));
  for (int32_t workgroup_mapping_n : {1, 2, 4, 8}) {
    EXPECT_TRUE(
        std::any_of(configs.begin(), configs.end(), [&](const auto& config) {
          const FlyGemmConfig& fly = config->fly();
          return fly.block_m() == 64 && fly.block_n() == 256 &&
                 fly.block_k() == 32 && fly.num_warps() == 8 &&
                 fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16 &&
                 !fly.stage_rhs() && !fly.stage_output() &&
                 fly.workgroup_mapping_n() == workgroup_mapping_n;
        }));
  }
}

TEST_F(FlyBackendTest, SupportsGemvKAndOutputTails) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_gemv_k_and_output_tails

gemv {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{0,1} parameter(1)
  ROOT dot = bf16[1,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{0,1} parameter(1)
  ROOT fusion = bf16[1,111]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_n() == 64 && fly.num_warps() == 4 &&
               fly.gemv_outputs_per_wave() == 4 &&
               fly.gemv_k_vector_width() == 1;
      }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().block_n() == 128; }));
}

TEST_F(FlyBackendTest, SupportsTransposedSingletonLhsGemv) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_transposed_singleton_lhs_gemv

gemv {
  lhs = bf16[512,1]{1,0} parameter(0)
  rhs = bf16[512,192]{1,0} parameter(1)
  ROOT dot = bf16[1,192]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[512,1]{1,0} parameter(0)
  rhs = bf16[512,192]{1,0} parameter(1)
  ROOT fusion = bf16[1,192]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");

  auto local_split = std::find_if(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().local_split_k(); });
  ASSERT_NE(local_split, configs.end());
  EXPECT_THAT(backend_.ApplyConfig(*fusion, **local_split), IsOk());
  ASSERT_OK_AND_ASSIGN(gpu_config, fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsBatchedTransposedSingletonLhsGemv) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_batched_transposed_singleton_lhs_gemv

gemv {
  lhs = bf16[3,257,1]{2,1,0} parameter(0)
  rhs = bf16[3,257,193]{2,1,0} parameter(1)
  ROOT dot = f32[3,1,193]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[3,257,1]{2,1,0} parameter(0)
  rhs = bf16[3,257,193]{2,1,0} parameter(1)
  ROOT fusion = f32[3,1,193]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, OffersMfmaRowVectorLocalSplitK) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF16GemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  auto candidate =
      std::find_if(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 16 && fly.block_n() == 16 &&
               fly.block_k() == 256 && fly.num_warps() == 2 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16 &&
               fly.waves_per_eu() == 2 && fly.stage_rhs() &&
               fly.preload_lds_fragments() && !fly.single_buffer_lds() &&
               fly.local_split_k();
      });
  ASSERT_NE(candidate, configs.end());
  EXPECT_THAT(backend_.ApplyConfig(*fusion, **candidate), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 16 && fly.block_n() == 16 &&
               fly.block_k() == 256 && fly.num_warps() == 4 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               !fly.single_buffer_lds() && fly.local_split_k();
      }));
}

TEST_F(FlyBackendTest, SupportsSplitGemvKAndOutputTails) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_split_gemv_k_and_output_tails

gemv {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{1,0} parameter(1)
  ROOT dot = bf16[1,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{1,0} parameter(1)
  ROOT fusion = bf16[1,111]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_n() == 128 && fly.num_warps() == 8 &&
               fly.gemv_split_k();
      }));
}

TEST_F(FlyBackendTest, SupportsGemvColumnEpilogue) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_gemv_column_epilogue

gemv {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{1,0} parameter(1)
  bias = bf16[111]{0} parameter(2)
  dot = f32[1,111]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  narrowed = bf16[1,111]{1,0} convert(dot)
  broadcast = bf16[1,111]{1,0} broadcast(bias), dimensions={1}
  ROOT add = bf16[1,111]{1,0} add(narrowed, broadcast)
}

ENTRY main {
  lhs = bf16[1,127]{1,0} parameter(0)
  rhs = bf16[127,111]{1,0} parameter(1)
  bias = bf16[111]{0} parameter(2)
  ROOT fusion = bf16[1,111]{1,0} fusion(lhs, rhs, bias), kind=kCustom,
      calls=gemv,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, OffersWideDirectToVgprForBothOutputTails) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_wide_both_output_tails

gemm {
  lhs = bf16[513,1024]{1,0} parameter(0)
  rhs = bf16[1024,11007]{0,1} parameter(1)
  ROOT dot = bf16[513,11007]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[513,1024]{1,0} parameter(0)
  rhs = bf16[1024,11007]{0,1} parameter(1)
  ROOT fusion = bf16[513,11007]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 256 && fly.block_n() == 224 &&
               fly.block_k() == 64 && fly.num_warps() == 4 && fly.stage_rhs() &&
               fly.preload_lds_fragments() && fly.direct_to_vgpr();
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 256 && fly.block_n() == 128 &&
               fly.block_k() == 64 && fly.num_warps() == 8 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               fly.single_buffer_lds() && fly.workgroup_mapping_n() == 8;
      }));
}

TEST_F(FlyFusionBackendTest, ReplacesRewrittenTritonSoftmaxWithF16Tail) {
  constexpr absl::string_view kHlo = R"(
HloModule rewritten_triton_softmax

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

softmax {
  p0 = f16[64,125]{1,0} parameter(0)
  converted = f32[64,125]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[64]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[64,125]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[64,125]{1,0} subtract(converted, broadcast_max)
  exponential = f32[64,125]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[64,125]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[64,125]{1,0} divide(exponential, broadcast_sum)
  ROOT result = f16[64,125]{1,0} convert(normalized)
}

ENTRY main {
  p0 = f16[64,125]{1,0} parameter(0)
  ROOT fusion = f16[64,125]{1,0} fusion(p0), kind=kCustom,
    calls=softmax,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 5);
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> retuned_configs,
      backend_.GetSupportedConfigs(*fusion));
  EXPECT_FALSE(retuned_configs.empty());
}

TEST_F(FlyFusionBackendTest, TunesNativeDependentLayerNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_dependent_layer_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

layer_norm {
  p0 = bf16[256,4096]{1,0} parameter(0)
  converted = f32[256,4096]{1,0} convert(p0)
  zero = f32[] constant(0)
  sum = f32[256]{0} reduce(converted, zero), dimensions={1}, to_apply=add
  reciprocal = f32[] constant(0.000244140625)
  reciprocals = f32[256]{0} broadcast(reciprocal), dimensions={}
  mean = f32[256]{0} multiply(sum, reciprocals)
  means = f32[256,4096]{1,0} broadcast(mean), dimensions={0}
  centered = f32[256,4096]{1,0} subtract(converted, means)
  squared = f32[256,4096]{1,0} multiply(centered, centered)
  square_sum = f32[256]{0} reduce(squared, zero), dimensions={1}, to_apply=add
  variance = f32[256]{0} multiply(square_sum, reciprocals)
  epsilon = f32[] constant(1e-5)
  epsilons = f32[256]{0} broadcast(epsilon), dimensions={}
  variance_epsilon = f32[256]{0} add(variance, epsilons)
  reciprocal_stddev = f32[256]{0} rsqrt(variance_epsilon)
  scales = f32[256,4096]{1,0} broadcast(reciprocal_stddev), dimensions={0}
  normalized = f32[256,4096]{1,0} multiply(centered, scales)
  ROOT result = bf16[256,4096]{1,0} convert(normalized)
}

ENTRY main {
  p0 = bf16[256,4096]{1,0} parameter(0)
  ROOT fusion = bf16[256,4096]{1,0} fusion(p0), kind=kCustom,
    calls=layer_norm,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","4096"]}],"num_warps":"4","num_ctas":"1","num_stages":"1"}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 5);
  std::vector<int64_t> num_warps;
  for (const std::unique_ptr<BackendConfig>& config : configs) {
    num_warps.push_back(config->block_level().num_warps());
    EXPECT_THAT(config->block_level().output_tiles(0).sizes(),
                ::testing::ElementsAre(1, 4096));
  }
  EXPECT_THAT(num_warps, ::testing::ElementsAre(1, 2, 4, 8, 16));
}

TEST_F(FlyFusionBackendTest, ReplacesGenericTritonElementwiseFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule generic_triton_elementwise

elementwise {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  add = bf16[128,192]{1,0} add(p0, p1)
  ROOT result = bf16[128,192]{1,0} maximum(add, p0)
}

ENTRY main {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  ROOT fusion = bf16[128,192]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1","128"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 64;
  }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 128;
  }));
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
  EXPECT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .vector_size_bits(),
            64);
}

TEST_F(FlyFusionBackendTest, TunesNativeIndexedConcatenate) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_indexed_concatenate

concatenate {
  p0 = bf16[4097]{0} parameter(0)
  p1 = bf16[4095]{0} parameter(1)
  absolute = bf16[4097]{0} abs(p0)
  negated = bf16[4095]{0} negate(p1)
  ROOT result = bf16[8192]{0} concatenate(absolute, negated), dimensions={0}
}

ENTRY main {
  p0 = bf16[4097]{0} parameter(0)
  p1 = bf16[4095]{0} parameter(1)
  ROOT fusion = bf16[8192]{0} fusion(p0, p1), kind=kCustom,
    calls=concatenate,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 64;
  }));
  EXPECT_EQ(configs.front()->block_level().num_warps(), 8);
  EXPECT_EQ(configs.front()->block_level().output_tiles(0).sizes(0), 4);
}

TEST_F(FlyFusionBackendTest, TunesNativeRowGather) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_row_gather

elementwise {
  values = bf16[127,259]{1,0} parameter(0)
  indices = s32[191,1]{1,0} parameter(1)
  gathered = bf16[191,1,259]{2,1,0} gather(values, indices),
    offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0},
    index_vector_dim=1, slice_sizes={1,259}
  ROOT result = bf16[191,1,259]{2,1,0} negate(gathered)
}

ENTRY main {
  values = bf16[127,259]{1,0} parameter(0)
  indices = s32[191,1]{1,0} parameter(1)
  ROOT fusion = bf16[191,1,259]{2,1,0} fusion(values, indices),
    kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 64;
  }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 128;
  }));
  EXPECT_EQ(configs.front()->block_level().num_warps(), 8);
  EXPECT_EQ(configs.front()->block_level().output_tiles(0).sizes(0), 8);
  EXPECT_EQ(configs.front()->block_level().vector_size_bits(), 128);
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, TunesNativeOverwriteRowScatter) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_overwrite_row_scatter

assign {
  old = bf16[] parameter(0)
  ROOT update = bf16[] parameter(1)
}

elementwise {
  base = bf16[257,259]{1,0} parameter(0)
  indices = s32[191,1]{1,0} parameter(1)
  updates = bf16[191,1,259]{2,1,0} parameter(2)
  ROOT scatter = bf16[257,259]{1,0} scatter(base, indices, updates),
    update_window_dims={1,2}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1,
    unique_indices=true, to_apply=assign
}

ENTRY main {
  base = bf16[257,259]{1,0} parameter(0)
  indices = s32[191,1]{1,0} parameter(1)
  updates = bf16[191,1,259]{2,1,0} parameter(2)
  ROOT fusion = bf16[257,259]{1,0} fusion(base, indices, updates),
    kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 64;
  }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 128;
  }));
  EXPECT_EQ(configs.front()->block_level().num_warps(), 8);
  EXPECT_EQ(configs.front()->block_level().output_tiles(0).sizes(0), 1);
  EXPECT_EQ(configs.front()->block_level().vector_size_bits(), 128);
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, TunesNativeAtomicOverwriteRowScatter) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_atomic_overwrite_row_scatter

assign {
  old = bf16[] parameter(0)
  ROOT update = bf16[] parameter(1)
}

elementwise {
  base = bf16[257,259]{1,0} parameter(0)
  indices = s32[191,1]{1,0} parameter(1)
  updates = bf16[191,1,259]{2,1,0} parameter(2)
  ROOT scatter = bf16[257,259]{1,0} scatter(base, indices, updates),
    update_window_dims={1,2}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1,
    unique_indices=false, to_apply=assign
}

ENTRY main {
  base = bf16[257,259]{1,0} parameter(0)
  indices = s32[191,1]{1,0} parameter(1)
  updates = bf16[191,1,259]{2,1,0} parameter(2)
  ROOT fusion = bf16[257,259]{1,0} fusion(base, indices, updates),
    kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 32;
  }));
  EXPECT_EQ(configs.front()->block_level().num_warps(), 8);
  EXPECT_EQ(configs.front()->block_level().output_tiles(0).sizes(0), 8);
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, TunesNativeTrailingBroadcast) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_trailing_broadcast

elementwise {
  p0 = bf16[4096,8193]{1,0} parameter(0)
  row = bf16[8193]{0} parameter(1)
  rows = bf16[4096,8193]{1,0} broadcast(row), dimensions={1}
  ROOT result = bf16[4096,8193]{1,0} add(p0, rows)
}

ENTRY main {
  p0 = bf16[4096,8193]{1,0} parameter(0)
  row = bf16[8193]{0} parameter(1)
  ROOT fusion = bf16[4096,8193]{1,0} fusion(p0, row), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 64;
  }));
  EXPECT_EQ(configs.front()->block_level().num_warps(), 8);
  EXPECT_EQ(configs.front()->block_level().output_tiles(0).sizes(0), 4);
}

TEST_F(FlyFusionBackendTest, TunesNativeContiguousSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_contiguous_slice

elementwise {
  p0 = bf16[4098,8193]{1,0} parameter(0)
  ROOT result = bf16[4096,8193]{1,0} slice(p0),
    slice={[1:4097], [0:8193]}
}

ENTRY main {
  p0 = bf16[4098,8193]{1,0} parameter(0)
  ROOT fusion = bf16[4096,8193]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<int64_t, 4> kWavesPerEu = {0, 1, 2, 4};
  for (int64_t index = 0; index < 4; ++index) {
    EXPECT_EQ(configs[index]->block_level().vector_size_bits(), 128);
    EXPECT_EQ(configs[index]->block_level().waves_per_eu(), kWavesPerEu[index]);
  }
  EXPECT_EQ(configs[4]->block_level().vector_size_bits(), 64);
  EXPECT_EQ(configs[5]->block_level().vector_size_bits(), 64);
  EXPECT_EQ(configs[6]->block_level().vector_size_bits(), 128);
  EXPECT_EQ(configs[7]->block_level().vector_size_bits(), 128);
  EXPECT_EQ(configs.front()->block_level().num_warps(), 8);
  EXPECT_EQ(configs.front()->block_level().output_tiles(0).sizes(0), 1);
}

TEST_F(FlyFusionBackendTest, TunesNativeGappedRectangularSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_gapped_slice

elementwise {
  p0 = bf16[4096,8194]{1,0} parameter(0)
  ROOT result = bf16[4096,8192]{1,0} slice(p0),
    slice={[0:4096], [1:8193]}
}

ENTRY main {
  p0 = bf16[4096,8194]{1,0} parameter(0)
  ROOT fusion = bf16[4096,8192]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<int64_t, 4> kWavesPerEu = {0, 1, 2, 4};
  for (int64_t index = 0; index < 4; ++index) {
    EXPECT_EQ(configs[index]->block_level().vector_size_bits(), 128);
    EXPECT_EQ(configs[index]->block_level().waves_per_eu(), kWavesPerEu[index]);
  }
}

TEST_F(FlyFusionBackendTest, TunesNativeDynamicRectangularSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_dynamic_slice

elementwise {
  p0 = bf16[4098,8194]{1,0} parameter(0)
  row = s32[] parameter(1)
  column = s32[] parameter(2)
  ROOT result = bf16[4096,8192]{1,0}
    dynamic-slice(p0, row, column), dynamic_slice_sizes={4096,8192}
}

ENTRY main {
  p0 = bf16[4098,8194]{1,0} parameter(0)
  row = s32[] parameter(1)
  column = s32[] parameter(2)
  ROOT fusion = bf16[4096,8192]{1,0} fusion(p0, row, column), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<int64_t, 4> kWavesPerEu = {0, 1, 2, 4};
  for (int64_t index = 0; index < 4; ++index) {
    EXPECT_EQ(configs[index]->block_level().vector_size_bits(), 128);
    EXPECT_EQ(configs[index]->block_level().waves_per_eu(), kWavesPerEu[index]);
  }
}

TEST_F(FlyFusionBackendTest, TunesNativeDynamicRectangularUpdate) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_dynamic_update_slice

elementwise {
  input = bf16[4096,8192]{1,0} parameter(0)
  update = bf16[4096,8190]{1,0} parameter(1)
  row = s32[] parameter(2)
  column = s32[] parameter(3)
  ROOT result = bf16[4096,8192]{1,0} dynamic-update-slice(
    input, update, row, column)
}

ENTRY main {
  input = bf16[4096,8192]{1,0} parameter(0)
  update = bf16[4096,8190]{1,0} parameter(1)
  row = s32[] parameter(2)
  column = s32[] parameter(3)
  ROOT fusion = bf16[4096,8192]{1,0}
    fusion(input, update, row, column), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<std::array<int64_t, 3>, 8> kExpected = {{
      {2, 4, 128},
      {1, 8, 128},
      {1, 1, 64},
      {1, 4, 128},
      {1, 1, 128},
      {4, 8, 128},
      {2, 2, 128},
      {4, 4, 128},
  }};
  for (int64_t index = 0; index < kExpected.size(); ++index) {
    const auto& expected = kExpected[index];
    EXPECT_EQ(configs[index]->block_level().output_tiles(0).sizes(0),
              expected[0]);
    EXPECT_EQ(configs[index]->block_level().num_warps(), expected[1]);
    EXPECT_EQ(configs[index]->block_level().vector_size_bits(), expected[2]);
  }
}

TEST_F(FlyFusionBackendTest, TunesNativeReduceWindow) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_reduce_window

maximum {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT result = bf16[] maximum(lhs, rhs)
}

elementwise {
  p0 = bf16[4096,8192]{1,0} parameter(0)
  negative_infinity = bf16[] constant(-inf)
  ROOT result = bf16[4096,8192]{1,0}
    reduce-window(p0, negative_infinity),
    window={size=1x3 pad=0_0x1_1}, to_apply=maximum
}

ENTRY main {
  p0 = bf16[4096,8192]{1,0} parameter(0)
  ROOT fusion = bf16[4096,8192]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<std::array<int64_t, 3>, 8> kExpected = {{
      {2, 8, 128},
      {2, 4, 128},
      {4, 4, 128},
      {1, 8, 128},
      {4, 8, 128},
      {1, 4, 128},
      {4, 8, 64},
      {8, 4, 64},
  }};
  for (int64_t index = 0; index < kExpected.size(); ++index) {
    const auto& expected = kExpected[index];
    EXPECT_EQ(configs[index]->block_level().output_tiles(0).sizes(0),
              expected[0]);
    EXPECT_EQ(configs[index]->block_level().num_warps(), expected[1]);
    EXPECT_EQ(configs[index]->block_level().vector_size_bits(), expected[2]);
  }
}

TEST_F(FlyFusionBackendTest, TunesNativeFlatReverse) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_flat_reverse

elementwise {
  p0 = bf16[4096,8193]{1,0} parameter(0)
  ROOT result = bf16[4096,8193]{1,0} reverse(p0), dimensions={0,1}
}

ENTRY main {
  p0 = bf16[4096,8193]{1,0} parameter(0)
  ROOT fusion = bf16[4096,8193]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<int64_t, 4> kWarps = {8, 4, 2, 1};
  for (int64_t pair = 0; pair < 4; ++pair) {
    EXPECT_EQ(configs[2 * pair]->block_level().num_warps(), kWarps[pair]);
    EXPECT_EQ(configs[2 * pair]->block_level().vector_size_bits(), 128);
    EXPECT_EQ(configs[2 * pair]->block_level().output_tiles(0).sizes(0), 1);
    EXPECT_EQ(configs[2 * pair + 1]->block_level().num_warps(), kWarps[pair]);
    EXPECT_EQ(configs[2 * pair + 1]->block_level().vector_size_bits(), 64);
    EXPECT_EQ(configs[2 * pair + 1]->block_level().output_tiles(0).sizes(0), 2);
  }
}

TEST_F(FlyFusionBackendTest, TunesNativeFlatEdgePad) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_flat_edge_pad

elementwise {
  p0 = bf16[4096,8193]{1,0} parameter(0)
  zero = bf16[] constant(0)
  ROOT result = bf16[4098,8193]{1,0} pad(p0, zero),
    padding=1_1x0_0
}

ENTRY main {
  p0 = bf16[4096,8193]{1,0} parameter(0)
  ROOT fusion = bf16[4098,8193]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<int64_t, 4> kWarps = {8, 4, 2, 1};
  for (int64_t pair = 0; pair < 4; ++pair) {
    EXPECT_EQ(configs[2 * pair]->block_level().num_warps(), kWarps[pair]);
    EXPECT_EQ(configs[2 * pair]->block_level().vector_size_bits(), 128);
    EXPECT_EQ(configs[2 * pair]->block_level().output_tiles(0).sizes(0), 1);
    EXPECT_EQ(configs[2 * pair + 1]->block_level().num_warps(), kWarps[pair]);
    EXPECT_EQ(configs[2 * pair + 1]->block_level().vector_size_bits(), 64);
    EXPECT_EQ(configs[2 * pair + 1]->block_level().output_tiles(0).sizes(0), 2);
  }
}

TEST_F(FlyFusionBackendTest, TunesNativeInteriorPad) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_interior_pad

elementwise {
  p0 = bf16[4096,4096]{1,0} parameter(0)
  zero = bf16[] constant(0)
  ROOT result = bf16[4096,8192]{1,0} pad(p0, zero),
    padding=0_0x0_1_1
}

ENTRY main {
  p0 = bf16[4096,4096]{1,0} parameter(0)
  ROOT fusion = bf16[4096,8192]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<int64_t, 4> kWarps = {8, 4, 2, 1};
  for (int64_t pair = 0; pair < 4; ++pair) {
    EXPECT_EQ(configs[2 * pair]->block_level().num_warps(), kWarps[pair]);
    EXPECT_EQ(configs[2 * pair]->block_level().vector_size_bits(), 128);
    EXPECT_EQ(configs[2 * pair]->block_level().output_tiles(0).sizes(0), 1);
    EXPECT_EQ(configs[2 * pair + 1]->block_level().num_warps(), kWarps[pair]);
    EXPECT_EQ(configs[2 * pair + 1]->block_level().vector_size_bits(), 64);
    EXPECT_EQ(configs[2 * pair + 1]->block_level().output_tiles(0).sizes(0), 2);
  }
}

TEST_F(FlyFusionBackendTest, TunesSmallSplitKResidualWithMixedBuffers) {
  constexpr absl::string_view kHlo = R"(
HloModule small_split_k_residual_autotune

add_reduce {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

residual {
  base = bf16[2,128,1024]{2,1,0} parameter(0)
  base_f32 = f32[2,128,1024]{2,1,0} convert(base)
  partial4 = f32[4,256,1024]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  sum4 = f32[256,1024]{1,0} reduce(partial4, zero), dimensions={0},
      to_apply=add_reduce
  rounded4 = bf16[256,1024]{1,0} convert(sum4)
  view4 = bf16[2,128,1024]{2,1,0} bitcast(rounded4)
  value4 = f32[2,128,1024]{2,1,0} convert(view4)
  partial2 = f32[2,256,1024]{2,1,0} parameter(2)
  sum2 = f32[256,1024]{1,0} reduce(partial2, zero), dimensions={0},
      to_apply=add_reduce
  rounded2 = bf16[256,1024]{1,0} convert(sum2)
  view2 = bf16[2,128,1024]{2,1,0} bitcast(rounded2)
  value2 = f32[2,128,1024]{2,1,0} convert(view2)
  first = f32[2,128,1024]{2,1,0} add(base_f32, value2)
  total = f32[2,128,1024]{2,1,0} add(first, value4)
  ROOT result = bf16[2,128,1024]{2,1,0} convert(total)
}

ENTRY main {
  base = bf16[2,128,1024]{2,1,0} parameter(0)
  partial4 = f32[4,256,1024]{2,1,0} parameter(1)
  partial2 = f32[2,256,1024]{2,1,0} parameter(2)
  ROOT fusion = bf16[2,128,1024]{2,1,0}
      fusion(base, partial4, partial2), kind=kCustom, calls=residual,
      backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 64;
  }));
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, LimitsAndRanksNativeElementwiseConfigs) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_elementwise_top_k

elementwise {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  ROOT result = bf16[128,192]{1,0} add(p0, p1)
}

ENTRY main {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  ROOT fusion = bf16[128,192]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1,
      "vector_size_bits":"64"}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(3);
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 3);
  EXPECT_EQ(configs[0]->block_level().output_tiles(0).sizes(0), 1);
  EXPECT_EQ(configs[0]->block_level().num_warps(), 2);
  EXPECT_EQ(configs[0]->block_level().vector_size_bits(), 64);
  EXPECT_EQ(configs[1]->block_level().output_tiles(0).sizes(0), 1);
  EXPECT_EQ(configs[1]->block_level().num_warps(), 2);
  EXPECT_EQ(configs[1]->block_level().vector_size_bits(), 128);
  EXPECT_EQ(configs[2]->block_level().output_tiles(0).sizes(0), 2);
  EXPECT_EQ(configs[2]->block_level().num_warps(), 4);
  EXPECT_EQ(configs[2]->block_level().vector_size_bits(), 64);
}

TEST_F(FlyFusionBackendTest, RanksShapeChangingPhysicalViewConfigs) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_physical_view_top_k

elementwise {
  p0 = s32[4194304,16]{1,0} parameter(0)
  view = s32[16,4194304]{0,1} bitcast(p0)
  ROOT result = s32[16,4194304]{0,1} not(view)
}

ENTRY main {
  p0 = s32[4194304,16]{1,0} parameter(0)
  ROOT fusion = s32[16,4194304]{0,1} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1,
      "vector_size_bits":"64"}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  constexpr std::array<std::array<int64_t, 3>, 8> kExpected = {{
      {1, 2, 64},
      {1, 8, 128},
      {2, 1, 128},
      {2, 4, 128},
      {1, 2, 128},
      {1, 4, 128},
      {2, 2, 64},
      {2, 2, 128},
  }};
  for (int64_t index = 0; index < kExpected.size(); ++index) {
    const auto& expected = kExpected[index];
    EXPECT_EQ(configs[index]->block_level().output_tiles(0).sizes(0),
              expected[0]);
    EXPECT_EQ(configs[index]->block_level().num_warps(), expected[1]);
    EXPECT_EQ(configs[index]->block_level().vector_size_bits(), expected[2]);
  }
}

TEST_F(FlyFusionBackendTest, KeepsWideTypeChangingBitcastsVectorized) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_type_bitcast_top_k

elementwise {
  p0 = s32[4194304]{0} parameter(0)
  bytes = s8[4194304,4]{1,0} bitcast-convert(p0)
  ROOT result = s8[4194304,4]{1,0} not(bytes)
}

ENTRY main {
  p0 = s32[4194304]{0} parameter(0)
  ROOT fusion = s8[4194304,4]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1,
      "vector_size_bits":"64"}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 8);
  EXPECT_EQ(configs[0]->block_level().vector_size_bits(), 64);
  EXPECT_EQ(configs[1]->block_level().vector_size_bits(), 128);
  EXPECT_EQ(configs[2]->block_level().vector_size_bits(), 128);
  EXPECT_EQ(configs[3]->block_level().vector_size_bits(), 128);
}

TEST_F(FlyFusionBackendTest, RanksScalarTailElementwiseConfigFirst) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_elementwise_tail_top_k

elementwise {
  p0 = bf16[127,65]{1,0} parameter(0)
  ROOT result = bf16[127,65]{1,0} negate(p0)
}

ENTRY main {
  p0 = bf16[127,65]{1,0} parameter(0)
  ROOT fusion = bf16[127,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"64"}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 1);
  EXPECT_EQ(configs[0]->block_level().output_tiles(0).sizes(0), 1);
  EXPECT_EQ(configs[0]->block_level().num_warps(), 2);
  EXPECT_EQ(configs[0]->block_level().vector_size_bits(), 64);
}

TEST_F(FlyFusionBackendTest, RanksCooperativeStridedReductionConfigFirst) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_cooperative_strided_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = f32[64,128,256]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT result = f32[64,256]{1,0} reduce(p0, zero), dimensions={1},
      to_apply=add
}

ENTRY main {
  p0 = f32[64,128,256]{2,1,0} parameter(0)
  ROOT fusion = f32[64,256]{1,0} fusion(p0), kind=kCustom,
    calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_EQ(configs[0]->block_level().output_tiles(0).sizes(0), 1);
  EXPECT_EQ(configs[0]->block_level().num_warps(), 4);
  EXPECT_EQ(configs[0]->block_level().vector_size_bits(), 128);
}

TEST_F(FlyFusionBackendTest, RanksArbitraryBroadcastReuseConfigFirst) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_arbitrary_broadcast

body {
  p0 = bf16[256,128,256]{2,1,0} parameter(0)
  plane = bf16[256,256]{1,0} parameter(1)
  absolute = bf16[256,256]{1,0} abs(plane)
  planes = bf16[256,128,256]{2,1,0} broadcast(absolute), dimensions={0,2}
  ROOT result = bf16[256,128,256]{2,1,0} add(p0, planes)
}

ENTRY main {
  p0 = bf16[256,128,256]{2,1,0} parameter(0)
  plane = bf16[256,256]{1,0} parameter(1)
  ROOT fusion = bf16[256,128,256]{2,1,0} fusion(p0, plane), kind=kCustom,
    calls=body,
    backend_config={"fusion_backend_config":{"kind":"__fly"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_EQ(configs[0]->block_level().output_tiles(0).sizes(0), 8);
  EXPECT_EQ(configs[0]->block_level().num_warps(), 8);
  EXPECT_EQ(configs[0]->block_level().vector_size_bits(), 64);
  EXPECT_EQ(configs[0]->block_level().waves_per_eu(), 0);
}

TEST_F(FlyFusionBackendTest, RanksMiddleConcatenateConfigFirst) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_middle_concatenate

body {
  p0 = bf16[256,64,256]{2,1,0} parameter(0)
  p1 = bf16[256,64,256]{2,1,0} parameter(1)
  absolute = bf16[256,64,256]{2,1,0} abs(p0)
  negated = bf16[256,64,256]{2,1,0} negate(p1)
  ROOT result = bf16[256,128,256]{2,1,0} concatenate(absolute, negated),
    dimensions={1}
}

ENTRY main {
  p0 = bf16[256,64,256]{2,1,0} parameter(0)
  p1 = bf16[256,64,256]{2,1,0} parameter(1)
  ROOT fusion = bf16[256,128,256]{2,1,0} fusion(p0, p1), kind=kCustom,
    calls=body,
    backend_config={"fusion_backend_config":{"kind":"__fly"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_EQ(configs[0]->block_level().output_tiles(0).sizes(0), 4);
  EXPECT_EQ(configs[0]->block_level().num_warps(), 4);
  EXPECT_EQ(configs[0]->block_level().vector_size_bits(), 128);
}

TEST_F(FlyFusionBackendTest, RanksElementwiseConfigByDtype) {
  constexpr absl::string_view kF16Hlo = R"(
HloModule native_fly_f16_elementwise_top_k

elementwise {
  p0 = f16[128,192]{1,0} parameter(0)
  ROOT result = f16[128,192]{1,0} negate(p0)
}

ENTRY main {
  p0 = f16[128,192]{1,0} parameter(0)
  ROOT fusion = f16[128,192]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"64"}}}
})";
  constexpr absl::string_view kF32Hlo = R"(
HloModule native_fly_f32_elementwise_top_k

elementwise {
  p0 = f32[128,192]{1,0} parameter(0)
  ROOT result = f32[128,192]{1,0} negate(p0)
}

ENTRY main {
  p0 = f32[128,192]{1,0} parameter(0)
  ROOT fusion = f32[128,192]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1,
      "vector_size_bits":"64"}}}
})";
  for (absl::string_view hlo : {kF16Hlo, kF32Hlo}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                         backend_.GetSupportedConfigs(
                             *module->entry_computation()->root_instruction()));

    ASSERT_EQ(configs.size(), 1);
    EXPECT_EQ(configs[0]->block_level().output_tiles(0).sizes(0), 1);
    EXPECT_EQ(configs[0]->block_level().num_warps(), 2);
    EXPECT_EQ(configs[0]->block_level().vector_size_bits(), 64);
  }
}

TEST_F(FlyFusionBackendTest, OffersLegalNativeFp8ConversionConfigs) {
  constexpr absl::string_view kF32ToFp8Hlo = R"(
HloModule native_fly_f32_to_fp8_top_k

elementwise {
  p0 = f32[67108864]{0} parameter(0)
  ROOT result = f8e5m2fnuz[67108864]{0} convert(p0)
}

ENTRY main {
  p0 = f32[67108864]{0} parameter(0)
  ROOT fusion = f8e5m2fnuz[67108864]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1,
      "vector_size_bits":"32"}}}
})";
  constexpr absl::string_view kFp8ToF32Hlo = R"(
HloModule native_fly_fp8_to_f32_top_k

elementwise {
  p0 = f8e5m2fnuz[67108864]{0} parameter(0)
  ROOT result = f32[67108864]{0} convert(p0)
}

ENTRY main {
  p0 = f8e5m2fnuz[67108864]{0} parameter(0)
  ROOT fusion = f32[67108864]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1,
      "vector_size_bits":"64"}}}
})";

  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> f32_to_fp8_module,
                       ParseAndReturnVerifiedModule(kF32ToFp8Hlo));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> f32_to_fp8_configs,
      backend_.GetSupportedConfigs(
          *f32_to_fp8_module->entry_computation()->root_instruction()));

  ASSERT_EQ(f32_to_fp8_configs.size(), 8);
  EXPECT_TRUE(std::all_of(
      f32_to_fp8_configs.begin(), f32_to_fp8_configs.end(),
      [](const auto& c) { return c->block_level().vector_size_bits() == 32; }));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> fp8_to_f32_module,
                       ParseAndReturnVerifiedModule(kFp8ToF32Hlo));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> fp8_to_f32_configs,
      backend_.GetSupportedConfigs(
          *fp8_to_f32_module->entry_computation()->root_instruction()));
  ASSERT_EQ(fp8_to_f32_configs.size(), 8);
  EXPECT_TRUE(std::any_of(
      fp8_to_f32_configs.begin(), fp8_to_f32_configs.end(),
      [](const auto& c) { return c->block_level().vector_size_bits() == 64; }));
  EXPECT_TRUE(std::any_of(fp8_to_f32_configs.begin(), fp8_to_f32_configs.end(),
                          [](const auto& c) {
                            return c->block_level().vector_size_bits() == 128;
                          }));
}

TEST_F(FlyFusionBackendTest, OffersNativePackedS4ConversionConfigs) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_packed_s4_top_k

elementwise {
  p0 = s4[67108864]{0:E(4)} parameter(0)
  widened = s8[67108864]{0} convert(p0)
  ROOT result = bf16[67108864]{0} convert(widened)
}

ENTRY main {
  p0 = s4[67108864]{0:E(4)} parameter(0)
  ROOT fusion = bf16[67108864]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1,
      "vector_size_bits":"64"}}}
})";

  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  ASSERT_EQ(configs.size(), 8);
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 64;
  }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 128;
  }));
}

TEST_F(FlyFusionBackendTest, OffersScalarizedWideInputPackedS4OutputConfigs) {
  constexpr absl::string_view kHlo = R"(
HloModule native_fly_packed_s4_output_top_k

elementwise {
  p0 = s64[65]{0} parameter(0)
  ROOT result = s4[65]{0:E(4)} convert(p0)
}

ENTRY main {
  p0 = s64[65]{0} parameter(0)
  ROOT fusion = s4[65]{0:E(4)} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1,
      "vector_size_bits":"16"}}}
})";

  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 16;
  }));
}

TEST_F(FlyFusionBackendTest, ReplacesMultiOutputTritonElementwiseFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule multi_output_triton_elementwise

elementwise {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  sum = bf16[128,192]{1,0} add(p0, p1)
  product = bf16[128,192]{1,0} multiply(sum, p0)
  ROOT tuple = (bf16[128,192]{1,0}, bf16[128,192]{1,0})
    tuple(sum, product)
}

ENTRY main {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  ROOT fusion = (bf16[128,192]{1,0}, bf16[128,192]{1,0})
    fusion(p0, p1), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1","128"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, LimitsVectorWidthForMixedTypeMultiOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule mixed_type_multi_output_vector_width

elementwise {
  p0 = bf16[128,192]{1,0} parameter(0)
  widened = f64[128,192]{1,0} convert(p0)
  ROOT tuple = (bf16[128,192]{1,0}, f64[128,192]{1,0})
    tuple(p0, widened)
}

ENTRY main {
  p0 = bf16[128,192]{1,0} parameter(0)
  ROOT fusion = (bf16[128,192]{1,0}, f64[128,192]{1,0})
    fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"2","num_ctas":1,"num_stages":1,
      "vector_size_bits":"32"}}}
})";

  debug_options_.set_xla_gpu_fusion_autotune_top_k_configs(8);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  ASSERT_EQ(configs.size(), 8);
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().vector_size_bits() == 32;
  }));
}

TEST_F(FlyFusionBackendTest, RetilesGenericTritonReductionFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule generic_triton_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = f32[64,256]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT result = f32[64]{0} reduce(p0, zero), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[64,256]{1,0} parameter(0)
  ROOT fusion = f32[64]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 10);
  for (int64_t vector_size_bits : {64, 128}) {
    for (int64_t num_warps : {1, 2, 4, 8, 16}) {
      EXPECT_TRUE(
          std::any_of(configs.begin(), configs.end(), [&](const auto& config) {
            return config->block_level().vector_size_bits() ==
                       vector_size_bits &&
                   config->block_level().num_warps() == num_warps;
          }));
    }
  }
}

TEST_F(FlyFusionBackendTest, TunesDynamicInitRowReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule dynamic_init_triton_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = f32[64,256]{1,0} parameter(0)
  init = f32[] parameter(1)
  ROOT result = f32[64]{0} reduce(p0, init), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f32[64,256]{1,0} parameter(0)
  init = f32[] parameter(1)
  ROOT fusion = f32[64]{0} fusion(p0, init), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  EXPECT_EQ(configs.size(), 10);
}

TEST_F(FlyFusionBackendTest, TunesRowReductionWithExtraOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule row_reduction_extra_output_autotune

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(lhs, rhs)
}

reduction {
  p0 = f32[128,512]{1,0} parameter(0)
  absolute = f32[128,512]{1,0} abs(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[128]{0} reduce(absolute, minus_inf), dimensions={1},
    to_apply=maximum
  ROOT result = (f32[128]{0}, f32[128,512]{1,0})
    tuple(row_max, absolute)
}

ENTRY main {
  p0 = f32[128,512]{1,0} parameter(0)
  ROOT fusion = (f32[128]{0}, f32[128,512]{1,0})
    fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64"]},{"sizes":["64","512"]}],
        "num_warps":"2","num_ctas":1,"num_stages":1}}}
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 10);
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().output_tiles_size() == 1 &&
           c->block_level().output_tiles(0).sizes(0) == 1;
  }));
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, TunesReductionWithDependentRowwiseOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule reduction_dependent_rowwise_output_autotune

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = f32[128,512]{1,0} parameter(0)
  squared = f32[128,512]{1,0} multiply(p0, p0)
  zero = f32[] constant(0)
  row_sum = f32[128]{0} reduce(squared, zero), dimensions={1}, to_apply=add
  epsilon = f32[] constant(1e-6)
  epsilons = f32[128]{0} broadcast(epsilon), dimensions={}
  variance = f32[128]{0} add(row_sum, epsilons)
  scale = f32[128]{0} rsqrt(variance)
  scales = f32[128,512]{1,0} broadcast(scale), dimensions={0}
  normalized = f32[128,512]{1,0} multiply(p0, scales)
  negative_sum = f32[128]{0} negate(row_sum)
  ROOT result = (f32[128,512]{1,0}, f32[128]{0})
    tuple(normalized, negative_sum)
}

ENTRY main {
  p0 = f32[128,512]{1,0} parameter(0)
  ROOT fusion = (f32[128,512]{1,0}, f32[128]{0})
    fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64","512"]},{"sizes":["64"]}],
        "num_warps":"2","num_ctas":1,"num_stages":1}}}
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 10);
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().output_tiles_size() == 1 &&
           c->block_level().output_tiles(0).sizes(0) == 1;
  }));
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, TunesMultipleRowReductions) {
  constexpr absl::string_view kHlo = R"(
HloModule multiple_row_reductions_autotune

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(lhs, rhs)
}

reduction {
  p0 = f32[128,512]{1,0} parameter(0)
  zero = f32[] constant(0)
  row_sum = f32[128]{0} reduce(p0, zero), dimensions={1}, to_apply=add
  squared = f32[128,512]{1,0} multiply(p0, p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[128]{0} reduce(squared, minus_inf), dimensions={1},
    to_apply=maximum
  ROOT result = (f32[128]{0}, f32[128]{0}) tuple(row_sum, row_max)
}

ENTRY main {
  p0 = f32[128,512]{1,0} parameter(0)
  ROOT fusion = (f32[128]{0}, f32[128]{0})
    fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64"]},{"sizes":["64"]}],
        "num_warps":"2","num_ctas":1,"num_stages":1}}}
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 10);
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().output_tiles_size() == 1 &&
           c->block_level().output_tiles(0).sizes(0) == 1;
  }));
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, TunesMultipleRowReductionsWithExtraOutput) {
  constexpr absl::string_view kHlo = R"(
HloModule multiple_row_reductions_extra_output_autotune

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(lhs, rhs)
}

reduction {
  p0 = f32[128,512]{1,0} parameter(0)
  absolute = f32[128,512]{1,0} abs(p0)
  zero = f32[] constant(0)
  row_sum = f32[128]{0} reduce(p0, zero), dimensions={1}, to_apply=add
  minus_inf = f32[] constant(-inf)
  row_max = f32[128]{0} reduce(absolute, minus_inf), dimensions={1},
    to_apply=maximum
  ROOT result = (f32[128]{0}, f32[128]{0}, f32[128,512]{1,0})
    tuple(row_sum, row_max, absolute)
}

ENTRY main {
  p0 = f32[128,512]{1,0} parameter(0)
  ROOT fusion = (f32[128]{0}, f32[128]{0}, f32[128,512]{1,0})
    fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64"]},{"sizes":["64"]},
                        {"sizes":["64","512"]}],
        "num_warps":"2","num_ctas":1,"num_stages":1}}}
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 10);
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->block_level().output_tiles_size() == 1 &&
           c->block_level().output_tiles(0).sizes(0) == 1;
  }));
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, UsesWideCopiesWithTailForRaggedBf16Reduction) {
  constexpr absl::string_view kHlo = R"(
HloModule ragged_bf16_triton_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = bf16[64,259]{1,0} parameter(0)
  converted = f32[64,259]{1,0} convert(p0)
  zero = f32[] constant(0)
  ROOT result = f32[64]{0} reduce(converted, zero), dimensions={1},
    to_apply=add
}

ENTRY main {
  p0 = bf16[64,259]{1,0} parameter(0)
  ROOT fusion = f32[64]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 10);
  for (int64_t vector_size_bits : {64, 128}) {
    EXPECT_EQ(std::count_if(configs.begin(), configs.end(),
                            [&](const auto& config) {
                              return config->block_level().vector_size_bits() ==
                                     vector_size_bits;
                            }),
              5);
  }
}

TEST_F(FlyFusionBackendTest, TunesFusedRowReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fused_triton_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = bf16[64,256]{1,0} parameter(0)
  p1 = bf16[64,256]{1,0} parameter(1)
  lhs = f32[64,256]{1,0} convert(p0)
  rhs = f32[64,256]{1,0} convert(p1)
  difference = f32[64,256]{1,0} subtract(lhs, rhs)
  square = f32[64,256]{1,0} multiply(difference, difference)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(square, zero), dimensions={1}, to_apply=add
  scale = f32[] constant(0.25)
  scales = f32[64]{0} broadcast(scale), dimensions={}
  ROOT result = f32[64]{0} multiply(row_sum, scales)
}

ENTRY main {
  p0 = bf16[64,256]{1,0} parameter(0)
  p1 = bf16[64,256]{1,0} parameter(1)
  ROOT fusion = f32[64]{0} fusion(p0, p1), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  EXPECT_EQ(configs.size(), 10);
}

TEST_F(FlyFusionBackendTest, TunesPartitionedRmsNormRows) {
  constexpr absl::string_view kHlo = R"(
HloModule partitioned_rms_norm_autotune

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[2,128,1024]{2,1,0} parameter(0)
  converted = f32[2,128,1024]{2,1,0} convert(p0)
  squared = f32[2,128,1024]{2,1,0} multiply(converted, converted)
  zero = f32[] constant(0)
  row_sum = f32[2,128]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.0009765625)
  widths = f32[2,128]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,128]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,128]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,128]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,128]{1,0} rsqrt(variance)
  scales = f32[2,128,1024]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,128,1024]{2,1,0} multiply(converted, scales)
  ROOT result = bf16[2,128,1024]{2,1,0} convert(normalized)
}

ENTRY main {
  p0 = bf16[2,128,1024]{2,1,0} parameter(0)
  ROOT fusion = bf16[2,128,1024]{2,1,0} fusion(p0), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 22);
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    const BlockLevelFusionConfig& block = c->block_level();
    return block.output_tiles(0).sizes(0) == 2 &&
           block.vector_size_bits() == 128 && block.num_warps() == 2;
  }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    const BlockLevelFusionConfig& block = c->block_level();
    return block.output_tiles(0).sizes(0) == 8 &&
           block.vector_size_bits() == 32 && block.num_warps() == 2;
  }));
}

TEST_F(FlyFusionBackendTest, TunesSingletonBitcastRmsNorm) {
  debug_options_.set_xla_gpu_flydsl_replace_triton(true);
  constexpr absl::string_view kHlo = R"(
HloModule singleton_bitcast_rms_norm_autotune

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[1,1,4096]{2,1,0} parameter(0)
  weight = bf16[4096]{0} parameter(1)
  converted = f32[1,1,4096]{2,1,0} convert(p0)
  squared = f32[1,1,4096]{2,1,0} multiply(converted, converted)
  flat_square = f32[4096]{0} bitcast(squared)
  zero = f32[] constant(0)
  row_sum = f32[] reduce(flat_square, zero), dimensions={0}, to_apply=add
  row_sum_view = f32[1,1]{1,0} bitcast(row_sum)
  reciprocal_width = f32[1,1]{1,0} constant({{0.000244140625}})
  mean_square = f32[1,1]{1,0} multiply(row_sum_view, reciprocal_width)
  epsilon = f32[1,1]{1,0} constant({{1e-06}})
  variance = f32[1,1]{1,0} add(mean_square, epsilon)
  reciprocal_stddev = f32[1,1]{1,0} rsqrt(variance)
  scalar_scale = f32[] bitcast(reciprocal_stddev)
  scales = f32[1,1,4096]{2,1,0} broadcast(scalar_scale), dimensions={}
  normalized = f32[1,1,4096]{2,1,0} multiply(converted, scales)
  weight_view = bf16[1,1,4096]{2,1,0} bitcast(weight)
  weight_f32 = f32[1,1,4096]{2,1,0} convert(weight_view)
  weighted = f32[1,1,4096]{2,1,0} multiply(normalized, weight_f32)
  ROOT result = bf16[1,1,4096]{2,1,0} convert(weighted)
}

ENTRY main {
  p0 = bf16[1,1,4096]{2,1,0} parameter(0)
  weight = bf16[4096]{0} parameter(1)
  ROOT fusion = bf16[1,1,4096]{2,1,0} fusion(p0, weight), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","64"]}],
      "num_warps":"4","num_ctas":"1","num_stages":"1"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  EXPECT_FALSE(configs.empty());
}

TEST_F(FlyFusionBackendTest, TunesBitcastReductionExponentialEpilogue) {
  debug_options_.set_xla_gpu_flydsl_replace_triton(true);
  constexpr absl::string_view kHlo = R"(
HloModule bitcast_reduction_exponential_epilogue_autotune

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

attention_epilogue {
  scale = f32[1,4]{1,0} parameter(0)
  input = f32[1,1,1,4,17]{4,3,2,1,0} parameter(1)
  matrix = f32[4,17]{1,0} bitcast(input)
  zero = f32[] constant(0)
  row_sum = f32[4]{0} reduce(matrix, zero), dimensions={1}, to_apply=add
  row_sum_view = f32[1,4]{1,0} bitcast(row_sum)
  scaled = f32[1,4]{1,0} multiply(row_sum_view, scale)
  shifted = f32[1,4]{1,0} subtract(scaled, scale)
  exponential = f32[1,4]{1,0} exponential(shifted)
  ROOT result = bf16[1,4]{1,0} convert(exponential)
}

ENTRY main {
  scale = f32[1,4]{1,0} parameter(0)
  input = f32[1,1,1,4,17]{4,3,2,1,0} parameter(1)
  ROOT fusion = bf16[1,4]{1,0} fusion(scale, input), kind=kCustom,
    calls=attention_epilogue,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  EXPECT_FALSE(configs.empty());
}

TEST_F(FlyFusionBackendTest, TunesPackedQkvSliceReduction) {
  debug_options_.set_xla_gpu_flydsl_replace_triton(true);
  constexpr absl::string_view kHlo = R"(
HloModule packed_qkv_slice_reduction_autotune

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

attention {
  scale = f32[] parameter(0)
  packed = bf16[4,3,4,17]{3,2,1,0} parameter(1)
  q = bf16[4,1,4,17]{3,2,1,0} slice(packed),
      slice={[0:4], [0:1], [0:4], [0:17]}
  k = bf16[4,1,4,17]{3,2,1,0} slice(packed),
      slice={[0:4], [1:2], [0:4], [0:17]}
  q_f32 = f32[4,1,4,17]{3,2,1,0} convert(q)
  k_f32 = f32[4,1,4,17]{3,2,1,0} convert(k)
  products = f32[4,1,4,17]{3,2,1,0} multiply(q_f32, k_f32)
  matrix = f32[16,17]{1,0} bitcast(products)
  zero = f32[] constant(0)
  row_sum = f32[16]{0} reduce(matrix, zero), dimensions={1}, to_apply=add
  rows = f32[4,4]{1,0} bitcast(row_sum)
  scales = f32[4,4]{1,0} broadcast(scale), dimensions={}
  ROOT result = f32[4,4]{1,0} multiply(rows, scales)
}

ENTRY main {
  scale = f32[] parameter(0)
  packed = bf16[4,3,4,17]{3,2,1,0} parameter(1)
  ROOT fusion = f32[4,4]{1,0} fusion(scale, packed), kind=kCustom,
    calls=attention,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  EXPECT_FALSE(configs.empty());
}

TEST_F(FlyFusionBackendTest, RetilesGenericTritonTransposeFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule generic_triton_transpose

transpose {
  p0 = bf16[256,384]{1,0} parameter(0)
  ROOT result = bf16[384,256]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[256,384]{1,0} parameter(0)
  ROOT fusion = bf16[384,256]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["64","64"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 9);
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.front()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, TunesBf16TransposeTileSize) {
  constexpr char kHlo[] = R"(
HloModule fly_transpose_autotune

transpose {
  input = bf16[256,384]{1,0} parameter(0)
  ROOT result = bf16[384,256]{1,0} transpose(input), dimensions={1,0}
}

ENTRY main {
  input = bf16[256,384]{1,0} parameter(0)
  ROOT fusion = bf16[384,256]{1,0} fusion(input), kind=kInput,
      calls=transpose
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 9);
  for (int64_t tile_rows : {32, 64, 128}) {
    for (int64_t tile_columns : {32, 64, 128}) {
      EXPECT_TRUE(
          std::any_of(configs.begin(), configs.end(), [&](const auto& config) {
            const BlockLevelFusionConfig& block = config->block_level();
            return block.num_warps() == tile_rows * tile_columns / 1024 &&
                   block.output_tiles_size() == 1 &&
                   block.output_tiles(0).sizes_size() == 2 &&
                   block.output_tiles(0).sizes(0) == tile_rows &&
                   block.output_tiles(0).sizes(1) == tile_columns;
          }));
    }
  }
}

TEST_F(FlyFusionBackendTest, TunesTransformerQkvSliceTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_transformer_qkv_transpose_autotune

transpose {
  input = bf16[256,3072]{1,0} parameter(0)
  view = bf16[2,128,3,16,64]{4,3,2,1,0} bitcast(input)
  value = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [2:3], [0:16], [0:64]}
  ROOT result = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(value),
    dimensions={0,2,3,4,1}
}

ENTRY main {
  input = bf16[256,3072]{1,0} parameter(0)
  ROOT fusion = bf16[2,1,16,64,128]{4,3,2,1,0} fusion(input),
    kind=kCustom, calls=transpose,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{
        "sizes":["1","1","1","64","128"]}],
        "num_warps":"4","num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 6);
  EXPECT_THAT(configs[0]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(32, 32));
  EXPECT_THAT(configs[1]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(32, 64));
  EXPECT_THAT(configs[3]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(64, 64));
  ASSERT_OK(backend_.ApplyConfig(*fusion, *configs.back()));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
}

TEST_F(FlyFusionBackendTest, TunesMultiOutputTransformerQkvTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_multi_output_transformer_qkv_transpose_autotune

transpose {
  input = bf16[256,3072]{1,0} parameter(0)
  view = bf16[2,128,3,16,64]{4,3,2,1,0} bitcast(input)
  q_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [0:1], [0:16], [0:64]}
  q = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(q_slice),
    dimensions={0,2,3,4,1}
  k_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [1:2], [0:16], [0:64]}
  k = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(k_slice),
    dimensions={0,2,3,4,1}
  v_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [2:3], [0:16], [0:64]}
  v = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(v_slice),
    dimensions={0,2,3,4,1}
  ROOT result = (bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0}) tuple(q, k, v)
}

ENTRY main {
  input = bf16[256,3072]{1,0} parameter(0)
  ROOT fusion = (bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0}) fusion(input), kind=kInput,
    calls=transpose
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 6);
  EXPECT_THAT(configs[0]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(32, 32));
  EXPECT_THAT(configs[1]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(32, 64));
}

TEST_F(FlyFusionBackendTest, TunesTransformerContextTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_transformer_context_transpose_autotune

transpose {
  input = bf16[32,64,128]{2,1,0} parameter(0)
  view = bf16[2,16,64,128]{3,2,1,0} bitcast(input)
  ROOT result = bf16[2,128,16,64]{3,2,1,0} transpose(view),
    dimensions={0,3,1,2}
}

ENTRY main {
  input = bf16[32,64,128]{2,1,0} parameter(0)
  ROOT fusion = bf16[2,128,16,64]{3,2,1,0} fusion(input), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{"kind":"__triton",
      "block_level_fusion_config":{"output_tiles":[{
        "sizes":["1","64","2","32"]}],
        "num_warps":"4","num_ctas":1,"num_stages":1}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 9);
  EXPECT_THAT(configs[1]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(32, 64));
  EXPECT_THAT(configs[2]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(32, 128));
  EXPECT_THAT(configs[8]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(128, 128));
}

TEST_F(FlyBackendTest, SupportsNativeF32MfmaGemm) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF32DotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X4_F32;
      }));
  auto mfma32 =
      std::find_if(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X2_F32;
      });
  ASSERT_NE(mfma32, configs.end());
  auto xf32 =
      std::find_if(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 64 && fly.block_n() == 32 &&
               fly.block_k() == 32 && fly.num_warps() == 2 && fly.stage_rhs() &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X4_XF32;
      });
  ASSERT_NE(xf32, configs.end());
  EXPECT_THAT(backend_.ApplyConfig(*fusion, **xf32), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsF32AlgorithmWithBf16Storage) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_algorithm_bf16_storage

gemm {
  lhs = bf16[64,32]{1,0} parameter(0)
  rhs = bf16[32,8]{1,0} parameter(1)
  ROOT dot = bf16[64,8]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_f32_f32_f32
}

ENTRY main {
  lhs = bf16[64,32]{1,0} parameter(0)
  rhs = bf16[32,8]{1,0} parameter(1)
  ROOT fusion = bf16[64,8]{1,0} fusion(lhs, rhs), kind=kCustom,
    calls=gemm,
    backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64","8"]}],
        "num_warps":"4","num_ctas":"1","num_stages":"1"}}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->fly().mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16 ||
           c->fly().mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8;
  }));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *configs.front()), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsSharedDotEpilogueOnSmallF32Gemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_small_f32_shared_dot

gemm {
  lhs = f32[4,8]{1,0} parameter(0)
  rhs = f32[8,16]{1,0} parameter(1)
  dot = f32[4,16]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT result = f32[4,16]{1,0} multiply(dot, dot)
}

ENTRY main {
  lhs = f32[4,8]{1,0} parameter(0)
  rhs = f32[8,16]{1,0} parameter(1)
  ROOT fusion = f32[4,16]{1,0} fusion(lhs, rhs), kind=kCustom,
    calls=gemm,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 1);
  const FlyGemmConfig& fly = configs.front()->fly();
  EXPECT_EQ(fly.block_m(), 16);
  EXPECT_EQ(fly.block_n(), 16);
  EXPECT_EQ(fly.block_k(), 16);
  EXPECT_EQ(fly.num_warps(), 1);
  EXPECT_EQ(fly.mfma_atom(), FlyGemmConfig::FLY_MFMA_16X16X4_F32);
}

TEST_F(FlyBackendTest, SupportsNativeS8MfmaGemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_s8_gemm

gemm {
  lhs = s8[128,256]{1,0} parameter(0)
  rhs = s8[256,128]{1,0} parameter(1)
  ROOT dot = s32[128,128]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = s8[128,256]{1,0} parameter(0)
  rhs = s8[256,128]{1,0} parameter(1)
  ROOT fusion = s32[128,128]{1,0} fusion(lhs, rhs), kind=kCustom,
    calls=gemm,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->fly().mfma_atom() ==
           FlyGemmConfig::FLY_MFMA_16X16X32_I8;
  }));
}

TEST_F(FlyBackendTest, DoesNotOfferXf32ForHighestPrecisionF32Gemm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_highest_precision_f32_dot

gemm {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      operand_precision={highest,highest}
}

ENTRY main {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT fusion = f32[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::none_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() ==
               FlyGemmConfig::FLY_MFMA_32X32X4_XF32;
      }));
}

TEST_F(FlyBackendTest, RejectsDotAlgorithmsThatRequireFission) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_x3_algorithm

gemm {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      algorithm=dot_bf16_bf16_f32_x3
}

ENTRY main {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT fusion = f32[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(configs.empty());
}

TEST_F(FlyBackendTest, ExplicitTf32AlgorithmOnlyOffersXf32Mfma) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_tf32_algorithm

gemm {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      algorithm=dot_tf32_tf32_f32
}

ENTRY main {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT fusion = f32[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->fly().mfma_atom() ==
           FlyGemmConfig::FLY_MFMA_32X32X4_XF32;
  }));
}

TEST_F(FlyBackendTest, Tf32X3UsesNativeFullPrecisionF32MfmaOnRocm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_tf32_x3_algorithm

gemm {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      algorithm=dot_tf32_tf32_f32_x3
}

ENTRY main {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT fusion = f32[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->fly().mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X4_F32 ||
           c->fly().mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X2_F32;
  }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    const FlyGemmConfig& fly = c->fly();
    return fly.block_m() == 128 && fly.block_n() == 128 &&
           fly.block_k() == 32 && fly.num_warps() == 8 && fly.stage_rhs() &&
           !fly.schedule_instructions() && fly.preload_lds_fragments() &&
           fly.single_buffer_lds() &&
           fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X2_F32;
  }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    const FlyGemmConfig& fly = c->fly();
    return fly.block_m() == 128 && fly.block_n() == 128 &&
           fly.block_k() == 64 && fly.num_warps() == 4 && fly.stage_rhs() &&
           fly.preload_lds_fragments() && fly.single_buffer_lds() &&
           fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X2_F32;
  }));
}

TEST_F(FlyBackendTest, SupportsF16GemmWithOptimizedXTilePipelines) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF16DotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return !config->fly().stage_rhs(); }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().stage_rhs() &&
               config->fly().preload_lds_fragments();
      }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().async_lhs(); }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().direct_to_vgpr(); }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 256 && fly.block_n() == 224 &&
               fly.block_k() == 64 && fly.num_warps() == 4 && fly.stage_rhs() &&
               fly.preload_lds_fragments() && fly.single_buffer_lds() &&
               fly.direct_to_vgpr();
      }));
  for (int32_t workgroup_mapping_n : {4, 6, 8}) {
    EXPECT_TRUE(
        std::any_of(configs.begin(), configs.end(), [&](const auto& config) {
          const FlyGemmConfig& fly = config->fly();
          return fly.block_m() == 256 && fly.block_n() == 224 &&
                 fly.block_k() == 64 && fly.num_warps() == 4 &&
                 fly.stage_rhs() && fly.preload_lds_fragments() &&
                 fly.direct_to_vgpr() &&
                 fly.workgroup_mapping_n() == workgroup_mapping_n;
        }));
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, DoesNotOfferBf16OnlyOutputStagingForF32Output) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF32OutputBf16GemmHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  for (const auto& config : configs) {
    EXPECT_FALSE(config->fly().stage_output()) << config->fly().DebugString();
  }
}

TEST_F(FlyBackendTest, SupportsF16Gemv) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF16GemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, SupportsBatchedF16Gemm) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF16BatchedGemmHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
  EXPECT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .output_tiles(0)
                .sizes(0),
            1);
}

TEST_F(FlyBackendTest, SupportsBatchedF16Gemv) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF16BatchedGemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().gemv_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, TunesGroupedVectorizedBatchedF16MatrixVector) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kF16BatchedMatrixVectorHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.gemv_outputs_per_wave() == 8 &&
               fly.gemv_k_vector_width() == 4 && fly.block_m() == 64 &&
               fly.num_warps() == 8;
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, SupportsBatchedF16BroadcastEpilogueChain) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF16BatchedEpilogueHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsUniformBf16ScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kScaledDotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().local_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsUniformFnuzFp8ScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kFnuzFp8ScaledDotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::all_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() ==
                   FlyGemmConfig::FLY_MFMA_16X16X32_FP8 ||
               config->fly().mfma_atom() ==
                   FlyGemmConfig::FLY_MFMA_32X32X16_FP8;
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsUniformBatchedFnuzFp8ScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kBatchedFnuzFp8ScaledDotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::all_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() ==
                   FlyGemmConfig::FLY_MFMA_16X16X32_FP8 ||
               config->fly().mfma_atom() ==
                   FlyGemmConfig::FLY_MFMA_32X32X16_FP8;
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
  ASSERT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .output_tiles_size(),
            1);
  EXPECT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .output_tiles(0)
                .sizes(0),
            1);
}

TEST_F(FlyBackendTest, SupportsNonuniformScale) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kNonuniformScaleHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));
  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::all_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_k() == 128 && !fly.stage_rhs() &&
               !fly.direct_to_vgpr() && !fly.local_split_k();
      }));
}

TEST_F(FlyBackendTest, TunesDirectAndDequantizedFnuzBlockScaling) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kFnuzFp8NonuniformScaleHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));
  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return !c->fly().dequantize_block_scales() &&
           c->fly().mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X32_FP8;
  }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->fly().dequantize_block_scales() &&
           c->fly().mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16;
  }));
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    const FlyGemmConfig& fly = c->fly();
    return fly.block_k() == 32 &&
           (!fly.dequantize_block_scales() ||
            fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16 ||
            fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8);
  }));
}

TEST_F(FlyBackendTest, SupportsNarrowingBf16Epilogue) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kNarrowingEpilogueHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().stage_output(); }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().local_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsAttentionValueOutputTranspose) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kAttentionValueOutputTransposeHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->fly().stage_output();
  }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    const FlyGemmConfig& fly = c->fly();
    return fly.block_m() == 64 && fly.block_n() == 32 && fly.block_k() == 32 &&
           fly.num_warps() == 4 && fly.waves_per_eu() == 4 &&
           fly.stage_output() && fly.stage_rhs();
  }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
  EXPECT_TRUE(
      gpu_config.fusion_backend_config().fly_gemm_config().stage_output());
}

TEST_F(FlyBackendTest, SupportsAttentionScoreInputTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_attention_score_input_transpose

gemm {
  q = bf16[32,64,128]{2,1,0} parameter(0)
  q_transposed = bf16[32,128,64]{2,1,0} transpose(q),
      dimensions={0,2,1}
  k = bf16[32,64,128]{2,1,0} parameter(1)
  dot = bf16[32,128,128]{2,1,0} dot(q_transposed, k),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
  converted = f32[32,128,128]{2,1,0} convert(dot)
  scale = bf16[] constant(0.125)
  broadcast = bf16[32,128,128]{2,1,0} broadcast(scale), dimensions={}
  widened_scale = f32[32,128,128]{2,1,0} convert(broadcast)
  scaled = f32[32,128,128]{2,1,0} multiply(converted, widened_scale)
  ROOT narrowed = bf16[32,128,128]{2,1,0} convert(scaled)
}

ENTRY main {
  q = bf16[32,64,128]{2,1,0} parameter(0)
  k = bf16[32,64,128]{2,1,0} parameter(1)
  ROOT fusion = bf16[32,128,128]{2,1,0} fusion(q, k), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    const FlyGemmConfig& fly = c->fly();
    return fly.block_m() == 128 && fly.block_n() == 128 &&
           fly.block_k() == 32 && fly.num_warps() == 8 && fly.stage_rhs();
  }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsPackedAttentionScoreSlices) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_packed_attention_score_slices

gemm {
  packed = bf16[3,4,32,1,64]{4,3,2,1,0} parameter(0)
  q_slice = bf16[1,4,32,1,64]{4,3,2,1,0} slice(packed),
      slice={[0:1], [0:4], [0:32], [0:1], [0:64]}
  q = bf16[4,32,64]{2,1,0} bitcast(q_slice)
  k_slice = bf16[1,4,32,1,64]{4,3,2,1,0} slice(packed),
      slice={[1:2], [0:4], [0:32], [0:1], [0:64]}
  k = bf16[4,32,64]{2,1,0} bitcast(k_slice)
  dot = bf16[4,64,64]{2,1,0} dot(q, k),
      lhs_batch_dims={0}, lhs_contracting_dims={1},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
  converted = f32[4,64,64]{2,1,0} convert(dot)
  scale = bf16[] constant(0.125)
  broadcast = bf16[4,64,64]{2,1,0} broadcast(scale), dimensions={}
  widened_scale = f32[4,64,64]{2,1,0} convert(broadcast)
  scaled = f32[4,64,64]{2,1,0} multiply(converted, widened_scale)
  ROOT narrowed = bf16[4,64,64]{2,1,0} convert(scaled)
}

ENTRY main {
  packed = bf16[3,4,32,1,64]{4,3,2,1,0} parameter(0)
  ROOT fusion = bf16[4,64,64]{2,1,0} fusion(packed), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
}

TEST_F(FlyBackendTest, TunesShortKWorkgroupMapping) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kNarrowingShortKEpilogueHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().workgroup_mapping_n() == 4;
      }));

  const auto is_staged_32x16x128 = [](const auto& config) {
    const FlyGemmConfig& fly = config->fly();
    return fly.block_m() == 32 && fly.block_n() == 16 &&
           fly.block_k() == 128 && fly.num_warps() == 2 && fly.stage_rhs();
  };
  EXPECT_EQ(absl::c_count_if(configs, is_staged_32x16x128), 12);

  debug_options_.set_xla_gpu_exhaustive_tiling_search(true);
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> exhaustive,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));
  EXPECT_EQ(absl::c_count_if(exhaustive, is_staged_32x16x128), 48);
}

TEST_F(FlyBackendTest, SupportsScalarEpilogue) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kScalarEpilogueHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsVectorEpilogue) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kVectorEpilogueHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsThreeStepEpilogueChain) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kEpilogueChainHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().local_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsBiasReluEpilogue) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kBiasReluEpilogueHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 128 && fly.block_n() == 64 &&
               fly.block_k() == 128 && fly.num_warps() == 8 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               fly.single_buffer_lds() && fly.rolling_refill();
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsF32ToBf16ContractionInputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kConvertedInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().local_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsF32ToBf16GemvInputs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kConvertedGemvInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, SupportsBitcastsAroundF32ToBf16Inputs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kConvertedBitcastInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsStaticUnitStrideInputSlices) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kConvertedSliceInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, RejectsStridedInputSlices) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kStridedSliceInputHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsEmpty());
}

TEST_F(FlyBackendTest, SupportsDynamicSliceInputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kDynamicSliceInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsFnuzFp8Inputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kFnuzFp8InputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() ==
               FlyGemmConfig::FLY_MFMA_16X16X32_FP8;
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() ==
               FlyGemmConfig::FLY_MFMA_32X32X16_FP8;
      }));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, OffersFlyDslStagedHomogeneousFnuzFp8) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHomogeneousFnuzFp8InputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 128 && fly.block_n() == 128 &&
               fly.block_k() == 128 && fly.num_warps() == 4 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X32_FP8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               !fly.single_buffer_lds();
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 256 && fly.block_n() == 224 &&
               fly.block_k() == 128 && fly.num_warps() == 4 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X32_FP8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               !fly.single_buffer_lds() && fly.direct_to_vgpr();
      }));
}

TEST_F(FlyBackendTest, OffersWideTileForF32HomogeneousFnuzFp8ScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHomogeneousFnuzFp8ScaledDotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 256 && fly.block_n() == 224 &&
               fly.block_k() == 128 && fly.num_warps() == 4 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X32_FP8 &&
               fly.schedule_instructions() && fly.stage_rhs() &&
               fly.preload_lds_fragments() && !fly.single_buffer_lds() &&
               fly.direct_to_vgpr();
      }));
}

TEST_F(FlyBackendTest, SupportsFnuzFp8GemvAndSplitK) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kFnuzFp8GemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.gemv_split_k() && fly.block_n() == 64 &&
               fly.num_warps() == 8 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X32_FP8;
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, SupportsUniformScaledFnuzFp8BatchedGemv) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kUniformScaledFnuzFp8BatchedGemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.gemv_split_k() && fly.block_n() >= 64 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X32_FP8;
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
  ASSERT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .output_tiles_size(),
            1);
  EXPECT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .output_tiles(0)
                .sizes(0),
            1);
}

TEST_F(FlyBackendTest, SupportsFnuzFp8BatchedGemm) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kFnuzFp8BatchedGemmHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::all_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() ==
                   FlyGemmConfig::FLY_MFMA_16X16X32_FP8 ||
               config->fly().mfma_atom() ==
                   FlyGemmConfig::FLY_MFMA_32X32X16_FP8;
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsFnuzFp8BatchedGemv) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kFnuzFp8BatchedGemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().gemv_split_k(); }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

TEST_F(FlyBackendTest, SupportsS4DequantizedRhs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kS4DequantizedRhsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::all_of(configs.begin(), configs.end(), [](const auto& config) {
        return !config->fly().async_lhs() && !config->fly().direct_to_vgpr();
      }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(),
      [](const auto& config) { return config->fly().stage_rhs(); }));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsS4DequantizedKContiguousRhs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kS4DequantizedKContiguousRhsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));
  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 256 && fly.block_n() == 224 &&
               fly.block_k() == 64 && fly.num_warps() == 4 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               !fly.single_buffer_lds() && fly.direct_to_vgpr();
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 224 && fly.block_n() == 256 &&
               fly.block_k() == 64 && fly.num_warps() == 4 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               !fly.single_buffer_lds() && fly.direct_to_vgpr();
      }));
  auto config = std::find_if(configs.begin(), configs.end(), [](const auto& c) {
    const FlyGemmConfig& fly = c->fly();
    return fly.block_m() == 32 && fly.block_n() == 128 && fly.block_k() == 64 &&
           fly.num_warps() == 8 &&
           fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16 &&
           !fly.stage_rhs();
  });
  ASSERT_NE(config, configs.end());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          backend_.Compile(*fusion, **config));
  EXPECT_THAT(executable->module_stats(), IsEmpty());
}

TEST_F(FlyBackendTest, SupportsS4ChannelScaledBatchedLhs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kS4ChannelScaledBatchedLhsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 16 && fly.block_n() == 64 &&
               fly.block_k() == 64 && fly.num_warps() == 2 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X16 &&
               !fly.stage_rhs();
      }));
  EXPECT_FALSE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 128 && fly.block_n() == 128 &&
               fly.block_k() == 64 && fly.num_warps() == 4 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               !fly.single_buffer_lds();
      }));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsS4SubchannelScaledBatchedLhs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kS4SubchannelScaledBatchedLhsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsS4ChannelScaledMultiBatchLhs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kS4ChannelScaledMultiBatchLhsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
  ASSERT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .output_tiles_size(),
            1);
  EXPECT_THAT(gpu_config.fusion_backend_config()
                  .block_level_fusion_config()
                  .output_tiles(0)
                  .sizes(),
              testing::ElementsAre(1, 1, config->fly().block_m(),
                                   config->fly().block_n()));
}

TEST_F(FlyBackendTest, SupportsTileAlignedConvertedConcatInputs) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kConvertedConcatInputsHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::all_of(configs.begin(), configs.end(), [](const auto& config) {
        return 512 % config->fly().block_m() == 0 &&
               512 % config->fly().block_n() == 0;
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, RejectsContractingDimensionConcatInput) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kContractingConcatInputHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsEmpty());
}

TEST_F(FlyBackendTest, SupportsTransposedUnevenConcatInput) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kTransposedUnevenConcatInputHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::all_of(configs.begin(), configs.end(), [](const auto& config) {
        return 1024 % config->fly().block_n() == 0 &&
               512 % config->fly().block_n() == 0;
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 32 && fly.block_n() == 64 &&
               fly.block_k() == 128 && fly.num_warps() == 2 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               fly.single_buffer_lds() && fly.rolling_refill();
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsBatchedBf16Gemm) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kBatchedGemmHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
  const BlockLevelFusionConfig& block =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  ASSERT_EQ(block.output_tiles_size(), 1);
  ASSERT_EQ(block.output_tiles(0).sizes_size(), 3);
  EXPECT_EQ(block.output_tiles(0).sizes(0), 1);
}

TEST_F(FlyBackendTest, SupportsCurrentGlobalSplitKOperandLayout) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_current_global_split_k_layout

split_gemm {
  lhs = bf16[128,2,512]{2,1,0} parameter(0)
  rhs = bf16[2,512,64]{2,1,0} parameter(1)
  ROOT dot = f32[2,128,64]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[128,2,512]{2,1,0} parameter(0)
  rhs = bf16[2,512,64]{2,1,0} parameter(1)
  ROOT fusion = f32[2,128,64]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=split_gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 64 && fly.block_n() == 32 &&
               fly.block_k() == 128 && fly.num_warps() == 2 && fly.stage_rhs();
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 128 && fly.block_n() == 64 &&
               fly.block_k() == 128 && fly.num_warps() == 8 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               fly.single_buffer_lds();
      }));
}

TEST_F(FlyBackendTest, SupportsShortKMiddleBatchDimension) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_short_k_middle_batch

gemm {
  lhs = bf16[256,4,128]{2,1,0} parameter(0)
  rhs = bf16[4,128,384]{2,1,0} parameter(1)
  ROOT dot = f32[4,256,384]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[256,4,128]{2,1,0} parameter(0)
  rhs = bf16[4,128,384]{2,1,0} parameter(1)
  ROOT fusion = f32[4,256,384]{2,1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__fly_gemm"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  EXPECT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& c) {
    return c->fly().block_k() <= 128 && !c->fly().stage_rhs();
  }));
}

TEST_F(FlyBackendTest, UsesGemmEmitterForMultiBatchMatrixVector) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_multi_batch_matrix_vector

gemm {
  lhs = bf16[16,32,64,128]{3,2,1,0} parameter(0)
  rhs = bf16[16,32,64,1]{3,2,1,0} parameter(1)
  ROOT dot = f32[32,16,128,1]{3,2,1,0} dot(lhs, rhs),
    lhs_batch_dims={1,0}, rhs_batch_dims={1,0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  lhs = bf16[16,32,64,128]{3,2,1,0} parameter(0)
  rhs = bf16[16,32,64,1]{3,2,1,0} parameter(1)
  ROOT fusion = f32[32,16,128,1]{3,2,1,0} fusion(lhs, rhs),
    kind=kCustom, calls=gemm,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemm");
}

TEST_F(FlyBackendTest, SupportsBatchedBf16Gemv) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kBatchedGemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_FALSE(configs.empty());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
  const BlockLevelFusionConfig& block =
      gpu_config.fusion_backend_config().block_level_fusion_config();
  ASSERT_EQ(block.output_tiles_size(), 1);
  ASSERT_EQ(block.output_tiles(0).sizes_size(), 3);
  EXPECT_EQ(block.output_tiles(0).sizes(0), 1);
}

TEST_F(FlyFusionBackendTest, TunesNativePagedAttentionOccupancy) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_paged_attention_autotune

fly_paged_attention {
  q = bf16[1,32,128]{2,1,0} parameter(0)
  k = bf16[15,16,8,128]{3,2,1,0} parameter(1)
  v = bf16[15,16,8,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,8]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = bf16[1,32,128]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}

ENTRY main {
  q = bf16[1,32,128]{2,1,0} parameter(0)
  k = bf16[15,16,8,128]{3,2,1,0} parameter(1)
  v = bf16[15,16,8,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,8]{1,0} parameter(4)
  ROOT fusion = bf16[1,32,128]{2,1,0}
    fusion(q, k, v, used_k, table), kind=kCustom,
    calls=fly_paged_attention,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","128"]}],"num_warps":"4","num_ctas":"1","num_stages":"1","waves_per_eu":"2"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 4);
  const std::vector<int64_t> expected_waves_per_eu = {0, 1, 2, 4};
  for (int64_t i = 0; i < configs.size(); ++i) {
    const BlockLevelFusionConfig& block = configs[i]->block_level();
    EXPECT_EQ(block.num_warps(), 4);
    EXPECT_EQ(block.waves_per_eu(), expected_waves_per_eu[i]);
    ASSERT_EQ(block.output_tiles_size(), 1);
    EXPECT_EQ(block.output_tiles(0).sizes(2), 128);
  }

  EXPECT_THAT(backend_.ApplyConfig(*fusion, *configs[1]), IsOk());
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly");
  EXPECT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .waves_per_eu(),
            1);
}

TEST_F(FlyFusionBackendTest, TunesSegmentedPagedAttentionWaveGeometry) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_segmented_paged_attention_autotune

producer_computation {
  q = bf16[1,16,128]{2,1,0} parameter(0)
  k = bf16[32,16,4,128]{3,2,1,0} parameter(1)
  v = bf16[32,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,32]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT partials = (f32[1,16,8,128]{3,2,1,0}, f32[1,16,8]{2,1,0}, f32[1,16,8]{2,1,0})
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode_segmented_producer"
}

reducer_computation {
  partial_o = f32[1,16,8,128]{3,2,1,0} parameter(0)
  partial_m = f32[1,16,8]{2,1,0} parameter(1)
  partial_l = f32[1,16,8]{2,1,0} parameter(2)
  ROOT attention = bf16[1,16,128]{2,1,0}
    custom-call(partial_o, partial_m, partial_l),
    custom_call_target="__fly$paged_attention_decode_segmented_reducer"
}

ENTRY main {
  q = bf16[1,16,128]{2,1,0} parameter(0)
  k = bf16[32,16,4,128]{3,2,1,0} parameter(1)
  v = bf16[32,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,32]{1,0} parameter(4)
  producer = (f32[1,16,8,128]{3,2,1,0}, f32[1,16,8]{2,1,0}, f32[1,16,8]{2,1,0})
    fusion(q, k, v, used_k, table), kind=kCustom,
    calls=producer_computation,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","1","128"]},{"sizes":["1","1","1"]},{"sizes":["1","1","1"]}],"num_warps":"2","num_ctas":"1","num_stages":"1","waves_per_eu":"2"}}}
  partial_o = f32[1,16,8,128]{3,2,1,0} get-tuple-element(producer), index=0
  partial_m = f32[1,16,8]{2,1,0} get-tuple-element(producer), index=1
  partial_l = f32[1,16,8]{2,1,0} get-tuple-element(producer), index=2
  ROOT reducer = bf16[1,16,128]{2,1,0}
    fusion(partial_o, partial_m, partial_l), kind=kCustom,
    calls=reducer_computation,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","128"]}],"num_warps":"2","num_ctas":"1","num_stages":"1","waves_per_eu":"2"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* reducer = module->entry_computation()->root_instruction();
  HloInstruction* producer = reducer->mutable_operand(0)->mutable_operand(0);

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> producer_configs,
      backend_.GetSupportedConfigs(*producer));
  ASSERT_EQ(producer_configs.size(), 4);
  for (const std::unique_ptr<BackendConfig>& config : producer_configs) {
    const BlockLevelFusionConfig& block = config->block_level();
    EXPECT_EQ(block.num_warps(), 2);
    ASSERT_EQ(block.output_tiles_size(), 3);
    EXPECT_THAT(block.output_tiles(0).sizes(),
                ::testing::ElementsAre(1, 1, 1, 128));
    EXPECT_THAT(block.output_tiles(1).sizes(), ::testing::ElementsAre(1, 1, 1));
    EXPECT_THAT(block.output_tiles(2).sizes(), ::testing::ElementsAre(1, 1, 1));
  }

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> reducer_configs,
      backend_.GetSupportedConfigs(*reducer));
  ASSERT_EQ(reducer_configs.size(), 4);
  for (const std::unique_ptr<BackendConfig>& config : reducer_configs) {
    const BlockLevelFusionConfig& block = config->block_level();
    EXPECT_EQ(block.num_warps(), 1);
    ASSERT_EQ(block.output_tiles_size(), 1);
    EXPECT_THAT(block.output_tiles(0).sizes(),
                ::testing::ElementsAre(1, 1, 128));
  }
}

TEST_F(FlyFusionBackendTest, UsesFourWavesForLongPagedAttentionSegments) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_long_segment_paged_attention_autotune

producer_computation {
  q = bf16[1,16,128]{2,1,0} parameter(0)
  k = bf16[256,16,4,128]{3,2,1,0} parameter(1)
  v = bf16[256,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,256]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT partials = (f32[1,16,32,128]{3,2,1,0}, f32[1,16,32]{2,1,0}, f32[1,16,32]{2,1,0})
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode_segmented_producer"
}

ENTRY main {
  q = bf16[1,16,128]{2,1,0} parameter(0)
  k = bf16[256,16,4,128]{3,2,1,0} parameter(1)
  v = bf16[256,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,256]{1,0} parameter(4)
  ROOT producer = (f32[1,16,32,128]{3,2,1,0}, f32[1,16,32]{2,1,0}, f32[1,16,32]{2,1,0})
    fusion(q, k, v, used_k, table), kind=kCustom,
    calls=producer_computation,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","1","128"]},{"sizes":["1","1","1"]},{"sizes":["1","1","1"]}],"num_warps":"4","num_ctas":"1","num_stages":"1","waves_per_eu":"2"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* producer = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*producer));
  ASSERT_EQ(configs.size(), 4);
  for (const std::unique_ptr<BackendConfig>& config : configs) {
    EXPECT_EQ(config->block_level().num_warps(), 4);
  }
}

TEST_F(FlyFusionBackendTest, TunesCooperativePagedAttentionStageDepth) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_cooperative_paged_attention_autotune

producer_computation {
  q = bf16[1,16,128]{2,1,0} parameter(0)
  k = bf16[4096,16,4,128]{3,2,1,0} parameter(1)
  v = bf16[4096,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,4096]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT partials = (f32[1,16,103,128]{3,2,1,0}, f32[1,16,103]{2,1,0}, f32[1,16,103]{2,1,0})
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode_segmented_producer"
}

ENTRY main {
  q = bf16[1,16,128]{2,1,0} parameter(0)
  k = bf16[4096,16,4,128]{3,2,1,0} parameter(1)
  v = bf16[4096,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,4096]{1,0} parameter(4)
  ROOT producer = (f32[1,16,103,128]{3,2,1,0}, f32[1,16,103]{2,1,0}, f32[1,16,103]{2,1,0})
    fusion(q, k, v, used_k, table), kind=kCustom,
    calls=producer_computation,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","1","128"]},{"sizes":["1","1","1"]},{"sizes":["1","1","1"]}],"num_warps":"2","num_ctas":"1","num_stages":"1","waves_per_eu":"2"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* producer = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*producer));

  ASSERT_EQ(configs.size(), 12);
  constexpr std::array<int64_t, 4> kOccupancies = {0, 1, 2, 4};
  for (int64_t stage = 1; stage <= 3; ++stage) {
    for (int64_t occupancy = 0; occupancy < kOccupancies.size(); ++occupancy) {
      const BlockLevelFusionConfig& block =
          configs[(stage - 1) * kOccupancies.size() + occupancy]->block_level();
      EXPECT_EQ(block.num_warps(), 2);
      EXPECT_EQ(block.num_stages(), stage);
      EXPECT_EQ(block.waves_per_eu(), kOccupancies[occupancy]);
    }
  }
}

TEST_F(FlyFusionBackendTest, TunesFusedCooperativePagedAttentionReducer) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fused_cooperative_paged_attention_autotune

fused_computation {
  q = bf16[1,16,128]{2,1,0} parameter(0)
  k = bf16[4096,16,4,128]{3,2,1,0} parameter(1)
  v = bf16[4096,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,4096]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT result = (bf16[1,16,128]{2,1,0}, f32[1,16,114,128]{3,2,1,0}, f32[1,16,114]{2,1,0}, f32[1,16,114]{2,1,0}, u32[1,4,2]{2,1,0})
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode_segmented_fused"
}

ENTRY main {
  q = bf16[1,16,128]{2,1,0} parameter(0)
  k = bf16[4096,16,4,128]{3,2,1,0} parameter(1)
  v = bf16[4096,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,4096]{1,0} parameter(4)
  ROOT fusion = (bf16[1,16,128]{2,1,0}, f32[1,16,114,128]{3,2,1,0}, f32[1,16,114]{2,1,0}, f32[1,16,114]{2,1,0}, u32[1,4,2]{2,1,0})
    fusion(q, k, v, used_k, table), kind=kCustom,
    calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","128"]},{"sizes":["1","1","1","128"]},{"sizes":["1","1","1"]},{"sizes":["1","1","1"]},{"sizes":["1","1","1"]}],"num_warps":"2","num_ctas":"1","num_stages":"1","waves_per_eu":"2"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_EQ(configs.size(), 12);
  constexpr std::array<int64_t, 4> kOccupancies = {0, 1, 2, 4};
  for (int64_t stage = 1; stage <= 3; ++stage) {
    for (int64_t occupancy = 0; occupancy < kOccupancies.size(); ++occupancy) {
      const BlockLevelFusionConfig& block =
          configs[(stage - 1) * kOccupancies.size() + occupancy]->block_level();
      EXPECT_EQ(block.num_warps(), 2);
      EXPECT_EQ(block.num_stages(), stage);
      EXPECT_EQ(block.waves_per_eu(), kOccupancies[occupancy]);
      ASSERT_EQ(block.output_tiles_size(), 5);
      EXPECT_THAT(block.output_tiles(0).sizes(),
                  ::testing::ElementsAre(1, 1, 128));
      EXPECT_THAT(block.output_tiles(1).sizes(),
                  ::testing::ElementsAre(1, 1, 1, 128));
      EXPECT_THAT(block.output_tiles(4).sizes(),
                  ::testing::ElementsAre(1, 1, 1));
    }
  }
}

TEST_F(FlyBackendTest, TunesGroupedVectorizedKContiguousBatchedGemv) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kBatchedGemvKContiguousHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.gemv_outputs_per_wave() == 8 &&
               fly.gemv_k_vector_width() == 2 && !fly.prefetch_rhs() &&
               fly.block_k() == 32;
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_n() == 2 && fly.num_warps() == 1 &&
               fly.gemv_outputs_per_wave() == 2;
      }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*fusion));
  EXPECT_THAT(backend_.ApplyConfig(*fusion, *config), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), "__fly_gemv");
}

}  // namespace
}  // namespace xla::gpu
