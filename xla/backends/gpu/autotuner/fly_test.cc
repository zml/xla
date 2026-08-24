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

TEST_F(FlyBackendTest, SupportsOddOutputTailsAndMinimumKTile) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kOddOutputTailBf16GemmHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 64 && fly.block_n() == 64 &&
               fly.block_k() == 16 && fly.num_warps() == 4 &&
               !fly.stage_rhs() && !fly.stage_output();
      }));
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
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 64 && fly.block_n() == 64 &&
               fly.block_k() == 64 && fly.num_warps() == 4 &&
               !fly.stage_rhs();
      }));
  EXPECT_TRUE(std::all_of(
      configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().block_k() <= 64;
      }));
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
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  auto config = std::find_if(configs.begin(), configs.end(), [](const auto& c) {
    const FlyGemmConfig& fly = c->fly();
    return fly.block_m() == 64 && fly.block_n() == 64 &&
           fly.block_k() == 64 && fly.num_warps() == 4 && fly.stage_rhs() &&
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
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
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
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(),
                          [&](const auto& config) {
                            return is_row_major_mfma32(config) &&
                                   !config->fly().stage_output() &&
                                   config->fly().schedule_instructions();
                          }));
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(),
                          [&](const auto& config) {
                            return is_row_major_mfma32(config) &&
                                   config->fly().stage_output() &&
                                   !config->fly().schedule_instructions();
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
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_n() == 64 && fly.num_warps() == 4 &&
               fly.gemv_outputs_per_wave() == 4 &&
               fly.gemv_k_vector_width() == 1;
      }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().block_n() == 128;
      }));
}

TEST_F(FlyBackendTest, OffersMfmaRowVectorLocalSplitK) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF16GemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  auto candidate = std::find_if(
      configs.begin(), configs.end(), [](const auto& config) {
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

  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 256 && fly.block_n() == 224 &&
               fly.block_k() == 64 && fly.num_warps() == 4 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               fly.direct_to_vgpr();
      }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
    ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<BackendConfig>> configs,
        backend_.GetSupportedConfigs(
            *module->entry_computation()->root_instruction()));

    ASSERT_EQ(configs.size(), 1);
    EXPECT_EQ(configs[0]->block_level().output_tiles(0).sizes(0), 1);
    EXPECT_EQ(configs[0]->block_level().num_warps(), 2);
    EXPECT_EQ(configs[0]->block_level().vector_size_bits(), 64);
  }
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
      EXPECT_TRUE(std::any_of(
          configs.begin(), configs.end(), [&](const auto& config) {
            return config->block_level().vector_size_bits() ==
                       vector_size_bits &&
                   config->block_level().num_warps() == num_warps;
          }));
    }
  }
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
    EXPECT_EQ(std::count_if(
                  configs.begin(), configs.end(), [&](const auto& config) {
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

  ASSERT_EQ(configs.size(), 3);
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

  ASSERT_EQ(configs.size(), 3);
  for (const auto& tile :
       {std::pair<int64_t, int64_t>{32, 1}, {64, 4}, {128, 16}}) {
    const int64_t tile_size = tile.first;
    const int64_t num_warps = tile.second;
    EXPECT_TRUE(std::any_of(
        configs.begin(), configs.end(), [&](const auto& config) {
          const BlockLevelFusionConfig& block = config->block_level();
          return block.num_warps() == num_warps &&
                 block.output_tiles_size() == 1 &&
                 block.output_tiles(0).sizes_size() == 2 &&
                 block.output_tiles(0).sizes(0) == tile_size &&
                 block.output_tiles(0).sizes(1) == tile_size;
        }));
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

  ASSERT_EQ(configs.size(), 2);
  EXPECT_THAT(configs[0]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(32, 32));
  EXPECT_THAT(configs[1]->block_level().output_tiles(0).sizes(),
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

  ASSERT_EQ(configs.size(), 2);
  EXPECT_THAT(configs[0]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(32, 32));
  EXPECT_THAT(configs[1]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(64, 64));
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

  ASSERT_EQ(configs.size(), 3);
  EXPECT_THAT(configs[2]->block_level().output_tiles(0).sizes(),
              ::testing::ElementsAre(128, 128));
}

TEST_F(FlyBackendTest, SupportsNativeF32MfmaGemm) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF32DotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() ==
               FlyGemmConfig::FLY_MFMA_16X16X4_F32;
      }));
  auto mfma32 = std::find_if(
      configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().mfma_atom() ==
               FlyGemmConfig::FLY_MFMA_32X32X2_F32;
      });
  ASSERT_NE(mfma32, configs.end());
  auto xf32 = std::find_if(configs.begin(), configs.end(), [](const auto& config) {
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
  EXPECT_TRUE(std::none_of(configs.begin(), configs.end(), [](const auto& config) {
    return config->fly().mfma_atom() ==
           FlyGemmConfig::FLY_MFMA_32X32X4_XF32;
  }));
}

TEST_F(FlyBackendTest, SupportsF16GemmWithOptimizedXTilePipelines) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kF16DotHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return !config->fly().stage_rhs();
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().stage_rhs() &&
               config->fly().preload_lds_fragments();
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().async_lhs();
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        return config->fly().direct_to_vgpr();
      }));
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 256 && fly.block_n() == 224 &&
               fly.block_k() == 64 && fly.num_warps() == 4 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               fly.single_buffer_lds() && fly.direct_to_vgpr();
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
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kF32OutputBf16GemmHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  ASSERT_FALSE(configs.empty());
  for (const auto& config : configs) {
    EXPECT_FALSE(config->fly().stage_output())
        << config->fly().DebugString();
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
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& config) {
    return config->fly().gemv_split_k();
  }));
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

  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
  EXPECT_TRUE(std::all_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
  EXPECT_TRUE(std::all_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& config) {
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
           c->fly().mfma_atom() ==
               FlyGemmConfig::FLY_MFMA_16X16X32_FP8;
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

  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 128 && fly.block_n() == 128 &&
               fly.block_k() == 128 && fly.num_warps() == 4 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_16X16X32_FP8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               !fly.single_buffer_lds();
      }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
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

  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kFnuzFp8BatchedGemmHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::all_of(configs.begin(), configs.end(), [](const auto& config) {
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
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kFnuzFp8BatchedGemvHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  ASSERT_FALSE(configs.empty());
  EXPECT_TRUE(std::any_of(configs.begin(), configs.end(), [](const auto& config) {
    return config->fly().gemv_split_k();
  }));
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
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      backend_.GetSupportedConfigs(
          *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 64 && fly.block_n() == 32 &&
               fly.block_k() == 128 && fly.num_warps() == 2 &&
               fly.stage_rhs();
      }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.block_m() == 128 && fly.block_n() == 64 &&
               fly.block_k() == 128 && fly.num_warps() == 8 &&
               fly.mfma_atom() == FlyGemmConfig::FLY_MFMA_32X32X8 &&
               fly.stage_rhs() && fly.preload_lds_fragments() &&
               fly.single_buffer_lds();
      }));
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

TEST_F(FlyBackendTest, TunesGroupedVectorizedKContiguousBatchedGemv) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kBatchedGemvKContiguousHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*fusion));

  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
        const FlyGemmConfig& fly = config->fly();
        return fly.gemv_outputs_per_wave() == 8 &&
               fly.gemv_k_vector_width() == 2 && !fly.prefetch_rhs() &&
               fly.block_k() == 32;
      }));
  EXPECT_TRUE(std::any_of(
      configs.begin(), configs.end(), [](const auto& config) {
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
