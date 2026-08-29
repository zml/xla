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

#include "xla/backends/gpu/transforms/attention_rewriter_fly.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/flydsl/attention_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr absl::string_view kAttentionHlo = R"(
HloModule fly_attention_rewriter

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

qkv_transpose {
  qkv = bf16[256,3072]{1,0} parameter(0)
  view = bf16[2,128,3,16,64]{4,3,2,1,0} bitcast(qkv)
  q_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [0:1], [0:16], [0:64]}
  k_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [1:2], [0:16], [0:64]}
  v_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [2:3], [0:16], [0:64]}
  // Multi-output fusion does not preserve semantic Q/K/V tuple ordering.
  ROOT tuple = (bf16[2,128,1,16,64]{4,3,2,1,0},
                bf16[2,128,1,16,64]{4,3,2,1,0},
                bf16[2,128,1,16,64]{4,3,2,1,0})
    tuple(k_slice, v_slice, q_slice)
}

qk {
  q = bf16[32,128,64]{2,1,0} parameter(0)
  k = bf16[32,128,64]{2,1,0} parameter(1)
  q_transposed = bf16[32,64,128]{2,1,0} transpose(q),
    dimensions={0,2,1}
  k_transposed = bf16[32,64,128]{2,1,0} transpose(k),
    dimensions={0,2,1}
  scores = bf16[32,128,128]{2,1,0} dot(q_transposed, k_transposed),
    lhs_batch_dims={0}, lhs_contracting_dims={1},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
  ROOT result = bf16[32,128,128]{2,1,0} copy(scores)
}

softmax {
  scores = bf16[32,128,128]{2,1,0} parameter(0)
  view = bf16[2,16,128,128]{3,2,1,0} bitcast(scores)
  converted = f32[2,16,128,128]{3,2,1,0} convert(view)
  minus_inf = f32[] constant(-inf)
  row_max = f32[2,16,128]{2,1,0} reduce(converted, minus_inf),
    dimensions={3}, to_apply=maximum
  maxima = f32[2,16,128,128]{3,2,1,0} broadcast(row_max),
    dimensions={0,1,2}
  shifted = f32[2,16,128,128]{3,2,1,0} subtract(converted, maxima)
  exponential = f32[2,16,128,128]{3,2,1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[2,16,128]{2,1,0} reduce(exponential, zero),
    dimensions={3}, to_apply=add
  sums = f32[2,16,128,128]{3,2,1,0} broadcast(row_sum),
    dimensions={0,1,2}
  normalized = f32[2,16,128,128]{3,2,1,0} divide(exponential, sums)
  ROOT result = bf16[2,16,128,128]{3,2,1,0} convert(normalized)
}

pv {
  v = bf16[32,128,64]{2,1,0} parameter(0)
  p = bf16[32,128,128]{2,1,0} parameter(1)
  v_transposed = bf16[32,64,128]{2,1,0} transpose(v),
    dimensions={0,2,1}
  context = bf16[32,64,128]{2,1,0} dot(v_transposed, p),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
  ROOT result = bf16[32,128,64]{2,1,0} transpose(context),
    dimensions={0,2,1}
}

ENTRY main {
  qkv = bf16[256,3072]{1,0} parameter(0)
  qkv_fusion = (bf16[2,128,1,16,64]{4,3,2,1,0},
                bf16[2,128,1,16,64]{4,3,2,1,0},
                bf16[2,128,1,16,64]{4,3,2,1,0})
    fusion(qkv), kind=kInput, calls=qkv_transpose
  q_tuple = bf16[2,128,1,16,64]{4,3,2,1,0}
    get-tuple-element(qkv_fusion), index=2
  k_tuple = bf16[2,128,1,16,64]{4,3,2,1,0}
    get-tuple-element(qkv_fusion), index=0
  v_tuple = bf16[2,128,1,16,64]{4,3,2,1,0}
    get-tuple-element(qkv_fusion), index=1
  q = bf16[32,128,64]{2,1,0} bitcast(q_tuple)
  k = bf16[32,128,64]{2,1,0} bitcast(k_tuple)
  v = bf16[32,128,64]{2,1,0} bitcast(v_tuple)
  qk_fusion = bf16[32,128,128]{2,1,0} fusion(q, k),
    kind=kCustom, calls=qk,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  scores_f32 = f32[32,128,128]{2,1,0} convert(qk_fusion)
  scale_bf16 = bf16[] constant(0.125)
  scale = f32[] convert(scale_bf16)
  scales = f32[32,128,128]{2,1,0} broadcast(scale), dimensions={}
  scaled = f32[32,128,128]{2,1,0} multiply(scores_f32, scales)
  scaled_bf16 = bf16[32,128,128]{2,1,0} convert(scaled)
  softmax_fusion = bf16[2,16,128,128]{3,2,1,0} fusion(scaled_bf16),
    kind=kCustom, calls=softmax,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","4","128"]}],
      "num_warps":"4","num_ctas":"1","num_stages":"1"}}}
  probabilities = bf16[32,128,128]{2,1,0} bitcast(softmax_fusion)
  pv_fusion = bf16[32,128,64]{2,1,0}
    fusion(v, probabilities), kind=kCustom, calls=pv,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  ROOT result = bf16[2,128,16,64]{3,2,1,0} bitcast(pv_fusion)
}
)";

// This is the representation seen at AttentionRewriterFly's actual position
// in the GPU compiler: Q/K/V are still raw packed slices and the softmax has
// not yet become a standalone fusion.
constexpr absl::string_view kEarlyPipelineAttentionHlo = R"(
HloModule fly_early_pipeline_attention_rewriter

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

qk {
  q = bf16[32,64,128]{2,1,0} parameter(0)
  k = bf16[32,64,128]{2,1,0} parameter(1)
  q_transposed = bf16[32,128,64]{2,1,0} transpose(q),
    dimensions={0,2,1}
  scores = bf16[32,128,128]{2,1,0} dot(q_transposed, k),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
  scale = bf16[] constant(0.125)
  scales = bf16[32,128,128]{2,1,0} broadcast(scale), dimensions={}
  ROOT result = bf16[32,128,128]{2,1,0} multiply(scores, scales)
}

pv {
  v = bf16[32,64,128]{2,1,0} parameter(0)
  p = bf16[32,128,128]{2,1,0} parameter(1)
  context = bf16[32,64,128]{2,1,0} dot(v, p),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
  view = bf16[2,16,64,128]{3,2,1,0} bitcast(context)
  ROOT result = bf16[2,128,16,64]{3,2,1,0} transpose(view),
    dimensions={0,3,1,2}
}

ENTRY main {
  qkv = bf16[256,3072]{1,0} parameter(0)
  packed = bf16[2,128,3,16,64]{4,3,2,1,0} bitcast(qkv)
  v_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(packed),
    slice={[0:2], [0:128], [2:3], [0:16], [0:64]}
  v_transpose = bf16[2,1,16,64,128]{4,3,2,1,0}
    transpose(v_slice), dimensions={0,2,3,4,1}
  v = bf16[32,64,128]{2,1,0} bitcast(v_transpose)
  q_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(packed),
    slice={[0:2], [0:128], [0:1], [0:16], [0:64]}
  q_transpose = bf16[2,1,16,64,128]{4,3,2,1,0}
    transpose(q_slice), dimensions={0,2,3,4,1}
  q = bf16[32,64,128]{2,1,0} bitcast(q_transpose)
  k_slice = bf16[2,128,1,16,64]{4,3,2,1,0} slice(packed),
    slice={[0:2], [0:128], [1:2], [0:16], [0:64]}
  k_transpose = bf16[2,1,16,64,128]{4,3,2,1,0}
    transpose(k_slice), dimensions={0,2,3,4,1}
  k = bf16[32,64,128]{2,1,0} bitcast(k_transpose)
  qk_fusion = bf16[32,128,128]{2,1,0} fusion(q, k),
    kind=kCustom, calls=qk,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  scores_view = bf16[2,16,128,128]{3,2,1,0} bitcast(qk_fusion)
  scores_f32 = f32[2,16,128,128]{3,2,1,0} convert(scores_view)
  key = s32[128,128]{1,0} iota(), iota_dimension=1
  query = s32[128,128]{1,0} iota(), iota_dimension=0
  key_le_query = pred[128,128]{1,0} compare(key, query), direction=LE
  causal = pred[2,16,128,128]{3,2,1,0} broadcast(key_le_query),
    dimensions={2,3}
  minus_inf = f32[] constant(-inf)
  masked_values = f32[2,16,128,128]{3,2,1,0} broadcast(minus_inf),
    dimensions={}
  masked_scores = f32[2,16,128,128]{3,2,1,0}
    select(causal, scores_f32, masked_values)
  row_max_0 = f32[2,16,128]{2,1,0} reduce(masked_scores, minus_inf),
    dimensions={3}, to_apply=maximum
  maxima_0 = f32[2,16,128,128]{3,2,1,0} broadcast(row_max_0),
    dimensions={0,1,2}
  shifted_0 = f32[2,16,128,128]{3,2,1,0}
    subtract(masked_scores, maxima_0)
  row_max_1 = f32[2,16,128]{2,1,0} reduce(shifted_0, minus_inf),
    dimensions={3}, to_apply=maximum
  maxima_1 = f32[2,16,128,128]{3,2,1,0} broadcast(row_max_1),
    dimensions={0,1,2}
  shifted_1 = f32[2,16,128,128]{3,2,1,0}
    subtract(shifted_0, maxima_1)
  exponential = f32[2,16,128,128]{3,2,1,0} exponential(shifted_1)
  zero = f32[] constant(0)
  row_sum = f32[2,16,128]{2,1,0} reduce(exponential, zero),
    dimensions={3}, to_apply=add
  sums = f32[2,16,128,128]{3,2,1,0} broadcast(row_sum),
    dimensions={0,1,2}
  normalized = f32[2,16,128,128]{3,2,1,0} divide(exponential, sums)
  probabilities_4d = bf16[2,16,128,128]{3,2,1,0} convert(normalized)
  probabilities = bf16[32,128,128]{2,1,0} bitcast(probabilities_4d)
  ROOT pv_fusion = bf16[2,128,16,64]{3,2,1,0}
    fusion(v, probabilities), kind=kCustom, calls=pv,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
)";

constexpr absl::string_view kGroupedQueryAttentionHlo = R"(
HloModule fly_grouped_query_attention_rewriter

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] maximum(lhs, rhs)
}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] add(lhs, rhs)
}

qk {
  key = bf16[8,128,64]{2,1,0} parameter(0)
  query = bf16[8,64,512]{2,1,0} parameter(1)
  scores = bf16[8,128,512]{2,1,0} dot(key, query),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
  scale = bf16[] constant(0.125)
  scales = bf16[8,128,512]{2,1,0} broadcast(scale), dimensions={}
  ROOT result = bf16[8,128,512]{2,1,0} multiply(scores, scales)
}

pv {
  value = bf16[8,64,128]{2,1,0} parameter(0)
  probabilities = bf16[8,128,512]{2,1,0} parameter(1)
  ROOT result = bf16[8,64,512]{2,1,0} dot(value, probabilities),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY main {
  packed_qkv = bf16[256,1536]{1,0} parameter(0)
  packed_view = bf16[2,128,1536]{2,1,0} bitcast(packed_qkv)

  kv_segment = bf16[2,128,512]{2,1,0} slice(packed_view),
    slice={[0:2], [0:128], [1024:1536]}
  kv_view = bf16[2,128,2,4,64]{4,3,2,1,0} bitcast(kv_segment)
  value_slice = bf16[2,128,1,4,64]{4,3,2,1,0} slice(kv_view),
    slice={[0:2], [0:128], [1:2], [0:4], [0:64]}
  value_transpose = bf16[2,1,4,64,128]{4,3,2,1,0}
    transpose(value_slice), dimensions={0,2,3,4,1}
  value = bf16[8,64,128]{2,1,0} bitcast(value_transpose)
  key_slice = bf16[2,128,1,4,64]{4,3,2,1,0} slice(kv_view),
    slice={[0:2], [0:128], [0:1], [0:4], [0:64]}
  key_transpose = bf16[2,1,4,128,64]{4,3,2,1,0}
    transpose(key_slice), dimensions={0,2,3,1,4}
  key = bf16[8,128,64]{2,1,0} bitcast(key_transpose)

  query_segment = bf16[2,128,1024]{2,1,0} slice(packed_view),
    slice={[0:2], [0:128], [0:1024]}
  query_view = bf16[2,128,4,4,64]{4,3,2,1,0}
    bitcast(query_segment)
  query_transpose = bf16[2,4,64,128,4]{4,3,2,1,0}
    transpose(query_view), dimensions={0,2,4,1,3}
  query = bf16[8,64,512]{2,1,0} bitcast(query_transpose)

  qk_fusion = bf16[8,128,512]{2,1,0} fusion(key, query),
    kind=kCustom, calls=qk,
    backend_config={"fusion_backend_config":{"kind":"__fly_gemm"}}
  scores = bf16[2,4,128,128,4]{4,3,2,1,0} bitcast(qk_fusion)
  scores_f32 = f32[2,4,128,128,4]{4,3,2,1,0} convert(scores)
  minus_inf = f32[] constant(-inf)
  row_max = f32[2,4,128,4]{3,2,1,0} reduce(scores_f32, minus_inf),
    dimensions={2}, to_apply=maximum
  maxima = f32[2,4,128,128,4]{4,3,2,1,0} broadcast(row_max),
    dimensions={0,1,3,4}
  shifted = f32[2,4,128,128,4]{4,3,2,1,0}
    subtract(scores_f32, maxima)
  exponential = f32[2,4,128,128,4]{4,3,2,1,0}
    exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[2,4,128,4]{3,2,1,0} reduce(exponential, zero),
    dimensions={2}, to_apply=add
  sums = f32[2,4,128,128,4]{4,3,2,1,0} broadcast(row_sum),
    dimensions={0,1,3,4}
  normalized = f32[2,4,128,128,4]{4,3,2,1,0}
    divide(exponential, sums)
  probabilities = bf16[2,4,128,128,4]{4,3,2,1,0}
    convert(normalized)
  probabilities_transpose = bf16[2,4,128,4,128]{4,3,2,1,0}
    transpose(probabilities), dimensions={0,1,2,4,3}
  probabilities_flat = bf16[8,128,512]{2,1,0}
    bitcast(probabilities_transpose)

  pv_fusion = bf16[8,64,512]{2,1,0} fusion(value, probabilities_flat),
    kind=kCustom, calls=pv,
    backend_config={"fusion_backend_config":{"kind":"__fly_gemm"}}
  context = bf16[2,4,64,4,128]{4,3,2,1,0} bitcast(pv_fusion)
  context_transpose = bf16[2,128,4,4,64]{4,3,2,1,0}
    transpose(context), dimensions={0,4,1,3,2}
  ROOT result = bf16[2,128,1024]{2,1,0} bitcast(context_transpose)
}
)";

constexpr absl::string_view kCrossAttentionHlo = R"(
HloModule fly_cross_attention_rewriter

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] maximum(lhs, rhs)
}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] add(lhs, rhs)
}

qk {
  query = bf16[32,128,64]{2,1,0} parameter(0)
  key = bf16[32,64,256]{2,1,0} parameter(1)
  scores = bf16[32,128,256]{2,1,0} dot(query, key),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
  scale = bf16[] constant(0.125)
  scales = bf16[32,128,256]{2,1,0} broadcast(scale), dimensions={}
  ROOT result = bf16[32,128,256]{2,1,0} multiply(scores, scales)
}

pv {
  value = bf16[32,64,256]{2,1,0} parameter(0)
  probabilities = bf16[32,128,256]{2,1,0} parameter(1)
  context = bf16[32,64,128]{2,1,0} dot(value, probabilities),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
  view = bf16[2,16,64,128]{3,2,1,0} bitcast(context)
  ROOT result = bf16[2,128,16,64]{3,2,1,0} transpose(view),
    dimensions={0,3,1,2}
}

ENTRY main {
  projected_query = bf16[256,1024]{1,0} parameter(0)
  query_view = bf16[2,128,16,64]{3,2,1,0} bitcast(projected_query)
  query_transpose = bf16[2,16,128,64]{3,2,1,0}
    transpose(query_view), dimensions={0,2,1,3}
  query = bf16[32,128,64]{2,1,0} bitcast(query_transpose)

  projected_key_value = bf16[512,2048]{1,0} parameter(1)
  key_value_view = bf16[2,256,2,16,64]{4,3,2,1,0}
    bitcast(projected_key_value)
  key_slice = bf16[2,256,1,16,64]{4,3,2,1,0}
    slice(key_value_view),
    slice={[0:2], [0:256], [0:1], [0:16], [0:64]}
  key_transpose = bf16[2,1,16,64,256]{4,3,2,1,0}
    transpose(key_slice), dimensions={0,2,3,4,1}
  key = bf16[32,64,256]{2,1,0} bitcast(key_transpose)
  value_slice = bf16[2,256,1,16,64]{4,3,2,1,0}
    slice(key_value_view),
    slice={[0:2], [0:256], [1:2], [0:16], [0:64]}
  value_transpose = bf16[2,1,16,64,256]{4,3,2,1,0}
    transpose(value_slice), dimensions={0,2,3,4,1}
  value = bf16[32,64,256]{2,1,0} bitcast(value_transpose)

  qk_fusion = bf16[32,128,256]{2,1,0} fusion(query, key),
    kind=kCustom, calls=qk,
    backend_config={"fusion_backend_config":{"kind":"__fly_gemm"}}
  score_view = bf16[2,16,128,256]{3,2,1,0} bitcast(qk_fusion)
  scores_f32 = f32[2,16,128,256]{3,2,1,0} convert(score_view)
  minus_inf = f32[] constant(-inf)
  row_max = f32[2,16,128]{2,1,0} reduce(scores_f32, minus_inf),
    dimensions={3}, to_apply=maximum
  maxima = f32[2,16,128,256]{3,2,1,0} broadcast(row_max),
    dimensions={0,1,2}
  shifted = f32[2,16,128,256]{3,2,1,0} subtract(scores_f32, maxima)
  exponential = f32[2,16,128,256]{3,2,1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[2,16,128]{2,1,0} reduce(exponential, zero),
    dimensions={3}, to_apply=add
  sums = f32[2,16,128,256]{3,2,1,0} broadcast(row_sum),
    dimensions={0,1,2}
  normalized = f32[2,16,128,256]{3,2,1,0}
    divide(exponential, sums)
  probabilities = bf16[2,16,128,256]{3,2,1,0} convert(normalized)
  probabilities_bh = bf16[32,128,256]{2,1,0} bitcast(probabilities)
  ROOT pv_fusion = bf16[2,128,16,64]{3,2,1,0}
    fusion(value, probabilities_bh), kind=kCustom, calls=pv,
    backend_config={"fusion_backend_config":{"kind":"__fly_gemm"}}
}
)";

class AttentionRewriterFlyTest : public HloHardwareIndependentTestBase {};

TEST_F(AttentionRewriterFlyTest, FusesQkSoftmaxPvAndQkvTranspose) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kAttentionHlo));

  EXPECT_THAT(AttentionRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(fusion->name(), HasSubstr("fly_attention"));
  ASSERT_EQ(fusion->operand_count(), 1);
  EXPECT_THAT(fusion->operand(0)->name(), HasSubstr("qkv"));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 2);

  int64_t dots = 0;
  int64_t nested_fusions = 0;
  for (const HloInstruction* instruction :
       fusion->fused_instructions_computation()->instructions()) {
    dots += instruction->opcode() == HloOpcode::kDot;
    nested_fusions += instruction->opcode() == HloOpcode::kFusion;
  }
  EXPECT_EQ(dots, 2);
  EXPECT_EQ(nested_fusions, 0);

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config = gpu_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind);
  EXPECT_THAT(fusion_config.block_level_fusion_config().output_tiles(0).sizes(),
              ElementsAre(1, 128, 1, 64));
  EXPECT_EQ(fusion_config.block_level_fusion_config().num_warps(), 4);
  EXPECT_EQ(fusion_config.block_level_fusion_config().waves_per_eu(), 2);
}

TEST_F(AttentionRewriterFlyTest, FusesEarlyPipelineCausalAttentionGraph) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kEarlyPipelineAttentionHlo));

  EXPECT_THAT(AttentionRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(fusion->name(), HasSubstr("fly_attention"));
  ASSERT_EQ(fusion->operand_count(), 1);
  EXPECT_EQ(fusion->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 2);

  int64_t dots = 0;
  int64_t nested_fusions = 0;
  int64_t selects = 0;
  for (const HloInstruction* instruction :
       fusion->fused_instructions_computation()->instructions()) {
    dots += instruction->opcode() == HloOpcode::kDot;
    nested_fusions += instruction->opcode() == HloOpcode::kFusion;
    selects += instruction->opcode() == HloOpcode::kSelect;
  }
  EXPECT_EQ(dots, 2);
  EXPECT_EQ(nested_fusions, 0);
  EXPECT_EQ(selects, 1);
}

TEST_F(AttentionRewriterFlyTest, FusesBroadcastScoreBiasAsAnOperand) {
  std::string hlo = absl::StrReplaceAll(
      kEarlyPipelineAttentionHlo,
      {{"  qkv = bf16[256,3072]{1,0} parameter(0)\n",
        "  qkv = bf16[256,3072]{1,0} parameter(0)\n"
        "  bias = f32[1,16,128,128]{3,2,1,0} parameter(1)\n"},
       {"  scores_f32 = f32[2,16,128,128]{3,2,1,0} convert(scores_view)\n",
        "  scores_f32 = f32[2,16,128,128]{3,2,1,0} convert(scores_view)\n"
        "  bias_view = f32[16,128,128]{2,1,0} bitcast(bias)\n"
        "  broadcast_bias = f32[2,16,128,128]{3,2,1,0} "
        "broadcast(bias_view), dimensions={1,2,3}\n"
        "  biased_scores = f32[2,16,128,128]{3,2,1,0} "
        "add(scores_f32, broadcast_bias)\n"},
       {"select(causal, scores_f32, masked_values)",
        "select(causal, biased_scores, masked_values)"}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  EXPECT_THAT(AttentionRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  ASSERT_EQ(fusion->operand_count(), 2);
  EXPECT_THAT(fusion->operand(0)->name(), HasSubstr("qkv"));
  EXPECT_THAT(fusion->operand(1)->name(), HasSubstr("bias"));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *fusion, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  std::optional<flydsl::FlyAttentionDescriptor> descriptor =
      flydsl::GetFlyAttentionDescriptor(analysis);
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->bias_parameter->parameter_number(), 1);
  EXPECT_THAT(descriptor->bias_strides,
              ElementsAre(0, 128 * 128, 128, 1));
  EXPECT_TRUE(descriptor->causal);
}

TEST_F(AttentionRewriterFlyTest, FusesBroadcastPaddingMaskAsAnOperand) {
  std::string hlo = absl::StrReplaceAll(
      kEarlyPipelineAttentionHlo,
      {{"  qkv = bf16[256,3072]{1,0} parameter(0)\n",
        "  qkv = bf16[256,3072]{1,0} parameter(0)\n"
        "  key_mask = pred[2,128]{1,0} parameter(1)\n"},
       {"  key = s32[128,128]{1,0} iota(), iota_dimension=1\n"
        "  query = s32[128,128]{1,0} iota(), iota_dimension=0\n"
        "  key_le_query = pred[128,128]{1,0} compare(key, query), direction=LE\n"
        "  causal = pred[2,16,128,128]{3,2,1,0} broadcast(key_le_query),\n"
        "    dimensions={2,3}\n",
        "  external_mask = pred[2,16,128,128]{3,2,1,0} "
        "broadcast(key_mask), dimensions={0,3}\n"},
       {"select(causal, scores_f32, masked_values)",
        "select(external_mask, scores_f32, masked_values)"}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  EXPECT_THAT(AttentionRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  ASSERT_EQ(fusion->operand_count(), 2);
  EXPECT_THAT(fusion->operand(0)->name(), HasSubstr("qkv"));
  EXPECT_THAT(fusion->operand(1)->name(), HasSubstr("key_mask"));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *fusion, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  std::optional<flydsl::FlyAttentionDescriptor> descriptor =
      flydsl::GetFlyAttentionDescriptor(analysis);
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->mask_parameter->parameter_number(), 1);
  EXPECT_THAT(descriptor->mask_strides, ElementsAre(128, 0, 0, 1));
  EXPECT_FALSE(descriptor->causal);
}

TEST_F(AttentionRewriterFlyTest,
       FusesCombinedCausalAndPaddingMaskAsAnOperand) {
  std::string hlo = absl::StrReplaceAll(
      kEarlyPipelineAttentionHlo,
      {{"  qkv = bf16[256,3072]{1,0} parameter(0)\n",
        "  qkv = bf16[256,3072]{1,0} parameter(0)\n"
        "  key_mask = pred[2,128]{1,0} parameter(1)\n"},
       {"  causal = pred[2,16,128,128]{3,2,1,0} broadcast(key_le_query),\n"
        "    dimensions={2,3}\n",
        "  causal = pred[2,16,128,128]{3,2,1,0} broadcast(key_le_query),\n"
        "    dimensions={2,3}\n"
        "  external_mask = pred[2,16,128,128]{3,2,1,0} "
        "broadcast(key_mask), dimensions={0,3}\n"
        "  combined_mask = pred[2,16,128,128]{3,2,1,0} "
        "and(causal, external_mask)\n"},
       {"select(causal, scores_f32, masked_values)",
        "select(combined_mask, scores_f32, masked_values)"}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  EXPECT_THAT(AttentionRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  ASSERT_EQ(fusion->operand_count(), 2);
  EXPECT_THAT(fusion->operand(0)->name(), HasSubstr("qkv"));
  EXPECT_THAT(fusion->operand(1)->name(), HasSubstr("key_mask"));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *fusion, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  std::optional<flydsl::FlyAttentionDescriptor> descriptor =
      flydsl::GetFlyAttentionDescriptor(analysis);
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->mask_parameter->parameter_number(), 1);
  EXPECT_THAT(descriptor->mask_strides, ElementsAre(128, 0, 0, 1));
  EXPECT_TRUE(descriptor->causal);
}

TEST_F(AttentionRewriterFlyTest, FusesGroupedQueryAttentionGraph) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kGroupedQueryAttentionHlo));

  EXPECT_THAT(AttentionRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(fusion->name(), HasSubstr("fly_attention"));
  ASSERT_EQ(fusion->operand_count(), 1);
  EXPECT_THAT(fusion->operand(0)->name(), HasSubstr("packed_qkv"));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 2);

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config = gpu_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind);
  EXPECT_THAT(fusion_config.block_level_fusion_config().output_tiles(0).sizes(),
              ElementsAre(1, 128, 1, 64));
}

TEST_F(AttentionRewriterFlyTest, FusesUnequalLengthCrossAttentionGraph) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kCrossAttentionHlo));

  EXPECT_THAT(AttentionRewriterFly().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_THAT(fusion->name(), HasSubstr("fly_attention"));
  ASSERT_EQ(fusion->operand_count(), 2);
  EXPECT_THAT(fusion->operand(0)->name(), HasSubstr("projected_query"));
  EXPECT_THAT(fusion->operand(1)->name(), HasSubstr("projected_key_value"));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config = gpu_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind);
  EXPECT_THAT(fusion_config.block_level_fusion_config().output_tiles(0).sizes(),
              ElementsAre(1, 128, 1, 64));
}

}  // namespace
}  // namespace xla::gpu
