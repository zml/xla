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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
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
  const FusionBackendConfig& fusion_config =
      gpu_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind);
  EXPECT_THAT(fusion_config.block_level_fusion_config()
                  .output_tiles(0)
                  .sizes(),
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

}  // namespace
}  // namespace xla::gpu
