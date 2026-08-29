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

#include <memory>
#include <string>

#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class FlyAttentionPipelineDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_flydsl_gemm(true);
    debug_options.set_xla_gpu_enable_flydsl_fusion(true);
    debug_options.set_xla_gpu_flydsl_replace_triton(true);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    debug_options.set_xla_gpu_experimental_enable_fusion_autotuner(false);
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }
};

TEST_F(FlyAttentionPipelineDeviceTest,
       KeepsSpecializedTileAfterSplitKCrossAttentionProjectionFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule jit_cross_attention, entry_computation_layout={(bf16[2,128,1024]{2,1,0}, bf16[2,256,1024]{2,1,0}, bf16[1024,1024]{1,0}, bf16[1024,2048]{1,0})->bf16[2,128,16,64]{3,2,1,0}}, allow_spmd_sharding_propagation_to_parameters={true,true,true,true}, allow_spmd_sharding_propagation_to_output={true}

FileNames
1 "fly_cross_attention_regression.py"

FunctionNames
1 "<module>"
2 "main"
3 "cross_attention"

FileLocations
1 {file_name_id=1 function_name_id=1 line=101 end_line=101 column=4 end_column=10}
2 {file_name_id=1 function_name_id=2 line=70 end_line=72 column=17 end_column=25}
3 {file_name_id=1 function_name_id=3 line=55 end_line=55 column=20 end_column=71}
4 {file_name_id=1 function_name_id=3 line=48 end_line=48 column=17 end_column=70}
5 {file_name_id=1 function_name_id=3 line=49 end_line=50 column=35 end_column=67}
6 {file_name_id=1 function_name_id=3 line=52 end_line=52 column=12 end_column=30}
7 {file_name_id=1 function_name_id=3 line=43 end_line=43 column=13 end_column=64}
8 {file_name_id=1 function_name_id=3 line=44 end_line=45 column=27 end_column=62}
9 {file_name_id=1 function_name_id=3 line=51 end_line=51 column=10 end_column=28}
10 {file_name_id=1 function_name_id=3 line=53 end_line=53 column=13 end_column=54}
11 {file_name_id=1 function_name_id=3 line=54 end_line=54 column=4 end_column=67}
12 {file_name_id=1 function_name_id=3 line=55 end_line=55 column=35 end_column=61}
13 {file_name_id=1 function_name_id=3 line=56 end_line=56 column=20 end_column=54}
14 {file_name_id=1 function_name_id=3 line=57 end_line=57 column=11 end_column=62}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=3}
4 {file_location_id=4 parent_frame_id=3}
5 {file_location_id=5 parent_frame_id=3}
6 {file_location_id=6 parent_frame_id=3}
7 {file_location_id=7 parent_frame_id=3}
8 {file_location_id=8 parent_frame_id=3}
9 {file_location_id=9 parent_frame_id=3}
10 {file_location_id=10 parent_frame_id=3}
11 {file_location_id=11 parent_frame_id=3}
12 {file_location_id=12 parent_frame_id=3}
13 {file_location_id=13 parent_frame_id=3}
14 {file_location_id=14 parent_frame_id=3}


%region_0.1 (reduce_max.3: f32[], reduce_max.4: f32[]) -> f32[] {
  %reduce_max.3 = f32[] parameter(0), metadata={op_name="reduce_max"}
  %reduce_max.4 = f32[] parameter(1), metadata={op_name="reduce_max"}
  ROOT %reduce_max.5 = f32[] maximum(%reduce_max.3, %reduce_max.4), metadata={op_name="jit(cross_attention)/reduce_max" stack_frame_id=3}
}

%region_1.2 (reduce_sum.3: f32[], reduce_sum.4: f32[]) -> f32[] {
  %reduce_sum.3 = f32[] parameter(0), metadata={op_name="reduce_sum"}
  %reduce_sum.4 = f32[] parameter(1), metadata={op_name="reduce_sum"}
  ROOT %reduce_sum.5 = f32[] add(%reduce_sum.3, %reduce_sum.4), metadata={op_name="jit(cross_attention)/reduce_sum" stack_frame_id=3}
}

ENTRY %main.3 (query_input.1: bf16[2,128,1024], key_value_input.1: bf16[2,256,1024], query_weight.1: bf16[1024,1024], key_value_weight.1: bf16[1024,2048]) -> bf16[2,128,16,64] {
  %key_value_input.1 = bf16[2,256,1024]{2,1,0} parameter(1), metadata={op_name="key_value_input"}
  %reshape.6 = bf16[512,1024]{1,0} reshape(%key_value_input.1), metadata={op_name="jit(cross_attention)/reshape" stack_frame_id=4}
  %key_value_weight.1 = bf16[1024,2048]{1,0} parameter(3), metadata={op_name="key_value_weight"}
  %dot_general.5 = bf16[512,2048]{1,0} dot(%reshape.6, %key_value_weight.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(cross_attention)/dot_general" stack_frame_id=4}
  %reshape.7 = bf16[2,256,2,16,64]{4,3,2,1,0} reshape(%dot_general.5), metadata={op_name="jit(cross_attention)/reshape" stack_frame_id=5}
  %slice.3 = bf16[2,256,1,16,64]{4,3,2,1,0} slice(%reshape.7), slice={[0:2], [0:256], [1:2], [0:16], [0:64]}, metadata={op_name="jit(cross_attention)/slice" stack_frame_id=6}
  %squeeze.3 = bf16[2,256,16,64]{3,2,1,0} reshape(%slice.3), metadata={op_name="jit(cross_attention)/squeeze" stack_frame_id=6}
  %query_input.1 = bf16[2,128,1024]{2,1,0} parameter(0), metadata={op_name="query_input"}
  %reshape.4 = bf16[256,1024]{1,0} reshape(%query_input.1), metadata={op_name="jit(cross_attention)/reshape" stack_frame_id=7}
  %query_weight.1 = bf16[1024,1024]{1,0} parameter(2), metadata={op_name="query_weight"}
  %dot_general.4 = bf16[256,1024]{1,0} dot(%reshape.4, %query_weight.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(cross_attention)/dot_general" stack_frame_id=7}
  %reshape.5 = bf16[2,128,16,64]{3,2,1,0} reshape(%dot_general.4), metadata={op_name="jit(cross_attention)/reshape" stack_frame_id=8}
  %slice.2 = bf16[2,256,1,16,64]{4,3,2,1,0} slice(%reshape.7), slice={[0:2], [0:256], [0:1], [0:16], [0:64]}, metadata={op_name="jit(cross_attention)/slice" stack_frame_id=9}
  %squeeze.2 = bf16[2,256,16,64]{3,2,1,0} reshape(%slice.2), metadata={op_name="jit(cross_attention)/squeeze" stack_frame_id=9}
  %dot_general.6 = bf16[2,16,128,256]{3,2,1,0} dot(%reshape.5, %squeeze.2), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}, metadata={op_name="jit(cross_attention)/bqhd,bkhd->bhqk/dot_general" stack_frame_id=10}
  %constant.5 = bf16[] constant(0.125)
  %mul.2 = bf16[2,16,128,256]{3,2,1,0} broadcast(%constant.5), dimensions={}, metadata={op_name="jit(cross_attention)/mul" stack_frame_id=11}
  %mul.3 = bf16[2,16,128,256]{3,2,1,0} multiply(%dot_general.6, %mul.2), metadata={op_name="jit(cross_attention)/mul" stack_frame_id=11}
  %convert_element_type.2 = f32[2,16,128,256]{3,2,1,0} convert(%mul.3), metadata={op_name="jit(cross_attention)/convert_element_type" stack_frame_id=12}
  %constant.7 = f32[] constant(-inf)
  %reduce_max.7 = f32[2,16,128]{2,1,0} reduce(%convert_element_type.2, %constant.7), dimensions={3}, to_apply=%region_0.1, metadata={op_name="jit(cross_attention)/reduce_max" stack_frame_id=3}
  %constant.4 = f32[] constant(-inf)
  %max.2 = f32[2,16,128]{2,1,0} broadcast(%constant.4), dimensions={}, metadata={op_name="jit(cross_attention)/max" stack_frame_id=3}
  %max.3 = f32[2,16,128]{2,1,0} maximum(%reduce_max.7, %max.2), metadata={op_name="jit(cross_attention)/max" stack_frame_id=3}
  %broadcast_in_dim.2 = f32[2,16,128,1]{3,2,1,0} reshape(%max.3), metadata={op_name="jit(cross_attention)/broadcast_in_dim" stack_frame_id=3}
  %sub.4 = f32[2,16,128,1]{3,2,1,0} broadcast(%broadcast_in_dim.2), dimensions={0,1,2,3}, metadata={op_name="jit(cross_attention)/sub" stack_frame_id=3}
  %sub.5 = f32[2,16,128]{2,1,0} reshape(%sub.4), metadata={op_name="jit(cross_attention)/sub" stack_frame_id=3}
  %sub.6 = f32[2,16,128,256]{3,2,1,0} broadcast(%sub.5), dimensions={0,1,2}, metadata={op_name="jit(cross_attention)/sub" stack_frame_id=3}
  %sub.7 = f32[2,16,128,256]{3,2,1,0} subtract(%convert_element_type.2, %sub.6), metadata={op_name="jit(cross_attention)/sub" stack_frame_id=3}
  %exp.1 = f32[2,16,128,256]{3,2,1,0} exponential(%sub.7), metadata={op_name="jit(cross_attention)/exp" stack_frame_id=3}
  %constant.6 = f32[] constant(0)
  %reduce_sum.7 = f32[2,16,128]{2,1,0} reduce(%exp.1, %constant.6), dimensions={3}, to_apply=%region_1.2, metadata={op_name="jit(cross_attention)/reduce_sum" stack_frame_id=3}
  %broadcast_in_dim.3 = f32[2,16,128,1]{3,2,1,0} reshape(%reduce_sum.7), metadata={op_name="jit(cross_attention)/broadcast_in_dim" stack_frame_id=3}
  %div.4 = f32[2,16,128,1]{3,2,1,0} broadcast(%broadcast_in_dim.3), dimensions={0,1,2,3}, metadata={op_name="jit(cross_attention)/div" stack_frame_id=3}
  %div.5 = f32[2,16,128]{2,1,0} reshape(%div.4), metadata={op_name="jit(cross_attention)/div" stack_frame_id=3}
  %div.6 = f32[2,16,128,256]{3,2,1,0} broadcast(%div.5), dimensions={0,1,2}, metadata={op_name="jit(cross_attention)/div" stack_frame_id=3}
  %div.7 = f32[2,16,128,256]{3,2,1,0} divide(%exp.1, %div.6), metadata={op_name="jit(cross_attention)/div" stack_frame_id=3}
  %convert_element_type.3 = bf16[2,16,128,256]{3,2,1,0} convert(%div.7), metadata={op_name="jit(cross_attention)/convert_element_type" stack_frame_id=13}
  %dot_general.7 = bf16[2,16,64,128]{3,2,1,0} dot(%squeeze.3, %convert_element_type.3), lhs_batch_dims={0,2}, lhs_contracting_dims={1}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}, metadata={op_name="jit(cross_attention)/bhqk,bkhd->bqhd/dot_general" stack_frame_id=14}
  ROOT %transpose.1 = bf16[2,128,16,64]{1,3,2,0} transpose(%dot_general.7), dimensions={0,3,1,2}, metadata={op_name="jit(cross_attention)/bhqk,bkhd->bqhd/transpose" stack_frame_id=14}
}

)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* attention =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(attention->opcode(), HloOpcode::kFusion)
      << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       attention->backend_config<GpuBackendConfig>());
  ASSERT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();

  int64_t embedded_reductions = 0;
  for (const HloInstruction* instruction :
       attention->fused_instructions_computation()->instructions()) {
    embedded_reductions += instruction->opcode() == HloOpcode::kReduce;
  }
  // Stable softmax contributes two reductions and the absorbed local split-K
  // query projection contributes the third.
  ASSERT_EQ(embedded_reductions, 3) << optimized->ToString();

  const BlockLevelFusionConfig& block =
      backend_config.fusion_backend_config().block_level_fusion_config();
  ASSERT_EQ(block.output_tiles_size(), 1);
  ASSERT_EQ(block.output_tiles(0).sizes_size(), 4);
  EXPECT_EQ(block.output_tiles(0).sizes(0), 1);
  EXPECT_EQ(block.output_tiles(0).sizes(1), 128);
  EXPECT_EQ(block.output_tiles(0).sizes(2), 1);
  EXPECT_EQ(block.output_tiles(0).sizes(3), 64);
  EXPECT_EQ(block.num_warps(), 4);
  EXPECT_EQ(block.waves_per_eu(), 2);
}

TEST_F(FlyAttentionPipelineDeviceTest,
       FormsAndExecutesCombinedCausalAndPaddingMaskAttention) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_padding_mask_attention_pipeline

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

ENTRY main {
  qkv = bf16[256,192]{1,0} parameter(0)
  key_mask = pred[2,128]{1,0} parameter(1)
  view = bf16[2,128,3,1,64]{4,3,2,1,0} reshape(qkv)
  q_slice = bf16[2,128,1,1,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [0:1], [0:1], [0:64]}
  q5 = bf16[2,1,1,64,128]{4,3,2,1,0} transpose(q_slice),
    dimensions={0,2,3,4,1}
  q = bf16[2,64,128]{2,1,0} reshape(q5)
  k_slice = bf16[2,128,1,1,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [1:2], [0:1], [0:64]}
  k5 = bf16[2,1,1,64,128]{4,3,2,1,0} transpose(k_slice),
    dimensions={0,2,3,4,1}
  k = bf16[2,64,128]{2,1,0} reshape(k5)
  v_slice = bf16[2,128,1,1,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [2:3], [0:1], [0:64]}
  v5 = bf16[2,1,1,64,128]{4,3,2,1,0} transpose(v_slice),
    dimensions={0,2,3,4,1}
  v = bf16[2,64,128]{2,1,0} reshape(v5)
  q_transposed = bf16[2,128,64]{2,1,0} transpose(q),
    dimensions={0,2,1}
  scores = bf16[2,128,128]{2,1,0} dot(q_transposed, k),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
  scores_f32 = f32[2,128,128]{2,1,0} convert(scores)
  scale_bf16 = bf16[] constant(0.125)
  scales_bf16 = bf16[2,128,128]{2,1,0} broadcast(scale_bf16),
    dimensions={}
  scales = f32[2,128,128]{2,1,0} convert(scales_bf16)
  scaled = f32[2,128,128]{2,1,0} multiply(scores_f32, scales)
  score_view = f32[2,1,128,128]{3,2,1,0} reshape(scaled)
  mask = pred[2,1,128,128]{3,2,1,0} broadcast(key_mask),
    dimensions={0,3}
  query_iota = s32[128,128]{1,0} iota(), iota_dimension=0
  key_iota = s32[128,128]{1,0} iota(), iota_dimension=1
  causal_2d = pred[128,128]{1,0} compare(query_iota, key_iota), direction=GE
  causal = pred[2,1,128,128]{3,2,1,0} broadcast(causal_2d),
    dimensions={2,3}
  combined_mask = pred[2,1,128,128]{3,2,1,0} and(causal, mask)
  minus_inf = f32[] constant(-inf)
  masked_value = f32[2,1,128,128]{3,2,1,0} broadcast(minus_inf),
    dimensions={}
  masked_scores = f32[2,1,128,128]{3,2,1,0}
    select(combined_mask, score_view, masked_value)
  row_max = f32[2,1,128]{2,1,0} reduce(masked_scores, minus_inf),
    dimensions={3}, to_apply=maximum
  maxima = f32[2,1,128,128]{3,2,1,0} broadcast(row_max),
    dimensions={0,1,2}
  shifted = f32[2,1,128,128]{3,2,1,0} subtract(masked_scores, maxima)
  exponential = f32[2,1,128,128]{3,2,1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[2,1,128]{2,1,0} reduce(exponential, zero),
    dimensions={3}, to_apply=add
  sums = f32[2,1,128,128]{3,2,1,0} broadcast(row_sum),
    dimensions={0,1,2}
  normalized = f32[2,1,128,128]{3,2,1,0} divide(exponential, sums)
  probabilities = bf16[2,1,128,128]{3,2,1,0} convert(normalized)
  probabilities_bh = bf16[2,128,128]{2,1,0} reshape(probabilities)
  context = bf16[2,64,128]{2,1,0} dot(v, probabilities_bh),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
  context_view = bf16[2,1,64,128]{3,2,1,0} reshape(context)
  ROOT result = bf16[2,128,1,64]{3,2,1,0} transpose(context_view),
    dimensions={0,3,1,2}
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloComputation* entry = optimized->entry_computation();
  EXPECT_EQ(entry->instruction_count(), 3) << optimized->ToString();
  const HloInstruction* root = entry->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_EQ(root->operand_count(), 2) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.03, /*arel=*/0.03}));

  // Sequences above four KV tiles use the looped emitter path. Keep a device
  // regression here so the scf.for-carried online-softmax state is checked
  // against the HLO interpreter, rather than only exercising the short
  // unrolled path above.
  std::string looped_hlo = absl::StrReplaceAll(
      kHlo, {{"HloModule fly_padding_mask_attention_pipeline",
              "HloModule fly_looped_padding_mask_attention"},
             {"ENTRY main {", "attention {"},
             {"256,192", "1024,192"},
             {"128", "512"}});
  looped_hlo += R"(

ENTRY main {
  qkv = bf16[1024,192]{1,0} parameter(0)
  key_mask = pred[2,512]{1,0} parameter(1)
  ROOT fusion = bf16[2,512,1,64]{3,2,1,0} fusion(qkv, key_mask),
    kind=kCustom, calls=attention,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","128","1","64"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "waves_per_eu":"2"}}}
}
)";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      looped_hlo, ErrorSpec{/*aabs=*/0.03, /*arel=*/0.03}));
}

}  // namespace
}  // namespace xla::gpu
