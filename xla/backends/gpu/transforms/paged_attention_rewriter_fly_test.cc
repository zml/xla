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

#include "xla/backends/gpu/transforms/paged_attention_rewriter_fly.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/codegen/flydsl/paged_attention_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

class PagedAttentionRewriterFlyTest : public HloHardwareIndependentTestBase {};

TEST_F(PagedAttentionRewriterFlyTest, FormsNativeFlyPagedDecodeFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule fly_paged_attention_decode

ENTRY main {
  q = f16[8,32,128]{2,1,0} parameter(0)
  k = f16[32,16,4,128]{3,2,1,0} parameter(1)
  v = f16[32,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[8]{0} parameter(3)
  table = s32[8,8]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = f16[8,32,128]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}
)"));

  EXPECT_THAT(PagedAttentionRewriterFly().Run(module.get()),
              IsOkAndHolds(true));
  EXPECT_THAT(verifier().Run(module.get()), IsOkAndHolds(false));

  const auto* fusion = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  EXPECT_EQ(fusion->operand_count(), 5);
  EXPECT_EQ(fusion->fused_expression_root()->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(fusion->fused_expression_root()->operand(5)->opcode(),
            HloOpcode::kConstant);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), kFlyFusionKind);
  EXPECT_THAT(gpu_config.fusion_backend_config()
                  .block_level_fusion_config()
                  .output_tiles(0)
                  .sizes(),
              ElementsAre(1, 1, 128));

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *fusion, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  std::optional<flydsl::FlyPagedAttentionDescriptor> descriptor =
      flydsl::GetFlyPagedAttentionDescriptor(analysis);
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->sequences, 8);
  EXPECT_EQ(descriptor->query_heads, 32);
  EXPECT_EQ(descriptor->kv_heads, 4);
  EXPECT_EQ(descriptor->gqa_group, 8);
  EXPECT_EQ(descriptor->element_type, F16);
  EXPECT_EQ(descriptor->page_size, 16);
  EXPECT_EQ(descriptor->max_context, 128);
}

TEST_F(PagedAttentionRewriterFlyTest, LeavesUnsupportedContractUntouched) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule unsupported_fly_paged_attention_decode

ENTRY main {
  q = f16[8,16,64]{2,1,0} parameter(0)
  k = f16[32,16,4,64]{3,2,1,0} parameter(1)
  v = f16[32,16,4,64]{3,2,1,0} parameter(2)
  used_k = s32[8]{0} parameter(3)
  table = s32[8,8]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = f16[8,16,64]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}
)"));

  EXPECT_THAT(PagedAttentionRewriterFly().Run(module.get()),
              IsOkAndHolds(false));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kCustomCall);
}

TEST_F(PagedAttentionRewriterFlyTest, FormsStreamingLongContextFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule fly_paged_attention_streaming_decode

ENTRY main {
  q = bf16[2,16,128]{2,1,0} parameter(0)
  k = bf16[64,16,4,128]{3,2,1,0} parameter(1)
  v = bf16[64,16,4,128]{3,2,1,0} parameter(2)
  used_k = s32[2]{0} parameter(3)
  table = s32[2,32]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = bf16[2,16,128]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}
)"));

  EXPECT_THAT(PagedAttentionRewriterFly().Run(module.get()),
              IsOkAndHolds(true));
  const auto* reducer = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  ASSERT_EQ(reducer->operand_count(), 3);
  for (int64_t index = 0; index < 3; ++index) {
    EXPECT_EQ(reducer->operand(index)->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_EQ(reducer->operand(index)->tuple_index(), index);
  }
  const auto* producer =
      Cast<const HloFusionInstruction>(reducer->operand(0)->operand(0));
  ASSERT_TRUE(producer->shape().IsTuple());
  ASSERT_EQ(producer->shape().tuple_shapes_size(), 3);
  EXPECT_THAT(producer->shape().tuple_shapes(0).dimensions(),
              ElementsAre(2, 16, 4, 128));
  EXPECT_THAT(producer->shape().tuple_shapes(1).dimensions(),
              ElementsAre(2, 16, 4));
  EXPECT_THAT(producer->shape().tuple_shapes(2).dimensions(),
              ElementsAre(2, 16, 4));

  HloFusionAnalysis producer_analysis = HloFusionAnalysis::Create(
      *producer, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  std::optional<flydsl::FlyPagedAttentionSegmentedProducerDescriptor>
      producer_descriptor =
          flydsl::GetFlyPagedAttentionSegmentedProducerDescriptor(
              producer_analysis);
  ASSERT_TRUE(producer_descriptor.has_value());
  EXPECT_EQ(producer_descriptor->attention.max_context, 512);
  EXPECT_EQ(producer_descriptor->attention.pages_per_sequence, 32);
  EXPECT_EQ(producer_descriptor->num_segments, 4);
  EXPECT_EQ(producer_descriptor->segment_tokens, 128);

  HloFusionAnalysis reducer_analysis = HloFusionAnalysis::Create(
      *reducer, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  std::optional<flydsl::FlyPagedAttentionSegmentedReducerDescriptor>
      reducer_descriptor =
          flydsl::GetFlyPagedAttentionSegmentedReducerDescriptor(
              reducer_analysis);
  ASSERT_TRUE(reducer_descriptor.has_value());
  EXPECT_EQ(reducer_descriptor->num_segments, 4);
  EXPECT_EQ(reducer_descriptor->head_dimension, 128);
}

TEST_F(PagedAttentionRewriterFlyTest, UsesTunedCooperativeSegmentGeometry) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule fly_paged_attention_cooperative_decode

ENTRY main {
  q = bf16[1,32,128]{2,1,0} parameter(0)
  k = bf16[4096,16,8,128]{3,2,1,0} parameter(1)
  v = bf16[4096,16,8,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,4096]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = bf16[1,32,128]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}
)"));

  EXPECT_THAT(PagedAttentionRewriterFly().Run(module.get()),
              IsOkAndHolds(true));
  const auto* reducer = Cast<const HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  const auto* producer =
      Cast<const HloFusionInstruction>(reducer->operand(0)->operand(0));
  HloFusionAnalysis producer_analysis = HloFusionAnalysis::Create(
      *producer, TestGpuDeviceInfo::CudaOrRocmDeviceInfo());
  std::optional<flydsl::FlyPagedAttentionSegmentedProducerDescriptor>
      descriptor = flydsl::GetFlyPagedAttentionSegmentedProducerDescriptor(
          producer_analysis);
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->num_segments, 114);
  EXPECT_EQ(descriptor->segment_tokens, 576);

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       producer->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config()
                .block_level_fusion_config()
                .num_warps(),
            2);
}

}  // namespace
}  // namespace xla::gpu
