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

#include "xla/backends/gpu/transforms/vulkan_flash_attention_rewriter.h"

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using VulkanFlashAttentionRewriterTest = HloHardwareIndependentTestBase;

TEST_F(VulkanFlashAttentionRewriterTest, RewritesPrefillCallToCustomFusion) {
  RunAndFilecheckHloRewrite(R"(
ENTRY main {
  q = bf16[8,16,128] parameter(0)
  k = bf16[2,32,128] parameter(1)
  v = bf16[2,32,128] parameter(2)
  token_index = s32[] parameter(3)
  num_tokens = u32[] parameter(4)
  ROOT attention = bf16[8,16,128] custom-call(q, k, v, token_index, num_tokens),
    custom_call_target="zml$flash_attn"
})",
                            VulkanFlashAttentionRewriter(), R"(
; CHECK: %[[FUSION_BODY:vulkan_flash_attention.*]] {
; CHECK: ROOT %attention = bf16[8,16,128] custom-call
; CHECK: ENTRY %main
; CHECK: ROOT %vulkan_flash_attention = bf16[8,16,128] fusion
; CHECK-SAME: kind=kCustom
; CHECK-SAME: calls=%[[FUSION_BODY]]
; CHECK-SAME: "fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"vulkan_flash_attention"}}
                          )");
}

TEST_F(VulkanFlashAttentionRewriterTest, LeavesOtherCustomCallsAlone) {
  RunAndFilecheckHloRewrite(R"(
ENTRY main {
  input = bf16[8,1,128] parameter(0)
  ROOT other = bf16[8,1,128] custom-call(input),
    custom_call_target="not_flash_attention"
})",
                            VulkanFlashAttentionRewriter(), R"(
; CHECK: ROOT %other = bf16[8,1,128] custom-call(%input), custom_call_target="not_flash_attention"
; CHECK-NOT: fusion
                          )");
}

}  // namespace
}  // namespace xla::gpu
