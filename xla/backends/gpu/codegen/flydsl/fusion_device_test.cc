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

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class FlyFusionDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {};

class FlyFusionPipelineDeviceTest
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

class FlyFusionAutotuningPipelineDeviceTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_flydsl_fusion(true);
    debug_options.set_xla_gpu_flydsl_replace_triton(true);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    debug_options.set_xla_gpu_experimental_enable_fusion_autotuner(true);
    debug_options.set_xla_gpu_autotune_level(3);
    debug_options.clear_xla_gpu_experimental_autotune_backends();
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::FLY_FUSION);
    return debug_options;
  }
};

TEST_F(FlyFusionPipelineDeviceTest,
       ExecutesRaggedPagedAttentionDecodeDtypesAndGqa) {
  constexpr absl::string_view kPagedHloTemplate = R"(
HloModule fly_paged_attention_decode

ENTRY main {
  q = __TYPE__[1,__GQA__,128]{2,1,0} parameter(0)
  k = __TYPE__[8,16,1,128]{3,2,1,0} parameter(1)
  v = __TYPE__[8,16,1,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,8]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = __TYPE__[1,__GQA__,128]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}
)";

  constexpr absl::string_view kReferenceHloTemplate = R"(
HloModule fly_paged_attention_decode_reference

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
  q = __TYPE__[1,__GQA__,128]{2,1,0} parameter(0)
  k = __TYPE__[8,16,1,128]{3,2,1,0} parameter(1)
  v = __TYPE__[8,16,1,128]{3,2,1,0} parameter(2)
  used_k_argument = s32[1]{0} parameter(3)
  table_argument = s32[1,8]{1,0} parameter(4)
  q_f32 = f32[1,__GQA__,128]{2,1,0} convert(q)
  k_flat = __TYPE__[128,128]{1,0} reshape(k)
  k_f32 = f32[128,128]{1,0} convert(k_flat)
  scores = f32[1,__GQA__,128]{2,1,0} dot(q_f32, k_f32),
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  scale = f32[] constant(0.0883883476)
  scale_broadcast = f32[1,__GQA__,128]{2,1,0} broadcast(scale), dimensions={}
  scaled = f32[1,__GQA__,128]{2,1,0} multiply(scores, scale_broadcast)
  token_iota = s32[128]{0} iota(), iota_dimension=0
  used_k = s32[] constant(37)
  used_broadcast = s32[128]{0} broadcast(used_k), dimensions={}
  token_valid = pred[128]{0} compare(token_iota, used_broadcast), direction=LT
  mask = pred[1,__GQA__,128]{2,1,0} broadcast(token_valid), dimensions={2}
  minus_inf = f32[] constant(-inf)
  minus_inf_broadcast = f32[1,__GQA__,128]{2,1,0} broadcast(minus_inf), dimensions={}
  masked = f32[1,__GQA__,128]{2,1,0} select(mask, scaled,
    minus_inf_broadcast)
  row_max = f32[1,__GQA__]{1,0} reduce(masked, minus_inf), dimensions={2},
    to_apply=maximum
  max_broadcast = f32[1,__GQA__,128]{2,1,0} broadcast(row_max), dimensions={0,1}
  shifted = f32[1,__GQA__,128]{2,1,0} subtract(masked, max_broadcast)
  exponential = f32[1,__GQA__,128]{2,1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[1,__GQA__]{1,0} reduce(exponential, zero), dimensions={2},
    to_apply=add
  sum_broadcast = f32[1,__GQA__,128]{2,1,0} broadcast(row_sum), dimensions={0,1}
  probabilities = f32[1,__GQA__,128]{2,1,0} divide(exponential, sum_broadcast)
  probabilities_element = __TYPE__[1,__GQA__,128]{2,1,0} convert(probabilities)
  staged_probabilities = f32[1,__GQA__,128]{2,1,0} convert(probabilities_element)
  v_flat = __TYPE__[128,128]{1,0} reshape(v)
  v_f32 = f32[128,128]{1,0} convert(v_flat)
  result = f32[1,__GQA__,128]{2,1,0} dot(staged_probabilities, v_f32),
    lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT output = __TYPE__[1,__GQA__,128]{2,1,0} convert(result)
}
)";

  constexpr std::array<std::pair<absl::string_view, absl::string_view>, 3>
      kCases = {{{"bf16", "4"}, {"f16", "8"}, {"bf16", "16"}}};
  for (const auto& [element_type, gqa_group] : kCases) {
    SCOPED_TRACE(std::string(element_type) + "/GQA" + std::string(gqa_group));
    const std::string paged_hlo = absl::StrReplaceAll(
        kPagedHloTemplate,
        {{"__TYPE__", element_type}, {"__GQA__", gqa_group}});
    const std::string reference_hlo = absl::StrReplaceAll(
        kReferenceHloTemplate,
        {{"__TYPE__", element_type}, {"__GQA__", gqa_group}});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(paged_hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> reference_module,
                         ParseAndReturnVerifiedModule(reference_hlo));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_disable_binary_libraries(true);
    reference_module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_disable_binary_libraries(true);
    ASSERT_OK_AND_ASSIGN(std::vector<Literal> arguments,
                         MakeFakeArguments(module.get()));
    arguments[3] = LiteralUtil::CreateR1<int32_t>({37});
    arguments[4] = LiteralUtil::CreateR2<int32_t>({{0, 1, 2, 3, 4, 5, 6, 7}});
    std::vector<const Literal*> argument_pointers;
    argument_pointers.reserve(arguments.size());
    for (const Literal& argument : arguments) {
      argument_pointers.push_back(&argument);
    }
    ASSERT_OK(PreprocessModuleForTestRunner(module.get()));
    ASSERT_OK_AND_ASSIGN(
        Literal actual,
        test_runner().Execute(std::move(module), argument_pointers,
                              /*run_hlo_passes=*/true));
    ASSERT_OK_AND_ASSIGN(Literal expected,
                         reference_runner().Execute(std::move(reference_module),
                                                    argument_pointers,
                                                    /*run_hlo_passes=*/false));
    EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
        expected, actual, ErrorSpec{/*aabs=*/0.03, /*arel=*/0.03}));
  }
}

TEST_F(FlyFusionPipelineDeviceTest,
       ExecutesStreamingLongContextPagedAttentionDecode) {
  constexpr absl::string_view kPagedHlo = R"(
HloModule fly_streaming_paged_attention_decode

ENTRY main {
  q = bf16[1,4,128]{2,1,0} parameter(0)
  k = bf16[32,16,1,128]{3,2,1,0} parameter(1)
  v = bf16[32,16,1,128]{3,2,1,0} parameter(2)
  used_k_argument = s32[1]{0} parameter(3)
  used_k = s32[1]{0} constant({401})
  table = s32[1,32]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = bf16[1,4,128]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}
)";

  constexpr absl::string_view kReferenceHlo = R"(
HloModule fly_streaming_paged_attention_decode_reference

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
  q = bf16[1,4,128]{2,1,0} parameter(0)
  k = bf16[32,16,1,128]{3,2,1,0} parameter(1)
  v = bf16[32,16,1,128]{3,2,1,0} parameter(2)
  used_k_argument = s32[1]{0} parameter(3)
  table_argument = s32[1,32]{1,0} parameter(4)
  q_f32 = f32[1,4,128]{2,1,0} convert(q)
  k_flat = bf16[512,128]{1,0} reshape(k)
  k_f32 = f32[512,128]{1,0} convert(k_flat)
  scores = f32[1,4,512]{2,1,0} dot(q_f32, k_f32),
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  scale = f32[] constant(0.0883883476)
  scale_broadcast = f32[1,4,512]{2,1,0} broadcast(scale), dimensions={}
  scaled = f32[1,4,512]{2,1,0} multiply(scores, scale_broadcast)
  token_iota = s32[512]{0} iota(), iota_dimension=0
  used_k = s32[] constant(401)
  used_broadcast = s32[512]{0} broadcast(used_k), dimensions={}
  token_valid = pred[512]{0} compare(token_iota, used_broadcast), direction=LT
  mask = pred[1,4,512]{2,1,0} broadcast(token_valid), dimensions={2}
  minus_inf = f32[] constant(-inf)
  minus_inf_broadcast = f32[1,4,512]{2,1,0} broadcast(minus_inf), dimensions={}
  masked = f32[1,4,512]{2,1,0} select(mask, scaled,
    minus_inf_broadcast)
  row_max = f32[1,4]{1,0} reduce(masked, minus_inf), dimensions={2},
    to_apply=maximum
  max_broadcast = f32[1,4,512]{2,1,0} broadcast(row_max), dimensions={0,1}
  shifted = f32[1,4,512]{2,1,0} subtract(masked, max_broadcast)
  exponential = f32[1,4,512]{2,1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[1,4]{1,0} reduce(exponential, zero), dimensions={2},
    to_apply=add
  sum_broadcast = f32[1,4,512]{2,1,0} broadcast(row_sum), dimensions={0,1}
  probabilities = f32[1,4,512]{2,1,0} divide(exponential, sum_broadcast)
  probabilities_element = bf16[1,4,512]{2,1,0} convert(probabilities)
  staged_probabilities = f32[1,4,512]{2,1,0} convert(probabilities_element)
  v_flat = bf16[512,128]{1,0} reshape(v)
  v_f32 = f32[512,128]{1,0} convert(v_flat)
  result = f32[1,4,128]{2,1,0} dot(staged_probabilities, v_f32),
    lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT output = bf16[1,4,128]{2,1,0} convert(result)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kPagedHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> reference_module,
                       ParseAndReturnVerifiedModule(kReferenceHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_disable_binary_libraries(true);
  reference_module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_disable_binary_libraries(true);
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> arguments,
                       MakeFakeArguments(module.get()));
  arguments[3] = LiteralUtil::CreateR1<int32_t>({401});
  arguments[4] = Literal::CreateFromShape(arguments[4].shape());
  for (int64_t page = 0; page < 32; ++page) {
    arguments[4].Set<int32_t>({0, page}, page);
  }
  std::vector<const Literal*> argument_pointers;
  argument_pointers.reserve(arguments.size());
  for (const Literal& argument : arguments) {
    argument_pointers.push_back(&argument);
  }
  ASSERT_OK(PreprocessModuleForTestRunner(module.get()));
  ASSERT_OK_AND_ASSIGN(Literal actual, test_runner().Execute(
                                           std::move(module), argument_pointers,
                                           /*run_hlo_passes=*/true));
  ASSERT_OK_AND_ASSIGN(
      Literal expected,
      reference_runner().Execute(std::move(reference_module), argument_pointers,
                                 /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
      expected, actual, ErrorSpec{/*aabs=*/0.04, /*arel=*/0.04}));
}

TEST_F(FlyFusionPipelineDeviceTest,
       ExecutesNonPowerOfTwoSegmentedPagedAttentionDecode) {
  constexpr absl::string_view kPagedHlo = R"(
HloModule fly_non_power_of_two_segmented_paged_attention_decode

ENTRY main {
  q = bf16[1,4,128]{2,1,0} parameter(0)
  k = bf16[__PAGES__,16,1,128]{3,2,1,0} parameter(1)
  v = bf16[__PAGES__,16,1,128]{3,2,1,0} parameter(2)
  used_k = s32[1]{0} parameter(3)
  table = s32[1,__PAGES__]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = bf16[1,4,128]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}
)";

  constexpr absl::string_view kReferenceHlo = R"(
HloModule fly_non_power_of_two_segmented_paged_attention_decode_reference

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
  q = bf16[1,4,128]{2,1,0} parameter(0)
  k = bf16[__PAGES__,16,1,128]{3,2,1,0} parameter(1)
  v = bf16[__PAGES__,16,1,128]{3,2,1,0} parameter(2)
  used_k_argument = s32[1]{0} parameter(3)
  table_argument = s32[1,__PAGES__]{1,0} parameter(4)
  q_f32 = f32[1,4,128]{2,1,0} convert(q)
  k_flat = bf16[__TOKENS__,128]{1,0} reshape(k)
  k_f32 = f32[__TOKENS__,128]{1,0} convert(k_flat)
  scores = f32[1,4,__TOKENS__]{2,1,0} dot(q_f32, k_f32),
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  scale = f32[] constant(0.0883883476)
  scale_broadcast = f32[1,4,__TOKENS__]{2,1,0} broadcast(scale), dimensions={}
  scaled = f32[1,4,__TOKENS__]{2,1,0} multiply(scores, scale_broadcast)
  token_iota = s32[__TOKENS__]{0} iota(), iota_dimension=0
  used_k = s32[] reshape(used_k_argument)
  used_broadcast = s32[__TOKENS__]{0} broadcast(used_k), dimensions={}
  token_valid = pred[__TOKENS__]{0} compare(token_iota, used_broadcast), direction=LT
  mask = pred[1,4,__TOKENS__]{2,1,0} broadcast(token_valid), dimensions={2}
  minus_inf = f32[] constant(-inf)
  minus_inf_broadcast = f32[1,4,__TOKENS__]{2,1,0} broadcast(minus_inf), dimensions={}
  masked = f32[1,4,__TOKENS__]{2,1,0} select(mask, scaled,
    minus_inf_broadcast)
  row_max = f32[1,4]{1,0} reduce(masked, minus_inf), dimensions={2},
    to_apply=maximum
  max_broadcast = f32[1,4,__TOKENS__]{2,1,0} broadcast(row_max), dimensions={0,1}
  shifted = f32[1,4,__TOKENS__]{2,1,0} subtract(masked, max_broadcast)
  exponential = f32[1,4,__TOKENS__]{2,1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[1,4]{1,0} reduce(exponential, zero), dimensions={2},
    to_apply=add
  sum_broadcast = f32[1,4,__TOKENS__]{2,1,0} broadcast(row_sum), dimensions={0,1}
  probabilities = f32[1,4,__TOKENS__]{2,1,0} divide(exponential, sum_broadcast)
  probabilities_element = bf16[1,4,__TOKENS__]{2,1,0} convert(probabilities)
  staged_probabilities = f32[1,4,__TOKENS__]{2,1,0} convert(probabilities_element)
  v_flat = bf16[__TOKENS__,128]{1,0} reshape(v)
  v_f32 = f32[__TOKENS__,128]{1,0} convert(v_flat)
  result = f32[1,4,128]{2,1,0} dot(staged_probabilities, v_f32),
    lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT output = bf16[1,4,128]{2,1,0} convert(result)
}
)";

  constexpr std::array<std::array<int64_t, 3>, 2> kCases = {
      {{4096, 65536, 65521}, {8192, 131072, 131057}}};
  for (const auto& [pages, tokens, used_tokens] : kCases) {
    SCOPED_TRACE(std::to_string(tokens));
    const std::string paged_hlo =
        absl::StrReplaceAll(kPagedHlo, {{"__PAGES__", std::to_string(pages)}});
    const std::string reference_hlo = absl::StrReplaceAll(
        kReferenceHlo, {{"__PAGES__", std::to_string(pages)},
                        {"__TOKENS__", std::to_string(tokens)}});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(paged_hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> reference_module,
                         ParseAndReturnVerifiedModule(reference_hlo));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_disable_binary_libraries(true);
    reference_module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_disable_binary_libraries(true);
    ASSERT_OK_AND_ASSIGN(std::vector<Literal> arguments,
                         MakeFakeArguments(module.get()));
    arguments[3] =
        LiteralUtil::CreateR1<int32_t>({static_cast<int32_t>(used_tokens)});
    arguments[4] = Literal::CreateFromShape(arguments[4].shape());
    for (int64_t page = 0; page < pages; ++page) {
      arguments[4].Set<int32_t>({0, page}, page);
    }
    std::vector<const Literal*> argument_pointers;
    argument_pointers.reserve(arguments.size());
    for (const Literal& argument : arguments) {
      argument_pointers.push_back(&argument);
    }
    ASSERT_OK(PreprocessModuleForTestRunner(module.get()));
    ASSERT_OK_AND_ASSIGN(
        Literal actual,
        test_runner().Execute(std::move(module), argument_pointers,
                              /*run_hlo_passes=*/true));
    ASSERT_OK_AND_ASSIGN(Literal expected,
                         reference_runner().Execute(std::move(reference_module),
                                                    argument_pointers,
                                                    /*run_hlo_passes=*/false));
    EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
        expected, actual, ErrorSpec{/*aabs=*/0.05, /*arel=*/0.05}));
  }
}

TEST_F(FlyFusionPipelineDeviceTest, FormsAndExecutesFlyFusionEndToEnd) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fusion_pipeline

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
  p0 = bf16[64,4096]{1,0} parameter(0)
  converted = f32[64,4096]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[64]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[64,4096]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[64,4096]{1,0} subtract(converted, broadcast_max)
  exponential = f32[64,4096]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[64,4096]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[64,4096]{1,0} divide(exponential, broadcast_sum)
  ROOT result = bf16[64,4096]{1,0} convert(normalized)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloComputation* entry = optimized->entry_computation();
  EXPECT_EQ(entry->instruction_count(), 2) << optimized->ToString();
  const HloInstruction* root = entry->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.01}));
}

TEST_F(FlyFusionPipelineDeviceTest,
       FormsAndExecutesRank4DoubleStabilizedSoftmax) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_rank4_double_stabilized_softmax_pipeline

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

maximum.1 {
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
  p0 = bf16[32,128,128]{2,1,0} parameter(0)
  view = bf16[2,16,128,128]{3,2,1,0} reshape(p0)
  converted = f32[2,16,128,128]{3,2,1,0} convert(view)
  minus_inf = f32[] constant(-inf)
  row_max.0 = f32[2,16,128]{2,1,0} reduce(converted, minus_inf),
    dimensions={3}, to_apply=maximum
  broadcast_max.0 = f32[2,16,128,128]{3,2,1,0}
    broadcast(row_max.0), dimensions={0,1,2}
  shifted.0 = f32[2,16,128,128]{3,2,1,0}
    subtract(converted, broadcast_max.0)
  row_max.1 = f32[2,16,128]{2,1,0} reduce(shifted.0, minus_inf),
    dimensions={3}, to_apply=maximum.1
  broadcast_max.1 = f32[2,16,128,128]{3,2,1,0}
    broadcast(row_max.1), dimensions={0,1,2}
  shifted.1 = f32[2,16,128,128]{3,2,1,0}
    subtract(shifted.0, broadcast_max.1)
  exponential = f32[2,16,128,128]{3,2,1,0} exponential(shifted.1)
  zero = f32[] constant(0)
  row_sum = f32[2,16,128]{2,1,0} reduce(exponential, zero),
    dimensions={3}, to_apply=add
  broadcast_sum = f32[2,16,128,128]{3,2,1,0}
    broadcast(row_sum), dimensions={0,1,2}
  normalized = f32[2,16,128,128]{3,2,1,0}
    divide(exponential, broadcast_sum)
  ROOT result = bf16[2,16,128,128]{3,2,1,0} convert(normalized)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.01}));
}

TEST_F(FlyFusionPipelineDeviceTest,
       FormsAndExecutesPackedQkvAttentionEndToEnd) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_packed_qkv_attention_pipeline

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
  scales_bf16 = bf16[2,128,128]{2,1,0} broadcast(scale_bf16), dimensions={}
  scales = f32[2,128,128]{2,1,0} convert(scales_bf16)
  scaled = f32[2,128,128]{2,1,0} multiply(scores_f32, scales)
  scaled_bf16 = bf16[2,128,128]{2,1,0} convert(scaled)
  score_view = bf16[2,1,128,128]{3,2,1,0} reshape(scaled_bf16)
  converted = f32[2,1,128,128]{3,2,1,0} convert(score_view)
  minus_inf = f32[] constant(-inf)
  row_max = f32[2,1,128]{2,1,0} reduce(converted, minus_inf),
    dimensions={3}, to_apply=maximum
  maxima = f32[2,1,128,128]{3,2,1,0} broadcast(row_max),
    dimensions={0,1,2}
  shifted = f32[2,1,128,128]{3,2,1,0} subtract(converted, maxima)
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
  EXPECT_EQ(entry->instruction_count(), 2) << optimized->ToString();
  const HloInstruction* root = entry->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_EQ(root->operand_count(), 1) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.03, /*arel=*/0.03}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesNativeElementwiseFusionEndToEnd) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_elementwise_pipeline

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  add = bf16[128,64]{1,0} add(p0, p1)
  scale = bf16[] constant(1.5)
  scale_broadcast = bf16[128,64]{1,0} broadcast(scale), dimensions={}
  scaled = bf16[128,64]{1,0} multiply(add, scale_broadcast)
  ROOT result = bf16[128,64]{1,0} maximum(scaled, p0)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesRaggedNativeElementwiseFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ragged_native_elementwise_pipeline

ENTRY main {
  p0 = bf16[127,65]{1,0} parameter(0)
  p1 = bf16[127,65]{1,0} parameter(1)
  add = bf16[127,65]{1,0} add(p0, p1)
  scale = bf16[] constant(1.5)
  scale_broadcast = bf16[127,65]{1,0} broadcast(scale), dimensions={}
  scaled = bf16[127,65]{1,0} multiply(add, scale_broadcast)
  ROOT result = bf16[127,65]{1,0} maximum(scaled, p0)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesRaggedSignedIntegerFusionWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ragged_signed_integer_pipeline

ENTRY main {
  p0 = s32[127,65]{1,0} parameter(0)
  p1 = s32[127,65]{1,0} parameter(1)
  sum = s32[127,65]{1,0} add(p0, p1)
  inverted = s32[127,65]{1,0} not(p0)
  mixed = s32[127,65]{1,0} xor(sum, inverted)
  less = pred[127,65]{1,0} compare(p0, p1), direction=LT
  ROOT result = s32[127,65]{1,0} select(less, mixed, sum)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesExternalPredicateFusionWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_external_predicate_pipeline

ENTRY main {
  p0 = pred[127,65]{1,0} parameter(0)
  p1 = s32[127,65]{1,0} parameter(1)
  p0_s32 = s32[127,65]{1,0} convert(p0)
  sum = s32[127,65]{1,0} add(p0_s32, p1)
  sum_pred = pred[127,65]{1,0} convert(sum)
  inverted = pred[127,65]{1,0} not(p0)
  not_sum = pred[127,65]{1,0} not(sum_pred)
  both = pred[127,65]{1,0} and(sum_pred, p0)
  neither = pred[127,65]{1,0} and(not_sum, inverted)
  ROOT result = pred[127,65]{1,0} or(both, neither)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesMultidimensionalIotaFusionWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_multidimensional_iota_pipeline

ENTRY main {
  p0 = f32[5,7,65,3]{3,2,1,0} parameter(0)
  iota = s32[5,7,65,3]{3,2,1,0} iota(), iota_dimension=1
  converted = f32[5,7,65,3]{3,2,1,0} convert(iota)
  ROOT result = f32[5,7,65,3]{3,2,1,0} add(p0, converted)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesShapeChangingViewWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_shape_changing_view_pipeline

ENTRY main {
  p0 = f32[65,67]{1,0} parameter(0)
  negated = f32[65,67]{1,0} negate(p0)
  view = f32[67,65]{1,0} reshape(negated)
  ROOT result = f32[67,65]{1,0} abs(view)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kBitcast) << optimized->ToString();
  root = root->operand(0);
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsWideF64ToNarrowS8FusionWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_wide_f64_to_narrow_s8_pipeline

ENTRY main {
  p0 = f64[65]{0} parameter(0)
  offset = f64[] constant(37)
  offsets = f64[65]{0} broadcast(offset), dimensions={}
  shifted = f64[65]{0} add(p0, offsets)
  lower = f64[] constant(-100)
  lowers = f64[65]{0} broadcast(lower), dimensions={}
  upper = f64[] constant(100)
  uppers = f64[65]{0} broadcast(upper), dimensions={}
  bounded = f64[65]{0} clamp(lowers, shifted, uppers)
  ROOT result = s8[65]{0} convert(bounded)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config =
      backend_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind) << optimized->ToString();
  EXPECT_LE(fusion_config.block_level_fusion_config().vector_size_bits(), 16)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesTypeChangingBitcastWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_type_changing_bitcast_pipeline

ENTRY main {
  p0 = s32[65]{0} parameter(0)
  bytes = s8[65,4]{1,0} bitcast-convert(p0)
  ROOT result = s8[65,4]{1,0} not(bytes)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesFp8ConversionFusionWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_fp8_conversion_pipeline

ENTRY main {
  p0 = f8e4m3fnuz[127,65]{1,0} parameter(0)
  converted = f32[127,65]{1,0} convert(p0)
  magnitude = f32[127,65]{1,0} abs(converted)
  scale = f32[] constant(0.5)
  scales = f32[127,65]{1,0} broadcast(scale), dimensions={}
  ROOT result = f32[127,65]{1,0} multiply(magnitude, scales)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesPackedS4ConversionFusionWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_packed_s4_conversion_pipeline

ENTRY main {
  p0 = s4[127,65]{1,0:E(4)} parameter(0)
  widened = s8[127,65]{1,0} convert(p0)
  converted = bf16[127,65]{1,0} convert(widened)
  magnitude = bf16[127,65]{1,0} abs(converted)
  scale = bf16[] constant(0.5)
  scales = bf16[127,65]{1,0} broadcast(scale), dimensions={}
  ROOT result = bf16[127,65]{1,0} multiply(magnitude, scales)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesPackedS4OutputFusionWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_packed_s4_output_pipeline

ENTRY main {
  p0 = s8[127,65]{1,0} parameter(0)
  bias = s8[] constant(9)
  biases = s8[127,65]{1,0} broadcast(bias), dimensions={}
  shifted = s8[127,65]{1,0} add(p0, biases)
  ROOT result = s4[127,65]{1,0:E(4)} convert(shifted)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesBf16ErfGeluWithoutTriton) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_erf_gelu_pipeline

ENTRY main {
  p0 = bf16[127,65]{1,0} parameter(0)
  p0_f32 = f32[127,65]{1,0} convert(p0)
  inv_sqrt_two = f32[] constant(0.7071067811865476)
  inv_sqrt_two_broadcast = f32[127,65]{1,0}
    broadcast(inv_sqrt_two), dimensions={}
  scaled = f32[127,65]{1,0} multiply(p0_f32, inv_sqrt_two_broadcast)
  erf = f32[127,65]{1,0} erf(scaled)
  one = f32[] constant(1)
  ones = f32[127,65]{1,0} broadcast(one), dimensions={}
  shifted = f32[127,65]{1,0} add(erf, ones)
  half = f32[] constant(0.5)
  halves = f32[127,65]{1,0} broadcast(half), dimensions={}
  gated = f32[127,65]{1,0} multiply(p0_f32, shifted)
  result_f32 = f32[127,65]{1,0} multiply(gated, halves)
  ROOT result = bf16[127,65]{1,0} convert(result_f32)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.015, /*arel=*/0.02}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesNarrowingBf16RowReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_narrowing_row_reduction_pipeline

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = bf16[65,256]{1,0} parameter(0)
  converted = f32[65,256]{1,0} convert(p0)
  zero = f32[] constant(0)
  row_sum = f32[65]{0} reduce(converted, zero), dimensions={1}, to_apply=add
  scale = f32[] constant(0.00390625)
  scales = f32[65]{0} broadcast(scale), dimensions={}
  mean = f32[65]{0} multiply(row_sum, scales)
  ROOT result = bf16[65]{0} convert(mean)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_EQ(root->fused_expression_root()->opcode(), HloOpcode::kConvert);
  bool contains_reduce = false;
  for (const HloInstruction* instruction :
       root->fused_instructions_computation()->instructions()) {
    contains_reduce |= instruction->opcode() == HloOpcode::kReduce;
  }
  EXPECT_TRUE(contains_reduce) << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesRaggedBf16RowReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_ragged_row_reduction_pipeline

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = bf16[65,259]{1,0} parameter(0)
  converted = f32[65,259]{1,0} convert(p0)
  zero = f32[] constant(0)
  ROOT row_sum = f32[65]{0} reduce(converted, zero), dimensions={1},
    to_apply=add
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config =
      backend_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind) << optimized->ToString();
  EXPECT_GE(fusion_config.block_level_fusion_config().vector_size_bits(), 64)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesTritonReductionTypes) {
  constexpr absl::string_view kS32Hlo = R"(
HloModule fly_native_s32_row_reduction_pipeline

maximum {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT max = s32[] maximum(lhs, rhs)
}

ENTRY main {
  p0 = s32[127,259]{1,0} parameter(0)
  init = s32[] constant(-2147483648)
  ROOT result = s32[127]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> s32_optimized,
                       GetOptimizedModule(kS32Hlo));
  const HloInstruction* s32_root =
      s32_optimized->entry_computation()->root_instruction();
  ASSERT_EQ(s32_root->opcode(), HloOpcode::kFusion)
      << s32_optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig s32_backend_config,
                       s32_root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(s32_backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << s32_optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kS32Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kF64Hlo = R"(
HloModule fly_native_f64_row_reduction_pipeline

maximum {
  lhs = f64[] parameter(0)
  rhs = f64[] parameter(1)
  ROOT max = f64[] maximum(lhs, rhs)
}

ENTRY main {
  p0 = f64[31,67]{1,0} parameter(0)
  init = f64[] constant(-inf)
  ROOT result = f64[31]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> f64_optimized,
                       GetOptimizedModule(kF64Hlo));
  const HloInstruction* f64_root =
      f64_optimized->entry_computation()->root_instruction();
  ASSERT_EQ(f64_root->opcode(), HloOpcode::kFusion)
      << f64_optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig f64_backend_config,
                       f64_root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(f64_backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << f64_optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kF64Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesRank3Bf16RmsNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank3_bf16_rms_norm_pipeline

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = bf16[2,17,259]{2,1,0} parameter(0)
  converted = f32[2,17,259]{2,1,0} convert(p0)
  squared = f32[2,17,259]{2,1,0} multiply(converted, converted)
  zero = f32[] constant(0)
  row_sum = f32[2,17]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[2,17]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,17]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,17]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,17]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,17]{1,0} rsqrt(variance)
  scales = f32[2,17,259]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,17,259]{2,1,0} multiply(converted, scales)
  ROOT result = bf16[2,17,259]{2,1,0} convert(normalized)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& fusion_config =
      backend_config.fusion_backend_config();
  EXPECT_EQ(fusion_config.kind(), kFlyFusionKind) << optimized->ToString();
  EXPECT_GE(fusion_config.block_level_fusion_config().vector_size_bits(), 64)
      << optimized->ToString();
  bool contains_reduce = false;
  bool contains_rsqrt = false;
  for (const HloInstruction* instruction :
       root->fused_instructions_computation()->instructions()) {
    contains_reduce |= instruction->opcode() == HloOpcode::kReduce;
    contains_rsqrt |= instruction->opcode() == HloOpcode::kRsqrt;
  }
  EXPECT_TRUE(contains_reduce) << optimized->ToString();
  EXPECT_TRUE(contains_rsqrt) << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.01, /*arel=*/0.01}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesResidualRmsNormAfterSplitKReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_residual_rms_norm_after_split_k_pipeline

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  residual = bf16[2,17,259]{2,1,0} parameter(0)
  partials = f32[2,34,259]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  projected = f32[34,259]{1,0} reduce(partials, zero), dimensions={0},
    to_apply=add
  projected_bf16 = bf16[34,259]{1,0} convert(projected)
  projected_view = bf16[2,17,259]{2,1,0} reshape(projected_bf16)
  residual_f32 = f32[2,17,259]{2,1,0} convert(residual)
  projected_f32 = f32[2,17,259]{2,1,0} convert(projected_view)
  added = f32[2,17,259]{2,1,0} add(residual_f32, projected_f32)
  squared = f32[2,17,259]{2,1,0} multiply(added, added)
  row_sum = f32[2,17]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[2,17]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,17]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,17]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,17]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,17]{1,0} rsqrt(variance)
  scales = f32[2,17,259]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,17,259]{2,1,0} multiply(added, scales)
  ROOT result = bf16[2,17,259]{2,1,0} convert(normalized)
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* root =
      optimized->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.02, /*arel=*/0.02}));
}

TEST_F(FlyFusionDeviceTest, ExecutesGeneralLeadingReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_general_leading_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

body {
  p0 = f32[257,259]{1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT result = f32[259]{0} reduce(p0, zero), dimensions={0}, to_apply=add
}

ENTRY main {
  p0 = f32[257,259]{1,0} parameter(0)
  ROOT fusion = f32[259]{0} fusion(p0), kind=kCustom, calls=body,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(FlyFusionDeviceTest, ExecutesCooperativeMiddleDimensionReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_cooperative_middle_dimension_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

body {
  p0 = f32[64,128,256]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT result = f32[64,256]{1,0} reduce(p0, zero), dimensions={1},
      to_apply=add
}

ENTRY main {
  p0 = f32[64,128,256]{2,1,0} parameter(0)
  ROOT fusion = f32[64,256]{1,0} fusion(p0), kind=kCustom, calls=body,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsNativeMultiOutputWithoutTritonFlag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_multi_output_pipeline

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  sum = bf16[128,64]{1,0} add(p0, p1)
  product = bf16[128,64]{1,0} multiply(sum, p0)
  ROOT result = (bf16[128,64]{1,0}, bf16[128,64]{1,0})
    tuple(sum, product)
})";
  ASSERT_FALSE(GetDebugOptionsForTest()
                   .xla_gpu_unsupported_enable_triton_multi_output_fusion());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* fly_fusion = nullptr;
  for (const HloInstruction* instruction :
       optimized->entry_computation()->instructions()) {
    if (instruction->opcode() != HloOpcode::kFusion ||
        !instruction->shape().IsTuple()) {
      continue;
    }
    ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                         instruction->backend_config<GpuBackendConfig>());
    if (backend_config.fusion_backend_config().kind() == kFlyFusionKind) {
      fly_fusion = instruction;
      break;
    }
  }
  ASSERT_NE(fly_fusion, nullptr) << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionPipelineDeviceTest,
       ExecutesLateFlyLoopFusionWithoutAutotuning) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_late_loop_fusion_pipeline

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  sum = bf16[128,64]{1,0} add(p0, p1)
  ROOT result = bf16[128,64]{1,0} multiply(sum, p0)
}

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0, p1), kind=kLoop,
    calls=elementwise
})";

  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionPipelineDeviceTest, ExecutesNativeFlyForwardAndReverseScans) {
  constexpr absl::string_view kHloTemplate = R"(
HloModule fly_native_scan

add {
  lhs = f32[257] parameter(0)
  rhs = f32[257] parameter(1)
  sum = f32[257] add(lhs, rhs)
  ROOT result = (f32[257], f32[257]) tuple(sum, sum)
}

ENTRY main {
  input = f32[257,129]{1,0} parameter(0)
  zero = f32[] constant(0)
  init = f32[257]{0} broadcast(zero), dimensions={}
  scan = (f32[257,129]{1,0}, f32[257]{0}) scan(input, init),
    dimensions={1}, num_carries=1, is_associative=true,
    is_reverse=__REVERSE__, to_apply=add
  ROOT output = f32[257,129]{1,0} get-tuple-element(scan), index=0
}
)";
  for (bool reverse : {false, true}) {
    std::string hlo = absl::StrReplaceAll(
        kHloTemplate, {{"__REVERSE__", reverse ? "true" : "false"}});
    EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
  }
}

TEST_F(FlyFusionPipelineDeviceTest, ExecutesNativeFlyS32ScanExactly) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_s32_scan

add {
  lhs = s32[19] parameter(0)
  rhs = s32[19] parameter(1)
  sum = s32[19] add(lhs, rhs)
  ROOT result = (s32[19], s32[19]) tuple(sum, sum)
}

ENTRY main {
  input = s32[19,193]{1,0} parameter(0)
  zero = s32[] constant(0)
  init = s32[19]{0} broadcast(zero), dimensions={}
  scan = (s32[19,193]{1,0}, s32[19]{0}) scan(input, init),
    dimensions={1}, num_carries=1, is_associative=true, to_apply=add
  ROOT output = s32[19,193]{1,0} get-tuple-element(scan), index=0
}
)";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionPipelineDeviceTest, ExecutesNativeFlyLongS32ScanExactly) {
  constexpr absl::string_view kHloTemplate = R"(
HloModule fly_native_long_s32_scan

add {
  lhs = s32[3] parameter(0)
  rhs = s32[3] parameter(1)
  sum = s32[3] add(lhs, rhs)
  ROOT result = (s32[3], s32[3]) tuple(sum, sum)
}

ENTRY main {
  input = s32[3,4097]{1,0} parameter(0)
  zero = s32[] constant(0)
  init = s32[3]{0} broadcast(zero), dimensions={}
  scan = (s32[3,4097]{1,0}, s32[3]{0}) scan(input, init),
    dimensions={1}, num_carries=1, is_associative=true,
    is_reverse=__REVERSE__, to_apply=add
  ROOT output = s32[3,4097]{1,0} get-tuple-element(scan), index=0
}
)";
  for (bool reverse : {false, true}) {
    std::string hlo = absl::StrReplaceAll(
        kHloTemplate, {{"__REVERSE__", reverse ? "true" : "false"}});
    EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
  }
}

TEST_F(FlyFusionPipelineDeviceTest, ExecutesNativeFlyBf16AndF16Scans) {
  constexpr absl::string_view kHloTemplate = R"(
HloModule fly_native_low_precision_scan

add {
  lhs = __TYPE__[73] parameter(0)
  rhs = __TYPE__[73] parameter(1)
  sum = __TYPE__[73] add(lhs, rhs)
  ROOT result = (__TYPE__[73], __TYPE__[73]) tuple(sum, sum)
}

ENTRY main {
  input = __TYPE__[73,193]{1,0} parameter(0)
  zero = __TYPE__[] constant(0)
  init = __TYPE__[73]{0} broadcast(zero), dimensions={}
  scan = (__TYPE__[73,193]{1,0}, __TYPE__[73]{0}) scan(input, init),
    dimensions={1}, num_carries=1, is_associative=true, to_apply=add
  ROOT output = __TYPE__[73,193]{1,0} get-tuple-element(scan), index=0
}
)";
  for (absl::string_view type : {"bf16", "f16"}) {
    std::string hlo = absl::StrReplaceAll(kHloTemplate, {{"__TYPE__", type}});
    EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{/*aabs=*/0.5, /*arel=*/0.06}));
  }
}

TEST_F(FlyFusionDeviceTest, Bf16Softmax64x4096) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_softmax

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
  p0 = bf16[64,4096]{1,0} parameter(0)
  converted = f32[64,4096]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[64]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[64,4096]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[64,4096]{1,0} subtract(converted, broadcast_max)
  exponential = f32[64,4096]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[64]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[64,4096]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[64,4096]{1,0} divide(exponential, broadcast_sum)
  ROOT result = bf16[64,4096]{1,0} convert(normalized)
}

ENTRY main {
  p0 = bf16[64,4096]{1,0} parameter(0)
  ROOT fusion = bf16[64,4096]{1,0} fusion(p0), kind=kCustom,
    calls=softmax,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","4096"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.01}));
}

TEST_F(FlyFusionDeviceTest, NativePackedQkvBf16Attention) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_qkv_attention

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

attention {
  qkv = bf16[128,192]{1,0} parameter(0)
  view = bf16[1,128,3,1,64]{4,3,2,1,0} bitcast(qkv)
  q_slice = bf16[1,128,1,1,64]{4,3,2,1,0} slice(view),
    slice={[0:1], [0:128], [0:1], [0:1], [0:64]}
  q5 = bf16[1,1,1,64,128]{4,3,2,1,0} transpose(q_slice),
    dimensions={0,2,3,4,1}
  q = bf16[1,64,128]{2,1,0} bitcast(q5)
  k_slice = bf16[1,128,1,1,64]{4,3,2,1,0} slice(view),
    slice={[0:1], [0:128], [1:2], [0:1], [0:64]}
  k5 = bf16[1,1,1,64,128]{4,3,2,1,0} transpose(k_slice),
    dimensions={0,2,3,4,1}
  k = bf16[1,64,128]{2,1,0} bitcast(k5)
  v_slice = bf16[1,128,1,1,64]{4,3,2,1,0} slice(view),
    slice={[0:1], [0:128], [2:3], [0:1], [0:64]}
  v5 = bf16[1,1,1,64,128]{4,3,2,1,0} transpose(v_slice),
    dimensions={0,2,3,4,1}
  v = bf16[1,64,128]{2,1,0} bitcast(v5)
  q_transposed = bf16[1,128,64]{2,1,0} transpose(q),
    dimensions={0,2,1}
  scores = bf16[1,128,128]{2,1,0} dot(q_transposed, k),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
  scores_f32 = f32[1,128,128]{2,1,0} convert(scores)
  scale_bf16 = bf16[] constant(0.125)
  scales_bf16 = bf16[1,128,128]{2,1,0} broadcast(scale_bf16),
    dimensions={}
  scales = f32[1,128,128]{2,1,0} convert(scales_bf16)
  scaled = f32[1,128,128]{2,1,0} multiply(scores_f32, scales)
  scaled_bf16 = bf16[1,128,128]{2,1,0} convert(scaled)
  score_view = bf16[1,1,128,128]{3,2,1,0} bitcast(scaled_bf16)
  converted = f32[1,1,128,128]{3,2,1,0} convert(score_view)
  minus_inf = f32[] constant(-inf)
  row_max = f32[1,1,128]{2,1,0} reduce(converted, minus_inf),
    dimensions={3}, to_apply=maximum
  maxima = f32[1,1,128,128]{3,2,1,0} broadcast(row_max),
    dimensions={0,1,2}
  shifted = f32[1,1,128,128]{3,2,1,0} subtract(converted, maxima)
  exponential = f32[1,1,128,128]{3,2,1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[1,1,128]{2,1,0} reduce(exponential, zero),
    dimensions={3}, to_apply=add
  sums = f32[1,1,128,128]{3,2,1,0} broadcast(row_sum),
    dimensions={0,1,2}
  normalized = f32[1,1,128,128]{3,2,1,0} divide(exponential, sums)
  probabilities = bf16[1,1,128,128]{3,2,1,0} convert(normalized)
  probabilities_bh = bf16[1,128,128]{2,1,0} bitcast(probabilities)
  context = bf16[1,64,128]{2,1,0} dot(v, probabilities_bh),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={2}
  context_view = bf16[1,1,64,128]{3,2,1,0} bitcast(context)
  ROOT result = bf16[1,128,1,64]{3,2,1,0} transpose(context_view),
    dimensions={0,3,1,2}
}

ENTRY main {
  qkv = bf16[128,192]{1,0} parameter(0)
  ROOT fusion = bf16[1,128,1,64]{3,2,1,0} fusion(qkv),
    kind=kCustom, calls=attention,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","128","1","64"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "waves_per_eu":"2"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.03, /*arel=*/0.03}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16ElementwiseDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_native_elementwise

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  add = bf16[128,64]{1,0} add(p0, p1)
  scale = bf16[] constant(1.5)
  scale_broadcast = bf16[128,64]{1,0} broadcast(scale), dimensions={}
  scaled = bf16[128,64]{1,0} multiply(add, scale_broadcast)
  ROOT result = bf16[128,64]{1,0} maximum(scaled, p0)
}

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRaggedF16MixedPrecisionElementwiseDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ragged_f16_native_elementwise

elementwise {
  p0 = f16[127,65]{1,0} parameter(0)
  p1 = f16[127,65]{1,0} parameter(1)
  p0_f32 = f32[127,65]{1,0} convert(p0)
  p1_f32 = f32[127,65]{1,0} convert(p1)
  sum = f32[127,65]{1,0} add(p0_f32, p1_f32)
  zero = f32[] constant(0)
  zeros = f32[127,65]{1,0} broadcast(zero), dimensions={}
  positive = pred[127,65]{1,0} compare(sum, zeros), direction=GT
  absolute = f32[127,65]{1,0} abs(sum)
  negated = f32[127,65]{1,0} negate(absolute)
  selected = f32[127,65]{1,0} select(positive, absolute, negated)
  ROOT result = f16[127,65]{1,0} convert(selected)
}

ENTRY main {
  p0 = f16[127,65]{1,0} parameter(0)
  p1 = f16[127,65]{1,0} parameter(1)
  ROOT fusion = f16[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionDeviceTest, NativeRaggedS32CompleteElementwiseSurface) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ragged_s32_complete_elementwise

elementwise {
  p0 = s32[127,65]{1,0} parameter(0)
  p1 = s32[127,65]{1,0} parameter(1)
  absolute = s32[127,65]{1,0} abs(p0)
  negated = s32[127,65]{1,0} negate(p1)
  inverted = s32[127,65]{1,0} not(p0)
  sum = s32[127,65]{1,0} add(absolute, negated)
  difference = s32[127,65]{1,0} subtract(sum, p1)
  product = s32[127,65]{1,0} multiply(difference, p0)
  one = s32[] constant(1)
  ones = s32[127,65]{1,0} broadcast(one), dimensions={}
  abs_denominator = s32[127,65]{1,0} abs(p1)
  denominator = s32[127,65]{1,0} maximum(abs_denominator, ones)
  quotient = s32[127,65]{1,0} divide(product, denominator)
  remainder = s32[127,65]{1,0} remainder(product, denominator)
  minimum = s32[127,65]{1,0} minimum(quotient, remainder)
  anded = s32[127,65]{1,0} and(minimum, inverted)
  ored = s32[127,65]{1,0} or(anded, p1)
  xored = s32[127,65]{1,0} xor(ored, p0)
  less = pred[127,65]{1,0} compare(sum, difference), direction=LT
  equal = pred[127,65]{1,0} compare(quotient, remainder), direction=EQ
  not_equal = pred[127,65]{1,0} not(equal)
  both = pred[127,65]{1,0} and(less, not_equal)
  either = pred[127,65]{1,0} or(less, equal)
  decision = pred[127,65]{1,0} xor(both, either)
  selected = s32[127,65]{1,0} select(decision, xored, product)
  lower = s32[] constant(-1000)
  lowers = s32[127,65]{1,0} broadcast(lower), dimensions={}
  upper = s32[] constant(1000)
  uppers = s32[127,65]{1,0} broadcast(upper), dimensions={}
  ROOT result = s32[127,65]{1,0} clamp(lowers, selected, uppers)
}

ENTRY main {
  p0 = s32[127,65]{1,0} parameter(0)
  p1 = s32[127,65]{1,0} parameter(1)
  ROOT fusion = s32[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRaggedS8AndS16ElementwiseVectors) {
  constexpr absl::string_view kS8Hlo = R"(
HloModule fly_ragged_s8_elementwise

elementwise {
  p0 = s8[127,65]{1,0} parameter(0)
  p1 = s8[127,65]{1,0} parameter(1)
  sum = s8[127,65]{1,0} add(p0, p1)
  inverted = s8[127,65]{1,0} not(p0)
  mixed = s8[127,65]{1,0} xor(sum, inverted)
  less = pred[127,65]{1,0} compare(p0, p1), direction=LT
  ROOT result = s8[127,65]{1,0} select(less, mixed, sum)
}

ENTRY main {
  p0 = s8[127,65]{1,0} parameter(0)
  p1 = s8[127,65]{1,0} parameter(1)
  ROOT fusion = s8[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kS8Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kS16Hlo = R"(
HloModule fly_ragged_s16_elementwise

elementwise {
  p0 = s16[127,65]{1,0} parameter(0)
  p1 = s16[127,65]{1,0} parameter(1)
  difference = s16[127,65]{1,0} subtract(p0, p1)
  product = s16[127,65]{1,0} multiply(difference, p0)
  maximum = s16[127,65]{1,0} maximum(product, p1)
  ROOT result = s16[127,65]{1,0} minimum(maximum, p0)
}

ENTRY main {
  p0 = s16[127,65]{1,0} parameter(0)
  p1 = s16[127,65]{1,0} parameter(1)
  ROOT fusion = s16[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kS16Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeMixedS32F32Conversions) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_mixed_s32_f32_conversions

elementwise {
  p0 = s32[127,65]{1,0} parameter(0)
  p1 = f32[127,65]{1,0} parameter(1)
  converted = f32[127,65]{1,0} convert(p0)
  finite = pred[127,65]{1,0} compare(p1, p1), direction=EQ
  zero = f32[] constant(0)
  zeros = f32[127,65]{1,0} broadcast(zero), dimensions={}
  sanitized = f32[127,65]{1,0} select(finite, p1, zeros)
  sum = f32[127,65]{1,0} add(converted, sanitized)
  lower = f32[] constant(-1000)
  lowers = f32[127,65]{1,0} broadcast(lower), dimensions={}
  upper = f32[] constant(1000)
  uppers = f32[127,65]{1,0} broadcast(upper), dimensions={}
  bounded = f32[127,65]{1,0} clamp(lowers, sum, uppers)
  ROOT result = s32[127,65]{1,0} convert(bounded)
}

ENTRY main {
  p0 = s32[127,65]{1,0} parameter(0)
  p1 = f32[127,65]{1,0} parameter(1)
  ROOT fusion = s32[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeFp8InputAndOutputConversions) {
  constexpr absl::string_view kInputHlo = R"(
HloModule fly_native_fp8_input_conversion

elementwise {
  p0 = f8e4m3fnuz[127,65]{1,0} parameter(0)
  converted = f32[127,65]{1,0} convert(p0)
  ROOT result = f32[127,65]{1,0} abs(converted)
}

ENTRY main {
  p0 = f8e4m3fnuz[127,65]{1,0} parameter(0)
  ROOT fusion = f32[127,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kInputHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kOutputHlo = R"(
HloModule fly_native_fp8_output_conversion

elementwise {
  p0 = f32[127,65]{1,0} parameter(0)
  converted = f8e5m2fnuz[127,65]{1,0} convert(p0)
  ROOT result = f8e5m2fnuz[127,65]{1,0} abs(converted)
}

ENTRY main {
  p0 = f32[127,65]{1,0} parameter(0)
  ROOT fusion = f8e5m2fnuz[127,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kOutputHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kOcpHlo = R"(
HloModule fly_native_ocp_fp8_conversions

elementwise {
  p0 = f8e4m3fn[127,65]{1,0} parameter(0)
  converted = f32[127,65]{1,0} convert(p0)
  ROOT result = f8e5m2[127,65]{1,0} convert(converted)
}

ENTRY main {
  p0 = f8e4m3fn[127,65]{1,0} parameter(0)
  ROOT fusion = f8e5m2[127,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kOcpHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeFp8ValueOperations) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_fp8_value_operations

elementwise {
  predicate = pred[127,65]{1,0} parameter(0)
  p0 = f8e4m3fnuz[127,65]{1,0} parameter(1)
  reduced = f8e4m3fnuz[127,65]{1,0} reduce-precision(p0),
    exponent_bits=3, mantissa_bits=1
  fallback = f8e4m3fnuz[] constant(-1.5)
  fallbacks = f8e4m3fnuz[127,65]{1,0}
    broadcast(fallback), dimensions={}
  selected = f8e4m3fnuz[127,65]{1,0}
    select(predicate, reduced, fallbacks)
  ROOT result = f8e4m3fnuz[127,65]{1,0} abs(selected)
}

ENTRY main {
  predicate = pred[127,65]{1,0} parameter(0)
  p0 = f8e4m3fnuz[127,65]{1,0} parameter(1)
  ROOT fusion = f8e4m3fnuz[127,65]{1,0}
    fusion(predicate, p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4InputConversionWithOddTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_input_conversion

elementwise {
  p0 = s4[127,65]{1,0:E(4)} parameter(0)
  widened = s8[127,65]{1,0} convert(p0)
  converted = bf16[127,65]{1,0} convert(widened)
  magnitude = bf16[127,65]{1,0} abs(converted)
  scale = bf16[] constant(0.5)
  scales = bf16[127,65]{1,0} broadcast(scale), dimensions={}
  ROOT result = bf16[127,65]{1,0} multiply(magnitude, scales)
}

ENTRY main {
  p0 = s4[127,65]{1,0:E(4)} parameter(0)
  ROOT fusion = bf16[127,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4PhysicalViewsWithOddTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_physical_views

elementwise {
  p0 = s4[127,65]{1,0:E(4)} parameter(0)
  flattened = s4[8255]{0:E(4)} bitcast(p0)
  widened = s8[8255]{0} convert(flattened)
  converted = bf16[8255]{0} convert(widened)
  ROOT result = bf16[8255]{0} abs(converted)
}

ENTRY main {
  p0 = s4[127,65]{1,0:E(4)} parameter(0)
  ROOT fusion = bf16[8255]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4OddOffsetRectangularSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_odd_offset_slice

elementwise {
  p0 = s4[127,66]{1,0:E(4)} parameter(0)
  sliced = s4[127,65]{1,0:E(4)} slice(p0),
    slice={[0:127], [1:66]}
  widened = s8[127,65]{1,0} convert(sliced)
  ROOT result = bf16[127,65]{1,0} convert(widened)
}

ENTRY main {
  p0 = s4[127,66]{1,0:E(4)} parameter(0)
  ROOT fusion = bf16[127,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4OddOffsetDynamicSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_odd_offset_dynamic_slice

elementwise {
  p0 = s4[129,67]{1,0:E(4)} parameter(0)
  row = s32[] constant(0)
  column = s32[] constant(1)
  sliced = s4[127,65]{1,0:E(4)} dynamic-slice(p0, row, column),
    dynamic_slice_sizes={127,65}
  widened = s8[127,65]{1,0} convert(sliced)
  ROOT result = bf16[127,65]{1,0} convert(widened)
}

ENTRY main {
  p0 = s4[129,67]{1,0:E(4)} parameter(0)
  ROOT fusion = bf16[127,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4ReversePadAndConcatenate) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_reverse_pad_concatenate

elementwise {
  p0 = s4[8255]{0:E(4)} parameter(0)
  reversed = s4[8255]{0:E(4)} reverse(p0), dimensions={0}
  widened = s8[8255]{0} convert(reversed)
  converted = bf16[8255]{0} convert(widened)
  zero = bf16[] constant(0)
  padded = bf16[8257]{0} pad(converted, zero), padding=1_1
  ROOT result = bf16[16514]{0}
    concatenate(padded, padded), dimensions={0}
}

ENTRY main {
  p0 = s4[8255]{0:E(4)} parameter(0)
  ROOT fusion = bf16[16514]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4DynamicUpdateSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_dynamic_update_slice

elementwise {
  base_s4 = s4[67]{0:E(4)} parameter(0)
  update_s4 = s4[9]{0:E(4)} parameter(1)
  base_s8 = s8[67]{0} convert(base_s4)
  update_s8 = s8[9]{0} convert(update_s4)
  base = bf16[67]{0} convert(base_s8)
  update = bf16[9]{0} convert(update_s8)
  start = s32[] constant(5)
  ROOT result = bf16[67]{0}
    dynamic-update-slice(base, update, start)
}

ENTRY main {
  base = s4[67]{0:E(4)} parameter(0)
  update = s4[9]{0:E(4)} parameter(1)
  ROOT fusion = bf16[67]{0} fusion(base, update), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4BitcastConvert) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_bitcast_convert

elementwise {
  p0 = s4[65,4]{1,0:E(4)} parameter(0)
  widened = s8[65,4]{1,0} convert(p0)
  ROOT result = s32[65]{0} bitcast-convert(widened)
}

ENTRY main {
  p0 = s4[65,4]{1,0:E(4)} parameter(0)
  ROOT fusion = s32[65]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4ReduceWindow) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_reduce_window

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT result = bf16[] add(lhs, rhs)
}

elementwise {
  p0 = s4[67]{0:E(4)} parameter(0)
  widened = s8[67]{0} convert(p0)
  converted = bf16[67]{0} convert(widened)
  zero = bf16[] constant(0)
  ROOT result = bf16[67]{0} reduce-window(converted, zero),
    window={size=3 pad=1_1}, to_apply=add
}

ENTRY main {
  p0 = s4[67]{0:E(4)} parameter(0)
  ROOT fusion = bf16[67]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4OutputConversionWithOddTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_output_conversion

elementwise {
  p0 = s8[127,65]{1,0} parameter(0)
  bias = s8[] constant(9)
  biases = s8[127,65]{1,0} broadcast(bias), dimensions={}
  shifted = s8[127,65]{1,0} add(p0, biases)
  ROOT result = s4[127,65]{1,0:E(4)} convert(shifted)
}

ENTRY main {
  p0 = s8[127,65]{1,0} parameter(0)
  ROOT fusion = s4[127,65]{1,0:E(4)} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4OutputFromS64) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_output_from_s64

elementwise {
  p0 = s64[65]{0} parameter(0)
  ROOT result = s4[65]{0:E(4)} convert(p0)
}

ENTRY main {
  p0 = s64[65]{0} parameter(0)
  ROOT fusion = s4[65]{0:E(4)} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"16"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4OutputAligned) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_output_aligned

elementwise {
  p0 = s8[128]{0} parameter(0)
  bias = s8[] constant(9)
  biases = s8[128]{0} broadcast(bias), dimensions={}
  shifted = s8[128]{0} add(p0, biases)
  ROOT result = s4[128]{0:E(4)} convert(shifted)
}

ENTRY main {
  p0 = s8[128]{0} parameter(0)
  ROOT fusion = s4[128]{0:E(4)} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativePackedS4OutputFromF32ClampsSpecialValues) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_s4_output_from_f32

elementwise {
  p0 = f32[9]{0} parameter(0)
  ROOT result = s4[9]{0:E(4)} convert(p0)
}

ENTRY main {
  values = f32[9]{0} constant({-inf, -9, -8, -7.9, 0, nan, 6.9, 7, inf})
  ROOT fusion = s4[9]{0:E(4)} fusion(values), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"16"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeExternalPredicateBuffersAndConversions) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_external_predicate_buffers

elementwise {
  p0 = pred[127,65]{1,0} parameter(0)
  p1 = s32[127,65]{1,0} parameter(1)
  p0_s32 = s32[127,65]{1,0} convert(p0)
  sum = s32[127,65]{1,0} add(p0_s32, p1)
  sum_pred = pred[127,65]{1,0} convert(sum)
  inverted = pred[127,65]{1,0} not(p0)
  not_sum = pred[127,65]{1,0} not(sum_pred)
  both = pred[127,65]{1,0} and(sum_pred, p0)
  neither = pred[127,65]{1,0} and(not_sum, inverted)
  ROOT result = pred[127,65]{1,0} or(both, neither)
}

ENTRY main {
  p0 = pred[127,65]{1,0} parameter(0)
  p1 = s32[127,65]{1,0} parameter(1)
  ROOT fusion = pred[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeTritonPredicateSurface) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_triton_predicate_surface

elementwise {
  p0 = pred[127,65]{1,0} parameter(0)
  p1 = pred[127,65]{1,0} parameter(1)
  sum = pred[127,65]{1,0} add(p0, p1)
  product = pred[127,65]{1,0} multiply(p0, p1)
  maximum = pred[127,65]{1,0} maximum(sum, product)
  minimum = pred[127,65]{1,0} minimum(maximum, p0)
  lower = pred[] constant(false)
  lowers = pred[127,65]{1,0} broadcast(lower), dimensions={}
  upper = pred[] constant(true)
  uppers = pred[127,65]{1,0} broadcast(upper), dimensions={}
  clamped = pred[127,65]{1,0} clamp(lowers, minimum, uppers)
  equal = pred[127,65]{1,0} compare(p0, p1), direction=EQ
  ROOT result = pred[127,65]{1,0} select(equal, clamped, sum)
}

ENTRY main {
  p0 = pred[127,65]{1,0} parameter(0)
  p1 = pred[127,65]{1,0} parameter(1)
  ROOT fusion = pred[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeMultidimensionalIotaDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_multidimensional_iota

elementwise {
  iota = s16[5,7,65,3]{3,2,1,0} iota(), iota_dimension=1
  converted = f64[5,7,65,3]{3,2,1,0} convert(iota)
  half = f64[] constant(0.5)
  half_broadcast = f64[5,7,65,3]{3,2,1,0} broadcast(half), dimensions={}
  ROOT result = f64[5,7,65,3]{3,2,1,0}
    multiply(converted, half_broadcast)
}

ENTRY main {
  ROOT fusion = f64[5,7,65,3]{3,2,1,0} fusion(), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeShapeChangingReshapeView) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_shape_changing_reshape_view

elementwise {
  p0 = s32[65,17]{1,0} parameter(0)
  view = s32[17,65]{1,0} reshape(p0)
  ROOT result = s32[17,65]{1,0} negate(view)
}

ENTRY main {
  p0 = s32[65,17]{1,0} parameter(0)
  ROOT fusion = s32[17,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeEffectiveTransposeView) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_effective_transpose_view

elementwise {
  p0 = f32[13,67]{1,0} parameter(0)
  view = f32[67,13]{0,1} transpose(p0), dimensions={1,0}
  ROOT result = f32[67,13]{0,1} abs(view)
}

ENTRY main {
  p0 = f32[13,67]{1,0} parameter(0)
  ROOT fusion = f32[67,13]{0,1} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeSameWidthTypeChangingBitcast) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_same_width_type_bitcast

elementwise {
  p0 = f32[65]{0} parameter(0)
  bits = s32[65]{0} bitcast(p0)
  ROOT result = s32[65]{0} not(bits)
}

ENTRY main {
  p0 = f32[65]{0} parameter(0)
  ROOT fusion = s32[65]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeNarrowingBitcastConvertWithTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_narrowing_bitcast_convert

elementwise {
  p0 = s32[65]{0} parameter(0)
  bytes = s8[65,4]{1,0} bitcast-convert(p0)
  ROOT result = s8[65,4]{1,0} not(bytes)
}

ENTRY main {
  p0 = s32[65]{0} parameter(0)
  ROOT fusion = s8[65,4]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeWideningBitcastConvertWithTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_widening_bitcast_convert

elementwise {
  p0 = s8[65,4]{1,0} parameter(0)
  words = s32[65]{0} bitcast-convert(p0)
  ROOT result = s32[65]{0} not(words)
}

ENTRY main {
  p0 = s8[65,4]{1,0} parameter(0)
  ROOT fusion = s32[65]{0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRaggedS64ElementwiseDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ragged_s64_elementwise

elementwise {
  p0 = s64[127,65]{1,0} parameter(0)
  p1 = s64[127,65]{1,0} parameter(1)
  absolute = s64[127,65]{1,0} abs(p0)
  negated = s64[127,65]{1,0} negate(p1)
  sum = s64[127,65]{1,0} add(absolute, negated)
  inverted = s64[127,65]{1,0} not(p1)
  mixed = s64[127,65]{1,0} xor(sum, inverted)
  less = pred[127,65]{1,0} compare(p0, p1), direction=LT
  ROOT result = s64[127,65]{1,0} select(less, mixed, sum)
}

ENTRY main {
  p0 = s64[127,65]{1,0} parameter(0)
  p1 = s64[127,65]{1,0} parameter(1)
  ROOT fusion = s64[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeWideF64ToNarrowS8Conversion) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_wide_f64_to_narrow_s8

elementwise {
  p0 = f64[65]{0} parameter(0)
  offset = f64[] constant(37)
  offsets = f64[65]{0} broadcast(offset), dimensions={}
  shifted = f64[65]{0} add(p0, offsets)
  lower = f64[] constant(-100)
  lowers = f64[65]{0} broadcast(lower), dimensions={}
  upper = f64[] constant(100)
  uppers = f64[65]{0} broadcast(upper), dimensions={}
  bounded = f64[65]{0} clamp(lowers, shifted, uppers)
  ROOT result = s8[65]{0} convert(bounded)
}

ENTRY main {
  p0 = f64[65]{0} parameter(0)
  ROOT fusion = s8[65]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"16"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRaggedF64ElementwiseDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_ragged_f64_elementwise

elementwise {
  p0 = f64[127,65]{1,0} parameter(0)
  p1 = f64[127,65]{1,0} parameter(1)
  absolute = f64[127,65]{1,0} abs(p0)
  sum = f64[127,65]{1,0} add(absolute, p1)
  sine = f64[127,65]{1,0} sine(sum)
  zero = f64[] constant(0)
  zeros = f64[127,65]{1,0} broadcast(zero), dimensions={}
  positive = pred[127,65]{1,0} compare(sine, zeros), direction=GT
  negated = f64[127,65]{1,0} negate(sine)
  ROOT result = f64[127,65]{1,0} select(positive, sine, negated)
}

ENTRY main {
  p0 = f64[127,65]{1,0} parameter(0)
  p1 = f64[127,65]{1,0} parameter(1)
  ROOT fusion = f64[127,65]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-9, /*arel=*/1e-9}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16CompleteFloatingMathSurface) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_complete_floating_math

elementwise {
  p0 = bf16[65]{0} parameter(0)
  lower = bf16[] constant(-0.75)
  lowers = bf16[65]{0} broadcast(lower), dimensions={}
  upper = bf16[] constant(0.75)
  uppers = bf16[65]{0} broadcast(upper), dimensions={}
  bounded = bf16[65]{0} clamp(lowers, p0, uppers)
  absolute = bf16[65]{0} abs(p0)
  offset = bf16[] constant(1.25)
  offsets = bf16[65]{0} broadcast(offset), dimensions={}
  positive = bf16[65]{0} add(absolute, offsets)
  acos = bf16[65]{0} acos(bounded)
  acosh = bf16[65]{0} acosh(positive)
  asin = bf16[65]{0} asin(bounded)
  asinh = bf16[65]{0} asinh(bounded)
  atanh = bf16[65]{0} atanh(bounded)
  cbrt = bf16[65]{0} cbrt(bounded)
  ceil = bf16[65]{0} ceil(p0)
  cosine = bf16[65]{0} cosine(bounded)
  cosh = bf16[65]{0} cosh(bounded)
  erf = bf16[65]{0} erf(bounded)
  expm1 = bf16[65]{0} exponential-minus-one(bounded)
  floor = bf16[65]{0} floor(p0)
  log1p = bf16[65]{0} log-plus-one(bounded)
  round = bf16[65]{0} round-nearest-even(p0)
  sine = bf16[65]{0} sine(bounded)
  sinh = bf16[65]{0} sinh(bounded)
  tan = bf16[65]{0} tan(bounded)
  reduced = bf16[65]{0} reduce-precision(p0),
    exponent_bits=5, mantissa_bits=5
  atan2 = bf16[65]{0} atan2(bounded, positive)
  power = bf16[65]{0} power(positive, bounded)
  remainder = bf16[65]{0} remainder(p0, positive)
  ROOT tuple = (bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0},
    bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0},
    bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0},
    bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0},
    bf16[65]{0}, bf16[65]{0})
    tuple(acos, acosh, asin, asinh, atanh, cbrt, ceil, cosine, cosh, erf,
      expm1, floor, log1p, round, sine, sinh, tan, reduced, atan2, power,
      remainder)
}

ENTRY main {
  p0 = bf16[65]{0} parameter(0)
  ROOT fusion = (bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0},
    bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0},
    bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0},
    bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0}, bf16[65]{0},
    bf16[65]{0}, bf16[65]{0}) fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.02, /*arel=*/0.03}));
}

TEST_F(FlyFusionDeviceTest, NativeTinyF32ElementwiseTail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_tiny_f32_native_elementwise

elementwise {
  p0 = f32[3]{0} parameter(0)
  absolute = f32[3]{0} abs(p0)
  one = f32[] constant(1)
  ones = f32[3]{0} broadcast(one), dimensions={}
  ROOT result = f32[3]{0} add(absolute, ones)
}

ENTRY main {
  p0 = f32[3]{0} parameter(0)
  ROOT fusion = f32[3]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"1", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(FlyFusionDeviceTest, NativeMultiOutputBf16ElementwiseDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_native_multi_output_elementwise

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  sum = bf16[128,64]{1,0} add(p0, p1)
  product = bf16[128,64]{1,0} multiply(sum, p0)
  ROOT tuple = (bf16[128,64]{1,0}, bf16[128,64]{1,0}) tuple(sum, product)
}

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = (bf16[128,64]{1,0}, bf16[128,64]{1,0})
    fusion(p0, p1), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16CompareSelectClampDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_compare_select_clamp

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  absolute = bf16[128,64]{1,0} abs(p0)
  zero = bf16[] constant(0)
  zero_broadcast = bf16[128,64]{1,0} broadcast(zero), dimensions={}
  compare = pred[128,64]{1,0} compare(p1, zero_broadcast), direction=GT
  lower = bf16[] constant(-1)
  lower_broadcast = bf16[128,64]{1,0} broadcast(lower), dimensions={}
  upper = bf16[] constant(1)
  upper_broadcast = bf16[128,64]{1,0} broadcast(upper), dimensions={}
  clamped = bf16[128,64]{1,0} clamp(lower_broadcast, p1, upper_broadcast)
  ROOT result = bf16[128,64]{1,0} select(compare, absolute, clamped)
}

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  p1 = bf16[128,64]{1,0} parameter(1)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16MixedPrecisionSigmoidDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_sigmoid

elementwise {
  p0 = bf16[128,64]{1,0} parameter(0)
  converted = f32[128,64]{1,0} convert(p0)
  negated = f32[128,64]{1,0} negate(converted)
  exponential = f32[128,64]{1,0} exponential(negated)
  one = f32[] constant(1)
  one_broadcast = f32[128,64]{1,0} broadcast(one), dimensions={}
  denominator = f32[128,64]{1,0} add(exponential, one_broadcast)
  sigmoid = f32[128,64]{1,0} divide(one_broadcast, denominator)
  ROOT result = bf16[128,64]{1,0} convert(sigmoid)
}

ENTRY main {
  p0 = bf16[128,64]{1,0} parameter(0)
  ROOT fusion = bf16[128,64]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.002, /*arel=*/0.01}));
}

TEST_F(FlyFusionDeviceTest, NativeF32TranscendentalDag) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_transcendentals

elementwise {
  p0 = f32[128,64]{1,0} parameter(0)
  absolute = f32[128,64]{1,0} abs(p0)
  one = f32[] constant(1)
  one_broadcast = f32[128,64]{1,0} broadcast(one), dimensions={}
  positive = f32[128,64]{1,0} add(absolute, one_broadcast)
  logarithm = f32[128,64]{1,0} log(positive)
  square_root = f32[128,64]{1,0} sqrt(positive)
  reciprocal_square_root = f32[128,64]{1,0} rsqrt(positive)
  hyperbolic_tangent = f32[128,64]{1,0} tanh(logarithm)
  sum = f32[128,64]{1,0} add(square_root, reciprocal_square_root)
  ROOT result = f32[128,64]{1,0} add(sum, hyperbolic_tangent)
}

ENTRY main {
  p0 = f32[128,64]{1,0} parameter(0)
  ROOT fusion = f32[128,64]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(FlyFusionDeviceTest, F16Softmax31x125Tail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f16_softmax_tail

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
  p0 = f16[31,125]{1,0} parameter(0)
  converted = f32[31,125]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[31]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[31,125]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[31,125]{1,0} subtract(converted, broadcast_max)
  exponential = f32[31,125]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[31]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[31,125]{1,0} broadcast(row_sum), dimensions={0}
  normalized = f32[31,125]{1,0} divide(exponential, broadcast_sum)
  ROOT result = f16[31,125]{1,0} convert(normalized)
}

ENTRY main {
  p0 = f16[31,125]{1,0} parameter(0)
  ROOT fusion = f16[31,125]{1,0} fusion(p0), kind=kCustom,
    calls=softmax,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","125"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0.001, /*arel=*/0.01}));
}

TEST_F(FlyFusionDeviceTest, F32Softmax31x125Tail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_softmax_tail

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
  p0 = f32[31,125]{1,0} parameter(0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[31]{0} reduce(p0, minus_inf), dimensions={1},
    to_apply=maximum
  broadcast_max = f32[31,125]{1,0} broadcast(row_max), dimensions={0}
  shifted = f32[31,125]{1,0} subtract(p0, broadcast_max)
  exponential = f32[31,125]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[31]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[31,125]{1,0} broadcast(row_sum), dimensions={0}
  ROOT result = f32[31,125]{1,0} divide(exponential, broadcast_sum)
}

ENTRY main {
  p0 = f32[31,125]{1,0} parameter(0)
  ROOT fusion = f32[31,125]{1,0} fusion(p0), kind=kCustom,
    calls=softmax,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","125"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/0.0001, /*arel=*/0.001}));
}

TEST_F(FlyFusionDeviceTest, F32ExternalRowOffsetSoftmax31x125Tail) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_f32_external_row_offset_softmax_tail

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

softmax {
  input = f32[31,125]{1,0} parameter(0)
  row_offset = f32[31]{0} parameter(1)
  row_offsets = f32[31,125]{1,0} broadcast(row_offset), dimensions={0}
  shifted = f32[31,125]{1,0} subtract(input, row_offsets)
  exponential = f32[31,125]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[31]{0} reduce(exponential, zero), dimensions={1}, to_apply=add
  broadcast_sum = f32[31,125]{1,0} broadcast(row_sum), dimensions={0}
  ROOT result = f32[31,125]{1,0} divide(exponential, broadcast_sum)
}

ENTRY main {
  input = f32[31,125]{1,0} parameter(0)
  row_offset = f32[31]{0} parameter(1)
  ROOT fusion = f32[31,125]{1,0} fusion(input, row_offset), kind=kCustom,
    calls=softmax,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","125"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHlo, ErrorSpec{/*aabs=*/0.0001, /*arel=*/0.001}));
}

TEST_F(FlyFusionDeviceTest, Bf16Transpose128x192) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose

transpose {
  p0 = bf16[128,192]{1,0} parameter(0)
  ROOT result = bf16[192,128]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[128,192]{1,0} parameter(0)
  ROOT fusion = bf16[192,128]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64","64"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, Bf16Transpose32Tile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose_32_tile

transpose {
  p0 = bf16[96,160]{1,0} parameter(0)
  ROOT result = bf16[160,96]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[96,160]{1,0} parameter(0)
  ROOT fusion = bf16[160,96]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32","32"]}],
        "num_stages":"1", "num_warps":"1", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, Bf16Transpose128Tile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose_128_tile

transpose {
  p0 = bf16[256,384]{1,0} parameter(0)
  ROOT result = bf16[384,256]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[256,384]{1,0} parameter(0)
  ROOT fusion = bf16[384,256]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["128","128"]}],
        "num_stages":"1", "num_warps":"16", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, Bf16TransposeWideRectangularTile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose_wide_rectangular_tile

transpose {
  p0 = bf16[128,256]{1,0} parameter(0)
  ROOT result = bf16[256,128]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[128,256]{1,0} parameter(0)
  ROOT fusion = bf16[256,128]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32","128"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, Bf16TransposeTallRectangularTile) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose_tall_rectangular_tile

transpose {
  p0 = bf16[256,128]{1,0} parameter(0)
  ROOT result = bf16[128,256]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[256,128]{1,0} parameter(0)
  ROOT fusion = bf16[128,256]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["128","32"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, TransformerQkvSliceTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_transformer_qkv_slice_transpose

transpose {
  p0 = bf16[256,3072]{1,0} parameter(0)
  view = bf16[2,128,3,16,64]{4,3,2,1,0} bitcast(p0)
  q = bf16[2,128,1,16,64]{4,3,2,1,0} slice(view),
    slice={[0:2], [0:128], [1:2], [0:16], [0:64]}
  ROOT result = bf16[2,1,16,64,128]{4,3,2,1,0} transpose(q),
    dimensions={0,2,3,4,1}
}

ENTRY main {
  p0 = bf16[256,3072]{1,0} parameter(0)
  ROOT fusion = bf16[2,1,16,64,128]{4,3,2,1,0} fusion(p0),
    kind=kCustom, calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64","64"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, MultiOutputTransformerQkvTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_multi_output_transformer_qkv_transpose

transpose {
  p0 = bf16[256,3072]{1,0} parameter(0)
  view = bf16[2,128,3,16,64]{4,3,2,1,0} bitcast(p0)
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
  p0 = bf16[256,3072]{1,0} parameter(0)
  ROOT fusion = (bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0},
    bf16[2,1,16,64,128]{4,3,2,1,0}) fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64","64"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionAutotuningPipelineDeviceTest,
       FormsAndExecutesMultiOutputTransformerQkvTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_multi_output_transformer_qkv_transpose_pipeline

ENTRY main {
  p0 = bf16[256,3072]{1,0} parameter(0)
  view = bf16[2,128,3,16,64]{4,3,2,1,0} reshape(p0)
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
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized,
                       GetOptimizedModule(kHlo));
  const HloInstruction* qkv_fusion = nullptr;
  for (const HloInstruction* instruction :
       optimized->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kFusion &&
        instruction->IsMultiOutputFusion()) {
      qkv_fusion = instruction;
    }
  }
  ASSERT_NE(qkv_fusion, nullptr) << optimized->ToString();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       qkv_fusion->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), kFlyFusionKind)
      << optimized->ToString();
  EXPECT_EQ(backend_config.fusion_backend_config()
                .block_level_fusion_config()
                .output_tiles(0)
                .sizes_size(),
            2)
      << optimized->ToString();
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, TransformerContextTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_transformer_context_transpose

transpose {
  p0 = bf16[32,64,128]{2,1,0} parameter(0)
  view = bf16[2,16,64,128]{3,2,1,0} bitcast(p0)
  ROOT result = bf16[2,128,16,64]{3,2,1,0} transpose(view),
    dimensions={0,3,1,2}
}

ENTRY main {
  p0 = bf16[32,64,128]{2,1,0} parameter(0)
  ROOT fusion = bf16[2,128,16,64]{3,2,1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32","64"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, Bf16TransposePartialTiles) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_bf16_transpose_partial_tiles

transpose {
  p0 = bf16[65,127]{1,0} parameter(0)
  ROOT result = bf16[127,65]{1,0} transpose(p0), dimensions={1,0}
}

ENTRY main {
  p0 = bf16[65,127]{1,0} parameter(0)
  ROOT fusion = bf16[127,65]{1,0} fusion(p0), kind=kCustom,
    calls=transpose,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1","1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, GenericBf16Elementwise) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_bf16_elementwise

elementwise {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  add = bf16[128,192]{1,0} add(p0, p1)
  scale = bf16[] constant(1.5)
  broadcast = bf16[128,192]{1,0} broadcast(scale), dimensions={}
  multiply = bf16[128,192]{1,0} multiply(add, broadcast)
  ROOT result = bf16[128,192]{1,0} maximum(multiply, p0)
}

ENTRY main {
  p0 = bf16[128,192]{1,0} parameter(0)
  p1 = bf16[128,192]{1,0} parameter(1)
  ROOT fusion = bf16[128,192]{1,0} fusion(p0, p1), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeSmallSplitKResidualReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_small_split_k_residual

add_reduce {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

residual {
  base = bf16[2,17,259]{2,1,0} parameter(0)
  base_f32 = f32[2,17,259]{2,1,0} convert(base)
  partial4 = f32[4,34,259]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  sum4 = f32[34,259]{1,0} reduce(partial4, zero), dimensions={0},
      to_apply=add_reduce
  rounded4 = bf16[34,259]{1,0} convert(sum4)
  view4 = bf16[2,17,259]{2,1,0} bitcast(rounded4)
  value4 = f32[2,17,259]{2,1,0} convert(view4)
  partial2 = f32[2,34,259]{2,1,0} parameter(2)
  sum2 = f32[34,259]{1,0} reduce(partial2, zero), dimensions={0},
      to_apply=add_reduce
  rounded2 = bf16[34,259]{1,0} convert(sum2)
  view2 = bf16[2,17,259]{2,1,0} bitcast(rounded2)
  value2 = f32[2,17,259]{2,1,0} convert(view2)
  first = f32[2,17,259]{2,1,0} add(base_f32, value2)
  total = f32[2,17,259]{2,1,0} add(first, value4)
  ROOT result = bf16[2,17,259]{2,1,0} convert(total)
}

ENTRY main {
  base = bf16[2,17,259]{2,1,0} parameter(0)
  partial4 = f32[4,34,259]{2,1,0} parameter(1)
  partial2 = f32[2,34,259]{2,1,0} parameter(2)
  ROOT fusion = bf16[2,17,259]{2,1,0}
      fusion(base, partial4, partial2), kind=kCustom, calls=residual,
      backend_config={"fusion_backend_config":{
        "kind":"__fly",
        "block_level_fusion_config":{
          "output_tiles":[{"sizes":["1"]}],
          "num_stages":"1", "num_warps":"2", "num_ctas":"1",
          "vector_size_bits":"64"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeF32RowReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_f32_row_reduction

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
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(FlyFusionDeviceTest, NativeRowReductionsWithDynamicInit) {
  constexpr absl::string_view kF32Hlo = R"(
HloModule fly_native_f32_dynamic_init_row_reduction

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
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kF32Hlo, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));

  constexpr absl::string_view kBf16Hlo = R"(
HloModule fly_native_bf16_dynamic_init_row_minimum

minimum {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT min = bf16[] minimum(lhs, rhs)
}

reduction {
  p0 = bf16[65,259]{1,0} parameter(0)
  init = bf16[] parameter(1)
  ROOT result = bf16[65]{0} reduce(p0, init), dimensions={1},
    to_apply=minimum
}

ENTRY main {
  p0 = bf16[65,259]{1,0} parameter(0)
  init = bf16[] parameter(1)
  ROOT fusion = bf16[65]{0} fusion(p0, init), kind=kCustom,
    calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kBf16Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kS32Hlo = R"(
HloModule fly_native_s32_dynamic_init_row_maximum

maximum {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT max = s32[] maximum(lhs, rhs)
}

reduction {
  p0 = s32[65,259]{1,0} parameter(0)
  init = s32[] parameter(1)
  ROOT result = s32[65]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = s32[65,259]{1,0} parameter(0)
  init = s32[] parameter(1)
  ROOT fusion = s32[65]{0} fusion(p0, init), kind=kCustom,
    calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kS32Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeExternalRowBroadcastInReductionInput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_external_row_broadcast_input_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  input = f32[65,259]{1,0} parameter(0)
  row_offset = f32[65]{0} parameter(1)
  row_offsets = f32[65,259]{1,0} broadcast(row_offset), dimensions={0}
  shifted = f32[65,259]{1,0} subtract(input, row_offsets)
  exponentials = f32[65,259]{1,0} exponential(shifted)
  zero = f32[] constant(0)
  row_sum = f32[65]{0} reduce(exponentials, zero), dimensions={1},
    to_apply=add
  row_sums = f32[65,259]{1,0} broadcast(row_sum), dimensions={0}
  ROOT result = f32[65,259]{1,0} divide(exponentials, row_sums)
}

ENTRY main {
  input = f32[65,259]{1,0} parameter(0)
  row_offset = f32[65]{0} parameter(1)
  ROOT fusion = f32[65,259]{1,0} fusion(input, row_offset), kind=kCustom,
    calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(FlyFusionDeviceTest, NativeConvertedBf16RowMaximum) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_bf16_row_maximum

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(lhs, rhs)
}

reduction {
  p0 = bf16[65,512]{1,0} parameter(0)
  converted = f32[65,512]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  ROOT result = f32[65]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = bf16[65,512]{1,0} parameter(0)
  ROOT fusion = f32[65]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeNarrowingF16RowMaximum) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_narrowing_f16_row_maximum

maximum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(lhs, rhs)
}

reduction {
  p0 = f16[65,256]{1,0} parameter(0)
  converted = f32[65,256]{1,0} convert(p0)
  minus_inf = f32[] constant(-inf)
  row_max = f32[65]{0} reduce(converted, minus_inf), dimensions={1},
    to_apply=maximum
  ROOT result = f16[65]{0} convert(row_max)
}

ENTRY main {
  p0 = f16[65,256]{1,0} parameter(0)
  ROOT fusion = f16[65]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeDirectLowPrecisionRowReductions) {
  constexpr absl::string_view kF16AddHlo = R"(
HloModule fly_native_direct_f16_row_add

add {
  lhs = f16[] parameter(0)
  rhs = f16[] parameter(1)
  ROOT sum = f16[] add(lhs, rhs)
}

reduction {
  p0 = f16[127,259]{1,0} parameter(0)
  zero = f16[] constant(0)
  ROOT result = f16[127]{0} reduce(p0, zero), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = f16[127,259]{1,0} parameter(0)
  ROOT fusion = f16[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kF16AddHlo,
                                       ErrorSpec{/*aabs=*/0.5, /*arel=*/0.06}));

  constexpr absl::string_view kF16Hlo = R"(
HloModule fly_native_direct_f16_row_maximum

maximum {
  lhs = f16[] parameter(0)
  rhs = f16[] parameter(1)
  ROOT max = f16[] maximum(lhs, rhs)
}

reduction {
  p0 = f16[127,259]{1,0} parameter(0)
  minus_inf = f16[] constant(-inf)
  ROOT result = f16[127]{0} reduce(p0, minus_inf), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = f16[127,259]{1,0} parameter(0)
  ROOT fusion = f16[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kF16Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kBf16Hlo = R"(
HloModule fly_native_direct_bf16_row_minimum

minimum {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT min = bf16[] minimum(lhs, rhs)
}

reduction {
  p0 = bf16[127,259]{1,0} parameter(0)
  plus_inf = bf16[] constant(inf)
  ROOT result = bf16[127]{0} reduce(p0, plus_inf), dimensions={1},
    to_apply=minimum
}

ENTRY main {
  p0 = bf16[127,259]{1,0} parameter(0)
  ROOT fusion = bf16[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kBf16Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeDirectIntegerRowReductions) {
  constexpr absl::string_view kPredHlo = R"(
HloModule fly_native_direct_pred_row_maximum

maximum {
  lhs = pred[] parameter(0)
  rhs = pred[] parameter(1)
  ROOT max = pred[] maximum(lhs, rhs)
}

reduction {
  p0 = pred[127,259]{1,0} parameter(0)
  init = pred[] constant(false)
  ROOT result = pred[127]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = pred[127,259]{1,0} parameter(0)
  ROOT fusion = pred[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kPredHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kS8Hlo = R"(
HloModule fly_native_direct_s8_row_add

add {
  lhs = s8[] parameter(0)
  rhs = s8[] parameter(1)
  ROOT sum = s8[] add(lhs, rhs)
}

reduction {
  p0 = s8[127,259]{1,0} parameter(0)
  init = s8[] constant(0)
  ROOT result = s8[127]{0} reduce(p0, init), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = s8[127,259]{1,0} parameter(0)
  ROOT fusion = s8[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kS8Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kS16Hlo = R"(
HloModule fly_native_direct_s16_row_minimum

minimum {
  lhs = s16[] parameter(0)
  rhs = s16[] parameter(1)
  ROOT min = s16[] minimum(lhs, rhs)
}

reduction {
  p0 = s16[127,259]{1,0} parameter(0)
  init = s16[] constant(32767)
  ROOT result = s16[127]{0} reduce(p0, init), dimensions={1},
    to_apply=minimum
}

ENTRY main {
  p0 = s16[127,259]{1,0} parameter(0)
  ROOT fusion = s16[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kS16Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kS32Hlo = R"(
HloModule fly_native_direct_s32_row_maximum

maximum {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT max = s32[] maximum(lhs, rhs)
}

reduction {
  p0 = s32[127,259]{1,0} parameter(0)
  init = s32[] constant(-2147483648)
  ROOT result = s32[127]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = s32[127,259]{1,0} parameter(0)
  ROOT fusion = s32[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kS32Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kS32XorHlo = R"(
HloModule fly_native_direct_s32_row_xor

xor {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT result = s32[] xor(lhs, rhs)
}

reduction {
  p0 = s32[127,259]{1,0} parameter(0)
  init = s32[] constant(0)
  ROOT result = s32[127]{0} reduce(p0, init), dimensions={1}, to_apply=xor
}

ENTRY main {
  p0 = s32[127,259]{1,0} parameter(0)
  ROOT fusion = s32[127]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kS32XorHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kS64Hlo = R"(
HloModule fly_native_direct_s64_row_maximum

maximum {
  lhs = s64[] parameter(0)
  rhs = s64[] parameter(1)
  ROOT max = s64[] maximum(lhs, rhs)
}

reduction {
  p0 = s64[31,67]{1,0} parameter(0)
  init = s64[] constant(-9223372036854775808)
  ROOT result = s64[31]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = s64[31,67]{1,0} parameter(0)
  ROOT fusion = s64[31]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kS64Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));

  constexpr absl::string_view kF64Hlo = R"(
HloModule fly_native_direct_f64_row_maximum

maximum {
  lhs = f64[] parameter(0)
  rhs = f64[] parameter(1)
  ROOT max = f64[] maximum(lhs, rhs)
}

reduction {
  p0 = f64[31,67]{1,0} parameter(0)
  init = f64[] constant(-inf)
  ROOT result = f64[31]{0} reduce(p0, init), dimensions={1},
    to_apply=maximum
}

ENTRY main {
  p0 = f64[31,67]{1,0} parameter(0)
  ROOT fusion = f64[31]{0} fusion(p0), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
})";
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kF64Hlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeFusedBf16SquaredDifferenceReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_fused_bf16_squared_difference_reduction

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

reduction {
  p0 = bf16[65,512]{1,0} parameter(0)
  p1 = bf16[65,512]{1,0} parameter(1)
  lhs = f32[65,512]{1,0} convert(p0)
  rhs = f32[65,512]{1,0} convert(p1)
  difference = f32[65,512]{1,0} subtract(lhs, rhs)
  square = f32[65,512]{1,0} multiply(difference, difference)
  zero = f32[] constant(0)
  row_sum = f32[65]{0} reduce(square, zero), dimensions={1}, to_apply=add
  scale = f32[] constant(0.25)
  scales = f32[65]{0} broadcast(scale), dimensions={}
  ROOT result = f32[65]{0} multiply(row_sum, scales)
}

ENTRY main {
  p0 = bf16[65,512]{1,0} parameter(0)
  p1 = bf16[65,512]{1,0} parameter(1)
  ROOT fusion = f32[65]{0} fusion(p0, p1), kind=kCustom, calls=reduction,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(FlyFusionDeviceTest, NativeRank3Bf16RmsNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank3_bf16_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[2,17,259]{2,1,0} parameter(0)
  converted = f32[2,17,259]{2,1,0} convert(p0)
  squared = f32[2,17,259]{2,1,0} multiply(converted, converted)
  zero = f32[] constant(0)
  row_sum = f32[2,17]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[2,17]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,17]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,17]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,17]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,17]{1,0} rsqrt(variance)
  scales = f32[2,17,259]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,17,259]{2,1,0} multiply(converted, scales)
  ROOT result = bf16[2,17,259]{2,1,0} convert(normalized)
}

ENTRY main {
  p0 = bf16[2,17,259]{2,1,0} parameter(0)
  ROOT fusion = bf16[2,17,259]{2,1,0} fusion(p0), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(FlyFusionDeviceTest, NativeSingletonBitcastBf16RmsNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_singleton_bitcast_bf16_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[1,1,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  converted = f32[1,1,259]{2,1,0} convert(p0)
  squared = f32[1,1,259]{2,1,0} multiply(converted, converted)
  flat_square = f32[259]{0} bitcast(squared)
  zero = f32[] constant(0)
  row_sum = f32[] reduce(flat_square, zero), dimensions={0}, to_apply=add
  row_sum_view = f32[1,1]{1,0} bitcast(row_sum)
  reciprocal_width = f32[1,1]{1,0} constant({{0.003861003861003861}})
  mean_square = f32[1,1]{1,0} multiply(row_sum_view, reciprocal_width)
  epsilon = f32[1,1]{1,0} constant({{1e-06}})
  variance = f32[1,1]{1,0} add(mean_square, epsilon)
  reciprocal_stddev = f32[1,1]{1,0} rsqrt(variance)
  scalar_scale = f32[] bitcast(reciprocal_stddev)
  scales = f32[1,1,259]{2,1,0} broadcast(scalar_scale), dimensions={}
  normalized = f32[1,1,259]{2,1,0} multiply(converted, scales)
  weight_view = bf16[1,1,259]{2,1,0} bitcast(weight)
  weight_f32 = f32[1,1,259]{2,1,0} convert(weight_view)
  weighted = f32[1,1,259]{2,1,0} multiply(normalized, weight_f32)
  ROOT result = bf16[1,1,259]{2,1,0} convert(weighted)
}

ENTRY main {
  p0 = bf16[1,1,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  ROOT fusion = bf16[1,1,259]{2,1,0} fusion(p0, weight), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(FlyFusionDeviceTest, NativeBitcastBf16ColumnScaleRmsNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_bitcast_bf16_column_scale_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[4,1,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  converted = f32[4,1,259]{2,1,0} convert(p0)
  squared = f32[4,1,259]{2,1,0} multiply(converted, converted)
  flat_square = f32[4,259]{1,0} bitcast(squared)
  zero = f32[] constant(0)
  row_sum = f32[4]{0} reduce(flat_square, zero), dimensions={1}, to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[4]{0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[4]{0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[4]{0} broadcast(epsilon), dimensions={}
  variance = f32[4]{0} add(mean_square, epsilons)
  reciprocal_stddev = f32[4]{0} rsqrt(variance)
  scales = f32[4,1,259]{2,1,0} broadcast(reciprocal_stddev), dimensions={0}
  normalized = f32[4,1,259]{2,1,0} multiply(converted, scales)
  narrowed = bf16[4,1,259]{2,1,0} convert(normalized)
  normalized_view = bf16[4,259]{1,0} bitcast(narrowed)
  normalized_f32 = f32[4,259]{1,0} convert(normalized_view)
  weights = bf16[4,259]{1,0} broadcast(weight), dimensions={1}
  weights_f32 = f32[4,259]{1,0} convert(weights)
  weighted = f32[4,259]{1,0} multiply(normalized_f32, weights_f32)
  ROOT result = bf16[4,259]{1,0} convert(weighted)
}

ENTRY main {
  p0 = bf16[4,1,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  ROOT fusion = bf16[4,259]{1,0} fusion(p0, weight), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(FlyFusionDeviceTest, NativeFlattenedRank3Bf16ColumnScaleRmsNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_flattened_rank3_bf16_column_scale_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[2,3,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  converted = f32[2,3,259]{2,1,0} convert(p0)
  squared = f32[2,3,259]{2,1,0} multiply(converted, converted)
  zero = f32[] constant(0)
  row_sum = f32[2,3]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.003861003861003861)
  widths = f32[2,3]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,3]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,3]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,3]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,3]{1,0} rsqrt(variance)
  scales = f32[2,3,259]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,3,259]{2,1,0} multiply(converted, scales)
  narrowed = bf16[2,3,259]{2,1,0} convert(normalized)
  normalized_view = bf16[6,259]{1,0} bitcast(narrowed)
  normalized_f32 = f32[6,259]{1,0} convert(normalized_view)
  weights = bf16[6,259]{1,0} broadcast(weight), dimensions={1}
  weights_f32 = f32[6,259]{1,0} convert(weights)
  weighted = f32[6,259]{1,0} multiply(normalized_f32, weights_f32)
  ROOT result = bf16[6,259]{1,0} convert(weighted)
}

ENTRY main {
  p0 = bf16[2,3,259]{2,1,0} parameter(0)
  weight = bf16[259]{0} parameter(1)
  ROOT fusion = bf16[6,259]{1,0} fusion(p0, weight), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(FlyFusionDeviceTest, NativeBitcastReductionExponentialEpilogue) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_bitcast_reduction_exponential_epilogue

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
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"1", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(FlyFusionDeviceTest, NativePackedQkvSliceReduction) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_packed_qkv_slice_reduction

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
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"1", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2e-2, /*arel=*/2e-2}));
}

TEST_F(FlyFusionDeviceTest, NativePartitionedRank3Bf16RmsNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_partitioned_rank3_bf16_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  p0 = bf16[2,17,1024]{2,1,0} parameter(0)
  converted = f32[2,17,1024]{2,1,0} convert(p0)
  squared = f32[2,17,1024]{2,1,0} multiply(converted, converted)
  zero = f32[] constant(0)
  row_sum = f32[2,17]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.0009765625)
  widths = f32[2,17]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,17]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,17]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,17]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,17]{1,0} rsqrt(variance)
  scales = f32[2,17,1024]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,17,1024]{2,1,0} multiply(converted, scales)
  ROOT result = bf16[2,17,1024]{2,1,0} convert(normalized)
}

ENTRY main {
  p0 = bf16[2,17,1024]{2,1,0} parameter(0)
  ROOT fusion = bf16[2,17,1024]{2,1,0} fusion(p0), kind=kCustom,
    calls=rms_norm,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"32"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(FlyFusionDeviceTest, NativePartitionedResidualRmsNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_partitioned_residual_rms_norm

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

rms_norm {
  residual = bf16[2,17,1024]{2,1,0} parameter(0)
  partials = f32[2,34,1024]{2,1,0} parameter(1)
  zero = f32[] constant(0)
  projected = f32[34,1024]{1,0} reduce(partials, zero), dimensions={0},
    to_apply=add
  projected_bf16 = bf16[34,1024]{1,0} convert(projected)
  projected_view = bf16[2,17,1024]{2,1,0} bitcast(projected_bf16)
  residual_f32 = f32[2,17,1024]{2,1,0} convert(residual)
  projected_f32 = f32[2,17,1024]{2,1,0} convert(projected_view)
  added = f32[2,17,1024]{2,1,0} add(residual_f32, projected_f32)
  squared = f32[2,17,1024]{2,1,0} multiply(added, added)
  row_sum = f32[2,17]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.0009765625)
  widths = f32[2,17]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,17]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,17]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,17]{1,0} add(mean_square, epsilons)
  reciprocal_stddev = f32[2,17]{1,0} rsqrt(variance)
  scales = f32[2,17,1024]{2,1,0} broadcast(reciprocal_stddev),
    dimensions={0,1}
  normalized = f32[2,17,1024]{2,1,0} multiply(added, scales)
  ROOT result = bf16[2,17,1024]{2,1,0} convert(normalized)
}

ENTRY main {
  residual = bf16[2,17,1024]{2,1,0} parameter(0)
  partials = f32[2,34,1024]{2,1,0} parameter(1)
  ROOT fusion = bf16[2,17,1024]{2,1,0} fusion(residual, partials),
    kind=kCustom, calls=rms_norm,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/2e-2, /*arel=*/2e-2}));
}

TEST_F(FlyFusionDeviceTest, NativeBitcastResidualRmsStatistic) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_bitcast_residual_rms_statistic

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

statistic {
  flat_residual = bf16[256,1024]{1,0} parameter(0)
  input = bf16[2,128,1024]{2,1,0} parameter(1)
  residual = bf16[2,128,1024]{2,1,0} bitcast(flat_residual)
  input_f32 = f32[2,128,1024]{2,1,0} convert(input)
  residual_f32 = f32[2,128,1024]{2,1,0} convert(residual)
  added = f32[2,128,1024]{2,1,0} add(input_f32, residual_f32)
  squared = f32[2,128,1024]{2,1,0} multiply(added, added)
  zero = f32[] constant(0)
  row_sum = f32[2,128]{1,0} reduce(squared, zero), dimensions={2},
    to_apply=add
  reciprocal_width = f32[] constant(0.0009765625)
  widths = f32[2,128]{1,0} broadcast(reciprocal_width), dimensions={}
  mean_square = f32[2,128]{1,0} multiply(row_sum, widths)
  epsilon = f32[] constant(1e-06)
  epsilons = f32[2,128]{1,0} broadcast(epsilon), dimensions={}
  variance = f32[2,128]{1,0} add(mean_square, epsilons)
  ROOT result = f32[2,128]{1,0} rsqrt(variance)
}

ENTRY main {
  flat_residual = bf16[256,1024]{1,0} parameter(0)
  input = bf16[2,128,1024]{2,1,0} parameter(1)
  ROOT fusion = f32[2,128]{1,0} fusion(flat_residual, input),
    kind=kCustom, calls=statistic,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(FlyFusionDeviceTest, NativeDynamicSliceBitcast) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_dynamic_slice_bitcast

slice {
  p0 = f32[32,16]{1,0} parameter(0)
  offset = s32[] parameter(1)
  dynamic_slice = f32[16,16]{1,0} dynamic-slice(p0, offset, offset),
    dynamic_slice_sizes={16,16}
  ROOT result = f32[256]{0} bitcast(dynamic_slice)
}

ENTRY main {
  p0 = f32[32,16]{1,0} parameter(0)
  offset = s32[] parameter(1)
  ROOT fusion = f32[256]{0} fusion(p0, offset), kind=kCustom, calls=slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeS64DynamicSliceClampsConstants) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_s64_dynamic_slice_constants

slice {
  p0 = bf16[19,67]{1,0} parameter(0)
  row = s64[] constant(-4294967297)
  column = s64[] constant(4294967298)
  ROOT result = bf16[15,65]{1,0} dynamic-slice(p0, row, column),
    dynamic_slice_sizes={15,65}
}

ENTRY main {
  p0 = bf16[19,67]{1,0} parameter(0)
  ROOT fusion = bf16[15,65]{1,0} fusion(p0), kind=kCustom, calls=slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeInPlaceDynamicUpdateSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_in_place_dynamic_update_slice

update_slice {
  input = f32[64,96]{1,0} parameter(0)
  update = f32[17,31]{1,0} parameter(1)
  row = s32[] parameter(2)
  column = s32[] parameter(3)
  scale = f32[] constant(1.25)
  broadcast = f32[17,31]{1,0} broadcast(scale), dimensions={}
  scaled = f32[17,31]{1,0} multiply(update, broadcast)
  ROOT result = f32[64,96]{1,0} dynamic-update-slice(
      input, scaled, row, column)
}

ENTRY main {
  input = f32[64,96]{1,0} parameter(0)
  update = f32[17,31]{1,0} parameter(1)
  row = s32[] parameter(2)
  column = s32[] parameter(3)
  ROOT fusion = f32[64,96]{1,0} fusion(input, update, row, column),
    kind=kCustom, calls=update_slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeS64InPlaceDynamicUpdateSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_s64_in_place_dynamic_update_slice

update_slice {
  input = bf16[19,67]{1,0} parameter(0)
  update = bf16[15,65]{1,0} parameter(1)
  row = s64[] parameter(2)
  column = s64[] parameter(3)
  absolute = bf16[15,65]{1,0} abs(update)
  ROOT result = bf16[19,67]{1,0} dynamic-update-slice(
      input, absolute, row, column)
}

ENTRY main {
  input = bf16[19,67]{1,0} parameter(0)
  update = bf16[15,65]{1,0} parameter(1)
  row = s64[] parameter(2)
  column = s64[] parameter(3)
  ROOT fusion = bf16[19,67]{1,0}
    fusion(input, update, row, column), kind=kCustom, calls=update_slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest,
       NativeOutOfPlaceDynamicUpdateSliceClampsConstantStarts) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_out_of_place_dynamic_update_slice

update_slice {
  input = bf16[19,67]{1,0} parameter(0)
  update = bf16[15,65]{1,0} parameter(1)
  absolute = bf16[15,65]{1,0} abs(update)
  row = s32[] constant(-7)
  column = s32[] constant(200)
  result = bf16[19,67]{1,0} dynamic-update-slice(
      input, absolute, row, column)
  ROOT tuple = (bf16[19,67]{1,0}, bf16[19,67]{1,0})
      tuple(result, input)
}

ENTRY main {
  input = bf16[19,67]{1,0} parameter(0)
  update = bf16[15,65]{1,0} parameter(1)
  ROOT fusion = (bf16[19,67]{1,0}, bf16[19,67]{1,0})
    fusion(input, update), kind=kCustom, calls=update_slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRank4S64InPlaceDynamicUpdateSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank4_s64_in_place_dynamic_update_slice

update_slice {
  input = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  update = bf16[2,4,6,9]{3,2,1,0} parameter(1)
  start0 = s64[] parameter(2)
  start1 = s64[] parameter(3)
  start2 = s64[] parameter(4)
  start3 = s64[] parameter(5)
  absolute = bf16[2,4,6,9]{3,2,1,0} abs(update)
  ROOT result = bf16[4,6,8,11]{3,2,1,0} dynamic-update-slice(
    input, absolute, start0, start1, start2, start3)
}

ENTRY main {
  input = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  update = bf16[2,4,6,9]{3,2,1,0} parameter(1)
  start0 = s64[] parameter(2)
  start1 = s64[] parameter(3)
  start2 = s64[] parameter(4)
  start3 = s64[] parameter(5)
  ROOT fusion = bf16[4,6,8,11]{3,2,1,0} fusion(
    input, update, start0, start1, start2, start3), kind=kCustom,
    calls=update_slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest,
       NativeRank4OutOfPlaceDynamicUpdateSliceClampsConstants) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank4_out_of_place_dynamic_update_slice

update_slice {
  input = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  update = bf16[2,4,6,9]{3,2,1,0} parameter(1)
  absolute = bf16[2,4,6,9]{3,2,1,0} abs(update)
  start0 = s64[] constant(-4294967297)
  start1 = s64[] constant(4294967298)
  start2 = s64[] constant(1)
  start3 = s64[] constant(2)
  result = bf16[4,6,8,11]{3,2,1,0} dynamic-update-slice(
    input, absolute, start0, start1, start2, start3)
  ROOT tuple = (bf16[4,6,8,11]{3,2,1,0}, bf16[4,6,8,11]{3,2,1,0})
    tuple(result, input)
}

ENTRY main {
  input = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  update = bf16[2,4,6,9]{3,2,1,0} parameter(1)
  ROOT fusion = (bf16[4,6,8,11]{3,2,1,0},
                 bf16[4,6,8,11]{3,2,1,0}) fusion(input, update),
    kind=kCustom, calls=update_slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRank1DynamicUpdateSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank1_dynamic_update_slice

update_slice {
  input = bf16[67]{0} parameter(0)
  update = bf16[65]{0} parameter(1)
  start = s32[] parameter(2)
  absolute = bf16[65]{0} abs(update)
  ROOT result = bf16[67]{0} dynamic-update-slice(input, absolute, start)
}

ENTRY main {
  input = bf16[67]{0} parameter(0)
  update = bf16[65]{0} parameter(1)
  start = s32[] parameter(2)
  ROOT fusion = bf16[67]{0} fusion(input, update, start), kind=kCustom,
    calls=update_slice,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16Concatenate) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_bf16_concatenate

concatenate {
  p0 = bf16[17]{0} parameter(0)
  p1 = bf16[16]{0} parameter(1)
  p2 = bf16[31]{0} parameter(2)
  abs0 = bf16[17]{0} abs(p0)
  negate = bf16[16]{0} negate(p1)
  abs2 = bf16[31]{0} abs(p2)
  ROOT result = bf16[64]{0} concatenate(abs0, negate, abs2), dimensions={0}
}

ENTRY main {
  p0 = bf16[17]{0} parameter(0)
  p1 = bf16[16]{0} parameter(1)
  p2 = bf16[31]{0} parameter(2)
  ROOT fusion = bf16[64]{0} fusion(p0, p1, p2), kind=kCustom,
    calls=concatenate,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16MiddleDimensionConcatenate) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_bf16_middle_dimension_concatenate

concatenate {
  p0 = bf16[3,5,17]{2,1,0} parameter(0)
  p1 = bf16[3,7,17]{2,1,0} parameter(1)
  p2 = bf16[3,2,17]{2,1,0} parameter(2)
  abs0 = bf16[3,5,17]{2,1,0} abs(p0)
  negate = bf16[3,7,17]{2,1,0} negate(p1)
  abs2 = bf16[3,2,17]{2,1,0} abs(p2)
  ROOT result = bf16[3,14,17]{2,1,0} concatenate(abs0, negate, abs2),
    dimensions={1}
}

ENTRY main {
  p0 = bf16[3,5,17]{2,1,0} parameter(0)
  p1 = bf16[3,7,17]{2,1,0} parameter(1)
  p2 = bf16[3,2,17]{2,1,0} parameter(2)
  ROOT fusion = bf16[3,14,17]{2,1,0} fusion(p0, p1, p2), kind=kCustom,
    calls=concatenate,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRaggedBf16TrailingBroadcast) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_ragged_bf16_broadcast

elementwise {
  p0 = bf16[17,65]{1,0} parameter(0)
  row = bf16[65]{0} parameter(1)
  absolute_row = bf16[65]{0} abs(row)
  rows = bf16[17,65]{1,0} broadcast(absolute_row), dimensions={1}
  ROOT result = bf16[17,65]{1,0} add(p0, rows)
}

ENTRY main {
  p0 = bf16[17,65]{1,0} parameter(0)
  row = bf16[65]{0} parameter(1)
  ROOT fusion = bf16[17,65]{1,0} fusion(p0, row), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeUnalignedBf16ContiguousSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_unaligned_bf16_slice

elementwise {
  p0 = bf16[19,67]{1,0} parameter(0)
  absolute = bf16[19,67]{1,0} abs(p0)
  ROOT result = bf16[15,67]{1,0} slice(absolute),
    slice={[2:17], [0:67]}
}

ENTRY main {
  p0 = bf16[19,67]{1,0} parameter(0)
  ROOT fusion = bf16[15,67]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16GappedRectangularSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_bf16_gapped_slice

elementwise {
  p0 = bf16[19,67]{1,0} parameter(0)
  absolute = bf16[19,67]{1,0} abs(p0)
  ROOT result = bf16[15,65]{1,0} slice(absolute),
    slice={[2:17], [1:66]}
}

ENTRY main {
  p0 = bf16[19,67]{1,0} parameter(0)
  ROOT fusion = bf16[15,65]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRank4Bf16RectangularSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank4_bf16_slice

elementwise {
  p0 = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  absolute = bf16[4,6,8,11]{3,2,1,0} abs(p0)
  ROOT result = bf16[2,4,6,9]{3,2,1,0} slice(absolute),
    slice={[1:3], [1:5], [1:7], [1:10]}
}

ENTRY main {
  p0 = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  ROOT fusion = bf16[2,4,6,9]{3,2,1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRank4S64DynamicSlice) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank4_s64_dynamic_slice

elementwise {
  p0 = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  start0 = s64[] parameter(1)
  start1 = s64[] parameter(2)
  start2 = s64[] parameter(3)
  start3 = s64[] parameter(4)
  absolute = bf16[4,6,8,11]{3,2,1,0} abs(p0)
  ROOT result = bf16[2,4,6,9]{3,2,1,0} dynamic-slice(
    absolute, start0, start1, start2, start3),
    dynamic_slice_sizes={2,4,6,9}
}

ENTRY main {
  p0 = bf16[4,6,8,11]{3,2,1,0} parameter(0)
  start0 = s64[] parameter(1)
  start1 = s64[] parameter(2)
  start2 = s64[] parameter(3)
  start3 = s64[] parameter(4)
  ROOT fusion = bf16[2,4,6,9]{3,2,1,0} fusion(
    p0, start0, start1, start2, start3), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRaggedBf16FlatReverse) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_ragged_bf16_reverse

elementwise {
  p0 = bf16[3,67]{1,0} parameter(0)
  absolute = bf16[3,67]{1,0} abs(p0)
  ROOT result = bf16[3,67]{1,0} reverse(absolute), dimensions={0,1}
}

ENTRY main {
  p0 = bf16[3,67]{1,0} parameter(0)
  ROOT fusion = bf16[3,67]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRaggedBf16PartialDimensionReverse) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_ragged_bf16_partial_reverse

elementwise {
  p0 = bf16[3,67]{1,0} parameter(0)
  absolute = bf16[3,67]{1,0} abs(p0)
  rows = bf16[3,67]{1,0} reverse(absolute), dimensions={0}
  columns = bf16[3,67]{1,0} reverse(absolute), dimensions={1}
  ROOT tuple = (bf16[3,67]{1,0}, bf16[3,67]{1,0}) tuple(rows, columns)
}

ENTRY main {
  p0 = bf16[3,67]{1,0} parameter(0)
  ROOT fusion = (bf16[3,67]{1,0}, bf16[3,67]{1,0})
    fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRank4Bf16PartialDimensionReverse) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank4_bf16_partial_reverse

elementwise {
  p0 = bf16[3,5,7,11]{3,2,1,0} parameter(0)
  absolute = bf16[3,5,7,11]{3,2,1,0} abs(p0)
  outer = bf16[3,5,7,11]{3,2,1,0} reverse(absolute), dimensions={0,2}
  minor = bf16[3,5,7,11]{3,2,1,0} reverse(absolute), dimensions={1,3}
  ROOT tuple = (bf16[3,5,7,11]{3,2,1,0},
                bf16[3,5,7,11]{3,2,1,0}) tuple(outer, minor)
}

ENTRY main {
  p0 = bf16[3,5,7,11]{3,2,1,0} parameter(0)
  ROOT fusion = (bf16[3,5,7,11]{3,2,1,0},
                 bf16[3,5,7,11]{3,2,1,0}) fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeUnalignedBf16FlatEdgePad) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_unaligned_bf16_flat_pad

elementwise {
  p0 = bf16[15,67]{1,0} parameter(0)
  absolute = bf16[15,67]{1,0} abs(p0)
  zero = bf16[] constant(0)
  ROOT result = bf16[19,67]{1,0} pad(absolute, zero),
    padding=2_2x0_0
}

ENTRY main {
  p0 = bf16[15,67]{1,0} parameter(0)
  ROOT fusion = bf16[19,67]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeUnalignedBf16RectangularEdgePad) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_unaligned_bf16_rectangular_pad

elementwise {
  p0 = bf16[15,65]{1,0} parameter(0)
  absolute = bf16[15,65]{1,0} abs(p0)
  zero = bf16[] constant(0)
  ROOT result = bf16[17,69]{1,0} pad(absolute, zero), padding=1_1x2_2
}

ENTRY main {
  p0 = bf16[15,65]{1,0} parameter(0)
  ROOT fusion = bf16[17,69]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRank4InteriorAndNegativeEdgePad) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank4_interior_negative_edge_pad

elementwise {
  p0 = bf16[4,5,6,7]{3,2,1,0} parameter(0)
  absolute = bf16[4,5,6,7]{3,2,1,0} abs(p0)
  padding_value = bf16[] constant(-0.5)
  ROOT result = bf16[5,9,7,15]{3,2,1,0} pad(absolute, padding_value),
    padding=-1_2x1_-1_1x0_1x2_0_1
}

ENTRY main {
  p0 = bf16[4,5,6,7]{3,2,1,0} parameter(0)
  ROOT fusion = bf16[5,9,7,15]{3,2,1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRank1InteriorAndNegativeEdgePad) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank1_interior_negative_edge_pad

elementwise {
  p0 = bf16[7]{0} parameter(0)
  padding_value = bf16[] constant(2)
  ROOT result = bf16[14]{0} pad(p0, padding_value), padding=-1_2_1
}

ENTRY main {
  p0 = bf16[7]{0} parameter(0)
  ROOT fusion = bf16[14]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRank1NegativeEdgePadVectorPath) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank1_negative_edge_pad_vector

elementwise {
  p0 = bf16[7]{0} parameter(0)
  absolute = bf16[7]{0} abs(p0)
  padding_value = bf16[] constant(2)
  ROOT result = bf16[6]{0} pad(absolute, padding_value), padding=-2_1
}

ENTRY main {
  p0 = bf16[7]{0} parameter(0)
  ROOT fusion = bf16[6]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeF32HighRectangularPadding) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_generic_f32_high_padding

padding {
  p0 = f32[17,137]{1,0} parameter(0)
  one = f32[] constant(1)
  ROOT result = f32[32,138]{1,0} pad(p0, one), padding=0_15x0_1
}

ENTRY main {
  p0 = f32[17,137]{1,0} parameter(0)
  ROOT fusion = f32[32,138]{1,0} fusion(p0), kind=kCustom,
    calls=padding,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1"}}}
})";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16PackedInteriorColumnPad) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_bf16_packed_interior_column_pad

padding {
  p0 = bf16[7,8]{1,0} parameter(0)
  absolute = bf16[7,8]{1,0} abs(p0)
  zero = bf16[] constant(0)
  ROOT result = bf16[7,16]{1,0} pad(absolute, zero),
    padding=0_0x0_1_1
}

ENTRY main {
  p0 = bf16[7,8]{1,0} parameter(0)
  ROOT fusion = bf16[7,16]{1,0} fusion(p0), kind=kCustom,
    calls=padding,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeArbitraryDimensionBroadcast) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_arbitrary_dimension_broadcast

elementwise {
  p0 = f32[17,7,65]{2,1,0} parameter(0)
  plane = f32[17,65]{1,0} parameter(1)
  plane_broadcast = f32[17,7,65]{2,1,0} broadcast(plane),
    dimensions={0,2}
  sum = f32[17,7,65]{2,1,0} add(p0, plane_broadcast)
  ROOT result = f32[17,7,65]{2,1,0} tanh(sum)
}

ENTRY main {
  p0 = f32[17,7,65]{2,1,0} parameter(0)
  plane = f32[17,65]{1,0} parameter(1)
  ROOT fusion = f32[17,7,65]{2,1,0} fusion(p0, plane), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(FlyFusionDeviceTest, NativeStripedInvariantBroadcastReuse) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_striped_invariant_broadcast_reuse

elementwise {
  p0 = f32[3,64,64]{2,1,0} parameter(0)
  plane = f32[3,64]{1,0} parameter(1)
  absolute = f32[3,64]{1,0} abs(plane)
  plane_broadcast = f32[3,64,64]{2,1,0} broadcast(absolute),
    dimensions={0,2}
  ROOT result = f32[3,64,64]{2,1,0} add(p0, plane_broadcast)
}

ENTRY main {
  p0 = f32[3,64,64]{2,1,0} parameter(0)
  plane = f32[3,64]{1,0} parameter(1)
  ROOT fusion = f32[3,64,64]{2,1,0} fusion(p0, plane), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["8"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16InteriorPadBothDimensions) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_bf16_interior_pad_both_dimensions

padding {
  p0 = bf16[5,7]{1,0} parameter(0)
  absolute = bf16[5,7]{1,0} abs(p0)
  zero = bf16[] constant(0)
  ROOT result = bf16[12,24]{1,0} pad(absolute, zero),
    padding=1_2_1x2_3_2
}

ENTRY main {
  p0 = bf16[5,7]{1,0} parameter(0)
  ROOT fusion = bf16[12,24]{1,0} fusion(p0), kind=kCustom,
    calls=padding,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["4"]}],
        "num_stages":"1", "num_warps":"4", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeRank4Bf16DilatedReduceWindow) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_rank4_bf16_dilated_reduce_window

maximum {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT result = bf16[] maximum(lhs, rhs)
}

elementwise {
  p0 = bf16[2,5,7,11]{3,2,1,0} parameter(0)
  negative_infinity = bf16[] constant(-inf)
  window = bf16[2,3,3,11]{3,2,1,0} reduce-window(p0, negative_infinity),
    window={size=1x3x3x1 stride=1x2x2x1
      pad=0_0x1_1x2_1x0_0 rhs_dilate=1x1x2x1}, to_apply=maximum
  ROOT result = bf16[2,3,3,11]{3,2,1,0} abs(window)
}

ENTRY main {
  p0 = bf16[2,5,7,11]{3,2,1,0} parameter(0)
  ROOT fusion = bf16[2,3,3,11]{3,2,1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeBf16SlidingReduceWindowReusesInput) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_bf16_sliding_reduce_window

maximum {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT result = bf16[] maximum(lhs, rhs)
}

elementwise {
  p0 = bf16[5,37]{1,0} parameter(0)
  absolute = bf16[5,37]{1,0} abs(p0)
  negative_infinity = bf16[] constant(-inf)
  ROOT result = bf16[5,37]{1,0}
    reduce-window(absolute, negative_infinity),
    window={size=1x15 pad=0_0x7_7}, to_apply=maximum
}

ENTRY main {
  p0 = bf16[5,37]{1,0} parameter(0)
  ROOT fusion = bf16[5,37]{1,0} fusion(p0), kind=kCustom,
    calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["2"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"128"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

TEST_F(FlyFusionDeviceTest, NativeF32BaseDilatedReduceWindow) {
  constexpr absl::string_view kHlo = R"(
HloModule fly_native_f32_base_dilated_reduce_window

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] add(lhs, rhs)
}

elementwise {
  p0 = f32[7]{0} parameter(0)
  zero = f32[] constant(0)
  window = f32[6]{0} reduce-window(p0, zero),
    window={size=3 stride=2 pad=2_1 lhs_dilate=2 rhs_dilate=2},
    to_apply=add
  ROOT result = f32[6]{0} negate(window)
}

ENTRY main {
  p0 = f32[7]{0} parameter(0)
  ROOT fusion = f32[6]{0} fusion(p0), kind=kCustom, calls=elementwise,
    backend_config={"fusion_backend_config":{
      "kind":"__fly",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1"]}],
        "num_stages":"1", "num_warps":"2", "num_ctas":"1",
        "vector_size_bits":"64"}}}
}
)";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(kHlo, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

}  // namespace
}  // namespace xla::gpu
