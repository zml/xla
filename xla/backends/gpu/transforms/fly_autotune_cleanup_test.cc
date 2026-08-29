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

#include "xla/backends/gpu/transforms/fly_autotune_cleanup.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

class FlyAutotuneCleanupTest : public HloHardwareIndependentTestBase {};

TEST_F(FlyAutotuneCleanupTest, RetagsImportedTritonCollectiveInReplacementMode) {
  constexpr absl::string_view kHlo = R"(
HloModule imported_triton_collective

sum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] add(lhs, rhs)
}

collective {
  input = f32[1024]{0} parameter(0)
  ROOT result = f32[1024]{0} all-reduce(input), replica_groups={{0}},
    to_apply=sum
}

ENTRY main {
  input = f32[1024]{0} parameter(0)
  ROOT fusion = f32[1024]{0} fusion(input), kind=kCustom,
    calls=collective,
    backend_config={"fusion_backend_config":{
      "kind":"__triton_collective", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1024"]}], "num_warps":"16",
        "num_ctas":"1", "num_stages":"1"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_flydsl_replace_triton(true);

  FlyAutotuneCleanup cleanup;
  ASSERT_OK_AND_ASSIGN(bool changed, cleanup.Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(
      GpuBackendConfig config,
      module->entry_computation()->root_instruction()->backend_config<
          GpuBackendConfig>());
  EXPECT_EQ(config.fusion_backend_config().kind(), kFlyCollectiveFusionKind);
}

TEST_F(FlyAutotuneCleanupTest, FoldsParameterFreeScalarFusionToConstant) {
  constexpr absl::string_view kHlo = R"(
HloModule scalar_materialization

materialize {
  value = bf16[] constant(0.125)
  ROOT converted = f32[] convert(value)
}

consume {
  scalar = f32[] parameter(0)
  values = f32[4]{0} parameter(1)
  broadcast = f32[4]{0} broadcast(scalar), dimensions={}
  ROOT result = f32[4]{0} add(values, broadcast)
}

ENTRY main {
  values = f32[4]{0} parameter(0)
  scalar = f32[] fusion(), kind=kLoop, calls=materialize
  ROOT consumer = f32[4]{0} fusion(scalar, values), kind=kCustom,
    calls=consume,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["4"]}],
      "num_warps":"1","num_ctas":1,"num_stages":1,
      "vector_size_bits":"128"}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  FlyAutotuneCleanup cleanup;
  ASSERT_OK_AND_ASSIGN(bool changed, cleanup.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* consumer =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(consumer->opcode(), HloOpcode::kFusion);
  ASSERT_EQ(consumer->operand(0)->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(consumer->operand(0)->literal().GetFirstElement<float>(), 0.125f);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
}

TEST_F(FlyAutotuneCleanupTest, MergesExactScaleIntoFlyProducer) {
  constexpr absl::string_view kHlo = R"(
HloModule scale_prologue

produce {
  lhs = bf16[4,1,8]{2,1,0} parameter(0)
  rhs = bf16[4,1,8]{2,1,0} parameter(1)
  ROOT result = bf16[4,1,8]{2,1,0} add(lhs, rhs)
}

scale {
  data = bf16[4,8]{1,0} parameter(0)
  weight = bf16[8]{0} parameter(1)
  data_f32 = f32[4,8]{1,0} convert(data)
  weights = bf16[4,8]{1,0} broadcast(weight), dimensions={1}
  weights_f32 = f32[4,8]{1,0} convert(weights)
  product = f32[4,8]{1,0} multiply(data_f32, weights_f32)
  ROOT result = bf16[4,8]{1,0} convert(product)
}

gemv {
  lhs = bf16[4,8]{1,0} parameter(0)
  rhs = bf16[8,16]{1,0} parameter(1)
  ROOT result = bf16[4,16]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[4,1,8]{2,1,0} parameter(0)
  residual = bf16[4,1,8]{2,1,0} parameter(1)
  weight = bf16[8]{0} parameter(2)
  rhs = bf16[8,16]{1,0} parameter(3)
  producer = bf16[4,1,8]{2,1,0} fusion(lhs, residual), kind=kCustom,
    calls=produce,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["4"]}],
      "num_warps":"1","num_ctas":1,"num_stages":1,
      "vector_size_bits":"64"}}}
  view = bf16[4,8]{1,0} bitcast(producer)
  scaled = bf16[4,8]{1,0} fusion(view, weight), kind=kLoop, calls=scale
  ROOT gemm = bf16[4,16]{1,0} fusion(scaled, rhs), kind=kCustom,
    calls=gemv,
    backend_config={"fusion_backend_config":{"kind":"__fly_gemv",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["4","16"]}],
      "num_warps":"1","num_ctas":1,"num_stages":1}}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  FlyAutotuneCleanup cleanup;
  ASSERT_OK_AND_ASSIGN(bool changed, cleanup.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* gemm = module->entry_computation()->root_instruction();
  ASSERT_EQ(gemm->opcode(), HloOpcode::kFusion);
  HloInstruction* scaled = gemm->mutable_operand(0);
  ASSERT_EQ(scaled->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(scaled->operand_count(), 3);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 6);

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                       scaled->backend_config<GpuBackendConfig>());
  EXPECT_EQ(config.fusion_backend_config().kind(), kFlyFusionKind);
  EXPECT_EQ(config.fusion_backend_config()
                .block_level_fusion_config()
                .vector_size_bits(),
            64);

  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kBitcast);
    EXPECT_NE(instruction->name(), "producer");
  }
}

TEST_F(FlyAutotuneCleanupTest, DeduplicatesIdenticalFlyFissionBoundaries) {
  constexpr absl::string_view kHlo = R"(
HloModule duplicate_fission_boundaries

transpose_first {
  input = bf16[2,3,4,64,64]{4,3,2,1,0} parameter(0)
  ROOT result = bf16[2,64,3,4,64]{4,3,2,1,0} transpose(input),
    dimensions={0,4,1,2,3}
}

transpose_second {
  input = bf16[2,3,4,64,64]{4,3,2,1,0} parameter(0)
  ROOT result = bf16[2,64,3,4,64]{4,3,2,1,0} transpose(input),
    dimensions={0,4,1,2,3}
}

ENTRY main {
  input = bf16[2,3,4,64,64]{4,3,2,1,0} parameter(0)
  first = bf16[2,64,3,4,64]{4,3,2,1,0} fusion(input), kind=kCustom,
    calls=transpose_first,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","1","1","1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1}}}
  second = bf16[2,64,3,4,64]{4,3,2,1,0} fusion(input), kind=kCustom,
    calls=transpose_second,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","1","1","1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1}}}
  ROOT result = (bf16[2,64,3,4,64]{4,3,2,1,0},
                 bf16[2,64,3,4,64]{4,3,2,1,0}) tuple(first, second)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  FlyAutotuneCleanup cleanup;
  ASSERT_OK_AND_ASSIGN(bool changed, cleanup.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0), root->operand(1));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
}

TEST_F(FlyAutotuneCleanupTest, DeduplicatesWrappedAndUnwrappedTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule wrapped_and_unwrapped_transpose

wrapped_computation {
  input = bf16[2,3,4,64,64]{4,3,2,1,0} parameter(0)
  ROOT result = bf16[2,64,3,4,64]{4,3,2,1,0} transpose(input),
    dimensions={0,4,1,2,3}
}

ENTRY main {
  input = bf16[2,3,4,64,64]{4,3,2,1,0} parameter(0)
  unwrapped = bf16[2,64,3,4,64]{4,3,2,1,0} transpose(input),
    dimensions={0,4,1,2,3}
  wrapped = bf16[2,64,3,4,64]{4,3,2,1,0} fusion(input), kind=kInput,
    calls=wrapped_computation
  ROOT result = (bf16[2,64,3,4,64]{4,3,2,1,0},
                 bf16[2,64,3,4,64]{4,3,2,1,0}) tuple(unwrapped, wrapped)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  FlyAutotuneCleanup cleanup;
  ASSERT_OK_AND_ASSIGN(bool changed, cleanup.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0), root->operand(1));
  EXPECT_EQ(root->operand(0)->fusion_kind(),
            HloInstruction::FusionKind::kInput);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
}

TEST_F(FlyAutotuneCleanupTest,
       DeduplicatesConfiguredFlyAndUnwrappedTranspose) {
  constexpr absl::string_view kHlo = R"(
HloModule configured_fly_and_unwrapped_transpose

wrapped_computation {
  input = bf16[2,3,4,64,64]{4,3,2,1,0} parameter(0)
  ROOT result = bf16[2,64,3,4,64]{4,3,2,1,0} transpose(input),
    dimensions={0,4,1,2,3}
}

ENTRY main {
  input = bf16[2,3,4,64,64]{4,3,2,1,0} parameter(0)
  unwrapped = bf16[2,64,3,4,64]{4,3,2,1,0} transpose(input),
    dimensions={0,4,1,2,3}
  wrapped = bf16[2,64,3,4,64]{4,3,2,1,0} fusion(input), kind=kCustom,
    calls=wrapped_computation,
    backend_config={"fusion_backend_config":{"kind":"__fly",
      "block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","1","1","1"]}],
      "num_warps":"4","num_ctas":1,"num_stages":1}}}
  ROOT result = (bf16[2,64,3,4,64]{4,3,2,1,0},
                 bf16[2,64,3,4,64]{4,3,2,1,0}) tuple(unwrapped, wrapped)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  FlyAutotuneCleanup cleanup;
  ASSERT_OK_AND_ASSIGN(bool changed, cleanup.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0), root->operand(1));
  EXPECT_EQ(root->operand(0)->fusion_kind(),
            HloInstruction::FusionKind::kCustom);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
}

}  // namespace
}  // namespace xla::gpu
