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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/codegen/flydsl/paged_attention_support.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/status_macros.h"

namespace xla::gpu {
namespace {

constexpr int64_t kTokensPerProducerTile = 128;

int64_t CeilOfRatio(int64_t numerator, int64_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

int64_t GetSegmentCount(
    const flydsl::FlyPagedAttentionDescriptor& descriptor) {
  // Follow FlyDSL's D128 decode dispatch.  The wide kernel uses four waves
  // with 32 tokens per wave through KV8192, then four waves with 128 tokens
  // per wave below KV65536. At KV65536 the source-swapped cooperative path
  // takes over. Its tuned segment sizes keep 114 producer CTAs
  // per KV head while amortizing the segmented reducer. The descriptor rounds
  // the resulting extent to the producer's 16-token MFMA granularity.
  if (descriptor.max_context <= 256) {
    return 1;
  }
  int64_t segment_tokens = kTokensPerProducerTile;
  if (descriptor.max_context > 131072) {
    segment_tokens = 2304;
  } else if (descriptor.max_context == 131072) {
    segment_tokens = 1152;
  } else if (descriptor.max_context >= 65536) {
    segment_tokens = 576;
  } else if (descriptor.max_context > 8192) {
    segment_tokens = 512;
  }
  return CeilOfRatio(descriptor.max_context, segment_tokens);
}

absl::Status SetPagedAttentionBackendConfig(HloInstruction* fusion,
                                            int64_t num_warps = 4) {
  GpuBackendConfig gpu_config;
  FusionBackendConfig* fusion_config =
      gpu_config.mutable_fusion_backend_config();
  fusion_config->set_kind(kFlyFusionKind);
  BlockLevelFusionConfig* block =
      fusion_config->mutable_block_level_fusion_config();
  auto add_output_tile = [&](const Shape& shape) {
    Tile* output_tile = block->add_output_tiles();
    for (int64_t dimension = 0;
         dimension + 1 < shape.dimensions_size(); ++dimension) {
      output_tile->add_sizes(1);
    }
    output_tile->add_sizes(
        std::min<int64_t>(128, shape.dimensions().back()));
  };
  if (fusion->shape().IsTuple()) {
    for (const Shape& output_shape : fusion->shape().tuple_shapes()) {
      add_output_tile(output_shape);
    }
  } else {
    add_output_tile(fusion->shape());
  }
  block->set_num_warps(num_warps);
  block->set_num_ctas(1);
  block->set_num_stages(1);
  block->set_waves_per_eu(2);
  return fusion->set_backend_config(std::move(gpu_config));
}

absl::StatusOr<HloFusionInstruction*> MakePagedAttentionFusion(
    HloCustomCallInstruction* call) {
  HloComputation::Builder builder("fly_paged_attention_computation");
  std::vector<HloInstruction*> fused_operands;
  fused_operands.reserve(call->operand_count());
  for (int64_t i = 0; i < 5; ++i) {
    fused_operands.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            i, call->operand(i)->shape(), call->operand(i)->name())));
  }
  fused_operands.push_back(builder.AddInstruction(call->operand(5)->Clone()));
  HloInstruction* root = builder.AddInstruction(
      call->CloneWithNewOperands(call->shape(), fused_operands));

  HloModule* module = call->GetModule();
  HloComputation* parent = call->parent();
  HloComputation* fused_computation =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(root),
                                                /*is_entry=*/false);
  std::vector<HloInstruction*> external_operands(call->operands().begin(),
                                                 call->operands().begin() + 5);
  HloInstruction* fusion = parent->AddInstruction(
      HloInstruction::CreateFusion(call->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   external_operands, fused_computation),
      /*new_name=*/"fly_paged_attention");
  fusion->set_metadata(call->metadata());

  TF_RETURN_IF_ERROR(SetPagedAttentionBackendConfig(fusion));
  TF_RETURN_IF_ERROR(parent->ReplaceInstruction(call, fusion));
  return Cast<HloFusionInstruction>(fusion);
}

absl::StatusOr<HloFusionInstruction*> MakeSegmentedProducerFusion(
    HloCustomCallInstruction* call, int64_t num_segments,
    int64_t num_warps) {
  const Shape& output_shape = call->shape();
  Shape partial_output_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32,
      {output_shape.dimensions(0), output_shape.dimensions(1), num_segments,
       output_shape.dimensions(2)},
      {3, 2, 1, 0});
  Shape partial_state_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32,
      {output_shape.dimensions(0), output_shape.dimensions(1), num_segments},
      {2, 1, 0});
  Shape producer_shape = ShapeUtil::MakeTupleShape(
      {partial_output_shape, partial_state_shape, partial_state_shape});
  HloComputation::Builder builder("fly_paged_attention_segmented_producer");
  std::vector<HloInstruction*> fused_operands;
  fused_operands.reserve(call->operand_count());
  for (int64_t i = 0; i < 5; ++i) {
    fused_operands.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            i, call->operand(i)->shape(), call->operand(i)->name())));
  }
  fused_operands.push_back(builder.AddInstruction(call->operand(5)->Clone()));
  HloInstruction* root = builder.AddInstruction(HloInstruction::CreateCustomCall(
      producer_shape, fused_operands,
      flydsl::kFlyPagedAttentionSegmentedProducerCallTarget));

  HloModule* module = call->GetModule();
  HloComputation* parent = call->parent();
  HloComputation* fused_computation =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(root),
                                                /*is_entry=*/false);
  std::vector<HloInstruction*> external_operands(call->operands().begin(),
                                                 call->operands().begin() + 5);
  HloInstruction* fusion = parent->AddInstruction(
      HloInstruction::CreateFusion(producer_shape,
                                   HloInstruction::FusionKind::kCustom,
                                   external_operands, fused_computation),
      /*new_name=*/"fly_paged_attention_segmented_producer");
  fusion->set_metadata(call->metadata());
  TF_RETURN_IF_ERROR(SetPagedAttentionBackendConfig(fusion, num_warps));
  return Cast<HloFusionInstruction>(fusion);
}

absl::StatusOr<HloFusionInstruction*> MakeSegmentedReducerFusion(
    HloCustomCallInstruction* call, HloInstruction* producer) {
  TF_RET_CHECK(producer->shape().IsTuple() &&
               producer->shape().tuple_shapes_size() == 3);
  std::vector<HloInstruction*> producer_outputs;
  producer_outputs.reserve(3);
  for (int64_t index = 0; index < 3; ++index) {
    producer_outputs.push_back(call->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            producer->shape().tuple_shapes(index), producer, index)));
  }
  HloComputation::Builder builder("fly_paged_attention_segmented_reducer");
  std::vector<HloInstruction*> fused_operands;
  fused_operands.reserve(3);
  for (int64_t index = 0; index < 3; ++index) {
    fused_operands.push_back(builder.AddInstruction(
        HloInstruction::CreateParameter(
            index, producer_outputs[index]->shape(),
            producer_outputs[index]->name())));
  }
  HloInstruction* root = builder.AddInstruction(HloInstruction::CreateCustomCall(
      call->shape(), fused_operands,
      flydsl::kFlyPagedAttentionSegmentedReducerCallTarget));

  HloModule* module = call->GetModule();
  HloComputation* parent = call->parent();
  HloComputation* fused_computation =
      module->AddComputationAndUnifyNamesAndIds(builder.Build(root),
                                                /*is_entry=*/false);
  HloInstruction* fusion = parent->AddInstruction(
      HloInstruction::CreateFusion(call->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   producer_outputs, fused_computation),
      /*new_name=*/"fly_paged_attention_segmented_reducer");
  fusion->set_metadata(call->metadata());
  TF_RETURN_IF_ERROR(SetPagedAttentionBackendConfig(fusion,
                                                    /*num_warps=*/2));
  TF_RETURN_IF_ERROR(parent->ReplaceInstruction(call, fusion));
  return Cast<HloFusionInstruction>(fusion);
}

}  // namespace

absl::StatusOr<bool> PagedAttentionRewriterFly::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloCustomCallInstruction*> matches;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kCustomCall ||
          instruction->custom_call_target() !=
              flydsl::kFlyPagedAttentionDecodeCallTarget ||
          instruction->operand_count() != 6 ||
          !flydsl::GetFlyPagedAttentionDescriptor(*instruction).has_value()) {
        continue;
      }
      matches.push_back(Cast<HloCustomCallInstruction>(instruction));
    }
  }
  for (HloCustomCallInstruction* call : matches) {
    std::optional<flydsl::FlyPagedAttentionDescriptor> descriptor =
        flydsl::GetFlyPagedAttentionDescriptor(*call);
    TF_RET_CHECK(descriptor.has_value());
    const int64_t num_segments = GetSegmentCount(*descriptor);
    if (num_segments == 1) {
      TF_ASSIGN_OR_RETURN(HloFusionInstruction * fusion,
                          MakePagedAttentionFusion(call));
      (void)fusion;
      continue;
    }
    const int64_t segment_tokens =
        CeilOfRatio(descriptor->max_context, num_segments);
    const int64_t num_warps =
        (descriptor->max_context >= 65536 || segment_tokens <= 64) ? 2 : 4;
    TF_ASSIGN_OR_RETURN(
        HloFusionInstruction * producer,
        MakeSegmentedProducerFusion(call, num_segments, num_warps));
    TF_ASSIGN_OR_RETURN(HloFusionInstruction * reducer,
                        MakeSegmentedReducerFusion(call, producer));
    (void)reducer;
  }
  return !matches.empty();
}

}  // namespace xla::gpu
