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

#include "xla/backends/gpu/codegen/flydsl/fusion_support.h"

#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/flydsl/attention_support.h"
#include "xla/backends/gpu/codegen/flydsl/paged_attention_support.h"
#include "xla/backends/gpu/codegen/flydsl/scan_support.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_elementwise.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_reduction.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_softmax.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_transpose.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla::gpu::flydsl {
namespace {

bool IsSupportedCustomCallRoot(const HloInstruction& root) {
  return GetFlyScanDescriptor(root).has_value() ||
         GetFlyPagedAttentionDescriptor(root).has_value() ||
         GetFlyPagedAttentionSegmentedProducerDescriptor(root).has_value() ||
         GetFlyPagedAttentionSegmentedReducerDescriptor(root).has_value();
}

bool IsSupportedCustomCallRoot(const HloFusionAnalysis& analysis) {
  return GetFlyScanDescriptor(analysis).has_value() ||
         GetFlyPagedAttentionDescriptor(analysis).has_value() ||
         GetFlyPagedAttentionSegmentedProducerDescriptor(analysis)
             .has_value() ||
         GetFlyPagedAttentionSegmentedReducerDescriptor(analysis).has_value();
}

}  // namespace

FlyFusionRoute ClassifyFlyFusion(const HloFusionAnalysis& analysis) {
  const absl::string_view kind = analysis.fusion_backend_config().kind();
  if (kind == kFlyGemmFusionKind) {
    return FlyFusionRoute::kGemm;
  }
  if (kind == kFlyGemvFusionKind) {
    return FlyFusionRoute::kGemv;
  }
  if (kind == kFlyCollectiveFusionKind) {
    return FlyFusionRoute::kCollective;
  }
  if (kind != kFlyFusionKind) {
    return FlyFusionRoute::kNotFly;
  }
  if (GetFlyScanDescriptor(analysis).has_value()) {
    return FlyFusionRoute::kScan;
  }
  if (GetFlyPagedAttentionDescriptor(analysis).has_value() ||
      GetFlyPagedAttentionSegmentedProducerDescriptor(analysis).has_value() ||
      GetFlyPagedAttentionSegmentedReducerDescriptor(analysis).has_value()) {
    return FlyFusionRoute::kPagedAttention;
  }
  if (GetFlyAttentionDescriptor(analysis).has_value()) {
    return FlyFusionRoute::kAttention;
  }
  if (ContainsUnsupportedCustomCall(analysis)) {
    return FlyFusionRoute::kUnsupportedCustomCall;
  }
  if (IsFlySoftmaxFusion(analysis)) {
    return FlyFusionRoute::kSoftmax;
  }
  if (IsFlyXTileTransposeConfigSupported(analysis)) {
    return FlyFusionRoute::kTranspose;
  }
  if (IsFlyXTileElementwiseFusion(analysis)) {
    return FlyFusionRoute::kElementwise;
  }
  if (IsFlyXTileRowReductionFusion(analysis)) {
    return FlyFusionRoute::kRowReduction;
  }
  return FlyFusionRoute::kGenericXla;
}

absl::string_view FlyFusionRouteName(FlyFusionRoute route) {
  switch (route) {
    case FlyFusionRoute::kNotFly:
      return "not-fly";
    case FlyFusionRoute::kGemm:
      return "native-gemm";
    case FlyFusionRoute::kGemv:
      return "native-gemv";
    case FlyFusionRoute::kCollective:
      return "native-collective";
    case FlyFusionRoute::kScan:
      return "native-scan";
    case FlyFusionRoute::kPagedAttention:
      return "native-paged-attention";
    case FlyFusionRoute::kAttention:
      return "native-attention";
    case FlyFusionRoute::kSoftmax:
      return "native-softmax";
    case FlyFusionRoute::kTranspose:
      return "native-transpose";
    case FlyFusionRoute::kElementwise:
      return "native-elementwise";
    case FlyFusionRoute::kRowReduction:
      return "native-row-reduction";
    case FlyFusionRoute::kGenericXla:
      return "generic-xla-emitter";
    case FlyFusionRoute::kUnsupportedCustomCall:
      return "unsupported-custom-call";
  }
  return "unknown";
}

bool IsNativeFlyFusionRoute(FlyFusionRoute route) {
  return route != FlyFusionRoute::kNotFly &&
         route != FlyFusionRoute::kGenericXla &&
         route != FlyFusionRoute::kUnsupportedCustomCall;
}

bool ContainsUnsupportedCustomCall(const HloInstruction& instruction) {
  if (instruction.opcode() != HloOpcode::kFusion) {
    return false;
  }
  const auto* fusion = Cast<const HloFusionInstruction>(&instruction);
  int custom_call_count = 0;
  for (const HloInstruction* fused : fusion->fused_instructions()) {
    custom_call_count += fused->opcode() == HloOpcode::kCustomCall;
  }
  if (custom_call_count == 0) {
    return false;
  }
  // Every current specialized Fly custom-call fusion consists of exactly one
  // owned custom call. Do not let a recognized root mask another opaque call
  // embedded in one of its operands.
  return custom_call_count != 1 ||
         !IsSupportedCustomCallRoot(*fusion->fused_expression_root());
}

bool ContainsUnsupportedCustomCall(const HloFusionAnalysis& analysis) {
  int custom_call_count = 0;
  analysis.fusion().ForEach([&](HloInstructionAdaptor instruction) {
    custom_call_count += instruction.opcode() == HloOpcode::kCustomCall;
  });
  if (custom_call_count == 0) {
    return false;
  }
  return custom_call_count != 1 || !IsSupportedCustomCallRoot(analysis);
}

}  // namespace xla::gpu::flydsl
