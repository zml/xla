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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

absl::Status ValidateFlashAttentionCall(const HloInstruction& call) {
  if (call.operand_count() != 4 && call.operand_count() != 5) {
    return InvalidArgument("%s expects 4 or 5 operands, got %d",
                           kVulkanFlashAttentionCallTarget,
                           call.operand_count());
  }

  const Shape& q = call.operand(0)->shape();
  const Shape& k = call.operand(1)->shape();
  const Shape& v = call.operand(2)->shape();
  const Shape& output = call.shape();
  if (!q.IsArray() || !k.IsArray() || !v.IsArray() || !output.IsArray() ||
      q.dimensions().size() != 3 || k.dimensions().size() != 3 ||
      v.dimensions().size() != 3 || output.dimensions().size() != 3) {
    return InvalidArgument("%s expects rank-3 Q, K, V, and output tensors",
                           kVulkanFlashAttentionCallTarget);
  }
  if (q.element_type() != BF16 || k.element_type() != BF16 ||
      v.element_type() != BF16 || output.element_type() != BF16) {
    return InvalidArgument("%s supports only BF16 Q, K, V, and output",
                           kVulkanFlashAttentionCallTarget);
  }
  if (!ShapeUtil::Compatible(q, output) || !ShapeUtil::Compatible(k, v)) {
    return InvalidArgument("%s requires output to match Q and V to match K",
                           kVulkanFlashAttentionCallTarget);
  }

  const int64_t query_heads = q.dimensions(0);
  const int64_t kv_heads = k.dimensions(0);
  const int64_t head_dim = q.dimensions(2);
  if (query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0 ||
      k.dimensions(2) != head_dim || head_dim < 16 || head_dim > 256 ||
      head_dim % 16 != 0) {
    return InvalidArgument(
        "%s requires compatible grouped-query heads and head dimension "
        "divisible by 16 in [16, 256]",
        kVulkanFlashAttentionCallTarget);
  }
  if (call.operand_count() == 4 && q.dimensions(1) != 1) {
    return InvalidArgument("%s decode form requires query length 1",
                           kVulkanFlashAttentionCallTarget);
  }

  const Shape& token_index = call.operand(3)->shape();
  if (!ShapeUtil::IsScalar(token_index) || token_index.element_type() != S32) {
    return InvalidArgument("%s token index must be an s32 scalar",
                           kVulkanFlashAttentionCallTarget);
  }
  if (call.operand_count() == 5) {
    const Shape& num_tokens = call.operand(4)->shape();
    if (!ShapeUtil::IsScalar(num_tokens) ||
        num_tokens.element_type() != U32) {
      return InvalidArgument("%s token count must be a u32 scalar",
                             kVulkanFlashAttentionCallTarget);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<HloComputation*> CreateFusionBody(HloInstruction& call) {
  HloComputation::Builder builder("vulkan_flash_attention");
  std::vector<HloInstruction*> parameters;
  parameters.reserve(call.operand_count());
  for (int64_t i = 0; i < call.operand_count(); ++i) {
    parameters.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i, call.operand(i)->shape(), "parameter_" + std::to_string(i))));
  }
  builder.AddInstruction(call.CloneWithNewOperands(call.shape(), parameters));
  return call.GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                             false);
}

absl::Status SetBackendConfig(HloInstruction& fusion) {
  GpuBackendConfig gpu_config;
  FusionBackendConfig& config = *gpu_config.mutable_fusion_backend_config();
  config.set_kind(std::string(kCustomFusionKind));
  config.mutable_custom_fusion_config()->set_name(
      std::string(kVulkanFlashAttentionFusionConfigName));
  return fusion.set_backend_config(std::move(gpu_config));
}

class FlashAttentionVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleCustomCall(HloInstruction* call) override {
    if (call->custom_call_target() != kVulkanFlashAttentionCallTarget ||
        call->parent()->IsFusionComputation()) {
      return absl::OkStatus();
    }
    RETURN_IF_ERROR(ValidateFlashAttentionCall(*call));
    ASSIGN_OR_RETURN(HloComputation * fusion_body, CreateFusionBody(*call));
    HloInstruction* fusion = call->parent()->AddInstruction(
        HloInstruction::CreateFusion(call->shape(),
                                     HloInstruction::FusionKind::kCustom,
                                     call->operands(), fusion_body));
    call->GetModule()->SetAndUniquifyInstrName(fusion,
                                               "vulkan_flash_attention");
    RETURN_IF_ERROR(SetBackendConfig(*fusion));
    return ReplaceInstruction(call, fusion);
  }
};

}  // namespace

absl::StatusOr<bool> VulkanFlashAttentionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return FlashAttentionVisitor().RunOnModule(module, execution_threads);
}

}  // namespace xla::gpu
