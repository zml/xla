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

#include "xla/backends/gpu/transforms/metal_workspace_rewriter.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/metal_workspace.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/metal_custom_calls.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::Status CheckRank(const Shape& shape, int64_t rank,
                       absl::string_view description) {
  if (!shape.IsArray() || shape.dimensions().size() != rank) {
    return absl::InvalidArgumentError(absl::StrCat(
        description, " must have rank ", rank, "; got ", shape.ToString()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<int64_t>> DenseNvfp4WorkspaceBytes(
    const HloCustomCallInstruction& call, char arch_size, int arch_gen) {
  if (!IsMetalScaledMatmul(call) || !call.shape().IsArray() ||
      call.operand_count() < 3 ||
      call.operand(1)->shape().element_type() != F4E2M1FN ||
      call.operand(2)->shape().element_type() != F8E4M3FN) {
    return std::nullopt;
  }
  if (call.operand_count() != 3) {
    return absl::InvalidArgumentError(
        absl::StrCat("Metal NVFP4 custom call must have 3 operands; got ",
                     call.operand_count()));
  }

  const Shape& x = call.operand(0)->shape();
  const Shape& w = call.operand(1)->shape();
  RETURN_IF_ERROR(CheckRank(x, 2, "NVFP4 x"));
  RETURN_IF_ERROR(CheckRank(w, 2, "NVFP4 weight"));
  RETURN_IF_ERROR(CheckRank(call.shape(), 2, "NVFP4 output"));
  const int64_t m = x.dimensions(0);
  const int64_t k = x.dimensions(1);
  const int64_t n = w.dimensions(0);
  return GetMetalNvfp4WorkspaceBytes(m, k, n, arch_size, arch_gen);
}

absl::StatusOr<std::optional<int64_t>> MoeWorkspaceBytes(
    const HloCustomCallInstruction& call) {
  if (!IsMetalMoeGemmAny(call) || !call.shape().IsArray()) {
    return std::nullopt;
  }
  if (call.operand_count() < 2) {
    return absl::InvalidArgumentError(
        "Metal MoE custom call must have at least x and weight operands");
  }

  // nvfp4 may carry an optional trailing f32[E] per-expert global scale.
  const int64_t expected_operands = IsMetalMoeGemmBf16(call) ? 3 : 4;
  const bool has_global_scale =
      IsMetalMoeGemmF4(call) && call.operand_count() == expected_operands + 1;
  if (call.operand_count() != expected_operands && !has_global_scale) {
    return absl::InvalidArgumentError(
        absl::StrCat("Metal MoE custom call must have ", expected_operands,
                     IsMetalMoeGemmF4(call) ? " or 5" : "", " operands; got ",
                     call.operand_count()));
  }

  const Shape& x = call.operand(0)->shape();
  const Shape& w = call.operand(1)->shape();
  RETURN_IF_ERROR(CheckRank(x, 2, "Metal MoE x"));
  RETURN_IF_ERROR(CheckRank(w, 3, "Metal MoE weight"));
  RETURN_IF_ERROR(CheckRank(call.shape(), 2, "Metal MoE output"));
  const int64_t r = x.dimensions(0);
  const int64_t k = x.dimensions(1);
  const int64_t e = w.dimensions(0);
  const int64_t n = w.dimensions(1);
  const bool is_nvfp4 = IsMetalMoeGemmF4(call);
  return GetMetalMoeWorkspaceBytes(r, e, k, n, is_nvfp4);
}

absl::StatusOr<std::optional<int64_t>> WorkspaceBytes(
    const HloCustomCallInstruction& call, char arch_size, int arch_gen) {
  TF_ASSIGN_OR_RETURN(std::optional<int64_t> dense,
                      DenseNvfp4WorkspaceBytes(call, arch_size, arch_gen));
  if (dense.has_value()) return dense;
  return MoeWorkspaceBytes(call);
}

absl::Status RewriteWithWorkspace(HloCustomCallInstruction* call,
                                  int64_t workspace_bytes) {
  HloComputation* computation = call->parent();
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {call->shape(), ShapeUtil::MakeShape(S8, {workspace_bytes})});
  HloInstruction* new_call = computation->AddInstruction(
      call->CloneWithNewOperands(tuple_shape, call->operands()));

  // None of the Metal scaled-matmul or MoE kernels are in-place safe. In
  // particular, the per-row paths can write an output row while another
  // threadgroup is still reading the aliased input. CloneWithNewOperands
  // preserves custom-call aliases, so clear them explicitly on the replacement.
  auto* new_custom_call = static_cast<HloCustomCallInstruction*>(new_call);
  new_custom_call->set_output_to_operand_aliasing({});

  HloInstruction* result = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(new_call, 0));
  return computation->ReplaceInstruction(call, result);
}

absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                      char arch_size, int arch_gen) {
  // Snapshot candidates: rewriting inserts another custom call, and processing
  // it in the same traversal would be unnecessary even though tuple-shape
  // idempotence would make it a no-op.
  std::vector<HloCustomCallInstruction*> calls;
  for (HloInstruction* instruction : computation->instructions()) {
    if (auto* call = DynCast<HloCustomCallInstruction>(instruction);
        call != nullptr &&
        (IsMetalScaledMatmul(*call) || IsMetalMoeGemmAny(*call))) {
      calls.push_back(call);
    }
  }

  bool changed = false;
  for (HloCustomCallInstruction* call : calls) {
    TF_ASSIGN_OR_RETURN(std::optional<int64_t> workspace_bytes,
                        WorkspaceBytes(*call, arch_size, arch_gen));
    // A path that needs no scratch keeps its plain array result: an s8[0]
    // tuple element would only add a dead allocation and a scheduling edge.
    if (workspace_bytes.has_value() && *workspace_bytes > 0) {
      RETURN_IF_ERROR(RewriteWithWorkspace(call, *workspace_bytes));
      changed = true;
      continue;
    }

    // Tuple calls have already been workspace-rewritten, and calls that need no
    // workspace keep their result shape. They still must not retain unsafe
    // output donation aliases.
    if (!call->output_operand_aliasing().empty()) {
      call->set_output_to_operand_aliasing({});
      changed = true;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> MetalWorkspaceRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool computation_changed,
                        RunOnComputation(computation, arch_size_, arch_gen_));
    changed |= computation_changed;
  }
  return changed;
}

}  // namespace xla::gpu
