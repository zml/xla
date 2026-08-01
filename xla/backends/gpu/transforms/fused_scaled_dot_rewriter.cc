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

#include "xla/backends/gpu/transforms/fused_scaled_dot_rewriter.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/gpu/metal_custom_calls.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// True if `s` is a bf16 all-ones constant representing an unquantized
// activation. The ZML weight-only form is either scalar or has every dimension
// equal to one. Do not infer identity from shape alone: a scalar parameter (or
// a non-one scalar constant) cannot be dropped by the fused three-operand ABI.
bool IsIdentityScale(const HloInstruction* s) {
  if (s->shape().element_type() != BF16 ||
      s->opcode() != HloOpcode::kConstant) {
    return false;
  }
  for (int64_t d : s->shape().dimensions()) {
    if (d != 1) return false;
  }
  return s->literal().IsAllFloat(1.0);
}

// A weight-only scaled dot that a backend may be able to fuse: bf16 activations
// whose last dimension is the contracting one, an identity activation scale,
// rank-2 quantized weights [N,K] with K minor, and a result whose shape is the
// lhs leading dimensions followed by N.
//
// Everything here is dot structure -- no backend ABI, no dimension bounds, no
// quantization scheme. A backend arm takes this and applies its own limits.
// `rows` is the product of the lhs leading dimensions: with no batch dimensions
// they are all independent rows, so a backend whose kernel is rank-2 may
// flatten them.
struct WeightOnlyScaledDot {
  HloInstruction* lhs;           // bf16 [..., K]
  HloInstruction* weights;       // quant [N, K]
  HloInstruction* weight_scale;  // scheme-dependent
  int64_t rows;                  // flattened M
  int64_t k;
  int64_t n;
};

// Product of every dimension but the last, or nullopt on a dynamic or
// degenerate dimension. `bound` caps both the product and each dimension, so a
// backend with a narrower index type than int64 can express that limit here.
std::optional<int64_t> FlattenedRowCount(const Shape& x, int64_t bound) {
  const int64_t rank = x.dimensions().size();
  if (rank == 0) return std::nullopt;
  if (x.is_dynamic_dimension(rank - 1) || x.dimensions(rank - 1) <= 0 ||
      x.dimensions(rank - 1) > bound) {
    return std::nullopt;
  }
  int64_t rows = 1;
  for (int64_t dim = 0; dim < rank - 1; ++dim) {
    if (x.is_dynamic_dimension(dim)) return std::nullopt;
    const int64_t size = x.dimensions(dim);
    if (size <= 0 || size > bound / rows) return std::nullopt;
    rows *= size;
  }
  return rows;
}

// Backend-neutral structural match. `dimension_bound` is the largest dimension
// the caller's kernel ABI can index; pass int64 max to impose no limit.
std::optional<WeightOnlyScaledDot> MatchWeightOnlyScaledDot(
    HloScaledDotInstruction* dot, int64_t dimension_bound) {
  const DotDimensionNumbers& dn = dot->dot_dimension_numbers();
  if (dn.lhs_batch_dimensions_size() != 0 ||
      dn.rhs_batch_dimensions_size() != 0 ||
      dn.lhs_contracting_dimensions_size() != 1 ||
      dn.rhs_contracting_dimensions_size() != 1) {
    return std::nullopt;
  }
  if (dot->shape().element_type() != BF16) return std::nullopt;

  HloInstruction* x = dot->mutable_operand(0);
  HloInstruction* w = dot->mutable_operand(1);
  HloInstruction* xs = dot->mutable_operand(2);
  HloInstruction* ws = dot->mutable_operand(3);

  const int64_t x_rank = x->shape().dimensions().size();
  if (x_rank == 0 || x->shape().element_type() != BF16 ||
      dn.lhs_contracting_dimensions(0) != x_rank - 1 || !IsIdentityScale(xs)) {
    return std::nullopt;
  }
  std::optional<int64_t> rows = FlattenedRowCount(x->shape(), dimension_bound);
  if (!rows.has_value()) return std::nullopt;

  // Rank-2 weights, K minor, statically sized within the caller's bound.
  if (w->shape().dimensions().size() != 2 ||
      dn.rhs_contracting_dimensions(0) != 1) {
    return std::nullopt;
  }
  for (int64_t dim = 0; dim < w->shape().dimensions().size(); ++dim) {
    if (w->shape().is_dynamic_dimension(dim) ||
        w->shape().dimensions(dim) <= 0 ||
        w->shape().dimensions(dim) > dimension_bound) {
      return std::nullopt;
    }
  }
  for (int64_t dim = 0; dim < ws->shape().dimensions().size(); ++dim) {
    if (ws->shape().is_dynamic_dimension(dim) ||
        ws->shape().dimensions(dim) <= 0) {
      return std::nullopt;
    }
  }
  const int64_t k = x->shape().dimensions(x_rank - 1);
  const int64_t n = w->shape().dimensions(0);
  if (w->shape().dimensions(1) != k) return std::nullopt;

  // With no batch dimensions and a rank-2 rhs, the result is exactly the lhs
  // leading dimensions followed by N. Re-derive it rather than trusting it:
  // this runs before the fusion replaces the dot, and callers construct a fixed
  // rank-2 call from these numbers.
  const Shape& result = dot->shape();
  if (result.dimensions().size() != x_rank) return std::nullopt;
  for (int64_t dim = 0; dim < x_rank - 1; ++dim) {
    if (result.is_dynamic_dimension(dim) ||
        result.dimensions(dim) != x->shape().dimensions(dim)) {
      return std::nullopt;
    }
  }
  if (result.is_dynamic_dimension(x_rank - 1) ||
      result.dimensions(x_rank - 1) != n) {
    return std::nullopt;
  }
  return WeightOnlyScaledDot{x, w, ws, *rows, k, n};
}

// Metal: emit zml$scaled_matmul {x, w, w_scale}; ThunkEmitter dispatches on the
// same ClassifyMetalScaledMatmul result, so the two cannot disagree about which
// schemes are lowerable.
//
// Metal-specific constraints applied here and nowhere above: the custom-call
// ABI is static int32 rank-2, so dimensions are int32-bounded and the lhs
// leading dimensions are flattened (as MLX does for a row-contiguous lhs); and
// the kernels read dense row-major buffers, so the call carries explicit
// row-major operand and result layouts.
absl::StatusOr<HloInstruction*> TryEmitMetalScaledMatmul(
    HloComputation* comp, HloScaledDotInstruction* dot) {
  constexpr int64_t kMaxMetalDimension = std::numeric_limits<int32_t>::max();
  std::optional<WeightOnlyScaledDot> match =
      MatchWeightOnlyScaledDot(dot, kMaxMetalDimension);
  if (!match.has_value()) return nullptr;
  if (!ClassifyMetalScaledMatmul(match->weights->shape(),
                                 match->weight_scale->shape())
           .has_value()) {
    return nullptr;
  }

  HloInstruction* x = match->lhs;
  HloInstruction* w = match->weights;
  HloInstruction* ws = match->weight_scale;
  const bool needs_flatten = x->shape().dimensions().size() != 2;
  const int64_t m = match->rows;
  const int64_t k = match->k;
  const int64_t n = match->n;

  Shape call_result_shape = dot->shape();
  if (needs_flatten) {
    Shape flat_x_shape = ShapeUtil::MakeShape(BF16, {m, k});
    LayoutUtil::SetToDefaultLayout(&flat_x_shape);
    x = comp->AddInstruction(
        HloInstruction::CreateReshape(flat_x_shape, x));
    call_result_shape = ShapeUtil::MakeShape(BF16, {m, n});
  }

  // The Metal kernels interpret all three inputs and the output as dense
  // row-major matrices. Carry that ABI requirement on the custom call so GPU
  // layout assignment inserts copies when an input arrives in another physical
  // layout. SetToDefaultLayout changes only minor_to_major, preserving the
  // sub-byte element size on packed fp4 weights.
  std::vector<HloInstruction*> operands = {x, w, ws};
  std::vector<Shape> operand_layouts;
  operand_layouts.reserve(operands.size());
  for (const HloInstruction* operand : operands) {
    operand_layouts.push_back(
        LayoutUtil::GetWithDefaultLayout(operand->shape()));
  }
  Shape result_layout =
      LayoutUtil::GetWithDefaultLayout(call_result_shape);
  HloInstruction* call = comp->AddInstruction(HloInstruction::CreateCustomCall(
      result_layout, operands, std::string(kMetalScaledMatmulCallTarget),
      operand_layouts));
  if (!needs_flatten) return call;

  return comp->AddInstruction(
      HloInstruction::CreateReshape(dot->shape(), call));
}

// Backend switch: try a fused custom call for this platform. Returns nullptr to
// leave the scaled-dot for the generic dequant+Dot expansion in
// ScaledDotRewriter.
//
// Metal is the only arm, and on the current pipeline it is the only arm that
// can ever run: this pass is registered under
// !xla_gpu_experimental_scaled_dot_with_triton, that flag defaults to true
// (debug_options_flags.cc), and MetalGpuCompiler is the only compiler that
// forces it false. On CUDA/ROCm the kScaledDot instead survives into
// GemmFusion, which wraps it in a Triton fusion, and the fused-kernel choice
// (hipBLASLt / cuDNN / Triton) is made by the autotuner's CodegenBackends
// rather than by an HLO pass. So a CUDA or ROCm fused kernel belongs in
// xla/backends/gpu/autotuner/, NOT in a second arm here -- an arm added here
// would be dead code under the default flag.
absl::StatusOr<HloInstruction*> TryFusedScaledMatmul(
    HloComputation* comp, HloScaledDotInstruction* dot,
    const se::GpuComputeCapability& gpu_version) {
  if (gpu_version.IsMetal()) {
    return TryEmitMetalScaledMatmul(comp, dot);
  }
  return nullptr;
}

}  // namespace

absl::StatusOr<bool> FusedScaledDotRewriter::RewriteComputation(
    HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kScaledDot) {
      continue;
    }
    HloScaledDotInstruction* dot = Cast<HloScaledDotInstruction>(instruction);
    ASSIGN_OR_RETURN(HloInstruction * fused,
                     TryFusedScaledMatmul(computation, dot, gpu_version_));
    if (fused == nullptr) {
      continue;  // No fused lowering for this layout; leave it for the
                 // generic ScaledDotRewriter expansion.
    }
    RETURN_IF_ERROR(dot->ReplaceAllUsesWith(fused));
    RETURN_IF_ERROR(computation->RemoveInstruction(dot));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> FusedScaledDotRewriter::RunImpl(
    HloModule* module, const absl::flat_hash_set<absl::string_view>&) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    ASSIGN_OR_RETURN(bool result, RewriteComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
