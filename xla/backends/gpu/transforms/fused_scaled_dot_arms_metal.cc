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

#include "xla/backends/gpu/transforms/fused_scaled_dot_arms_metal.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/gpu/metal_custom_calls.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "xla/backends/gpu/transforms/fused_scaled_dot_rewriter.h"

namespace xla {
namespace gpu {

namespace {

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

struct WeightOnlyScaledDot {
  HloInstruction* lhs;           // bf16 [..., K]
  HloInstruction* weights;       // quant [N, K]
  HloInstruction* weight_scale;  // scheme-dependent
  int64_t rows;                  // flattened M
  int64_t k;
  int64_t n;
};

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

}  // namespace

FusedScaledDotArm MetalScaledMatmulArm() {
  return [](HloComputation* comp, HloScaledDotInstruction* dot) {
    return TryEmitMetalScaledMatmul(comp, dot);
  };
}

std::vector<FusedScaledDotArm> MetalFusedScaledDotArms() {
  return {MetalScaledMatmulArm()};
}

}  // namespace gpu
}  // namespace xla
