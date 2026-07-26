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
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/nvfp4_scale_swizzle.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// True if `s` is a bf16 all-ones constant representing an unquantized
// activation. The ZML weight-only form is either scalar or has every dimension
// equal to one.
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
struct WeightOnlyScaledDot {
  HloInstruction* lhs;           // bf16 [..., K]
  HloInstruction* weights;       // quant [N, K]
  HloInstruction* weight_scale;  // scheme-dependent
  int64_t rows;                  // flattened M
  int64_t k;
  int64_t n;
};

// Product of every dimension but the last, or nullopt on a dynamic or
// degenerate dimension. `bound` caps both the product and each dimension.
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
  // leading dimensions followed by N. Re-derive it rather than trusting it.
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

// CUDA: emit cutlass$fp4_mm_dyn {x, weight, weight_scale, wgs, igs} for an
// NVFP4 W4A4 scaled-dot on sm120a. The scaled-dot carries the two per-tensor
// globals (kOperandsWithGlobals): operand 4 = input_global_scale (igs), operand
// 5 = weight_global_scale (wgs). The activation is bf16 and quantized inside the
// kernel (in-register), so nothing is materialized here.
//
// ABI (cutlass_fp4_ffi.cc CutlassFp4MmDynImpl): x bf16[..,K], b f4e2m1[N,K]
// (packed bytes read as-is; M/K taken from x), b_sf e4m3[szB] in cutlass's
// swizzled SF layout, wgs/igs f32 scalars, result bf16[..,N].
absl::StatusOr<HloInstruction*> TryEmitCutlassScaledMatmul(
    HloComputation* comp, HloScaledDotInstruction* dot,
    const se::GpuComputeCapability& gpu_version) {
  // sm120a only: the cutlass fp4 block-scaled MMA is arch-conditional.
  const se::CudaComputeCapability* cc = gpu_version.cuda_compute_capability();
  if (cc == nullptr || cc->major != 12) return nullptr;

  // NVFP4 W4A4 carries the two per-tensor globals; weight-only FP8/MX do not.
  if (!dot->has_global_scales()) return nullptr;

  std::optional<WeightOnlyScaledDot> match =
      MatchWeightOnlyScaledDot(dot, std::numeric_limits<int64_t>::max());
  if (!match.has_value()) return nullptr;

  // NVFP4: fp4 (e2m1) weight, e4m3 block scale, group size 16.
  if (match->weights->shape().element_type() != F4E2M1FN) return nullptr;
  if (match->weight_scale->shape().element_type() != F8E4M3FN) return nullptr;
  if (match->weight_scale->shape().dimensions().size() != 2) return nullptr;
  const int64_t scale_k = match->weight_scale->shape().dimensions(1);
  if (scale_k <= 0 || match->k % scale_k != 0 || match->k / scale_k != 16) {
    return nullptr;
  }

  HloInstruction* x = match->lhs;
  HloInstruction* w = match->weights;
  HloInstruction* w_sf = match->weight_scale;
  HloInstruction* igs = dot->mutable_operand(4);
  HloInstruction* wgs = dot->mutable_operand(5);

  // The kernel consumes the block scale in cutlass's swizzled SF layout and does
  // not convert: the natural scale must already be the un-swizzle of a blocked
  // buffer (the framework swizzles once at load time), so we can hand that buffer
  // straight through and let the un-swizzle chain die. When it isn't -- a scale
  // that really is natural in memory -- we decline, and the scaled-dot falls
  // through to Triton and then the dequant floor, which are always correct.
  HloInstruction* sf = MatchNvfp4ScaleUnswizzle(w_sf);
  if (sf == nullptr) {
    VLOG(1) << "NVFP4 scaled-dot block scale is not a swizzled SF buffer; "
               "leaving it for the generic lowerings: "
            << dot->ToString();
    return nullptr;
  }
  const Shape flat = ShapeUtil::MakeShape(sf->shape().element_type(),
                                          {ShapeUtil::ElementsIn(sf->shape())});
  w_sf = comp->AddInstruction(HloInstruction::CreateReshape(flat, sf));

  // The kernel reads dense row-major buffers; carry that ABI on the call so GPU
  // layout assignment inserts copies for any operand in another physical layout.
  // GetWithDefaultLayout changes only minor_to_major, preserving the sub-byte
  // element size on the packed fp4 weight.
  std::vector<HloInstruction*> operands = {x, w, w_sf, wgs, igs};
  std::vector<Shape> operand_layouts;
  operand_layouts.reserve(operands.size());
  for (const HloInstruction* operand : operands) {
    operand_layouts.push_back(
        LayoutUtil::GetWithDefaultLayout(operand->shape()));
  }
  Shape result_layout = LayoutUtil::GetWithDefaultLayout(dot->shape());

  return comp->AddInstruction(HloInstruction::CreateCustomCall(
      result_layout, operands, "cutlass$fp4_mm_dyn", operand_layouts,
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_TYPED_FFI));
}

// Backend switch: try a fused custom call for this platform. Returns nullptr to
// leave the scaled-dot for the generic dequant+Dot expansion in
// ScaledDotRewriter.
absl::StatusOr<HloInstruction*> TryFusedScaledMatmul(
    HloComputation* comp, HloScaledDotInstruction* dot,
    const se::GpuComputeCapability& gpu_version) {
  if (gpu_version.IsCuda()) {
    return TryEmitCutlassScaledMatmul(comp, dot, gpu_version);
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
    if (extra_filter_ && !extra_filter_(instruction)) {
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
