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

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_SCALED_DOT_GLOBAL_SCALE_EXPANDER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_SCALED_DOT_GLOBAL_SCALE_EXPANDER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// Rewrites a kScaledDot that carries the NVFP4 per-tensor global scales into
// the plain four-operand form plus explicit HLO:
//
//   scaled-dot(lhs, rhs, lhs_scale, rhs_scale, igs, wgs)
//     -> divide(scaled-dot(lhs, rhs, lhs_scale, rhs_scale), broadcast(wgs))
//
// This exists so the globals have a bounded lifetime. Only a backend arm that
// consumes them -- one that quantizes the activation itself, and so needs
// `input_global_scale` to place the block scale inside the e4m3 window -- reads
// them as operands. Everything downstream (Triton, cuDNN, GemmFusion, the
// generic dequant expansion, the evaluator, indexing analysis) then sees the
// long-standing four-operand op it already models correctly, instead of a wider
// one whose trailing operands it would drop silently.
//
// `input_global_scale` is dropped rather than expanded: it does not appear in
// the op's value. With s = ue4m3(amax/6 * igs), q = x * igs / s and
// alpha = 1/(igs * wgs), a quantizing kernel computes
// alpha * sum(q * s * w * sb) = (1/wgs) * sum(x_hat * w * sb) -- igs cancels
// identically, surviving only as the rounding grid of s. A lowering that
// ignores it is correct, differing only in activation-quantization rounding.
//
// Placement matters: this must run AFTER the backend arm has had its chance
// (the arm reads the globals by operand index, which is immune to any pattern
// being rewritten) and after SPMD partitioning (a contracting-dim split wraps
// the dot in an all-reduce, which would sit between the dot and any epilogue
// emitted earlier and break a matcher looking for one). Dividing after the
// all-reduce is exact: the divisor is a replicated scalar, so it commutes with
// the sum.
//
// Never declines, and is idempotent: a four-operand scaled-dot does not match.
class ScaledDotGlobalScaleExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override {
    return "scaled-dot-global-scale-expander";
  }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_SCALED_DOT_GLOBAL_SCALE_EXPANDER_H_
