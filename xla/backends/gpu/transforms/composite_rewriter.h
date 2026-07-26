/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_COMPOSITE_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_COMPOSITE_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites a `stablehlo.composite` call into the equivalent native HLO.
//
// Today that means the `xla.scaled_dot` composite -> kScaledDot. This composite
// is the SINGLE supported carrier for block-scaled matmul in this fork: ZML
// (zml/ops.zig scaledDot) emits it, this pass canonicalizes it, and everything
// downstream consumes kScaledDot. The default floor is ScaledDotRewriter's
// generic dequantize-and-Dot expansion (correct on every GPU backend). Optional
// fused kernels, when a backend has them, claim a matched subset ahead of that
// floor without changing this entry point.
//
// Deliberately NOT used here: the parallel `__op$block_scaled_dot` /
// `__op$quantize` / `__op$dequantize` custom-call route handled by
// BlockScalingRewriter. That route is upstream's JAX-facing entry point, is
// registered only for NVPTX (nvptx_compiler.cc), and is MX-centric. We do not
// emit into it and do not route through it; it is left untouched for upstream
// CUDA/ROCm users rather than removed. What this pass produces instead is a
// plain kScaledDot, which gpu_compiler.cc lowers through its ladder: a fused
// backend arm, then Triton, then the generic dequant floor.
//
// Quantization schemes accepted (any operand/scale pair whose dimensions
// divide cleanly). The primary ones ZML emits today:
//   - FP8 128x128-block: f8e4m3fn weights + bf16/f32 scales [N/128, K/128]
//   - FP8 per-channel:   f8e4m3fn weights + bf16/f32 scales [N, 1]
//   - NVFP4 group-16:    f4e2m1fn weights + f8e4m3fn scales [N, K/16]
// OCP microscaling (MX / e8m0 group-32) is also accepted and lowers through the
// same dequantize floor. Widening acceptance here never risks correctness: an
// operand/scale pair no fast path claims always falls through to that floor.
class CompositeRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "composite-rewriter"; }

  absl::StatusOr<bool> RewriteComputation(HloComputation* computation);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_COMPOSITE_REWRITER_H_
