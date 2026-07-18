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
// downstream -- FusedScaledDotRewriter's backend fast paths, then
// ScaledDotRewriter's generic dequantize-and-Dot floor -- consumes kScaledDot.
//
// Deliberately NOT used here: the parallel `__op$block_scaled_dot` /
// `__op$quantize` / `__op$dequantize` custom-call route handled by
// BlockScalingRewriter. That route is upstream's JAX-facing entry point, is
// registered only for NVPTX (nvptx_compiler.cc), and is MX-centric. We do not
// emit into it and do not route through it; it is left untouched for upstream
// CUDA/ROCm users rather than removed.
//
// Quantization schemes: the maintained FUSED paths are 128x128-block FP8 and
// NVFP4 group-16 (see MetalScaledMatmulScheme in metal_custom_calls.h). OCP
// microscaling (MX / e8m0 group-32) is accepted here on purpose but has no
// fused kernel on any backend we own, so it lowers through the generic
// dequantize-and-Dot floor -- correct, just not fast. Widening acceptance here
// never risks correctness: an operand/scale pair no fast path claims always
// falls through to that floor.
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
