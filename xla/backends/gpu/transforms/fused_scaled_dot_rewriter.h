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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_FUSED_SCALED_DOT_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_FUSED_SCALED_DOT_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Claims the kScaledDots this backend already has a fused lowering for, and
// leaves the rest untouched.
//
// Run this immediately before ScaledDotRewriter, which owns the generic
// dequantize-and-Dot expansion. Everything platform-specific about block-scaled
// matmul lives here; ScaledDotRewriter stays a pure, backend-agnostic rewrite.
// A scaled-dot whose operand layout no backend kernel accepts is simply not
// matched, and the generic expansion behind us handles it as before.
//
// The shared part is MatchWeightOnlyScaledDot: a backend-neutral structural
// match (no batch dimensions, bf16 activations contracting on their last
// dimension, an identity all-ones activation scale, rank-2 quantized weights
// with K minor). Which quantization schemes are lowerable, what dimension
// bounds apply, and what layout the call demands are all the backend arm's
// business -- Metal's are its int32 rank-2 ABI and row-major buffers.
//
// Metal is currently the only arm, and see TryFusedScaledMatmul in the .cc
// before adding another: under the default debug flags this pass does not run
// on CUDA or ROCm at all, and their fused block-scaled kernels are selected by
// the autotuner rather than here.
class FusedScaledDotRewriter : public HloModulePass {
 public:
  explicit FusedScaledDotRewriter(se::GpuComputeCapability gpu_version = {})
      : gpu_version_(gpu_version) {}

  absl::string_view name() const override {
    return "fused-scaled-dot-rewriter";
  }

  absl::StatusOr<bool> RewriteComputation(HloComputation* computation);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_FUSED_SCALED_DOT_REWRITER_H_
