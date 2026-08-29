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

#ifndef XLA_BACKENDS_GPU_CODEGEN_FLYDSL_SOFTMAX_SUPPORT_H_
#define XLA_BACKENDS_GPU_CODEGEN_FLYDSL_SOFTMAX_SUPPORT_H_

#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {

// The two tensor inputs and score scale of a canonical softmax derivative.
// The native kernel recomputes the stable softmax from `scores` and applies
//   scale * p * (upstream_gradient - sum(upstream_gradient * p)).
// The inputs are allowed to be view-compatible F16/BF16 interface tensors;
// widening conversions and element-preserving bitcasts remain inside the
// fusion. Their element count and final minor dimension match the output.
struct FlySoftmaxBackwardDescriptor {
  const HloInstruction* scores;
  const HloInstruction* upstream_gradient;
  const HloInstruction* output;
  double scale;
};

std::optional<FlySoftmaxBackwardDescriptor> GetFlySoftmaxBackwardDescriptor(
    const HloInstruction& root);
std::optional<FlySoftmaxBackwardDescriptor> GetFlySoftmaxBackwardDescriptor(
    const HloFusionAnalysis& analysis);

// Returns the tensor entering a canonical stable softmax, or nullptr when the
// root is not a supported softmax. Unlike IsFlySoftmaxRoot, this accepts a
// producer expression instead of requiring the input to be a fusion parameter.
// The producer may remain F32 when an F16/BF16 result narrows scaled attention
// scores; the native emitter performs the interface conversion on store.
const HloInstruction* GetFlySoftmaxInput(const HloInstruction& root);

// Returns a scalar producer scale folded into the native softmax input. The
// input returned above is before this scale, so the emitter applies it before
// the maximum and exponential reductions.
double GetFlySoftmaxInputScale(const HloInstruction& root);

// Returns a runtime row offset used by `exp(input - row_offset)` when the
// normalized result is otherwise a canonical softmax. The offset is optional:
// the usual stable-softmax form computes it with a row maximum internally.
const HloInstruction* GetFlySoftmaxExternalRowOffset(
    const HloInstruction& root);

// Returns true when the external row value is subtracted before a second,
// internal row-maximum reduction. This is the form XLA emits for an explicitly
// stabilized tensor passed to softmax; unlike the direct external-maximum form,
// the native kernel must retain both subtractions.
bool FlySoftmaxRecomputesMaximumAfterExternalRowOffset(
    const HloInstruction& root);

// Like GetFlySoftmaxInput, but retains a mixed-precision F32 producer when the
// softmax result is narrowed to F16/BF16. Compound attention uses this to see
// canonical select(-inf) masks instead of treating them as external inputs.
const HloInstruction* GetFlyCompoundSoftmaxInput(const HloInstruction& root);

// Generalized compound-softmax matcher for a specified logical reduction
// dimension. Attention layouts can keep the key dimension non-minor while
// still representing exactly the same stable row softmax.
const HloInstruction* GetFlyCompoundSoftmaxInputAlongDimension(
    const HloInstruction& root, int64_t reduction_dimension);

// Recognizes the canonical row-wise F16, BF16, or F32 softmax produced by XLA:
// optional conversion to FP32, row maximum, exp, row sum, normalize, and an
// optional conversion back to the interface type.
bool IsFlySoftmaxRoot(const HloInstruction& root);
bool IsFlySoftmaxFusion(const HloFusionAnalysis& analysis);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_SOFTMAX_SUPPORT_H_
