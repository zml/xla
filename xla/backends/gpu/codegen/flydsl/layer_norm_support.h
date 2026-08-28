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

#ifndef XLA_BACKENDS_GPU_CODEGEN_FLYDSL_LAYER_NORM_SUPPORT_H_
#define XLA_BACKENDS_GPU_CODEGEN_FLYDSL_LAYER_NORM_SUPPORT_H_

#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::flydsl {

struct FlyLayerNormDescriptor {
  const HloInstruction* output;
  const HloInstruction* input;
  const HloInstruction* gamma;
  const HloInstruction* beta;
  const HloInstruction* mean;
  const HloInstruction* reciprocal_stddev;
  const HloInstruction* reciprocal_stddev_cube;
  PrimitiveType element_type;
  int64_t rows;
  int64_t columns;
  double epsilon;
  int64_t output_count;
  bool uses_moments_variance;
};

// Recognizes inference LayerNorm over the contiguous minor dimension:
// mean(x), mean((x - mean)^2), rsqrt(variance + epsilon), normalize, and
// optional per-column gamma/beta affine transforms. A training form may also
// return the row mean, reciprocal standard deviation, and optionally the
// reciprocal-standard-deviation cube used by the backward pass. Both the
// centered-square and E[x^2] - E[x]^2 variance forms are recognized.
std::optional<FlyLayerNormDescriptor> GetFlyLayerNormDescriptor(
    const HloInstruction& root);
std::optional<FlyLayerNormDescriptor> GetFlyLayerNormDescriptor(
    const HloFusionAnalysis& analysis);

bool IsFlyLayerNormRoot(const HloInstruction& root);
bool IsFlyLayerNormFusion(const HloFusionAnalysis& analysis);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_LAYER_NORM_SUPPORT_H_
