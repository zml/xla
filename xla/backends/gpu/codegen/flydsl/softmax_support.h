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

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {

// Recognizes the canonical rank-2 F16, BF16, or F32 softmax produced by XLA:
// optional conversion to FP32, row maximum, exp, row sum, normalize, and an
// optional conversion back to the interface type.
bool IsFlySoftmaxRoot(const HloInstruction& root);
bool IsFlySoftmaxFusion(const HloFusionAnalysis& analysis);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_SOFTMAX_SUPPORT_H_
