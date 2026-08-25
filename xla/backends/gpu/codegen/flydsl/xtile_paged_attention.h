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

#ifndef XLA_BACKENDS_GPU_CODEGEN_FLYDSL_XTILE_PAGED_ATTENTION_H_
#define XLA_BACKENDS_GPU_CODEGEN_FLYDSL_XTILE_PAGED_ATTENTION_H_

#include <memory>

#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {

// Emits the native four-wave, two-phase MFMA16 GQA4/D128 paged-decode
// specialization from FlyDSL's unified_attention_decode_twophase kernel.
std::unique_ptr<MlirKernelEmitter> CreateFlyXTilePagedAttentionEmitter(
    const HloFusionAnalysis& analysis);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_XTILE_PAGED_ATTENTION_H_
