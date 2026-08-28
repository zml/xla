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

#ifndef XLA_BACKENDS_GPU_CODEGEN_FLYDSL_XTILE_TRANSPOSE_H_
#define XLA_BACKENDS_GPU_CODEGEN_FLYDSL_XTILE_TRANSPOSE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {

// Returns the logical input matrix shape (rows, columns) for a native BF16
// transpose. Besides a direct rank-2 transpose, this recognizes the batched
// bitcast/transpose and bitcast/slice/transpose views used by transformer
// attention. Partial edge tiles are supported for matrix extents of at least
// 32 elements.
std::optional<std::pair<int64_t, int64_t>> GetFlyXTileTransposeMatrixShape(
    const HloFusionAnalysis& analysis);

bool IsFlyXTileTransposeFusion(const HloFusionAnalysis& analysis);

// Returns whether the selected block-level config is compatible with the
// native xTile emitter. Structural matching remains separate so the Fly
// autotuner can replace an existing rank-preserving Triton/legacy config.
bool IsFlyXTileTransposeConfigSupported(const HloFusionAnalysis& analysis);

// Emits one rectangular matrix tile per workgroup. Every thread transfers two
// or four 128-bit vectors in each direction. Adjacent BF16 input rows are
// packed into LDS dwords; partial edge workgroups overlap the preceding tile
// to retain the same branch-free vectorized memory path.
std::unique_ptr<MlirKernelEmitter> CreateFlyXTileTransposeEmitter(
    const HloFusionAnalysis& analysis);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_XTILE_TRANSPOSE_H_
