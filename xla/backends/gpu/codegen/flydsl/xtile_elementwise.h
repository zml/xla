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

#ifndef XLA_BACKENDS_GPU_CODEGEN_FLYDSL_XTILE_ELEMENTWISE_H_
#define XLA_BACKENDS_GPU_CODEGEN_FLYDSL_XTILE_ELEMENTWISE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu::flydsl {

enum class FlyXTileMemoryPolicy { kCached, kNonTemporal };

// Returns whether a fusion can be emitted as a native Fly vector program.
bool IsFlyXTileElementwiseFusion(const HloFusionAnalysis& analysis);

// Returns whether the fusion is a supported row-scatter specialization owned
// by the native vector emitter.
bool IsFlyXTileOverwriteRowScatterFusion(const HloFusionAnalysis& analysis);

// Returns whether row-scatter stores require an atomic update because of
// collisions or a read-modify-write combiner.
bool IsFlyXTileAtomicOverwriteRowScatterFusion(
    const HloFusionAnalysis& analysis);

// Returns the physical element count traversed by the native vector program.
// This is normally the result size; an in-place row scatter traverses only its
// update payload.
int64_t GetFlyXTileElementwiseElementCount(
    const HloFusionAnalysis& analysis);

// Returns whether the native elementwise program has an indexed output domain
// assembled from differently sized buffers. Full transactions remain
// vectorized; the emitter scalarizes only vectors that cross a boundary.
bool IsFlyXTileIndexedFusion(const HloFusionAnalysis& analysis);

// Returns fusion-parameter numbers used exclusively to compute gather indices.
// Those operands are scalarized by the emitter and must not restrict the
// transaction width selected for gathered values and outputs.
std::vector<int64_t> GetFlyXTileScalarIndexParameterNumbers(
    const HloFusionAnalysis& analysis);

// Selects the cache policy for vectorized global-memory transactions.
FlyXTileMemoryPolicy GetFlyXTileMemoryPolicy(const HloFusionAnalysis& analysis);

// Creates a native Fly emitter for a contiguous elementwise fusion.
std::unique_ptr<MlirKernelEmitter> CreateFlyXTileElementwiseEmitter(
    const HloFusionAnalysis& analysis);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_XTILE_ELEMENTWISE_H_
