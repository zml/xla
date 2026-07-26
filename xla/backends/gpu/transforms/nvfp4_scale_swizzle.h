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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_NVFP4_SCALE_SWIZZLE_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_NVFP4_SCALE_SWIZZLE_H_

#include <cstdint>

#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

// The NVFP4 block-scale "swizzle" is the hardware block-scaled (SF) layout the
// tensor-core block-scaled MMA reads its scale factors from. It is a pure
// reindex of the natural [N, kg] scale (kg = K/16 groups), no arithmetic:
//
//   sf[n_blk][k_blk][m0][m1][j] = natural[n_blk*128 + m1*32 + m0][k_blk*4 + j]
//
// One tile is 128 rows of N by 4 k-groups = 512 scale bytes, and the N blocks
// are OUTERMOST. Equivalently: view [N, kg] as [N/128, 4, 32, kg/4, 4] and
// transpose by {0,3,2,1,4}. That permutation exchanges positions 1 and 3, so it
// is an involution -- the same permutation spells both directions, which is why
// there is one helper here and not two.
//
// Same layout as XLA's own cuDNN block-scale swizzle
// (block_scaling_rewriter.cc, {0,1,4,3,2,5} with a leading batch axis) and as
// vLLM's swizzle_blockscale (nvfp4_utils.py).
//
// Carrying the swizzled copy as an extra operand on kScaledDot was tried and
// rejected: it forces the 128/4 tiling into a target-independent sharding rule,
// needs a second legal spelling in the verifier, and cannot be split on the
// contracting dimension. It is a LAYOUT, so it lives in the graph as ordinary
// reshape/transpose that every existing pass already reasons about -- including
// the partitioner, which shards the chain without any bespoke rule.
inline constexpr int64_t kSfRowsPerBlock = 128;
inline constexpr int64_t kSfGroupsPerBlock = 4;
inline constexpr int64_t kSfTileElems = kSfRowsPerBlock * kSfGroupsPerBlock;

// Emits the chain turning a blocked [N/128, kg/4, 512] scale into the natural
// [N, kg] one the op is defined on. Returns nullptr if `sf` is not a rank-3
// blocked scale whose dimensions are consistent with [N, kg].
HloInstruction* EmitNvfp4ScaleUnswizzle(HloInstruction* sf, int64_t n,
                                        int64_t kg);

// The inverse of the above as a pattern: given the natural scale an op consumes,
// recovers the blocked buffer it was derived from, or nullptr if `natural` was
// not produced by EmitNvfp4ScaleUnswizzle. A miss is only ever a missed
// optimisation -- the caller falls back to the natural scale -- never a wrong
// answer.
HloInstruction* MatchNvfp4ScaleUnswizzle(HloInstruction* natural);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_NVFP4_SCALE_SWIZZLE_H_
