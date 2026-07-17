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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_WORKSPACE_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_WORKSPACE_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/metal_nvfp4_dispatch.h"

namespace xla::gpu {

// Mutable scratch used by a Metal custom call is represented as the final S8
// tuple result. Buffer assignment then owns its lifetime and gives concurrent
// executions distinct storage. Small immutable launch parameters are passed
// inline and are intentionally not part of this layout.
struct MetalWorkspaceRegion {
  int64_t offset = 0;
  int64_t size = 0;
};

struct MetalNvfp4WorkspaceLayout {
  // [split_k, M, N] bf16 split-K staging. Empty for every non-split path.
  MetalWorkspaceRegion staging;
  int64_t total_bytes = 0;
};

// Layout of the sorted MoE pipeline's mutable buffers. Every region starts at
// a 16-byte boundary, which is sufficient for the widest vector loads. The
// sizes themselves are exact.
struct MetalMoeWorkspaceLayout {
  MetalWorkspaceRegion order;
  MetalWorkspaceRegion expert_ids;
  MetalWorkspaceRegion x_sorted;
  MetalWorkspaceRegion out_sorted;
  int64_t total_bytes = 0;
};

inline constexpr int64_t kMetalWorkspaceAlignment = 16;

// Returns [split_k, M, N] bf16 staging storage. All arithmetic is checked and
// dimensions must be positive.
absl::StatusOr<MetalNvfp4WorkspaceLayout> GetMetalNvfp4SplitKWorkspaceLayout(
    int64_t split_k, int64_t m, int64_t n);

// Returns the split-K staging layout for the path selected at [M, K, N], or an
// empty layout when the selected path needs no mutable workspace.
absl::StatusOr<MetalNvfp4WorkspaceLayout> GetMetalNvfp4WorkspaceLayout(
    int64_t m, int64_t k, int64_t n, char arch_size = kNvfp4DefaultArchSize,
    int arch_gen = kNvfp4DefaultArchGen);

// Returns order[rows] s32, expert_ids[rows] s32, x_sorted[rows,K] bf16, and
// out_sorted[rows,N] bf16. All arithmetic is checked and dimensions must be
// positive.
absl::StatusOr<MetalMoeWorkspaceLayout> GetMetalMoeWorkspaceLayout(
    int64_t rows, int64_t k, int64_t n);

// Shared path selection + exact byte planners for the HLO rewriter, emitter,
// and runtime. Return zero for paths that need no mutable workspace. Dimensions
// are checked against both overflow and the Metal int4 constant ABI.
absl::StatusOr<int64_t> GetMetalNvfp4WorkspaceBytes(
    int64_t m, int64_t k, int64_t n, char arch_size = kNvfp4DefaultArchSize,
    int arch_gen = kNvfp4DefaultArchGen);
absl::StatusOr<int64_t> GetMetalMoeWorkspaceBytes(int64_t r, int64_t e,
                                                  int64_t k, int64_t n,
                                                  bool is_nvfp4);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_WORKSPACE_H_
