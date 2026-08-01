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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_NVFP4_DISPATCH_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_NVFP4_DISPATCH_H_

#include <algorithm>
#include <cstdint>

namespace xla {
namespace gpu {

// Dense NVFP4 Metal matmul path selection — MLX QuantizedMatmul / dispatch_qmv
// / qmm_splitk for transposed weight-only fp (no bias), not ad-hoc mid-batch
// cutoffs:
//   M == 1                         -> qmv
//   2 <= M < vector_limit          -> qmv_wide
//   M >= vector_limit, split_k<=1  -> steel qmm
//   M >= vector_limit, split_k>1   -> steel qmm_t_splitk + sum reduce
//
// vector_limit = MLX get_qmv_batch_limit(K, N, arch). Default arch when the
// host has no query: gen 15, non-'d' (M4-class).
//
// Layout/packing (MLX row-contiguous matrix ABI):
//   x: bf16 [M, K] row-major (K contiguous)
//   w: f4e2m1 packed [N, K/2] (K minor; 2 values/byte)
//   scales: e4m3 [N, K/16]
//   out: bf16 [M, N] row-major
// Steel/splitk honor align_N: our steel BN=64 → N%64==0 selects *_alN.

enum class Nvfp4DensePath : uint8_t {
  kQmv = 0,
  kQmvWide = 1,
  kQmm = 2,
  kQmmSplitK = 3,
};

// Stable fallback for deviceless/AOT compilation and tests. This is MLX's
// post-M2 non-Ultra table, which was also the policy before Metal exposed the
// raw MTL architecture through DeviceDescription.
inline constexpr char kNvfp4DefaultArchSize = ' ';
inline constexpr int kNvfp4DefaultArchGen = 15;

// Steel kernel tiles (mlx_steel_qgemm.h nvfp4_qmm_t / splitk body).
// Split-K *body* reuses BM=16,BN=64 (same as non-split) for correctness.
// Split-K *count* formula uses MLX's bm=bn=32 (see ComputeNvfp4QmmSplitK).
inline constexpr int kNvfp4QmmBM = 16;
inline constexpr int kNvfp4QmmBN = 64;
inline constexpr int kNvfp4SplitkBM = kNvfp4QmmBM;
inline constexpr int kNvfp4SplitkBN = kNvfp4QmmBN;
// MLX qmm_splitk launch tiles for the 512-threadgroup heuristic only.
inline constexpr int kMlxSplitkHeuristicBM = 32;
inline constexpr int kMlxSplitkHeuristicBN = 32;
inline constexpr int kNvfp4GroupSize = 16;

// MoE gather-qmm-rhs uses the MLX Steel BM=16, BN=32, BK=32 body. Its safe
// loader covers partial K/N tiles, and the gather/scatter row copies have
// scalar tails. The only alignment requirement is NVFP4's group-16 K axis.
// moe_argsort is a single-threadgroup counting sort with one bucket per expert
// and supports at most 256 experts.
inline constexpr int kMetalMoeMaxSortedExperts = 256;
// The custom bf16/fp8 kernels predate the MLX-derived NVFP4 path and keep their
// measured large-route cutoff. Keep it next to the shared selector so the HLO
// workspace planner and runtime dispatch cannot allocate different paths.
inline constexpr int64_t kMetalMoeSortedMinR = 1024;

// MLX GatherQMM::eval_gpu gather_qmm_rhs gate for the flattened MoE ABI:
// logical M=1, B=R, and the thunk makes right_sorted=true by sorting routes.
// E is an implementation limit; K/N are the common MoE custom-call ABI.
inline bool ShouldUseNvfp4MoeGatherQmm(int64_t R, int64_t E, int64_t K,
                                       int64_t N) {
  if (R < 16 || E <= 0 || E > kMetalMoeMaxSortedExperts || K <= 0 || N <= 0) {
    return false;
  }
  return R / E >= 4 && K % kNvfp4GroupSize == 0;
}

// Shared sorted-prefill selector for MetalMoeGemvThunk and the HLO workspace
// planner. NVFP4 follows MLX's gather_qmm_rhs reuse gate; bf16/fp8 retain the
// existing R>=1024 policy. Every sorted path uses moe_argsort and therefore
// shares its 256-expert implementation limit.
inline bool ShouldUseMetalMoeSortedPath(int64_t R, int64_t E, int64_t K,
                                        int64_t N, bool is_nvfp4) {
  if (is_nvfp4) return ShouldUseNvfp4MoeGatherQmm(R, E, K, N);
  return R >= kMetalMoeSortedMinR && E > 0 && E <= kMetalMoeMaxSortedExperts &&
         K > 0 && N > 0;
}

// MLX mlx/backend/metal/quantized.cpp get_qmv_batch_limit (verbatim table).
// D = K (contracting), O = N (output cols).
inline int GetNvfp4QmvBatchLimit(int D, int O,
                                 char arch_size = kNvfp4DefaultArchSize,
                                 int arch_gen = kNvfp4DefaultArchGen) {
  if (arch_gen == 13 || arch_gen == 14) {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 14;
        } else if (D <= 4096 && O <= 4096) {
          return 10;
        } else {
          return 6;
        }
    }
  } else {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 18;
        } else if (D <= 4096 && O <= 4096) {
          return 12;
        } else {
          return 10;
        }
    }
  }
}

// MLX qmm_splitk (quantized.cpp): bm=bn=32 for the tile count heuristic;
// target ~512 threadgroups; cap by K/group_size; require
// K % (split_k * group_size) == 0. Returns 1 when split-K is not used.
//
inline int ComputeNvfp4QmmSplitK(int M, int N, int K,
                                 int group_size = kNvfp4GroupSize) {
  if (M <= 0 || N <= 0 || K <= 0 || group_size <= 0) return 1;
  const int64_t n_tiles =
      (static_cast<int64_t>(N) + kMlxSplitkHeuristicBN - 1) /
      kMlxSplitkHeuristicBN;
  const int64_t m_tiles =
      (static_cast<int64_t>(M) + kMlxSplitkHeuristicBM - 1) /
      kMlxSplitkHeuristicBM;
  const int64_t current_tgs = std::max<int64_t>(1, n_tiles * m_tiles);
  int split_k = static_cast<int>(std::max<int64_t>(1, 512 / current_tgs));
  split_k = std::min(split_k, std::max(1, K / group_size));
  while (split_k > 1) {
    if (K % (split_k * group_size) == 0) {
      break;
    }
    split_k--;
  }
  return split_k;
}

// MLX qmv_wide launch: n_tiles = ceil(M/5), vecs_per_tg = ceil(M/n_tiles).
inline int Nvfp4QmvWideVecsPerTg(int M) {
  if (M <= 1) return 1;
  const int n_tiles = (M + 4) / 5;  // ceil(M / 5)
  return (M + n_tiles - 1) / n_tiles;
}

// N-alignment for steel tiles.
inline bool Nvfp4QmmAlignN(int N, int bn = kNvfp4QmmBN) {
  return (N % bn) == 0;
}
inline bool Nvfp4SplitkAlignN(int N) {
  return Nvfp4QmmAlignN(N, kNvfp4SplitkBN);
}

// MLX QuantizedMatmul::eval_gpu (transpose, non-batched):
//   M >= vector_limit → qmm_splitk (falls back to qmm if split_k<=1)
//   else → dispatch_qmv → qmv_wide if M>=2 else qmv
inline Nvfp4DensePath SelectNvfp4DensePath(
    int64_t M, int64_t K, int64_t N, char arch_size = kNvfp4DefaultArchSize,
    int arch_gen = kNvfp4DefaultArchGen) {
  const int limit = GetNvfp4QmvBatchLimit(
      static_cast<int>(K), static_cast<int>(N), arch_size, arch_gen);
  if (M < limit) {
    return (M >= 2) ? Nvfp4DensePath::kQmvWide : Nvfp4DensePath::kQmv;
  }
  const int split_k = ComputeNvfp4QmmSplitK(
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
  return (split_k > 1) ? Nvfp4DensePath::kQmmSplitK : Nvfp4DensePath::kQmm;
}

inline const char* Nvfp4QmvWideKernelName(int vecs_per_tg) {
  switch (vecs_per_tg) {
    case 2:
      return "nvfp4_qmv_wide_2";
    case 3:
      return "nvfp4_qmv_wide_3";
    case 4:
      return "nvfp4_qmv_wide_4";
    case 5:
      return "nvfp4_qmv_wide_5";
    default:
      return nullptr;
  }
}

inline const char* Nvfp4QmmKernelName(int N) {
  return Nvfp4QmmAlignN(N) ? "nvfp4_qmm_t_alN" : "nvfp4_qmm_t";
}

inline const char* Nvfp4QmmSplitkKernelName(int N) {
  return Nvfp4SplitkAlignN(N) ? "nvfp4_qmm_t_splitk_alN" : "nvfp4_qmm_t_splitk";
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_NVFP4_DISPATCH_H_
