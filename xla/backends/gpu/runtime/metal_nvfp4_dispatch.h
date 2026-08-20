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
#include <cstdlib>

namespace xla {
namespace gpu {

enum class Nvfp4DensePath : uint8_t {
  kQmv = 0,
  kQmvWide = 1,
  kQmm = 2,
  kQmmSplitK = 3,
};

inline constexpr char kNvfp4DefaultArchSize = ' ';
inline constexpr int kNvfp4DefaultArchGen = 15;

inline constexpr int kNvfp4QmmBM = 16;
inline constexpr int kNvfp4QmmBN = 64;
inline constexpr int kNvfp4SplitkBM = kNvfp4QmmBM;
inline constexpr int kNvfp4SplitkBN = kNvfp4QmmBN;
inline constexpr int kMlxSplitkHeuristicBM = 32;
inline constexpr int kMlxSplitkHeuristicBN = 32;
inline constexpr int kNvfp4GroupSize = 16;

inline constexpr int kMetalMoeMaxSortedExperts = 256;
inline constexpr int64_t kMetalMoeSortedMinR = 1024;

inline bool ShouldUseNvfp4MoeGatherQmm(int64_t R, int64_t E, int64_t K,
                                       int64_t N) {
  if (R < 16 || E <= 0 || E > kMetalMoeMaxSortedExperts || K <= 0 || N <= 0) {
    return false;
  }
  return R / E >= 4 && K % kNvfp4GroupSize == 0;
}

inline bool ShouldUseMetalMoeSortedPath(int64_t R, int64_t E, int64_t K,
                                        int64_t N, bool is_nvfp4) {
  if (is_nvfp4) return ShouldUseNvfp4MoeGatherQmm(R, E, K, N);
  return R >= kMetalMoeSortedMinR && E > 0 && E <= kMetalMoeMaxSortedExperts &&
         K > 0 && N > 0;
}

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

inline constexpr int kNvfp4QmvWideMaxVecs = 12;
inline int Nvfp4QmvWideMaxVecs() {
  static const int v = [] {
    const char* e = std::getenv("METAL_NVFP4_WIDE_VECS");
    const int n = e ? std::atoi(e) : 0;
    return (n >= 2 && n <= kNvfp4QmvWideMaxVecs) ? n : 5;
  }();
  return v;
}

inline int Nvfp4QmvWideVecsPerTg(int M) {
  if (M <= 1) return 1;
  const int cap = Nvfp4QmvWideMaxVecs();
  const int n_tiles = (M + cap - 1) / cap;
  return (M + n_tiles - 1) / n_tiles;
}

inline bool Nvfp4QmmAlignN(int N, int bn = kNvfp4QmmBN) {
  return (N % bn) == 0;
}
inline bool Nvfp4SplitkAlignN(int N) {
  return Nvfp4QmmAlignN(N, kNvfp4SplitkBN);
}

inline int Nvfp4QmvBatchLimitOverride() {
  static const int v = [] {
    const char* e = std::getenv("METAL_NVFP4_QMV_MAX");
    const int n = e ? std::atoi(e) : 0;
    return n > 0 ? n : 0;
  }();
  return v;
}

inline Nvfp4DensePath SelectNvfp4DensePath(
    int64_t M, int64_t K, int64_t N, char arch_size = kNvfp4DefaultArchSize,
    int arch_gen = kNvfp4DefaultArchGen) {
  const int override_limit = Nvfp4QmvBatchLimitOverride();
  const int limit = override_limit > 0
                        ? override_limit
                        : GetNvfp4QmvBatchLimit(static_cast<int>(K),
                                                static_cast<int>(N), arch_size,
                                                arch_gen);
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
    case 6:
      return "nvfp4_qmv_wide_6";
    case 7:
      return "nvfp4_qmv_wide_7";
    case 8:
      return "nvfp4_qmv_wide_8";
    case 9:
      return "nvfp4_qmv_wide_9";
    case 10:
      return "nvfp4_qmv_wide_10";
    case 11:
      return "nvfp4_qmv_wide_11";
    case 12:
      return "nvfp4_qmv_wide_12";
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
