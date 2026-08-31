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

#ifndef XLA_BACKENDS_GPU_CODEGEN_KERNELS_FP8_BLOCK_GEMM_CUTLASS_H_
#define XLA_BACKENDS_GPU_CODEGEN_KERNELS_FP8_BLOCK_GEMM_CUTLASS_H_

#include <cstddef>
#include <cstdint>

namespace xla::gpu::kernel {

// D[M,N] = (A * a_scales) @ (B * b_scales)^T through CUTLASS's blockwise collective. Every buffer
// is a row-major device pointer in the layout its producer emits, both scales K-major:
//   a         f8e4m3fn [M, K]
//   a_scales  f32 [M, K/128]
//   b         f8e4m3fn [N, K]
//   b_scales  f32 [N/128, K/128]
//   d         bf16 [M, N]
struct Fp8BlockGemmCutlassParams {
  const void* a = nullptr;
  const void* a_scales = nullptr;
  const void* b = nullptr;
  const void* b_scales = nullptr;
  void* d = nullptr;
  void* workspace = nullptr;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

// One compiled instantiation per config; the set is fixed at build time.
int Fp8BlockGemmCutlassNumConfigs();

// "plain_128x128x128_c1x1_tma" or "sm120_swapab_128x32x128".
const char* Fp8BlockGemmCutlassConfigName(int config);

// Whether `config` was built for this compute capability and the scale grids are whole. Any M runs.
bool Fp8BlockGemmCutlassCanRun(int config, int cc_major, int cc_minor,
                               int64_t m, int64_t n, int64_t k);

size_t Fp8BlockGemmCutlassWorkspaceSize(int config, int64_t m, int64_t n,
                                        int64_t k);

// Returns 0 on success, else non-zero with a static description in `error`. `stream` is a cudaStream_t.
int Fp8BlockGemmCutlassRun(int config, const Fp8BlockGemmCutlassParams& params,
                           void* stream, const char** error);

}  // namespace xla::gpu::kernel

#endif  // XLA_BACKENDS_GPU_CODEGEN_KERNELS_FP8_BLOCK_GEMM_CUTLASS_H_
