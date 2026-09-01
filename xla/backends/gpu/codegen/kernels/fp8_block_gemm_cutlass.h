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

// A block-scaled FP8 GEMM: D[M,N] = (A * a_scale) @ (B * b_scale)^T, where the
// activation carries one f32 scale per row per 128 of K and the weight one f32
// scale per 128x128 tile -- the shape every projection of a block-128 FP8
// checkpoint has. The kernel is CUTLASS's blockwise collective -- the SM100
// warp-specialized one on a datacenter part, the SM120 warp-level-mma one on a
// consumer part -- which promotes the f32 accumulator once per 128 of K inside
// the mainloop; that is the step a Triton emitter cannot hide.
//
// Buffers, all device pointers, every one row-major and every one already in
// the layout its producer emits -- no scale is rewritten on the way in:
//   a         f8e4m3fn [M, K]
//   a_scales  f32 [M, K/128]        (CUTLASS SFA/SFB, K-major)
//   b         f8e4m3fn [N, K]
//   b_scales  f32 [N/128, K/128]    (CUTLASS SFB/SFA, K-major)
//   d         bf16 [M, N]
//
// Both scale tensors are declared to the collective K-major, which is what
// `weight_scale_inv` ships in and what a per-row activation quantizer writes.
// The alternative spelling, MN-major, is the one vLLM uses -- it builds the
// permuted buffer in its quantizer -- and it costs a transpose per GEMM here
// for nothing.
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

// Configs are enumerated rather than described: each one is a distinct CUTLASS
// instantiation compiled into this translation unit, so the set is fixed at
// build time and an autotuner rung picks by index.
int Fp8BlockGemmCutlassNumConfigs();

// "plain_128x128x128_c1x1_tma" or "sm120_swapab_128x32x128".
const char* Fp8BlockGemmCutlassConfigName(int config);

// Whether the config can serve this problem on this GPU. `cc_major`/`cc_minor`
// are the CUDA compute capability -- 10.0 or 10.3 for a GB200/GB300, 12.0 for
// an RTX 5090 -- and a config only runs where its device code was actually
// built with the architecture features it needs (the SM120 half is 12.0 only:
// a plain 12.1 target compiles TMA out). Beyond that, every config takes an
// arbitrary M (a swap-A/B config puts the weight rows on the MMA's M axis so a
// decode-shaped batch tiles exactly, while the plain orientation pads M up to
// the tile), so this is a filter on the architecture and the scale grids, not
// on the batch.
bool Fp8BlockGemmCutlassCanRun(int config, int cc_major, int cc_minor,
                               int64_t m, int64_t n, int64_t k);

size_t Fp8BlockGemmCutlassWorkspaceSize(int config, int64_t m, int64_t n,
                                        int64_t k);

// Returns 0 on success. On failure returns non-zero and, if `error` is
// non-null, points it at a static description. `stream` is a cudaStream_t.
int Fp8BlockGemmCutlassRun(int config, const Fp8BlockGemmCutlassParams& params,
                           void* stream, const char** error);

}  // namespace xla::gpu::kernel

#endif  // XLA_BACKENDS_GPU_CODEGEN_KERNELS_FP8_BLOCK_GEMM_CUTLASS_H_
