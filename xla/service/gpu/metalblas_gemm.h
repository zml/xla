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

#ifndef XLA_SERVICE_GPU_METALBLAS_GEMM_H_
#define XLA_SERVICE_GPU_METALBLAS_GEMM_H_

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Everything needed to launch one metalBLAS GEMM kernel on a Metal device: the
// compiled metallib bytes, the kernel entry name, the launch dims, and the
// MBTensorDims{M,N,K,lda,ldb,ldc} params buffer the kernel reads at buffer(3).
struct MetalGemmLaunch {
  std::vector<uint8_t> metallib;
  std::string kernel_name;
  ::stream_executor::ThreadDim thread_dim;
  ::stream_executor::BlockDim block_dim;
  // The mpp_tensor_gemm params buffer: MBTensorDims {M, N, K, lda, ldb, ldc}.
  std::array<uint32_t, 6> params;
  // The M-tile size (BM) and swizzle log used to derive block_dim.y from the
  // M-tile count. Kept so a PREFILL GEMM can recompute block_dim.y at execute for
  // a runtime-clamped row count (M' = num_tokens ≤ M) WITHOUT touching the baked
  // SWIZZLE_LOG (compiled into the metallib), block_dim.x, or params.M (the kernel
  // bounds writes by params.M, so the few padding rows in the last dispatched
  // swizzle block stay in-bounds and unread).
  uint32_t bm = 0;
  uint32_t swizzle_log = 0;
  // GEMV kernels (gemv_t/gemv_bt) compute y = x @ B with the *matrix* B at
  // buffer(0) and the *vector* x at buffer(1) — the opposite of the GEMM (A,B,C)
  // order. When set, MetalGemmThunk binds rhs->buffer(0), lhs->buffer(1). The
  // prefill M-clamp self-skips for GEMV because bm stays 0.
  bool swap_ab = false;

  // MLX steel split-K (two-dispatch) family, set by CompileMetalblasSplitk.
  // splitk_partitions > 0 switches the thunk to: dispatch `kernel_name`
  // (gemm_splitk: A=x b0, B=W b1, f32 staging b2, GEMMSpiltKParams b3, grid =
  // (tiles_n, tiles_m, partitions)) then `accum_kernel_name`
  // (gemm_splitk_accum: staging b0, out b1, three constant ints at b2/b3/b4 —
  // bound at +0/+4/+8 of one tiny staged buffer). splitk_params is the 13-int
  // GEMMSpiltKParams image; staging_bytes = partitions*M*N*4.
  std::string accum_kernel_name;
  ::stream_executor::ThreadDim accum_thread_dim;
  ::stream_executor::BlockDim accum_block_dim;
  uint32_t splitk_partitions = 0;
  uint64_t staging_bytes = 0;
  std::array<uint32_t, 13> splitk_params{};
  std::array<uint32_t, 3> accum_params{};  // {k_partitions, partition_stride, ldd}
};

// Runs the imported metalBLAS dispatcher for a GEMM of the given geometry/dtype
// (row-major operands; trans_a/trans_b say whether each operand is stored
// transposed relative to op(A)=M×K, op(B)=K×N) and compiles the chosen kernel
// to a metallib. `dtype` must be F32, F16 or BF16.
//
// Wired: the mpp_tensor cooperative-tensor kernel (metalBLAS's primary GEMM
// backend), which handles NN/NT/TN/TT via matmul2d tensor views + a 3-tuple
// (BM,BN,NSG) tile. Non-float dtypes return UnimplementedError; callers fall
// back / fail loud (D9: never silently wrong). (GEMV / split-K / conv / int /
// complex / batched paths exist upstream but are not routed here yet.)
// `prefill_token_axis` marks a per-layer token-axis prefill GEMM ([seqlen,*] =
// QKV/O/gate+up/down): it is COMPILED at M=seqlen but the thunk clamps the M-row
// grid to the real prompt length (num_tokens ≤ seqlen) at execute. pick_m5_tensor_tile
// sees the padded M=seqlen and picks a large-BM tile that under-occupies the GPU
// once clamped to a short prompt; for these we pick a small-BM, high-threadgroup-
// count tile instead (measured 1.73x faster end-to-end, bit-identical output).
absl::StatusOr<MetalGemmLaunch> CompileMetalblasGemm(int64_t M, int64_t N,
                                                     int64_t K, bool trans_a,
                                                     bool trans_b,
                                                     PrimitiveType dtype,
                                                     bool prefill_token_axis);

// Compiles a dedicated metalBLAS GEMV kernel for a GEMV-shaped, row-major
// (non-transposed) matmul y = x @ B, picking VEC/NWARPS via the imported
// dispatcher. Routed cases:
//   * gemv_t  — M==1, N>=16 (the decode x@W projections + lm_head).
//   * gemv_bt — thin-M batch (2<=M<=16, f16/bf16, is_gemv_bt_regime): one B
//     stream feeds all MROWS rows, recovering the bandwidth mpp_tensor loses on
//     thin M.
//   * gemv_bt TRANS_B=1 — thin-M x[M,K] @ W[N,K]ᵀ with N <= 4096, K <= 8192
//     (the batched-decode attention + MLP-down projections, llmd batch_size=16):
//     one K-contiguous W column per warp, NCOLS register blocking. Routing +
//     spec come from the dispatcher port's decide() (PORT POLICY #2 in
//     metalblas_dispatch.cc / gen_golden.py); measured in
//     metal-xla-docs/scratch/air-ref/2026-06-10-gemv_bt_thin_m.mm.
//     METAL_GEMV_BT=0 disables, METAL_GEMV_BT=v,nw,nc pins the spec.
// `b_byte_offset` is the B (rhs) operand's slice offset; its element alignment is
// OR-ed into the VEC clamp (with the row stride) so a VEC-wide load never
// straddles. Returns UnimplementedError for any other shape/dtype/transpose so
// the caller falls back to CompileMetalblasGemm (D9: never silently wrong).
absl::StatusOr<MetalGemmLaunch> CompileMetalblasGemv(int64_t M, int64_t N,
                                                     int64_t K, bool trans_a,
                                                     bool trans_b,
                                                     PrimitiveType dtype,
                                                     int64_t b_byte_offset);

// Compiles the MLX steel split-K GEMM (vendored flattened in
// metal_kernels/vendored/mlx/mlx_steel_splitk.h, MIT) for a thin-M batched-decode
// x[M,K] @ W[N,K]ᵀ: grid-level K-split — (tiles_n × tiles_m × partitions)
// threadgroups each MMA-ing a K-chunk into f32 partial planes, plus a trivially
// parallel accum pass. Harness-measured (2026-06-11-steel_splitk_thin_m.mm,
// M4 Max, CPU-oracle): kv [16,3072]→1024 16.7us (gemv_bt 25.4), qo →3072
// 39.7us (50.7), down [16,8192]→3072 102.8us (147.4), gateup →8192 96.3us
// (tensor 133) — 380-520 GB/s ≈ bandwidth peak. lm_head-scale N measured a
// wash → regime is capped at N <= 8192. Returns UnimplementedError outside
// the regime (thin-M NT f16/bf16, 2<=M<=16, 512<=N<=8192, K>=2048 with
// K%16==0) so the caller falls through to GEMV/GEMM. METAL_SPLITK=0 disables;
// METAL_SPLITK_PARTS pins the partition count.
absl::StatusOr<MetalGemmLaunch> CompileMetalblasSplitk(int64_t M, int64_t N,
                                                       int64_t K, bool trans_a,
                                                       bool trans_b,
                                                       PrimitiveType dtype);

// Assembles + compiles one metalBLAS MSL kernel family to a metallib: seeds the
// shared epilogue (mb_epi.h, the only always-active metalBLAS header) then
// appends `family_source` (a single vendored family header verbatim), enables it
// via `#define <build_flag> 1`, substitutes its `__TOKEN__` tile/dtype
// placeholders, and runs the Metal compiler then `metallib`.
// Compiling `mb_epi + <one family>` is byte-identical to compiling the metalBLAS
// binder with that single MB_BUILD flag set — every other family's body is
// `#ifdef MB_BUILD_<NAME>`-guarded and compiles to nothing — so no binder /
// build-time include-inlining is needed. `extra_defines` is prepended verbatim
// after the build flag — for family options that are `#ifndef`-defaulted in the
// shader rather than `__TOKEN__` placeholders (e.g. gemv_bt's TRANS_B).
absl::StatusOr<std::vector<uint8_t>> CompileMetalblasKernelToMetallib(
    absl::string_view build_flag, absl::string_view family_source,
    absl::Span<const std::pair<absl::string_view, std::string>> subs,
    absl::string_view extra_defines = "");

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METALBLAS_GEMM_H_
