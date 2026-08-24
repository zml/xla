// MXFP8/MXFP4 quantized matvec (MIT License, Copyright (c) 2023 Apple Inc.,
// https://github.com/ml-explore/mlx). Entry source for the mlx_mxfp_qmv bundle:
// OCP microscaling, group_size=32, E8M0 scales. Loaded by MetalMxMatmulThunk
// for the decode (thin-M) leg of zml$scaled_matmul.
//
// ABI: x bf16[M,K], w u32-packed, scales u8 E8M0, y bf16[M,N], dims {M,K,N,K/g}.
// Grid (M, ceil(N/8), 1), tg (32, 2, 1).
//
// The upstream text is NOT checked in. The includes below resolve against the
// @mlx archive pinned in //third_party/mlx:workspace.bzl, which
// MetalIncludeRoot() hands to the Metal compiler as -I. So the compiler reads
// upstream's bytes and "verbatim" is true by construction. To change upstream's
// bytes, add a patch to //third_party/mlx:series.bzl, where a bump re-verifies
// it at fetch time.
//
// The includes are upstream's own prologue, copied from
// mlx/backend/metal/kernels/fp_quantized.metal:4-7. fp_quantized.h has no
// #pragma once and no self-includes -- cold, it does not even name float16_t --
// so it only compiles behind the prologue its own .metal file establishes.
//
// This is the most upstream bundle of the family: the flattened copy carried
// none of our forks. fp_qmv_impl and fp_qmv_fast_impl were byte-identical
// bodies, and the group-32 instantiation reaches E8M0 through upstream's own
// dequantize_scale<T, group_size>, which selects fp8_e8m0 at any group size but
// 16. So everything below the includes is the four entry points, and nothing
// else.
//
// These entries are covered by runtime golden values:
// backends/gpu/runtime/metal_mx_kernel_test.cc checks both formats against the
// exact arithmetic answer on a real device, over qmv_fast, the guarded qmv and
// a partial N; the thin-batch case is MXFP8 only.
//
// TODO: the k_aligned fix that //third_party/mlx:series.bzl argues about would
// reach this path -- it patches fp_qmv_impl in the archive, which this bundle
// includes unmodified (the NVFP4 bundle calls our own fork instead, so a patch
// would not move it). Landing it therefore moves MXFP codegen, which the
// goldens above check for correctness but nothing checks for speed. Mint an
// MXFP8 bench before taking that patch.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"
// clang-format on

// ---- thin entry points (ours) ----
// ABI: buffer0 x, buffer1 w, buffer2 scales, buffer3 y, buffer4 dims
// (see file header). Grid = (M, ceil(N / 8), 1), threadgroup = (32, 2, 1).
//
// XLA DELTA: dims is read through a pointer cast rather than as `dims.y`.
// Upstream's fp_qmv_impl takes `const constant int&`, and a vector swizzle is an
// rvalue, so `dims.y` cannot bind to it ("reference of type 'const constant int
// &' cannot bind to a temporary object because of address space mismatch") --
// which is why the flattened bundle forked the signature to take int by value
// instead. Indexing the constant-space object keeps BOTH upstream's signature
// and this bundle's 16-byte setBytes layout, so neither side has to move: the
// emitted GEPs are lanes 1 and 2 of the same `<4 x i32> addrspace(2)*` that the
// by-value version loaded whole. Measured cost is one instruction per kernel --
// a 16-byte vector load + 2 extractelement becomes 2 scalar loads, which LLVM
// hoists and CSEs, so `const constant int&` does NOT reload per use (the
// constant address space is read-only). Every other instruction is unmoved.
#define MXFP_QMV_ENTRY(NAME, IMPL, GS, BITS)                             \
  kernel void NAME(                                                      \
      const device bfloat* x [[buffer(0)]],                              \
      const device uint32_t* w [[buffer(1)]],                            \
      const device uint8_t* scales [[buffer(2)]],                        \
      device bfloat* y [[buffer(3)]],                                    \
      const constant int4& dims [[buffer(4)]],                           \
      uint3 tid [[threadgroup_position_in_grid]],                        \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                  \
      uint simd_lid [[thread_index_in_simdgroup]]) {                     \
    IMPL<bfloat, GS, BITS>(w, scales, x, y, ((constant int*)&dims)[1],   \
                           ((constant int*)&dims)[2], tid, simd_gid,     \
                           simd_lid);                                    \
  }

MXFP_QMV_ENTRY(mxfp8_qmv_fast, fp_qmv_fast_impl, 32, 8)
MXFP_QMV_ENTRY(mxfp8_qmv, fp_qmv_impl, 32, 8)
MXFP_QMV_ENTRY(mxfp4_qmv_fast, fp_qmv_fast_impl, 32, 4)
MXFP_QMV_ENTRY(mxfp4_qmv, fp_qmv_impl, 32, 4)
