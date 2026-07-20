// MLX tiled quantized Steel GEMM (MIT License, Copyright (c) 2023 Apple Inc.,
// https://github.com/ml-explore/mlx). Entry source for the mlx_steel_qgemm
// bundle: the Steel simdgroup-matrix q-GEMM with MLX's dequant swapped for our
// schemes -- DeepSeek-style 128x128 block FP8 (dense + MoE gather), MXFP
// group-32, and NVFP4 group-16 -- plus the bf16 MoE gather twin.
//
// The upstream text is NOT checked in. The includes below resolve against the
// @mlx archive pinned in //third_party/mlx:workspace.bzl, which
// MetalIncludeRoot() hands to the Metal compiler as -I. So the compiler reads
// upstream's bytes and "verbatim" is true by construction. To change upstream's
// bytes, add a patch to //third_party/mlx:series.bzl, where a bump re-verifies
// it at fetch time.
//
// The includes are the first three lines of upstream's own prologue, copied from
// mlx/backend/metal/kernels/fp_quantized.metal:4-6. fp_quantized.h, its fourth,
// is deliberately NOT included: this bundle does not use upstream's q-GEMM
// impls (ours take a different ABI and are templated on our own loaders), and
// pulling it in would put a same-named fp_qmm_t_impl / fp_gather_qmm_rhs_impl
// into the overload set for no gain. What the first three give us is what this
// file actually needs:
//   * utils.h + steel/gemm/gemm.h -- the Steel machinery (BlockLoader, BlockMMA,
//     BaseMMAFrag/MMATile, TransformNone, integral_constant, complex64_t). This
//     is the 2,653-line span the flattened bundle carried, which was byte-
//     identical to the split-K bundle's copy of the same span: one duplicate,
//     two files, kept in sync by nobody. The archive dissolves it.
//   * quantized_utils.h -- gemm_loop_aligned / gemm_loop_unaligned /
//     gemm_loop_finalize, which the gather kernels call.
//
// Everything below is OURS: the block-FP8 / NVFP4 / MoE loaders, impls and entry
// points that have no upstream counterpart, plus two adaptations of upstream
// bodies that are marked XLA DELTA at their definitions -- XlaQuantizedBlockLoader
// (a rename-fork) and the dequantize_scale_mx / DequantizeMx pair.
//
// This bundle ships the dense block-FP8 path Qwen3.6-27B-FP8 decodes on
// (fp8_qmm_t{,_bm64,_pc} via MetalFp8GemvThunk), the MoE block-FP8 gather path
// (fp8_gather_qmm_rhs via MetalMoeGemvThunk / __metal$moe_gemm$f8), and the
// NVFP4 MoE path gemma-4-26B-A4B-NVFP4 decodes on (nvfp4_gather_qmm_rhs via
// MetalMoeGemvThunk). Dense + MoE FP8 share Fp8BlockLoader.
//
// NOTE: this bundle is a standalone TU compiled by CompileMetalSourceToMetallib.
// It is a .h rather than a .metal only to match the family's convention -- the
// bundles it sits beside are fragments that cannot type-check standalone.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
// clang-format on

///////////////////////////////////////////////////////////////////////////////
// FP8 block-quantized GEMM (DeepSeek 128x128 block scales)
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

template <int wsize = 8, int bits = 4>
inline constexpr short get_pack_factor() {
  return wsize / bits;
}

template <int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  return wsize / 8;
}

// Decode one OCP f8e4m3fn byte to float (exponent bias 7, 3-bit mantissa, no
// inf; 0x7f/0xff encode NaN, which block-quantized weights never produce).
//   e==0 : subnormal,  value = m * 2^-9
//   e!=0 : normal,     value = (1 + m/8) * 2^(e-7) = (8+m) * 2^(e-10)
static inline float decode_e4m3fn(uchar b) {
  int s = (b >> 7) & 0x1;
  int e = (b >> 3) & 0xf;
  int m = b & 0x7;
  float v = (e == 0) ? (float(m) * 1.953125e-3f)
                     : (float(8 + m) * exp2(float(e) - 10.0f));
  return s ? -v : v;
}

// Function constants matching MLX's fp_quantized.h (align_M/N/K), used by the
// gather kernel to elide bounds checks when the tile is known to be aligned.
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

// NVFP4 MoE only: whether the call carries a trailing f32[E] per-expert global
// scale (the compressed-tensors weight-encode divisor). When unset, no buffer
// is bound and nvfp4_gather_qmm_rhs behaves exactly as before. The MX entries
// never reference this constant.
//
// XLA DELTA: 440 is the same logical constant as mlx_fp4_qmv.h's; both bundles
// bind it from MetalMoeGemvThunk. It sits in our 4xx range because upstream MLX
// owns 0-26, 100, 101, 110, 199, 200-202 and 300-302 across its kernel tree, and
// our custom/ shaders already number from 420. Never allocate ours below 400.
constant bool moe_has_global_scale [[function_constant(440)]];

///////////////////////////////////////////////////////////////////////////////
// Fp8BlockLoader — DeepSeek 128x128-block adaptation of MLX QuantizedBlockLoader
//
// bits is hardcoded to 8 (=> pack_factor = bytes_per_pack = 1, so the f8 byte
// stream and the dst tile share the same column count). group_size is 128 in
// both K and N. `scales` is a bf16 tensor with one entry per 128x128 weight
// block; the value is shared across the whole block (no per-row scale). Each
// load reads the single block scale once and applies it to every decoded byte.
// In next(), for reduction_dim==1 (the qmm_t / transpose layout) the scale only
// advances after group_steps = 128/BK K-tiles (one full 128-column block).
///////////////////////////////////////////////////////////////////////////////
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct Fp8BlockLoader {
  MLX_MTL_CONST short pack_factor = 1;     // bits == 8
  MLX_MTL_CONST short bytes_per_pack = 1;  // bits == 8
  MLX_MTL_CONST short group_size = 128;    // DeepSeek block edge
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size < BCOLS ? 1 : group_size / BCOLS;
  MLX_MTL_CONST short scale_step = group_size < BCOLS ? BCOLS / group_size : 1;

  static_assert(
      (n_reads * pack_factor) <= group_size,
      "The number of reads per thread must be less than the group size.");

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uchar* src;
  const device T* scales;

  Fp8BlockLoader(
      const device uchar* src_,
      const device T* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        // Block stride in scale rows when walking the non-reduction dim
        // (reduction_dim == 0). Both kernels here use reduction_dim == 1
        // (transpose layout), so this path is never executed; the value is the
        // 2-D-block analogue of MLX's BROWS*src_ld/group_size for completeness.
        // TODO: validate this stride if a reduction_dim == 0 (n-major) variant
        // is ever added.
        group_stride(BROWS * (src_ld / group_size) / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        // 2-D block-scale base: the source LD in *blocks* is src_ld/128, and the
        // thread's row block is bi/128 (always 0 here since BROWS <= 128), its
        // K block is bj/128 (also 0 since bj < BCOLS <= 128). The tile's own
        // (N-block, K-block) offset is already folded into scales_ by the impl.
        scales(scales_ + (bi / group_size) * (src_ld / group_size) +
               (bj / group_size)) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    for (int i = 0; i < n_reads; i++) {
      dst[i] = static_cast<T>(decode_e4m3fn(src[i * bytes_per_pack])) * scale;
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    for (int i = 0; i < n_reads; i++) {
      dst[i] = static_cast<T>(decode_e4m3fn(src[i * bytes_per_pack])) * scale;
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
        }
      } else {
        scales += scale_step;
      }
    } else {
      scales += group_stride;
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// fp_qmm_t_impl — dense tiled q-GEMM (transpose=true). Adapted from MLX's
// fp_qmm_t_impl: QuantizedBlockLoader -> Fp8BlockLoader, and the per-N-tile
// scale base changed from the 1-D `scales += y_col * K_g` to the 2-D block form
// `scales += (y_col/128) * (K/128)`. `scales` is bf16 (const device T*).
///////////////////////////////////////////////////////////////////////////////
template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 16,
    const int BK = 32,
    const int BN = 64>
METAL_FUNC void fp_qmm_t_impl(
    const device uchar* w,
    const device T* scales,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    const int K_eff,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t =
      Fp8BlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;

  // Set the block. K is in elements (pack_factor == 1 for f8). The scale tensor
  // is [N/128, K/128] bf16, so its leading dim (in entries) is K/128.
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uchar*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * static_cast<int64_t>(K);
  scales += (y_col / group_size) * K_g;  // 2-D block-scale base
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

///////////////////////////////////////////////////////////////////////////////
// The per-K-tile gemm loop helpers the gather kernels call -- gemm_loop_aligned,
// gemm_loop_unaligned and gemm_loop_finalize -- now come from upstream's
// quantized_utils.h via the include above. The copy that used to sit here was
// byte-identical to it (diff-zero against quantized_utils.h:6-EOF).
///////////////////////////////////////////////////////////////////////////////

// fp_gather_qmm_rhs_impl — MoE gather q-GEMM (transpose=true). Adapted from
// MLX's fp_gather_qmm_rhs: QuantizedBlockLoader -> Fp8BlockLoader; the
// per-expert scale stride stride_s = N * K_g becomes the 2-D block form
// (N/128) * (K/128), the expert base is index * stride_s, and the per-N-tile
// base advance scales += y_col * K_g becomes scales += (y_col/128) * (K/128).
// `scales` is bf16 (const device T*).
///////////////////////////////////////////////////////////////////////////////
template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
METAL_FUNC void fp_gather_qmm_rhs_impl(
    const device T* x,
    const device uchar* w,
    const device T* scales,
    const device uint32_t* indices,
    device T* y,
    const int M,
    const int N,
    const int K,
    threadgroup T* Xs,
    threadgroup T* Ws,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      transpose,
      BK_padded,
      transpose ? BK_padded : BN_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = Fp8BlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE>;

  // Compute the block. Weights are f8 (pack_factor == 1) so K_w == K, N_w == N.
  // The scale tensor is [E, N/128, K/128] bf16; per-expert stride and per-tile
  // base are in *blocks*.
  const int K_g = K / group_size;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? size_t(N) * K : size_t(K) * N;
  const size_t stride_s = transpose ? size_t(N_g) * K_g : size_t(K) * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uchar*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K : y_col;
  scales += transpose ? (y_col_long / group_size) * K_g : (y_col / group_size);

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    // Prepare threadgroup loading operations
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    // Matrices are all aligned check nothing
    if (align_M && align_N) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }

      // Store results to device memory
      if (offset_next - offset == BM) {
        mma_op.store_result(y, N);
      } else {
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }
    } else {
      // Tile aligned so check outside of the hot loop
      if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
        gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }

        // Store results to device memory
        if (offset_next - offset == BM) {
          mma_op.store_result(y, N);
        } else {
          mma_op.store_result_slice(
              y, N, short2(0, offset), short2(BN, offset_next));
        }
      }

      // Tile partially aligned check rows
      else if (align_N || tgp_bn == BN) {
        gemm_loop_unaligned<false, true, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }

      // Tile partially aligned check cols
      else if (align_M || tgp_bm == BM) {
        gemm_loop_unaligned<true, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }

      // Nothing aligned so check both rows and cols
      else {
        gemm_loop_unaligned<false, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(
              Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// MXFP8 / MXFP4 (OCP microscaling) tiled q-GEMM (prefill).

///////////////////////////////////////////////////////////////////////////////
// NVFP4 tiled q-GEMM (prefill) -- the 1-D group-scale family.
//
// Unlike the DeepSeek Fp8BlockLoader above (128x128 bf16 block scale), this is
// MLX's ORIGINAL fp_quantized path: a per-(output-row, K-group) uint8 scale and
// a sub-byte weight. The helpers XlaQuantizedBlockLoader / mxfp_qmm_t_impl are
// derived from MLX's fp8.h / fp4.h / fp_quantized.h, with XLA constant-buffer
// ABI, rectangular-tile bounds, and K tail handling, reusing the same Steel
// BlockMMA / BlockLoader core.
//
// This family was written for OCP microscaling (mxfp8/mxfp4: group-32 E8M0
// scales), hence the mxfp_/`_mx` names throughout. Those entries are gone -- no
// model emits e8m0 -- and NVFP4 (group-16 e4m3 scales, E2M1 weights) is the only
// scheme left instantiating them.
///////////////////////////////////////////////////////////////////////////////

// fp8.h / fp4.h value+scale decodes (verbatim, MLX numerics).
//
// XLA DELTA: these are renamed copies of upstream's fp8_e4m3 / fp4_e2m1
// (fp8.h, fp4.h), reachable only through fp_quantized.h -- which this
// TU deliberately does not include (see the file header). The `mlx_` prefix is
// a flattening artifact from when this bundle was one TU with the Steel span
// and the names collided; it is NOT a fork. Each body is upstream's, and each
// keeps only the float conversion this file uses.
// TODO: these become plain deletions the day this TU can include fp_quantized.h
// -- i.e. once our fp_qmm_t_impl / fp_gather_qmm_rhs_impl are either renamed off
// upstream's names or replaced by upstream's own, via the ScaleDecoder seam.
struct mlx_fp8_e4m3 {
  operator float() {
    uint16_t v = (bits & 127) << 7;
    half converted = as_type<half>(v);
    converted *= 256.0;
    auto sign = bits & 128;
    return static_cast<float>(sign ? -converted : converted);
  }
  operator bfloat() { return static_cast<bfloat>(this->operator float()); }
  uint8_t bits;
};
struct mlx_fp4_e2m1 {
  operator float() {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return static_cast<float>(bits & 8 ? -converted : converted);
  }
  operator bfloat() { return static_cast<bfloat>(this->operator float()); }
  uint8_t bits;
};

// NVFP4's group-16 e4m3 scale decode. The E8M0 arm (group_size 32) went with
// the mxfp entries; static_assert rather than a silent fallthrough, so adding a
// group size back is a compile error and not a wrong answer.
template <typename T, int group_size>
static inline T dequantize_scale_mx(uint8_t s) {
  static_assert(group_size == 16, "only NVFP4 group-16 e4m3 scales remain");
  return T(*(thread mlx_fp8_e4m3*)(&s));
}

// NVFP4 weight decode. The bits==8 arm decoded mxfp8's e4m3 WEIGHTS (not to be
// confused with e4m3 SCALES above) and went with the mxfp entries.
template <int bits, typename U = float>
struct DequantizeMx {
  U operator()(uint8_t x) {
    static_assert(bits == 4, "only NVFP4 4-bit weights remain");
    return U(*(thread mlx_fp4_e2m1*)(&x));
  }
};

template <typename U, int bits>
inline void dequantize_mx(uint8_t w, U scale, threadgroup U* w_local) {
  static_assert(bits == 4, "only NVFP4 4-bit weights remain");
  w_local[0] = scale * DequantizeMx<4, U>{}(w);
  w_local[1] = scale * DequantizeMx<4, U>{}(w >> 4);
}

// NVFP4 MoE global-scale fold: same as dequantize_mx, but the group scale
// arrives in f32 (already multiplied by the per-expert global reciprocal) and
// is rounded into the bf16 tile exactly once. Rounding the folded scale to
// bf16 first would round twice: an e4m3 scale times an f4 value is otherwise
// exact in bf16, so the unfolded path stores an exact product.
template <typename T, int bits>
inline void dequantize_mx_global(uint8_t w, float scale, threadgroup T* w_local) {
  static_assert(bits == 4, "only NVFP4 4-bit weights remain");
  w_local[0] = T(scale * DequantizeMx<4, float>{}(w));
  w_local[1] = T(scale * DequantizeMx<4, float>{}(w >> 4));
}


// XlaQuantizedBlockLoader — 1-D per-(row, K-group) uint8 scales.
//
// XLA DELTA: a RENAME-FORK of upstream's QuantizedBlockLoader (fp_quantized.h).
// The fork is real: upstream's reduction_dim == 1 safe-load path guards only the
// whole row (`if (reduction_dim == 1 && bi >= src_tile_dim.x)`), while ours
// checks every packed K read below -- one thread can own several packs, so a
// valid first pack does not imply that all of its later packs are valid.
// Upstream's own BN == BK tile makes the row/column distinction unobservable, so
// the bug is latent there; the rectangular 64x32 tile below requires both
// dimensions to be checked independently, which is how we found it. The members
// are also widened to int64_t (tile_stride / group_stride), since upstream's
// int32 `out_row * in_vec_size_w` overflows at large N*K.
//
// It is a rename rather than a patch on purpose: a rename-fork can never collide
// on a bump and can never silently un-apply, which a patch to a body can. It
// plugs into upstream's own shape through the `using loader_w_t = ...` typedef
// in mxfp_qmm_t_impl below.
//
// TODO: file the per-pack K-tail guard upstream -- it is a genuine correctness
// bug in MLX for any rectangular tile -- and the int64_t widening with it. If
// both land, this whole struct becomes a deletion.
constexpr bool quantized_mx_row_in_bounds(short row, short valid_rows) {
  return row < valid_rows;
}

template <short pack_factor>
constexpr short quantized_mx_valid_values(
    short packed_col, short read, short valid_k) {
  const short k = (packed_col + read) * pack_factor;
  const short remaining = valid_k - k;
  return remaining <= 0
      ? 0
      : (remaining < pack_factor ? remaining : pack_factor);
}

// Rectangular BN=64, BK=32 regression coverage for the f4 (two/value) loader.
static_assert(quantized_mx_row_in_bounds(62, 63), "row 62 must be loaded");
static_assert(
    !quantized_mx_row_in_bounds(63, 63), "row 63 must be zero-filled");
static_assert(
    quantized_mx_valid_values<2>(8, 1, 20) == 2,
    "the packed K=18 read must be loaded");
static_assert(
    quantized_mx_valid_values<2>(8, 2, 20) == 0,
    "the packed K=20 read must be zero-filled");
static_assert(
    quantized_mx_valid_values<2>(8, 0, 17) == 1,
    "a partial f4 pack must preserve only its valid value");

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct XlaQuantizedBlockLoader {
  MLX_MTL_CONST short pack_factor = get_pack_factor<8, bits>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size < BCOLS ? 1 : group_size / BCOLS;
  MLX_MTL_CONST short scale_step = group_size < BCOLS ? BCOLS / group_size : 1;

  static_assert(
      (n_reads * pack_factor) <= group_size,
      "The number of reads per thread must be less than the group size.");

  const int src_ld;
  const int64_t tile_stride;
  short group_step_cnt;
  const int64_t group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device uint8_t* scales;

  XlaQuantizedBlockLoader(
      const device uint8_t* src_,
      const device uint8_t* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id,
      ushort simd_lane_id)
      : src_ld(src_ld_),
        tile_stride(reduction_dim
                        ? static_cast<int64_t>(BCOLS_PACKED) * bytes_per_pack
                        : static_cast<int64_t>(BROWS) * src_ld *
                            bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(
            static_cast<int64_t>(BROWS) * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + static_cast<int64_t>(bi) * src_ld * bytes_per_pack /
                pack_factor +
            static_cast<int64_t>(bj) * bytes_per_pack),
        scales(
            scales_ + static_cast<int64_t>(bi) * src_ld / group_size +
            (bj * pack_factor) / group_size) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }
    T scale = dequantize_scale_mx<T, group_size>(*scales);
    for (int i = 0; i < n_reads; i++) {
      dequantize_mx<T, bits>(src[i * bytes_per_pack], scale, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    // Steel's source-tile convention is (valid columns, valid rows).  For the
    // transposed quantized-weight loader, rows are output channels and columns
    // are K.  Check every packed K read: one thread can own several packs, so a
    // valid first pack does not imply that all of its later packs are valid.
    if (reduction_dim == 1) {
      const short valid_k = src_tile_dim.x;
      const short valid_rows = src_tile_dim.y;
      if (!quantized_mx_row_in_bounds(bi, valid_rows) ||
          quantized_mx_valid_values<pack_factor>(bj, 0, valid_k) == 0) {
        for (int i = 0; i < n_reads * pack_factor; i++) {
          dst[i] = T(0);
        }
        return;
      }

      T scale = dequantize_scale_mx<T, group_size>(*scales);
      for (int i = 0; i < n_reads; i++) {
        const short valid_values =
            quantized_mx_valid_values<pack_factor>(bj, short(i), valid_k);
        if (valid_values == 0) {
          for (short j = 0; j < pack_factor; j++) {
            dst[i * pack_factor + j] = T(0);
          }
          continue;
        }

        dequantize_mx<T, bits>(
            src[i * bytes_per_pack], scale, dst + i * pack_factor);
        for (short j = valid_values; j < pack_factor; j++) {
          dst[i * pack_factor + j] = T(0);
        }
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }
    T scale = dequantize_scale_mx<T, group_size>(*scales);
    for (int i = 0; i < n_reads; i++) {
      dequantize_mx<T, bits>(src[i * bytes_per_pack], scale, dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
        }
      } else {
        scales += scale_step;
      }
    } else {
      scales += group_stride;
    }
  }
};

// QuantizedBlockLoader with the NVFP4 MoE per-expert global reciprocal folded
// into the group scale (see nvfp4_gather_qmm_rhs). Structurally identical to
// the loader above; only the two scale decodes differ.
//
// This is a separate struct, not a template parameter on QuantizedBlockLoader,
// because MSL supports neither inheritance nor a conditionally-present member:
// an unconditional `float` member repacks the shared struct and grows the
// shipping mxfp4_qmm_t / mxfp8_qmm_t loader stack frame from 64 to 72 bytes
// (verified by diffing the emitted AIR). The MX entries must stay untouched, so
// the fold lives here. Keep the two loaders in sync.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct Nvfp4GlobalQuantizedBlockLoader {
  MLX_MTL_CONST short pack_factor = get_pack_factor<8, bits>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size < BCOLS ? 1 : group_size / BCOLS;
  MLX_MTL_CONST short scale_step = group_size < BCOLS ? BCOLS / group_size : 1;

  static_assert(
      (n_reads * pack_factor) <= group_size,
      "The number of reads per thread must be less than the group size.");

  const int src_ld;
  const int64_t tile_stride;
  short group_step_cnt;
  const int64_t group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device uint8_t* scales;
  // 1/g_ct for this tile's expert, or 1.0f when the call carries no global
  // scale -- in which case every store below is bit-identical to the loader
  // above (an e4m3 scale times an f4 value is exact in bf16, so the f32 and T
  // multiplies round to the same product).
  const float global_scale_recip;

  // The fold must happen in f32 and round into the tile exactly once; folding
  // into a T group scale first would round twice.
  float group_scale() const {
    return dequantize_scale_mx<float, group_size>(*scales) * global_scale_recip;
  }

  Nvfp4GlobalQuantizedBlockLoader(
      const device uint8_t* src_,
      const device uint8_t* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id,
      ushort simd_lane_id,
      float global_scale_recip_)
      : src_ld(src_ld_),
        tile_stride(reduction_dim
                        ? static_cast<int64_t>(BCOLS_PACKED) * bytes_per_pack
                        : static_cast<int64_t>(BROWS) * src_ld *
                            bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(
            static_cast<int64_t>(BROWS) * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + static_cast<int64_t>(bi) * src_ld * bytes_per_pack /
                pack_factor +
            static_cast<int64_t>(bj) * bytes_per_pack),
        scales(
            scales_ + static_cast<int64_t>(bi) * src_ld / group_size +
            (bj * pack_factor) / group_size),
        global_scale_recip(global_scale_recip_) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }
    float scale = group_scale();
    for (int i = 0; i < n_reads; i++) {
      dequantize_mx_global<T, bits>(
          src[i * bytes_per_pack], scale, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1) {
      const short valid_k = src_tile_dim.x;
      const short valid_rows = src_tile_dim.y;
      if (!quantized_mx_row_in_bounds(bi, valid_rows) ||
          quantized_mx_valid_values<pack_factor>(bj, 0, valid_k) == 0) {
        for (int i = 0; i < n_reads * pack_factor; i++) {
          dst[i] = T(0);
        }
        return;
      }

      float scale = group_scale();
      for (int i = 0; i < n_reads; i++) {
        const short valid_values =
            quantized_mx_valid_values<pack_factor>(bj, short(i), valid_k);
        if (valid_values == 0) {
          for (short j = 0; j < pack_factor; j++) {
            dst[i * pack_factor + j] = T(0);
          }
          continue;
        }

        dequantize_mx_global<T, bits>(
            src[i * bytes_per_pack], scale, dst + i * pack_factor);
        for (short j = valid_values; j < pack_factor; j++) {
          dst[i * pack_factor + j] = T(0);
        }
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }
    float scale = group_scale();
    for (int i = 0; i < n_reads; i++) {
      dequantize_mx_global<T, bits>(
          src[i * bytes_per_pack], scale, dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
        }
      } else {
        scales += scale_step;
      }
    } else {
      scales += group_stride;
    }
  }
};

// mxfp_qmm_t_impl — adapted MLX fp_qmm_t_impl (by-value dimensions,
// rectangular tiles, and a predicated final K tile).
template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 16,
    const int BK = 32,
    const int BN = 64>
METAL_FUNC void mxfp_qmm_t_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int static_M,
    const int active_M,
    const int K_eff,
    uint3 tid,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int threads_per_threadgroup = WM * WN * SIMD_SIZE;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = XlaQuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int64_t K_w =
      static_cast<int64_t>(K) * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  const short static_num_els =
      static_cast<short>(clamp(static_M - y_row, 0, BM));
  const short num_els = static_cast<short>(
      clamp(active_M - y_row, 0, static_cast<int>(static_num_els)));
  const short num_outs =
      static_cast<short>(clamp(N - y_col, 0, BN));

  // The execution grid covers static_M. Explicitly clear the inactive suffix
  // in every partial/full inactive tile, while keeping active output rows
  // disjoint from these stores. Fully inactive tiles return before reading x.
  device T* y_tile = y + y_row * static_cast<int64_t>(N) + y_col;
  const int inactive_count =
      (static_cast<int>(static_num_els) - static_cast<int>(num_els)) *
      static_cast<int>(num_outs);
  for (int elem = static_cast<int>(lid); elem < inactive_count;
       elem += threads_per_threadgroup) {
    const int row = static_cast<int>(num_els) + elem / num_outs;
    const int col = elem % num_outs;
    y_tile[row * static_cast<int64_t>(N) + col] = static_cast<T>(0.0f);
  }
  if (num_els == 0 || num_outs == 0) {
    return;
  }

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += static_cast<int64_t>(y_col) * K_w;
  scales += static_cast<int64_t>(y_col) * K_g;
  // 1-D per-row group-scale base
  y = y_tile;

  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  const int K_full = (K_eff / BK) * BK;
  const short K_rem = K_eff - K_full;

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_full; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_full; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_full; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_full; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  if (K_rem > 0) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(short2(K_rem, num_els));
    loader_w.load_safe(short2(K_rem, num_outs));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Concrete kernel entry points
///////////////////////////////////////////////////////////////////////////////

// Dense tiled q-GEMM (prefill). Shares the {x, w, scales, y, dims} ABI. w is
// uint32-packed (cast to bytes), scales uint8 [N, K/GS]. Tiles BM=16, BK=32,
// BN=64 (WM=WN=2 => 128 threads). ALIGNED_N selects full-tile N unsafe loads
// when N%BN==0; grid = (ceil(N/64), ceil(M/16), 1), tg=(32,2,2).
//
// The MXFP_ names are historical: this macro and mxfp_qmm_t_impl were written
// for the OCP-microscaling mxfp8/mxfp4 entries, which are gone (no producer).
// NVFP4 is the only scheme instantiating them now; the names are kept because
// renaming would churn the compiler input for the 26B golden and buy nothing.
#define MXFP_QMM_T_ENTRY_ALIGN(NAME, GS, BITS, ALIGNED_N)                    \
  kernel void NAME(                                                          \
      device const bfloat* x [[buffer(0)]],                                 \
      device const uint32_t* w [[buffer(1)]],                               \
      device const uchar* scales [[buffer(2)]],                             \
      device bfloat* y [[buffer(3)]],                                       \
      constant int4& dims [[buffer(4)]],                                    \
      uint3 tid [[threadgroup_position_in_grid]],                           \
      uint lid [[thread_index_in_threadgroup]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                           \
      uint sl [[thread_index_in_simdgroup]]) {                              \
    constexpr int BM = 16, BK = 32, BN = 64;                                \
    constexpr int BK_padded = (BK + 16 / sizeof(bfloat));                   \
    threadgroup bfloat Xs[BM * BK_padded];                                  \
    threadgroup bfloat Ws[BN * BK_padded];                                  \
    mxfp_qmm_t_impl<bfloat, GS, BITS, ALIGNED_N, BM, BK, BN>(               \
        w, scales, x, y, Xs, Ws, dims.y, dims.z, dims.x, dims.x, dims.y,    \
        tid, lid, sg, sl);                                                  \
  }

// NVFP4 dense q-GEMM: group_size 16, bits 4. The grid spans M, so every tile row
// is a real row (static_M == active_M == dims.x).
MXFP_QMM_T_ENTRY_ALIGN(nvfp4_qmm_t, 16, 4, false)
MXFP_QMM_T_ENTRY_ALIGN(nvfp4_qmm_t_alN, 16, 4, true)

// NVFP4 split-K dense q-GEMM (MLX fp_qmm_t_splitk intent).
// Uses the SAME BM=16,BK=32,BN=64 tile as nvfp4_qmm_t (proven multi-M path);
// tid.z is the K-partition. Intermediate y is bf16 [split_k, M, N].
// dims    = {M, K, N, unused}
// control = {M, split_k, k_partition_size, M*N}
// Both payloads are int4/uint4 for a stable 16-byte constant/device ABI.
// aligned_N true when N%64==0 (BN=64). Host uses non-split when split_k<=1.
#define NVFP4_QMM_T_SPLITK_ENTRY(NAME, ALIGNED_N)                              \
  kernel void NAME(                                                            \
      device const bfloat* x [[buffer(0)]],                                    \
      device const uint32_t* w [[buffer(1)]],                                  \
      device const uchar* scales [[buffer(2)]],                                \
      device bfloat* y [[buffer(3)]],                                          \
      constant int4& dims [[buffer(4)]],                                       \
      constant uint4& control [[buffer(5)]],                                   \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint lid [[thread_index_in_threadgroup]],                                \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint sl [[thread_index_in_simdgroup]]) {                                 \
    constexpr int BM = 16, BK = 32, BN = 64;                                   \
    constexpr int BK_padded = (BK + 16 / sizeof(bfloat));                      \
    threadgroup bfloat Xs[BM * BK_padded];                                     \
    threadgroup bfloat Ws[BN * BK_padded];                                     \
    const int M = dims.x;                                                      \
    const int K = dims.y;                                                      \
    const int N = dims.z;                                                      \
    const int effective_rows = static_cast<int>(                               \
        min(control.x, static_cast<uint>(max(M, 0))));                         \
    const int split_k = static_cast<int>(control.y);                           \
    const int k_partition_size = static_cast<int>(control.z);                  \
    const int split_k_partition_stride = static_cast<int>(control.w);          \
    if (effective_rows <= 0 || split_k <= 1 || k_partition_size <= 0 ||        \
        split_k_partition_stride != effective_rows * N ||                     \
        static_cast<int>(tid.z) >= split_k) {                                  \
      return;                                                                  \
    }                                                                          \
    constexpr int pack_factor = get_pack_factor<8, 4>();                         \
    constexpr int bytes_per_pack = get_bytes_per_pack();                       \
    const int64_t k_start =                                                    \
        static_cast<int64_t>(tid.z) * k_partition_size;                       \
    /* Row-major x[M,K]: advance K; w packed [N,K/2]; scales [N,K/16]. */     \
    const device bfloat* xp = x + k_start;                                     \
    auto wl = (const device uint8_t*)w;                                        \
    wl += k_start * bytes_per_pack / pack_factor;                               \
    const device uchar* sp = scales + k_start / 16;                            \
    device bfloat* yp =                                                        \
        y + tid.z * static_cast<int64_t>(split_k_partition_stride);            \
    /* Inactive staging rows are scratch and the sum kernel never reads them. */ \
    mxfp_qmm_t_impl<bfloat, 16, 4, ALIGNED_N, BM, BK, BN>(                     \
        (const device uint32_t*)wl, sp, xp, yp, Xs, Ws, K, N, effective_rows,  \
        effective_rows, k_partition_size, tid, lid, sg, sl);                   \
  }

NVFP4_QMM_T_SPLITK_ENTRY(nvfp4_qmm_t_splitk, false)
NVFP4_QMM_T_SPLITK_ENTRY(nvfp4_qmm_t_splitk_alN, true)

// Sum intermediate[split_k, MN] bf16 along axis 0 into out[MN] bf16 (MLX
// ContiguousStridedReduce sum after qmm_t_splitk). One element per thread.
// control = {M, split_k, k_partition_size, M*N}.
kernel void nvfp4_splitk_sum(
    device const bfloat* intermediate [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant uint4& control [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int sk = static_cast<int>(control.y);
  const int MN = static_cast<int>(control.w);
  const int idx = int(gid.x);
  if (idx >= MN) return;
  float acc = 0.0f;
  for (int z = 0; z < sk; ++z) {
    acc += float(intermediate[(long)z * MN + idx]);
  }
  out[idx] = bfloat(acc);
}

// Dense tiled q-GEMM: out[m,n] = sum_k x[m,k] * dequant(w[n,k]).
//   x:     bfloat [M, K]          row-major
//   w:     uchar  [N, K]          f8e4m3fn, row-major
//   scale: bfloat [N/128, K/128]  one bf16 per 128x128 weight block
//   y:     bfloat [M, N]          row-major
//   dims = {M, K, N, K/128}
// Tiles BM=16, BK=32, BN=64 (WM=WN=2 => 128 threads). aligned_N=true (N % 128
// == 0 => N % BN == 0).
kernel void fp8_qmm_t(
    device const bfloat* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const bfloat* scale [[buffer(2)]],
    device bfloat* y [[buffer(3)]],
    constant int4& dims [[buffer(4)]],  // {M, K, N, K/128}
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]]) {
  constexpr int BM = 16;
  constexpr int BK = 32;
  constexpr int BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(bfloat));

  threadgroup bfloat Xs[BM * BK_padded];
  threadgroup bfloat Ws[BN * BK_padded];

  const int M = dims.x;
  const int K = dims.y;
  const int N = dims.z;

  fp_qmm_t_impl<bfloat, 128, 8, true, BM, BK, BN>(
      w, scale, x, y, Xs, Ws, K, N, M, K, tid, lid, sg, sl);
}

// Prefill-tuned variant of fp8_qmm_t: BM=64 (more rows per threadgroup) for
// large-M (prefill) matmuls -- the best BM in {16,32,64,128} for M~256 (A/B'd:
// ~1.3x over BM=16 and over the dequant floor). Same 128x128-block f8 dequant,
// BN=64, WM=WN=2 (128 threads); only the M-tile grows. Grid BlockDim over
// ceil(M/BM). Small-M batched decode stays on the BM=16 fp8_qmm_t.
kernel void fp8_qmm_t_bm64(
    device const bfloat* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const bfloat* scale [[buffer(2)]],
    device bfloat* y [[buffer(3)]],
    constant int4& dims [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]]) {
  constexpr int BM = 64;
  constexpr int BK = 32;
  constexpr int BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(bfloat));
  threadgroup bfloat Xs[BM * BK_padded];
  threadgroup bfloat Ws[BN * BK_padded];
  const int M = dims.x;
  const int K = dims.y;
  const int N = dims.z;
  fp_qmm_t_impl<bfloat, 128, 8, true, BM, BK, BN>(
      w, scale, x, y, Xs, Ws, K, N, M, K, tid, lid, sg, sl);
}

// PerChannelFp8BlockLoader — per-OUTPUT-CHANNEL (compressed-tensors) analogue of
// Fp8BlockLoader. The scale is bf16 [N, 1] (one per output row n), CONSTANT across
// K -- it factors out of the K reduction. So each thread's scale is scales[bi] (its
// output column within the [BN, BK] weight tile), read once and applied to every
// decoded byte, and next() never advances it. The impl bases `scales` at the tile's
// N column (y_col); this loader adds the thread's row bi. reduction_dim is always 1.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct PerChannelFp8BlockLoader {
  MLX_MTL_CONST short pack_factor = 1;
  MLX_MTL_CONST short bytes_per_pack = 1;
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;

  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uchar* src;
  const device T* scales;  // one bf16 per output row n; points at scales[bi]

  PerChannelFp8BlockLoader(
      const device uchar* src_,
      const device T* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi) {}  // per-row scale (constant across K)

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }
    T scale = *scales;
    for (int i = 0; i < n_reads; i++) {
      dst[i] = static_cast<T>(decode_e4m3fn(src[i * bytes_per_pack])) * scale;
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }
    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }
    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }
    T scale = *scales;
    for (int i = 0; i < n_reads; i++) {
      dst[i] = static_cast<T>(decode_e4m3fn(src[i * bytes_per_pack])) * scale;
    }
  }

  // Per-channel scale is constant across K: only the weight source advances.
  void next() { src += tile_stride; }
};

// fp_qmm_t_pc_impl — per-channel twin of fp_qmm_t_impl: same tiled q-GEMM but with
// the PerChannelFp8BlockLoader (bf16 [N,1] scale, applied per output column) and
// the scale base at y_col (not the 2-D block index).
template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 16,
    const int BK = 32,
    const int BN = 64>
METAL_FUNC void fp_qmm_t_pc_impl(
    const device uchar* w,
    const device T* scales,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    const int K_eff,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");
  (void)lid;
  (void)group_size;
  (void)bits;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t =
      PerChannelFp8BlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uchar*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * static_cast<int64_t>(K);
  scales += y_col;  // per-channel scale base: the tile's N column
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

// Per-channel dense q-GEMM entries (bf16[M,K] . f8e4m3fn[N,K] with bf16 [N,1]
// scale -> bf16[M,N]). Small-M (batched decode) BM=16, large-M (prefill) BM=64.
kernel void fp8_qmm_t_pc(
    device const bfloat* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const bfloat* scale [[buffer(2)]],
    device bfloat* y [[buffer(3)]],
    constant int4& dims [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]]) {
  constexpr int BM = 16;
  constexpr int BK = 32;
  constexpr int BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(bfloat));
  threadgroup bfloat Xs[BM * BK_padded];
  threadgroup bfloat Ws[BN * BK_padded];
  const int M = dims.x;
  const int K = dims.y;
  const int N = dims.z;
  fp_qmm_t_pc_impl<bfloat, 128, 8, false, BM, BK, BN>(
      w, scale, x, y, Xs, Ws, K, N, M, K, tid, lid, sg, sl);
}

kernel void fp8_qmm_t_pc_bm64(
    device const bfloat* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const bfloat* scale [[buffer(2)]],
    device bfloat* y [[buffer(3)]],
    constant int4& dims [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]]) {
  constexpr int BM = 64;
  constexpr int BK = 32;
  constexpr int BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(bfloat));
  threadgroup bfloat Xs[BM * BK_padded];
  threadgroup bfloat Ws[BN * BK_padded];
  const int M = dims.x;
  const int K = dims.y;
  const int N = dims.z;
  fp_qmm_t_pc_impl<bfloat, 128, 8, false, BM, BK, BN>(
      w, scale, x, y, Xs, Ws, K, N, M, K, tid, lid, sg, sl);
}

// MoE gather variant (transpose=true): each output row r selects expert
// indices[r]; out[r,n] = sum_k x[r,k] * dequant(w[indices[r], n, k]).
//   x:       bfloat [R, K]              row-major
//   w:       uchar  [E, N, K]           f8e4m3fn, flat row-major
//   scale:   bfloat [E, N/128, K/128]   bf16, flat (one per 128x128 block)
//   indices: uint32 [R]                 expert index per output row
//   y:       bfloat [R, N]              row-major
//   mnk = {R, N, K}
// Tiles BM=16, BN=32, BK=32, WM=1, WN=2 (=> 64 threads).
kernel void fp8_gather_qmm_rhs(
    device const bfloat* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const bfloat* scale [[buffer(2)]],
    device const uint* indices [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant int4& mnk [[buffer(5)]],  // {R, N, K, _}
    uint3 tid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]]) {
  constexpr int BM = 16;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int WM = 1;
  constexpr int WN = 2;
  constexpr bool transpose = true;
  constexpr int BK_padded = (BK + 16 / sizeof(bfloat));
  constexpr int BN_padded = (BN + 16 / sizeof(bfloat));

  threadgroup bfloat Xs[BM * BK_padded];
  threadgroup bfloat Ws[transpose ? BN * BK_padded : BK * BN_padded];

  const int R = max(mnk.x, 0);
  const int N = mnk.y;
  const int K = mnk.z;

  fp_gather_qmm_rhs_impl<bfloat, 128, 8, BM, BN, BK, WM, WN, transpose>(
      x, w, scale, indices, y, R, N, K, Xs, Ws, tid, sg, sl);
}

///////////////////////////////////////////////////////////////////////////////
// bf16 (un-quantized) MoE gather variant: bf16 twin of fp_gather_qmm_rhs_impl
// with no block scales. The weight loader is the plain bf16 mlx::steel
// BlockLoader (the W-tile analogue of the X loader); the sort/gather/store
// machinery is identical.
///////////////////////////////////////////////////////////////////////////////
template <typename T, int BM, int BN, int BK, int WM, int WN, bool transpose>
METAL_FUNC void bf16_gather_mm_rhs_impl(
    const device T* x,
    const device T* w,
    const device uint32_t* indices,
    device T* y,
    const int M,
    const int N,
    const int K,
    threadgroup T* Xs,
    threadgroup T* Ws,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T, T, BM, BN, BK, WM, WN, false, transpose, BK_padded,
      transpose ? BK_padded : BN_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = mlx::steel::BlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE>;

  const int K_it = K / BK;
  const size_t stride_w = transpose ? size_t(N) * K : size_t(K) * N;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  auto wl = (const device T*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K : y_col;

  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    thread mma_t mma_op(simd_group_id, simd_lane_id);
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(
        wl + index * stride_w, transpose ? K : N, Ws, simd_group_id,
        simd_lane_id);

    if (align_M && align_N) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      if (offset_next - offset == BM) {
        mma_op.store_result(y, N);
      } else {
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }
    } else {
      if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
        gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        if (offset_next - offset == BM) {
          mma_op.store_result(y, N);
        } else {
          mma_op.store_result_slice(
              y, N, short2(0, offset), short2(BN, offset_next));
        }
      } else if (align_N || tgp_bn == BN) {
        gemm_loop_unaligned<false, true, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      } else if (align_M || tgp_bm == BM) {
        gemm_loop_unaligned<true, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      } else {
        gemm_loop_unaligned<false, false, transpose>(
            Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
        }
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(tgp_bn, offset_next));
      }
    }
  }
}

// bf16 MoE gather q-GEMM: out[r,n] = sum_k x[r,k] * w[indices[r], n, k] (bf16).
//   x:       bfloat [R, K]    row-major
//   w:       bfloat [E, N, K] flat row-major
//   indices: uint32 [R]       expert index per output row (rows sorted by expert)
//   y:       bfloat [R, N]    row-major
//   mnk = {R, N, K}.  Tiles BM=16, BN=32, BK=32, WM=1, WN=2 (=> 64 threads).
kernel void bf16_gather_mm_rhs(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    device bfloat* y [[buffer(3)]],
    constant int4& mnk [[buffer(4)]],  // {R, N, K, _}
    uint3 tid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]]) {
  constexpr int BM = 16;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int WM = 1;
  constexpr int WN = 2;
  constexpr bool transpose = true;
  constexpr int BK_padded = (BK + 16 / sizeof(bfloat));
  constexpr int BN_padded = (BN + 16 / sizeof(bfloat));

  threadgroup bfloat Xs[BM * BK_padded];
  threadgroup bfloat Ws[transpose ? BN * BK_padded : BK * BN_padded];

  const int R = max(mnk.x, 0);
  const int N = mnk.y;
  const int K = mnk.z;

  bf16_gather_mm_rhs_impl<bfloat, BM, BN, BK, WM, WN, transpose>(
      x, w, indices, y, R, N, K, Xs, Ws, tid, sg, sl);
}

///////////////////////////////////////////////////////////////////////////////
// NVFP4 MoE gather q-GEMM (transpose=true): MLX fp_gather_qmm_rhs with
// QuantizedBlockLoader group_size=16 bits=4 (e4m3 scales, packed f4 weights).
//   x:       bfloat [R, K]
//   w:       uchar  [E, N, K/2]   packed f4e2m1, expert-major
//   scale:   uchar  [E, N, K/16]  e4m3 group-16
//   indices: uint32 [R]
//   y:       bfloat [R, N]
//   mnk = {R, N, K, _}
// Tiles BM=16, BN=32, BK=32, WM=1, WN=2 (64 threads) — same as fp8 gather.
///////////////////////////////////////////////////////////////////////////////
template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
METAL_FUNC void nvfp4_gather_qmm_rhs_impl(
    const device T* x,
    const device uchar* w,
    const device uchar* scales,
    const device uint32_t* indices,
    device T* y,
    const int M,
    const int N,
    const int K,
    const device float* w_global_scale,  // f32[E], or null when absent
    threadgroup T* Xs,
    threadgroup T* Ws,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T, T, BM, BN, BK, WM, WN, false, transpose, BK_padded,
      transpose ? BK_padded : BN_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  // One instantiation carries the fold; an absent operand passes a reciprocal
  // of 1.0f, which is bit-identical (an e4m3 scale times an f4 value is exact
  // in bf16, so both paths round to the same product) and reads no buffer.
  using loader_w_t = Nvfp4GlobalQuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  const size_t K_w =
      size_t(K) * size_t(bytes_per_pack) / size_t(pack_factor);  // K/2
  const size_t K_g = size_t(K) / size_t(group_size);
  const size_t N_w =
      size_t(N) * size_t(bytes_per_pack) / size_t(pack_factor);
  const size_t N_g = size_t(N) / size_t(group_size);
  const int K_it = K / BK;
  const size_t stride_w = transpose ? size_t(N) * K_w : size_t(K) * N_w;
  const size_t stride_s = transpose ? size_t(N) * K_g : size_t(K) * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  auto wl = (const device uchar*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose
      ? y_col_long * K_w
      : y_col_long * size_t(bytes_per_pack) / size_t(pack_factor);
  scales += transpose ? y_col_long * K_g
                      : y_col_long / size_t(group_size);

  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    thread mma_t mma_op(simd_group_id, simd_lane_id);
    thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
    // Same `index` the weight/scale offsets above already trust; the fold is
    // one indexed f32 load per expert run.
    thread loader_w_t loader_w(
        wl + index * stride_w, scales + index * stride_s,
        transpose ? K : N, Ws, simd_group_id, simd_lane_id,
        w_global_scale ? 1.0f / w_global_scale[index] : 1.0f);

    if (align_M && align_N) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      if (offset_next - offset == BM) {
        mma_op.store_result(y, N);
      } else {
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }
    } else if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      if (offset_next - offset == BM) {
        mma_op.store_result(y, N);
      } else {
        mma_op.store_result_slice(
            y, N, short2(0, offset), short2(BN, offset_next));
      }
    } else if (align_N || tgp_bn == BN) {
      gemm_loop_unaligned<false, true, transpose>(
          Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result_slice(
          y, N, short2(0, offset), short2(BN, offset_next));
    } else if (align_M || tgp_bm == BM) {
      gemm_loop_unaligned<true, false, transpose>(
          Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result_slice(
          y, N, short2(0, offset), short2(tgp_bn, offset_next));
    } else {
      gemm_loop_unaligned<false, false, transpose>(
          Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result_slice(
          y, N, short2(0, offset), short2(tgp_bn, offset_next));
    }
  }
}

kernel void nvfp4_gather_qmm_rhs(
    device const bfloat* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const uchar* scale [[buffer(2)]],
    device const uint* indices [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant int4& mnk [[buffer(5)]],  // {R, N, K, _}
    device const float* w_global_scale
        [[buffer(6), function_constant(moe_has_global_scale)]],  // f32[E]
    uint3 tid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]]) {
  constexpr int BM = 16;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int WM = 1;
  constexpr int WN = 2;
  constexpr bool transpose = true;
  constexpr int BK_padded = (BK + 16 / sizeof(bfloat));
  constexpr int BN_padded = (BN + 16 / sizeof(bfloat));

  threadgroup bfloat Xs[BM * BK_padded];
  threadgroup bfloat Ws[transpose ? BN * BK_padded : BK * BN_padded];

  const int R = max(mnk.x, 0);
  const int N = mnk.y;
  const int K = mnk.z;

  // A conditionally-declared argument may only be read under its own function
  // constant; hoisting it to a null pointer keeps the fold in one instantiation.
  const device float* gs = nullptr;
  if (moe_has_global_scale) {
    gs = w_global_scale;
  }

  nvfp4_gather_qmm_rhs_impl<bfloat, 16, 4, BM, BN, BK, WM, WN, transpose>(
      x, w, scale, indices, y, R, N, K, gs, Xs, Ws, tid, sg, sl);
}
