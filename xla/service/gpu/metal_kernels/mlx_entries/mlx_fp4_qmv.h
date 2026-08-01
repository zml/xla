// NVFP4 fused qmv (MIT License, Copyright (c) 2023 Apple Inc.,
// https://github.com/ml-explore/mlx). Entry source for the mlx_fp4_qmv bundle:
// the NVFP4 (group_size=16, e4m3 scales) decode GEMV and MoE gather GEMV that
// gemma-4-26B-A4B-NVFP4 runs on -- MetalNvfp4MatmulThunk and MetalMoeGemvThunk.
//
// out = sum_k x * f4(w) * e4m3_scale. For compressed-tensors weights, the caller
// pre-scales x by 1/g_ct, the saved weight-global encode divisor. MLX's
// global_scale_w instead means amax and equals 2688/g_ct.
//
// ABI: w uchar[N,K/2], scales e4m3[N,K/16], x bf16[M,K], y bf16[M,N],
// dims int4{M,K,N,_}. grid (M, ceil(N/8), 1) x 64 threads.
//
// The upstream text is NOT checked in. The includes below resolve against the
// @mlx archive pinned in //third_party/mlx:workspace.bzl, which
// MetalIncludeRoot() hands to the Metal compiler as -I. To change upstream's
// bytes, add a patch to //third_party/mlx:series.bzl, where a bump re-verifies
// it at fetch time.
//
// The includes are upstream's own prologue, copied from
// mlx/backend/metal/kernels/fp_quantized.metal:4-7. They supply every piece of
// this bundle that was genuinely upstream's -- fp4_e2m1, fp8_e4m3, fp8_e8m0,
// get_pack_factor, get_bytes_per_pack, dequantize_scale, Dequantize,
// load_vector{,_safe} and qdot{,_safe} -- which the flattened bundle carried as
// copies. Note what that deletes: the flattened fp8_e8m0 here had been
// hand-rewritten to exp2(bits - 127.0f) where upstream (and our OWN other copy,
// in the mxfp bundle) bitcasts. Numerically equal, a transcendental instead of a
// bitcast, dead (dequantize_scale<T,16> selects fp8_e4m3), and silently diverged
// from its twin with nothing in the tree able to compare them. That is the exact
// failure this migration exists to make impossible, and the include ends it.
//
// UNLIKE the other bundles in this family, this one is only ~11% upstream. The
// four impls below are OURS -- they are forks of upstream's fp_qmv_impl /
// fp_qmv_fast_impl / fp_qmv_wide_impl and of the shape of fp_gather_qmv, and
// every one of them diverges in the body, not the signature. They are therefore
// RENAME-FORKS, xla_-prefixed: upstream's originals are now in scope via the
// include and would otherwise silently overload with ours (fp_qmv_fast_impl in
// particular differs only in a parameter type, so the "right" overload would be
// picked by an accident of address-space binding). Each fork is marked XLA DELTA
// at its definition with what diverges and why, and //third_party/mlx:series.bzl
// records why none of them is carried as a patch.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"
// clang-format on

// ---- K-tail alignment -------------------------------------------------------
// The impls below have NO partial-K tail, and this is why. `remaining` is
// clamp(K - k - simd_lid*values_per_thread, 0, vpt). When K % values_per_thread
// == 0 every term is a multiple of values_per_thread, so remaining is exactly
// values_per_thread or exactly 0 -- never partial. values_per_thread is 8
// (get_pack_factor<32,4>), and NVFP4's group_size of 16 means a well-formed
// weight has K%16==0, hence K%8==0. The tail is unreachable BY CONSTRUCTION.
//
// It is not merely unreachable, it is enforced at the emitter, which is what
// makes deleting the tail safe rather than optimistic: thunk_emitter.cc:1473
// (dense zml$scaled_matmul) and :1673 (MoE __metal$moe_gemm$f4) both return
// UnimplementedError for k % 16 != 0 BEFORE the thunk is built, so a kernel with
// a partial K never gets launched -- it never gets compiled.
//
// Carrying the tail anyway was expensive, because reaching it at all is the
// cost: the safe bodies index the `thread U x_thread[]` array with a runtime
// bound, which defeats SROA, so the array is demoted out of registers to scratch
// for the WHOLE function -- the hot unrolled loop included. Worth 90.8 -> 97.3
// tok/s here. These impls previously carried it behind a `k_aligned` template
// axis with both instantiations compiled and selected by a runtime K%8 test;
// the un-aligned half could not run, so the axis, its safe bodies and the
// dispatch are all gone and the aligned body is simply the body.
//
// The tail defect is upstream's, not an artifact of our adaptation -- it is live
// in upstream's own fp_qmv_impl. TODO: file it upstream. It is NOT carried as a
// series.bzl patch: see that file's note -- our forks below supersede upstream's
// bodies, so patching them would change no code we compile.
// ---- xla_fp_qmv_impl --------------------------------------------------------
// XLA DELTA: a RENAME-FORK of upstream's fp_qmv_impl (fp_quantized.h:386).
// Upstream's is in scope via the include and this one diverges in the BODY, in
// three independent ways -- so it is renamed rather than patched, because a
// rename cannot collide on a bump and cannot silently un-apply:
//
//  1. No partial-K tail. Upstream's body carries a runtime-bounded safe tail
//     that costs the whole function its registers; ours omits it, because the
//     emitter guarantees K%16==0. See the K-tail note above.
//  2. The partial-tile guard. Upstream moves a partial final four-column SIMD
//     tile BACKWARDS (used_out_row = min(out_vec_size - results_per_simdgroup,
//     out_row)) and guards on a WHOLE-MATRIX property (out_vec_size < 8); ours
//     predicates the genuinely partial tile on a PER-TILE property (out_row +
//     results_per_simdgroup > out_vec_size). At N%8!=0 upstream's move-back makes
//     two threadgroups write some of the same bf16 elements; ours gives every
//     output exactly one writer.
//  3. int64_t address arithmetic. Upstream forms `out_row * in_vec_size_w` in
//     int32, which overflows at large N*K.
//
// TODO: (2) and (3) are both worth filing upstream, and both are candidates for
// deletion here -- but NEITHER is proven droppable and neither should be dropped
// on the arithmetic alone. For (2), the obvious experiment ("at N%8==0 the
// move-back is inert, so the paths agree") tests precisely the regime where the
// fork does nothing: the real question is whether upstream's whole-matrix guard
// is sufficient for every N we dispatch, and upstream is actively patching
// inside the branch we deleted. For (3) our shipping shapes provably do not
// overflow (vocab N ~150k x K/2 ~2.5k ~= 375M < 2^31), but dropping it still
// moves the emitted AIR, so it needs the golden bench, not an argument.
// Also: dimensions are passed by value -- see the dims note on the entries.
template <typename T, int group_size, int bits>
void xla_fp_qmv_impl(
    const device uint32_t* w,
    const device uint8_t* scales,  // e4m3 group-16; x was divided by CT g_ct
    const device T* x,
    device bfloat* y,
    int in_vec_size,
    int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int64_t in_vec_size_w =
      static_cast<int64_t>(in_vec_size) * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  // MLX moves a partial final four-column SIMD tile backwards so every lane
  // computes four outputs.  With N%8!=0 that makes the previous threadgroup
  // and the final threadgroup write some of the same BF16 elements.  Keep the
  // full-tile path for disjoint four-column ranges, and predicate only the
  // genuinely partial SIMD tile so every public output has one writer.
  if (out_row + results_per_simdgroup > out_vec_size) {
    ws += static_cast<int64_t>(out_row) * in_vec_size_w +
        static_cast<int64_t>(simd_lid) * packs_per_thread * bytes_per_pack;
    scales += static_cast<int64_t>(out_row) * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    x += static_cast<int64_t>(tid.x) * in_vec_size +
        simd_lid * values_per_thread;
    y += static_cast<int64_t>(tid.x) * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      load_vector<T, U, values_per_thread>(x, x_thread);
      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl =
            scales + static_cast<int64_t>(row) * in_vec_size_g;
        U s = dequantize_scale<U, group_size>(sl[0]);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }
      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining == values_per_thread) {
      load_vector<T, U, values_per_thread>(x, x_thread);
      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl =
            scales + static_cast<int64_t>(row) * in_vec_size_g;
        U s = dequantize_scale<U, group_size>(sl[0]);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }
    }
    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<bfloat>(result[row]);
      }
    }
  } else {
    ws += static_cast<int64_t>(out_row) * in_vec_size_w +
        static_cast<int64_t>(simd_lid) * packs_per_thread * bytes_per_pack;
    scales += static_cast<int64_t>(out_row) * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    x += static_cast<int64_t>(tid.x) * in_vec_size +
        simd_lid * values_per_thread;
    y += static_cast<int64_t>(tid.x) * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      load_vector<T, U, values_per_thread>(x, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl =
            scales + static_cast<int64_t>(row) * in_vec_size_g;
        U s = dequantize_scale<U, group_size>(sl[0]);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }
      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining == values_per_thread) {
      load_vector<T, U, values_per_thread>(x, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl =
            scales + static_cast<int64_t>(row) * in_vec_size_g;
        U s = dequantize_scale<U, group_size>(sl[0]);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<bfloat>(result[row]);
      }
    }
  }
}

// ---- xla_fp_qmv_fast_impl ---------------------------------------------------
// MLX's aligned qmv body: two packed f4 words per lane and no K/N tail guards.
// The host selects this entry only for N%8==0 && K%512==0.
//
// XLA DELTA: a RENAME-FORK of upstream's fp_qmv_fast_impl (fp_quantized.h:325).
// The body is upstream's; what diverges is int64_t address arithmetic (as in
// xla_fp_qmv_impl above), by-value dimensions, and a bfloat-typed y.
//
// The rename is load-bearing here rather than cosmetic: upstream's overload has
// the SAME template parameter list and differs only in a parameter TYPE
// (`const constant int&` vs our `int`). Left both named fp_qmv_fast_impl, the
// call below would still compile and still pick ours -- but only because an
// rvalue cannot bind to a constant-space reference, which makes upstream's
// non-viable. That is an accident of address-space binding deciding which body
// runs. A rename makes it a decision.
template <typename T, int group_size, int bits>
void xla_fp_qmv_fast_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device bfloat* y,
    int in_vec_size,
    int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  const int64_t in_vec_size_w =
      static_cast<int64_t>(in_vec_size) * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint8_t* ws = reinterpret_cast<const device uint8_t*>(w);

  ws += static_cast<int64_t>(out_row) * in_vec_size_w +
      static_cast<int64_t>(simd_lid) * packs_per_thread * bytes_per_pack;
  scales += static_cast<int64_t>(out_row) * in_vec_size_g +
      simd_lid / scale_step_per_thread;
  x += static_cast<int64_t>(tid.x) * in_vec_size +
      simd_lid * values_per_thread;
  y += static_cast<int64_t>(tid.x) * out_vec_size + out_row;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x, x_thread);

#pragma unroll
    for (int row = 0; row < results_per_simdgroup; ++row) {
      const device uint8_t* wl =
          ws + static_cast<int64_t>(row) * in_vec_size_w;
      const device uint8_t* sl =
          scales + static_cast<int64_t>(row) * in_vec_size_g;
      U s = dequantize_scale<U, group_size>(sl[0]);
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    x += block_size;
  }

#pragma unroll
  for (int row = 0; row < results_per_simdgroup; ++row) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<bfloat>(result[row]);
    }
  }
}

// ---- clean NVFP4 GEMV entry -------------------------------------------------
kernel void nvfp4_qmv(
    device const uchar* w [[buffer(0)]],       // packed f4 [N, K/2]
    device const uchar* scales [[buffer(1)]],  // e4m3 [N, K/16] block scale
    device const bfloat* x [[buffer(2)]],      // [M, K]
    device bfloat* y [[buffer(3)]],            // [M, N]
    constant int4& dims [[buffer(4)]],         // {M, K, N, _}
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // values_per_thread is 8 here (pack_factor 8 * packs_per_thread 1) and NVFP4's
  // group_size of 16 forces K%16==0, so K%8==0 holds unconditionally and the
  // partial-K tail cannot occur. See the K-tail note at the top of this file.
  xla_fp_qmv_impl<bfloat, 16, 4>(
      (const device uint32_t*)w, scales, x, y, dims.y, dims.z, tid, simd_gid,
      simd_lid);
}

// Aligned MLX qmv entry.  The host guarantees N%8==0 and K%512==0.
kernel void nvfp4_qmv_fast(
    device const uchar* w [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const bfloat* x [[buffer(2)]],
    device bfloat* y [[buffer(3)]],
    constant int4& dims [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  xla_fp_qmv_fast_impl<bfloat, 16, 4>(
      reinterpret_cast<const device uint32_t*>(w), scales, x, y, dims.y, dims.z,
      tid, simd_gid, simd_lid);
}

// =============================================================================
// MLX fp_qmv_wide: dequant each weight group once, reuse across vecs_per_tg
// input rows. NVFP4: group_size=16, bits=4, k_lanes=16 (fp mode).
// Launch (MLX): group (32, 2, 1), grid (ceil(M/vecs), ceil(N/4), 1).
// Host picks vecs_per_tg via n_tiles = ceil(M/5), vecs = ceil(M/n_tiles).
//
// XLA DELTA: a RENAME-FORK of upstream's fp_qmv_wide_impl (fp_quantized.h:533),
// which the include puts in scope. Diverges in by-value dimensions, int64_t
// address arithmetic, and an explicit M bound (upstream reaches the row count
// through its batched-shape params; ours takes it directly because our ABI has
// no such params).
// =============================================================================
template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>
METAL_FUNC void xla_fp_qmv_wide_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    int in_vec_size,
    int out_vec_size,
    int M,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int nf4 = group_size / 4;  // float4 lanes per quant group

  typedef float U;

  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;

  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +
                      results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;

  if (out_row >= out_vec_size || vec0 >= M) {
    return;
  }

  const int row = min(out_row, out_vec_size - 1);

  const int64_t in_vec_size_w =
      static_cast<int64_t>(in_vec_size) * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const device uint8_t* wrow =
      (const device uint8_t*)w + static_cast<int64_t>(row) * in_vec_size_w;
  const device uint8_t* srow =
      scales + static_cast<int64_t>(row) * in_vec_size_g;

  const device T* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) {
    // Vectors past the last row alias the final row's pointer but are never
    // dereferenced below (the accumulate and the store both bound v by M).
    xv[v] = x + static_cast<int64_t>(min(vec0 + v, M - 1)) * in_vec_size;
  }

  U result[vecs_per_tg] = {0};

  for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {
    const int k0 = g * group_size;
    U s = dequantize_scale<U, group_size>(srow[g]);
    const device uint8_t* wg =
        wrow + static_cast<int64_t>(k0) * bytes_per_pack / pack_factor;

    float4 w4[nf4];
    if constexpr (bits == 4) {
      const device uint16_t* wq = (const device uint16_t*)wg;
#pragma unroll
      for (int i = 0; i < nf4; i++) {
        w4[i] = float4(
            Dequantize<4>{}(uint8_t(wq[i])),
            Dequantize<4>{}(uint8_t(wq[i] >> 4)),
            Dequantize<4>{}(uint8_t(wq[i] >> 8)),
            Dequantize<4>{}(uint8_t(wq[i] >> 12)));
      }
    } else {
#pragma unroll
      for (int i = 0; i < nf4; i++) {
        w4[i] = float4(
            Dequantize<8>{}(wg[4 * i]),
            Dequantize<8>{}(wg[4 * i + 1]),
            Dequantize<8>{}(wg[4 * i + 2]),
            Dequantize<8>{}(wg[4 * i + 3]));
      }
    }

#pragma unroll
    for (int v = 0; v < vecs_per_tg; v++) {
      const device vec<T, 4>* xv4 = (const device vec<T, 4>*)(xv[v] + k0);
      float acc = 0;
#pragma unroll
      for (int j = 0; j < nf4; j++) {
        acc += dot(w4[j], float4(xv4[j]));
      }
      result[v] += s * acc;
    }
  }

  for (int v = 0; v < vecs_per_tg; v++) {
    if constexpr (k_lanes >= 32) {
      result[v] += simd_shuffle_down(result[v], 16);
    }
    if constexpr (k_lanes >= 16) {
      result[v] += simd_shuffle_down(result[v], 8);
    }
    if constexpr (k_lanes >= 8) {
      result[v] += simd_shuffle_down(result[v], 4);
    }
    if constexpr (k_lanes >= 4) {
      result[v] += simd_shuffle_down(result[v], 2);
    }
    if constexpr (k_lanes >= 2) {
      result[v] += simd_shuffle_down(result[v], 1);
    }
  }

  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) {
        y[static_cast<int64_t>(vec0 + v) * out_vec_size + out_row] =
            static_cast<T>(result[v]);
      }
    }
  }
}

// Named entries for vecs_per_tg = 2..5 (MLX launch formula caps tile at 5).
// ABI matches nvfp4_qmv: w, scales, x, y, dims{M,K,N,_}.
#define NVFP4_QMV_WIDE_ENTRY(V)                                                \
  kernel void nvfp4_qmv_wide_##V(                                              \
      device const uchar* w [[buffer(0)]],                                     \
      device const uchar* scales [[buffer(1)]],                                \
      device const bfloat* x [[buffer(2)]],                                    \
      device bfloat* y [[buffer(3)]],                                          \
      constant int4& dims [[buffer(4)]],                                       \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]) {                           \
    xla_fp_qmv_wide_impl<bfloat, 16, 4, V, 16>(                                    \
        (const device uint32_t*)w, scales, x, y, dims.y, dims.z,               \
        max(dims.x, 0), tid, simd_gid, simd_lid);                              \
  }

NVFP4_QMV_WIDE_ENTRY(2)
NVFP4_QMV_WIDE_ENTRY(3)
NVFP4_QMV_WIDE_ENTRY(4)
NVFP4_QMV_WIDE_ENTRY(5)
#undef NVFP4_QMV_WIDE_ENTRY

// =============================================================================
// MLX gather_qmv for MoE (nvfp4, group_size=16, bits=4)
//
// MLX's fp_gather_qmv adjusts w/scales by rhs_indices[batch] then runs
// xla_fp_qmv_impl. Our MoE ABI is the same idea with a simpler layout:
//   x[R,K], w[E,N,K/2], scale[E,N,K/16], expert_id[R]  →  out[R,N]
// tid.x = route row r, tid.y = N-tile (8 cols). 64 threads = 2 simdgroups.
// Matches MLX gather_qmv launch for M=1, B=R (one vector per expert pick).
//
// When `moe_has_global_scale` is set the call carries a trailing f32[E]
// per-expert global scale (the compressed-tensors weight-encode divisor) and
// the kernel folds its reciprocal into the f32 group scale, one indexed load
// per threadgroup. Folding into the *weight* group scale keeps the f32
// accumulator at output magnitude: `sum(x*w*s*inv_g)` never holds the
// global-inflated `sum(x*w*s)` that a divide-the-output scheme would round.
// When the constant is unset no buffer is bound and no fold is compiled in.
//
// XLA DELTA: 440 is the same logical constant as mlx_steel_qgemm.h's; both
// bundles bind it from MetalMoeGemvThunk. It sits in our 4xx range because
// upstream MLX owns 0-26, 100, 101, 110, 199, 200-202 and 300-302 across its
// kernel tree, and our custom/ shaders already number from 420. Never allocate
// ours below 400.
// =============================================================================
constant bool moe_has_global_scale [[function_constant(440)]];

// The e4m3 group scale, folded with the per-expert global reciprocal when the
// call carries one. `U` is f32 here, so the fold is a pure exponent shift and
// costs no precision; without the operand this is the unmodified MLX decode.
//
// This is ours and has no upstream counterpart: it wraps upstream's
// dequantize_scale, which the include supplies. It cannot be injected into an
// upstream body from outside -- the call would be `dequantize_scale<U,
// group_size>(sl[0])`, dependent on group_size, but its argument is uint8_t, a
// builtin with no associated namespace, so ADL finds nothing and the name binds
// at definition time to ::dequantize_scale.
//
// TODO: the seam upstream would need is a defaulted `typename ScaleDecoder =
// DefaultScaleDecoder` template parameter on fp_qmv_impl -- defaulted, so zero
// behaviour change for upstream, and a signature patch survives a reshuffle
// where a body patch does not. Worth filing. It does NOT help us today: the
// gather below is a fork of upstream's fp_qmv_impl BODY (partial-tile guard,
// int64_t, no K-tail -- see the XLA DELTA there), so there is no upstream body
// for a ScaleDecoder to be threaded into. The seam only pays off if the fork
// items above land upstream first.
template <typename U, int group_size, bool has_global_scale>
static inline U moe_group_scale(uint8_t s, float inv_g) {
  U out = dequantize_scale<U, group_size>(s);
  if constexpr (has_global_scale) {
    out *= U(inv_g);
  }
  return out;
}

// XLA DELTA: ours. Same idea as upstream's fp_gather_qmv (adjust w/scales by the
// expert index, then run the qmv body), but the body is xla_fp_qmv_impl's, not
// upstream's -- it carries the same partial-tile predication and int64_t
// hardening, plus the inv_g fold above. Our MoE ABI is also flatter than
// upstream's batched-shape one: x[R,K], w[E,N,K/2], scale[E,N,K/16],
// expert_id[R] -> out[R,N].
//
// The partial K-tail is unreachable here for the same reason as in
// xla_fp_qmv_impl -- see the K-tail note at the top of this file.
template <typename T, int group_size, int bits, bool has_global_scale>
void gather_qmv_moe_impl(
    const device uint8_t* w_base,
    const device uint8_t* scales_base,
    int expert,
    const device T* x,
    device bfloat* y,
    int R, int K, int N,
    float inv_g,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int ri = int(tid.x);
  if (ri >= R) return;

  // The entry point validates `expert` before calling us, so the MLX-style
  // matrix offset below can never be formed from an untrusted expert id.
  const int64_t expert_w_stride = static_cast<int64_t>(N) * (K / 2);
  const int64_t expert_s_stride =
      static_cast<int64_t>(N) * (K / group_size);
  const device uint8_t* ws =
      w_base + static_cast<int64_t>(expert) * expert_w_stride;
  const device uint8_t* scales =
      scales_base + static_cast<int64_t>(expert) * expert_s_stride;
  const device T* xrow = x + static_cast<int64_t>(ri) * K;
  device bfloat* yrow = y + static_cast<int64_t>(ri) * N;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int64_t in_vec_size_w =
      static_cast<int64_t>(K) * bytes_per_pack / pack_factor;
  const int in_vec_size_g = K / group_size;
  const int out_row = int(tid.y) * (num_simdgroups * results_per_simdgroup) +
                      int(simd_gid) * results_per_simdgroup;

  if (out_row >= N) return;

  // Body is xla_fp_qmv_impl with x stride = one row (already selected).
  // Do not move a partial final SIMD tile backwards: that creates duplicate
  // cross-threadgroup stores when N is not a multiple of eight.  Predicating
  // the final one-to-three columns keeps the launch tail-safe and race-free.
  if (out_row + results_per_simdgroup > N) {
    ws += static_cast<int64_t>(out_row) * in_vec_size_w +
        static_cast<int64_t>(simd_lid) * packs_per_thread * bytes_per_pack;
    scales += static_cast<int64_t>(out_row) * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device T* xp = xrow + simd_lid * values_per_thread;

    int k = 0;
    for (; k < K - block_size; k += block_size) {
      load_vector<T, U, values_per_thread>(xp, x_thread);
      for (int row = 0; row < results_per_simdgroup && out_row + row < N;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl =
            scales + static_cast<int64_t>(row) * in_vec_size_g;
        U s = moe_group_scale<U, group_size, has_global_scale>(sl[0], inv_g);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }
      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      xp += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(K - k - int(simd_lid) * values_per_thread), 0,
        values_per_thread);
    if (remaining == values_per_thread) {
      load_vector<T, U, values_per_thread>(xp, x_thread);
      for (int row = 0; row < results_per_simdgroup && out_row + row < N;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl =
            scales + static_cast<int64_t>(row) * in_vec_size_g;
        U s = moe_group_scale<U, group_size, has_global_scale>(sl[0], inv_g);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }
    }
    for (int row = 0; row < results_per_simdgroup && out_row + row < N; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) yrow[out_row + row] = bfloat(result[row]);
    }
  } else {
    ws += static_cast<int64_t>(out_row) * in_vec_size_w +
          static_cast<int64_t>(simd_lid) * packs_per_thread * bytes_per_pack;
    scales += static_cast<int64_t>(out_row) * in_vec_size_g +
        simd_lid / scale_step_per_thread;
    const device T* xp = xrow + simd_lid * values_per_thread;

    int k = 0;
    for (; k < K - block_size; k += block_size) {
      load_vector<T, U, values_per_thread>(xp, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl =
            scales + static_cast<int64_t>(row) * in_vec_size_g;
        U s = moe_group_scale<U, group_size, has_global_scale>(sl[0], inv_g);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }
      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      xp += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(K - k - int(simd_lid) * values_per_thread), 0,
        values_per_thread);
    if (remaining == values_per_thread) {
      load_vector<T, U, values_per_thread>(xp, x_thread);
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl =
            scales + static_cast<int64_t>(row) * in_vec_size_g;
        U s = moe_group_scale<U, group_size, has_global_scale>(sl[0], inv_g);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) yrow[out_row + row] = bfloat(result[row]);
    }
  }
}

// MLX name: nvfp4 gather_qmv (MoE decode / small-R path).
kernel void nvfp4_gather_qmv(
    device const bfloat* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const uchar* scale [[buffer(2)]],
    device const int* expert_id [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant int4& dims [[buffer(5)]],  // {R, K, N, E}
    device const float* w_global_scale
        [[buffer(6), function_constant(moe_has_global_scale)]],  // f32[E]
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  const int R = max(dims.x, 0);
  const int K = max(dims.y, 0);
  const int N = max(dims.z, 0);
  const int ri = int(tid.x);
  if (ri >= R) return;

  const int e = expert_id[ri];
  const bool valid = dims.w > 0 && e >= 0 && e < dims.w;
  if (!valid) {
    const int out_row =
        int(tid.y) * (num_simdgroups * results_per_simdgroup) +
        int(simd_gid) * results_per_simdgroup;
    if (simd_lid == 0) {
      for (int row = 0;
           row < results_per_simdgroup && out_row + row < N; ++row) {
        out[static_cast<int64_t>(ri) * N + out_row + row] = bfloat(0);
      }
    }
    return;
  }

  // `e` is bounds-checked above, so this is one safe indexed f32 load per
  // threadgroup. Both branches are compiled, but the function constant is
  // resolved at pipeline creation, so each specialization keeps only one.
  //
  // K%8 holds unconditionally (group_size 16 forces K%16==0), so there is no
  // partial-K tail to guard. See the K-tail note at the top of this file.
  if (moe_has_global_scale) {
    const float inv_g = 1.0f / w_global_scale[e];
    gather_qmv_moe_impl<bfloat, 16, 4, true>(
        w, scale, e, x, out, R, K, N, inv_g, tid, simd_gid, simd_lid);
  } else {
    gather_qmv_moe_impl<bfloat, 16, 4, false>(
        w, scale, e, x, out, R, K, N, 1.0f, tid, simd_gid, simd_lid);
  }
}
