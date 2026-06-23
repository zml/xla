// mpp_tensor.h - matmul2d tensor-view GEMM - the primary backend.
#ifdef MB_BUILD_MPP_TENSOR
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T        __IN_T__
#define OUT_T       __OUT_T__
#define BM          __BM__
#define BN          __BN__
#define NSG         __NSG__
#define TRANS_A     __TRANS_A__
#define TRANS_B     __TRANS_B__
#define RELAXED     __RELAXED__
#define SWIZZLE_LOG __SWIZZLE_LOG__
#define MN_ALIGNED  __MN_ALIGNED__
#define STATIC_SLICE __STATIC_SLICE__
// MB_TOKCLAMP (prefill token-axis GEMM): the host dispatches the FULL static M
// grid and passes the real prompt length as a DEVICE pointer (mb_numtok); each
// threadgroup early-returns if its whole M-tile is past num_tokens. This replaces
// the old host-side encode-time read of num_tokens (which raced the GPU producer
// of that metadata on Metal — there is no totally-ordered stream), so the grid is
// clamped purely on-GPU like CUDA, with no host-side device read. 0 = full grid.
#define MB_TOKCLAMP __MB_TOKCLAMP__

// BATCHED (bmm/baddbmm): grid z is the batch index; A/B/C (and the bias) are offset
// by their per-matrix strides. Defaults to 0 so the 2-D build is unchanged.
#ifndef BATCHED
#define BATCHED 0
#endif
#if EPILOGUE
#define MB_BATCH_BUF 8
#else
#define MB_BATCH_BUF 4
#endif
#if BATCHED && EPILOGUE
#define MB_BBAT _bbat                      // per-batch bias base (baddbmm)
#else
#define MB_BBAT 0
#endif

// Write the BM x BN tile from cT_f32 into cT_out (addmm epilogue when EPILOGUE); the
// fragment index gives (col, row) so the broadcast bias lands right. VALIDATE skips
// out-of-tile elements (partial edge / unaligned tiles).
#if EPILOGUE
#define MB_STORE_TILE(VALIDATE) do {                                              \
    _Pragma("unroll")                                                            \
    for (uint16_t _e = 0; _e < cT_f32.get_capacity(); ++_e)                      \
        if (!(VALIDATE) || cT_f32.is_valid_element(_e)) {                         \
            auto _idx = cT_f32.get_multidimensional_index(_e);                    \
            int _r = m_off + (int)_idx[1], _c = n_off + (int)_idx[0];             \
            cT_out[_e] = mb_epi<OUT_T, float, float>(                             \
                cT_f32[_e], bias, MB_BBAT + _r * bstride.x + _c * bstride.y, beta, alpha); \
        }                                                                         \
} while (0)
#else
#define MB_STORE_TILE(VALIDATE) do {                                              \
    for (uint16_t _i = 0; _i < cT_f32.get_capacity(); ++_i)                      \
        if (!(VALIDATE) || cT_f32.is_valid_element(_i))                           \
            cT_out[_i] = (OUT_T)cT_f32[_i];                                        \
} while (0)
#endif
struct MBTensorDims { int M, N, K, lda, ldb, ldc; };

kernel void mpp_tensor_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    constant MBTensorDims& gP [[buffer(3)]],   // (M, N, K, lda, ldb, ldc)
#if EPILOGUE
    device const OUT_T *bias [[buffer(4)]],   // addmm input; bstride = (row, col) broadcast strides
    constant int2&  bstride  [[buffer(5)]],
    constant float& beta     [[buffer(6)]],
    constant float& alpha    [[buffer(7)]],
#endif
#if MB_TOKCLAMP
    // Real prompt length (cu_seqlens/query_start_len[num_seqs]); read on-GPU.
    // buffer(4) is free here: the prefill token-axis GEMM is never EPILOGUE.
    device const int* mb_numtok [[buffer(4)]],
#endif
#if BATCHED
    constant int4& batch [[buffer(MB_BATCH_BUF)]],   // (sA, sB, sC, sBias) per-batch element strides
#endif
    uint3 tgid         [[threadgroup_position_in_grid]])
{
#if BATCHED
    A += (int64_t)tgid.z * (int64_t)batch.x;
    B += (int64_t)tgid.z * (int64_t)batch.y;
    C += (int64_t)tgid.z * (int64_t)batch.z;
  #if EPILOGUE
    int _bbat = (int)tgid.z * batch.w;
  #endif
#endif
    int gM = gP.M, gN = gP.N, gK = gP.K;
    auto eA = TRANS_A ? dextents<int32_t, 2>(gM, gK) : dextents<int32_t, 2>(gK, gM);
    auto eB = TRANS_B ? dextents<int32_t, 2>(gK, gN) : dextents<int32_t, 2>(gN, gK);
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(A, eA, array<int32_t, 2>{1, gP.lda});
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(B, eB, array<int32_t, 2>{1, gP.ldb});
    tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(gN, gM), array<int32_t, 2>{1, gP.ldc});

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, dynamic_extent, TRANS_A, TRANS_B, RELAXED,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    // Swizzle threadgroup ids for L2 reuse.
    int tiles_m = (gM + BM - 1) / BM;
    int tiles_n = (gN + BN - 1) / BN;
    int sw_mask = (1 << SWIZZLE_LOG) - 1;
    int tgy = (int(tgid.y) << SWIZZLE_LOG) | (int(tgid.x) & sw_mask);
    int tgx = int(tgid.x) >> SWIZZLE_LOG;
    if (tgx >= tiles_n || tgy >= tiles_m) return;

    int m_off = tgy * BM;
    int n_off = tgx * BN;

#if MB_TOKCLAMP
    // GPU-side prefill row clamp: skip any M-tile entirely past the real prompt
    // length. Tiles straddling num_tokens still run (their valid rows feed the
    // causal output); for a full prompt num_tokens>=M so no tile is skipped
    // (byte-identical to the unclamped path).
    if (m_off >= *mb_numtok) return;
#endif

#if STATIC_SLICE
    // Non-transposed fast path: static-extent slices mark each interior tile exactly
    // BM x BN and in-bounds, dropping dynamic-slice edge predication (still fp32 accum).
  #if !MN_ALIGNED
    // Partial edge tiles (M%BM or N%BN != 0) fall back to a dynamic slice with
    // the per-element validity mask; interior tiles take the static path below.
    bool inside = (m_off + BM <= gM) && (n_off + BN <= gN);
    if (!inside) {
        auto mA = tA.slice(0, m_off);
        auto mB = tB.slice(n_off, 0);
        auto mC = tC.slice(n_off, m_off);
        auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        op.run(mA, mB, cT_f32);
        auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
        MB_STORE_TILE(1);
        cT_out.store(mC);
        return;
    }
  #endif
    auto mA = tA.slice<dynamic_extent, BM>(0, m_off);
    auto mB = tB.slice<BN, dynamic_extent>(n_off, 0);
    auto mC = tC.slice<BN, BM>(n_off, m_off);
    auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT_f32);
    auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
    MB_STORE_TILE(0);
    cT_out.store(mC);
#else
    auto mA = TRANS_A ? tA.slice(m_off, 0) : tA.slice(0, m_off);
    auto mB = TRANS_B ? tB.slice(0, n_off) : tB.slice(n_off, 0);
    auto mC = tC.slice(n_off, m_off);
    auto cT_f32 = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT_f32);
    auto cT_out = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
  #if MN_ALIGNED
    MB_STORE_TILE(0);
  #else
    MB_STORE_TILE(1);
  #endif
    cT_out.store(mC);
#endif
}
#endif  // MB_BUILD_MPP_TENSOR
