// splitk.h - Split-K matmul2d GEMM + reduction pass (deep-K, few-tile shapes).
#ifdef MB_BUILD_SPLITK
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T    __IN_T__
#define OUT_T   __OUT_T__
#define BM      __BM__
#define BN      __BN__
#define NSG     __NSG__
#define KCHUNK  __KCHUNK__
#define RELAXED __RELAXED__

// tgid.z = K-chunk g; g==0 writes C directly, g>=1 writes fp32 partial plane
// (g-1) of Cp. Tile K range is [g*KCHUNK, (g+1)*KCHUNK).
kernel void splitk_gemm(
    device IN_T   *A   [[buffer(0)]],
    device IN_T   *B   [[buffer(1)]],
    device OUT_T  *C   [[buffer(2)]],
    device float  *Cp  [[buffer(3)]],
    constant int4&  gP [[buffer(4)]],   // packed (gM, gN, gK)
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    int gM = gP.x, gN = gP.y, gK = gP.z;
    int g  = int(tgid.z);
    int k0 = g * KCHUNK;
    if (k0 >= gK) return;
    int tiles_n = (gN + BN - 1) / BN;
    int tgx = int(tgid.x);
    int tgy = int(tgid.y);
    if (tgx >= tiles_n) return;
    int m_off = tgy * BM;
    int n_off = tgx * BN;

    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(gK, gM));
    tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(gN, gK));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, KCHUNK, false, false, RELAXED, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    // KCHUNK divides gK, so every chunk is a full static-extent K slice.
    auto mA = tA.slice<KCHUNK, BM>(k0, m_off);
    auto mB = tB.slice<BN, KCHUNK>(n_off, k0);
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);

    if (g == 0) {
        tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(gN, gM));
        auto mC = tC.slice<BN, BM>(n_off, m_off);
        auto cO = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) cO[i] = (OUT_T)cT[i];
        cO.store(mC);
    } else {
        device float *Cg = Cp + (size_t)(g - 1) * gM * gN;
        tensor<device float, dextents<int32_t, 2>, tensor_inline> tCp(Cg, dextents<int32_t, 2>(gN, gM));
        auto mC = tCp.slice<BN, BM>(n_off, m_off);
        cT.store(mC);
    }
}

// C[i] += sum_{p<Gm1} Cp[p, i]: C holds chunk 0, accumulate remaining fp32 planes.
// One thread per output element.
kernel void splitk_reduce(
    device const float *Cp [[buffer(0)]],
    device OUT_T *C        [[buffer(1)]],
    constant int2&  gP     [[buffer(2)]],   // packed (n, Gm1)
    uint gid [[thread_position_in_grid]])
{
    int n = gP.x, Gm1 = gP.y;
    int i = int(gid);
    if (i >= n) return;
    float s = (float)C[i];
    size_t off = (size_t)i;
    for (int p = 0; p < Gm1; ++p) { s += Cp[off]; off += (size_t)n; }
    C[i] = (OUT_T)s;
}
#endif  // MB_BUILD_SPLITK
