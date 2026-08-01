#include <metal_stdlib>
using namespace metal;

static inline float decode_e4m3fn(uchar b) {
    int s = (b >> 7) & 0x1;
    int e = (b >> 3) & 0xf;
    int m = b & 0x7;
    float v = (e == 0) ? (float(m) * 1.953125e-3f)
                       : (float(8 + m) * exp2(float(e) - 10.0f));
    return s ? -v : v;
}

kernel void fp8_moe_gemv(
    device const bfloat *x          [[buffer(0)]],
    device const uchar  *w          [[buffer(1)]],
    device const bfloat *scale      [[buffer(2)]],
    device const int    *expert_id  [[buffer(3)]],
    device       bfloat *out        [[buffer(4)]],
    constant int4&       dims        [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const int R = dims.x, K = dims.y, N = dims.z;
    const int Nb = N / 128;     // scale rows per expert
    const int Kb = K / 128;     // scale cols per expert (= scale tiles over K)
    constexpr int TN = 8;       // simdgroups == output columns per threadgroup
    constexpr int BK = 128;     // K tile == one 128-element scale block

    threadgroup float  lut[256];
    threadgroup bfloat Xs[BK];
    lut[tid] = decode_e4m3fn(uchar(tid));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ri = int(tgid.y);
    if (ri >= R) return;
    const int ni = int(tgid.x) * TN + int(sgid);   // this simdgroup's column
    const bool active = (ni < N);

    const int e = expert_id[ri];
    const device bfloat *xrow = x + (long)ri * K;
    const device uchar  *wrow = w + ((long)e * N + ni) * K;
    const device bfloat *srow = scale + ((long)e * Nb + (ni >> 7)) * Kb;

    float acc = 0.0f;
    for (int kt = 0; kt < K; kt += BK) {
        if (tid < BK) Xs[tid] = xrow[kt + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            const int k = int(lane) * 4;           // 32 lanes * 4 = 128 = BK
            uchar4 wv = *(const device uchar4 *)(wrow + kt + k);
            bfloat4 xv = *(const threadgroup bfloat4 *)(Xs + k);
            float s = float(srow[kt >> 7]);         // one block scale per tile
            acc += s * (float(xv.x) * lut[wv.x] + float(xv.y) * lut[wv.y] +
                        float(xv.z) * lut[wv.z] + float(xv.w) * lut[wv.w]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    acc = simd_sum(acc);
    if (active && lane == 0) out[(long)ri * N + ni] = bfloat(acc);
}
