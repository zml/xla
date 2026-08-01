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

constant constexpr int MAX_M = 16;

kernel void fp8_gemm_tiled(
    device const bfloat *x      [[buffer(0)]],
    device const uchar  *w      [[buffer(1)]],
    device const bfloat *scale  [[buffer(2)]],
    device       bfloat *out    [[buffer(3)]],
    constant int4&       dims   [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const int M = dims.x, K = dims.y, N = dims.z, Kb = dims.w;

    threadgroup float lut[256];
    lut[tid] = decode_e4m3fn(uchar(tid));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ni = int(tgid.x);
    if (ni >= N) return;

    const device uchar  *wrow = w + (long)ni * K;
    const device bfloat *srow = scale + (long)(ni >> 7) * Kb;

    float acc[MAX_M];
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int k = int(tid) * 4; k + 4 <= K; k += 256 * 4) {
        uchar4 wv = *(const device uchar4 *)(wrow + k);
        float s = float(srow[k >> 7]);
        float w0 = s * lut[wv.x], w1 = s * lut[wv.y];
        float w2 = s * lut[wv.z], w3 = s * lut[wv.w];
        for (int m = 0; m < M; ++m) {
            bfloat4 xv = *(const device bfloat4 *)(x + (long)m * K + k);
            acc[m] += float(xv.x) * w0 + float(xv.y) * w1 +
                      float(xv.z) * w2 + float(xv.w) * w3;
        }
    }

    threadgroup float part[8];
    for (int m = 0; m < M; ++m) {
        float a = simd_sum(acc[m]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) part[sgid] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            float t = (lane < 8) ? part[lane] : 0.0f;
            t = simd_sum(t);
            if (lane == 0) out[(long)m * N + ni] = bfloat(t);
        }
    }
}
