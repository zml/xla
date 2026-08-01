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

// Per-CHANNEL block-scaled FP8 GEMV (decode, B==1). The scale is one bf16 per
// output channel and is constant across K, so it factors out of the reduction:
//   out[bi, ni] = scale[ni] * sum_k x[bi, k] * decode_e4m3fn(w[ni, k])
//   x:     bfloat [B, K]     w: uchar [N, K] (f8e4m3fn)
//   scale: bfloat [N, 1]     out: bfloat [B, N]     dims = (B, K, N, unused)
kernel void fp8_gemv_pc(
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
    const int B = dims.x, K = dims.y, N = dims.z;
    threadgroup float lut[256];
    lut[tid] = decode_e4m3fn(uchar(tid));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ni = int(tgid.x);
    const int bi = int(tgid.y);
    if (ni >= N || bi >= B) return;

    const device uchar  *wrow = w + (long)ni * K;
    const device bfloat *xrow = x + (long)bi * K;

    float acc = 0.0f;
    for (int k = int(tid) * 4; k + 4 <= K; k += 256 * 4) {
        uchar4 wv = *(const device uchar4 *)(wrow + k);
        bfloat4 xv = *(const device bfloat4 *)(xrow + k);
        acc += float(xv.x) * lut[wv.x] + float(xv.y) * lut[wv.y] +
               float(xv.z) * lut[wv.z] + float(xv.w) * lut[wv.w];
    }
    acc = simd_sum(acc);
    threadgroup float part[8];
    if (lane == 0) part[sgid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float t = (lane < 8) ? part[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) out[(long)bi * N + ni] = bfloat(float(scale[ni]) * t);
    }
}
