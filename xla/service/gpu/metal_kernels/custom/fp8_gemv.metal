#include <metal_stdlib>
using namespace metal;

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

// Fused FP8 (block-scaled) GEMV / thin GEMM, computing
//   out[bi, ni] = sum_k  x[bi, k] * dequant(w[ni, k])
//   dequant(w[ni, k]) = decode_e4m3fn(w[ni, k]) * scale[ni/128, k/128]
//
//   x:     bfloat [B, K]            (row-major; K is the contraction dim)
//   w:     uchar  [N, K]            (f8e4m3fn, row-major)
//   scale: bfloat [N/128, K/128]    (DeepSeek-style 128x128 block scales)
//   out:   bfloat [B, N]            (row-major)
//   dims = (B, K, N, Kb=K/128)
//
// One threadgroup (256 threads = 8 SIMD groups of 32) per (ni, bi) output,
// cooperatively reducing over K. The f8 weight is read once from DRAM
// (coalesced: consecutive lanes read consecutive bytes); the tiny x row stays
// resident in L2 across the N threadgroups, so DRAM traffic is ~one pass over
// the 1-byte weight — roughly half the bandwidth of a bf16 GEMV and far better
// than the generic reduce-fusion path. A 256-entry threadgroup LUT turns the
// per-element f8 decode into a single load (no per-element exp2).
kernel void fp8_gemv(
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
    const int B = dims.x, K = dims.y, N = dims.z, Kb = dims.w;

    // Each of the 256 threads decodes one of the 256 possible f8 byte values.
    threadgroup float lut[256];
    lut[tid] = decode_e4m3fn(uchar(tid));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ni = int(tgid.x);
    const int bi = int(tgid.y);
    if (ni >= N || bi >= B) return;

    const device uchar  *wrow = w + (long)ni * K;
    const device bfloat *xrow = x + (long)bi * K;
    const device bfloat *srow = scale + (long)(ni >> 7) * Kb;

    // Vectorized over K by 4 (uchar4 / bfloat4 loads: 128-byte / 256-byte
    // coalesced transactions per SIMD group). k..k+3 lie in one 128-element
    // scale block, so the block scale is read once per 4. The 256*4 stride
    // tiles K exactly for this model (all contraction dims are multiples of
    // 1024); the `k + 4 <= K` guard keeps any other K in-bounds (it would just
    // drop a <4-wide tail, which never occurs here).
    float acc = 0.0f;
    for (int k = int(tid) * 4; k + 4 <= K; k += 256 * 4) {
        uchar4 wv = *(const device uchar4 *)(wrow + k);
        bfloat4 xv = *(const device bfloat4 *)(xrow + k);
        float s = float(srow[k >> 7]);
        acc += s * (float(xv.x) * lut[wv.x] + float(xv.y) * lut[wv.y] +
                    float(xv.z) * lut[wv.z] + float(xv.w) * lut[wv.w]);
    }

    // Reduce 256 partials -> one output. simd_sum within each of the 8 SIMD
    // groups, then a final simd_sum over the 8 per-group results.
    acc = simd_sum(acc);
    threadgroup float part[8];
    if (lane == 0) part[sgid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float t = (lane < 8) ? part[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) out[(long)bi * N + ni] = bfloat(t);
    }
}
