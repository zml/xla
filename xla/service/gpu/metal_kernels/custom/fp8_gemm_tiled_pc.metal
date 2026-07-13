#include <metal_stdlib>
using namespace metal;

// Per-CHANNEL FP8 thin GEMM for batched decode (M in 1..16).
//
//   out[m, n] = scale[n] * sum_k x[m, k] * decode_e4m3fn(w[n, k])
//
//   x:     bfloat [M, K]     w: uchar [N, K] (f8e4m3fn)
//   scale: bfloat [N, 1]     out: bfloat [M, N]
//   dims = (M, K, N, unused)
//
// Weight-reuse: each output column's f8 row is read from DRAM once and fused
// into all M activation rows. A naive per-row GEMV re-reads W M times and
// collapses at batch 8–16.
//
// Tuned on Apple M4 Max (tools/fp8_pc_bench), target shape ~M=8/16, K=5120,
// N=17408 (Qwen-scale MLP). Winning configs:
//   M==8  → fp8_gemm_tiled_pc_m8  (TN=4 cols/TG, 64 thr/col)  ~248 GB/s
//   M==16 → fp8_gemm_tiled_pc_m16 (TN=2 cols/TG, 128 thr/col) ~128 GB/s
//   other → fp8_gemm_tiled_pc     (dyn M<=16, 1 col/TG)       fallback
//
// NOT wired into the thunk yet — call sites still to be added. Launch:
//   m8:  ThreadDim(256,1,1), BlockDim(ceil(N/4), 1, 1)
//   m16: ThreadDim(256,1,1), BlockDim(ceil(N/2), 1, 1)
//   dyn: ThreadDim(256,1,1), BlockDim(N, 1, 1)

static inline float decode_e4m3fn(uchar b) {
    int s = (b >> 7) & 0x1;
    int e = (b >> 3) & 0xf;
    int m = b & 0x7;
    float v = (e == 0) ? (float(m) * 1.953125e-3f)
                       : (float(8 + m) * exp2(float(e) - 10.0f));
    return s ? -v : v;
}

constant constexpr int MAX_M = 16;

// ---- dyn M <= 16, one column per TG ---------------------------------------
kernel void fp8_gemm_tiled_pc(
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
    const int M = dims.x, K = dims.y, N = dims.z;
    threadgroup float lut[256];
    lut[tid] = decode_e4m3fn(uchar(tid));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ni = int(tgid.x);
    if (ni >= N) return;

    const device uchar *wrow = w + (long)ni * K;
    const float sn = float(scale[ni]);

    float acc[MAX_M];
    #pragma clang loop unroll(full)
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int k = int(tid) * 4; k + 4 <= K; k += 256 * 4) {
        uchar4 wv = *(const device uchar4 *)(wrow + k);
        float w0 = lut[wv.x], w1 = lut[wv.y], w2 = lut[wv.z], w3 = lut[wv.w];
        #pragma clang loop unroll(full)
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                bfloat4 xv = *(const device bfloat4 *)(x + (long)m * K + k);
                acc[m] += float(xv.x) * w0 + float(xv.y) * w1 +
                          float(xv.z) * w2 + float(xv.w) * w3;
            }
        }
    }

    // One barrier after all simd_sums (not 2*M). part[m][sgid].
    threadgroup float part[MAX_M * 8];
    #pragma clang loop unroll(full)
    for (int m = 0; m < MAX_M; ++m) {
        float a = simd_sum(acc[m]);
        if (lane == 0) part[m * 8 + sgid] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        #pragma clang loop unroll(full)
        for (int m = 0; m < MAX_M; ++m) {
            float t = (lane < 8) ? part[m * 8 + lane] : 0.0f;
            t = simd_sum(t);
            if (lane == 0 && m < M) out[(long)m * N + ni] = bfloat(sn * t);
        }
    }
}

// ---- M==8, TN=4 columns per TG (best for large-N MLP on M4 Max) -----------
// 8 simdgroups → 4 cols × 2 SG/col = 64 threads/col. Grid x = ceil(N/4).
kernel void fp8_gemm_tiled_pc_m8(
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
    const int K = dims.y, N = dims.z;
    threadgroup float lut[256];
    lut[tid] = decode_e4m3fn(uchar(tid));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int col = int(sgid) / 2;            // 0..3
    const int local_sg = int(sgid) % 2;       // 0..1 within column
    const int ni = int(tgid.x) * 4 + col;
    const bool active = (ni < N);
    const int local_tid = local_sg * 32 + int(lane);

    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    if (active) {
        const device uchar *wrow = w + (long)ni * K;
        for (int k = local_tid * 4; k + 4 <= K; k += 64 * 4) {
            uchar4 wv = *(const device uchar4 *)(wrow + k);
            float w0 = lut[wv.x], w1 = lut[wv.y], w2 = lut[wv.z], w3 = lut[wv.w];
            bfloat4 x0 = *(const device bfloat4 *)(x + 0L * K + k);
            bfloat4 x1 = *(const device bfloat4 *)(x + 1L * K + k);
            bfloat4 x2 = *(const device bfloat4 *)(x + 2L * K + k);
            bfloat4 x3 = *(const device bfloat4 *)(x + 3L * K + k);
            bfloat4 x4 = *(const device bfloat4 *)(x + 4L * K + k);
            bfloat4 x5 = *(const device bfloat4 *)(x + 5L * K + k);
            bfloat4 x6 = *(const device bfloat4 *)(x + 6L * K + k);
            bfloat4 x7 = *(const device bfloat4 *)(x + 7L * K + k);
            a0 += float(x0.x)*w0+float(x0.y)*w1+float(x0.z)*w2+float(x0.w)*w3;
            a1 += float(x1.x)*w0+float(x1.y)*w1+float(x1.z)*w2+float(x1.w)*w3;
            a2 += float(x2.x)*w0+float(x2.y)*w1+float(x2.z)*w2+float(x2.w)*w3;
            a3 += float(x3.x)*w0+float(x3.y)*w1+float(x3.z)*w2+float(x3.w)*w3;
            a4 += float(x4.x)*w0+float(x4.y)*w1+float(x4.z)*w2+float(x4.w)*w3;
            a5 += float(x5.x)*w0+float(x5.y)*w1+float(x5.z)*w2+float(x5.w)*w3;
            a6 += float(x6.x)*w0+float(x6.y)*w1+float(x6.z)*w2+float(x6.w)*w3;
            a7 += float(x7.x)*w0+float(x7.y)*w1+float(x7.z)*w2+float(x7.w)*w3;
        }
    }

    float accs[8] = {a0,a1,a2,a3,a4,a5,a6,a7};
    threadgroup float part[4 * 8 * 2];  // [col][m][local_sg]
    for (int m = 0; m < 8; ++m) {
        float a = simd_sum(accs[m]);
        if (lane == 0) part[(col * 8 + m) * 2 + local_sg] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_sg == 0 && active) {
        const float sn = float(scale[ni]);
        for (int m = 0; m < 8; ++m) {
            float t = (lane < 2) ? part[(col * 8 + m) * 2 + lane] : 0.0f;
            t = simd_sum(t);
            if (lane == 0) out[(long)m * N + ni] = bfloat(sn * t);
        }
    }
}

// ---- M==16, TN=2 columns per TG -------------------------------------------
// 8 simdgroups → 2 cols × 4 SG/col = 128 threads/col. Grid x = ceil(N/2).
kernel void fp8_gemm_tiled_pc_m16(
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
    const int K = dims.y, N = dims.z;
    threadgroup float lut[256];
    lut[tid] = decode_e4m3fn(uchar(tid));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int col = int(sgid) / 4;            // 0..1
    const int local_sg = int(sgid) % 4;       // 0..3 within column
    const int ni = int(tgid.x) * 2 + col;
    const bool active = (ni < N);
    const int local_tid = local_sg * 32 + int(lane);

    float acc[16];
    #pragma clang loop unroll(full)
    for (int m = 0; m < 16; ++m) acc[m] = 0.0f;

    if (active) {
        const device uchar *wrow = w + (long)ni * K;
        for (int k = local_tid * 4; k + 4 <= K; k += 128 * 4) {
            uchar4 wv = *(const device uchar4 *)(wrow + k);
            float w0 = lut[wv.x], w1 = lut[wv.y], w2 = lut[wv.z], w3 = lut[wv.w];
            #pragma clang loop unroll(full)
            for (int m = 0; m < 16; ++m) {
                bfloat4 xv = *(const device bfloat4 *)(x + (long)m * K + k);
                acc[m] += float(xv.x)*w0 + float(xv.y)*w1 +
                          float(xv.z)*w2 + float(xv.w)*w3;
            }
        }
    }

    threadgroup float part[2 * 16 * 4];  // [col][m][local_sg]
    #pragma clang loop unroll(full)
    for (int m = 0; m < 16; ++m) {
        float a = simd_sum(acc[m]);
        if (lane == 0) part[(col * 16 + m) * 4 + local_sg] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_sg == 0 && active) {
        const float sn = float(scale[ni]);
        #pragma clang loop unroll(full)
        for (int m = 0; m < 16; ++m) {
            float t = (lane < 4) ? part[(col * 16 + m) * 4 + lane] : 0.0f;
            t = simd_sum(t);
            if (lane == 0) out[(long)m * N + ni] = bfloat(sn * t);
        }
    }
}
