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

// Per-CHANNEL block-scaled FP8 GEMV (decode, B==1). The scale is one value per
// output channel and is constant across K, so it factors out of the reduction:
//   out[bi, ni] = scale[ni] * sum_k x[bi, k] * decode_e4m3fn(w[ni, k])
//   x:     bfloat [B, K]     w: uchar [N, K] (f8e4m3fn)
//   scale: ST     [N, 1]     out: bfloat [B, N]
//   dims = (B, K, N, scale row stride: 1 for [N,1], negative for a [1,1]
//           whole-tensor scale, which is stride 0)
//
// One threadgroup computes kROWS output channels, not one. With one row per
// group the x row is re-read by every group -- at N=14336, K=5120 that is
// 14336 x 10 KB = 143 MB of x traffic against only 73 MB of weights, and the
// LUT build and the final reduction are paid per row. Four rows quarters the x
// traffic and amortises both, which measured 165 -> 509 GB/s on an M4 Max
// (~510 GB/s achievable), i.e. 3.1x, and it also removes a large run-to-run
// variance the one-row form had. Eight rows and 32-byte loads were both tried
// and are slower.
//
// But kROWS also DIVIDES the threadgroup count, and 4 gives up more in
// occupancy than it wins back in x traffic. Benchmarked in isolation on the
// four shapes this model runs (median of 32 timed reps after 40 warmup, over a
// rotating >=512 MB pool of distinct weights so every rep reads cold DRAM;
// GB/s, ceiling ~510):
//
//     shape                  kROWS=1  kROWS=2  kROWS=4  kROWS=8
//     qkv    N=14336 K=5120       98      509      516      500
//     inqkv  N=10240 K=5120      338      514      506      485
//     in_z   N=6144  K=5120      100      500      487      458
//     o_proj N=5120  K=6144      392      499      486      460
//
// 2 wins or ties everywhere and 8 is uniformly worse, so there is nothing to
// select on -- one row is starved for the reasons above, and past two the
// threadgroup count falls faster than the x saving pays for it. In situ 2 also
// beat 4 on the two wide shapes (inqkv 107.3 -> 104.9 us, qkv 147.8 -> 145.7),
// which is why this is a constant and not a heuristic on N.
//
// ST is the scale's storage type. The accumulator is already f32 and the scale
// multiplies it after the reduction, so an f32 scale is used at full precision
// here -- unlike the qmm path, where the product still lands in a bf16 tile.

constant constexpr int kROWS = 2;

template <typename ST>
[[kernel]] void fp8_gemv_pc_entry(
    device const bfloat *x      [[buffer(0)]],
    device const uchar  *w      [[buffer(1)]],
    device const ST     *scale  [[buffer(2)]],
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

    const int n0 = int(tgid.x) * kROWS;
    const int bi = int(tgid.y);
    if (n0 >= N || bi >= B) return;

    // Rows past N are clamped onto the last valid row: they are computed but
    // never stored, which keeps the inner loop branch-free.
    const device uchar *wrow[kROWS];
    for (int r = 0; r < kROWS; ++r)
        wrow[r] = w + (long)min(n0 + r, N - 1) * K;
    const device bfloat *xrow = x + (long)bi * K;

    float acc[kROWS];
    for (int r = 0; r < kROWS; ++r) acc[r] = 0.0f;

    // 16 bytes of w per thread per step. K % 32 == 0 is enforced by
    // ClassifyMetalScaledMatmul, so K % 16 == 0 and there is no tail.
    for (int k = int(tid) * 16; k + 16 <= K; k += 256 * 16) {
        bfloat4 b0 = *(const device bfloat4 *)(xrow + k);
        bfloat4 b1 = *(const device bfloat4 *)(xrow + k + 4);
        bfloat4 b2 = *(const device bfloat4 *)(xrow + k + 8);
        bfloat4 b3 = *(const device bfloat4 *)(xrow + k + 12);
        for (int r = 0; r < kROWS; ++r) {
            uchar4 a0 = *(const device uchar4 *)(wrow[r] + k);
            uchar4 a1 = *(const device uchar4 *)(wrow[r] + k + 4);
            uchar4 a2 = *(const device uchar4 *)(wrow[r] + k + 8);
            uchar4 a3 = *(const device uchar4 *)(wrow[r] + k + 12);
            acc[r] += float(b0.x)*lut[a0.x] + float(b0.y)*lut[a0.y]
                    + float(b0.z)*lut[a0.z] + float(b0.w)*lut[a0.w]
                    + float(b1.x)*lut[a1.x] + float(b1.y)*lut[a1.y]
                    + float(b1.z)*lut[a1.z] + float(b1.w)*lut[a1.w]
                    + float(b2.x)*lut[a2.x] + float(b2.y)*lut[a2.y]
                    + float(b2.z)*lut[a2.z] + float(b2.w)*lut[a2.w]
                    + float(b3.x)*lut[a3.x] + float(b3.y)*lut[a3.y]
                    + float(b3.z)*lut[a3.z] + float(b3.w)*lut[a3.w];
        }
    }

    threadgroup float part[kROWS][8];
    for (int r = 0; r < kROWS; ++r) {
        acc[r] = simd_sum(acc[r]);
        if (lane == 0) part[r][sgid] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        // Stride 0 makes every row read scale[0] -- a uniform load, cheaper
        // than the per-row one -- which is a whole-tensor scale.
        const bool per_tensor = dims.w < 0;
        for (int r = 0; r < kROWS; ++r) {
            float t = (lane < 8) ? part[r][lane] : 0.0f;
            t = simd_sum(t);
            const int ni = n0 + r;
            if (lane == 0 && ni < N) {
                const int si = per_tensor ? 0 : ni;
                out[(long)bi * N + ni] = bfloat(float(scale[si]) * t);
            }
        }
    }
}

// An MSL entry point cannot itself be a template, so each scale dtype is
// stamped out by name -- the same idiom as gdn_linear_attention.
#define instantiate_fp8_gemv_pc(name, st)                                 \
  template [[host_name(name)]] [[kernel]] void fp8_gemv_pc_entry<st>(     \
      device const bfloat*, device const uchar*, device const st*,        \
      device bfloat*, constant int4&, uint3, uint, uint, uint);

instantiate_fp8_gemv_pc("fp8_gemv_pc", bfloat)
instantiate_fp8_gemv_pc("fp8_gemv_pc_f32", float)
