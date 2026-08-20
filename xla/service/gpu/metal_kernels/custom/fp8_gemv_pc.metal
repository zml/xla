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

// The same decode with no table: e4m3fn's fields land in half's, four at a
// time, giving decode_e4m3fn's value divided by 2^8 (half's bias is 15, e4m3's
// is 7). Subnormals fall out for free and 0x7F/0xFF decode to +-480 exactly as
// the table does, since neither form special-cases fn's NaN. The 2^8 is a power
// of two, so the caller folds it into the scale exactly.
static inline float4 dec4_over256(uchar4 b) {
    const ushort4 u = ushort4(b);
    return float4(as_type<half4>(((u & ushort4(0x007F)) << 7) |
                                 ((u & ushort4(0x0080)) << 8)));
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

// Thin-M decode: one threadgroup serves kVECS input rows at once.
//
// B>1 used to fall to the Steel BM=16 qmm, where at M=2 fourteen of sixteen
// tile rows are padding. The weight is the expensive operand, so load it ONCE
// per threadgroup and apply it to kVECS input vectors instead -- the same trick
// nvfp4_qmv_wide_V uses. Weight traffic per output row drops by kVECS while x
// traffic (a few KB, cached) rises.
//
// The live frame is kROWS*kVECS accumulators plus kVECS*(kCHUNK/4) staged x
// vectors, and the second term is what limits how wide this can go:
//
// 1. x must be staged in one array PER CHUNK POSITION. A single
//    bfloat4 xb[kVECS][4] -- or the flat bfloat4 xb[kVECS*4] -- gets demoted
//    out of registers into thread scratch once it passes ~12 elements, at a 4x
//    cost. Same trap mlx_fp4_qmv.h:63-70 documents, and it is what used to make
//    kVECS=8 slower than kVECS=4 despite reading the weights half as often.
//
// 2. kCHUNK has to SHRINK as kVECS grows, since the frame is their product:
//    16 bytes of w per thread up to kVECS=4, 8 through 7, 4 from 8 on.
template <typename ST, int kVECS>
[[kernel]] void fp8_gemv_pc_wide_entry(
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
    // Bytes of w per thread per step, and the bfloat4 of x that pairs with it.
    constexpr int kCHUNK = (kVECS >= 8) ? 4 : (kVECS >= 5 ? 8 : 16);
    constexpr int NF = kCHUNK / 4;

    const int B = dims.x, K = dims.y, N = dims.z;
    const int n0 = int(tgid.x) * kROWS;
    const int b0 = int(tgid.y) * kVECS;
    if (n0 >= N || b0 >= B) return;

    // Both tails clamp onto the last valid index rather than branching: the
    // extra lanes are computed and then not stored.
    const device uchar *wrow[kROWS];
#pragma unroll
    for (int r = 0; r < kROWS; ++r)
        wrow[r] = w + (long)min(n0 + r, N - 1) * K;
    const device bfloat *xrow[kVECS];
#pragma unroll
    for (int v = 0; v < kVECS; ++v)
        xrow[v] = x + (long)min(b0 + v, B - 1) * K;

    float acc[kROWS][kVECS];
#pragma unroll
    for (int r = 0; r < kROWS; ++r)
#pragma unroll
        for (int v = 0; v < kVECS; ++v) acc[r][v] = 0.0f;

    // K % 32 == 0 is enforced by ClassifyMetalScaledMatmul, so every kCHUNK
    // divides K and there is no tail. The unrolls are load-bearing: a residual
    // loop over v keeps the x staging addressable, which is what demotes it.
    for (int k = int(tid) * kCHUNK; k + kCHUNK <= K; k += 256 * kCHUNK) {
        bfloat4 x0[kVECS], x1[kVECS], x2[kVECS], x3[kVECS];
#pragma unroll
        for (int v = 0; v < kVECS; ++v) {
            x0[v] = *(const device bfloat4 *)(xrow[v] + k);
            if (NF > 1) x1[v] = *(const device bfloat4 *)(xrow[v] + k + 4);
            if (NF > 2) x2[v] = *(const device bfloat4 *)(xrow[v] + k + 8);
            if (NF > 3) x3[v] = *(const device bfloat4 *)(xrow[v] + k + 12);
        }
#pragma unroll
        for (int r = 0; r < kROWS; ++r) {
            float4 wq[NF];
#pragma unroll
            for (int j = 0; j < NF; ++j)
                wq[j] = dec4_over256(
                    *(const device uchar4 *)(wrow[r] + k + 4 * j));
#pragma unroll
            for (int v = 0; v < kVECS; ++v) {
                float s = dot(wq[0], float4(x0[v]));
                if (NF > 1) s += dot(wq[1], float4(x1[v]));
                if (NF > 2) s += dot(wq[2], float4(x2[v]));
                if (NF > 3) s += dot(wq[3], float4(x3[v]));
                acc[r][v] += s;
            }
        }
    }

    threadgroup float part[kROWS][kVECS][8];
    for (int r = 0; r < kROWS; ++r) {
        for (int v = 0; v < kVECS; ++v) {
            float a = simd_sum(acc[r][v]);
            if (lane == 0) part[r][v][sgid] = a;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        const bool per_tensor = dims.w < 0;
        for (int r = 0; r < kROWS; ++r) {
            const int ni = n0 + r;
            // A clamped row still reads a scale, so clamp that index too.
            const int si = per_tensor ? 0 : min(ni, N - 1);
            // Undo dec4_over256's 2^-8, exactly, once per output row.
            const float s = float(scale[si]) * 256.0f;
            for (int v = 0; v < kVECS; ++v) {
                float t = (lane < 8) ? part[r][v][lane] : 0.0f;
                t = simd_sum(t);
                const int bi = b0 + v;
                if (lane == 0 && ni < N && bi < B) {
                    out[(long)bi * N + ni] = bfloat(s * t);
                }
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

#define instantiate_fp8_gemv_pc_wide(name, st, vecs)                           \
  template [[host_name(name)]] [[kernel]] void                                 \
  fp8_gemv_pc_wide_entry<st, vecs>(                                            \
      device const bfloat*, device const uchar*, device const st*,             \
      device bfloat*, constant int4&, uint3, uint, uint, uint);

instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_2", bfloat, 2)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_2_f32", float, 2)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_3", bfloat, 3)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_3_f32", float, 3)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_4", bfloat, 4)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_4_f32", float, 4)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_5", bfloat, 5)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_5_f32", float, 5)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_6", bfloat, 6)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_6_f32", float, 6)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_7", bfloat, 7)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_7_f32", float, 7)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_8", bfloat, 8)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_8_f32", float, 8)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_9", bfloat, 9)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_9_f32", float, 9)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_10", bfloat, 10)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_10_f32", float, 10)
// 11 and 12 are past MaxVecs and unreachable by default; they exist so
// METAL_FP8_WIDE_VECS can sweep the range without a rebuild.
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_11", bfloat, 11)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_11_f32", float, 11)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_12", bfloat, 12)
instantiate_fp8_gemv_pc_wide("fp8_gemv_pc_wide_12_f32", float, 12)
