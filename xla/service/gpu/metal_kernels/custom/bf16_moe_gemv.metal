#include <metal_stdlib>
using namespace metal;

// bf16 (un-quantized) twin of fp8_moe_gemv: grouped GEMV for mixture-of-experts
// expert projections, x-caching tiled variant. Each output row picks its own
// expert weight matrix:
//   e = expert_id[ri]
//   out[ri, ni] = sum_k  x[ri, k] * w[e, ni, k]
//
//   x:         bfloat [R, K]    (row-major; K is the contraction dim)
//   w:         bfloat [E, N, K] (row-major; expert-major)
//   expert_id: int    [R]       (which expert each output row uses)
//   out:       bfloat [R, N]    (row-major)
//   dims = (R, K, N, *)         (4th word ignored; shared layout with fp8 path)
//
// One threadgroup owns a block of TN=8 columns for a single row (TN simdgroups,
// one column each) and streams x in BK=128 tiles through threadgroup memory: each
// x tile is loaded from DRAM once and reused by all TN columns, so x is read
// ~N/TN times per row instead of N. The bf16 weight column is read once
// (coalesced) per (ni, ri).
kernel void bf16_moe_gemv(
    device const bfloat *x          [[buffer(0)]],
    device const bfloat *w          [[buffer(1)]],
    device const int    *expert_id  [[buffer(2)]],
    device       bfloat *out        [[buffer(3)]],
    constant int4&       dims        [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const int R = dims.x, K = dims.y, N = dims.z;
    constexpr int TN = 8;       // simdgroups == output columns per threadgroup
    constexpr int BK = 128;     // K tile

    threadgroup bfloat Xs[BK];

    const int ri = int(tgid.y);
    if (ri >= R) return;
    const int ni = int(tgid.x) * TN + int(sgid);   // this simdgroup's column
    const bool active = (ni < N);

    const int e = expert_id[ri];
    const device bfloat *xrow = x + (long)ri * K;
    // wrow is only dereferenced when active (ni < N).
    const device bfloat *wrow = w + ((long)e * N + ni) * K;

    // K-tiled reduction: load each x tile into Xs once (shared by all TN
    // columns), then each simdgroup's 32 lanes cover the 128-wide tile (4 k
    // each, vectorized). All threads cross the barriers uniformly; only the
    // math/store are guarded by `active`.
    //
    // K need NOT be a multiple of BK=128 (e.g. Gemma4-A4B's down projection has
    // K=704). The last tile may be partial; guard both the x load and the
    // weight read so neither reads past row K (the weight read would otherwise
    // spill into the next output column, or past the buffer for the last one).
    // Only requirement: K % 4 == 0 (the bfloat4-vectorized read). Lanes past the
    // tail simply contribute nothing.
    float acc = 0.0f;
    for (int kt = 0; kt < K; kt += BK) {
        if (tid < BK) Xs[tid] = (kt + int(tid) < K) ? xrow[kt + tid] : bfloat(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            const int k = int(lane) * 4;           // 32 lanes * 4 = 128 = BK
            if (kt + k < K) {                      // skip lanes past row K
                bfloat4 wv = *(const device bfloat4 *)(wrow + kt + k);
                bfloat4 xv = *(const threadgroup bfloat4 *)(Xs + k);
                acc += float(xv.x) * float(wv.x) + float(xv.y) * float(wv.y) +
                       float(xv.z) * float(wv.z) + float(xv.w) * float(wv.w);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Each simdgroup owns one full column: a single simd_sum over its 32 lanes.
    acc = simd_sum(acc);
    if (active && lane == 0) out[(long)ri * N + ni] = bfloat(acc);
}
