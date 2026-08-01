#include <metal_stdlib>
using namespace metal;

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
    const device bfloat *wrow = w + ((long)e * N + ni) * K;

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

    acc = simd_sum(acc);
    if (active && lane == 0) out[(long)ri * N + ni] = bfloat(acc);
}
