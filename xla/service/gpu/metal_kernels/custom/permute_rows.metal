#include <metal_stdlib>
using namespace metal;

// Row gather / scatter for the sorted-prefill MoE path. W (the row width) is a
// multiple of 4 for this model (hidden=2048, moe_inter=512, 2*inter=1024), so
// rows are copied as bfloat4. grid = (ceil(W/4), R, 1).
//
// PREFILL PADDING CLAMP: like moe_argsort, only the first R_active =
// num_tokens[0]*top_k routes are real (padding is a contiguous suffix). The
// clamp here is also REQUIRED for safety, not just speed: idx[] (the argsort
// permutation) is only written for [0, R_active); beyond it idx[i] is stale, so
// reading/dereferencing it (src[idx]/dst[idx]) would be an out-of-bounds access.
// When there is nothing to clamp the thunk passes num_tokens=R, top_k=1.
//   dims = (R, W, top_k)

// dst[i, :] = src[idx[i], :]   -- gather routed rows into expert-sorted order.
kernel void gather_rows(
    device const bfloat* src [[buffer(0)]],   // [*, W]
    device const int*    idx [[buffer(1)]],   // [R]
    device       bfloat* dst [[buffer(2)]],   // [R, W]
    constant int3&       dims [[buffer(3)]],
    device const int*    num_tokens [[buffer(4)]],  // [1] real prompt length
    uint2 gid [[thread_position_in_grid]])
{
    const int R = min(dims.x, num_tokens[0] * dims.z), W = dims.y;  // R_active
    const int i = int(gid.y);
    const int w = int(gid.x) * 4;
    if (i >= R || w >= W) return;
    const long s = idx[i];
    *(device bfloat4*)(dst + (long)i * W + w) =
        *(const device bfloat4*)(src + s * W + w);
}

// dst[idx[i], :] = src[i, :]   -- scatter the sorted GEMM output back to the
// original routed-row order.
kernel void scatter_rows(
    device const bfloat* src [[buffer(0)]],   // [R, W]
    device const int*    idx [[buffer(1)]],   // [R]
    device       bfloat* dst [[buffer(2)]],   // [*, W]
    constant int3&       dims [[buffer(3)]],
    device const int*    num_tokens [[buffer(4)]],  // [1] real prompt length
    uint2 gid [[thread_position_in_grid]])
{
    const int R = min(dims.x, num_tokens[0] * dims.z), W = dims.y;  // R_active
    const int i = int(gid.y);
    const int w = int(gid.x) * 4;
    if (i >= R || w >= W) return;
    const long d = idx[i];
    *(device bfloat4*)(dst + d * W + w) =
        *(const device bfloat4*)(src + (long)i * W + w);
}
