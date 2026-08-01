#include <metal_stdlib>
using namespace metal;

// Row gather / scatter for the sorted-prefill MoE path. Aligned, 4-wide rows
// use bfloat4; odd widths use scalar lanes so neither tail accesses nor
// row-stride misalignment can cross an allocation. grid = (ceil(W/4), R, 1).
//
// The order entry is validated before pointer arithmetic. Invalid experts
// gather a zero row, so the sorted GEMM can safely use expert zero, and
// scatter_rows deterministically zeroes their outputs. Sorted kernels support
// E in [1, 256].
//   dims = (R, W, E, unused)

// dst[i, :] = src[idx[i], :]   -- gather routed rows into expert-sorted order.
kernel void gather_rows(
    device const bfloat* src [[buffer(0)]],   // [*, W]
    device const int*    idx [[buffer(1)]],   // [R]
    device       bfloat* dst [[buffer(2)]],   // [R, W]
    constant int4&       dims [[buffer(3)]],
    device const int*    expert_ids [[buffer(4)]],  // [R], original order
    uint2 gid [[thread_position_in_grid]])
{
    const int R_total = max(dims.x, 0), W = max(dims.y, 0), E = dims.z;
    const bool supported = E > 0 && E <= 256;
    const int i = int(gid.y);
    const int w = int(gid.x) * 4;
    if (i >= R_total || w >= W) return;
    if (!supported) return;

    bool valid = true;
    int s = idx[i];
    valid = valid && s >= 0 && s < R_total;
    const int e = valid ? expert_ids[s] : -1;
    valid = valid && e >= 0 && e < E;

    if (!valid) {
        if ((W & 3) == 0 && w + 3 < W) {
            *(device bfloat4*)(dst + (long)i * W + w) = bfloat4(0);
        } else {
            for (int lane = 0; lane < 4 && w + lane < W; ++lane)
                dst[(long)i * W + w + lane] = bfloat(0);
        }
        return;
    }
    if ((W & 3) == 0 && w + 3 < W) {
        *(device bfloat4*)(dst + (long)i * W + w) =
            *(const device bfloat4*)(src + (long)s * W + w);
    } else {
        for (int lane = 0; lane < 4 && w + lane < W; ++lane)
            dst[(long)i * W + w + lane] = src[(long)s * W + w + lane];
    }
}

// dst[idx[i], :] = src[i, :]   -- scatter the sorted GEMM output back to the
// original routed-row order.
kernel void scatter_rows(
    device const bfloat* src [[buffer(0)]],   // [R, W]
    device const int*    idx [[buffer(1)]],   // [R]
    device       bfloat* dst [[buffer(2)]],   // [*, W]
    constant int4&       dims [[buffer(3)]],
    device const int*    expert_ids [[buffer(4)]],  // [R], original order
    uint2 gid [[thread_position_in_grid]])
{
    const int R_total = max(dims.x, 0), W = max(dims.y, 0), E = dims.z;
    const bool supported = E > 0 && E <= 256;
    const int i = int(gid.y);
    const int w = int(gid.x) * 4;
    if (i >= R_total || w >= W) return;
    if (!supported) {
        if ((W & 3) == 0 && w + 3 < W) {
            *(device bfloat4*)(dst + (long)i * W + w) = bfloat4(0);
        } else {
            for (int lane = 0; lane < 4 && w + lane < W; ++lane)
                dst[(long)i * W + w + lane] = bfloat(0);
        }
        return;
    }

    const int d = idx[i];
    if (d < 0 || d >= R_total) return;
    const int e = expert_ids[d];
    if (e < 0 || e >= E) {
        if ((W & 3) == 0 && w + 3 < W) {
            *(device bfloat4*)(dst + (long)d * W + w) = bfloat4(0);
        } else {
            for (int lane = 0; lane < 4 && w + lane < W; ++lane)
                dst[(long)d * W + w + lane] = bfloat(0);
        }
        return;
    }
    if ((W & 3) == 0 && w + 3 < W) {
        *(device bfloat4*)(dst + (long)d * W + w) =
            *(const device bfloat4*)(src + (long)i * W + w);
    } else {
        for (int lane = 0; lane < 4 && w + lane < W; ++lane)
            dst[(long)d * W + w + lane] = src[(long)i * W + w + lane];
    }
}
