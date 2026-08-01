#include <metal_stdlib>
using namespace metal;

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
