#include <metal_stdlib>

using namespace metal;

struct MatmulParams {
  uint m;
  uint n;
  uint k;
  uint reserved;
};

kernel void matmul_tiled16(const device float* a [[buffer(0)]],
                           const device float* b [[buffer(1)]],
                           device float* c [[buffer(2)]],
                           constant MatmulParams& params [[buffer(3)]],
                           uint2 tid [[thread_position_in_threadgroup]],
                           uint2 gid [[threadgroup_position_in_grid]]) {
  constexpr uint kTile = 16;
  threadgroup float tile_a[kTile][kTile];
  threadgroup float tile_b[kTile][kTile];

  const uint row = gid.y * kTile + tid.y;
  const uint col = gid.x * kTile + tid.x;
  float acc = 0.0f;

  for (uint base = 0; base < params.k; base += kTile) {
    const uint a_col = base + tid.x;
    const uint b_row = base + tid.y;
    tile_a[tid.y][tid.x] =
        (row < params.m && a_col < params.k) ? a[row * params.k + a_col] : 0.0f;
    tile_b[tid.y][tid.x] =
        (b_row < params.k && col < params.n) ? b[b_row * params.n + col] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < kTile; ++kk) {
      acc = fma(tile_a[tid.y][kk], tile_b[kk][tid.x], acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < params.m && col < params.n) {
    c[row * params.n + col] = acc;
  }
}

kernel void matmul_simdgroup_8x8(
    const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
    device float* c [[buffer(2)]], constant MatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kSimdTile = 8;
  constexpr uint kSimdgroupsX = 4;
  constexpr uint kSimdgroupsY = 2;

  const uint local_tile_x = simdgroup_id % kSimdgroupsX;
  const uint local_tile_y = simdgroup_id / kSimdgroupsX;
  const uint row = (group_id.y * kSimdgroupsY + local_tile_y) * kSimdTile;
  const uint col = (group_id.x * kSimdgroupsX + local_tile_x) * kSimdTile;

  simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint base = 0; base < params.k; base += kSimdTile) {
    simdgroup_float8x8 tile_a;
    simdgroup_float8x8 tile_b;
    simdgroup_load(tile_a, a + row * params.k + base, ulong(params.k));
    simdgroup_load(tile_b, b + base * params.n + col, ulong(params.n));
    simdgroup_multiply_accumulate(acc, tile_a, tile_b, acc);
  }

  simdgroup_store(acc, c + row * params.n + col, ulong(params.n));
}

kernel void matmul_relu_simdgroup_8x8(
    const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
    device float* c [[buffer(2)]], constant MatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kSimdTile = 8;
  constexpr uint kSimdgroupsX = 4;
  constexpr uint kSimdgroupsY = 2;

  const uint local_tile_x = simdgroup_id % kSimdgroupsX;
  const uint local_tile_y = simdgroup_id / kSimdgroupsX;
  const uint row = (group_id.y * kSimdgroupsY + local_tile_y) * kSimdTile;
  const uint col = (group_id.x * kSimdgroupsX + local_tile_x) * kSimdTile;

  simdgroup_float8x8 zero = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 acc = zero;

  for (uint base = 0; base < params.k; base += kSimdTile) {
    simdgroup_float8x8 tile_a;
    simdgroup_float8x8 tile_b;
    simdgroup_load(tile_a, a + row * params.k + base, ulong(params.k));
    simdgroup_load(tile_b, b + base * params.n + col, ulong(params.n));
    simdgroup_multiply_accumulate(acc, tile_a, tile_b, acc);
  }

  acc.thread_elements() =
      __builtin_elementwise_max(acc.thread_elements(), zero.thread_elements());
  simdgroup_store(acc, c + row * params.n + col, ulong(params.n));
}
