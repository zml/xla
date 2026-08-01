// Predicated KV-cache store for the XLA Metal backend (custom): one thread per
// (head, dim); writes v_cache raw and k_cache RoPE-rotated at the slot, skipping
// padding / out-of-range slots. Replaces zml's slice-old + select + in-place DUS
// cluster; the RoPE arithmetic replicates the fused HLO exactly.

#include <metal_stdlib>
using namespace metal;

#define KVH $0
#define HD $1
#define NUM_SLOTS $2

kernel void kv_write(
    device bfloat* k_cache              [[buffer(0)]],
    const device bfloat* k_new          [[buffer(1)]],
    device bfloat* v_cache              [[buffer(2)]],
    const device bfloat* v_new          [[buffer(3)]],
    const device int* slot              [[buffer(4)]],
    const device int* pos               [[buffer(5)]],
    const device float* freq            [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= uint(KVH * HD)) return;
  const int s = slot[0];
  if (s < 0 || s >= NUM_SLOTS) return;
  const uint h = tid / uint(HD);
  const uint d = tid % uint(HD);
  const uint base = (uint(s) * uint(KVH) + h) * uint(HD);
  v_cache[base + d] = v_new[tid];
  const uint half_hd = uint(HD) / 2u;
  const uint j = (d < half_hd) ? d : (d - half_hd);
  const float ang = float(pos[0]) * freq[j];
  const float c = float(bfloat(cos(ang)));
  const float sn = float(bfloat(sin(ang)));
  const float x1 = float(k_new[h * uint(HD) + j]);
  const float x2 = float(k_new[h * uint(HD) + half_hd + j]);
  const float r = (d < half_hd) ? (x1 * c - x2 * sn) : (x1 * sn + x2 * c);
  k_cache[base + d] = bfloat(r);
}
