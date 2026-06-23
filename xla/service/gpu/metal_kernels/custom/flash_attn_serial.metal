// Serial flash-attention fallback for the XLA Metal backend (custom): one
// threadgroup of 32 lanes per (kv_head, group), a single streaming pass over the
// KV cache to the causal cutoff. Used when head_dim != 128 or dtype != bf16.

#include <metal_stdlib>
using namespace metal;
kernel void flash_attn_vec(
    device const bfloat* q   [[buffer(0)]],
    device const bfloat* k   [[buffer(1)]],
    device const bfloat* v   [[buffer(2)]],
    device const int*    tok [[buffer(3)]],
    device       bfloat* out [[buffer(4)]],
    uint2  tgid [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint N_GROUPS = __N_GROUPS__;
  constexpr uint SEQLEN   = __SEQLEN__;
  constexpr uint HD       = __HD__;
  constexpr uint DPL      = HD / 32u;
  const float scale = 1.0f / sqrt(float(HD));
  const uint hq = tgid.x, h = tgid.y;
  const uint qhead  = h * N_GROUPS + hq;
  const uint qbase  = qhead * HD;
  const uint kvbase = h * SEQLEN * HD;
  const uint tokidx = uint(tok[0]);
  float qreg[DPL];
  for (uint i = 0; i < DPL; ++i) qreg[i] = float(q[qbase + lane + i*32]);
  float acc[DPL];
  for (uint i = 0; i < DPL; ++i) acc[i] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (uint kp = 0; kp <= tokidx && kp < SEQLEN; ++kp) {
    const uint kb = kvbase + kp * HD;
    float part = 0.0f;
    for (uint i = 0; i < DPL; ++i) part += qreg[i] * float(k[kb + lane + i*32]);
    float score = simd_sum(part) * scale;
    float mn = max(m, score);
    float corr = exp(m - mn);
    float pe = exp(score - mn);
    l = l * corr + pe;
    for (uint i = 0; i < DPL; ++i) acc[i] = acc[i] * corr + pe * float(v[kb + lane + i*32]);
    m = mn;
  }
  float inv = l > 0.0f ? 1.0f / l : 0.0f;
  for (uint i = 0; i < DPL; ++i) out[qbase + lane + i*32] = bfloat(acc[i] * inv);
}
