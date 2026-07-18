// Bucket/radix-select TopK kernels for the XLA Metal backend (custom): histogram
// the top-16 bits of each sortable key, find the threshold prefix, gather
// candidates, select the exact top-k on the full key. bf16/f16 (16-bit) + f32.

#include <metal_stdlib>
using namespace metal;
#define BSHIFT 18

struct TKA { uint n; uint k; uint cap; uint pad; };

inline uint okey32_from_u16(ushort b) { ushort o = (b & 0x8000) ? (ushort)(~b) : (ushort)(b | 0x8000); return (uint)o << 16; }
inline uint okey32_from_u32(uint b)   { return (b & 0x80000000u) ? (~b) : (b | 0x80000000u); }
inline ushort u16_from_okey32(uint o) { ushort k = (ushort)(o >> 16); return (k & 0x8000) ? (ushort)(k & 0x7fff) : (ushort)(~k); }
inline uint   u32_from_okey32(uint o) { return (o & 0x80000000u) ? (o & 0x7fffffffu) : (~o); }

// MSL: position attrs must all be scalar or all the same vector -> all uint3.

kernel void radix_hist16(device const ushort* logits [[buffer(0)]], constant TKA& a [[buffer(1)]],
                         device atomic_uint* hist [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint3 tgpg [[threadgroups_per_grid]], uint3 _t [[thread_position_in_threadgroup]], uint3 _nt [[threads_per_threadgroup]]) {
  uint b = tg.y, tid = _t.x, nt = _nt.x; device const ushort* row = logits + (ulong)b * a.n; device atomic_uint* h = hist + (ulong)b * 16384;
  for (uint i = tg.x * nt + tid, s = tgpg.x * nt; i < a.n; i += s) atomic_fetch_add_explicit(&h[okey32_from_u16(row[i]) >> BSHIFT], 1u, memory_order_relaxed);
}
kernel void radix_hist32(device const uint* logits [[buffer(0)]], constant TKA& a [[buffer(1)]],
                         device atomic_uint* hist [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint3 tgpg [[threadgroups_per_grid]], uint3 _t [[thread_position_in_threadgroup]], uint3 _nt [[threads_per_threadgroup]]) {
  uint b = tg.y, tid = _t.x, nt = _nt.x; device const uint* row = logits + (ulong)b * a.n; device atomic_uint* h = hist + (ulong)b * 16384;
  for (uint i = tg.x * nt + tid, s = tgpg.x * nt; i < a.n; i += s) atomic_fetch_add_explicit(&h[okey32_from_u32(row[i]) >> BSHIFT], 1u, memory_order_relaxed);
}

kernel void radix_scan(device uint* hist [[buffer(0)]], constant TKA& a [[buffer(1)]], device uint* thresh [[buffer(2)]],
                       uint3 tg [[threadgroup_position_in_grid]], uint3 _t [[thread_position_in_threadgroup]], uint3 _nt [[threads_per_threadgroup]]) {
  uint b = tg.y, tid = _t.x, nt = _nt.x, k = a.k; device uint* h = hist + (ulong)b * 16384;
  const uint NB = 16384; uint per = NB / nt; threadgroup uint csum[1024];
  uint s = 0; for (uint j = 0; j < per; ++j) s += h[tid * per + j];
  csum[tid] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint above = 0, tch = 0;
    for (int c = (int)nt - 1; c >= 0; --c) { if (above + csum[c] >= k) { tch = (uint)c; break; } above += csum[c]; }
    uint pth = 0, ca = above;
    for (int j = (int)per - 1; j >= 0; --j) { uint bk = tch * per + (uint)j; uint c = h[bk]; if (ca + c >= k) { pth = bk; break; } ca += c; }
    thresh[b] = pth;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < per; ++j) h[tid * per + j] = 0u;
}

kernel void radix_gather16(device const ushort* logits [[buffer(0)]], constant TKA& a [[buffer(1)]], device const uint* thresh [[buffer(2)]],
                           device atomic_uint* ccount [[buffer(3)]], device uint* cok [[buffer(4)]], device uint* cix [[buffer(5)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint3 tgpg [[threadgroups_per_grid]], uint3 _t [[thread_position_in_threadgroup]], uint3 _nt [[threads_per_threadgroup]]) {
  uint b = tg.y, tid = _t.x, nt = _nt.x, pth = thresh[b]; device const ushort* row = logits + (ulong)b * a.n;
  device uint* co = cok + (ulong)b * a.cap; device uint* ci = cix + (ulong)b * a.cap;
  // Two-tier gather: bins STRICTLY above pth are the genuine winners (< k total,
  // so they fit and must never be evicted); the threshold bin (== pth) is the
  // over-populated remainder. Use two real counters per row: ccount[2*b] for
  // winners, ccount[2*b+1] for threshold-bin candidates. Do not bit-pack them:
  // a full-vocab -inf threshold bin can overflow 16 bits and corrupt the winner
  // count.
  device atomic_uint* cg = ccount + (ulong)b * 2;
  device atomic_uint* ce = cg + 1;
  for (uint i = tg.x * nt + tid, s = tgpg.x * nt; i < a.n; i += s) { uint o = okey32_from_u16(row[i]); uint bin = o >> BSHIFT; if (bin > pth) { uint p = atomic_fetch_add_explicit(cg, 1u, memory_order_relaxed); if (p < a.k) { co[p] = o; ci[p] = i; } } else if (bin == pth) { uint p = atomic_fetch_add_explicit(ce, 1u, memory_order_relaxed); if (p < a.cap - a.k) { co[a.k + p] = o; ci[a.k + p] = i; } } }
}
kernel void radix_gather32(device const uint* logits [[buffer(0)]], constant TKA& a [[buffer(1)]], device const uint* thresh [[buffer(2)]],
                           device atomic_uint* ccount [[buffer(3)]], device uint* cok [[buffer(4)]], device uint* cix [[buffer(5)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint3 tgpg [[threadgroups_per_grid]], uint3 _t [[thread_position_in_threadgroup]], uint3 _nt [[threads_per_threadgroup]]) {
  uint b = tg.y, tid = _t.x, nt = _nt.x, pth = thresh[b]; device const uint* row = logits + (ulong)b * a.n;
  device uint* co = cok + (ulong)b * a.cap; device uint* ci = cix + (ulong)b * a.cap;
  // Two-tier gather (see radix_gather16).
  device atomic_uint* cg = ccount + (ulong)b * 2;
  device atomic_uint* ce = cg + 1;
  for (uint i = tg.x * nt + tid, s = tgpg.x * nt; i < a.n; i += s) { uint o = okey32_from_u32(row[i]); uint bin = o >> BSHIFT; if (bin > pth) { uint p = atomic_fetch_add_explicit(cg, 1u, memory_order_relaxed); if (p < a.k) { co[p] = o; ci[p] = i; } } else if (bin == pth) { uint p = atomic_fetch_add_explicit(ce, 1u, memory_order_relaxed); if (p < a.cap - a.k) { co[a.k + p] = o; ci[a.k + p] = i; } } }
}

inline bool sgt(uint2 a, uint2 b) { return a.x == b.x ? a.y < b.y : a.x > b.x; }
#define SEL(NAME, K, OUTT, CONV) \
kernel void NAME(device const uint* cok [[buffer(0)]], device const uint* cix [[buffer(1)]], device atomic_uint* ccount [[buffer(2)]], \
                 constant TKA& a [[buffer(3)]], device OUTT* outv [[buffer(4)]], device uint* outi [[buffer(5)]], \
                 threadgroup uint2* sc [[threadgroup(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 _t [[thread_position_in_threadgroup]], uint3 _nt [[threads_per_threadgroup]]) { \
  uint b = tg.y, tid = _t.x, nt = _nt.x, kreq = a.k, cap = a.cap; \
  device const uint* co = cok + (ulong)b * cap; device const uint* ci = cix + (ulong)b * cap; device atomic_uint* cg = ccount + (ulong)b * 2; device atomic_uint* ce = cg + 1; \
  uint n_gt = min(atomic_load_explicit(cg, memory_order_relaxed), kreq); uint n_eq = min(atomic_load_explicit(ce, memory_order_relaxed), cap - kreq); uint cn = n_gt + n_eq; \
  const uint SCAP = 2048u; uint nsc = min(n_eq, SCAP); \
  for (uint i = tid; i < nsc; i += nt) sc[i] = uint2(co[kreq + i], ci[kreq + i]); \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  if (tid == 0u) { uint2 t[K]; for (int i = 0; i < K; ++i) t[i] = uint2(0u, 0xffffffffu); \
    for (uint c = 0; c < cn; ++c) { uint2 kv; \
      if (c < n_gt) kv = uint2(co[c], ci[c]); \
      else { uint e = c - n_gt; kv = e < nsc ? sc[e] : uint2(co[kreq + e], ci[kreq + e]); } \
      bool p = sgt(t[K-1], kv); t[K-1] = p ? t[K-1] : kv; \
      for (int j = K - 2; j >= 0; --j) { bool q = sgt(t[j], kv); uint2 tt = t[j]; t[j] = q ? t[j] : t[j+1]; t[j+1] = q ? t[j+1] : tt; } } \
    device OUTT* ov = outv + (ulong)b * kreq; device uint* oi = outi + (ulong)b * kreq; \
    for (uint i = 0; i < kreq; ++i) { ov[i] = CONV(t[i].x); oi[i] = t[i].y; } \
    atomic_store_explicit(cg, 0u, memory_order_relaxed); atomic_store_explicit(ce, 0u, memory_order_relaxed); } }
SEL(radix_sel16_k1, 1, ushort, u16_from_okey32) SEL(radix_sel16_k2, 2, ushort, u16_from_okey32)
SEL(radix_sel16_k4, 4, ushort, u16_from_okey32) SEL(radix_sel16_k8, 8, ushort, u16_from_okey32) SEL(radix_sel16_k16, 16, ushort, u16_from_okey32)
SEL(radix_sel16_k32, 32, ushort, u16_from_okey32) SEL(radix_sel16_k64, 64, ushort, u16_from_okey32)
SEL(radix_sel32_k1, 1, uint, u32_from_okey32) SEL(radix_sel32_k2, 2, uint, u32_from_okey32)
SEL(radix_sel32_k4, 4, uint, u32_from_okey32) SEL(radix_sel32_k8, 8, uint, u32_from_okey32) SEL(radix_sel32_k16, 16, uint, u32_from_okey32)
SEL(radix_sel32_k32, 32, uint, u32_from_okey32) SEL(radix_sel32_k64, 64, uint, u32_from_okey32)

// Single-pass small-n TopK. Runs the SAME (okey desc, index asc) insertion
// select as the radix SEL path, reading the input directly — no histogram/scan/
// gather, so ONE dispatch instead of four. The top-k under that total order is
// unique, so this is byte-identical to the 4-pass radix result.
// Used for small n only (MoE router topk: n = #experts ~32-128); a full-vocab
// scan would be far too slow, so large-n keeps radix.
//
// Parallelised across the whole (32-thread) threadgroup: each thread runs the
// exact insertion select over a strided slice of the input into its OWN sorted
// top-K (descending), then the T per-thread lists are tree-merged in threadgroup
// memory (2-pointer merge of two sorted top-K lists -> their combined top-K).
// Because sgt() is a strict total order (indices are unique), the top-K set AND
// order are uniquely determined, so the parallel merge is bit-identical to the
// prior single-thread scan. TG (=32) matches the thunk's ThreadDim; sc is sized
// for the max threadgroup and the merge is guarded so any nt<=TG is correct.
#define SEL1_TG 32u
#define SEL1(NAME, K, INT, OUTT, OKEY, CONV) \
kernel void NAME(device const INT* logits [[buffer(0)]], constant TKA& a [[buffer(1)]], \
                 device OUTT* outv [[buffer(2)]], device uint* outi [[buffer(3)]], \
                 uint3 tg [[threadgroup_position_in_grid]], uint3 _t [[thread_position_in_threadgroup]], \
                 uint3 _nt [[threads_per_threadgroup]]) { \
  threadgroup uint2 sc[SEL1_TG * K]; \
  uint b = tg.y, n = a.n, kreq = a.k, tid = _t.x, nt = min(_nt.x, SEL1_TG); \
  device const INT* row = logits + (ulong)b * n; \
  uint2 t[K]; for (int i = 0; i < K; ++i) t[i] = uint2(0u, 0xffffffffu); \
  for (uint c = tid; c < n; c += nt) { uint2 kv = uint2(OKEY(row[c]), c); \
    bool p = sgt(t[K-1], kv); t[K-1] = p ? t[K-1] : kv; \
    for (int j = K - 2; j >= 0; --j) { bool q = sgt(t[j], kv); uint2 tt = t[j]; t[j] = q ? t[j] : t[j+1]; t[j+1] = q ? t[j+1] : tt; } } \
  for (int i = 0; i < K; ++i) sc[tid * K + i] = t[i]; \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  for (uint stride = SEL1_TG >> 1; stride > 0u; stride >>= 1) { \
    if (tid < stride && tid + stride < nt) { \
      threadgroup uint2* la = sc + tid * K; threadgroup uint2* lb = sc + (tid + stride) * K; \
      uint2 m[K]; uint ia = 0u, ib = 0u; \
      for (int i = 0; i < K; ++i) { uint2 va = la[ia], vb = lb[ib]; bool takea = sgt(va, vb); \
        m[i] = takea ? va : vb; if (takea) ia++; else ib++; } \
      for (int i = 0; i < K; ++i) la[i] = m[i]; } \
    threadgroup_barrier(mem_flags::mem_threadgroup); } \
  if (tid != 0u) return; \
  device OUTT* ov = outv + (ulong)b * kreq; device uint* oi = outi + (ulong)b * kreq; \
  for (uint i = 0; i < kreq; ++i) { ov[i] = CONV(sc[i].x); oi[i] = sc[i].y; } }
SEL1(topk1_16_k1, 1, ushort, ushort, okey32_from_u16, u16_from_okey32) SEL1(topk1_16_k2, 2, ushort, ushort, okey32_from_u16, u16_from_okey32)
SEL1(topk1_16_k4, 4, ushort, ushort, okey32_from_u16, u16_from_okey32) SEL1(topk1_16_k8, 8, ushort, ushort, okey32_from_u16, u16_from_okey32) SEL1(topk1_16_k16, 16, ushort, ushort, okey32_from_u16, u16_from_okey32)
SEL1(topk1_16_k32, 32, ushort, ushort, okey32_from_u16, u16_from_okey32) SEL1(topk1_16_k64, 64, ushort, ushort, okey32_from_u16, u16_from_okey32)
SEL1(topk1_32_k1, 1, uint, uint, okey32_from_u32, u32_from_okey32) SEL1(topk1_32_k2, 2, uint, uint, okey32_from_u32, u32_from_okey32)
SEL1(topk1_32_k4, 4, uint, uint, okey32_from_u32, u32_from_okey32) SEL1(topk1_32_k8, 8, uint, uint, okey32_from_u32, u32_from_okey32) SEL1(topk1_32_k16, 16, uint, uint, okey32_from_u32, u32_from_okey32)
SEL1(topk1_32_k32, 32, uint, uint, okey32_from_u32, u32_from_okey32) SEL1(topk1_32_k64, 64, uint, uint, okey32_from_u32, u32_from_okey32)
