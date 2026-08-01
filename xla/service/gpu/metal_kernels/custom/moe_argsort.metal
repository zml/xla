#include <metal_stdlib>
using namespace metal;

// Counting sort of the R routed rows by expert id, in a single threadgroup
// (256 threads). expert_ids[r] in [0, E), E <= 256. Produces:
//   order[pos]      = original row index at sorted position `pos`
//   idx_sorted[pos] = expert id at sorted position `pos`
// The permutation GROUPS rows by expert (order within a group is arbitrary).
// This is what the MLX gather q-GEMM (fp8_gather_qmm_rhs) needs: its per-BM-tile
// run-walk only reuses an expert's weight across rows that are CONTIGUOUS, so
// the routed rows must be pre-grouped by expert. Used only for the large-R
// (prefill) MoE GEMM, where each expert serves many rows (big weight reuse).
//
// PREFILL PADDING CLAMP: the compiled R = padded_seqlen*top_k, but only the
// real-prompt routes [0, num_tokens*top_k) carry meaningful rows (routes are
// token-major, so padding is a contiguous suffix). num_tokens[0] is the real
// prompt length (a device scalar produced by the same prefill exe's attention --
// the same one the dense GEMM clamps off). We sort only R_active routes; the tail
// is left stale and the downstream gather/steel/scatter clamp to R_active too, so
// the stale order/idx_sorted entries are never read. When there is nothing to
// clamp the thunk passes num_tokens=R, top_k=1 -> R_active=R (no-op).
//   dims = (R, E, top_k)
kernel void moe_argsort(
    device const int* expert_ids [[buffer(0)]],   // [R]
    device       int* order      [[buffer(1)]],   // [R] out
    device       int* idx_sorted [[buffer(2)]],   // [R] out
    constant int3&    dims        [[buffer(3)]],
    device const int* num_tokens  [[buffer(4)]],   // [1] real prompt length
    uint tid [[thread_index_in_threadgroup]])
{
    const int R = min(dims.x, num_tokens[0] * dims.z);  // R_active
    const int E = dims.y;            // <= 256
    threadgroup atomic_int count[256];
    threadgroup int        offset[256];
    threadgroup atomic_int cursor[256];

    if (int(tid) < E)
        atomic_store_explicit(&count[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Histogram: 256 threads stride over the rows.
    for (int r = int(tid); r < R; r += 256)
        atomic_fetch_add_explicit(&count[expert_ids[r]], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exclusive prefix sum over the E bucket counts -> per-expert base offset.
    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < E; e++) {
            offset[e] = acc;
            acc += atomic_load_explicit(&count[e], memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < E)
        atomic_store_explicit(&cursor[tid], offset[tid], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter each row to the next free slot of its expert's contiguous range.
    for (int r = int(tid); r < R; r += 256) {
        int e = expert_ids[r];
        int pos = atomic_fetch_add_explicit(&cursor[e], 1, memory_order_relaxed);
        order[pos] = r;
        idx_sorted[pos] = e;
    }
}

// Computes the steel-GEMM INDIRECT-dispatch grid {n_tiles, ceil(R_active/BM), 1}
// from the device num_tokens, so the prefill MoE GEMM launches only the active
// route-tiles instead of the full ceil(r_/BM) baked from the padded bucket --
// killing the padded-tile launch overhead the per-tile early-out leaves behind.
// args = (R, n_tiles, top_k, BM)
kernel void moe_steel_grid(
    device const int* num_tokens [[buffer(0)]],   // [1] real prompt length
    constant int4&    args        [[buffer(1)]],   // {R, n_tiles, top_k, BM}
    device       uint* grid_out   [[buffer(2)]],   // {gx, gy, gz}
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    const int R = args.x, n_tiles = args.y, top_k = args.z, BM = args.w;
    const int r_active = min(R, num_tokens[0] * top_k);
    grid_out[0] = (uint)n_tiles;                         // gx = N-tiles (fixed)
    grid_out[1] = (uint)((r_active + BM - 1) / BM);      // gy = active M-tiles
    grid_out[2] = 1u;
}
