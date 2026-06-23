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
//   dims = (R, E)
kernel void moe_argsort(
    device const int* expert_ids [[buffer(0)]],   // [R]
    device       int* order      [[buffer(1)]],   // [R] out
    device       int* idx_sorted [[buffer(2)]],   // [R] out
    constant int2&    dims        [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    const int R = dims.x;
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
