#include <metal_stdlib>
using namespace metal;

// Counting sort of the R routed rows by expert id, in a single threadgroup
// (256 threads). E must be in [1, 256]. Produces:
//   order[pos]      = original row index at sorted position `pos`
//   idx_sorted[pos] = expert id at sorted position `pos`
// The permutation GROUPS rows by expert (order within a group is arbitrary).
// This is what the MLX gather q-GEMM (nvfp4_gather_qmm_rhs) needs: its per-BM-tile
// run-walk only reuses an expert's weight across rows that are CONTIGUOUS, so
// the routed rows must be pre-grouped by expert. Used only for the large-R
// (prefill) MoE GEMM, where each expert serves many rows (big weight reuse).
//
// Invalid expert ids are placed in bucket zero with idx_sorted=0; gather_rows
// zeros their inputs and scatter_rows zeros their final outputs. This keeps
// every sorted slot initialized without ever indexing a threadgroup array by an
// untrusted id.
//   dims = (R, E, unused, unused)
kernel void moe_argsort(
    device const int* expert_ids [[buffer(0)]],   // [R]
    device       int* order      [[buffer(1)]],   // [R] out
    device       int* idx_sorted [[buffer(2)]],   // [R] out
    constant int4&    dims        [[buffer(3)]],   // {R, E, unused, unused}
    uint tid [[thread_index_in_threadgroup]])
{
    const int R_total = max(dims.x, 0);
    const int E = dims.y;
    const bool supported = E > 0 && E <= 256;
    const uint R = supported ? uint(R_total) : 0u;
    threadgroup atomic_int count[256];
    threadgroup int        offset[256];
    threadgroup atomic_int cursor[256];

    // The launch is exactly one 256-thread group. Initialize every slot so the
    // unsupported-E and invalid-id paths cannot observe stale threadgroup data.
    atomic_store_explicit(&count[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Histogram: 256 threads stride over the rows.
    for (uint r = tid; r < R; r += 256) {
        const int e = expert_ids[r];
        const int bucket = (e >= 0 && e < E) ? e : 0;
        atomic_fetch_add_explicit(&count[bucket], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exclusive prefix sum over the E bucket counts -> per-expert base offset.
    if (tid == 0 && supported) {
        int acc = 0;
        for (int e = 0; e < E; e++) {
            offset[e] = acc;
            acc += atomic_load_explicit(&count[e], memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (supported && int(tid) < E)
        atomic_store_explicit(&cursor[tid], offset[tid], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter each row to the next free slot of its expert's contiguous range.
    for (uint r = tid; r < R; r += 256) {
        const int e = expert_ids[r];
        const bool valid = e >= 0 && e < E;
        const int bucket = valid ? e : 0;
        const int pos = atomic_fetch_add_explicit(
            &cursor[bucket], 1, memory_order_relaxed);
        order[pos] = int(r);
        idx_sorted[pos] = valid ? e : 0;
    }
}
