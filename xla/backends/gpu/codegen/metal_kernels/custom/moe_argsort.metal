#include <metal_stdlib>
using namespace metal;

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

    atomic_store_explicit(&count[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = tid; r < R; r += 256) {
        const int e = expert_ids[r];
        const int bucket = (e >= 0 && e < E) ? e : 0;
        atomic_fetch_add_explicit(&count[bucket], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
