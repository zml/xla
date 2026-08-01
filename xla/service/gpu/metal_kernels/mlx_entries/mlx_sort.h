// MLX GPU merge sort (MIT License, Copyright (c) 2023 Apple Inc.,
// https://github.com/ml-explore/mlx). Entry source for the mlx_sort bundle:
// the XLA Metal backend's native `metal$sort` primitive (routed via
// RewriteSortToMetalThunk). It replaces the legacy LLVM bitonic sort emitter,
// which cannot lower to valid AIR (it emits NVVM intrinsics for thread/block
// ids).
//
// The upstream text is NOT checked in. The includes below resolve against the
// @mlx archive pinned in //third_party/mlx:workspace.bzl, which
// MetalIncludeRoot() hands to the Metal compiler as -I, so the device-side
// merge-sort building blocks (thread_swap, ThreadSort, BlockMergeSort,
// KernelMultiBlockMergeSort) are upstream's own bytes rather than a copy of
// them. To change upstream's bytes, add a patch to //third_party/mlx:series.bzl.
//
// The includes are upstream's own prologue, copied from
// mlx/backend/metal/kernels/sort.metal:4-7. sort.h has no #pragma once and
// opens on a bare `using namespace metal;`, so it only compiles behind the
// prologue its own .metal file establishes.
//
// Everything below the includes is ours -- only the thin `[[kernel]]` entry
// points are, and they differ from upstream's deliberately:
//   * Only the MULTI-BLOCK path is exposed (xla_mb_block_sort /
//     xla_mb_block_partition / xla_mb_block_merge). n_blocks==1 degenerates to a
//     single xla_mb_block_sort that sorts the whole row in one threadgroup -- so
//     one code path covers any n.
//   * Contiguous last-axis [rows, n] only: MLX's nc_dim/nc_shape/nc_strides
//     buffers are dropped and the row base is `tid.y * size_sorted_axis`
//     (sorted-axis stride == 1).
//   * All three entries are ARG_SORT (they emit sorted VALUES and the permuted
//     INDICES together -- topk/argsort need both), and are templated on CompareOp
//     so DESCENDING is a real instantiation (upstream mb_block_sort /
//     mb_block_partition hardcode ascending).
//
// Stability: the merge predicate keeps the A-side (earlier block == lower
// original index) element on ties, so equal keys retain input order == XLA
// is_stable (index-ascending ties), for both directions. Padding lanes use a
// NaN sentinel, which the comparators sink to the END of the sort in both
// directions -- so a slice<=n never sees padding.
#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sort.h"
// clang-format on

// Sorted-axis stride is always 1 for the contiguous [rows, n] case; a `constant`
// so it can bind to the `const constant int&` stride parameter.
constant constexpr const int one_helper = 1;


///////////////////////////////////////////////////////////////////////////////
// Comparators
///////////////////////////////////////////////////////////////////////////////

// XLA DELTA: rename-fork of upstream's LessThan (sort.h:39). Upstream's is
// already in scope via the include and is behaviourally the same for every type
// we instantiate, but ours reaches `init` directly instead of through
// Init<T>::v, so keeping our own body keeps the sort's emitted code -- and the
// greedy golden -- provably unmoved by this migration. It is a rename rather
// than a patch because a rename-fork cannot collide on a bump and cannot
// silently un-apply. Fold it into upstream's if the Init<T> indirection is ever
// worth the churn.
//
// XlaGreaterThan has no upstream counterpart: it is a real capability gap.
// Upstream's mb_block_partition hardcodes ascending, and descending is a
// direction XLA's Sort must express. It is renamed alongside XlaLessThan so the
// pair reads as one thing that is ours.
//
// Padding sentinel sinks to the END of the sort in both directions: quiet NaN
// for floats (via the NaN branch), type-max for ascending integers, type-min for
// descending integers. Keys can be float (bf16/f16/f32) or int (8/16/32-bit,
// signed/unsigned); the NaN branch is compiled out for integer keys.
template <typename T>
struct XlaLessThan {
  static constexpr constant T init =
      metal::is_floating_point_v<T>
          ? static_cast<T>(metal::numeric_limits<float>::quiet_NaN())
          : metal::numeric_limits<T>::max();
  METAL_FUNC bool operator()(T a, T b) const {
    if constexpr (metal::is_floating_point_v<T>) {
      bool an = metal::isnan(a), bn = metal::isnan(b);
      if (an | bn) return (!an) & bn;  // non-NaN before NaN => NaN sinks last
    }
    return a < b;
  }
};

template <typename T>
struct XlaGreaterThan {
  static constexpr constant T init =
      metal::is_floating_point_v<T>
          ? static_cast<T>(metal::numeric_limits<float>::quiet_NaN())
          : metal::numeric_limits<T>::min();
  METAL_FUNC bool operator()(T a, T b) const {
    if constexpr (metal::is_floating_point_v<T>) {
      bool an = metal::isnan(a), bn = metal::isnan(b);
      if (an | bn) return (!an) & bn;  // NaN sinks last in descending too
    }
    return a > b;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Entry points (ours): contiguous [rows, n], dual-output, CompareOp-templated.
// IdxT is uint (uint32). Grid dims are threadgroup counts.
///////////////////////////////////////////////////////////////////////////////

// Per-block sort. Grid = (n_blocks, n_rows, 1), group = (BLOCK_THREADS, 1, 1).
// When n_blocks == 1 the caller points out_vals/out_idxs at the real output and
// this alone is the full sort.
template <
    typename ValT,
    short BLOCK_THREADS,
    short N_PER_THREAD,
    typename CompareOp>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void xla_mb_block_sort(
    const device ValT* inp [[buffer(0)]],
    device ValT* out_vals [[buffer(1)]],
    device uint* out_idxs [[buffer(2)]],
    const constant int& size_sorted_axis [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using sort_kernel = KernelMultiBlockMergeSort<
      ValT,
      uint,
      true,
      BLOCK_THREADS,
      N_PER_THREAD,
      CompareOp>;

  inp += tid.y * size_sorted_axis;
  out_vals += tid.y * size_sorted_axis;
  out_idxs += tid.y * size_sorted_axis;

  threadgroup ValT tgp_vals[sort_kernel::N_PER_BLOCK];
  threadgroup uint tgp_idxs[sort_kernel::N_PER_BLOCK];

  sort_kernel::block_sort(
      inp,
      out_vals,
      out_idxs,
      size_sorted_axis,
      one_helper,
      tgp_vals,
      tgp_idxs,
      tid,
      lid);
}

// Merge partition points. Grid = (1, n_rows, 1),
// group = (min(n_blocks+1, 1024), 1, 1). Requires n_blocks + 1 <= 1024 so the
// per-row partition stride (tgp_dims.x) matches the merge's (num_tiles + 1).
template <
    typename ValT,
    short BLOCK_THREADS,
    short N_PER_THREAD,
    typename CompareOp>
[[kernel]] void xla_mb_block_partition(
    device uint* block_partitions [[buffer(0)]],
    const device ValT* dev_vals [[buffer(1)]],
    const device uint* dev_idxs [[buffer(2)]],
    const constant int& size_sorted_axis [[buffer(3)]],
    const constant int& merge_tiles [[buffer(4)]],
    const constant int& n_blocks [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_dims [[threads_per_threadgroup]]) {
  using sort_kernel = KernelMultiBlockMergeSort<
      ValT,
      uint,
      true,
      BLOCK_THREADS,
      N_PER_THREAD,
      CompareOp>;

  block_partitions += tid.y * tgp_dims.x;
  dev_vals += tid.y * size_sorted_axis;
  dev_idxs += tid.y * size_sorted_axis;

  for (int i = lid.x; i <= n_blocks; i += tgp_dims.x) {
    // Find location in merge step
    int merge_group = i / merge_tiles;
    int merge_lane = i % merge_tiles;

    int sort_sz = sort_kernel::N_PER_BLOCK * merge_tiles;
    int sort_st = sort_kernel::N_PER_BLOCK * merge_tiles * merge_group;

    int A_st = min(size_sorted_axis, sort_st);
    int A_ed = min(size_sorted_axis, sort_st + sort_sz / 2);
    int B_st = A_ed;
    int B_ed = min(size_sorted_axis, B_st + sort_sz / 2);

    int partition_at = min(B_ed - A_st, sort_kernel::N_PER_BLOCK * merge_lane);
    int partition = sort_kernel::merge_partition(
        dev_vals + A_st,
        dev_vals + B_st,
        A_ed - A_st,
        B_ed - B_st,
        partition_at);

    block_partitions[i] = A_st + partition;
  }
}

// One multi-block merge pass. Grid = (n_blocks, n_rows, 1),
// group = (BLOCK_THREADS, 1, 1). num_tiles == n_blocks.
template <
    typename ValT,
    short BLOCK_THREADS,
    short N_PER_THREAD,
    typename CompareOp>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
xla_mb_block_merge(
    const device uint* block_partitions [[buffer(0)]],
    const device ValT* dev_vals_in [[buffer(1)]],
    const device uint* dev_idxs_in [[buffer(2)]],
    device ValT* dev_vals_out [[buffer(3)]],
    device uint* dev_idxs_out [[buffer(4)]],
    const constant int& size_sorted_axis [[buffer(5)]],
    const constant int& merge_tiles [[buffer(6)]],
    const constant int& num_tiles [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using sort_kernel = KernelMultiBlockMergeSort<
      ValT,
      uint,
      true,
      BLOCK_THREADS,
      N_PER_THREAD,
      CompareOp>;

  using block_sort_t = typename sort_kernel::block_merge_sort_t;

  block_partitions += tid.y * (num_tiles + 1);
  dev_vals_in += tid.y * size_sorted_axis;
  dev_idxs_in += tid.y * size_sorted_axis;
  dev_vals_out += tid.y * size_sorted_axis;
  dev_idxs_out += tid.y * size_sorted_axis;

  int block_idx = tid.x;
  int merge_group = block_idx / merge_tiles;
  int sort_st = sort_kernel::N_PER_BLOCK * merge_tiles * merge_group;
  int sort_sz = sort_kernel::N_PER_BLOCK * merge_tiles;
  int sort_md = sort_kernel::N_PER_BLOCK * block_idx - sort_st;

  int A_st = block_partitions[block_idx + 0];
  int A_ed = block_partitions[block_idx + 1];
  int B_st = min(size_sorted_axis, 2 * sort_st + sort_sz / 2 + sort_md - A_st);
  int B_ed = min(
      size_sorted_axis,
      2 * sort_st + sort_sz / 2 + sort_md + sort_kernel::N_PER_BLOCK - A_ed);

  if ((block_idx % merge_tiles) == merge_tiles - 1) {
    A_ed = min(size_sorted_axis, sort_st + sort_sz / 2);
    B_ed = min(size_sorted_axis, sort_st + sort_sz);
  }

  int A_sz = A_ed - A_st;
  int B_sz = B_ed - B_st;

  // Load from global memory
  thread ValT thread_vals[N_PER_THREAD];
  thread uint thread_idxs[N_PER_THREAD];
  for (int i = 0; i < N_PER_THREAD; i++) {
    int idx = BLOCK_THREADS * i + lid.x;
    if (idx < (A_sz + B_sz)) {
      thread_vals[i] = (idx < A_sz) ? dev_vals_in[A_st + idx]
                                    : dev_vals_in[B_st + idx - A_sz];
      thread_idxs[i] = (idx < A_sz) ? dev_idxs_in[A_st + idx]
                                    : dev_idxs_in[B_st + idx - A_sz];
    } else {
      thread_vals[i] = CompareOp::init;
      thread_idxs[i] = 0;
    }
  }

  // Write to shared memory
  threadgroup ValT tgp_vals[sort_kernel::N_PER_BLOCK];
  threadgroup uint tgp_idxs[sort_kernel::N_PER_BLOCK];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < N_PER_THREAD; i++) {
    int idx = BLOCK_THREADS * i + lid.x;
    tgp_vals[idx] = thread_vals[i];
    tgp_idxs[idx] = thread_idxs[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Merge
  int sort_md_local = min(A_sz + B_sz, N_PER_THREAD * int(lid.x));

  int A_st_local = block_sort_t::merge_partition(
      tgp_vals, tgp_vals + A_sz, A_sz, B_sz, sort_md_local);
  int A_ed_local = A_sz;

  int B_st_local = sort_md_local - A_st_local;
  int B_ed_local = B_sz;

  int A_sz_local = A_ed_local - A_st_local;
  int B_sz_local = B_ed_local - B_st_local;

  // Do merge
  block_sort_t::merge_step(
      tgp_vals + A_st_local,
      tgp_vals + A_ed_local + B_st_local,
      tgp_idxs + A_st_local,
      tgp_idxs + A_ed_local + B_st_local,
      A_sz_local,
      B_sz_local,
      thread_vals,
      thread_idxs);

  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < N_PER_THREAD; ++i) {
    int idx = lid.x * N_PER_THREAD;
    tgp_vals[idx + i] = thread_vals[i];
    tgp_idxs[idx + i] = thread_idxs[i];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Write output
  int base_idx = tid.x * sort_kernel::N_PER_BLOCK;
  for (int i = lid.x; i < sort_kernel::N_PER_BLOCK; i += BLOCK_THREADS) {
    int idx = base_idx + i;
    if (idx < size_sorted_axis) {
      dev_vals_out[idx] = tgp_vals[i];
      dev_idxs_out[idx] = tgp_idxs[i];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Instantiations: {float, half, bfloat16_t, int, short, char, uint, ushort,
// uchar} x {asc, desc}. Fixed bn=512, tn=4 (N_PER_BLOCK = 2048), MLX's choice
// for large n; correct for any n via padding. Names are matched by the host
// (MetalSortThunk) as xla_sort_{block,part,merge}_<dtype>_<dir>.
//
// instantiate_kernel comes from upstream's defines.h (reached via utils.h); our
// own identical copy of it is gone with the flattened bundle.
///////////////////////////////////////////////////////////////////////////////

#define instantiate_xla_sort_dir(dname, dtype, cmp)                            \
  instantiate_kernel(                                                          \
      "xla_sort_block_" #dtype "_" dname, xla_mb_block_sort, dtype, 512, 4,    \
      cmp<dtype>)                                                              \
  instantiate_kernel(                                                          \
      "xla_sort_part_" #dtype "_" dname, xla_mb_block_partition, dtype, 512,   \
      4, cmp<dtype>)                                                           \
  instantiate_kernel(                                                          \
      "xla_sort_merge_" #dtype "_" dname, xla_mb_block_merge, dtype, 512, 4,   \
      cmp<dtype>)

#define instantiate_xla_sort_dtype(dtype)      \
  instantiate_xla_sort_dir("asc", dtype, XlaLessThan) \
  instantiate_xla_sort_dir("desc", dtype, XlaGreaterThan)

instantiate_xla_sort_dtype(float)
instantiate_xla_sort_dtype(half)
instantiate_xla_sort_dtype(bfloat16_t)
instantiate_xla_sort_dtype(int)
instantiate_xla_sort_dtype(short)
instantiate_xla_sort_dtype(char)
instantiate_xla_sort_dtype(uint)
instantiate_xla_sort_dtype(ushort)
instantiate_xla_sort_dtype(uchar)
