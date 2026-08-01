// Vendored from MLX (github.com/ml-explore/mlx, MIT), backend/metal/kernels/sort.h.
//
// MLX GPU merge sort (thread -> threadgroup -> multi-block merge), used here as
// the XLA Metal backend's native `metal$sort` primitive (routed via
// RewriteSortToMetalThunk). It replaces the legacy LLVM bitonic sort emitter,
// which cannot lower to valid AIR (it emits NVVM intrinsics for thread/block
// ids). Only the thin `[[kernel]]` entry points at the bottom are ours; the
// device-side merge-sort building blocks (thread_swap, ThreadSort,
// BlockMergeSort, KernelMultiBlockMergeSort) are copied VERBATIM from MLX.
//
// Deltas from upstream MLX, all in the entry points (not the algorithm):
//   * Only the MULTI-BLOCK path is exposed (mb_block_sort / mb_block_partition /
//     mb_block_merge). n_blocks==1 degenerates to a single mb_block_sort that
//     sorts the whole row in one threadgroup -- so one code path covers any n.
//   * Contiguous last-axis [rows, n] only: MLX's nc_dim/nc_shape/nc_strides
//     buffers are dropped and the row base is `tid.y * size_sorted_axis`
//     (sorted-axis stride == 1).
//   * All three entries are ARG_SORT (they emit sorted VALUES and the permuted
//     INDICES together -- topk/argsort need both), and are templated on CompareOp
//     so DESCENDING is a real instantiation (upstream mb_block_sort /
//     mb_block_partition hardcode ascending LessThan).
//
// Stability: the merge predicate keeps the A-side (earlier block == lower
// original index) element on ties, so equal keys retain input order == XLA
// is_stable (index-ascending ties), for both directions. Padding lanes use a
// NaN sentinel, which the comparators sink to the END of the sort in both
// directions -- so a slice<=n never sees padding.

#include <metal_stdlib>

using namespace metal;

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif
#define MLX_MTL_CONST static constant constexpr const
#define MLX_MTL_LOOP_UNROLL _Pragma("clang loop unroll(full)")

// Metal 3.1+ native bfloat; MLX names it bfloat16_t.
typedef bfloat bfloat16_t;

// Sorted-axis stride is always 1 for the contiguous [rows, n] case; a `constant`
// so it can bind to the `const constant int&` stride parameter.
constant constexpr const int one_helper = 1;

///////////////////////////////////////////////////////////////////////////////
// Comparators (key type is always floating: bf16/f16/f32 logits)
///////////////////////////////////////////////////////////////////////////////

// Padding sentinel sinks to the END of the sort in both directions: quiet NaN
// for floats (via the NaN branch), type-max for ascending integers, type-min for
// descending integers. Keys can be float (bf16/f16/f32) or int (8/16/32-bit,
// signed/unsigned); the NaN branch is compiled out for integer keys.
template <typename T>
struct LessThan {
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
struct GreaterThan {
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
// Thread-level sort (MLX verbatim)
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC void thread_swap(thread T& a, thread T& b) {
  T w = a;
  a = b;
  b = w;
}

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    short N_PER_THREAD,
    typename CompareOp>
struct ThreadSort {
  static METAL_FUNC void sort(
      thread ValT (&vals)[N_PER_THREAD],
      thread IdxT (&idxs)[N_PER_THREAD]) {
    CompareOp op;
    MLX_MTL_LOOP_UNROLL
    for (short i = 0; i < N_PER_THREAD; ++i) {
      MLX_MTL_LOOP_UNROLL
      for (short j = i & 1; j < N_PER_THREAD - 1; j += 2) {
        if (op(vals[j + 1], vals[j])) {
          thread_swap(vals[j + 1], vals[j]);
          if (ARG_SORT) {
            thread_swap(idxs[j + 1], idxs[j]);
          }
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Threadgroup-level sort (MLX verbatim)
///////////////////////////////////////////////////////////////////////////////

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    short BLOCK_THREADS,
    short N_PER_THREAD,
    typename CompareOp>
struct BlockMergeSort {
  using thread_sort_t =
      ThreadSort<ValT, IdxT, ARG_SORT, N_PER_THREAD, CompareOp>;
  static METAL_FUNC int merge_partition(
      const threadgroup ValT* As,
      const threadgroup ValT* Bs,
      short A_sz,
      short B_sz,
      short sort_md) {
    CompareOp op;

    short A_st = max(0, sort_md - B_sz);
    short A_ed = min(sort_md, A_sz);

    while (A_st < A_ed) {
      short md = A_st + (A_ed - A_st) / 2;
      auto a = As[md];
      auto b = Bs[sort_md - 1 - md];

      if (op(b, a)) {
        A_ed = md;
      } else {
        A_st = md + 1;
      }
    }

    return A_ed;
  }

  static METAL_FUNC void merge_step(
      const threadgroup ValT* As,
      const threadgroup ValT* Bs,
      const threadgroup IdxT* As_idx,
      const threadgroup IdxT* Bs_idx,
      short A_sz,
      short B_sz,
      thread ValT (&vals)[N_PER_THREAD],
      thread IdxT (&idxs)[N_PER_THREAD]) {
    CompareOp op;
    short a_idx = 0;
    short b_idx = 0;

    for (int i = 0; i < N_PER_THREAD; ++i) {
      auto a = (a_idx < A_sz) ? As[a_idx] : ValT(CompareOp::init);
      auto b = (b_idx < B_sz) ? Bs[b_idx] : ValT(CompareOp::init);
      bool pred = (b_idx < B_sz) && (a_idx >= A_sz || op(b, a));

      vals[i] = pred ? b : a;
      if (ARG_SORT) {
        if (pred) {
          idxs[i] = Bs_idx[b_idx];
        } else {
          idxs[i] = (a_idx < A_sz) ? As_idx[a_idx] : IdxT(0);
        }
      }

      b_idx += short(pred);
      a_idx += short(!pred);
    }
  }

  static METAL_FUNC void sort(
      threadgroup ValT* tgp_vals [[threadgroup(0)]],
      threadgroup IdxT* tgp_idxs [[threadgroup(1)]],
      int size_sorted_axis,
      uint3 lid [[thread_position_in_threadgroup]]) {
    // Get thread location
    int idx = lid.x * N_PER_THREAD;

    // Load from shared memory
    thread ValT thread_vals[N_PER_THREAD];
    thread IdxT thread_idxs[N_PER_THREAD];
    for (int i = 0; i < N_PER_THREAD; ++i) {
      thread_vals[i] = tgp_vals[idx + i];
      if (ARG_SORT) {
        thread_idxs[i] = tgp_idxs[idx + i];
      }
    }

    // Per thread sort
    if (idx < size_sorted_axis) {
      thread_sort_t::sort(thread_vals, thread_idxs);
    }

    // Do merges using threadgroup memory
    for (int merge_threads = 2; merge_threads <= BLOCK_THREADS;
         merge_threads *= 2) {
      // Update threadgroup memory
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < N_PER_THREAD; ++i) {
        tgp_vals[idx + i] = thread_vals[i];
        if (ARG_SORT) {
          tgp_idxs[idx + i] = thread_idxs[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Find location in merge step
      int merge_group = lid.x / merge_threads;
      int merge_lane = lid.x % merge_threads;

      int sort_sz = N_PER_THREAD * merge_threads;
      int sort_st = N_PER_THREAD * merge_threads * merge_group;

      // As = tgp_vals[A_st:A_ed] is sorted
      // Bs = tgp_vals[B_st:B_ed] is sorted
      int A_st = sort_st;
      int A_ed = sort_st + sort_sz / 2;
      int B_st = sort_st + sort_sz / 2;
      int B_ed = sort_st + sort_sz;

      const threadgroup ValT* As = tgp_vals + A_st;
      const threadgroup ValT* Bs = tgp_vals + B_st;
      int A_sz = A_ed - A_st;
      int B_sz = B_ed - B_st;

      // Find a partition of merge elements
      //  Ci = merge(As[partition:], Bs[sort_md - partition:])
      //       of size N_PER_THREAD for each merge lane i
      //  C = [Ci] is sorted
      int sort_md = N_PER_THREAD * merge_lane;
      int partition = merge_partition(As, Bs, A_sz, B_sz, sort_md);

      As += partition;
      Bs += sort_md - partition;

      A_sz -= partition;
      B_sz -= sort_md - partition;

      const threadgroup IdxT* As_idx =
          ARG_SORT ? tgp_idxs + A_st + partition : nullptr;
      const threadgroup IdxT* Bs_idx =
          ARG_SORT ? tgp_idxs + B_st + sort_md - partition : nullptr;

      // Merge starting at the partition and store results in thread registers
      merge_step(As, Bs, As_idx, Bs_idx, A_sz, B_sz, thread_vals, thread_idxs);
    }

    // Write out to shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_PER_THREAD; ++i) {
      tgp_vals[idx + i] = thread_vals[i];
      if (ARG_SORT) {
        tgp_idxs[idx + i] = thread_idxs[i];
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Multi-block merge sort (MLX verbatim)
///////////////////////////////////////////////////////////////////////////////

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    short BLOCK_THREADS,
    short N_PER_THREAD,
    typename CompareOp = LessThan<ValT>>
struct KernelMultiBlockMergeSort {
  using block_merge_sort_t = BlockMergeSort<
      ValT,
      IdxT,
      ARG_SORT,
      BLOCK_THREADS,
      N_PER_THREAD,
      CompareOp>;

  MLX_MTL_CONST short N_PER_BLOCK = BLOCK_THREADS * N_PER_THREAD;

  static METAL_FUNC void block_sort(
      const device ValT* inp,
      device ValT* out_vals,
      device IdxT* out_idxs,
      const constant int& size_sorted_axis,
      const constant int& stride_sorted_axis,
      threadgroup ValT* tgp_vals,
      threadgroup IdxT* tgp_idxs,
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]]) {
    // tid.y tells us the segment index
    int base_idx = tid.x * N_PER_BLOCK;

    // Copy into threadgroup memory
    for (short i = lid.x; i < N_PER_BLOCK; i += BLOCK_THREADS) {
      int idx = base_idx + i;
      tgp_vals[i] = idx < size_sorted_axis ? inp[idx * stride_sorted_axis]
                                           : ValT(CompareOp::init);
      tgp_idxs[i] = idx;
    }

    // Sort elements within the block
    threadgroup_barrier(mem_flags::mem_threadgroup);

    block_merge_sort_t::sort(tgp_vals, tgp_idxs, size_sorted_axis, lid);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output
    for (int i = lid.x; i < N_PER_BLOCK; i += BLOCK_THREADS) {
      int idx = base_idx + i;
      if (idx < size_sorted_axis) {
        out_vals[idx] = tgp_vals[i];
        out_idxs[idx] = tgp_idxs[i];
      }
    }
  }

  static METAL_FUNC int merge_partition(
      const device ValT* As,
      const device ValT* Bs,
      int A_sz,
      int B_sz,
      int sort_md) {
    CompareOp op;

    int A_st = max(0, sort_md - B_sz);
    int A_ed = min(sort_md, A_sz);

    while (A_st < A_ed) {
      int md = A_st + (A_ed - A_st) / 2;
      auto a = As[md];
      auto b = Bs[sort_md - 1 - md];

      if (op(b, a)) {
        A_ed = md;
      } else {
        A_st = md + 1;
      }
    }

    return A_ed;
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
// Instantiations: {float, half, bfloat16_t} x {asc, desc}. Fixed bn=512, tn=4
// (N_PER_BLOCK = 2048), MLX's choice for large n; correct for any n via padding.
// Names are matched by the host (MetalSortThunk) as
//   xla_sort_{block,part,merge}_<dtype>_<dir>
///////////////////////////////////////////////////////////////////////////////

#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

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
  instantiate_xla_sort_dir("asc", dtype, LessThan) \
  instantiate_xla_sort_dir("desc", dtype, GreaterThan)

instantiate_xla_sort_dtype(float)
instantiate_xla_sort_dtype(half)
instantiate_xla_sort_dtype(bfloat16_t)
instantiate_xla_sort_dtype(int)
instantiate_xla_sort_dtype(short)
instantiate_xla_sort_dtype(char)
instantiate_xla_sort_dtype(uint)
instantiate_xla_sort_dtype(ushort)
instantiate_xla_sort_dtype(uchar)
