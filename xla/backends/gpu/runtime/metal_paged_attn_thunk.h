/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_PAGED_ATTN_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_PAGED_ATTN_THUNK_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Implements the "zml$paged_attn" custom call on the v2 Metal backend with TWO
// kernels, selected statically the way vllm's paged_ops.cpp does (has_prefill =
// total_q_tokens > num_seqs):
//   * PREFILL (total_q_tokens > num_seqs): vllm-metal's tiled FA-2 simdgroup-
//     matrix kernel, `pagedattention_tiled.metal` copied VERBATIM (vendored +
//     embedded via get_pagedattention_tiled(); only the source-concat preamble —
//     metal_stdlib / using namespace metal / DIVIDE_ROUND_UP / MIN / MAX /
//     typedef bfloat bfloat16_t — is prepended at compile time). Validated vs a
//     CPU causal paged-SDPA reference in
//     metal-xla-docs/scratch/air-ref/2026-06-08-paged_attn_tiled_verbatim.mm.
//   * DECODE (total_q_tokens == num_seqs, head_dim 128, bf16): `fa_vec_paged`,
//     the verified fa_vec matrix-VECTOR decode kernel (metal_flash_attn_thunk)
//     reading K/V through the block table. The tiled kernel's BQ=32 MMA query
//     tile is 31/32 padding for single-token queries; the vector kernel is the
//     right shape. It is VARLEN-causal (per-row cu_seqlens_q binary search), so
//     a small chunked-prefill executable that happens to share the decode's
//     static shape still computes correctly. Falls back to the tiled kernel
//     when the shape doesn't qualify (or METAL_PAGED_VEC=0).
// Both are wired like MetalFlashAttnThunk (embedded MSL -> xcrun metal ->
// LoadKernelWithConstants -> dispatchThreadgroups).
//
// Operand contract (the paged inputs, mirroring ZML's triton paged path):
//   q             bf16/f16 [total_q_tokens, num_heads, head_dim]
//   k_cache       bf16/f16 [num_blocks, block_size, num_kv_heads, head_dim]
//   v_cache       bf16/f16 [num_blocks, block_size, num_kv_heads, head_dim]
//   block_table   u32/i32  [num_seqs, max_num_blocks_per_seq]
//   seq_lens      u32/i32  [num_seqs]            (TOTAL KV length per seq)
//   query_start_len i32    [num_seqs + 1]        (cumulative query lengths)
//   -> out        bf16/f16 [total_q_tokens, num_heads, head_dim]
//
// Static, non-default params (scale/softcapping/sliding_window) default to
// scale=1/sqrt(head_dim), softcapping=0, sliding_window=-1 (Llama). Non-default
// values are carried in scale_/softcapping_/sliding_window_ (set by the matcher
// from backend-config attributes; TODO once Gemma softcap / Mistral SWA need it).
class MetalPagedAttnThunk : public Thunk {
 public:
  MetalPagedAttnThunk(ThunkInfo thunk_info, BufferAllocation::Slice q,
                      Shape q_shape, BufferAllocation::Slice k_cache,
                      Shape k_cache_shape, BufferAllocation::Slice v_cache,
                      Shape v_cache_shape, BufferAllocation::Slice block_table,
                      Shape block_table_shape, BufferAllocation::Slice seq_lens,
                      Shape seq_lens_shape,
                      BufferAllocation::Slice query_start_len,
                      Shape query_start_len_shape, BufferAllocation::Slice out,
                      Shape out_shape, int64_t num_heads, int64_t num_kv_heads,
                      int64_t head_dim, int64_t block_size, int64_t num_seqs,
                      int64_t max_num_blocks_per_seq, int64_t total_q_tokens,
                      float scale, float softcapping, int sliding_window,
                      PrimitiveType element_type);

  MetalPagedAttnThunk(const MetalPagedAttnThunk&) = delete;
  MetalPagedAttnThunk& operator=(const MetalPagedAttnThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

  // Compile-time prewarm (MetalGpuCompiler::RunHloPasses): compile the tiled-
  // prefill metallib + create its PSO; and, when the decode vector path
  // qualifies (head_dim 128, bf16), compile the fa_vec_paged metallib + create
  // ALL THREE nsg PSOs (4/8/16) into Apple's driver pipeline cache. Warming all
  // nsg unconditionally (not seqlen-gated) covers both the depth ramp and the
  // defensive kv==0 -> nsg=16 branch, so no decode token — first or ramp
  // transition — pays `xcrun`/PSO build. Best-effort — failures are swallowed.
  static void Prewarm(stream_executor::StreamExecutor* executor,
                      PrimitiveType dtype, int64_t head_dim, int64_t block_size,
                      int64_t num_kv_heads);

 private:
  // Lazily, on the first execute for a given executor: compile the embedded
  // metallib (process-cached), load the (dtype, head_size, block_size) kernel
  // variant, and stage the static scalar params into small device buffers (the
  // tiled kernel takes them as individual inline `constant&` args at fixed sparse
  // buffer indices). Must hold mu_.
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Decode path setup: compile the fa_vec_paged metallib + stage its kargs, and
  // build the nsg=4 pipeline variant (larger nsg variants are built lazily as
  // the live KV depth crosses the ramp thresholds). Must hold mu_.
  absl::Status EnsureVecDecode(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status EnsureVecVariant(stream_executor::StreamExecutor* executor,
                                int idx) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice q_, k_cache_, v_cache_, block_table_, seq_lens_,
      query_start_len_, out_;
  const Shape q_shape_, k_cache_shape_, v_cache_shape_, block_table_shape_,
      seq_lens_shape_, query_start_len_shape_, out_shape_;
  const int64_t num_heads_, num_kv_heads_, head_dim_, block_size_, num_seqs_,
      max_num_blocks_per_seq_, total_q_tokens_;
  const float scale_, softcapping_;
  const int sliding_window_;
  const PrimitiveType element_type_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);
  size_t shmem_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  int bq_ ABSL_GUARDED_BY(mu_) = 0;
  int num_threads_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t total_q_blocks_ ABSL_GUARDED_BY(mu_) = 0;

  // Static scalar params staged into individual device buffers (one per kernel
  // inline-`constant&` arg), allocated once per executor. The kernel's sparse
  // buffer ABI binds these at indices 8,9,10,13,15,16,17,20,21 (see the .cc).
  stream_executor::DeviceAddressBase p_num_kv_heads_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_scale_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_softcapping_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_max_blocks_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_q_stride_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_kv_block_stride_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_kv_head_stride_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_num_seqs_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_sliding_window_ ABSL_GUARDED_BY(mu_);
  // A throwaway buffer bound at the sparse ABI's unused slots (0,1,6,7,14,18) so
  // the positional KernelArgsPackedArray can reach the kernel's real indices
  // without modifying the verbatim kernel's [[buffer(N)]] numbering.
  stream_executor::DeviceAddressBase dummy_ ABSL_GUARDED_BY(mu_);

  // DECODE (fa_vec_paged) state. Variants by NSG (the per-threadgroup KV split),
  // ramped per-execute from the live max KV depth like the contiguous FA thunk.
  static constexpr int kVecNsgVals[3] = {4, 8, 16};
  bool use_vec_decode_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<uint8_t> vec_lib_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> vec_kernel_by_nsg_[3]
      ABSL_GUARDED_BY(mu_);
  size_t vec_smem_by_nsg_[3] ABSL_GUARDED_BY(mu_) = {0, 0, 0};
  stream_executor::DeviceAddressBase p_vec_args_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_PAGED_ATTN_THUNK_H_
