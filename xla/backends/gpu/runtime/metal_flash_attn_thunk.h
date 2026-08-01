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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_FLASH_ATTN_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_FLASH_ATTN_THUNK_H_

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

namespace xla {
namespace gpu {

// Implements the "zml$flash_attn" custom call on the v2 Metal backend: a fused
// GQA flash-attention DECODE (single-query, M=1) kernel that replaces ZML's
// sdpa (the 2 MPSGraph batched dots + softmax). The fast path is the KV-PARALLEL
// flash decode kernel `fa_vec` (+ `fa_vec_reduce`) — llama.cpp's
// kernel_flash_attn_ext_vec extracted into a small self-contained metallib and
// SPECIALIZED to ZML's decode case (bf16 q/k/v/out read directly; causality from
// the device token_index computed inline; no mask/convert glue dispatches). It
// distributes the KV cache across NSG simdgroups (shared-memory combine). NSG is
// chosen PER-EXECUTE from the LIVE KV length (the raw host-set token position,
// read host-side at encode time with no GPU sync): nsg ramps 4->8->16 as context
// grows, so decode does not collapse at long context, while low context keeps the
// proven nsg=4. (An NWG=32 split-K + `fa_vec_reduce` path exists as a gated
// experimental lever via METAL_FA_NWG=32, but measured slower than the nwg=1 nsg
// ramp for contexts <= 8192 — the extra reduce dispatch + f32 partial traffic
// outweigh the parallelism in our per-op dispatch model.) Overridable via
// METAL_FA_NSG / METAL_FA_NWG. A serial single-simdgroup kernel remains as the
// fallback for head_dim != 128 or unsupported dtypes.
// Validated vs a CPU causal reference in
// metal-xla-docs/scratch/air-ref/2026-06-06-fa_vec_specialized.mm.
//
// Operand contract (matches ZML sdpa for decode):
//   q  bf16 [n_q=n_kv*groups, 1, hd]     (the query for this token)
//   k  bf16 [n_kv, seqlen, hd]           (KV cache keys)
//   v  bf16 [n_kv, seqlen, hd]           (KV cache values)
//   tok s32/u32 scalar                   (token_index; key kpos valid iff <= tok)
//   -> out bf16 [n_q, 1, hd]             (same shape/dtype sdpa returns)
class MetalFlashAttnThunk : public Thunk {
 public:
  MetalFlashAttnThunk(ThunkInfo thunk_info, BufferAllocation::Slice q,
                      Shape q_shape, BufferAllocation::Slice k, Shape k_shape,
                      BufferAllocation::Slice v, Shape v_shape,
                      BufferAllocation::Slice tok, Shape tok_shape,
                      BufferAllocation::Slice out, Shape out_shape,
                      int64_t n_kv, int64_t n_groups, int64_t seqlen,
                      int64_t head_dim, bool kv_position_major,
                      bool kv_full_cache, BufferAllocation::Slice layer,
                      Shape layer_shape, BufferAllocation::Slice num_tokens,
                      Shape num_tokens_shape, bool tok_host_coherent);

  MetalFlashAttnThunk(const MetalFlashAttnThunk&) = delete;
  MetalFlashAttnThunk& operator=(const MetalFlashAttnThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

  // Precompile, at HLO-compile time (when the compiler detects the zml$flash_attn
  // custom call and still has a live executor), BOTH the metallib (process-
  // globally cached) AND the GPU pipeline state (newComputePipelineState, cached by
  // Apple's driver), so the first execute pays neither (compile + ~70ms PSO).
  // is_prefill selects kernel_flash_attn_ext (prefill, one PSO) vs fa_vec (decode);
  // for decode it warms every nsg the adaptive ramp can reach for this seqlen
  // (nsg=4 always, +8 if seqlen>1024, +16 if seqlen>2048), so no decode token —
  // first or ramp transition — pays PSO creation. kv_pos_stride must match the
  // thunk's so the warmed PSO's function constants match. Best-effort — failures
  // are swallowed (the thunk rebuilds at execute).
  static void PrewarmPipeline(stream_executor::StreamExecutor* executor,
                              bool is_prefill, int64_t kv_pos_stride,
                              int64_t seqlen, int64_t head_dim);

 private:
  // Lazily compile + cache the fa_vec pipeline variants (from favec_lib_) the
  // first time each is selected, so a decode never pays for a variant it doesn't
  // use. EnsureFaVecMain builds the nwg=1 variant at kNsgVals[idx]. No-op if
  // already built. Must hold mu_.
  absl::Status EnsureFaVecMain(stream_executor::StreamExecutor* executor, int idx)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Build the head-contiguous split-K decode pipeline (fa_vec_hc) + its reduce.
  // fa_vec_hc maps one simdgroup per KV head (nsg=n_kv) so the heads' reads tile
  // one CONTIGUOUS position run — recovering sequential KV bandwidth from the
  // position-major cache (the per-query-head fa_vec reads each head strided).
  // Must hold mu_.
  absl::Status EnsureFaVecHC(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Build the prefill (q_len>1) pipeline from prefill_lib_ (the simdgroup-matrix
  // kernel_flash_attn_ext, specialized bf16/hd128/causal). One PSO (nsg=4). Must
  // hold mu_.
  absl::Status EnsurePrefill(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice q_, k_, v_, tok_, out_;
  const Shape q_shape_, k_shape_, v_shape_, tok_shape_, out_shape_;
  const int64_t n_kv_, n_groups_, seqlen_, head_dim_;
  // True when K/V are fed in the cache-native position-major layout
  // ([seqlen, n_kv, hd], i.e. operand layout {2,0,1}) instead of head-major
  // ([n_kv, seqlen, hd], {2,1,0}) — set by RelaxFlashAttnKVLayout so the ZML
  // transpose folds to a free bitcast (no per-token KV-cache copy).
  const bool kv_position_major_;
  // True when K/V are fed as the WHOLE [n_layer, seqlen, n_kv, hd] cache + a
  // `layer` index operand (RelaxFlashAttnKVLayout also folded away the
  // per-token dynamic-slice that extracted the current layer). The kernel reads
  // layer `layer_[0]` in place. False = single-layer operand (layer offset 0).
  const bool kv_full_cache_;
  const BufferAllocation::Slice layer_;
  const Shape layer_shape_;
  // PREFILL ONLY (q_len>1): the real prompt-token count (a raw u32 entry param,
  // host-valid at encode). When present, the prefill dispatch clamps its query-row
  // grid to num_tokens instead of the padded seqlen — a short prompt then skips the
  // padding rows (correct because the model is causal + position-wise, so rows
  // >= num_tokens never feed the real output). Empty/has_num_tokens_=false for
  // decode and for prefill calls that predate the operand (falls back to seqlen).
  const BufferAllocation::Slice num_tokens_;
  const Shape num_tokens_shape_;
  const bool has_num_tokens_;
  // True iff `tok` (the decode token-position operand) is a host-staged entry
  // parameter, so the thunk may read it host-side at encode to pick the decode
  // perf variant (nsg/HC). False = `tok` is a GPU-produced value (the layout-only
  // relax fallback / no raw param); reading it host-side would race the producer
  // on Metal, so the thunk uses a static default (kv = seqlen) and never derefs it.
  // The choice is perf-only: fa_vec is numerically identical across nsg (its KV
  // ceiling is the static seqlen; causality is the on-device tok[0]). Set by the
  // emitter from operand(3)'s producer; see RelaxFlashAttnKVLayout/TraceRawTokenParam.
  const bool tok_host_coherent_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;

  // KV-parallel fast path: the specialized fa_vec (+ fa_vec_reduce for nwg=32).
  // Selected when head_dim == 128, bf16 q/out, seqlen % 32 == 0.
  //
  // Per-execute adaptive selection: each decode step reads the LIVE KV length
  // host-side (the raw token-position param, host-valid at encode time — no GPU
  // sync) and dispatches the kernel variant matching the ACTUAL context. nsg/nwg
  // are baked function constants, so each variant is its own pipeline, compiled
  // from one shared metallib (favec_lib_). Variants are built LAZILY on first
  // selection and cached — a small-context decode only ever builds nsg=4 (the
  // same single-PSO first-token cost as before; no startup regression), while
  // larger contexts amortize a variant's one-time build over many tokens. Two
  // tracks:
  //  * nwg=1 ramp (fa_main_by_nsg_): one pipeline per nsg in kNsgVals. A SINGLE
  //    dispatch; more simdgroups = more KV parallelism as context grows. Covers
  //    the low/mid context that dominates, with no reduce pass.
  bool use_fa_vec_ ABSL_GUARDED_BY(mu_) = false;
  static constexpr int kNsgVals[5] = {1, 2, 4, 8, 16};
  std::vector<uint8_t> favec_lib_ ABSL_GUARDED_BY(mu_);  // shared fa_vec metallib
  int64_t kv_pos_stride_ ABSL_GUARDED_BY(mu_) = 0;       // K/V position stride (elems)
  std::unique_ptr<stream_executor::Kernel> fa_main_by_nsg_[5] ABSL_GUARDED_BY(mu_);
  size_t smem_by_nsg_[5] ABSL_GUARDED_BY(mu_) = {};
  std::unique_ptr<stream_executor::Kernel> fa_reduce_ ABSL_GUARDED_BY(mu_);
  //  * head-contiguous split-K (fa_main_hc_ + fa_reduce_): the long-context lever.
  //    One simdgroup per KV head (nsg=n_kv), grid (1, n_groups, 32); the n_kv heads
  //    of a position are read by the n_kv simdgroups together -> one CONTIGUOUS
  //    load, matching llama's head-major bandwidth out of OUR position-major cache.
  //    Reuses the position-split reduce (tmp_/fa_reduce_/kargs_reduce_).
  std::unique_ptr<stream_executor::Kernel> fa_main_hc_ ABSL_GUARDED_BY(mu_);
  size_t smem_hc_ ABSL_GUARDED_BY(mu_) = 0;
  // PREFILL fast path (q_len>1): the specialized kernel_flash_attn_ext (FA-2,
  // simdgroup-matrix). Selected when q_len>1 && head_dim==128 && bf16 && seqlen%64==0.
  // A single PSO (nsg=4) built on the first prefill execute (one compile + one
  // PSO, same first-call cost shape as the decode nsg=4 build). Parallelism comes
  // from the Q-row blocks (grid dim0 = ceil(q_len/8)), so there is no
  // per-execute nsg ramp and no host token read — the kernel derives the 2D
  // causal mask on-device from tok[0] + the query-row index.
  bool use_prefill_ ABSL_GUARDED_BY(mu_) = false;
  // The simdgroup-matrix fa_ext kernel is the sole prefill kernel; it covers any
  // head_dim % 16 == 0 (one compiled specialization per hd via PrefillSubs).
  std::vector<uint8_t> prefill_lib_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> fa_prefill_ ABSL_GUARDED_BY(mu_);
  size_t prefill_smem_ ABSL_GUARDED_BY(mu_) = 0;
  stream_executor::DeviceAddressBase kargs_prefill_ ABSL_GUARDED_BY(mu_);

  // Cached device scratch (allocated once per executor; reused each call).
  stream_executor::DeviceAddressBase kargs_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase kargs_reduce_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase tmp_ ABSL_GUARDED_BY(mu_);  // nwg=32 partials
  stream_executor::DeviceAddressBase zero_layer_ ABSL_GUARDED_BY(mu_);  // sliced-mode layer=0
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_FLASH_ATTN_THUNK_H_
