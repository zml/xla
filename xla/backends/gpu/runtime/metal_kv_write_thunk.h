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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_KV_WRITE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_KV_WRITE_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Implements the "metal$kv_write" custom call: the decode-step paged KV-cache
// write, fused into ONE kernel. The pattern it replaces (see
// RewriteKvCacheWrites in metal_gpu_compiler.cc) is the 6-kernel cluster XLA
// emits for zml's conditional cache update — per layer per token:
//
//   slot_valid = and-reduce(0 <= [page,off] <= [P-1,B-1])     (pred[] reduce)
//   pred       = broadcast(slot_valid)                        (wrapped bcast)
//   old_k/old_v = dynamic_slice(cache, page, off)             (2 slice kernels)
//   k_cache[page,off] = select(pred, rope(k_new), old_k)      (in-place DUS
//   v_cache[page,off] = select(pred, v_new,       old_v)       fusion, both)
//
// plus the s32[4] zero-vector feeding the reduce. Pure HLO cannot express a
// PREDICATED STORE, so "don't touch the cache when the slot is padding
// (slot==-1)" costs five extra dispatches of index plumbing — ~12us/layer of
// launch floor at decode. The kernel here is that predicated store:
// `if (slot < 0 || slot >= P*B) return;` then write v_new raw and k_new
// RoPE-rotated, exactly replicating the fused rotate-half arithmetic
// (f32 angle = f32(pos) * freq[j]; cos/sin -> bf16 -> f32 — air.fast_cos, the
// same intrinsic the MLIR emitter lowers HLO cosine to; operands bf16 -> f32;
// result -> bf16).
//
// Operand order (fixed by the rewriter):
//   0 k_cache  bf16 [P, B, H, D]   (rope arm target; updated in place)
//   1 k_new    bf16 [H*D]          (pre-RoPE)
//   2 v_cache  bf16 [P, B, H, D]   (raw arm target; updated in place)
//   3 v_new    bf16 [H*D]
//   4 slot     s32  [1]            (absolute slot = page*B + off; -1 = padding)
//   5 pos      s32  [1]            (token position for RoPE)
//   6 freq     f32  [1, D/2]       (inverse-frequency table)
// Output: tuple(k_cache, v_cache) — output_to_operand_aliasing makes both
// updates in place, like the DUS fusion it replaces.
class MetalKvWriteThunk : public Thunk {
 public:
  MetalKvWriteThunk(ThunkInfo thunk_info, BufferAllocation::Slice k_cache,
                    Shape k_cache_shape, BufferAllocation::Slice k_new,
                    Shape k_new_shape, BufferAllocation::Slice v_cache,
                    Shape v_cache_shape, BufferAllocation::Slice v_new,
                    Shape v_new_shape, BufferAllocation::Slice slot,
                    Shape slot_shape, BufferAllocation::Slice pos,
                    Shape pos_shape, BufferAllocation::Slice freq,
                    Shape freq_shape, int64_t num_slots, int64_t kv_heads,
                    int64_t head_dim);

  MetalKvWriteThunk(const MetalKvWriteThunk&) = delete;
  MetalKvWriteThunk& operator=(const MetalKvWriteThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

  // Compile-time prewarm (MetalGpuCompiler::RunHloPasses): compile the
  // substituted metallib + create the PSO into Apple's driver pipeline cache so
  // the first decode token's EnsureLoaded is a cache hit (no `xcrun`, no PSO
  // build). Params must match the thunk's so the warmed kernel matches.
  // Best-effort — failures are swallowed (the thunk rebuilds at execute).
  static void Prewarm(stream_executor::StreamExecutor* executor,
                      int64_t num_slots, int64_t kv_heads, int64_t head_dim);

 private:
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice k_cache_, k_new_, v_cache_, v_new_, slot_,
      pos_, freq_;
  const Shape k_cache_shape_, k_new_shape_, v_cache_shape_, v_new_shape_,
      slot_shape_, pos_shape_, freq_shape_;
  const int64_t num_slots_, kv_heads_, head_dim_;

  absl::Mutex mu_;
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_KV_WRITE_THUNK_H_
