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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_GDN_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_GDN_THUNK_H_

#include <cstdint>
#include <memory>

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

// Implements the "zml$gdn" custom call on the v2 Metal backend: the Gated
// DeltaNet recurrent linear-attention rule (`gdn_linear_attention.metal`,
// vendored VERBATIM from vllm-metal). It is the recurrence that the Triton
// `fused_recurrent_gated_delta_rule_fwd_kernel` runs on CUDA; Metal has no
// Triton compiler, so ZML's gated_delta_net.zig emits this custom call instead
// on the Metal backend.
//
// One SIMD group (32 threads) per (request, value-head, value-dim) cooperates
// over the key dim (Dk <= 256) and loops the recurrence over the request's
// tokens, updating a per-request recurrent state IN PLACE.
//
// Division of labour vs the Triton kernel: the Metal kernel does NOT
// L2-normalize q/k, does NOT scale q, and uses g directly as the per-step
// multiplicative decay. ZML therefore folds those into plain XLA ops before the
// call (q := l2norm(q)*scale, k := l2norm(k), g := exp(g)) and passes f32
// operands (the recurrent accumulation needs f32, matching vllm's float
// dispatch). So this thunk only runs the bare recurrence.
//
// Operand contract (positional):
//   0 q            f32 [total_tokens, Hk, Dk]   (already l2-normed * scale)
//   1 k            f32 [total_tokens, Hk, Dk]   (already l2-normed)
//   2 v            f32 [total_tokens, Hv, Dv]
//   3 g            f32 [total_tokens, Hv]        (already exp'd decay)
//   4 beta         f32 [total_tokens, Hv]
//   5 h0           f32 [num_seqs, Hv, Dk, Dv]    (initial recurrent state)
//   6 cu_seqlens   i32 [num_seqs + 1]            (cumulative token counts)
//   7 slot_mapping i32 [num_seqs]                (request -> state slot; iota)
//   -> 0 y         f32 [total_tokens, Hv, Dv]    (output activations)
//   -> 1 ht        f32 [num_seqs, Hv, Dk, Dv]    (final state; aliases h0)
//
// ht aliases h0 (output_operand_aliasing), so the kernel reads the initial
// state and overwrites it in place to produce the final state — no copy. The
// (Dk,Dv) state block is opaque to ZML (only this kernel reads/writes it), so
// the kernel's internal [slot,Hv,Dv,Dk] layout vs ZML's [.,Hv,Dk,Dv] labelling
// is irrelevant as long as it is self-consistent across chunks.
class MetalGdnThunk : public Thunk {
 public:
  MetalGdnThunk(ThunkInfo thunk_info, BufferAllocation::Slice q, Shape q_shape,
                BufferAllocation::Slice k, Shape k_shape,
                BufferAllocation::Slice v, Shape v_shape,
                BufferAllocation::Slice g, Shape g_shape,
                BufferAllocation::Slice beta, Shape beta_shape,
                BufferAllocation::Slice h0, Shape h0_shape,
                BufferAllocation::Slice cu_seqlens, Shape cu_seqlens_shape,
                BufferAllocation::Slice slot_mapping, Shape slot_mapping_shape,
                BufferAllocation::Slice y, Shape y_shape,
                BufferAllocation::Slice ht, Shape ht_shape, int64_t num_seqs,
                int64_t hk, int64_t hv, int64_t dk, int64_t dv,
                PrimitiveType element_type);

  MetalGdnThunk(const MetalGdnThunk&) = delete;
  MetalGdnThunk& operator=(const MetalGdnThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  // Lazily, on the first execute for a given executor: compile the embedded
  // metallib (process-cached), load the dtype kernel variant, and stage the
  // five `constant int&` scalar params (num_requests, Hk, Hv, Dk, Dv) into small
  // device buffers. Must hold mu_.
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice q_, k_, v_, g_, beta_, h0_, cu_seqlens_,
      slot_mapping_, y_, ht_;
  const Shape q_shape_, k_shape_, v_shape_, g_shape_, beta_shape_, h0_shape_,
      cu_seqlens_shape_, slot_mapping_shape_, y_shape_, ht_shape_;
  const int64_t num_seqs_, hk_, hv_, dk_, dv_;
  const PrimitiveType element_type_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);

  // The five scalar params staged into individual device buffers (the kernel
  // takes them as inline `constant int&` args at buffer indices 9..13),
  // allocated once per executor.
  stream_executor::DeviceAddressBase p_num_requests_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_hk_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_hv_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_dk_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_dv_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_GDN_THUNK_H_
