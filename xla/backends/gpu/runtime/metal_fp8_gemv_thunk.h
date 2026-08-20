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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_FP8_GEMV_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_FP8_GEMV_THUNK_H_

#include <cstdint>
#include <cstdlib>
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

// FP8 arm of zml$scaled_matmul on Metal: fused weight-only GEMV / thin GEMM for
// f8e4m3fn weights with bf16 scales. NVFP4 lives in MetalNvfp4MatmulThunk.
//
// Scale layouts:
//   * 128-block:  scale bf16[N/128, K/128]
//   * per-channel: scale bf16[N, 1]
//
// Operand contract:
//   0 x      bf16     [B, K]
//   1 w      f8e4m3fn [N, K]
//   2 scale  bf16     [N/128, K/128] or [N, 1]
//   -> out   bf16     [B, N]
class MetalFp8GemvThunk : public Thunk {
 public:
  MetalFp8GemvThunk(ThunkInfo thunk_info, BufferAllocation::Slice x,
                    Shape x_shape, BufferAllocation::Slice w, Shape w_shape,
                    BufferAllocation::Slice scale, Shape scale_shape,
                    BufferAllocation::Slice out, Shape out_shape, int64_t b,
                    int64_t k, int64_t n, bool per_channel);

  MetalFp8GemvThunk(const MetalFp8GemvThunk&) = delete;
  MetalFp8GemvThunk& operator=(const MetalFp8GemvThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Output channels one decode-GEMV threadgroup computes; must equal kROWS in
  // custom/fp8_gemv_pc.metal, which carries the benchmark that picked 2. The
  // block-128 GEMV still does one row.
  int64_t rows_per_group() const { return per_channel_ ? 2 : 1; }

  // Input rows one decode threadgroup serves, or 0 for "not the wide path".
  //
  // Every tile re-reads the weights -- traffic is N*K*ceil(B/kVECS) -- so pick
  // the FEWEST tiles and then the smallest tile that fills them: M=6 is 2x3,
  // not 5+1. Same rule MLX uses for fp_qmv_wide (quantized.cpp) and llama.cpp
  // for mul_mv_ext's r1ptg.
  //
  // Both of those stop at 5 and this stops at 10: their cap follows from their
  // kernel's register frame, and ours stages x per chunk position and narrows
  // its K chunk as kVECS grows (custom/fp8_gemv_pc.metal). This was 5 too until
  // one tile stopped measuring slower than two -- that was a staging array
  // spilling to thread scratch, not an occupancy cliff, so do not re-derive the
  // cap from a spilled frame. METAL_FP8_WIDE_VECS tunes it; entries reach 12.
  static int64_t MaxVecs() {
    static const int64_t v = [] {
      const char* e = std::getenv("METAL_FP8_WIDE_VECS");
      const int64_t n = e ? std::atoll(e) : 0;
      return (n >= 2 && n <= 12) ? n : 10;
    }();
    return v;
  }

  // Above this the Steel BM=16 tile fills well enough to win, so the wide
  // kernel steps aside. 12 is where the two actually cross, measured by moving
  // this bound alone; below it the wide kernel's tile is given away for
  // nothing. METAL_FP8_WIDE_MAX tunes it. (MLX puts its mat-vec limit at 12-32
  // depending on K and N, llama.cpp at 8.)
  static int64_t WideMaxBatch() {
    static const int64_t v = [] {
      const char* e = std::getenv("METAL_FP8_WIDE_MAX");
      const int64_t n = e ? std::atoll(e) : 0;
      return n > 0 ? n : 12;
    }();
    return v;
  }

  int64_t wide_vecs() const {
    if (!per_channel_ || b_ < 2 || b_ > WideMaxBatch()) return 0;
    const int64_t max_vecs = MaxVecs();
    const int64_t tiles = (b_ + max_vecs - 1) / max_vecs;
    return (b_ + tiles - 1) / tiles;
  }

  const BufferAllocation::Slice x_, w_, scale_, out_;
  const Shape x_shape_, w_shape_, scale_shape_, out_shape_;
  const int64_t b_, k_, n_;
  const bool per_channel_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  // Exactly one entry point can ever launch: which one is a function of
  // per_channel_, b_ and the scale's dtype, and all three are fixed when the
  // emitter builds this thunk. b==1 takes the per-row GEMV, b through
  // WideMaxBatch() the wide decode kernel, and the rest the Steel tiled q-GEMM
  // (BM=16 small M, BM=64 prefill). So load one rather than the family.
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);

  stream_executor::DeviceAddressBase p_dims_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_FP8_GEMV_THUNK_H_
