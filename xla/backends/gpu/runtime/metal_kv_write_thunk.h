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
