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

  static void PrewarmPipeline(stream_executor::StreamExecutor* executor,
                              bool is_prefill, int64_t kv_pos_stride,
                              int64_t seqlen, int64_t head_dim, int64_t n_kv);

 private:
  absl::Status EnsureFaVecMain(stream_executor::StreamExecutor* executor, int idx)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status EnsureFaVecHC(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status EnsurePrefill(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice q_, k_, v_, tok_, out_;
  const Shape q_shape_, k_shape_, v_shape_, tok_shape_, out_shape_;
  const int64_t n_kv_, n_groups_, seqlen_, head_dim_;
  const bool kv_position_major_;
  const bool kv_full_cache_;
  const BufferAllocation::Slice layer_;
  const Shape layer_shape_;
  const BufferAllocation::Slice num_tokens_;
  const Shape num_tokens_shape_;
  const bool has_num_tokens_;
  const bool tok_host_coherent_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;

  bool use_fa_vec_ ABSL_GUARDED_BY(mu_) = false;
  static constexpr int kNsgVals[5] = {1, 2, 4, 8, 16};
  std::vector<uint8_t> favec_lib_ ABSL_GUARDED_BY(mu_);  // shared fa_vec metallib
  int64_t kv_pos_stride_ ABSL_GUARDED_BY(mu_) = 0;       // K/V position stride (elems)
  std::unique_ptr<stream_executor::Kernel> fa_main_by_nsg_[5] ABSL_GUARDED_BY(mu_);
  size_t smem_by_nsg_[5] ABSL_GUARDED_BY(mu_) = {};
  std::unique_ptr<stream_executor::Kernel> fa_reduce_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> fa_main_hc_ ABSL_GUARDED_BY(mu_);
  size_t smem_hc_ ABSL_GUARDED_BY(mu_) = 0;
  bool use_prefill_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<uint8_t> prefill_lib_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> fa_prefill_ ABSL_GUARDED_BY(mu_);
  size_t prefill_smem_ ABSL_GUARDED_BY(mu_) = 0;
  stream_executor::DeviceAddressBase kargs_prefill_ ABSL_GUARDED_BY(mu_);

  stream_executor::DeviceAddressBase kargs_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase kargs_reduce_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase tmp_ ABSL_GUARDED_BY(mu_);  // nwg=32 partials
  stream_executor::DeviceAddressBase zero_layer_ ABSL_GUARDED_BY(mu_);  // sliced-mode layer=0
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_FLASH_ATTN_THUNK_H_
