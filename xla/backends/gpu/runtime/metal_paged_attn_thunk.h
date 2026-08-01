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
                      bool is_causal, PrimitiveType element_type);

  MetalPagedAttnThunk(const MetalPagedAttnThunk&) = delete;
  MetalPagedAttnThunk& operator=(const MetalPagedAttnThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

  static void Prewarm(stream_executor::StreamExecutor* executor,
                      PrimitiveType dtype, int64_t head_dim, int64_t block_size,
                      int64_t num_kv_heads);

 private:
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

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
  const bool is_causal_;
  const PrimitiveType element_type_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> zero_kernel_ ABSL_GUARDED_BY(mu_);
  size_t shmem_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  int bq_ ABSL_GUARDED_BY(mu_) = 0;
  int num_threads_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t total_q_blocks_ ABSL_GUARDED_BY(mu_) = 0;

  stream_executor::DeviceAddressBase p_num_kv_heads_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_scale_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_softcapping_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_max_blocks_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_q_stride_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_kv_block_stride_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_kv_head_stride_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_num_seqs_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_sliding_window_ ABSL_GUARDED_BY(mu_);
  // Bound at the verbatim kernel's unused sparse buffer slots so positional
  // packing reaches its real [[buffer(N)]] indices.
  stream_executor::DeviceAddressBase dummy_ ABSL_GUARDED_BY(mu_);

  static constexpr int kVecNsgVals[3] = {4, 8, 16};
  bool use_vec_decode_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<uint8_t> vec_lib_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> vec_kernel_by_nsg_[3]
      ABSL_GUARDED_BY(mu_);
  size_t vec_smem_by_nsg_[3] ABSL_GUARDED_BY(mu_) = {0, 0, 0};
  stream_executor::DeviceAddressBase p_vec_args_ ABSL_GUARDED_BY(mu_);

  static constexpr int kSplitKNwg = 4;     // position-split workgroups
  static constexpr int kSplitKMinKv = 4096;  // static-capacity gate
  bool use_split_k_ ABSL_GUARDED_BY(mu_) = false;
  std::unique_ptr<stream_executor::Kernel> vec_partial_kernel_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> vec_reduce_kernel_ ABSL_GUARDED_BY(mu_);
  size_t vec_partial_smem_ ABSL_GUARDED_BY(mu_) = 0;
  stream_executor::DeviceAddressBase p_partial_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_reduce_args_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_PAGED_ATTN_THUNK_H_
