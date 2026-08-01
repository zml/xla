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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_MOE_GEMV_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_MOE_GEMV_THUNK_H_

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

class MetalMoeGemvThunk : public Thunk {
 public:
  MetalMoeGemvThunk(ThunkInfo thunk_info, BufferAllocation::Slice x,
                    Shape x_shape, BufferAllocation::Slice w, Shape w_shape,
                    BufferAllocation::Slice scale, Shape scale_shape,
                    BufferAllocation::Slice expert_id, Shape expert_id_shape,
                    BufferAllocation::Slice out, Shape out_shape, int64_t r,
                    int64_t k, int64_t n,
                    BufferAllocation::Slice num_tokens, Shape num_tokens_shape,
                    int64_t top_k);

  MetalMoeGemvThunk(const MetalMoeGemvThunk&) = delete;
  MetalMoeGemvThunk& operator=(const MetalMoeGemvThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  absl::Status EnsureLoaded(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice x_, w_, scale_, expert_id_, out_;
  const Shape x_shape_, w_shape_, scale_shape_, expert_id_shape_, out_shape_;
  const int64_t r_, k_, n_;
  const BufferAllocation::Slice num_tokens_;
  const Shape num_tokens_shape_;
  const int64_t top_k_;
  const bool has_num_tokens_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  static constexpr int64_t kSortedMinR = 1024;

  bool sorted_path_ ABSL_GUARDED_BY(mu_) = false;
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_steel_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_argsort_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_gather_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_scatter_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> kernel_steel_grid_
      ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_steel_grid_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_steel_grid_args_ ABSL_GUARDED_BY(mu_);

  stream_executor::DeviceAddressBase p_dims_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_dims_steel_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_order_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_idx_sorted_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_x_sorted_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_out_sorted_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_argsort_dims_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_gx_dims_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_gout_dims_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_num_tokens_fallback_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_MOE_GEMV_THUNK_H_
