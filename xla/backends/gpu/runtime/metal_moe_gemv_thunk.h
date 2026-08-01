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

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
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
                    BufferAllocation::Slice out, Shape out_shape,
                    BufferAllocation::Slice workspace, Shape workspace_shape,
                    BufferAllocation::Slice global_scale, Shape
                    global_scale_shape, bool has_global_scale, int64_t r,
                    int64_t k, int64_t n);
  ~MetalMoeGemvThunk() override;

  MetalMoeGemvThunk(const MetalMoeGemvThunk&) = delete;
  MetalMoeGemvThunk& operator=(const MetalMoeGemvThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  struct LoadedState;

  absl::StatusOr<std::shared_ptr<const LoadedState>> EnsureLoaded(
      stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice x_, w_, scale_, expert_id_, out_, workspace_,
      global_scale_;
  const Shape x_shape_, w_shape_, scale_shape_, expert_id_shape_, out_shape_,
      workspace_shape_, global_scale_shape_;
  const bool has_global_scale_;
  const int64_t r_, k_, n_;

  absl::Mutex mu_;
  absl::flat_hash_map<stream_executor::StreamExecutor*,
                      std::shared_ptr<const LoadedState>>
      states_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_MOE_GEMV_THUNK_H_
