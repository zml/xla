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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_GEMM_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_GEMM_THUNK_H_

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/metalblas_gemm.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

class MetalGemmThunk : public Thunk {
 public:
  MetalGemmThunk(ThunkInfo thunk_info, MetalGemmLaunch launch,
                 BufferAllocation::Slice lhs, Shape lhs_shape,
                 BufferAllocation::Slice rhs, Shape rhs_shape,
                 BufferAllocation::Slice out, Shape out_shape,
                 BufferAllocation::Slice num_tokens, Shape num_tokens_shape);
  ~MetalGemmThunk() override;

  MetalGemmThunk(const MetalGemmThunk&) = delete;
  MetalGemmThunk& operator=(const MetalGemmThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  const MetalGemmLaunch launch_;
  const BufferAllocation::Slice lhs_;
  const Shape lhs_shape_;
  const BufferAllocation::Slice rhs_;
  const Shape rhs_shape_;
  const BufferAllocation::Slice out_;
  const Shape out_shape_;
  const BufferAllocation::Slice num_tokens_;
  const Shape num_tokens_shape_;
  const bool has_num_tokens_;

  absl::Mutex mu_;
  std::unique_ptr<::stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);
  ::stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  ::stream_executor::DeviceAddressBase params_mem_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<::stream_executor::Kernel> accum_kernel_ ABSL_GUARDED_BY(mu_);
  ::stream_executor::DeviceAddressBase staging_mem_ ABSL_GUARDED_BY(mu_);
  ::stream_executor::DeviceAddressBase accum_params_mem_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_GEMM_THUNK_H_
