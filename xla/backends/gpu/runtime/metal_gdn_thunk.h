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

  stream_executor::DeviceAddressBase p_num_requests_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_hk_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_hv_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_dk_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase p_dv_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_GDN_THUNK_H_
