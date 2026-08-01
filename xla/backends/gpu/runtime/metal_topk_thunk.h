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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_TOPK_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_TOPK_THUNK_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class MetalTopKThunk : public Thunk {
 public:
  MetalTopKThunk(ThunkInfo thunk_info, BufferAllocation::Slice data,
                 BufferAllocation::Slice top_vals,
                 BufferAllocation::Slice top_idxs, PrimitiveType dtype,
                 int64_t batch_size, int64_t num_elements, int64_t k);

  MetalTopKThunk(const MetalTopKThunk&) = delete;
  MetalTopKThunk& operator=(const MetalTopKThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

  static void Prewarm(stream_executor::StreamExecutor* executor,
                      PrimitiveType dtype, int64_t k);

 private:
  absl::Status Ensure(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice data_, top_vals_, top_idxs_;
  const PrimitiveType dtype_;
  const int64_t batch_;
  const int64_t n_;
  const int64_t k_;
  const int64_t k_rounded_;  // bit_ceil(k) in {1,2,4,8,16,32,64} -> select template

  static constexpr int64_t kCap = 16384;

  static constexpr int64_t kSinglePassMaxN = 1024;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  bool single_pass_ ABSL_GUARDED_BY(mu_) = false;
  std::unique_ptr<stream_executor::Kernel> single_pass_pso_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> hist_pso_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> scan_pso_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> gather_pso_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> select_pso_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase hist_ ABSL_GUARDED_BY(mu_);     // B*65536 u32
  stream_executor::DeviceAddressBase thresh_ ABSL_GUARDED_BY(mu_);   // B u32
  stream_executor::DeviceAddressBase ccount_ ABSL_GUARDED_BY(mu_);   // B*2 u32
  stream_executor::DeviceAddressBase cok_ ABSL_GUARDED_BY(mu_);      // B*kCap u32
  stream_executor::DeviceAddressBase cix_ ABSL_GUARDED_BY(mu_);      // B*kCap u32
  stream_executor::DeviceAddressBase args_ ABSL_GUARDED_BY(mu_);     // {n,k,cap}
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_TOPK_THUNK_H_
