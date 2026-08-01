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

#ifndef XLA_BACKENDS_GPU_RUNTIME_METAL_SORT_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_METAL_SORT_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class MetalSortThunk : public Thunk {
 public:
  MetalSortThunk(ThunkInfo thunk_info, BufferAllocation::Slice data,
                 BufferAllocation::Slice out_vals,
                 BufferAllocation::Slice out_idxs, PrimitiveType dtype,
                 int64_t rows, int64_t num_elements, bool descending);

  MetalSortThunk(const MetalSortThunk&) = delete;
  MetalSortThunk& operator=(const MetalSortThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

  static void Prewarm(stream_executor::StreamExecutor* executor,
                      PrimitiveType dtype, bool descending);

 private:
  absl::Status Ensure(stream_executor::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const BufferAllocation::Slice data_, out_vals_, out_idxs_;
  const PrimitiveType dtype_;
  const int64_t rows_;
  const int64_t n_;
  const bool descending_;

  static constexpr int kBlockThreads = 512;
  static constexpr int kPerThread = 4;
  static constexpr int kPerBlock = kBlockThreads * kPerThread;  // 2048
  // The partition kernel indexes by its threadgroup width and the merge by
  // n_blocks+1; they agree only while n_blocks+1 fits in one threadgroup.
  static constexpr int64_t kMaxBlocks = 1023;

  const int64_t n_blocks_;

  absl::Mutex mu_;
  stream_executor::StreamExecutor* executor_ ABSL_GUARDED_BY(mu_) = nullptr;
  std::unique_ptr<stream_executor::Kernel> block_pso_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> part_pso_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<stream_executor::Kernel> merge_pso_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase args_n_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase args_nb_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase args_mt_ ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase dev_vals_[2] ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase dev_idxs_[2] ABSL_GUARDED_BY(mu_);
  stream_executor::DeviceAddressBase parts_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_SORT_THUNK_H_
