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

  // Must equal kROWS in custom/fp8_gemv_pc.metal.
  int64_t rows_per_group() const { return per_channel_ ? 2 : 1; }

  static int64_t MaxVecs() {
    static const int64_t v = [] {
      const char* e = std::getenv("METAL_FP8_WIDE_VECS");
      const int64_t n = e ? std::atoll(e) : 0;
      return (n >= 2 && n <= 12) ? n : 10;
    }();
    return v;
  }

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
  std::unique_ptr<stream_executor::Kernel> kernel_ ABSL_GUARDED_BY(mu_);

  stream_executor::DeviceAddressBase p_dims_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_METAL_FP8_GEMV_THUNK_H_
