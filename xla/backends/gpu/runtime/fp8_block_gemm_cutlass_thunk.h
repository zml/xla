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

#ifndef XLA_BACKENDS_GPU_RUNTIME_FP8_BLOCK_GEMM_CUTLASS_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_FP8_BLOCK_GEMM_CUTLASS_THUNK_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/service/buffer_assignment.h"

namespace xla::gpu {

// Runs one config of the vendored CUTLASS blockwise FP8 GEMM. Unlike the
// hand-written decode kernel this is not a bare __global__ symbol -- CUTLASS's
// GemmUniversalAdapter is a host-side launcher that computes its own grid and
// cluster dimensions -- so it cannot go through CustomKernelThunk, which only
// knows how to launch an in-process symbol. TracedCommand gives it a command
// buffer node by tracing ExecuteOnStream, the same way the cuBLAS thunk does.
class Fp8BlockGemmCutlassThunk : public TracedCommand {
 public:
  Fp8BlockGemmCutlassThunk(ThunkInfo thunk_info, int config,
                           const BufferAllocation::Slice& a,
                           const BufferAllocation::Slice& a_scales,
                           const BufferAllocation::Slice& b,
                           const BufferAllocation::Slice& b_scales,
                           const BufferAllocation::Slice& d, int64_t m,
                           int64_t n, int64_t k);

  Fp8BlockGemmCutlassThunk(const Fp8BlockGemmCutlassThunk&) = delete;
  Fp8BlockGemmCutlassThunk& operator=(const Fp8BlockGemmCutlassThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  std::string ToString(int indent) const override;

  BufferUses buffer_uses() const override;

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  const int config_;
  const BufferAllocation::Slice a_;
  const BufferAllocation::Slice a_scales_;
  const BufferAllocation::Slice b_;
  const BufferAllocation::Slice b_scales_;
  const BufferAllocation::Slice d_;
  const int64_t m_;
  const int64_t n_;
  const int64_t k_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_FP8_BLOCK_GEMM_CUTLASS_THUNK_H_
