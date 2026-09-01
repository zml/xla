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

#include "xla/backends/gpu/runtime/fp8_block_gemm_cutlass_thunk.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/codegen/kernels/fp8_block_gemm_cutlass.h"
#include "xla/stream_executor/stream.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla::gpu {

Fp8BlockGemmCutlassThunk::Fp8BlockGemmCutlassThunk(
    ThunkInfo thunk_info, int config, const BufferAllocation::Slice& a,
    const BufferAllocation::Slice& a_scales, const BufferAllocation::Slice& b,
    const BufferAllocation::Slice& b_scales, const BufferAllocation::Slice& d,
    int64_t m, int64_t n, int64_t k)
    : TracedCommand(Thunk::Kind::kCustomKernel, std::move(thunk_info)),
      config_(config),
      a_(a),
      a_scales_(a_scales),
      b_(b),
      b_scales_(b_scales),
      d_(d),
      m_(m),
      n_(n),
      k_(k) {}

absl::Status Fp8BlockGemmCutlassThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  const BufferAllocations& allocs = *params.buffer_allocations;

  // CanRun declines a config that wants a workspace, so the autotuner cannot
  // choose one; this is the belt to that braces -- reaching it means the
  // predicate and the table disagree, and a null workspace pointer would be
  // written through.
  if (size_t ws = kernel::Fp8BlockGemmCutlassWorkspaceSize(config_, m_, n_, k_);
      ws != 0) {
    return Internal(
        "fp8 block gemm cutlass config %d wants a %d byte workspace, which "
        "this thunk does not allocate",
        config_, static_cast<int64_t>(ws));
  }

  kernel::Fp8BlockGemmCutlassParams kernel_params;
  kernel_params.a = allocs.GetDeviceAddress(a_).opaque();
  kernel_params.a_scales = allocs.GetDeviceAddress(a_scales_).opaque();
  kernel_params.b = allocs.GetDeviceAddress(b_).opaque();
  kernel_params.b_scales = allocs.GetDeviceAddress(b_scales_).opaque();
  kernel_params.d = allocs.GetDeviceAddress(d_).opaque();
  kernel_params.workspace = nullptr;
  kernel_params.m = m_;
  kernel_params.n = n_;
  kernel_params.k = k_;

  void* stream = params.stream->platform_specific_handle().stream;
  const char* error = nullptr;
  if (int rc = kernel::Fp8BlockGemmCutlassRun(config_, kernel_params, stream,
                                              &error);
      rc != 0) {
    return Internal("fp8 block gemm cutlass config %s failed: %s",
                    kernel::Fp8BlockGemmCutlassConfigName(config_),
                    error == nullptr ? "unknown" : error);
  }
  return absl::OkStatus();
}

std::string Fp8BlockGemmCutlassThunk::ToString(int indent) const {
  return absl::StrCat(" config=",
                      kernel::Fp8BlockGemmCutlassConfigName(config_), " m=", m_,
                      " n=", n_, " k=", k_);
}

Thunk::BufferUses Fp8BlockGemmCutlassThunk::buffer_uses() const {
  return {
      BufferUse::Read(a_, ShapeUtil::MakeShape(S8, {a_.size()})),
      BufferUse::Read(a_scales_, ShapeUtil::MakeShape(S8, {a_scales_.size()})),
      BufferUse::Read(b_, ShapeUtil::MakeShape(S8, {b_.size()})),
      BufferUse::Read(b_scales_, ShapeUtil::MakeShape(S8, {b_scales_.size()})),
      BufferUse::Write(d_, ShapeUtil::MakeShape(S8, {d_.size()})),
  };
}

absl::StatusOr<ThunkProto> Fp8BlockGemmCutlassThunk::ToProto() const {
  return absl::UnimplementedError(
      "Fp8BlockGemmCutlassThunk is not serializable");
}

}  // namespace xla::gpu
