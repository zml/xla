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

#ifndef XLA_SERVICE_GPU_METALBLAS_GEMM_H_
#define XLA_SERVICE_GPU_METALBLAS_GEMM_H_

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct MetalGemmLaunch {
  std::vector<uint8_t> metallib;
  std::string kernel_name;
  ::stream_executor::ThreadDim thread_dim;
  ::stream_executor::BlockDim block_dim;
  std::array<uint32_t, 6> params;
  uint32_t bm = 0;
  uint32_t swizzle_log = 0;
  bool swap_ab = false;

  std::string accum_kernel_name;
  ::stream_executor::ThreadDim accum_thread_dim;
  ::stream_executor::BlockDim accum_block_dim;
  uint32_t splitk_partitions = 0;
  uint64_t staging_bytes = 0;
  std::array<uint32_t, 13> splitk_params{};
  std::array<uint32_t, 3> accum_params{};  // {k_partitions, partition_stride, ldd}
};

absl::StatusOr<MetalGemmLaunch> CompileMetalblasGemm(int64_t M, int64_t N,
                                                     int64_t K, bool trans_a,
                                                     bool trans_b,
                                                     PrimitiveType dtype,
                                                     bool prefill_token_axis);

absl::StatusOr<MetalGemmLaunch> CompileMetalblasGemv(int64_t M, int64_t N,
                                                     int64_t K, bool trans_a,
                                                     bool trans_b,
                                                     PrimitiveType dtype,
                                                     int64_t b_byte_offset);

absl::StatusOr<MetalGemmLaunch> CompileMetalblasSplitk(int64_t M, int64_t N,
                                                       int64_t K, bool trans_a,
                                                       bool trans_b,
                                                       PrimitiveType dtype);

absl::StatusOr<std::vector<uint8_t>> CompileMetalblasKernelToMetallib(
    absl::string_view build_flag, absl::string_view family_source,
    absl::Span<const std::pair<absl::string_view, std::string>> subs,
    absl::string_view extra_defines = "");

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METALBLAS_GEMM_H_
