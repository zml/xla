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

#ifndef XLA_BACKENDS_GPU_CODEGEN_KERNELS_FP8_BLOCK_GEMV_KERNEL_H_
#define XLA_BACKENDS_GPU_CODEGEN_KERNELS_FP8_BLOCK_GEMV_KERNEL_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"

namespace xla::gpu::kernel {

struct Fp8BlockGemvKernelConfig {
  int num_warps;      // threads per block = num_warps * 32
  int rows_per_warp;  // output rows each warp reduces
  int unroll;         // 16-byte weight loads in flight per row
};

bool IsSupportedFp8BlockGemvKernelConfig(const Fp8BlockGemvKernelConfig& config);

absl::StatusOr<CustomKernel> GetFp8BlockGemvKernel(
    const Fp8BlockGemvKernelConfig& config, int64_t activation_arg,
    int64_t weight_arg, int64_t scale_arg, int64_t output_arg, int64_t n,
    int64_t k);

}  // namespace xla::gpu::kernel

#endif  // XLA_BACKENDS_GPU_CODEGEN_KERNELS_FP8_BLOCK_GEMV_KERNEL_H_
