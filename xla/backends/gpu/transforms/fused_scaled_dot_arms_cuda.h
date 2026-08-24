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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_FUSED_SCALED_DOT_ARMS_CUDA_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_FUSED_SCALED_DOT_ARMS_CUDA_H_

#include "xla/backends/gpu/transforms/fused_scaled_dot_rewriter.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

FusedScaledDotArm Fp8BlockGemvArm(const se::GpuComputeCapability& gpu_version);

FusedScaledDotArm Nvfp4DecodeDotArm(
    const se::GpuComputeCapability& gpu_version);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_FUSED_SCALED_DOT_ARMS_CUDA_H_
