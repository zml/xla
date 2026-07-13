/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_SCALED_DOT_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_SCALED_DOT_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Lowers kScaledDot when the Triton path is not used
// (xla_gpu_experimental_scaled_dot_with_triton == false).
//
// For each scaled-dot:
//   1. Try a backend-fused custom call via TryFusedScaledMatmul (platform
//      switch: Metal emits zml$scaled_matmul for supported weight-only layouts;
//      other backends currently return null).
//   2. Else expand to Convert/Broadcast/Reshape/Multiply + Dot (generic
//      dequant fallback).
//
// Shared layout predicates (MX group-32, NVFP4, 128-block, per-channel) live
// in the .cc; only the emit target is platform-specific.
class ScaledDotRewriter : public HloModulePass {
 public:
  explicit ScaledDotRewriter(se::GpuComputeCapability gpu_version = {})
      : gpu_version_(gpu_version) {}

  absl::string_view name() const override { return "scaled-dot-rewriter"; }

  absl::StatusOr<bool> RewriteComputation(HloComputation* computation);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_SCALED_DOT_REWRITER_H_
