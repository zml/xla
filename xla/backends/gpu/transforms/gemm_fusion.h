/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_BACKENDS_GPU_TRANSFORMS_GEMM_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_GEMM_FUSION_H_

// This file contains the code for fusing dots and other operations into GPU
// GEMM fusions consumed by the Triton or FlyDSL autotuning backends.

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

enum class GemmFusionTarget { kTriton, kFly };

// Rewrite compatible dot() calls into custom fusions for the selected matmul
// emitter family. Autotuning later installs a concrete backend configuration.
class GemmFusion : public HloModulePass {
 public:
  explicit GemmFusion(
      const se::GpuComputeCapability& compute_capability,
      GemmFusionTarget target = GemmFusionTarget::kTriton)
      : compute_capability_(compute_capability), target_(target) {}
  absl::string_view name() const override {
    return target_ == GemmFusionTarget::kFly ? "fly-gemm-rewriter"
                                             : "triton-gemm-rewriter";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability compute_capability_;
  GemmFusionTarget target_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_GEMM_FUSION_H_
