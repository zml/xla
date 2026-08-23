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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_FLY_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_FLY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Autotuning backend for GEMMs emitted through the Fly/FlyROCDL dialects.
class FlyBackend final : public GpuCodegenBackend {
 public:
  FlyBackend(const DebugOptions* debug_options, Compiler* compiler,
             const Compiler::GpuTargetConfig* target_config)
      : GpuCodegenBackend(autotuner::Backend::FLY, debug_options, compiler,
                          target_config) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

  bool CanProduceWrongResults() const override { return true; }
  std::string version() const override { return "fly-rocdl-v2"; }

 private:
  bool IsSupported(const HloInstruction& instr) override;
};

// Autotuning backend for generic elementwise fusions emitted through the
// Fly/FlyROCDL dialects. The initial search space tunes the amount of work per
// thread; block-level Fly layouts are added on top of the same backend.
class FlyFusionBackend final : public GpuCodegenBackend {
 public:
  FlyFusionBackend(const DebugOptions* debug_options, Compiler* compiler,
                   const Compiler::GpuTargetConfig* target_config)
      : GpuCodegenBackend(autotuner::Backend::FLY_FUSION, debug_options,
                          compiler, target_config) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

  bool CanProduceWrongResults() const override { return true; }
  std::string version() const override { return "fly-rocdl-fusion-v1"; }

 private:
  bool IsSupported(const HloInstruction& instr) override;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_FLY_H_
