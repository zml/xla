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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_FP8_BLOCK_GEMV_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_FP8_BLOCK_GEMV_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class Fp8BlockGemvBackend : public GpuCodegenBackend {
 public:
  enum class Rung { kTriton, kTileIr };

  explicit Fp8BlockGemvBackend(const DebugOptions* debug_options,
                               Compiler* compiler,
                               const Compiler::GpuTargetConfig* target_config,
                               Rung rung = Rung::kTriton)
      : GpuCodegenBackend(rung == Rung::kTriton
                              ? autotuner::Backend::FP8_BLOCK_GEMV
                              : autotuner::Backend::FP8_BLOCK_GEMV_TILE_IR,
                          debug_options, compiler, target_config),
        rung_(rung) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

  bool IsSupported(const HloInstruction& instr) override;

  bool CanProduceWrongResults() const override { return false; }

  std::string version() const override {
    return rung_ == Rung::kTriton ? "1" : "tile_ir_13.3";
  }

 private:
  Rung rung_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_FP8_BLOCK_GEMV_H_
