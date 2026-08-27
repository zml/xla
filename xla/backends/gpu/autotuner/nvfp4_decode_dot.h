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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_NVFP4_DECODE_DOT_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_NVFP4_DECODE_DOT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class Nvfp4DecodeDotBackend : public GpuCodegenBackend {
 public:
  enum class Rung { kTriton, kTileIr };

  explicit Nvfp4DecodeDotBackend(const DebugOptions* debug_options,
                                 Compiler* compiler,
                                 const Compiler::GpuTargetConfig* target_config,
                                 mlir::MLIRContext* mlir_context,
                                 Rung rung = Rung::kTriton)
      : GpuCodegenBackend(BackendFor(rung), debug_options, compiler,
                          target_config),
        mlir_context_(mlir_context),
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
    return rung_ == Rung::kTileIr ? "tile_ir_13.3" : "1";
  }

 private:
  static autotuner::Backend BackendFor(Rung rung) {
    return rung == Rung::kTileIr ? autotuner::Backend::NVFP4_DECODE_DOT_TILE_IR
                                 : autotuner::Backend::NVFP4_DECODE_DOT;
  }

  mlir::MLIRContext* mlir_context_;
  Rung rung_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_NVFP4_DECODE_DOT_H_
