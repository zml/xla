#ifndef XLA_BACKENDS_GPU_AUTOTUNER_TILE_IR_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_TILE_IR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class TileIrBackend : public GpuCodegenBackend {
 public:
  TileIrBackend(const DebugOptions* debug_options, Compiler* compiler,
                const Compiler::GpuTargetConfig* target_config,
                mlir::MLIRContext* mlir_context)
      : GpuCodegenBackend(autotuner::Backend::TILE_IR, debug_options, compiler,
                          target_config),
        mlir_context_(mlir_context) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

  bool IsSupported(const HloInstruction& instr) override;

  bool CanProduceWrongResults() const override { return true; }

  std::string version() const override;

 private:
  struct GemmBounds {
    const HloInstruction* dot;
    int64_t max_m;
    int64_t max_n;
    int64_t contracting_size;
  };

  std::optional<GemmBounds> GetGemmBounds(const HloInstruction& instr);

  absl::StatusOr<xla::xtile::BlockLevelFusionConfig> BlockLevelConfigForTile(
      const HloInstruction& dot, int64_t block_m, int64_t block_n,
      int64_t block_k);

  absl::StatusOr<std::unique_ptr<BackendConfig>> ConfigForTile(
      const HloInstruction& dot, int64_t block_m, int64_t block_n,
      int64_t block_k);

  mlir::MLIRContext* mlir_context_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_TILE_IR_H_
