#ifndef XLA_BACKENDS_GPU_CODEGEN_TILE_IR_FUSION_H_
#define XLA_BACKENDS_GPU_CODEGEN_TILE_IR_FUSION_H_

#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla::gpu::tile_ir {

class TileIrFusion : public FusionInterface {
 public:
  explicit TileIrFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis) {}

  AsyncThunkSequence Emit(IrEmitterContext& ir_emitter_context,
                          const HloFusionInstruction& fusion) const final;

 private:
  const HloFusionAnalysis& analysis_;
};

}  // namespace xla::gpu::tile_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TILE_IR_FUSION_H_
