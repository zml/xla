#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/cuda_tile_custom_kernel_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla {
namespace gpu {

xla::Future<ThunkSequence> EmitCudaTileCustomKernelThunk(
    const HloCustomCallInstruction* /*instr*/, IrEmitterContext* /*context*/) {
  return absl::UnimplementedError(
      "CUDA Tile IR custom calls are only supported on the CUDA platform");
}

}  // namespace gpu
}  // namespace xla
