#ifndef XLA_SERVICE_GPU_CUDA_TILE_CUSTOM_KERNEL_EMITTER_H_
#define XLA_SERVICE_GPU_CUDA_TILE_CUSTOM_KERNEL_EMITTER_H_

#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/future.h"

namespace xla {

class HloCustomCallInstruction;

namespace gpu {

class IrEmitterContext;

// Assembles the call's cuda_tile module with tileiras and launches it as a
// CustomKernelThunk. Only the CUDA implementation does anything.
xla::Future<ThunkSequence> EmitCudaTileCustomKernelThunk(
    const HloCustomCallInstruction* instr, IrEmitterContext* context);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUDA_TILE_CUSTOM_KERNEL_EMITTER_H_
