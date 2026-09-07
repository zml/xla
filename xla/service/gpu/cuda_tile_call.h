#ifndef XLA_SERVICE_GPU_CUDA_TILE_CALL_H_
#define XLA_SERVICE_GPU_CUDA_TILE_CALL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"

namespace xla::gpu {

// The backend_config of a __gpu$xla.gpu.cuda_tile custom call: a printed MLIR
// dictionary of `name` (the cuda_tile.entry symbol), `kernel_type`, `ir`
// (textual module), `grid_x/y/z`, optional `ir_version` and `output_indices`.
// Launch-shape keys are rejected: cuda-tile fixes the block at (1,1,1) with no
// shared memory, and warp/CTA tuning is optimization_hints inside the IR.
struct CudaTileCall {
  std::string name;
  std::string ir;
  int32_t grid_x;
  int32_t grid_y;
  int32_t grid_z;
  uint8_t bytecode_major = 13;
  uint8_t bytecode_minor = 3;
  std::vector<int32_t> output_indices;

  static absl::StatusOr<CudaTileCall> Parse(absl::string_view backend_config,
                                            mlir::MLIRContext* mlir_context);
};

// The config plus the argument count: a cache hit skips the generator that
// checks the entry's arity.
std::string CudaTileKernelFingerprint(absl::string_view backend_config,
                                      int num_args);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_CUDA_TILE_CALL_H_
