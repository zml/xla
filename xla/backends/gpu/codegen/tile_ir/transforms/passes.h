#ifndef XLA_BACKENDS_GPU_CODEGEN_TILE_IR_TRANSFORMS_PASSES_H_
#define XLA_BACKENDS_GPU_CODEGEN_TILE_IR_TRANSFORMS_PASSES_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"      // IWYU pragma: keep
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_dialect.h"     // IWYU pragma: keep

namespace xla::gpu::tile_ir {


#define GEN_PASS_DECL
#include "xla/backends/gpu/codegen/tile_ir/transforms/passes.h.inc"

bool IsLowerableToCudaTile(mlir::Operation* module);

#define GEN_PASS_REGISTRATION
#include "xla/backends/gpu/codegen/tile_ir/transforms/passes.h.inc"

}  // namespace xla::gpu::tile_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TILE_IR_TRANSFORMS_PASSES_H_
