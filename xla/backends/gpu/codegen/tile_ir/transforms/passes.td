#ifndef XLA_BACKENDS_GPU_CODEGEN_TILE_IR_PASSES_TD_
#define XLA_BACKENDS_GPU_CODEGEN_TILE_IR_PASSES_TD_

include "mlir/Pass/PassBase.td"

def XTileLowerToCudaTilePass :
    Pass<"xtile-lower-to-cuda-tile", "mlir::ModuleOp"> {
  let summary = "Lowers an xtile entry function to a CUDA Tile IR kernel.";

  let description = [{
    Rewrites the `xtile.entry_func` in the module into a `cuda_tile.module`
    holding a single `cuda_tile.entry`, which `tileiras` can assemble into a
    cubin.

    This is the CUDA Tile IR sibling of `triton-xla-lower-xtile`: both consume
    the same backend-neutral XTile module produced by `TileAndEmitXTileModule`,
    so a fusion can be routed to either backend without re-tiling it.

    The pass is deliberately conservative. XTile is a much larger surface than
    the subset a block-scaled matmul needs, and an op this pass does not
    understand is an error rather than something to approximate -- the caller
    is expected to fall back to the Triton rung. See
    `IsLowerableToCudaTile` for the supported subset.
  }];

  let dependentDialects = [
    "::mlir::cuda_tile::CudaTileDialect",
    "::xla::xtile::XTileDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::stablehlo::StablehloDialect",
  ];
}

#endif  // XLA_BACKENDS_GPU_CODEGEN_TILE_IR_PASSES_TD_
