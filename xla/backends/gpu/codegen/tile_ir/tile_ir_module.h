#ifndef XLA_BACKENDS_GPU_CODEGEN_TILE_IR_TILE_IR_MODULE_H_
#define XLA_BACKENDS_GPU_CODEGEN_TILE_IR_TILE_IR_MODULE_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"

namespace xla::gpu::tile_ir {

// Loads the cuda_tile dialect; the bytecode writer rejects any other op.
void LoadMlirDialectsForCudaTile(mlir::MLIRContext& ctx);

struct TileIrModuleRequest {
  // A textual cuda_tile module, bare or wrapped in a builtin.module.
  absl::string_view text;
  // The cuda_tile.entry symbol to compile.
  absl::string_view entry_name;
  // Renames the entry before serializing; empty keeps its name.
  absl::string_view rename_entry_to;
  // Rejects an entry whose parameter count differs; -1 skips the check.
  int expected_num_entry_args = -1;
  // The bytecode format version handed to tileiras.
  uint8_t bytecode_major = 13;
  uint8_t bytecode_minor = 3;
};

// The post-rename textual module, for --xla_dump_to.
struct TileIrModuleText {
  std::string cuda_tile_text;
};

// Parses, renames the entry, drops every location (the bytecode writer
// rejects NameLoc) and writes Tile IR bytecode. A bad payload is an
// InvalidArgument.
absl::StatusOr<std::string> CudaTileTextToBytecode(
    const TileIrModuleRequest& req, mlir::MLIRContext& ctx,
    TileIrModuleText* dump_out);

}  // namespace xla::gpu::tile_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TILE_IR_TILE_IR_MODULE_H_
