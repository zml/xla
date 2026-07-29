
#include <cstdint>
#include <optional>
#include <string>

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "cuda_tile/Bytecode/Common/Version.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

namespace {

constexpr char kModuleText[] = R"mlir(
cuda_tile.module @m {
  entry @k(%p : tile<ptr<f32>>) {
    %off = iota : tile<128xi32>
    return
  }
}
)mlir";

}  // namespace

int main(int argc, char** argv) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::cuda_tile::CudaTileDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> owning =
      mlir::parseSourceString<mlir::ModuleOp>(kModuleText, &context);
  if (!owning) {
    llvm::errs() << "FAIL: could not parse the cuda_tile module\n";
    return 1;
  }

  mlir::cuda_tile::ModuleOp tile_module;
  owning->walk([&](mlir::cuda_tile::ModuleOp m) { tile_module = m; });
  if (!tile_module) {
    llvm::errs() << "FAIL: no cuda_tile.module found after parsing\n";
    return 1;
  }
  llvm::outs() << "parsed cuda_tile.module @" << tile_module.getSymName()
               << "\n";

  std::optional<mlir::cuda_tile::BytecodeVersion> version =
      mlir::cuda_tile::BytecodeVersion::fromVersion(13, 3);
  if (!version) {
    llvm::errs() << "FAIL: bytecode version 13.3 is not supported\n";
    return 1;
  }

  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  if (mlir::failed(
          mlir::cuda_tile::writeBytecode(os, tile_module, *version))) {
    llvm::errs() << "FAIL: writeBytecode returned failure\n";
    return 1;
  }
  os.flush();

  if (buffer.empty()) {
    llvm::errs() << "FAIL: bytecode buffer is empty\n";
    return 1;
  }

  llvm::outs() << "target version: " << version->toString() << "\n";
  llvm::outs() << "bytecode size: " << buffer.size() << " bytes\n";
  llvm::outs() << "first 8 bytes:";
  for (int i = 0; i < 8 && i < static_cast<int>(buffer.size()); ++i) {
    llvm::outs() << llvm::format(" %02x",
                                 static_cast<uint8_t>(buffer[i]));
  }
  llvm::outs() << "\n";
  llvm::outs() << "OK\n";
  return 0;
}
