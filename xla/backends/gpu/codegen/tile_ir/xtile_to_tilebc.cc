
#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "xla/backends/gpu/codegen/tile_ir/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"

namespace {

llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                          llvm::cl::desc("<input xtile mlir>"),
                                          llvm::cl::Required);
llvm::cl::opt<std::string> output_filename(
    "o", llvm::cl::desc("write CUDA Tile IR bytecode here"),
    llvm::cl::value_desc("filename"));

}  // namespace

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "xtile -> CUDA Tile IR\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect, mlir::NVVM::NVVMDialect,
                  mlir::stablehlo::StablehloDialect,
                  mlir::cuda_tile::CudaTileDialect, xla::XlaDialect,
                  xla::xtile::XTileDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFileOrSTDIN(input_filename);
  if (!file) {
    llvm::errs() << "cannot read " << input_filename << "\n";
    return 1;
  }
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(file.get()->getBuffer(), &context);
  if (!module) {
    llvm::errs() << "cannot parse " << input_filename << "\n";
    return 1;
  }

  mlir::PassManager pm(&context);
  pm.addPass(xla::emitters::createSimplifyAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(xla::gpu::tile_ir::createXTileLowerToCudaTilePass());
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "lowering failed\n";
    return 1;
  }

  module->print(llvm::outs());
  llvm::outs() << "\n";

  if (output_filename.empty()) return 0;

  mlir::cuda_tile::ModuleOp tile_module;
  module->walk([&](mlir::cuda_tile::ModuleOp m) { tile_module = m; });
  if (!tile_module) {
    llvm::errs() << "no cuda_tile.module was produced\n";
    return 1;
  }
  auto version = mlir::cuda_tile::BytecodeVersion::fromVersion(13, 3);
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  if (mlir::failed(
          mlir::cuda_tile::writeBytecode(os, tile_module, *version))) {
    llvm::errs() << "writeBytecode failed\n";
    return 1;
  }
  std::error_code ec;
  llvm::raw_fd_ostream out(output_filename, ec);
  if (ec) {
    llvm::errs() << "cannot write " << output_filename << ": " << ec.message()
                 << "\n";
    return 1;
  }
  out << buffer;
  llvm::errs() << "wrote " << buffer.size() << " bytes of Tile IR bytecode\n";
  return 0;
}
