#include "xla/backends/gpu/codegen/tile_ir/tile_ir_module.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "cuda_tile/Bytecode/Common/Version.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "xla/mlir/utils/error_util.h"

namespace xla::gpu::tile_ir {

void LoadMlirDialectsForCudaTile(mlir::MLIRContext& ctx) {
  ctx.loadDialect<mlir::cuda_tile::CudaTileDialect>();
}

namespace {

llvm::StringRef ToStringRef(absl::string_view s) {
  return llvm::StringRef(s.data(), s.size());
}

std::string TypeToString(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

std::string Echo(absl::string_view text) {
  constexpr size_t kLimit = 512;
  std::string escaped = absl::CHexEscape(text.substr(0, kLimit));
  if (text.size() > kLimit) absl::StrAppend(&escaped, "...");
  return escaped;
}

}  // namespace

absl::StatusOr<std::string> CudaTileTextToBytecode(
    const TileIrModuleRequest& req, mlir::MLIRContext& ctx,
    TileIrModuleText* dump_out) {
  mlir::BaseScopedDiagnosticHandler diagnostics(&ctx);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(ToStringRef(req.text), &ctx);
  if (!module) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CUDA Tile IR custom call payload does not parse: ",
        diagnostics.ConsumeStatus().message(), "; payload: ", Echo(req.text)));
  }

  std::vector<mlir::cuda_tile::ModuleOp> tile_modules;
  module->walk([&](mlir::cuda_tile::ModuleOp m) { tile_modules.push_back(m); });
  if (tile_modules.size() != 1) {
    diagnostics.ConsumeStatus().IgnoreError();
    return absl::InvalidArgumentError(
        absl::StrCat("CUDA Tile IR custom call payload must hold exactly one "
                     "cuda_tile.module, found ",
                     tile_modules.size()));
  }
  mlir::cuda_tile::ModuleOp tile_module = tile_modules.front();

  mlir::SymbolTable symbols(tile_module);
  auto entry =
      symbols.lookup<mlir::cuda_tile::EntryOp>(ToStringRef(req.entry_name));
  if (!entry) {
    std::vector<std::string> present;
    tile_module->walk([&](mlir::cuda_tile::EntryOp e) {
      present.push_back(e.getSymName().str());
    });
    diagnostics.ConsumeStatus().IgnoreError();
    return absl::InvalidArgumentError(absl::StrCat(
        "CUDA Tile IR custom call names entry '", req.entry_name,
        "' but the module defines [", absl::StrJoin(present, ", "), "]"));
  }

  // Every argument is a raw device pointer. EntryOp's verifier only rejects
  // rank != 0, so a scalar tile<i32> receiving the low bits of an address
  // would pass it.
  for (auto [i, ty] : llvm::enumerate(entry.getArgumentTypes())) {
    auto tile = mlir::dyn_cast<mlir::cuda_tile::TileType>(ty);
    if (!tile || tile.getRank() != 0 ||
        !mlir::isa<mlir::cuda_tile::PointerType>(tile.getElementType())) {
      diagnostics.ConsumeStatus().IgnoreError();
      return absl::InvalidArgumentError(absl::StrCat(
          "CUDA Tile IR kernel '", req.entry_name, "' parameter ", i,
          " has type ", TypeToString(ty),
          "; every custom-call argument is a raw device pointer, so each "
          "entry parameter must be a rank-0 tile<ptr<T>>"));
    }
  }

  const int num_entry_args = entry.getArgumentTypes().size();
  if (req.expected_num_entry_args >= 0 &&
      num_entry_args != req.expected_num_entry_args) {
    diagnostics.ConsumeStatus().IgnoreError();
    return absl::InvalidArgumentError(absl::StrCat(
        "CUDA Tile IR kernel '", req.entry_name, "' takes ", num_entry_args,
        " arguments but the custom call supplies ",
        req.expected_num_entry_args));
  }

  if (!req.rename_entry_to.empty()) {
    mlir::SymbolTable::setSymbolName(entry, ToStringRef(req.rename_entry_to));
  }

  if (dump_out != nullptr) {
    llvm::raw_string_ostream os(dump_out->cuda_tile_text);
    tile_module->print(os);
  }

  // The bytecode writer rejects NameLoc.
  mlir::Location unknown = mlir::UnknownLoc::get(&ctx);
  tile_module->walk([&](mlir::Operation* op) { op->setLoc(unknown); });

  std::optional<mlir::cuda_tile::BytecodeVersion> version =
      mlir::cuda_tile::BytecodeVersion::fromVersion(req.bytecode_major,
                                                    req.bytecode_minor);
  if (!version.has_value()) {
    diagnostics.ConsumeStatus().IgnoreError();
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported Tile IR bytecode version ",
                     req.bytecode_major, ".", req.bytecode_minor));
  }

  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  if (mlir::failed(mlir::cuda_tile::writeBytecode(os, tile_module, *version))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CUDA Tile IR kernel '", req.entry_name,
        "' could not be serialized as bytecode ", req.bytecode_major, ".",
        req.bytecode_minor, ": ", diagnostics.ConsumeStatus().message()));
  }
  diagnostics.ConsumeStatus().IgnoreError();
  return bytecode;
}

}  // namespace xla::gpu::tile_ir
