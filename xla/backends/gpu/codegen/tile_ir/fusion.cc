#include "xla/backends/gpu/codegen/tile_ir/fusion.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/tile_ir/tileiras_compiler.h"
#include "xla/backends/gpu/codegen/tile_ir/transforms/passes.h"
#include "xla/backends/gpu/codegen/xtile/xtile_module.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/computation_fingerprint.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu::tile_ir {
namespace {

int64_t NumberOfTiles(absl::Span<const int64_t> dimensions,
                      absl::Span<const int64_t> tile_sizes) {
  int64_t tiles = 1;
  for (auto [dim_size, tile_size] : llvm::zip(dimensions, tile_sizes)) {
    tiles *= (dim_size + tile_size - 1) / tile_size;
  }
  return tiles;
}

// CancelledError: backend declines (vs hard failure); cache once per fusion.
bool IsDeclined(const absl::Status& status) {
  return absl::IsCancelled(status);
}

absl::StatusOr<KernelReuseCache::Entry> BuildTileIrKernel(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    const HloFusionAnalysis& analysis,
    const BlockLevelParameters& block_level_parameters,
    absl::string_view kernel_name, int64_t num_arguments) {
  const DebugOptions& debug_options =
      fusion.GetModule()->config().debug_options();

  auto borrowed_context = ir_emitter_context.BorrowMlirContext();
  mlir::MLIRContext& mlir_context = **borrowed_context;
  LoadMlirDialectsForXTile(mlir_context);
  mlir_context.loadDialect<mlir::cuda_tile::CudaTileDialect>();

  llvm::SmallVector<mlir::Type> no_opaque_args;
  ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      TileAndEmitXTileModule(
          kernel_name, fusion, ir_emitter_context.gpu_device_info(),
          block_level_parameters, absl::MakeSpan(no_opaque_args), mlir_context,
          debug_options.xla_gpu_experimental_enable_tiling_propagation()));

  if (!IsLowerableToCudaTile(module.get())) {
    return absl::CancelledError(
        absl::StrCat(fusion.name(),
                     ": the XTile module uses ops the CUDA Tile IR backend "
                     "does not implement"));
  }

  // XLA_TILE_IR_DUMP_DIR also collects the XTile module, so a kernel that
  // misbehaves can be replayed through xtile_to_tilebc offline.
  const char* dump_dir = std::getenv("XLA_TILE_IR_DUMP_DIR");
  if (dump_dir != nullptr) {
    std::string text;
    llvm::raw_string_ostream os(text);
    module->print(os);
    tsl::WriteStringToFile(tsl::Env::Default(),
                           absl::StrCat(dump_dir, "/", kernel_name, ".xtile"),
                           text)
        .IgnoreError();
  }

  mlir::PassManager pm(&mlir_context);
  pm.addPass(emitters::createSimplifyAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createXTileLowerToCudaTilePass());
  if (mlir::failed(pm.run(*module))) {
  // Lowering failure after IsLowerable: fall back to Triton.
    LOG(WARNING) << "CUDA Tile IR lowering failed for " << fusion.name()
                 << " after IsLowerableToCudaTile accepted it; falling back to "
                    "Triton.";
    return absl::CancelledError("lowering failed");
  }

  mlir::cuda_tile::ModuleOp tile_module;
  module->walk([&](mlir::cuda_tile::ModuleOp m) { tile_module = m; });
  if (!tile_module) {
    return Internal("Lowering %s produced no cuda_tile.module.", fusion.name());
  }

  if (dump_dir != nullptr) {
    std::string text;
    llvm::raw_string_ostream os(text);
    tile_module->print(os);
    tsl::WriteStringToFile(
        tsl::Env::Default(),
        absl::StrCat(dump_dir, "/", kernel_name, ".cuda_tile.mlir"), text)
        .IgnoreError();
  }

  // Bytecode writer rejects NameLoc; clear locations (no DILoc).
  mlir::Location unknown = mlir::UnknownLoc::get(&mlir_context);
  tile_module->walk([&](mlir::Operation* op) { op->setLoc(unknown); });
  tile_module->setLoc(unknown);

  // Bytecode 13.3: first with mmaf_scaled + f4E2M1FN.
  auto version = mlir::cuda_tile::BytecodeVersion::fromVersion(13, 3);
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  if (mlir::failed(
          mlir::cuda_tile::writeBytecode(os, tile_module, *version))) {
    return Internal("Failed to serialize %s to Tile IR bytecode.",
                    fusion.name());
  }

  ASSIGN_OR_RETURN(
      std::vector<uint8_t> cubin,
      CompileTileIrBytecode(bytecode, ir_emitter_context.gpu_device_info(),
                            debug_options));

  int64_t num_tiles =
      NumberOfTiles(analysis.fusion_root(0).shape().dimensions(),
                    block_level_parameters.output_tile_sizes[0]);

  LOG(INFO) << "Emitted " << fusion.name() << " as CUDA Tile IR kernel "
            << kernel_name << " over " << num_tiles << " tiles, output "
            << analysis.fusion_root(0).shape().ToString() << " tile {"
            << absl::StrJoin(block_level_parameters.output_tile_sizes[0], ",")
            << "} args=" << num_arguments << " ("
            << cubin.size() << " byte cubin).";

  // Tile IR grid is the tile grid; block dims are (1,1,1).
  return KernelReuseCache::Entry{
      std::string(kernel_name),
      LaunchDimensions(se::BlockDim(num_tiles), se::ThreadDim(1, 1, 1)),
      /*cluster_dim=*/std::nullopt,
      /*shmem_bytes=*/0,
      std::move(cubin)};
}

}  // namespace

AsyncThunkSequence TileIrFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  if (!analysis_.fusion_backend_config().has_block_level_fusion_config()) {
    return AsyncThunkSequence(absl::FailedPreconditionError(absl::StrCat(
        "CUDA Tile IR fusion ", fusion.name(), " carries ", kTileIrFusionKind,
        " but no block-level fusion config; only TileIrBackend::ApplyConfig "
        "should produce this kind, and it always sets one.")));
  }
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          analysis_.fusion_backend_config().block_level_fusion_config());

  ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context.buffer_assignment(),
                       GetDefaultBufferAlignment(), &fusion));

  // Kernel name must not be per-instance: SE caches cubins by name.
  std::string contracting;
  if (const HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
          *fusion.fused_instructions_computation(), HloOpcode::kScaledDot)) {
    if (absl::StatusOr<Tile> dot_tile = dot->backend_config<Tile>();
        dot_tile.ok()) {
      contracting = absl::StrJoin(dot_tile->sizes(), "x");
    }
  }
  std::string discriminator = absl::StrCat(
      "TileIrFusion,blc=",
      analysis_.fusion_backend_config().block_level_fusion_config()
          .ShortDebugString(),
      ",k=", contracting);
  std::string kernel_name = absl::StrCat(
      "tile_ir_",
      absl::Hex(absl::HashOf(emitters::GetComputationFingerprint(
          fusion.fused_instructions_computation(), kernel_arguments.args(),
          discriminator))));

  auto generate = [&]() -> tsl::Future<KernelReuseCache::Entry> {
    return tsl::Future<KernelReuseCache::Entry>(BuildTileIrKernel(
        ir_emitter_context, fusion, analysis_, block_level_parameters,
        kernel_name, kernel_arguments.args().size()));
  };
  auto [future, was_cached] = ir_emitter_context.kernel_cache().GetWithStatus(
      fusion.fused_instructions_computation(), kernel_arguments.args(),
      discriminator, generate);

  const absl::StatusOr<const KernelReuseCache::Entry*>& entry = future.Await();
  if (!entry.ok()) {
    if (IsDeclined(entry.status())) {
      return AsyncThunkSequence(absl::FailedPreconditionError(absl::StrCat(
          "CUDA Tile IR could not lower ", fusion.name(),
          " at emit time, but TileIrBackend::CanLower accepted it at bid "
          "time. The bid-time check and the lowering pass have drifted: ",
          entry.status().message())));
    }
    return AsyncThunkSequence(entry.status());
  }

  ASSIGN_OR_RETURN(
      CustomKernel custom_kernel,
      kernel::CreateOwnedCubinCustomKernel(
          (*entry)->kernel_name, (*entry)->binary,
          kernel_arguments.args().size(),
          (*entry)->launch_dimensions.block_counts(),
          (*entry)->launch_dimensions.thread_counts_per_block(),
          (*entry)->shmem_bytes));

  return ThunkSequence::Of<CustomKernelThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          &fusion, ir_emitter_context.GetNextThunkId()),
      std::move(custom_kernel), kernel_arguments);
}

}  // namespace xla::gpu::tile_ir
