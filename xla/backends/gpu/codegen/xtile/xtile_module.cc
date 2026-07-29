#include "xla/backends/gpu/codegen/xtile/xtile_module.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/codegen/experimental_fusion_emitter.h"
#include "xla/codegen/xtile/codegen/fusion_emitter.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/decision.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/tiling_from_block_parameters.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

void LoadMlirDialectsForXTile(mlir::MLIRContext& mlir_context) {
  mlir_context.loadDialect<mlir::arith::ArithDialect,
                           mlir::affine::AffineDialect,
                           mlir::LLVM::LLVMDialect, xla::XlaDialect,
                           xla::gpu::XlaGpuDialect, mlir::func::FuncDialect,
                           mlir::tensor::TensorDialect, xla::xtile::XTileDialect,
                           mlir::stablehlo::StablehloDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir_context.appendDialectRegistry(registry);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> TileAndEmitXTileModule(
    absl::string_view fn_name, const HloFusionInstruction& fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    absl::Span<mlir::Type> opaque_args_types, mlir::MLIRContext& mlir_context,
    bool use_experimental_tiling) {
  const HloComputation* computation = fusion.fused_instructions_computation();

  if (use_experimental_tiling) {
    using experimental::TiledHloComputation;
    using experimental::TilingSpace;

    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(&fusion);
    ASSIGN_OR_RETURN(std::unique_ptr<TilingSpace> tiling_space,
                     TilingSpace::Create(*fusion_adaptor, &mlir_context));

    VLOG(3) << "fusion instruction: " << fusion.ToString() << "\n";
    VLOG(3) << "tiling space: " << tiling_space->ToString();
    if (VLOG_IS_ON(4)) {
      XLA_VLOG_LINES(
          4, absl::StrCat("HLO module to reproduce:\n",
                          ExtractInstructionIntoNewModule(fusion)->ToString(
                              HloPrintOptions::ShortParsable())));
    }
    ASSIGN_OR_RETURN(
        llvm::SmallVector<int64_t> tile_sizes,
        GetTilingSpaceConcreteSizes(*tiling_space, block_level_parameters));
    RETURN_IF_ERROR(
        tiling_space->AssignTileSizes(xtile::GetPaddedTileSizes(tile_sizes)));

    ASSIGN_OR_RETURN(
        TiledHloComputation tiled_computation,
        TiledHloComputation::Tile(*fusion_adaptor, std::move(tiling_space)));
    tiled_computation.Simplify();
    tiled_computation.SortInstructionsPostOrder();
    if (Decision constraints = experimental::VerifyTritonConstraints(
            tiled_computation, device_info);
        !constraints) {
      return absl::InvalidArgumentError(
          absl::StrCat("Triton constraints violated during codegen: ",
                       constraints.Explain()));
    }
    VLOG(4) << "tiled computation: " << tiled_computation.ToString();
    return xtile::EmitXTileModule(
        fn_name, fusion, tiled_computation, mlir_context,
        absl::MakeSpan(opaque_args_types),
        std::make_optional(device_info.gpu_compute_capability()),
        block_level_parameters.num_tiles_per_pid);
  }
  SymbolicTileAnalysisOrError symbolic_tile_analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(
          *computation, &mlir_context,
          TritonEmitterConstraints::GetBuilder(device_info));

  if (std::holds_alternative<FusionDecision>(symbolic_tile_analysis_or)) {
    return Internal(
        "Unsupported fusion in TileAndEmitXTileModule: %s",
        std::get<FusionDecision>(symbolic_tile_analysis_or).Explain());
  }

  const auto& symbolic_tile_analysis =
      std::get<SymbolicTileAnalysis>(symbolic_tile_analysis_or);

  ASSIGN_OR_RETURN(Tiling tiling,
                   TilingFromAnnotatedFusion(symbolic_tile_analysis,
                                             block_level_parameters));

  return xtile::EmitXTileModule(
      fn_name, fusion, symbolic_tile_analysis, tiling, mlir_context,
      absl::MakeSpan(opaque_args_types),
      std::make_optional(device_info.gpu_compute_capability()));
}

}  // namespace xla::gpu
