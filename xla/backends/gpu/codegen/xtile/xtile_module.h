#ifndef XLA_BACKENDS_GPU_CODEGEN_XTILE_XTILE_MODULE_H_
#define XLA_BACKENDS_GPU_CODEGEN_XTILE_XTILE_MODULE_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

  // Load dialects before EmitXTileModule (null registration otherwise).
void LoadMlirDialectsForXTile(mlir::MLIRContext& mlir_context);

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> TileAndEmitXTileModule(
    absl::string_view fn_name, const HloFusionInstruction& fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    absl::Span<mlir::Type> opaque_args_types, mlir::MLIRContext& mlir_context,
    bool use_experimental_tiling);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_XTILE_XTILE_MODULE_H_
