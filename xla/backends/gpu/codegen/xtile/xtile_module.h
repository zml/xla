/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_CODEGEN_XTILE_XTILE_MODULE_H_
#define XLA_BACKENDS_GPU_CODEGEN_XTILE_XTILE_MODULE_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

void LoadMlirDialectsForXTile(mlir::MLIRContext& mlir_context);

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> TileAndEmitXTileModule(
    absl::string_view fn_name, const HloFusionInstruction& fusion,
    const se::DeviceDescription& device_info,
    const xla::xtile::BlockLevelParameters& block_level_parameters,
    absl::Span<mlir::Type> opaque_args_types, mlir::MLIRContext& mlir_context,
    bool use_experimental_tiling, bool enable_same_shape_multi_output_fusion);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_XTILE_XTILE_MODULE_H_
