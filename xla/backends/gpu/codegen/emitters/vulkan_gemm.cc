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

#include "xla/backends/gpu/codegen/emitters/vulkan_gemm.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {
namespace {

using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::OpBuilder;
using mlir::RankedTensorType;
using mlir::TypeRange;
using mlir::Value;
using mlir::ValueRange;
namespace arith = mlir::arith;
namespace scf = mlir::scf;
namespace tensor = mlir::tensor;

constexpr int64_t kTileSize = 16;
constexpr int64_t kThreadsPerBlock = kTileSize * kTileSize;

Value IndexConstant(ImplicitLocOpBuilder& builder, int64_t value) {
  return arith::ConstantIndexOp::create(builder, value);
}

Value F32Constant(ImplicitLocOpBuilder& builder, float value) {
  return arith::ConstantFloatOp::create(builder, builder.getF32Type(),
                                        llvm::APFloat(value));
}

Value ZeroOfType(ImplicitLocOpBuilder& builder, mlir::Type type) {
  return arith::ConstantOp::create(builder, builder.getZeroAttr(type));
}

Value SelectTensorElementOrZero(ImplicitLocOpBuilder& builder, Value condition,
                                Value tensor_value, ValueRange indices) {
  mlir::Type element_type =
      mlir::cast<mlir::ShapedType>(tensor_value.getType()).getElementType();
  scf::IfOp if_op = scf::IfOp::create(builder, TypeRange{element_type},
                                      condition, /*withElseRegion=*/true);
  OpBuilder then_builder = if_op.getThenBodyBuilder();
  ImplicitLocOpBuilder implicit_then(builder.getLoc(), then_builder);
  Value loaded =
      tensor::ExtractOp::create(implicit_then, tensor_value, indices);
  scf::YieldOp::create(implicit_then, loaded);
  OpBuilder else_builder = if_op.getElseBodyBuilder();
  ImplicitLocOpBuilder implicit_else(builder.getLoc(), else_builder);
  scf::YieldOp::create(implicit_else,
                       ZeroOfType(implicit_else, element_type));
  return if_op.getResult(0);
}

bool IsStaticRowMajorBf16Matrix(const Shape& shape) {
  if (!shape.IsArray() || shape.dimensions().size() != 2 ||
      shape.element_type() != PrimitiveType::BF16 || !shape.has_layout() ||
      !shape.layout().tiles().empty() ||
      !LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return false;
  }
  return !shape.is_dynamic_dimension(0) && !shape.is_dynamic_dimension(1);
}

}  // namespace

std::optional<VulkanGemmConfig> MatchVulkanGemm(
    const HloFusionAnalysis& analysis) {
  const se::DeviceDescription& device = analysis.device_info();
  const se::VulkanComputeCapability* capability =
      device.gpu_compute_capability().vulkan_compute_capability();
  if (capability == nullptr || !capability->shader_bfloat16() ||
      !capability->storage_buffer_16bit_access() ||
      device.threads_per_block_limit() < kThreadsPerBlock ||
      analysis.fusion_root_count() != 1) {
    return std::nullopt;
  }

  const HloInstruction& dot = analysis.fusion_root(0).instruction();
  if (dot.opcode() != HloOpcode::kDot || dot.operand_count() != 2 ||
      dot.operand(0)->opcode() != HloOpcode::kParameter ||
      dot.operand(1)->opcode() != HloOpcode::kParameter ||
      dot.parent()->instruction_count() != 3) {
    return std::nullopt;
  }

  const Shape& lhs = dot.operand(0)->shape();
  const Shape& rhs = dot.operand(1)->shape();
  const Shape& output = dot.shape();
  if (!IsStaticRowMajorBf16Matrix(lhs) ||
      !IsStaticRowMajorBf16Matrix(rhs) ||
      !IsStaticRowMajorBf16Matrix(output)) {
    return std::nullopt;
  }

  const DotDimensionNumbers& dimensions = dot.dot_dimension_numbers();
  if (dimensions.lhs_batch_dimensions_size() != 0 ||
      dimensions.rhs_batch_dimensions_size() != 0 ||
      dimensions.lhs_contracting_dimensions_size() != 1 ||
      dimensions.rhs_contracting_dimensions_size() != 1 ||
      dimensions.lhs_contracting_dimensions(0) != 1 ||
      dimensions.rhs_contracting_dimensions(0) != 0 ||
      lhs.dimensions(1) != rhs.dimensions(0) ||
      output.dimensions(0) != lhs.dimensions(0) ||
      output.dimensions(1) != rhs.dimensions(1) ||
      lhs.dimensions(0) <= 0 || lhs.dimensions(1) <= 0 ||
      rhs.dimensions(1) <= 0) {
    return std::nullopt;
  }

  return VulkanGemmConfig{lhs.dimensions(0), rhs.dimensions(1),
                          lhs.dimensions(1)};
}

VulkanGemmEmitter::VulkanGemmEmitter(VulkanGemmConfig config)
    : m_(config.m), n_(config.n), k_(config.k) {}

LaunchDimensions VulkanGemmEmitter::launch_dimensions() const {
  return LaunchDimensions(
      se::BlockDim((n_ + kTileSize - 1) / kTileSize,
                   (m_ + kTileSize - 1) / kTileSize, 1),
      se::ThreadDim(kTileSize, kTileSize, 1));
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
VulkanGemmEmitter::CreateMLIRModule(
    mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  mlir::OpBuilder builder(&mlir_context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion.name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);
  ASSIGN_OR_RETURN(mlir::func::FuncOp entry_function,
                   emitters::EmitKernelApi(*module, fusion, buffer_assignment,
                                           GetDefaultBufferAlignment(),
                                           entry_function_name));
  SetBackendKind(&mlir_context, entry_function, BackendKind::kGpu);
  emitters::SetIndexDataLayout(*module, fusion);
  RETURN_IF_ERROR(EmitKernel(entry_function, fusion));
  TF_RET_CHECK(mlir::succeeded(mlir::verify(*module)))
      << "Vulkan GEMM emitter produced invalid MLIR";
  return module;
}

std::optional<IndexingMap>
VulkanGemmEmitter::ComputeThreadIdToOutputIndexing(
    int64_t, mlir::MLIRContext*) const {
  return std::nullopt;
}

std::optional<std::vector<IndexingMap>>
VulkanGemmEmitter::ComputeThreadIdToInputIndexing(
    int64_t, mlir::MLIRContext*) const {
  return std::nullopt;
}

absl::Status VulkanGemmEmitter::EmitEntryFunction(
    const emitters::PartitionedComputations&,
    const emitters::CallTargetProvider&, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  return EmitKernel(entry_function, fusion);
}

absl::Status VulkanGemmEmitter::EmitKernel(
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());
  Value lhs = entry_function.getArgument(0);
  Value rhs = entry_function.getArgument(1);
  Value output = entry_function.getArgument(fusion.operand_count());

  Value thread_col = EmitThreadId(builder, 0);
  Value thread_row = EmitThreadId(builder, 1);
  Value block_col = EmitBlockId(builder, 0);
  Value block_row = EmitBlockId(builder, 1);
  Value tile_size = IndexConstant(builder, kTileSize);
  Value row = arith::AddIOp::create(
      builder, arith::MulIOp::create(builder, block_row, tile_size),
      thread_row);
  Value col = arith::AddIOp::create(
      builder, arith::MulIOp::create(builder, block_col, tile_size),
      thread_col);

  RankedTensorType tile_type =
      RankedTensorType::get({kTileSize, kTileSize}, builder.getBF16Type());
  Value lhs_tile = AllocateSharedOp::create(builder, tile_type);
  Value rhs_tile = AllocateSharedOp::create(builder, tile_type);
  Value zero_index = IndexConstant(builder, 0);
  Value one_index = IndexConstant(builder, 1);
  Value tile_count = IndexConstant(builder, (k_ + kTileSize - 1) / kTileSize);
  scf::ForOp tile_loop = scf::ForOp::create(
      builder, zero_index, tile_count, one_index,
      ValueRange{F32Constant(builder, 0.0f), lhs_tile, rhs_tile});
  {
    OpBuilder::InsertionGuard tile_guard(builder);
    builder.setInsertionPointToStart(tile_loop.getBody());
    Value tile = tile_loop.getInductionVar();
    Value accumulator = tile_loop.getRegionIterArgs()[0];
    Value loop_lhs_tile = tile_loop.getRegionIterArgs()[1];
    Value loop_rhs_tile = tile_loop.getRegionIterArgs()[2];
    Value tile_base = arith::MulIOp::create(builder, tile, tile_size);
    Value lhs_k = arith::AddIOp::create(builder, tile_base, thread_col);
    Value rhs_k = arith::AddIOp::create(builder, tile_base, thread_row);

    Value lhs_row_valid = arith::CmpIOp::create(
        builder, arith::CmpIPredicate::ult, row, IndexConstant(builder, m_));
    Value lhs_k_valid = arith::CmpIOp::create(
        builder, arith::CmpIPredicate::ult, lhs_k, IndexConstant(builder, k_));
    Value lhs_valid =
        arith::AndIOp::create(builder, lhs_row_valid, lhs_k_valid);
    Value lhs_value = SelectTensorElementOrZero(
        builder, lhs_valid, lhs, ValueRange{row, lhs_k});
    Value written_lhs = tensor::InsertOp::create(
        builder, lhs_value, loop_lhs_tile,
        ValueRange{thread_row, thread_col});

    Value rhs_k_valid = arith::CmpIOp::create(
        builder, arith::CmpIPredicate::ult, rhs_k, IndexConstant(builder, k_));
    Value rhs_col_valid = arith::CmpIOp::create(
        builder, arith::CmpIPredicate::ult, col, IndexConstant(builder, n_));
    Value rhs_valid =
        arith::AndIOp::create(builder, rhs_k_valid, rhs_col_valid);
    Value rhs_value = SelectTensorElementOrZero(
        builder, rhs_valid, rhs, ValueRange{rhs_k, col});
    Value written_rhs = tensor::InsertOp::create(
        builder, rhs_value, loop_rhs_tile,
        ValueRange{thread_row, thread_col});

    SmallVector<mlir::Type> tile_types{written_lhs.getType(),
                                       written_rhs.getType()};
    auto synchronized =
        SyncThreadsOp::create(builder, TypeRange(tile_types),
                              ValueRange{written_lhs, written_rhs})
            .getResults();
    Value synchronized_lhs = synchronized[0];
    Value synchronized_rhs = synchronized[1];

    scf::ForOp k_loop = scf::ForOp::create(
        builder, zero_index, tile_size, one_index, ValueRange{accumulator});
    {
      OpBuilder::InsertionGuard k_guard(builder);
      builder.setInsertionPointToStart(k_loop.getBody());
      Value inner_k = k_loop.getInductionVar();
      Value sum = k_loop.getRegionIterArgs()[0];
      Value lhs_element = tensor::ExtractOp::create(
          builder, synchronized_lhs, ValueRange{thread_row, inner_k});
      Value rhs_element = tensor::ExtractOp::create(
          builder, synchronized_rhs, ValueRange{inner_k, thread_col});
      Value product = arith::MulFOp::create(
          builder,
          arith::ExtFOp::create(builder, builder.getF32Type(), lhs_element),
          arith::ExtFOp::create(builder, builder.getF32Type(), rhs_element));
      Value updated_sum = arith::AddFOp::create(builder, sum, product);
      scf::YieldOp::create(builder, updated_sum);
    }

    auto reusable_tiles =
        SyncThreadsOp::create(builder, TypeRange(tile_types),
                              ValueRange{synchronized_lhs, synchronized_rhs})
            .getResults();
    scf::YieldOp::create(
        builder, ValueRange{k_loop.getResult(0), reusable_tiles[0],
                            reusable_tiles[1]});
  }

  Value row_valid = arith::CmpIOp::create(
      builder, arith::CmpIPredicate::ult, row, IndexConstant(builder, m_));
  Value col_valid = arith::CmpIOp::create(
      builder, arith::CmpIPredicate::ult, col, IndexConstant(builder, n_));
  Value output_valid = arith::AndIOp::create(builder, row_valid, col_valid);
  scf::IfOp store = scf::IfOp::create(
      builder, TypeRange{output.getType()}, output_valid,
      /*withElseRegion=*/true);
  OpBuilder then_builder = store.getThenBodyBuilder();
  ImplicitLocOpBuilder implicit_then(builder.getLoc(), then_builder);
  Value result = arith::TruncFOp::create(
      implicit_then, implicit_then.getBF16Type(), tile_loop.getResult(0));
  Value stored = tensor::InsertOp::create(implicit_then, result, output,
                                          ValueRange{row, col});
  scf::YieldOp::create(implicit_then, stored);
  OpBuilder else_builder = store.getElseBodyBuilder();
  ImplicitLocOpBuilder implicit_else(builder.getLoc(), else_builder);
  scf::YieldOp::create(implicit_else, output);
  mlir::func::ReturnOp::create(builder, store.getResult(0));
  return absl::OkStatus();
}

}  // namespace xla::gpu
