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
#include "absl/status/status_macros.h"
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

Value I32Constant(ImplicitLocOpBuilder& builder, uint32_t value) {
  return arith::ConstantOp::create(builder, builder.getI32IntegerAttr(value));
}

mlir::Type PortableBf16StorageType(mlir::Type type, OpBuilder& builder) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (shaped == nullptr || !shaped.getElementType().isBF16()) {
    return type;
  }
  return shaped.clone(builder.getI16Type());
}

void UsePortableBf16StorageAbi(mlir::func::FuncOp function,
                               OpBuilder& builder) {
  mlir::FunctionType type = function.getFunctionType();
  SmallVector<mlir::Type> inputs;
  SmallVector<mlir::Type> results;
  inputs.reserve(type.getNumInputs());
  results.reserve(type.getNumResults());
  for (mlir::Type input : type.getInputs()) {
    inputs.push_back(PortableBf16StorageType(input, builder));
  }
  for (mlir::Type result : type.getResults()) {
    results.push_back(PortableBf16StorageType(result, builder));
  }
  function.setType(mlir::FunctionType::get(function.getContext(), inputs,
                                            results));
}

Value Bf16BitsToF32(ImplicitLocOpBuilder& builder, Value value) {
  Value bits = arith::ExtUIOp::create(builder, builder.getI32Type(), value);
  bits = arith::ShLIOp::create(builder, bits, I32Constant(builder, 16));
  return arith::BitcastOp::create(builder, builder.getF32Type(), bits);
}

Value F32ToBf16Bits(ImplicitLocOpBuilder& builder, Value value) {
  Value bits = arith::BitcastOp::create(builder, builder.getI32Type(), value);
  Value upper = arith::ShRUIOp::create(builder, bits, I32Constant(builder, 16));
  Value lsb = arith::AndIOp::create(builder, upper, I32Constant(builder, 1));
  Value bias = arith::AddIOp::create(builder, I32Constant(builder, 0x7fff), lsb);
  Value rounded = arith::AddIOp::create(builder, bits, bias);
  Value encoded = arith::TruncIOp::create(
      builder, builder.getI16Type(),
      arith::ShRUIOp::create(builder, rounded, I32Constant(builder, 16)));

  Value exponent = arith::AndIOp::create(
      builder, bits, I32Constant(builder, 0x7f800000));
  Value mantissa = arith::AndIOp::create(
      builder, bits, I32Constant(builder, 0x007fffff));
  Value exponent_all_ones = arith::CmpIOp::create(
      builder, arith::CmpIPredicate::eq, exponent,
      I32Constant(builder, 0x7f800000));
  Value mantissa_nonzero = arith::CmpIOp::create(
      builder, arith::CmpIPredicate::ne, mantissa, I32Constant(builder, 0));
  Value is_nan =
      arith::AndIOp::create(builder, exponent_all_ones, mantissa_nonzero);
  Value quiet_nan = arith::OrIOp::create(
      builder, encoded,
      arith::TruncIOp::create(builder, builder.getI16Type(),
                              I32Constant(builder, 0x40)));
  return arith::SelectOp::create(builder, is_nan, quiet_nan, encoded);
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
  if (capability == nullptr ||
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
      dimensions.lhs_contracting_dimensions(0) != 1) {
    return std::nullopt;
  }

  int64_t m = lhs.dimensions(0);
  int64_t k = lhs.dimensions(1);
  int64_t n;
  VulkanGemmRhsLayout rhs_layout;
  switch (dimensions.rhs_contracting_dimensions(0)) {
    case 0:
      n = rhs.dimensions(1);
      rhs_layout = VulkanGemmRhsLayout::kKxN;
      if (k != rhs.dimensions(0)) {
        return std::nullopt;
      }
      break;
    case 1:
      n = rhs.dimensions(0);
      rhs_layout = VulkanGemmRhsLayout::kNxK;
      if (k != rhs.dimensions(1)) {
        return std::nullopt;
      }
      break;
    default:
      return std::nullopt;
  }

  if (output.dimensions(0) != m || output.dimensions(1) != n || m <= 0 ||
      n <= 0 || k <= 0) {
    return std::nullopt;
  }

  return VulkanGemmConfig{m, n, k, rhs_layout};
}

VulkanGemmEmitter::VulkanGemmEmitter(VulkanGemmConfig config)
    : m_(config.m),
      n_(config.n),
      k_(config.k),
      rhs_layout_(config.rhs_layout) {}

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
  ABSL_ASSIGN_OR_RETURN(mlir::func::FuncOp entry_function,
                   emitters::EmitKernelApi(*module, fusion, buffer_assignment,
                                           GetDefaultBufferAlignment(),
                                           entry_function_name));
  UsePortableBf16StorageAbi(entry_function, builder);
  SetBackendKind(&mlir_context, entry_function, BackendKind::kGpu);
  emitters::SetIndexDataLayout(*module, fusion);
  ABSL_RETURN_IF_ERROR(EmitKernel(entry_function, fusion));
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
      RankedTensorType::get({kTileSize, kTileSize}, builder.getF32Type());
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
    // Keep thread X on the contiguous RHS dimension. For [N,K], transpose the
    // cooperative load when inserting it into the logical [K,N] shared tile.
    Value rhs_k_thread = rhs_layout_ == VulkanGemmRhsLayout::kKxN
                             ? thread_row
                             : thread_col;
    Value rhs_n_thread = rhs_layout_ == VulkanGemmRhsLayout::kKxN
                             ? thread_col
                             : thread_row;
    Value rhs_k = arith::AddIOp::create(builder, tile_base, rhs_k_thread);
    Value rhs_n = arith::AddIOp::create(
        builder, arith::MulIOp::create(builder, block_col, tile_size),
        rhs_n_thread);

    Value lhs_row_valid = arith::CmpIOp::create(
        builder, arith::CmpIPredicate::ult, row, IndexConstant(builder, m_));
    Value lhs_k_valid = arith::CmpIOp::create(
        builder, arith::CmpIPredicate::ult, lhs_k, IndexConstant(builder, k_));
    Value lhs_valid =
        arith::AndIOp::create(builder, lhs_row_valid, lhs_k_valid);
    Value lhs_value = SelectTensorElementOrZero(
        builder, lhs_valid, lhs, ValueRange{row, lhs_k});
    lhs_value = Bf16BitsToF32(builder, lhs_value);
    Value written_lhs = tensor::InsertOp::create(
        builder, lhs_value, loop_lhs_tile,
        ValueRange{thread_row, thread_col});

    Value rhs_k_valid = arith::CmpIOp::create(
        builder, arith::CmpIPredicate::ult, rhs_k, IndexConstant(builder, k_));
    Value rhs_n_valid = arith::CmpIOp::create(
        builder, arith::CmpIPredicate::ult, rhs_n, IndexConstant(builder, n_));
    Value rhs_valid =
        arith::AndIOp::create(builder, rhs_k_valid, rhs_n_valid);
    SmallVector<Value, 2> rhs_indices =
        rhs_layout_ == VulkanGemmRhsLayout::kKxN
            ? SmallVector<Value, 2>{rhs_k, rhs_n}
            : SmallVector<Value, 2>{rhs_n, rhs_k};
    SmallVector<Value, 2> rhs_tile_indices =
        rhs_layout_ == VulkanGemmRhsLayout::kKxN
            ? SmallVector<Value, 2>{thread_row, thread_col}
            : SmallVector<Value, 2>{thread_col, thread_row};
    Value rhs_value = SelectTensorElementOrZero(
        builder, rhs_valid, rhs, ValueRange(rhs_indices));
    rhs_value = Bf16BitsToF32(builder, rhs_value);
    Value written_rhs = tensor::InsertOp::create(
        builder, rhs_value, loop_rhs_tile, ValueRange(rhs_tile_indices));

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
      Value product =
          arith::MulFOp::create(builder, lhs_element, rhs_element);
      Value updated_sum = arith::AddFOp::create(builder, sum, product);
      scf::YieldOp::create(builder, updated_sum);
    }

    SyncThreadsOp::create(builder, TypeRange(tile_types),
                          ValueRange{loop_lhs_tile, loop_rhs_tile});
    scf::YieldOp::create(
        builder,
        ValueRange{k_loop.getResult(0), loop_lhs_tile, loop_rhs_tile});
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
  Value result = F32ToBf16Bits(implicit_then, tile_loop.getResult(0));
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
