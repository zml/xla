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

#include "xla/backends/gpu/codegen/flydsl/xtile_transpose.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/flydsl/compiler.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using mlir::Value;

constexpr int64_t kVectorWidth = 8;

Value IndexConstant(mlir::ImplicitLocOpBuilder& builder, int64_t value) {
  return mlir::arith::ConstantIndexOp::create(builder, value);
}

Value Add(mlir::ImplicitLocOpBuilder& builder, Value lhs, Value rhs) {
  return mlir::arith::AddIOp::create(builder, lhs, rhs);
}

Value Mul(mlir::ImplicitLocOpBuilder& builder, Value lhs, int64_t rhs) {
  return mlir::arith::MulIOp::create(builder, lhs, IndexConstant(builder, rhs));
}

Value Div(mlir::ImplicitLocOpBuilder& builder, Value lhs, int64_t rhs) {
  return mlir::arith::DivUIOp::create(builder, lhs,
                                      IndexConstant(builder, rhs));
}

Value Rem(mlir::ImplicitLocOpBuilder& builder, Value lhs, int64_t rhs) {
  return mlir::arith::RemUIOp::create(builder, lhs,
                                      IndexConstant(builder, rhs));
}

Value Xor(mlir::ImplicitLocOpBuilder& builder, Value lhs, Value rhs) {
  return mlir::arith::XOrIOp::create(builder, lhs, rhs);
}

class FlyXTileTransposeEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileTransposeEmitter(const HloFusionAnalysis& analysis) {
    const HloInstruction& root = analysis.fusion_root(0).instruction();
    rows_ = root.operand(0)->shape().dimensions(0);
    columns_ = root.operand(0)->shape().dimensions(1);
    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    CHECK_EQ(config.output_tiles_size(), 1);
    CHECK_EQ(config.output_tiles(0).sizes_size(), 2);
    CHECK_EQ(config.output_tiles(0).sizes(0),
             config.output_tiles(0).sizes(1));
    tile_size_ = config.output_tiles(0).sizes(0);
    CHECK(tile_size_ == 32 || tile_size_ == 64 || tile_size_ == 128);
    threads_ = tile_size_ * tile_size_ / (2 * kVectorWidth);
    CHECK_EQ(config.num_warps() * 64, threads_);
    launch_dimensions_ = LaunchDimensions(
        se::BlockDim((rows_ / tile_size_) * (columns_ / tile_size_), 1, 1),
        se::ThreadDim(threads_, 1, 1));
  }

  LaunchDimensions launch_dimensions() const override {
    return launch_dimensions_;
  }

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t, mlir::MLIRContext*) const override {
    return std::nullopt;
  }

  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t, mlir::MLIRContext*) const override {
    return std::nullopt;
  }

 private:
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateMLIRModule(
      mlir::MLIRContext& context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const override {
    mlir::OpBuilder module_builder(&context);
    mlir::Location location =
        mlir::NameLoc::get(module_builder.getStringAttr(fusion.name()));
    mlir::OwningOpRef<mlir::ModuleOp> module =
        llvm_ir::CreateMlirModuleOp(location);
    ASSIGN_OR_RETURN(mlir::func::FuncOp entry_function,
                     emitters::EmitKernelApi(*module, fusion, buffer_assignment,
                                             GetDefaultBufferAlignment(),
                                             entry_function_name));
    SetBackendKind(&context, entry_function, BackendKind::kGpu);
    emitters::SetIndexDataLayout(*module, fusion);
    RETURN_IF_ERROR(EmitKernel(entry_function));
    MarkGenericFusion(*module);
    return module;
  }

  absl::Status EmitEntryFunction(const emitters::PartitionedComputations&,
                                 const emitters::CallTargetProvider&,
                                 mlir::func::FuncOp,
                                 const HloFusionInstruction&) const override {
    return absl::UnimplementedError(
        "FlyXTileTransposeEmitter builds its module directly.");
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function) const {
    TF_RET_CHECK(entry_function.getNumArguments() == 2);
    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());

    Value input = entry_function.getArgument(0);
    Value output = entry_function.getArgument(1);
    Value thread_id = EmitThreadId(builder, 0);
    Value block_id = EmitBlockId(builder, 0);
    // Keep adjacent workgroups adjacent in the transposed output. This is the
    // workgroup order used by Triton's transpose emitter: the input row tile
    // varies fastest, so neighboring workgroups write neighboring output
    // column tiles instead of regions separated by an entire output row.
    Value tile_rows = IndexConstant(builder, rows_ / tile_size_);
    Value block_row =
        mlir::arith::RemUIOp::create(builder, block_id, tile_rows);
    Value block_column = Div(builder, block_id, rows_ / tile_size_);
    Value input_row_base = Mul(builder, block_row, tile_size_);
    Value input_column_base = Mul(builder, block_column, tile_size_);

    Value local_row_pair = Div(builder, thread_id, tile_size_ / kVectorWidth);
    Value vector_group = Rem(builder, thread_id, tile_size_ / kVectorWidth);
    Value local_column = Mul(builder, vector_group, kVectorWidth);
    Value first_input_row = Mul(builder, local_row_pair, 2);
    Value second_input_row =
        Add(builder, first_input_row, IndexConstant(builder, 1));
    auto vector_type =
        mlir::VectorType::get({kVectorWidth}, builder.getBF16Type());
    auto tile_buffer_load = [&](Value source, Value vector_index,
                                Value tile_index) {
      mlir::OperationState state(entry_function.getLoc(),
                                 "xla_gpu.tile_buffer_load");
      state.addOperands({source, vector_index, tile_index});
      state.addTypes(vector_type);
      return builder.create(state)->getResult(0);
    };
    auto tile_buffer_store = [&](Value value, Value destination,
                                 Value vector_index, Value tile_index) {
      mlir::OperationState state(entry_function.getLoc(),
                                 "xla_gpu.tile_buffer_store");
      state.addOperands({value, destination, vector_index, tile_index});
      state.addTypes(destination.getType());
      state.addAttribute("cache_policy", builder.getI32IntegerAttr(2));
      return builder.create(state)->getResult(0);
    };
    Value input_tile_base =
        Add(builder, Mul(builder, input_row_base, columns_), input_column_base);
    auto input_local_index = [&](Value row) {
      return Add(builder, Mul(builder, row, columns_), local_column);
    };
    Value first_loaded = tile_buffer_load(
        input, input_local_index(first_input_row), input_tile_base);
    Value second_loaded = tile_buffer_load(
        input, input_local_index(second_input_row), input_tile_base);

    // Store adjacent input rows as packed dwords. The physical slot XORs the
    // row pair with a four-bank rotation per eight input columns. Stores then
    // hit every LDS bank exactly twice per wave (the wave64 minimum), while
    // four adjacent row pairs remain a contiguous 128-bit output read.
    const int64_t row_pairs = tile_size_ / 2;
    auto shared_type = mlir::RankedTensorType::get({tile_size_ * row_pairs},
                                                   builder.getI32Type());
    Value shared = AllocateSharedOp::create(builder, shared_type);
    auto pair_type = mlir::VectorType::get({2}, builder.getBF16Type());
    auto word_vector_type = mlir::VectorType::get({1}, builder.getI32Type());
    for (int64_t element = 0; element < kVectorWidth; ++element) {
      Value first_scalar = mlir::vector::ExtractOp::create(
          builder, first_loaded, llvm::ArrayRef<int64_t>{element});
      Value second_scalar = mlir::vector::ExtractOp::create(
          builder, second_loaded, llvm::ArrayRef<int64_t>{element});
      Value pair = mlir::vector::FromElementsOp::create(
          builder, pair_type, mlir::ValueRange{first_scalar, second_scalar});
      Value word_vector =
          mlir::vector::BitCastOp::create(builder, word_vector_type, pair);
      Value word = mlir::vector::ExtractOp::create(builder, word_vector,
                                                   llvm::ArrayRef<int64_t>{0});
      Value column =
          Add(builder, local_column, IndexConstant(builder, element));
      Value column_group_rotation = Mul(builder, Div(builder, column, 8), 4);
      Value physical_pair = Xor(builder, local_row_pair, column_group_rotation);
      Value shared_write_index =
          Add(builder, Mul(builder, column, row_pairs), physical_pair);
      shared = mlir::tensor::InsertOp::create(
          builder, word, shared, mlir::ValueRange{shared_write_index});
    }
    shared =
        SyncThreadsOp::create(builder, mlir::TypeRange{shared_type}, shared)
            .getResult(0);

    Value output_local_row = Div(builder, thread_id, tile_size_ / kVectorWidth);
    Value second_output_local_row =
        Add(builder, output_local_row, IndexConstant(builder, tile_size_ / 2));
    Value output_local_column =
        Mul(builder, Rem(builder, thread_id, tile_size_ / kVectorWidth),
            kVectorWidth);
    auto packed_words_type =
        mlir::VectorType::get({kVectorWidth / 2}, builder.getI32Type());
    auto read_output = [&](Value row) {
      Value pair_start = Div(builder, output_local_column, 2);
      Value column_group_rotation = Mul(builder, Div(builder, row, 8), 4);
      Value physical_pair = Xor(builder, pair_start, column_group_rotation);
      Value shared_read_index =
          Add(builder, Mul(builder, row, row_pairs), physical_pair);
      Value packed_words = mlir::vector::TransferReadOp::create(
          builder, packed_words_type, shared,
          mlir::ValueRange{shared_read_index}, /*padding=*/std::nullopt,
          llvm::ArrayRef<bool>{true});
      return mlir::vector::BitCastOp::create(builder, vector_type,
                                             packed_words);
    };
    Value first_transposed = read_output(output_local_row);
    Value second_transposed = read_output(second_output_local_row);

    Value output_tile_base =
        Add(builder, Mul(builder, input_column_base, rows_), input_row_base);
    auto output_local_index = [&](Value row) {
      return Add(builder, Mul(builder, row, rows_), output_local_column);
    };
    output = tile_buffer_store(first_transposed, output,
                               output_local_index(output_local_row),
                               output_tile_base);
    output = tile_buffer_store(second_transposed, output,
                               output_local_index(second_output_local_row),
                               output_tile_base);
    mlir::func::ReturnOp::create(builder, output);
    return absl::OkStatus();
  }

  int64_t rows_ = 0;
  int64_t columns_ = 0;
  int64_t tile_size_ = 0;
  int64_t threads_ = 0;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

bool IsFlyXTileTransposeFusion(const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return false;
  }
  const HloInstruction& root = analysis.fusion_root(0).instruction();
  return root.opcode() == HloOpcode::kTranspose && root.operand_count() == 1 &&
         root.operand(0)->opcode() == HloOpcode::kParameter &&
         root.shape().element_type() == BF16 &&
         root.shape().dimensions_size() == 2 && root.dimensions().size() == 2 &&
         root.dimensions(0) == 1 && root.dimensions(1) == 0 &&
         root.operand(0)->shape().dimensions(0) % 32 == 0 &&
         root.operand(0)->shape().dimensions(1) % 32 == 0;
}

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileTransposeEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileTransposeEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
