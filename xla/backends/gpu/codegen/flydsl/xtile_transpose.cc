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
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using mlir::Value;

struct TransposeDescription {
  const HloInstruction* parameter = nullptr;
  int64_t batch_count = 0;
  int64_t rows = 0;
  int64_t columns = 0;
  int64_t input_row_stride = 0;
  int64_t input_group_size = 1;
  int64_t input_group_stride = 0;
  int64_t input_matrix_stride = 0;
  int64_t input_offset = 0;
  int64_t qkv_selector = -1;
  int64_t qkv_count = 0;
  PrimitiveType element_type = BF16;
};

bool IsSupportedTransposeType(PrimitiveType type) {
  return type == F16 || type == BF16 || type == F32;
}

int64_t TransposeElementBits(PrimitiveType type) {
  return type == F32 ? 32 : 16;
}

bool HasDescendingLayout(const Shape& shape) {
  if (!shape.has_layout() ||
      shape.layout().minor_to_major_size() != shape.dimensions_size()) {
    return false;
  }
  for (int64_t dimension = 0; dimension < shape.dimensions_size();
       ++dimension) {
    if (shape.layout().minor_to_major(dimension) !=
        shape.dimensions_size() - 1 - dimension) {
      return false;
    }
  }
  return true;
}

bool HasDimensions(const HloInstruction& instruction,
                   std::initializer_list<int64_t> dimensions) {
  if (instruction.dimensions().size() != dimensions.size()) {
    return false;
  }
  int64_t index = 0;
  for (int64_t dimension : dimensions) {
    if (instruction.dimensions(index++) != dimension) {
      return false;
    }
  }
  return true;
}

std::optional<TransposeDescription> MatchTransposeRoot(
    const HloInstruction& root) {
  if (root.opcode() != HloOpcode::kTranspose || root.operand_count() != 1 ||
      !IsSupportedTransposeType(root.shape().element_type()) ||
      !HasDescendingLayout(root.shape())) {
    return std::nullopt;
  }

  const HloInstruction* input = root.operand(0);
  // A leading rotation [D0,...,Dn] -> [Dn,D0,...,D(n-1)] is physically one
  // [prod(D0..D(n-1)), Dn] matrix transpose. Flattening the prefix lets the
  // LDS path cover ordinary 2D transposes and Triton's rank-3 rotating form
  // with the same tiled kernel.
  const int64_t rank = root.shape().dimensions_size();
  if (rank >= 2 && input->opcode() == HloOpcode::kParameter &&
      input->shape().element_type() == root.shape().element_type() &&
      input->shape().dimensions_size() == rank &&
      HasDescendingLayout(input->shape()) &&
      root.dimensions().size() == rank && root.dimensions(0) == rank - 1) {
    bool is_leading_rotation = true;
    int64_t rows = 1;
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      const int64_t expected_input_dimension =
          dimension == 0 ? rank - 1 : dimension - 1;
      is_leading_rotation &=
          root.dimensions(dimension) == expected_input_dimension &&
          root.shape().dimensions(dimension) ==
              input->shape().dimensions(expected_input_dimension);
      if (dimension + 1 < rank) {
        rows *= input->shape().dimensions(dimension);
      }
    }
    const int64_t columns = input->shape().dimensions(rank - 1);
    if (is_leading_rotation && rows > 0 && columns > 0) {
      return TransposeDescription{
          /*parameter=*/input,
          /*batch_count=*/1,
          /*rows=*/rows,
          /*columns=*/columns,
          /*input_row_stride=*/columns,
          /*input_group_size=*/1,
          /*input_group_stride=*/rows * columns,
          /*input_matrix_stride=*/0,
          /*input_offset=*/0,
          /*qkv_selector=*/-1,
          /*qkv_count=*/0,
          /*element_type=*/root.shape().element_type()};
    }
  }

  // Attention context changes [B,H,D,S] into [B,S,H,D]. Physically this is a
  // batch of contiguous [H*D,S] matrices transposed to [S,H*D].
  if (root.shape().dimensions_size() == 4 &&
      HasDimensions(root, {0, 3, 1, 2}) &&
      input->opcode() == HloOpcode::kBitcast && input->operand_count() == 1 &&
      input->shape().dimensions_size() == 4 &&
      input->shape().element_type() == BF16 &&
      HasDescendingLayout(input->shape())) {
    const HloInstruction* parameter = input->operand(0);
    const int64_t batch = input->shape().dimensions(0);
    const int64_t heads = input->shape().dimensions(1);
    const int64_t head_dimension = input->shape().dimensions(2);
    const int64_t sequence = input->shape().dimensions(3);
    const int64_t rows = heads * head_dimension;
    const int64_t columns = sequence;
    if (parameter->opcode() != HloOpcode::kParameter ||
        parameter->shape().element_type() != BF16 ||
        !HasDescendingLayout(parameter->shape()) ||
        ShapeUtil::ElementsIn(parameter->shape()) != batch * rows * columns ||
        root.shape().dimensions(0) != batch ||
        root.shape().dimensions(1) != sequence ||
        root.shape().dimensions(2) != heads ||
        root.shape().dimensions(3) != head_dimension) {
      return std::nullopt;
    }
    return TransposeDescription{/*parameter=*/parameter,
                                /*batch_count=*/batch,
                                /*rows=*/rows,
                                /*columns=*/columns,
                                /*input_row_stride=*/columns,
                                /*input_group_size=*/1,
                                /*input_group_stride=*/rows * columns,
                                /*input_matrix_stride=*/0,
                                /*input_offset=*/0,
                                /*qkv_selector=*/-1,
                                /*qkv_count=*/0};
  }

  // Q/K/V extraction changes one [B,S,Q,H,D] slice into [B,1,H,D,S]. For
  // each (batch, head), this is a strided [S,D] matrix transpose. The QKV
  // selector becomes a constant physical base offset instead of a separate
  // materialized slice.
  if (root.shape().dimensions_size() == 5 &&
      HasDimensions(root, {0, 2, 3, 4, 1}) &&
      input->opcode() == HloOpcode::kSlice && input->operand_count() == 1 &&
      input->shape().dimensions_size() == 5 &&
      input->shape().element_type() == BF16 &&
      HasDescendingLayout(input->shape())) {
    const HloInstruction* bitcast = input->operand(0);
    if (bitcast->opcode() != HloOpcode::kBitcast ||
        bitcast->operand_count() != 1 ||
        bitcast->shape().dimensions_size() != 5 ||
        bitcast->shape().element_type() != BF16 ||
        !HasDescendingLayout(bitcast->shape())) {
      return std::nullopt;
    }
    const HloInstruction* parameter = bitcast->operand(0);
    const int64_t batch = bitcast->shape().dimensions(0);
    const int64_t sequence = bitcast->shape().dimensions(1);
    const int64_t qkv = bitcast->shape().dimensions(2);
    const int64_t heads = bitcast->shape().dimensions(3);
    const int64_t head_dimension = bitcast->shape().dimensions(4);
    if (parameter->opcode() != HloOpcode::kParameter ||
        parameter->shape().element_type() != BF16 ||
        !HasDescendingLayout(parameter->shape()) ||
        ShapeUtil::ElementsIn(parameter->shape()) !=
            batch * sequence * qkv * heads * head_dimension ||
        input->slice_starts().size() != 5 ||
        input->slice_limits().size() != 5 ||
        input->slice_strides().size() != 5) {
      return std::nullopt;
    }
    for (int64_t dimension = 0; dimension < 5; ++dimension) {
      const int64_t expected_start =
          dimension == 2 ? input->slice_starts()[dimension] : 0;
      const int64_t expected_limit =
          dimension == 2 ? expected_start + 1
                         : bitcast->shape().dimensions(dimension);
      if (input->slice_starts()[dimension] != expected_start ||
          input->slice_limits()[dimension] != expected_limit ||
          input->slice_strides()[dimension] != 1) {
        return std::nullopt;
      }
    }
    const int64_t selected_qkv = input->slice_starts()[2];
    if (selected_qkv < 0 || selected_qkv >= qkv ||
        input->shape().dimensions(0) != batch ||
        input->shape().dimensions(1) != sequence ||
        input->shape().dimensions(2) != 1 ||
        input->shape().dimensions(3) != heads ||
        input->shape().dimensions(4) != head_dimension ||
        root.shape().dimensions(0) != batch ||
        root.shape().dimensions(1) != 1 ||
        root.shape().dimensions(2) != heads ||
        root.shape().dimensions(3) != head_dimension ||
        root.shape().dimensions(4) != sequence) {
      return std::nullopt;
    }
    const int64_t input_row_stride = qkv * heads * head_dimension;
    return TransposeDescription{
        /*parameter=*/parameter,
        /*batch_count=*/batch * heads,
        /*rows=*/sequence,
        /*columns=*/head_dimension,
        /*input_row_stride=*/input_row_stride,
        /*input_group_size=*/heads,
        /*input_group_stride=*/sequence * input_row_stride,
        /*input_matrix_stride=*/head_dimension,
        /*input_offset=*/selected_qkv * heads * head_dimension,
        /*qkv_selector=*/selected_qkv,
        /*qkv_count=*/qkv};
  }
  return std::nullopt;
}

std::optional<std::vector<TransposeDescription>> MatchTransposes(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() == 0) {
    return std::nullopt;
  }
  std::vector<TransposeDescription> descriptions;
  descriptions.reserve(analysis.fusion_root_count());
  for (int64_t root_index = 0; root_index < analysis.fusion_root_count();
       ++root_index) {
    std::optional<TransposeDescription> description =
        MatchTransposeRoot(analysis.fusion_root(root_index).instruction());
    if (!description.has_value()) {
      return std::nullopt;
    }
    descriptions.push_back(*description);
  }
  if (descriptions.size() == 1) {
    return descriptions;
  }

  // Multi-output native transpose is deliberately limited to two or three
  // outputs from the transformer Q/K/V split. The roots must select disjoint
  // regions of the same fusion parameter and have identical matrix geometry.
  // This lets one workgroup reuse its LDS tile while removing launch latency.
  if (descriptions.size() > 3) {
    return std::nullopt;
  }
  const TransposeDescription& first = descriptions.front();
  bool selectors[3] = {false, false, false};
  for (const TransposeDescription& description : descriptions) {
    if (description.parameter != first.parameter ||
        description.parameter->parameter_number() != 0 ||
        description.batch_count != first.batch_count ||
        description.rows != first.rows ||
        description.columns != first.columns ||
        description.input_row_stride != first.input_row_stride ||
        description.input_group_size != first.input_group_size ||
        description.input_group_stride != first.input_group_stride ||
        description.input_matrix_stride != first.input_matrix_stride ||
        description.element_type != first.element_type ||
        description.qkv_count != 3 || description.qkv_selector < 0 ||
        description.qkv_selector >= 3 || selectors[description.qkv_selector]) {
      return std::nullopt;
    }
    selectors[description.qkv_selector] = true;
  }
  return descriptions;
}

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
    std::optional<std::vector<TransposeDescription>> descriptions =
        MatchTransposes(analysis);
    CHECK(descriptions.has_value());
    descriptions_ = std::move(*descriptions);
    element_type_ = descriptions_.front().element_type;
    element_bits_ = TransposeElementBits(element_type_);
    vector_width_ = 128 / element_bits_;
    elements_per_word_ = 32 / element_bits_;
    rows_ = descriptions_.front().rows;
    columns_ = descriptions_.front().columns;
    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    CHECK_EQ(config.output_tiles_size(), 1);
    CHECK_EQ(config.output_tiles(0).sizes_size(), 2);
    tile_rows_ = config.output_tiles(0).sizes(0);
    tile_columns_ = config.output_tiles(0).sizes(1);
    CHECK_GE(tile_rows_, 32);
    CHECK_LE(tile_rows_, 512);
    CHECK_EQ(tile_rows_ % 32, 0);
    CHECK_GE(tile_columns_, 32);
    CHECK_LE(tile_columns_, 512);
    CHECK_EQ(tile_columns_ % 32, 0);
    threads_ = config.num_warps() * 64;
    CHECK_LE(threads_, 1024);
    CHECK_EQ((tile_rows_ * tile_columns_) % (threads_ * vector_width_), 0);
    vectors_per_thread_ =
        tile_rows_ * tile_columns_ / (threads_ * vector_width_);
    CHECK(vectors_per_thread_ == 2 || vectors_per_thread_ == 4);
    CHECK_EQ(vectors_per_thread_ % elements_per_word_, 0);
    const int64_t row_tile_count =
        (rows_ + tile_rows_ - 1) / tile_rows_;
    const int64_t column_tile_count =
        (columns_ + tile_columns_ - 1) / tile_columns_;
    launch_dimensions_ = LaunchDimensions(
        se::BlockDim(descriptions_.front().batch_count * row_tile_count *
                         column_tile_count,
                     1, 1),
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
    TF_RET_CHECK(entry_function.getNumArguments() == 1 + descriptions_.size());
    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());

    Value input = entry_function.getArgument(0);
    Value thread_id = EmitThreadId(builder, 0);
    Value block_id = EmitBlockId(builder, 0);
    // Keep adjacent workgroups adjacent in the transposed output. This is the
    // workgroup order used by Triton's transpose emitter: the input row tile
    // varies fastest, so neighboring workgroups write neighboring output
    // column tiles instead of regions separated by an entire output row.
    const int64_t row_tile_count =
        (rows_ + tile_rows_ - 1) / tile_rows_;
    const int64_t column_tile_count =
        (columns_ + tile_columns_ - 1) / tile_columns_;
    const int64_t tiles_per_matrix = row_tile_count * column_tile_count;
    Value matrix = Div(builder, block_id, tiles_per_matrix);
    Value matrix_tile = Rem(builder, block_id, tiles_per_matrix);
    Value block_row = Rem(builder, matrix_tile, row_tile_count);
    Value block_column = Div(builder, matrix_tile, row_tile_count);
    Value input_row_base = mlir::arith::MinUIOp::create(
        builder, Mul(builder, block_row, tile_rows_),
        IndexConstant(builder, rows_ - tile_rows_));
    Value input_column_base = mlir::arith::MinUIOp::create(
        builder, Mul(builder, block_column, tile_columns_),
        IndexConstant(builder, columns_ - tile_columns_));
    const int64_t column_vectors = tile_columns_ / vector_width_;
    const int64_t row_group_stride =
        elements_per_word_ == 1 ? threads_ / column_vectors : 1;
    Value local_row_group_base =
        Div(builder, thread_id, column_vectors);
    if (elements_per_word_ != 1) {
      local_row_group_base =
          Mul(builder, local_row_group_base,
              vectors_per_thread_ / elements_per_word_);
    }
    Value vector_group =
        Rem(builder, thread_id, column_vectors);
    Value local_column = Mul(builder, vector_group, vector_width_);
    mlir::Type scalar_type =
        emitters::PrimitiveTypeToMlirType(element_type_, builder);
    auto vector_type =
        mlir::VectorType::get({vector_width_}, scalar_type);
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

    // Store adjacent 16-bit input rows as packed dwords. The physical slot
    // XORs the row pair with a four-bank rotation per input vector. F32 uses
    // rows striped across the participating threads and two padding words per
    // shared row. This is the same conflict-minimizing geometry as Triton's
    // transpose: a Wave64 reaches every LDS bank exactly twice while four
    // adjacent rows remain one 128-bit output read.
    const int64_t row_groups = tile_rows_ / elements_per_word_;
    const int64_t shared_row_stride =
        row_groups + (elements_per_word_ == 1 ? 2 : 0);
    auto shared_type = mlir::RankedTensorType::get(
        {tile_columns_ * shared_row_stride}, builder.getI32Type());
    Value shared = AllocateSharedOp::create(builder, shared_type);
    auto word_elements_type =
        mlir::VectorType::get({elements_per_word_}, scalar_type);
    auto word_vector_type = mlir::VectorType::get({1}, builder.getI32Type());

    Value output_local_row_base =
        Div(builder, thread_id, tile_rows_ / vector_width_);
    Value output_local_column =
        Mul(builder, Rem(builder, thread_id, tile_rows_ / vector_width_),
            vector_width_);
    auto packed_words_type =
        mlir::VectorType::get({vector_width_ / elements_per_word_},
                              builder.getI32Type());
    auto read_output = [&](Value row) {
      Value word_start =
          Div(builder, output_local_column, elements_per_word_);
      const int64_t words_per_vector = vector_width_ / elements_per_word_;
      Value physical_word = word_start;
      if (elements_per_word_ != 1) {
        Value column_group_rotation =
            Mul(builder, Div(builder, row, vector_width_), words_per_vector);
        if (tile_columns_ > tile_rows_) {
          column_group_rotation =
              Rem(builder, column_group_rotation, row_groups);
        }
        physical_word =
            (row_groups & (row_groups - 1)) == 0
                ? Xor(builder, word_start, column_group_rotation)
                : Rem(builder,
                      Add(builder, word_start, column_group_rotation),
                      row_groups);
      }
      Value shared_read_index =
          Add(builder, Mul(builder, row, shared_row_stride), physical_word);
      Value packed_words = mlir::vector::TransferReadOp::create(
          builder, packed_words_type, shared,
          mlir::ValueRange{shared_read_index}, /*padding=*/std::nullopt,
          llvm::ArrayRef<bool>{true});
      return mlir::vector::BitCastOp::create(builder, vector_type,
                                             packed_words);
    };

    Value output_matrix_base = Mul(builder, matrix, rows_ * columns_);
    Value output_tile_base = Add(
        builder, output_matrix_base,
        Add(builder, Mul(builder, input_column_base, rows_), input_row_base));
    auto output_local_index = [&](Value row) {
      return Add(builder, Mul(builder, row, rows_), output_local_column);
    };
    llvm::SmallVector<Value> outputs;
    outputs.reserve(descriptions_.size());
    for (auto indexed_description : llvm::enumerate(descriptions_)) {
      const int64_t output_index = indexed_description.index();
      const TransposeDescription& description = indexed_description.value();
      Value input_group = Div(builder, matrix, description.input_group_size);
      Value input_matrix = Rem(builder, matrix, description.input_group_size);
      Value input_matrix_base = Add(
          builder, Mul(builder, input_group, description.input_group_stride),
          Add(builder,
              Mul(builder, input_matrix, description.input_matrix_stride),
              IndexConstant(builder, description.input_offset)));
      Value input_tile_base =
          Add(builder, input_matrix_base,
              Add(builder,
                  Mul(builder, input_row_base, description.input_row_stride),
                  input_column_base));
      auto input_local_index = [&](Value row) {
        return Add(builder, Mul(builder, row, description.input_row_stride),
                   local_column);
      };
      for (int64_t group_index = 0;
           group_index < vectors_per_thread_ / elements_per_word_;
           ++group_index) {
        Value local_row_group =
            Add(builder, local_row_group_base,
                IndexConstant(builder, group_index * row_group_stride));
        Value first_input_row =
            Mul(builder, local_row_group, elements_per_word_);
        llvm::SmallVector<Value> loaded_rows;
        loaded_rows.reserve(elements_per_word_);
        for (int64_t row = 0; row < elements_per_word_; ++row) {
          Value input_row = first_input_row;
          if (row != 0) {
            input_row = Add(builder, first_input_row,
                            IndexConstant(builder, row));
          }
          loaded_rows.push_back(tile_buffer_load(
              input, input_local_index(input_row), input_tile_base));
        }

        for (int64_t element = 0; element < vector_width_; ++element) {
          llvm::SmallVector<Value> word_elements;
          word_elements.reserve(elements_per_word_);
          for (Value loaded : loaded_rows) {
            word_elements.push_back(mlir::vector::ExtractOp::create(
                builder, loaded, llvm::ArrayRef<int64_t>{element}));
          }
          Value packed_elements = mlir::vector::FromElementsOp::create(
              builder, word_elements_type, word_elements);
          Value word_vector =
              mlir::vector::BitCastOp::create(builder, word_vector_type,
                                              packed_elements);
          Value word = mlir::vector::ExtractOp::create(
              builder, word_vector, llvm::ArrayRef<int64_t>{0});
          Value column =
              Add(builder, local_column, IndexConstant(builder, element));
          Value physical_word = local_row_group;
          if (elements_per_word_ != 1) {
            const int64_t words_per_vector =
                vector_width_ / elements_per_word_;
            Value column_group_rotation = Mul(
                builder, Div(builder, column, vector_width_),
                words_per_vector);
            if (tile_columns_ > tile_rows_) {
              column_group_rotation =
                  Rem(builder, column_group_rotation, row_groups);
            }
            physical_word =
                (row_groups & (row_groups - 1)) == 0
                    ? Xor(builder, local_row_group, column_group_rotation)
                    : Rem(builder,
                          Add(builder, local_row_group, column_group_rotation),
                          row_groups);
          }
          Value shared_write_index =
              Add(builder, Mul(builder, column, shared_row_stride),
                  physical_word);
          shared = mlir::tensor::InsertOp::create(
              builder, word, shared, mlir::ValueRange{shared_write_index});
        }
      }
      shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{shared_type}, shared)
              .getResult(0);

      Value output = entry_function.getArgument(1 + output_index);
      for (int64_t vector_index = 0;
           vector_index < vectors_per_thread_; ++vector_index) {
        Value output_local_row = Add(
            builder, output_local_row_base,
            IndexConstant(builder,
                          vector_index * tile_columns_ /
                              vectors_per_thread_));
        Value transposed = read_output(output_local_row);
        Value vector_offset = output_local_index(output_local_row);
        output = tile_buffer_store(transposed, output, vector_offset,
                                   output_tile_base);
      }
      outputs.push_back(output);

      // The next Q/K/V phase reuses the same LDS allocation. Synchronize after
      // all threads have consumed this tile before any thread overwrites it.
      if (output_index + 1 != descriptions_.size()) {
        shared =
            SyncThreadsOp::create(builder, mlir::TypeRange{shared_type}, shared)
                .getResult(0);
      }
    }
    mlir::func::ReturnOp::create(builder, outputs);
    return absl::OkStatus();
  }

  PrimitiveType element_type_ = PRIMITIVE_TYPE_INVALID;
  int64_t element_bits_ = 0;
  int64_t vector_width_ = 0;
  int64_t elements_per_word_ = 0;
  int64_t rows_ = 0;
  int64_t columns_ = 0;
  int64_t tile_rows_ = 0;
  int64_t tile_columns_ = 0;
  int64_t threads_ = 0;
  int64_t vectors_per_thread_ = 0;
  std::vector<TransposeDescription> descriptions_;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

std::optional<std::pair<int64_t, int64_t>> GetFlyXTileTransposeMatrixShape(
    const HloFusionAnalysis& analysis) {
  std::optional<std::vector<TransposeDescription>> descriptions =
      MatchTransposes(analysis);
  if (!descriptions.has_value() || descriptions->empty() ||
      descriptions->front().rows < 32 ||
      descriptions->front().columns < 32) {
    return std::nullopt;
  }
  return std::pair<int64_t, int64_t>{descriptions->front().rows,
                                     descriptions->front().columns};
}

bool IsFlyXTileTransposeFusion(const HloFusionAnalysis& analysis) {
  return GetFlyXTileTransposeMatrixShape(analysis).has_value();
}

bool IsFlyXTileTransposeConfigSupported(const HloFusionAnalysis& analysis) {
  std::optional<std::pair<int64_t, int64_t>> matrix_shape =
      GetFlyXTileTransposeMatrixShape(analysis);
  if (!matrix_shape.has_value()) {
    return false;
  }
  const BlockLevelFusionConfig& config =
      analysis.fusion_backend_config().block_level_fusion_config();
  if (config.output_tiles_size() != 1 ||
      config.output_tiles(0).sizes_size() != 2) {
    return false;
  }
  const int64_t tile_rows = config.output_tiles(0).sizes(0);
  const int64_t tile_columns = config.output_tiles(0).sizes(1);
  const auto is_supported_tile_extent = [](int64_t extent) {
    return extent >= 32 && extent <= 512 && extent % 32 == 0;
  };
  auto [rows, columns] = *matrix_shape;
  const int64_t element_bits =
      TransposeElementBits(analysis.first_result_shape().element_type());
  const int64_t vector_width = 128 / element_bits;
  const int64_t maximum_tile_elements = 65536 * 8 / element_bits;
  const int64_t shared_tile_elements =
      (tile_rows + (element_bits == 32 ? 2 : 0)) * tile_columns;
  const int64_t threads = config.num_warps() * 64;
  const int64_t tile_elements = tile_rows * tile_columns;
  if (!is_supported_tile_extent(tile_rows) ||
      !is_supported_tile_extent(tile_columns) || tile_rows > rows ||
      tile_columns > columns ||
      shared_tile_elements > maximum_tile_elements ||
      threads <= 0 || threads > 1024 ||
      tile_elements % (threads * vector_width) != 0) {
    return false;
  }
  const int64_t vectors_per_thread =
      tile_elements / (threads * vector_width);
  return vectors_per_thread == 2 || vectors_per_thread == 4;
}

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileTransposeEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileTransposeEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
