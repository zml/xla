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

#include "xla/backends/gpu/codegen/flydsl/xtile_softmax.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/flydsl/compiler.h"
#include "xla/backends/gpu/codegen/flydsl/layer_norm_support.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using mlir::Value;

Value IndexConstant(mlir::ImplicitLocOpBuilder& builder, int64_t value) {
  return mlir::arith::ConstantIndexOp::create(builder, value);
}

Value Add(mlir::ImplicitLocOpBuilder& builder, Value lhs, Value rhs) {
  return mlir::arith::AddIOp::create(builder, lhs, rhs);
}

Value Mul(mlir::ImplicitLocOpBuilder& builder, Value lhs, Value rhs) {
  return mlir::arith::MulIOp::create(builder, lhs, rhs);
}

Value Div(mlir::ImplicitLocOpBuilder& builder, Value lhs, int64_t rhs) {
  return mlir::arith::DivUIOp::create(builder, lhs,
                                      IndexConstant(builder, rhs));
}

Value Rem(mlir::ImplicitLocOpBuilder& builder, Value lhs, int64_t rhs) {
  return mlir::arith::RemUIOp::create(builder, lhs,
                                      IndexConstant(builder, rhs));
}

mlir::func::FuncOp CreateReducer(mlir::ModuleOp module, mlir::StringRef name,
                                 bool is_maximum) {
  mlir::ImplicitLocOpBuilder builder(module.getLoc(), module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto type = builder.getFunctionType(
      {builder.getF32Type(), builder.getF32Type()}, {builder.getF32Type()});
  auto function = mlir::func::FuncOp::create(module.getLoc(), name, type);
  module.push_back(function);
  mlir::Block* block = function.addEntryBlock();
  builder.setInsertionPointToStart(block);
  Value reduced =
      is_maximum ? mlir::arith::MaxNumFOp::create(
                       builder, block->getArgument(0), block->getArgument(1))
                       .getResult()
                 : mlir::arith::AddFOp::create(builder, block->getArgument(0),
                                               block->getArgument(1))
                       .getResult();
  mlir::func::ReturnOp::create(builder, reduced);
  return function;
}

class FlyXTileSoftmaxEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileSoftmaxEmitter(const HloFusionAnalysis& analysis) {
    const HloInstruction& root = analysis.fusion_root(0).instruction();
    std::optional<FlySoftmaxBackwardDescriptor> softmax_backward =
        GetFlySoftmaxBackwardDescriptor(analysis);
    std::optional<FlyLayerNormDescriptor> layer_norm =
        GetFlyLayerNormDescriptor(analysis);
    is_softmax_backward_ = softmax_backward.has_value();
    softmax_backward_scale_ =
        softmax_backward.has_value() ? softmax_backward->scale : 1.0;
    softmax_input_scale_ =
        softmax_backward.has_value() || layer_norm.has_value()
            ? 1.0
            : GetFlySoftmaxInputScale(root);
    is_layer_norm_ = layer_norm.has_value();
    epsilon_ = layer_norm.has_value() ? layer_norm->epsilon : 0.0;
    output_count_ = layer_norm.has_value() ? layer_norm->output_count : 1;
    uses_moments_variance_ =
        layer_norm.has_value() && layer_norm->uses_moments_variance;
    layer_norm_parameter_count_ =
        layer_norm.has_value() ? layer_norm->input->parent()->num_parameters()
                               : 0;
    gamma_parameter_number_ =
        layer_norm.has_value() && layer_norm->gamma != nullptr
            ? layer_norm->gamma->parameter_number()
            : -1;
    beta_parameter_number_ =
        layer_norm.has_value() && layer_norm->beta != nullptr
            ? layer_norm->beta->parameter_number()
            : -1;
    const HloInstruction* input =
        layer_norm.has_value()
            ? layer_norm->input
            : (softmax_backward.has_value() ? softmax_backward->scores
                                            : GetFlySoftmaxInput(root));
    CHECK(input != nullptr);
    while (input->opcode() == HloOpcode::kBitcast &&
           input->operand_count() == 1) {
      input = input->operand(0);
    }
    CHECK_EQ(input->opcode(), HloOpcode::kParameter);
    input_parameter_number_ = input->parameter_number();
    if (softmax_backward.has_value()) {
      const HloInstruction* upstream = softmax_backward->upstream_gradient;
      while (upstream->opcode() == HloOpcode::kBitcast &&
             upstream->operand_count() == 1) {
        upstream = upstream->operand(0);
      }
      CHECK_EQ(upstream->opcode(), HloOpcode::kParameter);
      upstream_gradient_parameter_number_ = upstream->parameter_number();
    } else if (!is_layer_norm_) {
      const HloInstruction* row_offset = GetFlySoftmaxExternalRowOffset(root);
      if (row_offset != nullptr) {
        external_row_parameter_number_ = row_offset->parameter_number();
        recompute_maximum_after_external_row_offset_ =
            FlySoftmaxRecomputesMaximumAfterExternalRowOffset(root);
      }
    }
    const Shape& shape = analysis.first_result_shape();
    dimensions_.assign(shape.dimensions().begin(), shape.dimensions().end());
    columns_ = dimensions_.back();
    rows_ = ShapeUtil::ElementsIn(shape) / columns_;
    element_type_ = shape.element_type();
    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    num_warps_ = config.num_warps();
    independent_rows_ = columns_ <= 256;
    const int64_t blocks =
        independent_rows_ ? (rows_ + num_warps_ - 1) / num_warps_ : rows_;
    launch_dimensions_ = LaunchDimensions(se::BlockDim(blocks, 1, 1),
                                          se::ThreadDim(num_warps_ * 64, 1, 1));
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
    context.getOrLoadDialect<mlir::amdgpu::AMDGPUDialect>();
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
    context.getOrLoadDialect<mlir::ub::UBDialect>();
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
    mlir::func::FuncOp maximum =
        CreateReducer(*module, "fly_softmax_maximum", /*is_maximum=*/true);
    mlir::func::FuncOp add =
        CreateReducer(*module, "fly_softmax_add", /*is_maximum=*/false);
    RETURN_IF_ERROR(EmitKernel(entry_function, maximum, add));
    MarkGenericFusion(*module);
    return module;
  }

  absl::Status EmitEntryFunction(const emitters::PartitionedComputations&,
                                 const emitters::CallTargetProvider&,
                                 mlir::func::FuncOp,
                                 const HloFusionInstruction&) const override {
    return absl::UnimplementedError(
        "FlyXTileSoftmaxEmitter builds its module directly.");
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function,
                          mlir::func::FuncOp maximum,
                          mlir::func::FuncOp add_reducer) const {
    const bool has_external_row = external_row_parameter_number_ >= 0;
    const int64_t parameter_count = is_layer_norm_ ? layer_norm_parameter_count_
                                    : is_softmax_backward_
                                        ? 2
                                        : (has_external_row ? 2 : 1);
    const int64_t expected_arguments = parameter_count + output_count_;
    TF_RET_CHECK(entry_function.getNumArguments() == expected_arguments);
    TF_RET_CHECK(input_parameter_number_ >= 0 &&
                 input_parameter_number_ < parameter_count);
    TF_RET_CHECK(gamma_parameter_number_ < parameter_count);
    TF_RET_CHECK(beta_parameter_number_ < parameter_count);
    TF_RET_CHECK(!has_external_row ||
                 external_row_parameter_number_ < parameter_count);
    TF_RET_CHECK(
        !is_softmax_backward_ ||
        (upstream_gradient_parameter_number_ >= 0 &&
         upstream_gradient_parameter_number_ < parameter_count &&
         upstream_gradient_parameter_number_ != input_parameter_number_));
    const int64_t threads = independent_rows_ ? 64 : num_warps_ * 64;
    TF_RET_CHECK(num_warps_ > 0 && num_warps_ <= 16 && columns_ > 0 &&
                 (columns_ + threads - 1) / threads <= 64);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());
    Value input = entry_function.getArgument(input_parameter_number_);
    Value upstream_gradient =
        is_softmax_backward_
            ? entry_function.getArgument(upstream_gradient_parameter_number_)
            : Value();
    Value gamma = gamma_parameter_number_ >= 0
                      ? entry_function.getArgument(gamma_parameter_number_)
                      : Value();
    Value beta = beta_parameter_number_ >= 0
                     ? entry_function.getArgument(beta_parameter_number_)
                     : Value();
    Value external_row =
        has_external_row
            ? entry_function.getArgument(external_row_parameter_number_)
            : Value();
    Value output = entry_function.getArgument(parameter_count);
    Value mean_output = output_count_ > 1
                            ? entry_function.getArgument(parameter_count + 1)
                            : Value();
    Value reciprocal_stddev_output =
        output_count_ > 1 ? entry_function.getArgument(parameter_count + 2)
                          : Value();
    Value reciprocal_stddev_cube_output =
        output_count_ > 3 ? entry_function.getArgument(parameter_count + 3)
                          : Value();
    Value thread_id = EmitThreadId(builder, 0);
    Value block_id = EmitBlockId(builder, 0);
    Value wave_id = Div(builder, thread_id, 64);
    Value lane_id = Rem(builder, thread_id, 64);
    Value row = block_id;
    Value element_thread_id = thread_id;
    if (independent_rows_) {
      row = Add(builder,
                Mul(builder, block_id, IndexConstant(builder, num_warps_)),
                wave_id);
      element_thread_id = lane_id;
    }
    Value row_in_bounds =
        mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::ult,
                                    row, IndexConstant(builder, rows_));

    auto make_row_indices = [&](const std::vector<int64_t>& dimensions) {
      std::vector<Value> indices(dimensions.size() - 1);
      Value remaining_row = row;
      for (int64_t dimension = static_cast<int64_t>(dimensions.size()) - 2;
           dimension > 0; --dimension) {
        indices[dimension] = Rem(builder, remaining_row, dimensions[dimension]);
        remaining_row = Div(builder, remaining_row, dimensions[dimension]);
      }
      indices[0] = remaining_row;
      return indices;
    };
    std::vector<Value> output_row_indices = make_row_indices(dimensions_);
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    std::vector<int64_t> input_dimensions(input_type.getShape().begin(),
                                          input_type.getShape().end());
    TF_RET_CHECK(input_dimensions.size() >= 2 &&
                 input_dimensions.back() == columns_);
    auto has_minor_vector_shape = [&](Value value) {
      if (!value) return true;
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
      return type && type.getRank() == 1 && type.getDimSize(0) == columns_;
    };
    TF_RET_CHECK(has_minor_vector_shape(gamma) && has_minor_vector_shape(beta));
    std::vector<Value> input_row_indices = make_row_indices(input_dimensions);
    if (upstream_gradient) {
      auto upstream_type =
          mlir::dyn_cast<mlir::RankedTensorType>(upstream_gradient.getType());
      TF_RET_CHECK(upstream_type &&
                   upstream_type.getShape() ==
                       llvm::ArrayRef<int64_t>(input_dimensions));
    }
    auto has_row_output_shape = [&](Value value) {
      if (!value) return true;
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
      if (!type || !type.getElementType().isF32() ||
          type.getRank() != dimensions_.size() - 1) {
        return false;
      }
      for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
        if (type.getDimSize(dimension) != dimensions_[dimension]) return false;
      }
      return true;
    };
    TF_RET_CHECK(has_row_output_shape(mean_output) &&
                 has_row_output_shape(reciprocal_stddev_output) &&
                 has_row_output_shape(reciprocal_stddev_cube_output));

    Value zero = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(), llvm::APFloat(0.0f));
    Value log2e = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(), llvm::APFloat(1.4426950408889634f));
    Value minus_inf = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(),
        llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(),
                              /*negative=*/true));
    const bool all_rows_in_bounds =
        !independent_rows_ || rows_ % num_warps_ == 0;

    auto store_row_output = [&](Value row_output, Value value) -> Value {
      Value writer_id = independent_rows_ ? lane_id : thread_id;
      Value condition =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                      writer_id, IndexConstant(builder, 0));
      if (!all_rows_in_bounds) {
        condition =
            mlir::arith::AndIOp::create(builder, condition, row_in_bounds);
      }
      mlir::scf::IfOp store = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{row_output.getType()}, condition,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(store.thenBlock());
        Value updated = mlir::tensor::InsertOp::create(
            builder, value, row_output, mlir::ValueRange(output_row_indices));
        mlir::scf::YieldOp::create(builder, updated);
        builder.setInsertionPointToStart(store.elseBlock());
        mlir::scf::YieldOp::create(builder, row_output);
      }
      builder.setInsertionPointAfter(store);
      return store.getResult(0);
    };

    Value row_offset;
    if (has_external_row) {
      if (all_rows_in_bounds) {
        row_offset = mlir::tensor::ExtractOp::create(
            builder, external_row, mlir::ValueRange(input_row_indices));
      } else {
        mlir::scf::IfOp load = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{builder.getF32Type()}, row_in_bounds,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(load.thenBlock());
          Value value = mlir::tensor::ExtractOp::create(
              builder, external_row, mlir::ValueRange(input_row_indices));
          mlir::scf::YieldOp::create(builder, value);
          builder.setInsertionPointToStart(load.elseBlock());
          mlir::scf::YieldOp::create(builder, zero);
        }
        builder.setInsertionPointAfter(load);
        row_offset = load.getResult(0);
      }
    }

    auto convert_input = [&](Value value) -> Value {
      if (!value.getType().isF32()) {
        value =
            mlir::arith::ExtFOp::create(builder, builder.getF32Type(), value);
      }
      return value;
    };

    auto load_minor_vector = [&](Value vector, Value column,
                                 bool all_columns_in_bounds,
                                 Value fallback) -> Value {
      if (all_columns_in_bounds) {
        return convert_input(mlir::tensor::ExtractOp::create(
            builder, vector, mlir::ValueRange{column}));
      }
      Value in_bounds =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::ult,
                                      column, IndexConstant(builder, columns_));
      mlir::scf::IfOp load = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{builder.getF32Type()}, in_bounds,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(load.thenBlock());
        Value value = mlir::tensor::ExtractOp::create(builder, vector,
                                                      mlir::ValueRange{column});
        mlir::scf::YieldOp::create(builder, convert_input(value));
        builder.setInsertionPointToStart(load.elseBlock());
        mlir::scf::YieldOp::create(builder, fallback);
      }
      builder.setInsertionPointAfter(load);
      return load.getResult(0);
    };

    auto load_input = [&](Value column, bool all_columns_in_bounds) -> Value {
      const bool all_threads_in_bounds =
          all_rows_in_bounds && all_columns_in_bounds;
      if (all_threads_in_bounds) {
        std::vector<Value> indices = input_row_indices;
        indices.push_back(column);
        return convert_input(mlir::tensor::ExtractOp::create(
            builder, input, mlir::ValueRange(indices)));
      }
      Value in_bounds =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::ult,
                                      column, IndexConstant(builder, columns_));
      if (!all_rows_in_bounds) {
        in_bounds =
            mlir::arith::AndIOp::create(builder, in_bounds, row_in_bounds);
      }
      mlir::scf::IfOp load = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{builder.getF32Type()}, in_bounds,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(load.thenBlock());
        std::vector<Value> indices = input_row_indices;
        indices.push_back(column);
        Value value = mlir::tensor::ExtractOp::create(
            builder, input, mlir::ValueRange(indices));
        mlir::scf::YieldOp::create(builder, convert_input(value));
        builder.setInsertionPointToStart(load.elseBlock());
        mlir::scf::YieldOp::create(builder, is_layer_norm_ ? zero : minus_inf);
      }
      builder.setInsertionPointAfter(load);
      return load.getResult(0);
    };

    auto load_upstream_gradient = [&](Value column,
                                      bool all_columns_in_bounds) -> Value {
      CHECK(upstream_gradient);
      const bool all_threads_in_bounds =
          all_rows_in_bounds && all_columns_in_bounds;
      if (all_threads_in_bounds) {
        std::vector<Value> indices = input_row_indices;
        indices.push_back(column);
        return convert_input(mlir::tensor::ExtractOp::create(
            builder, upstream_gradient, mlir::ValueRange(indices)));
      }
      Value in_bounds =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::ult,
                                      column, IndexConstant(builder, columns_));
      if (!all_rows_in_bounds) {
        in_bounds =
            mlir::arith::AndIOp::create(builder, in_bounds, row_in_bounds);
      }
      mlir::scf::IfOp load = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{builder.getF32Type()}, in_bounds,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(load.thenBlock());
        std::vector<Value> indices = input_row_indices;
        indices.push_back(column);
        Value value = mlir::tensor::ExtractOp::create(
            builder, upstream_gradient, mlir::ValueRange(indices));
        mlir::scf::YieldOp::create(builder, convert_input(value));
        builder.setInsertionPointToStart(load.elseBlock());
        mlir::scf::YieldOp::create(builder, zero);
      }
      builder.setInsertionPointAfter(load);
      return load.getResult(0);
    };

    std::vector<Value> values;
    std::vector<Value> upstream_gradient_values;
    const int64_t values_per_thread = (columns_ + threads - 1) / threads;
    values.reserve(values_per_thread);
    if (is_softmax_backward_) {
      upstream_gradient_values.reserve(values_per_thread);
    }
    Value backward_scale = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(),
        llvm::APFloat(static_cast<float>(softmax_backward_scale_)));
    Value softmax_input_scale = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(),
        llvm::APFloat(static_cast<float>(softmax_input_scale_)));
    Value local_max = minus_inf;
    Value local_sum = zero;
    Value local_input_squared_sum = zero;
    for (int64_t i = 0; i < values_per_thread; ++i) {
      Value column =
          Add(builder, element_thread_id, IndexConstant(builder, i * threads));
      Value value = load_input(column, (i + 1) * threads <= columns_);
      if (is_softmax_backward_) {
        upstream_gradient_values.push_back(
            load_upstream_gradient(column, (i + 1) * threads <= columns_));
        value = mlir::arith::MulFOp::create(builder, value, backward_scale);
      } else if (!is_layer_norm_ && softmax_input_scale_ != 1.0) {
        value =
            mlir::arith::MulFOp::create(builder, value, softmax_input_scale);
      }
      if (recompute_maximum_after_external_row_offset_) {
        value = mlir::arith::SubFOp::create(builder, value, row_offset);
      }
      values.push_back(value);
      local_max = mlir::arith::MaxNumFOp::create(builder, local_max, value);
      local_sum = mlir::arith::AddFOp::create(builder, local_sum, value);
      if (uses_moments_variance_) {
        Value square = mlir::arith::MulFOp::create(builder, value, value);
        local_input_squared_sum = mlir::arith::AddFOp::create(
            builder, local_input_squared_sum, square);
      }
    }

    auto shared_type =
        mlir::RankedTensorType::get({num_warps_}, builder.getF32Type());
    Value shared;
    if (!independent_rows_ && num_warps_ > 1) {
      shared = AllocateSharedOp::create(builder, shared_type);
    }

    auto conditional_store = [&](Value shared_value, Value condition,
                                 Value stored_value, Value index) -> Value {
      mlir::scf::IfOp store = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{shared_type}, condition,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(store.thenBlock());
        Value updated = mlir::tensor::InsertOp::create(
            builder, stored_value, shared_value, mlir::ValueRange{index});
        mlir::scf::YieldOp::create(builder, updated);
        builder.setInsertionPointToStart(store.elseBlock());
        mlir::scf::YieldOp::create(builder, shared_value);
      }
      builder.setInsertionPointAfter(store);
      return store.getResult(0);
    };

    auto conditional_load = [&](Value shared_value, Value condition,
                                Value index, Value fallback) -> Value {
      mlir::scf::IfOp load = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{builder.getF32Type()}, condition,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(load.thenBlock());
        Value loaded = mlir::tensor::ExtractOp::create(builder, shared_value,
                                                       mlir::ValueRange{index});
        mlir::scf::YieldOp::create(builder, loaded);
        builder.setInsertionPointToStart(load.elseBlock());
        mlir::scf::YieldOp::create(builder, fallback);
      }
      builder.setInsertionPointAfter(load);
      return load.getResult(0);
    };

    auto block_reduce = [&](Value local, mlir::func::FuncOp reducer,
                            Value identity, bool is_maximum) -> Value {
      if (independent_rows_ || num_warps_ == 1) {
        auto combine = [&](Value lhs, Value rhs) -> Value {
          return is_maximum ? mlir::arith::MaxNumFOp::create(builder, lhs, rhs)
                                  .getResult()
                            : mlir::arith::AddFOp::create(builder, lhs, rhs)
                                  .getResult();
        };
        Value poison = mlir::ub::PoisonOp::create(builder, local.getType());
        for (int32_t distance : {8, 4, 2, 1}) {
          Value shuffled = mlir::amdgpu::DPPOp::create(
              builder, local.getType(), /*old=*/poison, /*src=*/local,
              mlir::amdgpu::DPPPerm::row_shr,
              builder.getI32IntegerAttr(distance), /*row_mask=*/0xF,
              /*bank_mask=*/0xF, /*bound_ctrl=*/true);
          local = combine(local, shuffled);
        }
        Value adjacent_row = mlir::amdgpu::DPPOp::create(
            builder, local.getType(), /*old=*/local, /*src=*/local,
            mlir::amdgpu::DPPPerm::row_bcast_15,
            /*permArgument=*/nullptr, /*row_mask=*/0xA, /*bank_mask=*/0xF,
            /*bound_ctrl=*/true);
        local = combine(local, adjacent_row);
        Value lower_half = mlir::amdgpu::DPPOp::create(
            builder, local.getType(), /*old=*/poison, /*src=*/local,
            mlir::amdgpu::DPPPerm::row_bcast_31,
            /*permArgument=*/nullptr, /*row_mask=*/0xF, /*bank_mask=*/0xF,
            /*bound_ctrl=*/true);
        local = combine(local, lower_half);
        return mlir::gpu::ShuffleOp::create(builder, local, /*offset=*/63,
                                            /*width=*/64,
                                            mlir::gpu::ShuffleMode::IDX)
            .getShuffleResult();
      }
      Value wave_reduced =
          ShuffleReduceOp::create(builder, reducer, mlir::ValueRange{local}, 32)
              .getResult(0);
      Value is_lane_zero =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                      lane_id, IndexConstant(builder, 0));
      shared = conditional_store(shared, is_lane_zero, wave_reduced, wave_id);
      shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{shared_type}, shared)
              .getResult(0);

      Value is_first_wave =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                      wave_id, IndexConstant(builder, 0));
      Value has_wave_value = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, lane_id,
          IndexConstant(builder, num_warps_));
      Value reads_cross_wave =
          mlir::arith::AndIOp::create(builder, is_first_wave, has_wave_value);
      Value cross_wave =
          conditional_load(shared, reads_cross_wave, lane_id, identity);
      Value block_reduced =
          ShuffleReduceOp::create(builder, reducer,
                                  mlir::ValueRange{cross_wave}, 32)
              .getResult(0);
      Value is_thread_zero =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                      thread_id, IndexConstant(builder, 0));
      shared = conditional_store(shared, is_thread_zero, block_reduced,
                                 IndexConstant(builder, 0));
      shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{shared_type}, shared)
              .getResult(0);
      return mlir::tensor::ExtractOp::create(
          builder, shared, mlir::ValueRange{IndexConstant(builder, 0)});
    };

    std::vector<Value> normalized_values;
    normalized_values.reserve(values.size());
    if (is_layer_norm_) {
      Value block_sum = block_reduce(local_sum, add_reducer, zero,
                                     /*is_maximum=*/false);
      Value reciprocal_columns = mlir::arith::ConstantFloatOp::create(
          builder, builder.getF32Type(),
          llvm::APFloat(1.0f / static_cast<float>(columns_)));
      Value mean =
          mlir::arith::MulFOp::create(builder, block_sum, reciprocal_columns);
      if (mean_output) {
        mean_output = store_row_output(mean_output, mean);
      }
      std::vector<Value> centered;
      centered.reserve(values.size());
      Value local_squared_sum = zero;
      for (Value value : values) {
        Value centered_value =
            mlir::arith::SubFOp::create(builder, value, mean);
        centered.push_back(centered_value);
        if (!uses_moments_variance_) {
          Value square = mlir::arith::MulFOp::create(builder, centered_value,
                                                     centered_value);
          local_squared_sum =
              mlir::arith::AddFOp::create(builder, local_squared_sum, square);
        }
      }
      Value variance;
      if (uses_moments_variance_) {
        Value block_input_squared_sum =
            block_reduce(local_input_squared_sum, add_reducer, zero,
                         /*is_maximum=*/false);
        Value square_mean = mlir::arith::MulFOp::create(
            builder, block_input_squared_sum, reciprocal_columns);
        Value mean_square = mlir::arith::MulFOp::create(builder, mean, mean);
        variance =
            mlir::arith::SubFOp::create(builder, square_mean, mean_square);
      } else {
        Value block_squared_sum = block_reduce(local_squared_sum, add_reducer,
                                               zero, /*is_maximum=*/false);
        variance = mlir::arith::MulFOp::create(builder, block_squared_sum,
                                               reciprocal_columns);
      }
      Value epsilon = mlir::arith::ConstantFloatOp::create(
          builder, builder.getF32Type(),
          llvm::APFloat(static_cast<float>(epsilon_)));
      Value variance_epsilon =
          mlir::arith::AddFOp::create(builder, variance, epsilon);
      Value reciprocal_stddev =
          mlir::math::RsqrtOp::create(builder, variance_epsilon);
      if (reciprocal_stddev_output) {
        reciprocal_stddev_output =
            store_row_output(reciprocal_stddev_output, reciprocal_stddev);
      }
      if (reciprocal_stddev_cube_output) {
        Value reciprocal_stddev_cube = mlir::arith::DivFOp::create(
            builder, reciprocal_stddev, variance_epsilon);
        reciprocal_stddev_cube_output = store_row_output(
            reciprocal_stddev_cube_output, reciprocal_stddev_cube);
      }
      Value one = mlir::arith::ConstantFloatOp::create(
          builder, builder.getF32Type(), llvm::APFloat(1.0f));
      for (int64_t i = 0; i < centered.size(); ++i) {
        Value result = mlir::arith::MulFOp::create(builder, centered[i],
                                                   reciprocal_stddev);
        Value column = Add(builder, element_thread_id,
                           IndexConstant(builder, i * threads));
        const bool all_columns_in_bounds = (i + 1) * threads <= columns_;
        if (gamma) {
          Value scale =
              load_minor_vector(gamma, column, all_columns_in_bounds, one);
          result = mlir::arith::MulFOp::create(builder, result, scale);
        }
        if (beta) {
          Value bias =
              load_minor_vector(beta, column, all_columns_in_bounds, zero);
          result = mlir::arith::AddFOp::create(builder, result, bias);
        }
        normalized_values.push_back(result);
      }
    } else {
      Value block_max;
      if (has_external_row && !recompute_maximum_after_external_row_offset_) {
        block_max = row_offset;
      } else {
        block_max = block_reduce(local_max, maximum, minus_inf,
                                 /*is_maximum=*/true);
      }
      std::vector<Value> exponentials;
      exponentials.reserve(values.size());
      Value local_exponential_sum = zero;
      for (Value value : values) {
        Value shifted = mlir::arith::SubFOp::create(builder, value, block_max);
        Value scaled = mlir::arith::MulFOp::create(builder, shifted, log2e);
        Value exponential = mlir::ROCDL::ROCDLExp2::create(
            builder, builder.getF32Type(), scaled);
        exponentials.push_back(exponential);
        local_exponential_sum = mlir::arith::AddFOp::create(
            builder, local_exponential_sum, exponential);
      }
      Value block_sum = block_reduce(local_exponential_sum, add_reducer, zero,
                                     /*is_maximum=*/false);
      Value reciprocal = mlir::ROCDL::ROCDLRcp::create(
          builder, builder.getF32Type(), block_sum);
      if (is_softmax_backward_) {
        Value local_weighted_sum = zero;
        for (int64_t i = 0; i < exponentials.size(); ++i) {
          Value weighted = mlir::arith::MulFOp::create(
              builder, exponentials[i], upstream_gradient_values[i]);
          local_weighted_sum = mlir::arith::AddFOp::create(
              builder, local_weighted_sum, weighted);
        }
        Value block_weighted_sum =
            block_reduce(local_weighted_sum, add_reducer, zero,
                         /*is_maximum=*/false);
        Value mean_gradient = mlir::arith::MulFOp::create(
            builder, block_weighted_sum, reciprocal);
        for (int64_t i = 0; i < exponentials.size(); ++i) {
          Value centered_gradient = mlir::arith::SubFOp::create(
              builder, upstream_gradient_values[i], mean_gradient);
          Value probability =
              mlir::arith::MulFOp::create(builder, exponentials[i], reciprocal);
          Value result = mlir::arith::MulFOp::create(builder, centered_gradient,
                                                     probability);
          normalized_values.push_back(
              mlir::arith::MulFOp::create(builder, result, backward_scale));
        }
      } else {
        for (Value exponential : exponentials) {
          normalized_values.push_back(
              mlir::arith::MulFOp::create(builder, exponential, reciprocal));
        }
      }
    }
    mlir::Type result_type;
    switch (element_type_) {
      case F16:
        result_type = builder.getF16Type();
        break;
      case BF16:
        result_type = builder.getBF16Type();
        break;
      case F32:
        result_type = builder.getF32Type();
        break;
      default:
        return absl::InvalidArgumentError(
            "Fly softmax requires F16, BF16, or F32 results.");
    }
    for (int64_t i = 0; i < normalized_values.size(); ++i) {
      Value result = normalized_values[i];
      if (!result_type.isF32()) {
        result = mlir::arith::TruncFOp::create(builder, result_type, result);
      }
      Value column =
          Add(builder, element_thread_id, IndexConstant(builder, i * threads));
      if (all_rows_in_bounds && (i + 1) * threads <= columns_) {
        std::vector<Value> indices = output_row_indices;
        indices.push_back(column);
        output = mlir::tensor::InsertOp::create(builder, result, output,
                                                mlir::ValueRange(indices));
        continue;
      }
      Value in_bounds =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::ult,
                                      column, IndexConstant(builder, columns_));
      if (!all_rows_in_bounds) {
        in_bounds =
            mlir::arith::AndIOp::create(builder, in_bounds, row_in_bounds);
      }
      mlir::scf::IfOp store = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{output.getType()}, in_bounds,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(store.thenBlock());
        std::vector<Value> indices = output_row_indices;
        indices.push_back(column);
        Value updated = mlir::tensor::InsertOp::create(
            builder, result, output, mlir::ValueRange(indices));
        mlir::scf::YieldOp::create(builder, updated);
        builder.setInsertionPointToStart(store.elseBlock());
        mlir::scf::YieldOp::create(builder, output);
      }
      builder.setInsertionPointAfter(store);
      output = store.getResult(0);
    }

    llvm::SmallVector<Value> outputs{output};
    if (output_count_ > 1) {
      outputs.push_back(mean_output);
      outputs.push_back(reciprocal_stddev_output);
    }
    if (output_count_ > 3) {
      outputs.push_back(reciprocal_stddev_cube_output);
    }
    mlir::func::ReturnOp::create(builder, outputs);
    return absl::OkStatus();
  }

  int64_t rows_;
  int64_t columns_;
  PrimitiveType element_type_;
  std::vector<int64_t> dimensions_;
  bool independent_rows_ = false;
  int64_t input_parameter_number_ = 0;
  int64_t upstream_gradient_parameter_number_ = -1;
  int64_t external_row_parameter_number_ = -1;
  bool recompute_maximum_after_external_row_offset_ = false;
  bool is_layer_norm_ = false;
  bool is_softmax_backward_ = false;
  double softmax_backward_scale_ = 1.0;
  double softmax_input_scale_ = 1.0;
  double epsilon_ = 0.0;
  int64_t layer_norm_parameter_count_ = 0;
  int64_t gamma_parameter_number_ = -1;
  int64_t beta_parameter_number_ = -1;
  int64_t output_count_ = 1;
  bool uses_moments_variance_ = false;
  int64_t num_warps_ = 4;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileSoftmaxEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileSoftmaxEmitter>(analysis);
}

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileLayerNormEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileSoftmaxEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
