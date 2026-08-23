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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using mlir::Value;

bool IsScalarConstant(const HloInstruction* instruction, double value) {
  if (instruction->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(instruction->shape())) {
    return false;
  }
  return instruction->literal().GetAsDouble({}) == value;
}

bool IsRowReduction(const HloInstruction* reduction, HloOpcode reducer_opcode,
                    const HloInstruction* input) {
  return reduction->opcode() == HloOpcode::kReduce &&
         reduction->operand_count() == 2 && reduction->operand(0) == input &&
         reduction->dimensions().size() == 1 && reduction->dimensions(0) == 1 &&
         reduction->called_computations().size() == 1 &&
         reduction->called_computations()
                 .front()
                 ->root_instruction()
                 ->opcode() == reducer_opcode;
}

const HloInstruction* BroadcastOperand(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kBroadcast ||
      instruction->dimensions().size() != 1 ||
      instruction->dimensions(0) != 0) {
    return nullptr;
  }
  return instruction->operand(0);
}

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
      is_maximum ? mlir::arith::MaximumFOp::create(
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
  explicit FlyXTileSoftmaxEmitter(const HloFusionAnalysis& analysis)
      : rows_(analysis.first_result_shape().dimensions(0)),
        columns_(analysis.first_result_shape().dimensions(1)) {
    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    num_warps_ = config.num_warps();
    launch_dimensions_ = LaunchDimensions(se::BlockDim(rows_, 1, 1),
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
    TF_RET_CHECK(entry_function.getNumArguments() == 2);
    const int64_t threads = num_warps_ * 64;
    TF_RET_CHECK(num_warps_ > 0 && num_warps_ <= 16 && columns_ % threads == 0);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());
    Value input = entry_function.getArgument(0);
    Value output = entry_function.getArgument(1);
    Value thread_id = EmitThreadId(builder, 0);
    Value block_id = EmitBlockId(builder, 0);
    Value wave_id = Div(builder, thread_id, 64);
    Value lane_id = Rem(builder, thread_id, 64);

    Value zero = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(), llvm::APFloat(0.0f));
    Value log2e = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(), llvm::APFloat(1.4426950408889634f));
    Value minus_inf = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(),
        llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(),
                              /*negative=*/true));

    std::vector<Value> values;
    values.reserve(columns_ / threads);
    Value local_max = minus_inf;
    for (int64_t i = 0; i < columns_ / threads; ++i) {
      Value column =
          Add(builder, thread_id, IndexConstant(builder, i * threads));
      Value value = mlir::tensor::ExtractOp::create(
          builder, input, mlir::ValueRange{block_id, column});
      value = mlir::arith::ExtFOp::create(builder, builder.getF32Type(), value);
      values.push_back(value);
      local_max = mlir::arith::MaximumFOp::create(builder, local_max, value);
    }

    auto shared_type =
        mlir::RankedTensorType::get({num_warps_}, builder.getF32Type());
    Value shared = AllocateSharedOp::create(builder, shared_type);

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
                            Value identity) -> Value {
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

    Value block_max = block_reduce(local_max, maximum, minus_inf);
    std::vector<Value> exponentials;
    exponentials.reserve(values.size());
    Value local_sum = zero;
    for (Value value : values) {
      Value shifted = mlir::arith::SubFOp::create(builder, value, block_max);
      Value scaled = mlir::arith::MulFOp::create(builder, shifted, log2e);
      Value exponential =
          mlir::ROCDL::ROCDLExp2::create(builder, builder.getF32Type(), scaled);
      exponentials.push_back(exponential);
      local_sum = mlir::arith::AddFOp::create(builder, local_sum, exponential);
    }

    Value block_sum = block_reduce(local_sum, add_reducer, zero);
    Value reciprocal =
        mlir::ROCDL::ROCDLRcp::create(builder, builder.getF32Type(), block_sum);
    for (int64_t i = 0; i < exponentials.size(); ++i) {
      Value normalized =
          mlir::arith::MulFOp::create(builder, exponentials[i], reciprocal);
      Value result = mlir::arith::TruncFOp::create(
          builder, builder.getBF16Type(), normalized);
      Value column =
          Add(builder, thread_id, IndexConstant(builder, i * threads));
      output = mlir::tensor::InsertOp::create(
          builder, result, output, mlir::ValueRange{block_id, column});
    }

    mlir::func::ReturnOp::create(builder, output);
    return absl::OkStatus();
  }

  int64_t rows_;
  int64_t columns_;
  int64_t num_warps_ = 4;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

bool IsFlySoftmaxFusion(const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return false;
  }
  const HloInstruction* root = &analysis.fusion_root(0).instruction();
  if (root->opcode() != HloOpcode::kConvert ||
      root->shape().element_type() != BF16 ||
      root->shape().dimensions_size() != 2) {
    return false;
  }
  const HloInstruction* normalized = root->operand(0);
  if (normalized->opcode() != HloOpcode::kDivide) {
    return false;
  }
  const HloInstruction* exponential = normalized->operand(0);
  const HloInstruction* row_sum = BroadcastOperand(normalized->operand(1));
  if (exponential->opcode() != HloOpcode::kExp || row_sum == nullptr ||
      !IsRowReduction(row_sum, HloOpcode::kAdd, exponential) ||
      !IsScalarConstant(row_sum->operand(1), 0.0)) {
    return false;
  }
  const HloInstruction* shifted = exponential->operand(0);
  if (shifted->opcode() != HloOpcode::kSubtract) {
    return false;
  }
  const HloInstruction* converted = shifted->operand(0);
  const HloInstruction* row_max = BroadcastOperand(shifted->operand(1));
  if (converted->opcode() != HloOpcode::kConvert ||
      converted->shape().element_type() != F32 ||
      converted->operand(0)->opcode() != HloOpcode::kParameter ||
      converted->operand(0)->shape().element_type() != BF16 ||
      row_max == nullptr ||
      !IsRowReduction(row_max, HloOpcode::kMaximum, converted) ||
      !IsScalarConstant(row_max->operand(1),
                        -std::numeric_limits<double>::infinity())) {
    return false;
  }
  const int64_t columns = root->shape().dimensions(1);
  return columns >= 64 && columns % 64 == 0 &&
         root->shape() == converted->operand(0)->shape();
}

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileSoftmaxEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileSoftmaxEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
