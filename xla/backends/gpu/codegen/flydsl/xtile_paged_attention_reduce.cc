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

#include "xla/backends/gpu/codegen/flydsl/xtile_paged_attention_reduce.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "xla/backends/gpu/codegen/flydsl/paged_attention_support.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using mlir::Value;

constexpr int64_t kWaveSize = 64;
constexpr int64_t kBlockThreads = kWaveSize;
constexpr int64_t kOutputElementsPerThread = 2;

Value IndexConstant(mlir::ImplicitLocOpBuilder& builder, int64_t value) {
  return mlir::arith::ConstantIndexOp::create(builder, value);
}

Value Add(mlir::ImplicitLocOpBuilder& builder, Value lhs, Value rhs) {
  return mlir::arith::AddIOp::create(builder, lhs, rhs);
}

Value Mul(mlir::ImplicitLocOpBuilder& builder, Value lhs, Value rhs) {
  return mlir::arith::MulIOp::create(builder, lhs, rhs);
}

Value Rem(mlir::ImplicitLocOpBuilder& builder, Value lhs, int64_t rhs) {
  return mlir::arith::RemUIOp::create(builder, lhs,
                                      IndexConstant(builder, rhs));
}

Value F32Constant(mlir::ImplicitLocOpBuilder& builder, float value) {
  return mlir::arith::ConstantFloatOp::create(builder, builder.getF32Type(),
                                              llvm::APFloat(value));
}

Value NegativeInfinity(mlir::ImplicitLocOpBuilder& builder) {
  return mlir::arith::ConstantFloatOp::create(
      builder, builder.getF32Type(),
      llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(),
                            /*negative=*/true));
}

Value EmitBufferLoad(mlir::ImplicitLocOpBuilder& builder,
                     mlir::Location location, Value source, Value index,
                     mlir::Type result_type) {
  mlir::OperationState state(location, "xla_gpu.buffer_load");
  state.addOperands({source, index});
  state.addTypes(result_type);
  return builder.create(state)->getResult(0);
}

Value EmitBufferStore(mlir::ImplicitLocOpBuilder& builder,
                      mlir::Location location, Value value, Value destination,
                      Value index) {
  mlir::OperationState state(location, "xla_gpu.buffer_store");
  state.addOperands({value, destination, index});
  state.addTypes(destination.getType());
  state.addAttribute("cache_policy", builder.getI32IntegerAttr(0));
  return builder.create(state)->getResult(0);
}

class FlyXTilePagedAttentionSegmentedReducerEmitter final
    : public MlirKernelEmitter {
 public:
  explicit FlyXTilePagedAttentionSegmentedReducerEmitter(
      const HloFusionAnalysis& analysis)
      : descriptor_(*GetFlyPagedAttentionSegmentedReducerDescriptor(analysis)),
        launch_dimensions_(
            se::BlockDim(descriptor_.sequences, descriptor_.query_heads, 1),
            se::ThreadDim(kBlockThreads, 1, 1)) {}

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
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
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
    return module;
  }

  absl::Status EmitEntryFunction(const emitters::PartitionedComputations&,
                                 const emitters::CallTargetProvider&,
                                 mlir::func::FuncOp,
                                 const HloFusionInstruction&) const override {
    return absl::UnimplementedError(
        "The Fly paged-attention reducer builds its module directly.");
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function) const {
    TF_RET_CHECK(entry_function.getNumArguments() == 4)
        << "reducer kernel arguments: " << entry_function.getNumArguments();
    TF_RET_CHECK(descriptor_.num_segments > 1 &&
                 descriptor_.num_segments <= 256 &&
                 descriptor_.head_dimension ==
                     kBlockThreads * kOutputElementsPerThread);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());
    Value partial_output = entry_function.getArgument(0);
    Value partial_maximum = entry_function.getArgument(1);
    Value partial_sum_buffer = entry_function.getArgument(2);
    Value output = entry_function.getArgument(3);
    mlir::Type element_type = descriptor_.element_type == BF16
                                  ? builder.getBF16Type()
                                  : builder.getF16Type();
    auto f32_pair_type =
        mlir::VectorType::get({kOutputElementsPerThread}, builder.getF32Type());
    auto output_pair_type =
        mlir::VectorType::get({kOutputElementsPerThread}, element_type);
    Value zero = F32Constant(builder, 0.0f);
    Value one = F32Constant(builder, 1.0f);
    Value minus_inf = NegativeInfinity(builder);

    Value thread = EmitThreadId(builder, 0);
    Value lane = Rem(builder, thread, kWaveSize);
    Value sequence = EmitBlockId(builder, 0);
    Value query_head = EmitBlockId(builder, 1);
    auto wave_reduce = [&](Value value, bool maximum) {
      for (int64_t distance : {32, 16, 8, 4, 2, 1}) {
        Value peer = mlir::gpu::ShuffleOp::create(
                         builder, value, distance, kWaveSize,
                         mlir::gpu::ShuffleMode::XOR)
                         .getShuffleResult();
        value = maximum
                    ? mlir::arith::MaxNumFOp::create(builder, value, peer)
                          .getResult()
                    : mlir::arith::AddFOp::create(builder, value, peer)
                          .getResult();
      }
      return value;
    };

    constexpr int64_t kMaxPartsPerLane = 4;
    const int64_t parts_per_lane =
        (descriptor_.num_segments + kWaveSize - 1) / kWaveSize;
    TF_RET_CHECK(parts_per_lane <= kMaxPartsPerLane);
    std::vector<Value> part_valid;
    std::vector<Value> part_sum;
    std::vector<Value> part_max;
    part_valid.reserve(parts_per_lane);
    part_sum.reserve(parts_per_lane);
    part_max.reserve(parts_per_lane);
    Value local_max = minus_inf;
    for (int64_t lane_part = 0; lane_part < parts_per_lane; ++lane_part) {
      Value part = Add(builder, lane,
                       IndexConstant(builder, lane_part * kWaveSize));
      Value valid_index = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, part,
          IndexConstant(builder, descriptor_.num_segments));
      Value safe_part = mlir::arith::SelectOp::create(
          builder, valid_index, part, IndexConstant(builder, 0));
      Value partial_state_index = Add(
          builder,
          Mul(builder,
              Add(builder,
                  Mul(builder, sequence,
                      IndexConstant(builder, descriptor_.query_heads)),
                  query_head),
              IndexConstant(builder, descriptor_.num_segments)),
          safe_part);
      Value sum = EmitBufferLoad(
          builder, entry_function.getLoc(), partial_sum_buffer,
          partial_state_index,
          builder.getF32Type());
      Value maximum = EmitBufferLoad(
          builder, entry_function.getLoc(), partial_maximum,
          partial_state_index,
          builder.getF32Type());
      Value has_mass = mlir::arith::CmpFOp::create(
          builder, mlir::arith::CmpFPredicate::OGT, sum, zero);
      Value valid = mlir::arith::AndIOp::create(builder, valid_index, has_mass);
      sum = mlir::arith::SelectOp::create(builder, valid, sum, zero);
      maximum =
          mlir::arith::SelectOp::create(builder, valid, maximum, minus_inf);
      part_valid.push_back(valid);
      part_sum.push_back(sum);
      part_max.push_back(maximum);
      local_max =
          mlir::arith::MaxNumFOp::create(builder, local_max, maximum);
    }
    Value global_max = wave_reduce(local_max, /*maximum=*/true);

    std::vector<Value> part_scale;
    part_scale.reserve(parts_per_lane);
    Value local_sum = zero;
    for (int64_t lane_part = 0; lane_part < parts_per_lane; ++lane_part) {
      Value exponent = mlir::arith::SubFOp::create(
          builder, part_max[lane_part], global_max);
      Value candidate = mlir::ROCDL::ROCDLExp2::create(
          builder, builder.getF32Type(), exponent);
      Value scale = mlir::arith::SelectOp::create(
          builder, part_valid[lane_part], candidate, zero);
      part_scale.push_back(scale);
      local_sum = mlir::arith::AddFOp::create(
          builder, local_sum,
          mlir::arith::MulFOp::create(builder, part_sum[lane_part], scale));
    }
    Value global_sum = wave_reduce(local_sum, /*maximum=*/false);
    Value positive_sum = mlir::arith::CmpFOp::create(
        builder, mlir::arith::CmpFPredicate::OGT, global_sum, zero);
    Value safe_sum =
        mlir::arith::SelectOp::create(builder, positive_sum, global_sum, one);
    Value inverse_sum = mlir::ROCDL::ROCDLRcp::create(
        builder, builder.getF32Type(), safe_sum);

    std::vector<Value> part_weight;
    part_weight.reserve(parts_per_lane);
    for (int64_t lane_part = 0; lane_part < parts_per_lane; ++lane_part) {
      part_weight.push_back(mlir::arith::MulFOp::create(
          builder, part_scale[lane_part], inverse_sum));
    }

    // A single Wave64 owns all D=128 output elements. Each lane carries two
    // adjacent columns, so the state reduction and weight broadcasts execute
    // once instead of being duplicated by two identical waves. The paired
    // F32 loads and BF16/F16 stores are naturally aligned because every
    // partial row and output row is 128 elements wide.
    Value output_column = Mul(
        builder, thread, IndexConstant(builder, kOutputElementsPerThread));
    Value result =
        mlir::vector::BroadcastOp::create(builder, f32_pair_type, zero);
    for (int64_t part = 0; part < descriptor_.num_segments; ++part) {
      Value source_lane_byte = mlir::arith::ConstantIntOp::create(
          builder, builder.getI32Type(), (part % kWaveSize) * 4);
      Value weight_bits = mlir::arith::BitcastOp::create(
          builder, builder.getI32Type(), part_weight[part / kWaveSize]);
      weight_bits = mlir::ROCDL::DsBpermuteOp::create(
          builder, builder.getI32Type(), source_lane_byte, weight_bits);
      Value weight = mlir::arith::BitcastOp::create(
          builder, builder.getF32Type(), weight_bits);
      Value partial_base = Mul(
          builder,
          Add(builder,
              Mul(builder,
                  Add(builder,
                      Mul(builder, sequence,
                          IndexConstant(builder, descriptor_.query_heads)),
                      query_head),
                  IndexConstant(builder, descriptor_.num_segments)),
              IndexConstant(builder, part)),
          IndexConstant(builder, descriptor_.head_dimension));
      Value partial = EmitBufferLoad(
          builder, entry_function.getLoc(), partial_output,
          Add(builder, partial_base, output_column), f32_pair_type);
      Value weight_pair = mlir::vector::BroadcastOp::create(
          builder, f32_pair_type, weight);
      result = mlir::arith::AddFOp::create(
          builder, result,
          mlir::arith::MulFOp::create(builder, partial, weight_pair));
    }
    Value output_offset = Add(
        builder,
        Mul(builder,
            Add(builder,
                Mul(builder, sequence,
                    IndexConstant(builder, descriptor_.query_heads)),
                query_head),
            IndexConstant(builder, descriptor_.head_dimension)),
        output_column);
    Value output_value =
        mlir::arith::TruncFOp::create(builder, output_pair_type, result);
    output = EmitBufferStore(builder, entry_function.getLoc(), output_value,
                             output, output_offset);
    mlir::func::ReturnOp::create(builder, output);
    return absl::OkStatus();
  }

  FlyPagedAttentionSegmentedReducerDescriptor descriptor_;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

std::unique_ptr<MlirKernelEmitter>
CreateFlyXTilePagedAttentionSegmentedReducerEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<
      FlyXTilePagedAttentionSegmentedReducerEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
