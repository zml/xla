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

#include "xla/backends/gpu/codegen/emitters/vulkan_flash_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status_macros.h"
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

constexpr int64_t kThreadsPerBlock = 128;

Value IndexConstant(ImplicitLocOpBuilder& builder, int64_t value) {
  return arith::ConstantIndexOp::create(builder, value);
}

Value F32Constant(ImplicitLocOpBuilder& builder, float value) {
  return arith::ConstantFloatOp::create(builder, builder.getF32Type(),
                                        llvm::APFloat(value));
}

Value I1Constant(ImplicitLocOpBuilder& builder, bool value) {
  return arith::ConstantIntOp::create(builder, builder.getI1Type(), value);
}

mlir::Type GetElementType(Value tensor) {
  return mlir::cast<mlir::ShapedType>(tensor.getType()).getElementType();
}

Value ZeroOfType(ImplicitLocOpBuilder& builder, mlir::Type type) {
  return arith::ConstantOp::create(builder, builder.getZeroAttr(type));
}

}  // namespace

namespace internal {

Value SelectTensorElementOrZero(ImplicitLocOpBuilder& builder, Value condition,
                                Value tensor, ValueRange indices) {
  mlir::Type element_type = GetElementType(tensor);
  scf::IfOp if_op = scf::IfOp::create(builder, TypeRange{element_type},
                                      condition, /*withElseRegion=*/true);
  OpBuilder then_builder = if_op.getThenBodyBuilder();
  ImplicitLocOpBuilder implicit_then(builder.getLoc(), then_builder);
  Value loaded = tensor::ExtractOp::create(implicit_then, tensor, indices);
  scf::YieldOp::create(implicit_then, loaded);
  OpBuilder else_builder = if_op.getElseBodyBuilder();
  ImplicitLocOpBuilder implicit_else(builder.getLoc(), else_builder);
  scf::YieldOp::create(implicit_else, ZeroOfType(implicit_else, element_type));
  return if_op.getResult(0);
}

}  // namespace internal

namespace {

ValueRange UpdateIf(
    ImplicitLocOpBuilder& builder, Value condition, ValueRange current,
    llvm::function_ref<SmallVector<Value>(ImplicitLocOpBuilder&)> update) {
  scf::IfOp if_op = scf::IfOp::create(builder, TypeRange(current), condition,
                                      /*withElseRegion=*/true);
  OpBuilder then_builder = if_op.getThenBodyBuilder();
  ImplicitLocOpBuilder implicit_then(builder.getLoc(), then_builder);
  scf::YieldOp::create(implicit_then, update(implicit_then));
  OpBuilder else_builder = if_op.getElseBodyBuilder();
  ImplicitLocOpBuilder implicit_else(builder.getLoc(), else_builder);
  scf::YieldOp::create(implicit_else, current);
  return if_op.getResults();
}

Value ToF32(ImplicitLocOpBuilder& builder, Value value) {
  return arith::ExtFOp::create(builder, builder.getF32Type(), value);
}

Value ShuffleSum(ImplicitLocOpBuilder& builder, Value value,
                 int64_t subgroup_size) {
  Value zero = F32Constant(builder, 0.0f);
  for (int64_t offset = subgroup_size / 2; offset > 0; offset /= 2) {
    Value offset_value =
        arith::ConstantIntOp::create(builder, builder.getI32Type(), offset);
    Value width_value = arith::ConstantIntOp::create(
        builder, builder.getI32Type(), subgroup_size);
    mlir::gpu::ShuffleOp shuffle =
        mlir::gpu::ShuffleOp::create(builder, value, offset_value, width_value,
                                     mlir::gpu::ShuffleMode::DOWN);
    Value shuffled = arith::SelectOp::create(builder, shuffle.getValid(),
                                             shuffle.getShuffleResult(), zero);
    value = arith::AddFOp::create(builder, value, shuffled);
  }
  return value;
}

Value ReduceWorkgroup(ImplicitLocOpBuilder& builder, Value partial, Value tid,
                      Value lane, Value subgroup, Value shared,
                      int64_t subgroup_size) {
  const int64_t subgroup_count = kThreadsPerBlock / subgroup_size;
  Value shared_allocation = shared;
  Value reduced = ShuffleSum(builder, partial, subgroup_size);
  Value lane_zero = arith::CmpIOp::create(builder, arith::CmpIPredicate::eq,
                                          lane, IndexConstant(builder, 0));
  Value subgroup_values =
      UpdateIf(builder, lane_zero, ValueRange{shared},
               [&](ImplicitLocOpBuilder& nested) -> SmallVector<Value> {
                 return {tensor::InsertOp::create(nested, reduced, shared,
                                                  ValueRange{subgroup})};
               })[0];
  SyncThreadsOp::create(builder, TypeRange{subgroup_values.getType()},
                        ValueRange{subgroup_values});

  Value is_first_subgroup =
      arith::CmpIOp::create(builder, arith::CmpIPredicate::ult, tid,
                            IndexConstant(builder, subgroup_count));
  Value subgroup_sum = internal::SelectTensorElementOrZero(
      builder, is_first_subgroup, shared_allocation, ValueRange{tid});
  subgroup_sum = ShuffleSum(builder, subgroup_sum, subgroup_size);
  Value thread_zero = arith::CmpIOp::create(builder, arith::CmpIPredicate::eq,
                                            tid, IndexConstant(builder, 0));
  Value workgroup_sum =
      UpdateIf(builder, thread_zero, ValueRange{shared_allocation},
               [&](ImplicitLocOpBuilder& nested) -> SmallVector<Value> {
                 return {tensor::InsertOp::create(
                     nested, subgroup_sum, shared_allocation,
                     ValueRange{IndexConstant(nested, 0)})};
               })[0];
  SyncThreadsOp::create(builder, TypeRange{workgroup_sum.getType()},
                        ValueRange{workgroup_sum});
  return tensor::ExtractOp::create(builder, shared_allocation,
                                   ValueRange{IndexConstant(builder, 0)});
}

}  // namespace

VulkanFlashAttentionEmitter::VulkanFlashAttentionEmitter(
    const HloFusionInstruction& fusion, const se::DeviceDescription& device)
    : query_heads_(fusion.operand(0)->shape().dimensions(0)),
      query_length_(fusion.operand(0)->shape().dimensions(1)),
      kv_heads_(fusion.operand(1)->shape().dimensions(0)),
      kv_length_(fusion.operand(1)->shape().dimensions(1)),
      head_dim_(fusion.operand(0)->shape().dimensions(2)),
      subgroup_size_(device.threads_per_warp()) {}

LaunchDimensions VulkanFlashAttentionEmitter::launch_dimensions() const {
  return LaunchDimensions(se::BlockDim(query_heads_ * query_length_, 1, 1),
                          se::ThreadDim(kThreadsPerBlock, 1, 1));
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
VulkanFlashAttentionEmitter::CreateMLIRModule(
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
      << "Vulkan flash-attention emitter produced invalid MLIR";
  return module;
}

std::optional<IndexingMap>
VulkanFlashAttentionEmitter::ComputeThreadIdToOutputIndexing(
    int64_t, mlir::MLIRContext*) const {
  return std::nullopt;
}

std::optional<std::vector<IndexingMap>>
VulkanFlashAttentionEmitter::ComputeThreadIdToInputIndexing(
    int64_t, mlir::MLIRContext*) const {
  return std::nullopt;
}

absl::Status VulkanFlashAttentionEmitter::EmitEntryFunction(
    const emitters::PartitionedComputations&,
    const emitters::CallTargetProvider&, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  return EmitKernel(entry_function, fusion);
}

absl::Status VulkanFlashAttentionEmitter::EmitKernel(
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  if (subgroup_size_ != 16 && subgroup_size_ != 32 && subgroup_size_ != 64) {
    return absl::FailedPreconditionError(
        "Vulkan flash-attention requires subgroup size 16, 32, or 64");
  }

  ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());
  Value q = entry_function.getArgument(0);
  Value k = entry_function.getArgument(1);
  Value v = entry_function.getArgument(2);
  Value token_index_tensor = entry_function.getArgument(3);
  Value output = entry_function.getArgument(fusion.operand_count());

  Value tid = EmitThreadId(builder, 0);
  Value block = EmitBlockId(builder, 0);
  Value query_length = IndexConstant(builder, query_length_);
  Value query_head = arith::DivUIOp::create(builder, block, query_length);
  Value query_row = arith::RemUIOp::create(builder, block, query_length);
  Value group_size = IndexConstant(builder, query_heads_ / kv_heads_);
  Value kv_head = arith::DivUIOp::create(builder, query_head, group_size);
  Value subgroup_size = IndexConstant(builder, subgroup_size_);
  Value lane = arith::RemUIOp::create(builder, tid, subgroup_size);
  Value subgroup = arith::DivUIOp::create(builder, tid, subgroup_size);

  Value token_index_i32 =
      tensor::ExtractOp::create(builder, token_index_tensor, ValueRange{});
  Value token_index = arith::IndexCastOp::create(
      builder, builder.getIndexType(), token_index_i32);
  Value key_limit = arith::AddIOp::create(
      builder, arith::AddIOp::create(builder, token_index, query_row),
      IndexConstant(builder, 1));
  Value zero_index = IndexConstant(builder, 0);
  Value key_limit_positive = arith::CmpIOp::create(
      builder, arith::CmpIPredicate::sgt, key_limit, zero_index);
  key_limit = arith::SelectOp::create(builder, key_limit_positive, key_limit,
                                      zero_index);
  Value kv_length = IndexConstant(builder, kv_length_);
  Value below_kv_length = arith::CmpIOp::create(
      builder, arith::CmpIPredicate::ult, key_limit, kv_length);
  key_limit =
      arith::SelectOp::create(builder, below_kv_length, key_limit, kv_length);

  Value row_is_valid = I1Constant(builder, true);
  if (fusion.operand_count() == 5) {
    Value num_tokens_u32 = tensor::ExtractOp::create(
        builder, entry_function.getArgument(4), ValueRange{});
    Value num_tokens = arith::IndexCastOp::create(
        builder, builder.getIndexType(), num_tokens_u32);
    row_is_valid = arith::CmpIOp::create(builder, arith::CmpIPredicate::ult,
                                         query_row, num_tokens);
    key_limit =
        arith::SelectOp::create(builder, row_is_valid, key_limit, zero_index);
  }

  Value shared = AllocateSharedOp::create(
      builder, RankedTensorType::get({kThreadsPerBlock / subgroup_size_},
                                     builder.getF32Type()));
  Value negative_infinity =
      F32Constant(builder, -std::numeric_limits<float>::infinity());
  Value zero = F32Constant(builder, 0.0f);

  SmallVector<Value> initial{negative_infinity, zero, zero};
  if (head_dim_ > kThreadsPerBlock) initial.push_back(zero);
  initial.push_back(shared);
  scf::ForOp key_loop = scf::ForOp::create(builder, zero_index, key_limit,
                                           IndexConstant(builder, 1), initial);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(key_loop.getBody());
    Value key_index = key_loop.getInductionVar();
    Value running_max = key_loop.getRegionIterArgs()[0];
    Value running_sum = key_loop.getRegionIterArgs()[1];
    Value output0 = key_loop.getRegionIterArgs()[2];
    int shared_index = head_dim_ > kThreadsPerBlock ? 4 : 3;
    Value output1 =
        head_dim_ > kThreadsPerBlock ? key_loop.getRegionIterArgs()[3] : zero;
    Value loop_shared = key_loop.getRegionIterArgs()[shared_index];

    Value dim0_valid =
        arith::CmpIOp::create(builder, arith::CmpIPredicate::ult, tid,
                              IndexConstant(builder, head_dim_));
    Value q0 = internal::SelectTensorElementOrZero(
        builder, dim0_valid, q, ValueRange{query_head, query_row, tid});
    Value k0 = internal::SelectTensorElementOrZero(
        builder, dim0_valid, k, ValueRange{kv_head, key_index, tid});
    Value partial =
        arith::MulFOp::create(builder, ToF32(builder, q0), ToF32(builder, k0));
    if (head_dim_ > kThreadsPerBlock) {
      Value dim1 = arith::AddIOp::create(
          builder, tid, IndexConstant(builder, kThreadsPerBlock));
      Value dim1_valid =
          arith::CmpIOp::create(builder, arith::CmpIPredicate::ult, dim1,
                                IndexConstant(builder, head_dim_));
      Value q1 = internal::SelectTensorElementOrZero(
          builder, dim1_valid, q, ValueRange{query_head, query_row, dim1});
      Value k1 = internal::SelectTensorElementOrZero(
          builder, dim1_valid, k, ValueRange{kv_head, key_index, dim1});
      partial = arith::AddFOp::create(
          builder, partial,
          arith::MulFOp::create(builder, ToF32(builder, q1),
                                ToF32(builder, k1)));
    }

    Value dot = ReduceWorkgroup(builder, partial, tid, lane, subgroup,
                                loop_shared, subgroup_size_);
    Value scale = F32Constant(builder, 1.0f / std::sqrt(head_dim_));
    Value score = arith::MulFOp::create(builder, dot, scale);
    Value new_max = arith::MaximumFOp::create(builder, running_max, score);
    Value alpha = mlir::math::ExpOp::create(
        builder, arith::SubFOp::create(builder, running_max, new_max));
    Value beta = mlir::math::ExpOp::create(
        builder, arith::SubFOp::create(builder, score, new_max));
    Value new_sum = arith::AddFOp::create(
        builder, arith::MulFOp::create(builder, running_sum, alpha), beta);
    Value v0 = internal::SelectTensorElementOrZero(
        builder, dim0_valid, v, ValueRange{kv_head, key_index, tid});
    Value new_output0 = arith::AddFOp::create(
        builder, arith::MulFOp::create(builder, output0, alpha),
        arith::MulFOp::create(builder, beta, ToF32(builder, v0)));

    SmallVector<Value> yielded{new_max, new_sum, new_output0};
    if (head_dim_ > kThreadsPerBlock) {
      Value dim1 = arith::AddIOp::create(
          builder, tid, IndexConstant(builder, kThreadsPerBlock));
      Value dim1_valid =
          arith::CmpIOp::create(builder, arith::CmpIPredicate::ult, dim1,
                                IndexConstant(builder, head_dim_));
      Value v1 = internal::SelectTensorElementOrZero(
          builder, dim1_valid, v, ValueRange{kv_head, key_index, dim1});
      Value new_output1 = arith::AddFOp::create(
          builder, arith::MulFOp::create(builder, output1, alpha),
          arith::MulFOp::create(builder, beta, ToF32(builder, v1)));
      yielded.push_back(new_output1);
    }
    yielded.push_back(loop_shared);
    scf::YieldOp::create(builder, yielded);
  }

  Value final_sum = key_loop.getResult(1);
  Value has_keys = arith::CmpFOp::create(builder, arith::CmpFPredicate::OGT,
                                         final_sum, zero);
  Value write0 = arith::CmpIOp::create(builder, arith::CmpIPredicate::ult, tid,
                                       IndexConstant(builder, head_dim_));
  Value normalized0 =
      arith::DivFOp::create(builder, key_loop.getResult(2), final_sum);
  normalized0 = arith::SelectOp::create(builder, has_keys, normalized0, zero);
  output = UpdateIf(
      builder, write0, ValueRange{output},
      [&](ImplicitLocOpBuilder& nested) -> SmallVector<Value> {
        Value bf16 =
            arith::TruncFOp::create(nested, nested.getBF16Type(), normalized0);
        return {tensor::InsertOp::create(
            nested, bf16, output, ValueRange{query_head, query_row, tid})};
      })[0];

  if (head_dim_ > kThreadsPerBlock) {
    Value dim1 = arith::AddIOp::create(
        builder, tid, IndexConstant(builder, kThreadsPerBlock));
    Value write1 =
        arith::CmpIOp::create(builder, arith::CmpIPredicate::ult, dim1,
                              IndexConstant(builder, head_dim_));
    Value normalized1 =
        arith::DivFOp::create(builder, key_loop.getResult(3), final_sum);
    normalized1 = arith::SelectOp::create(builder, has_keys, normalized1, zero);
    output = UpdateIf(
        builder, write1, ValueRange{output},
        [&](ImplicitLocOpBuilder& nested) -> SmallVector<Value> {
          Value bf16 = arith::TruncFOp::create(nested, nested.getBF16Type(),
                                               normalized1);
          return {tensor::InsertOp::create(
              nested, bf16, output, ValueRange{query_head, query_row, dim1})};
        })[0];
  }

  mlir::func::ReturnOp::create(builder, output);
  return absl::OkStatus();
}

}  // namespace xla::gpu
