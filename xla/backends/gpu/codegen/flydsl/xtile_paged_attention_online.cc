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

#include "xla/backends/gpu/codegen/flydsl/xtile_paged_attention_online.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/flydsl/compiler.h"
#include "xla/backends/gpu/codegen/flydsl/paged_attention_support.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using mlir::Value;

constexpr int64_t kWaveSize = 64;
constexpr int64_t kMfmaN = 16;
constexpr int64_t kMfmaK = 16;
constexpr int64_t kMfmaInputsPerLane = 4;
constexpr int64_t kMfmaOutputsPerLane = 4;
constexpr int64_t kHeadSize = 128;
constexpr float kLog2E = 1.4426950408889634f;

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

Value ExtractVectorElement(mlir::ImplicitLocOpBuilder& builder, Value vector,
                           int64_t index) {
  return mlir::vector::ExtractOp::create(builder, vector,
                                         llvm::SmallVector<int64_t>{index});
}

Value EmitBufferLoad(mlir::ImplicitLocOpBuilder& builder,
                     mlir::Location location, Value source, Value index,
                     mlir::Type result_type, bool is_scalar = false,
                     int64_t cache_policy = 0) {
  mlir::OperationState state(location, is_scalar ? "xla_gpu.scalar_buffer_load"
                                                 : "xla_gpu.buffer_load");
  state.addOperands({source, index});
  state.addTypes(result_type);
  if (!is_scalar) {
    state.addAttribute("cache_policy",
                       builder.getI32IntegerAttr(cache_policy));
  }
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

Value EmitBufferCompletionTicket(mlir::ImplicitLocOpBuilder& builder,
                                 mlir::Location location, Value tickets,
                                 Value ticket_index) {
  mlir::OperationState state(location, "xla_gpu.buffer_completion_ticket");
  state.addOperands({tickets, ticket_index});
  state.addTypes(builder.getI32Type());
  return builder.create(state)->getResult(0);
}

Value MakeFlyIndex(mlir::ImplicitLocOpBuilder& builder, Value index) {
  if (index.getType().isIndex()) {
    index = mlir::arith::IndexCastOp::create(builder, builder.getI32Type(),
                                             index);
  } else if (index.getType().isInteger(64)) {
    index = mlir::arith::TruncIOp::create(builder, builder.getI32Type(), index);
  }
  auto index_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
      builder.getContext(), /*width=*/32, /*divisibility=*/1);
  auto index_type = mlir::fly::IntTupleType::get(index_attr);
  return mlir::fly::MakeIntTupleOp::create(builder, index_type,
                                           mlir::ValueRange{index});
}

Value CreateSharedMemRef(mlir::ImplicitLocOpBuilder& builder,
                         mlir::Type element_type, int64_t elements) {
  mlir::MLIRContext* context = builder.getContext();
  constexpr int64_t kAlignment = 16;
  const int64_t element_bytes = element_type.getIntOrFloatBitWidth() / 8;
  auto shared_address = mlir::fly::AddressSpaceAttr::get(
      context, mlir::fly::AddressSpace::Shared);
  auto alignment = mlir::fly::AlignAttr::get(context, kAlignment);
  auto raw_pointer_type = mlir::fly::PointerType::get(
      builder.getI8Type(), shared_address, alignment);
  auto allocation = builder.getDictionaryAttr(
      {builder.getNamedAttr("allocAlign",
                            builder.getI64IntegerAttr(kAlignment)),
       builder.getNamedAttr("allocBytes",
                            builder.getI64IntegerAttr(elements *
                                                      element_bytes))});
  Value raw_pointer = mlir::fly::MakePtrOp::create(
      builder, raw_pointer_type, mlir::ValueRange{}, allocation);
  auto pointer_type =
      mlir::fly::PointerType::get(element_type, shared_address, alignment);
  Value pointer = mlir::fly::RecastIterOp::create(builder, pointer_type,
                                                  raw_pointer);

  auto shape_attr =
      mlir::fly::IntTupleAttr::getLeafStatic(context, elements);
  auto stride_attr = mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
  auto shape_type = mlir::fly::IntTupleType::get(shape_attr);
  auto stride_type = mlir::fly::IntTupleType::get(stride_attr);
  Value shape = mlir::fly::MakeIntTupleOp::create(builder, shape_type,
                                                  mlir::ValueRange{});
  Value stride = mlir::fly::MakeIntTupleOp::create(builder, stride_type,
                                                   mlir::ValueRange{});
  auto layout_type = mlir::fly::LayoutType::get(shape_attr, stride_attr);
  Value layout = mlir::fly::MakeLayoutOp::create(builder, layout_type, shape,
                                                 stride);
  auto memref_type = mlir::fly::MemRefType::get(
      element_type, shared_address, layout_type.getAttr(), alignment);
  return mlir::fly::MakeViewOp::create(builder, memref_type, pointer, layout);
}

Value LoadShared(mlir::ImplicitLocOpBuilder& builder, Value memref,
                 Value index) {
  return mlir::fly::MemRefLoadOp::create(builder, memref,
                                         MakeFlyIndex(builder, index));
}

void StoreShared(mlir::ImplicitLocOpBuilder& builder, Value value,
                 Value memref, Value index) {
  mlir::fly::MemRefStoreOp::create(builder, value, memref,
                                   MakeFlyIndex(builder, index));
}

Value CreateSharedPointer(mlir::ImplicitLocOpBuilder& builder,
                          mlir::Type element_type, int64_t elements) {
  mlir::MLIRContext* context = builder.getContext();
  constexpr int64_t kAlignment = 16;
  const int64_t element_bytes = element_type.getIntOrFloatBitWidth() / 8;
  auto shared_address = mlir::fly::AddressSpaceAttr::get(
      context, mlir::fly::AddressSpace::Shared);
  auto alignment = mlir::fly::AlignAttr::get(context, kAlignment);
  auto raw_pointer_type = mlir::fly::PointerType::get(
      builder.getI8Type(), shared_address, alignment);
  auto allocation = builder.getDictionaryAttr(
      {builder.getNamedAttr("allocAlign",
                            builder.getI64IntegerAttr(kAlignment)),
       builder.getNamedAttr("allocBytes",
                            builder.getI64IntegerAttr(elements *
                                                      element_bytes))});
  Value raw_pointer = mlir::fly::MakePtrOp::create(
      builder, raw_pointer_type, mlir::ValueRange{}, allocation);
  auto pointer_type =
      mlir::fly::PointerType::get(element_type, shared_address, alignment);
  return mlir::fly::RecastIterOp::create(builder, pointer_type, raw_pointer);
}

Value AddSharedPointerOffset(mlir::ImplicitLocOpBuilder& builder, Value pointer,
                             Value element_offset) {
  if (element_offset.getType().isIndex()) {
    element_offset = mlir::arith::IndexCastOp::create(
        builder, builder.getI32Type(), element_offset);
  } else if (element_offset.getType().isInteger(64)) {
    element_offset = mlir::arith::TruncIOp::create(
        builder, builder.getI32Type(), element_offset);
  }
  auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
      builder.getContext(), /*width=*/32, /*divisibility=*/1);
  auto offset_type = mlir::fly::IntTupleType::get(offset_attr);
  Value offset = mlir::fly::MakeIntTupleOp::create(
      builder, offset_type, mlir::ValueRange{element_offset});
  return mlir::fly::AddOffsetOp::create(builder, pointer.getType(), pointer,
                                        offset);
}

Value LoadSharedPointer(mlir::ImplicitLocOpBuilder& builder, Value pointer,
                        Value element_offset, mlir::Type result_type) {
  return mlir::fly::PtrLoadOp::create(
      builder, result_type,
      AddSharedPointerOffset(builder, pointer, element_offset));
}

void StoreSharedPointer(mlir::ImplicitLocOpBuilder& builder, Value value,
                        Value pointer, Value element_offset) {
  mlir::fly::PtrStoreOp::create(
      builder, value,
      AddSharedPointerOffset(builder, pointer, element_offset));
}

void EmitLdsWait(mlir::ImplicitLocOpBuilder& builder) {
  // CDNA3: leave VMEM/EXP counters unconstrained and drain LGKM so a wave can
  // consume its private probability and V slots without a workgroup barrier.
  mlir::OperationState state(builder.getLoc(), "rocdl.s.waitcnt");
  state.addAttribute("bitfield", builder.getI32IntegerAttr(0xC07F));
  builder.create(state);
}

void EmitWaitAll(mlir::ImplicitLocOpBuilder& builder) {
  mlir::OperationState state(builder.getLoc(), "rocdl.s.waitcnt");
  state.addAttribute("bitfield", builder.getI32IntegerAttr(0));
  builder.create(state);
}

class FlyXTilePagedAttentionOnlineEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTilePagedAttentionOnlineEmitter(
      const HloFusionAnalysis& analysis)
      : producer_descriptor_(
            GetFlyPagedAttentionSegmentedProducerDescriptor(analysis)),
        descriptor_(producer_descriptor_.has_value()
                        ? producer_descriptor_->attention
                        : *GetFlyPagedAttentionDescriptor(analysis)),
        num_segments_(producer_descriptor_.has_value()
                          ? producer_descriptor_->num_segments
                          : 1),
        segment_tokens_(producer_descriptor_.has_value()
                            ? producer_descriptor_->segment_tokens
                            : descriptor_.max_context),
        num_waves_(producer_descriptor_.has_value() &&
                           ((descriptor_.max_context >= 65536 &&
                             descriptor_.max_context < 131072) ||
                            segment_tokens_ <= 64)
                       ? 2
                       : 4),
        tokens_per_wave_(producer_descriptor_.has_value() &&
                                 descriptor_.max_context >= 65536 &&
                                 descriptor_.max_context < 131072
                             ? 224
                             : (producer_descriptor_.has_value() &&
                                        descriptor_.max_context > 8192 &&
                                        descriptor_.max_context < 65536
                                    ? 128
                                    : 32)),
        token_halves_(tokens_per_wave_ / kMfmaN),
        tokens_per_tile_(num_waves_ * tokens_per_wave_),
        d_blocks_per_wave_(kHeadSize / (num_waves_ * kMfmaN)),
        launch_dimensions_(
            se::BlockDim(descriptor_.sequences, descriptor_.kv_heads,
                         num_segments_),
            se::ThreadDim(num_waves_ * kWaveSize, 1, 1)) {}

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
    context.getOrLoadDialect<mlir::fly::FlyDialect>();
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
        "FlyXTilePagedAttentionOnlineEmitter builds its module directly.");
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function) const {
    const int64_t parameter_count = descriptor_.call->parent()->num_parameters();
    const int64_t output_count = producer_descriptor_.has_value() ? 3 : 1;
    TF_RET_CHECK(entry_function.getNumArguments() ==
                 parameter_count + output_count);
    TF_RET_CHECK(descriptor_.query->opcode() == HloOpcode::kParameter);
    TF_RET_CHECK(descriptor_.key_cache->opcode() == HloOpcode::kParameter);
    TF_RET_CHECK(descriptor_.value_cache->opcode() == HloOpcode::kParameter);
    TF_RET_CHECK(descriptor_.block_table->opcode() == HloOpcode::kParameter);
    TF_RET_CHECK(descriptor_.used_k->opcode() == HloOpcode::kParameter ||
                 descriptor_.used_k->opcode() == HloOpcode::kConstant);
    TF_RET_CHECK(
        (descriptor_.element_type == BF16 || descriptor_.element_type == F16) &&
        descriptor_.query_heads ==
            descriptor_.gqa_group * descriptor_.kv_heads &&
        descriptor_.gqa_group > 0 && descriptor_.gqa_group <= kMfmaN &&
        descriptor_.head_dimension == kHeadSize &&
        descriptor_.max_context > tokens_per_tile_ &&
        descriptor_.max_context <= 262144 && num_segments_ >= 1 &&
        segment_tokens_ >= tokens_per_tile_);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());
    Value query = entry_function.getArgument(
        descriptor_.query->parameter_number());
    Value key_cache = entry_function.getArgument(
        descriptor_.key_cache->parameter_number());
    Value value_cache = entry_function.getArgument(
        descriptor_.value_cache->parameter_number());
    Value used_k;
    if (descriptor_.used_k->opcode() == HloOpcode::kParameter) {
      used_k = entry_function.getArgument(
          descriptor_.used_k->parameter_number());
    }
    Value block_table = entry_function.getArgument(
        descriptor_.block_table->parameter_number());
    Value output = entry_function.getArgument(parameter_count);
    Value partial_maximum;
    Value partial_sum;
    if (producer_descriptor_.has_value()) {
      partial_maximum = entry_function.getArgument(parameter_count + 1);
      partial_sum = entry_function.getArgument(parameter_count + 2);
    }

    mlir::Type element_type = descriptor_.element_type == BF16
                                  ? builder.getBF16Type()
                                  : builder.getF16Type();
    auto v4_element = mlir::VectorType::get({4}, element_type);
    auto v4_f32 = mlir::VectorType::get({4}, builder.getF32Type());

    const char* atom_type_name =
        descriptor_.element_type == BF16
            ? "!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, "
              "(bf16,bf16)->f32>>"
            : "!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, "
              "(f16,f16)->f32>>";
    mlir::Type atom_type =
        mlir::parseType(atom_type_name, entry_function.getContext());
    TF_RET_CHECK(atom_type != nullptr);
    mlir::OperationState atom_state(entry_function.getLoc(),
                                    "fly.make_mma_atom");
    atom_state.addTypes(atom_type);
    Value atom = builder.create(atom_state)->getResult(0);
    auto mma = [&](Value lhs, Value rhs, Value accumulator) {
      mlir::OperationState state(entry_function.getLoc(),
                                 "fly.mma_atom_call_ssa");
      state.addOperands({atom, lhs, rhs, accumulator});
      state.addTypes(v4_f32);
      return builder.create(state)->getResult(0);
    };

    auto vector_from = [&](llvm::ArrayRef<Value> values, mlir::Type type) {
      return mlir::vector::FromElementsOp::create(
                 builder, mlir::cast<mlir::VectorType>(type), values)
          .getResult();
    };
    Value zero_f32 = F32Constant(builder, 0.0f);
    Value minus_inf = NegativeInfinity(builder);
    Value zero_element =
        mlir::arith::TruncFOp::create(builder, element_type, zero_f32);
    Value zero4 = mlir::vector::BroadcastOp::create(builder, v4_f32, zero_f32);
    Value minus_inf4 =
        mlir::vector::BroadcastOp::create(builder, v4_f32, minus_inf);
    Value scale4 = mlir::vector::BroadcastOp::create(
        builder, v4_f32,
        F32Constant(builder,
                    static_cast<float>(descriptor_.scale) * kLog2E));

    auto exp2_vector = [&](Value exponent) {
      llvm::SmallVector<Value, kMfmaOutputsPerLane> values;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        values.push_back(mlir::ROCDL::ROCDLExp2::create(
            builder, builder.getF32Type(),
            ExtractVectorElement(builder, exponent, component)));
      }
      return vector_from(values, v4_f32);
    };
    auto safe_maximum = [&](Value maximum) {
      llvm::SmallVector<Value, kMfmaOutputsPerLane> values;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value candidate = ExtractVectorElement(builder, maximum, component);
        Value valid = mlir::arith::CmpFOp::create(
            builder, mlir::arith::CmpFPredicate::OGT, candidate, minus_inf);
        values.push_back(mlir::arith::SelectOp::create(
            builder, valid, candidate, zero_f32));
      }
      return vector_from(values, v4_f32);
    };
    auto positive_select = [&](Value condition_source, Value candidate) {
      llvm::SmallVector<Value, kMfmaOutputsPerLane> values;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value positive = mlir::arith::CmpFOp::create(
            builder, mlir::arith::CmpFPredicate::OGT,
            ExtractVectorElement(builder, condition_source, component),
            zero_f32);
        values.push_back(mlir::arith::SelectOp::create(
            builder, positive,
            ExtractVectorElement(builder, candidate, component), zero_f32));
      }
      return vector_from(values, v4_f32);
    };
    auto reduce_vector = [&](Value value, bool is_maximum) {
      for (int64_t distance : {8, 4, 2, 1}) {
        llvm::SmallVector<Value, kMfmaOutputsPerLane> reduced;
        for (int64_t component = 0; component < kMfmaOutputsPerLane;
             ++component) {
          Value current = ExtractVectorElement(builder, value, component);
          Value peer = mlir::gpu::ShuffleOp::create(builder, current, distance,
                                                    kWaveSize,
                                                    mlir::gpu::ShuffleMode::XOR)
                           .getShuffleResult();
          reduced.push_back(
              is_maximum
                  ? mlir::arith::MaxNumFOp::create(builder, current, peer)
                        .getResult()
                  : mlir::arith::AddFOp::create(builder, current, peer)
                        .getResult());
        }
        value = vector_from(reduced, v4_f32);
      }
      return value;
    };

    Value thread_id = EmitThreadId(builder, 0);
    Value wave = Div(builder, thread_id, kWaveSize);
    Value lane = Rem(builder, thread_id, kWaveSize);
    Value lane16 = Rem(builder, lane, kMfmaN);
    Value lane_group = Div(builder, lane, kMfmaN);
    Value sequence = EmitBlockId(builder, 0);
    Value kv_head = EmitBlockId(builder, 1);
    Value segment = EmitBlockId(builder, 2);
    Value segment_start =
        Mul(builder, segment, IndexConstant(builder, segment_tokens_));
    Value segment_end =
        Add(builder, segment_start, IndexConstant(builder, segment_tokens_));
    Value kv_len_i32;
    if (used_k) {
      kv_len_i32 = EmitBufferLoad(builder, entry_function.getLoc(), used_k,
                                  sequence, builder.getI32Type(),
                                  /*is_scalar=*/true);
    } else {
      TF_RET_CHECK(descriptor_.used_k->literal().shape().element_type() == S32);
      kv_len_i32 = mlir::arith::ConstantIntOp::create(
          builder, builder.getI32Type(),
          descriptor_.used_k->literal().Get<int32_t>({0}));
      for (int64_t i = 1; i < descriptor_.sequences; ++i) {
        Value is_sequence = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::eq, sequence,
            IndexConstant(builder, i));
        Value constant_length = mlir::arith::ConstantIntOp::create(
            builder, builder.getI32Type(),
            descriptor_.used_k->literal().Get<int32_t>({i}));
        kv_len_i32 = mlir::arith::SelectOp::create(
            builder, is_sequence, constant_length, kv_len_i32);
      }
    }
    Value kv_len = mlir::arith::IndexCastOp::create(
        builder, builder.getIndexType(), kv_len_i32);

    Value query_row = lane16;
    Value query_row_valid = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::ult, query_row,
        IndexConstant(builder, descriptor_.gqa_group));
    Value query_head = Add(
        builder,
        Mul(builder, kv_head, IndexConstant(builder, descriptor_.gqa_group)),
        query_row);
    std::vector<Value> query_fragments;
    query_fragments.reserve(kHeadSize / kMfmaK);
    for (int64_t k_step = 0; k_step < kHeadSize / kMfmaK; ++k_step) {
      Value q_offset = Add(
          builder,
          Mul(builder,
              Add(builder,
                  Mul(builder, sequence,
                      IndexConstant(builder, descriptor_.query_heads)),
                  query_head),
              IndexConstant(builder, kHeadSize)),
          Add(builder, IndexConstant(builder, k_step * kMfmaK),
              Mul(builder, lane_group,
                  IndexConstant(builder, kMfmaInputsPerLane))));
      Value safe_q_offset = mlir::arith::SelectOp::create(
          builder, query_row_valid, q_offset, IndexConstant(builder, 0));
      query_fragments.push_back(EmitBufferLoad(
          builder, entry_function.getLoc(), query, safe_q_offset, v4_element));
    }

    // Match FlyDSL's native MFMA16 row layout.  GQA4 only makes rows 0..3
    // visible, but retaining the full 16-row stride avoids compact-layout
    // address remapping in every LDS producer and consumer.
    const int64_t probability_elements =
        num_waves_ * kMfmaN * tokens_per_wave_;
    const int64_t scalar_elements = num_waves_ * kMfmaN;
    Value shared_probabilities =
        CreateSharedMemRef(builder, element_type, probability_elements);
    Value shared_l = CreateSharedMemRef(builder, builder.getF32Type(),
                                        scalar_elements);
    Value shared_m = CreateSharedMemRef(builder, builder.getF32Type(),
                                        scalar_elements);
    Value shared_rows = CreateSharedMemRef(builder, builder.getI32Type(),
                                           tokens_per_tile_);

    auto conditional_store = [&](Value memref, Value condition, Value value,
                                 Value index) {
      mlir::scf::IfOp store = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{}, condition,
          /*withElseRegion=*/false);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(store.thenBlock());
        StoreShared(builder, value, memref, index);
      }
      builder.setInsertionPointAfter(store);
    };
    auto sync_shared = [&]() { mlir::gpu::BarrierOp::create(builder); };
    auto cache_offset = [&](Value physical_row, Value dimension) {
      return Add(
          builder,
          Mul(builder,
              Add(builder,
                  Mul(builder, physical_row,
                      IndexConstant(builder, descriptor_.kv_heads)),
                  kv_head),
              IndexConstant(builder, kHeadSize)),
          dimension);
    };

    llvm::SmallVector<Value> initial_state(d_blocks_per_wave_, zero4);
    initial_state.append({minus_inf4, zero4});
    mlir::scf::ForOp tile_loop = mlir::scf::ForOp::create(
        builder, IndexConstant(builder, 0),
        IndexConstant(builder, segment_tokens_),
        IndexConstant(builder, tokens_per_tile_), initial_state,
        [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
    {
      mlir::OpBuilder::InsertionGuard loop_guard(builder);
      builder.setInsertionPointToStart(tile_loop.getBody());
      Value tile_base = tile_loop.getInductionVar();
      llvm::SmallVector<Value> previous_outputs;
      for (int64_t d_block = 0; d_block < d_blocks_per_wave_; ++d_block) {
        previous_outputs.push_back(tile_loop.getRegionIterArg(d_block));
      }
      Value previous_max = tile_loop.getRegionIterArg(d_blocks_per_wave_);
      Value previous_sum = tile_loop.getRegionIterArg(d_blocks_per_wave_ + 1);
      Value wave_token_start =
          Mul(builder, wave, IndexConstant(builder, tokens_per_wave_));
      Value query_row_base =
          Mul(builder, lane_group,
              IndexConstant(builder, kMfmaOutputsPerLane));
      llvm::SmallVector<Value> masked_scores(token_halves_);
      llvm::SmallVector<Value> token_validity(token_halves_);
      for (int64_t token_half = 0; token_half < token_halves_; ++token_half) {
        Value local_token = Add(
            builder, wave_token_start,
            Add(builder, IndexConstant(builder, token_half * kMfmaN), lane16));
        Value logical_token = Add(
            builder, segment_start, Add(builder, tile_base, local_token));
        Value token_used = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, logical_token, kv_len);
        Value token_in_segment = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, logical_token,
            segment_end);
        Value token_in_context = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, logical_token,
            IndexConstant(builder, descriptor_.max_context));
        Value token_valid = mlir::arith::AndIOp::create(
            builder, token_used,
            mlir::arith::AndIOp::create(builder, token_in_segment,
                                        token_in_context));
        token_validity[token_half] = token_valid;
        Value safe_logical_token = mlir::arith::SelectOp::create(
            builder, token_valid, logical_token, IndexConstant(builder, 0));
        Value logical_page = Div(builder, safe_logical_token,
                                 descriptor_.page_size);
        Value page_slot = Rem(builder, safe_logical_token,
                              descriptor_.page_size);
        Value table_offset = Add(
            builder,
            Mul(builder, sequence,
                IndexConstant(builder, descriptor_.pages_per_sequence)),
            logical_page);
        Value physical_page_i32 = EmitBufferLoad(
            builder, entry_function.getLoc(), block_table, table_offset,
            builder.getI32Type());
        Value physical_page = mlir::arith::IndexCastOp::create(
            builder, builder.getIndexType(), physical_page_i32);
        Value physical_row = Add(
            builder,
            Mul(builder, physical_page,
                IndexConstant(builder, descriptor_.page_size)),
            page_slot);
        Value owns_row = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::eq, lane_group,
            IndexConstant(builder, 0));
        conditional_store(
            shared_rows, owns_row,
            mlir::arith::IndexCastOp::create(builder, builder.getI32Type(),
                                             physical_row),
            local_token);

        Value score = zero4;
        for (int64_t k_step = 0; k_step < query_fragments.size(); ++k_step) {
          Value dimension = Add(
              builder, IndexConstant(builder, k_step * kMfmaK),
              Mul(builder, lane_group,
                  IndexConstant(builder, kMfmaInputsPerLane)));
          Value k_fragment = EmitBufferLoad(
              builder, entry_function.getLoc(), key_cache,
              cache_offset(physical_row, dimension), v4_element);
          score = mma(query_fragments[k_step], k_fragment, score);
        }
        score = mlir::arith::MulFOp::create(builder, score, scale4);
        llvm::SmallVector<Value, kMfmaOutputsPerLane> score_elements;
        for (int64_t component = 0; component < kMfmaOutputsPerLane;
             ++component) {
          Value row =
              Add(builder, query_row_base, IndexConstant(builder, component));
          Value row_valid = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, row,
              IndexConstant(builder, descriptor_.gqa_group));
          Value visible =
              mlir::arith::AndIOp::create(builder, token_valid, row_valid);
          score_elements.push_back(mlir::arith::SelectOp::create(
              builder, visible, ExtractVectorElement(builder, score, component),
              minus_inf));
        }
        masked_scores[token_half] = vector_from(score_elements, v4_f32);
      }

      Value local_max = masked_scores[0];
      for (int64_t token_half = 1; token_half < token_halves_; ++token_half) {
        local_max = mlir::arith::MaxNumFOp::create(
            builder, local_max, masked_scores[token_half]);
      }
      local_max = reduce_vector(local_max, /*is_maximum=*/true);
      Value safe_local_max = safe_maximum(local_max);
      llvm::SmallVector<Value> probabilities(token_halves_);
      Value local_sum = zero4;
      for (int64_t token_half = 0; token_half < token_halves_; ++token_half) {
        Value exponent = mlir::arith::SubFOp::create(
            builder, masked_scores[token_half], safe_local_max);
        Value candidates = exp2_vector(exponent);
        llvm::SmallVector<Value, kMfmaOutputsPerLane> values;
        for (int64_t component = 0; component < kMfmaOutputsPerLane;
             ++component) {
          Value row =
              Add(builder, query_row_base, IndexConstant(builder, component));
          Value row_valid = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, row,
              IndexConstant(builder, descriptor_.gqa_group));
          Value visible = mlir::arith::AndIOp::create(
              builder, token_validity[token_half], row_valid);
          values.push_back(mlir::arith::SelectOp::create(
              builder, visible,
              ExtractVectorElement(builder, candidates, component), zero_f32));
        }
        probabilities[token_half] = vector_from(values, v4_f32);
        local_sum = mlir::arith::AddFOp::create(builder, local_sum,
                                               probabilities[token_half]);
      }
      local_sum = reduce_vector(local_sum, /*is_maximum=*/false);

      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value row =
            Add(builder, query_row_base, IndexConstant(builder, component));
        Value row_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, row,
            IndexConstant(builder, descriptor_.gqa_group));
        Value lane_owns_scalar = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::eq, lane16,
            IndexConstant(builder, component));
        lane_owns_scalar = mlir::arith::AndIOp::create(
            builder, lane_owns_scalar, row_valid);
        Value scalar_slot = Add(
            builder,
            Mul(builder, wave, IndexConstant(builder, kMfmaN)),
            row);
        conditional_store(shared_m, lane_owns_scalar,
                          ExtractVectorElement(builder, local_max, component),
                          scalar_slot);
        conditional_store(shared_l, lane_owns_scalar,
                          ExtractVectorElement(builder, local_sum, component),
                          scalar_slot);
        for (int64_t token_half = 0; token_half < token_halves_; ++token_half) {
          Value probability_slot = Add(
                builder,
                Mul(builder, scalar_slot,
                    IndexConstant(builder, tokens_per_wave_)),
              Add(builder, IndexConstant(builder, token_half * kMfmaN),
                  lane16));
          Value probability = mlir::arith::TruncFOp::create(
              builder, element_type,
              ExtractVectorElement(builder, probabilities[token_half],
                                   component));
          conditional_store(shared_probabilities, row_valid, probability,
                            probability_slot);
        }
      }
      sync_shared();

      llvm::SmallVector<Value> partial_maxima(num_waves_);
      llvm::SmallVector<Value> partial_sums(num_waves_);
      for (int64_t source_wave = 0; source_wave < num_waves_; ++source_wave) {
        llvm::SmallVector<Value, kMfmaOutputsPerLane> maxima;
        llvm::SmallVector<Value, kMfmaOutputsPerLane> sums;
        for (int64_t component = 0; component < kMfmaOutputsPerLane;
             ++component) {
          Value row =
              Add(builder, query_row_base, IndexConstant(builder, component));
          Value row_valid = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, row,
              IndexConstant(builder, descriptor_.gqa_group));
          Value safe_row = mlir::arith::SelectOp::create(
              builder, row_valid, row, IndexConstant(builder, 0));
          Value scalar_slot = Add(
              builder,
              IndexConstant(builder, source_wave * kMfmaN),
              safe_row);
          Value maximum = LoadShared(builder, shared_m, scalar_slot);
          Value sum = LoadShared(builder, shared_l, scalar_slot);
          maxima.push_back(mlir::arith::SelectOp::create(
              builder, row_valid, maximum, minus_inf));
          sums.push_back(mlir::arith::SelectOp::create(
              builder, row_valid, sum, zero_f32));
        }
        partial_maxima[source_wave] = vector_from(maxima, v4_f32);
        partial_sums[source_wave] = vector_from(sums, v4_f32);
      }

      Value tile_max = partial_maxima[0];
      for (int64_t source_wave = 1; source_wave < num_waves_; ++source_wave) {
        tile_max = mlir::arith::MaxNumFOp::create(
            builder, tile_max, partial_maxima[source_wave]);
      }
      Value safe_tile_max = safe_maximum(tile_max);
      llvm::SmallVector<Value> partition_weights(num_waves_);
      Value tile_sum = zero4;
      for (int64_t source_wave = 0; source_wave < num_waves_; ++source_wave) {
        Value exponent = mlir::arith::SubFOp::create(
            builder, partial_maxima[source_wave], safe_tile_max);
        partition_weights[source_wave] = positive_select(
            partial_sums[source_wave], exp2_vector(exponent));
        tile_sum = mlir::arith::AddFOp::create(
            builder, tile_sum,
            mlir::arith::MulFOp::create(builder, partial_sums[source_wave],
                                        partition_weights[source_wave]));
      }

      llvm::SmallVector<Value> tile_outputs(d_blocks_per_wave_, zero4);
      for (int64_t source_wave = 0; source_wave < num_waves_; ++source_wave) {
        llvm::SmallVector<Value> source_outputs(d_blocks_per_wave_, zero4);
        for (int64_t token_half = 0; token_half < token_halves_; ++token_half) {
          Value probability_row_valid = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, lane16,
              IndexConstant(builder, descriptor_.gqa_group));
          Value safe_probability_row = mlir::arith::SelectOp::create(
              builder, probability_row_valid, lane16,
              IndexConstant(builder, 0));
          llvm::SmallVector<Value, kMfmaInputsPerLane> probability_elements;
          for (int64_t k_lane = 0; k_lane < kMfmaInputsPerLane; ++k_lane) {
            Value token_in_partition = Add(
                builder, IndexConstant(builder, token_half * kMfmaN + k_lane),
                Mul(builder, lane_group,
                    IndexConstant(builder, kMfmaInputsPerLane)));
            Value probability_slot = Add(
                builder,
                Mul(builder,
                    Add(builder,
                        IndexConstant(
                            builder,
                            source_wave * kMfmaN),
                        safe_probability_row),
                    IndexConstant(builder, tokens_per_wave_)),
                token_in_partition);
            Value probability =
                LoadShared(builder, shared_probabilities, probability_slot);
            probability_elements.push_back(mlir::arith::SelectOp::create(
                builder, probability_row_valid, probability, zero_element));
          }
          Value probability_fragment =
              vector_from(probability_elements, v4_element);
          for (int64_t local_d_block = 0;
               local_d_block < d_blocks_per_wave_; ++local_d_block) {
            Value d_block = Add(
                builder,
                Mul(builder, wave,
                    IndexConstant(builder, d_blocks_per_wave_)),
                IndexConstant(builder, local_d_block));
            Value dimension = Add(
                builder, Mul(builder, d_block, IndexConstant(builder, kMfmaN)),
                lane16);
            llvm::SmallVector<Value, kMfmaInputsPerLane> value_elements;
            for (int64_t k_lane = 0; k_lane < kMfmaInputsPerLane; ++k_lane) {
              Value local_token = Add(
                  builder,
                  IndexConstant(builder, source_wave * tokens_per_wave_ +
                                             token_half * kMfmaN + k_lane),
                  Mul(builder, lane_group,
                      IndexConstant(builder, kMfmaInputsPerLane)));
              Value physical_row_i32 =
                  LoadShared(builder, shared_rows, local_token);
              Value physical_row = mlir::arith::IndexCastOp::create(
                  builder, builder.getIndexType(), physical_row_i32);
              value_elements.push_back(EmitBufferLoad(
                  builder, entry_function.getLoc(), value_cache,
                  cache_offset(physical_row, dimension), element_type));
            }
            Value value_fragment = vector_from(value_elements, v4_element);
            source_outputs[local_d_block] = mma(
                probability_fragment, value_fragment,
                source_outputs[local_d_block]);
          }
        }
        for (int64_t local_d_block = 0; local_d_block < d_blocks_per_wave_;
             ++local_d_block) {
          tile_outputs[local_d_block] = mlir::arith::AddFOp::create(
              builder, tile_outputs[local_d_block],
              mlir::arith::MulFOp::create(
                  builder, source_outputs[local_d_block],
                  partition_weights[source_wave]));
        }
      }

      Value new_max;
      Value new_sum;
      llvm::SmallVector<Value> new_outputs;
      if (segment_tokens_ == tokens_per_tile_) {
        // The first and only tile is already in the segment's softmax basis.
        // Avoid expressing the generic online merge with an empty prior state:
        // LLVM does not fold all four vector exp2 calls through the max/select
        // sequence, even after the one-trip loop has been promoted.
        new_max = tile_max;
        new_sum = tile_sum;
        new_outputs.assign(tile_outputs.begin(), tile_outputs.end());
      } else {
        new_max =
            mlir::arith::MaxNumFOp::create(builder, previous_max, tile_max);
        Value safe_new_max = safe_maximum(new_max);
        Value previous_exponent = mlir::arith::SubFOp::create(
            builder, previous_max, safe_new_max);
        Value tile_exponent = mlir::arith::SubFOp::create(
            builder, tile_max, safe_new_max);
        Value previous_weight =
            positive_select(previous_sum, exp2_vector(previous_exponent));
        Value tile_weight =
            positive_select(tile_sum, exp2_vector(tile_exponent));
        new_sum = mlir::arith::AddFOp::create(
            builder,
            mlir::arith::MulFOp::create(builder, previous_sum,
                                        previous_weight),
            mlir::arith::MulFOp::create(builder, tile_sum, tile_weight));
        new_outputs.resize(d_blocks_per_wave_);
        for (int64_t local_d_block = 0;
             local_d_block < d_blocks_per_wave_; ++local_d_block) {
          new_outputs[local_d_block] = mlir::arith::AddFOp::create(
              builder,
              mlir::arith::MulFOp::create(builder,
                                          previous_outputs[local_d_block],
                                          previous_weight),
              mlir::arith::MulFOp::create(builder,
                                          tile_outputs[local_d_block],
                                          tile_weight));
        }
      }

      // The wide D128 dispatcher deliberately chooses one producer tile per
      // segment through KV8192. There is no following tile that can overwrite
      // LDS in that case, so a second workgroup barrier only stalls the kernel.
      // Keep it for the compact online path, where the next loop iteration
      // reuses the probability and physical-row allocations.
      if (segment_tokens_ > tokens_per_tile_) {
        sync_shared();
      }
      llvm::SmallVector<Value> yielded(new_outputs.begin(), new_outputs.end());
      yielded.append({new_max, new_sum});
      mlir::scf::YieldOp::create(builder, yielded);
    }
    builder.setInsertionPointAfter(tile_loop);

    llvm::SmallVector<Value> final_outputs;
    for (int64_t d_block = 0; d_block < d_blocks_per_wave_; ++d_block) {
      final_outputs.push_back(tile_loop.getResult(d_block));
    }
    Value final_sum = tile_loop.getResult(d_blocks_per_wave_ + 1);
    std::optional<Value> inverse_sum;
    if (!producer_descriptor_.has_value()) {
      Value one_f32 = F32Constant(builder, 1.0f);
      llvm::SmallVector<Value, kMfmaOutputsPerLane> inverse_elements;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value sum = ExtractVectorElement(builder, final_sum, component);
        Value positive = mlir::arith::CmpFOp::create(
            builder, mlir::arith::CmpFPredicate::OGT, sum, zero_f32);
        Value safe_sum =
            mlir::arith::SelectOp::create(builder, positive, sum, one_f32);
        inverse_elements.push_back(mlir::ROCDL::ROCDLRcp::create(
            builder, builder.getF32Type(), safe_sum));
      }
      inverse_sum = vector_from(inverse_elements, v4_f32);
    }
    Value query_row_base =
        Mul(builder, lane_group,
            IndexConstant(builder, kMfmaOutputsPerLane));
    if (producer_descriptor_.has_value()) {
      Value owns_softmax_state = mlir::arith::AndIOp::create(
          builder,
          mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::eq, wave,
              IndexConstant(builder, 0)),
          mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::eq, lane16,
              IndexConstant(builder, 0)));
      Value final_max = tile_loop.getResult(d_blocks_per_wave_);
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value row =
            Add(builder, query_row_base, IndexConstant(builder, component));
        Value row_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, row,
            IndexConstant(builder, descriptor_.gqa_group));
        Value store_state = mlir::arith::AndIOp::create(
            builder, owns_softmax_state, row_valid);
        Value output_head = Add(
            builder,
            Mul(builder, kv_head,
                IndexConstant(builder, descriptor_.gqa_group)),
            row);
        Value partial_state_index = Add(
            builder,
            Mul(builder,
                Add(builder,
                    Mul(builder, sequence,
                        IndexConstant(builder, descriptor_.query_heads)),
                    output_head),
                IndexConstant(builder, num_segments_)),
            segment);
        // FlyDSL guards the M/L pair once. Keep both buffer tokens in the same
        // region so LLVM sees one uniform tail branch rather than two branches
        // with identical predicates for every query row.
        mlir::scf::IfOp store = mlir::scf::IfOp::create(
            builder,
            mlir::TypeRange{partial_maximum.getType(), partial_sum.getType()},
            store_state, /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard store_guard(builder);
          builder.setInsertionPointToStart(store.thenBlock());
          Value updated_maximum = EmitBufferStore(
              builder, entry_function.getLoc(),
              ExtractVectorElement(builder, final_max, component),
              partial_maximum, partial_state_index);
          Value updated_sum = EmitBufferStore(
              builder, entry_function.getLoc(),
              ExtractVectorElement(builder, final_sum, component), partial_sum,
              partial_state_index);
          mlir::scf::YieldOp::create(
              builder, mlir::ValueRange{updated_maximum, updated_sum});
          builder.setInsertionPointToStart(store.elseBlock());
          mlir::scf::YieldOp::create(
              builder, mlir::ValueRange{partial_maximum, partial_sum});
        }
        builder.setInsertionPointAfter(store);
        partial_maximum = store.getResult(0);
        partial_sum = store.getResult(1);
      }
    }
    for (int64_t local_d_block = 0; local_d_block < d_blocks_per_wave_;
         ++local_d_block) {
      Value d_block = Add(
          builder,
          Mul(builder, wave, IndexConstant(builder, d_blocks_per_wave_)),
          IndexConstant(builder, local_d_block));
      Value dimension = Add(
          builder, Mul(builder, d_block, IndexConstant(builder, kMfmaN)),
          lane16);
      Value result = producer_descriptor_.has_value()
                         ? final_outputs[local_d_block]
                         : mlir::arith::MulFOp::create(
                               builder, final_outputs[local_d_block],
                               *inverse_sum)
                               .getResult();
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value row =
            Add(builder, query_row_base, IndexConstant(builder, component));
        Value row_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, row,
            IndexConstant(builder, descriptor_.gqa_group));
        Value output_head = Add(
            builder,
            Mul(builder, kv_head,
                IndexConstant(builder, descriptor_.gqa_group)),
            row);
        Value output_offset;
        Value output_value;
        if (producer_descriptor_.has_value()) {
          output_offset = Add(
              builder,
              Mul(builder,
                  Add(builder,
                      Mul(builder,
                          Add(builder,
                              Mul(builder, sequence,
                                  IndexConstant(builder,
                                                descriptor_.query_heads)),
                              output_head),
                          IndexConstant(builder, num_segments_)),
                      segment),
                  IndexConstant(builder, kHeadSize)),
              dimension);
          output_value = ExtractVectorElement(builder, result, component);
        } else {
          output_offset = Add(
              builder,
              Mul(builder,
                  Add(builder,
                      Mul(builder, sequence,
                          IndexConstant(builder, descriptor_.query_heads)),
                      output_head),
                  IndexConstant(builder, kHeadSize)),
              dimension);
          output_value = mlir::arith::TruncFOp::create(
              builder, element_type,
              ExtractVectorElement(builder, result, component));
        }
        mlir::scf::IfOp store = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{output.getType()}, row_valid,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard store_guard(builder);
          builder.setInsertionPointToStart(store.thenBlock());
          Value updated = EmitBufferStore(builder, entry_function.getLoc(),
                                          output_value, output, output_offset);
          mlir::scf::YieldOp::create(builder, updated);
          builder.setInsertionPointToStart(store.elseBlock());
          mlir::scf::YieldOp::create(builder, output);
        }
        builder.setInsertionPointAfter(store);
        output = store.getResult(0);
      }
    }

    if (producer_descriptor_.has_value()) {
      mlir::func::ReturnOp::create(
          builder, mlir::ValueRange{output, partial_maximum, partial_sum});
    } else {
      mlir::func::ReturnOp::create(builder, output);
    }

    // A one-tile segment is FlyDSL's straight-line wide decode algorithm, not
    // the reusable online algorithm. Promote the statically one-trip loop so
    // LLVM sees the initial (-inf, 0, 0) state directly. This folds away the
    // online rescaling exponentials, loop-carried state, and control flow and
    // materially reduces the live VGPR set on gfx942.
    if (segment_tokens_ == tokens_per_tile_ &&
        mlir::failed(mlir::loopUnrollFull(tile_loop))) {
      return absl::InternalError(
          "Failed to promote one-tile Fly paged-attention loop.");
    }
    return absl::OkStatus();
  }

  std::optional<FlyPagedAttentionSegmentedProducerDescriptor>
      producer_descriptor_;
  FlyPagedAttentionDescriptor descriptor_;
  int64_t num_segments_;
  int64_t segment_tokens_;
  int64_t num_waves_;
  int64_t tokens_per_wave_;
  int64_t token_halves_;
  int64_t tokens_per_tile_;
  int64_t d_blocks_per_wave_;
  LaunchDimensions launch_dimensions_;
};

// Source-faithful long-context producer from
// unified_attention_decode_cooperative.py.  Both waves compute the same
// 16-token QK tile, while each wave owns half of D128 for V and PV. V is
// staged wave-privately. The source-faithful configuration stages K
// cooperatively and prefetches the next K tile underneath the current PV
// MFMAs. This is deliberately separate from the wide producer above: the
// latter partitions tokens across waves, whereas this kernel partitions the
// output dimension across waves.
class FlyXTilePagedAttentionCooperativeEmitter final
    : public MlirKernelEmitter {
 public:
  explicit FlyXTilePagedAttentionCooperativeEmitter(
      const HloFusionAnalysis& analysis)
      : producer_descriptor_(
            *GetFlyPagedAttentionSegmentedProducerDescriptor(analysis)),
        descriptor_(producer_descriptor_.attention),
        unroll_factor_(analysis.fusion_backend_config()
                           .block_level_fusion_config()
                           .num_stages()),
        launch_dimensions_(
            se::BlockDim(descriptor_.sequences, descriptor_.kv_heads,
                         producer_descriptor_.num_segments),
            se::ThreadDim(kNumWaves * kWaveSize, 1, 1)) {}

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
  static constexpr int64_t kNumWaves = 2;
  static constexpr int64_t kBlockThreads = kNumWaves * kWaveSize;
  static constexpr int64_t kTileTokens = kMfmaN;
  static constexpr int64_t kStageWidth = 8;
  static constexpr int64_t kWaveHeadSize = kHeadSize / kNumWaves;
  static constexpr int64_t kDBlocksPerWave =
      kWaveHeadSize / kMfmaN;
  static constexpr int64_t kStageIterations =
      (kTileTokens * kHeadSize) / (kBlockThreads * kStageWidth);
  static constexpr int64_t kProbabilityStride = kWaveSize * kStageWidth;
  static constexpr int64_t kProbabilityBase =
      (kStageIterations - 1) * kBlockThreads * kStageWidth +
      kProbabilityStride - 4 * kTileTokens;
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateMLIRModule(
      mlir::MLIRContext& context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const override {
    context.getOrLoadDialect<mlir::fly::FlyDialect>();
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
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
        "Fly cooperative paged attention builds its module directly.");
  }

  absl::Status EmitFusedReducer(
      mlir::ImplicitLocOpBuilder& builder,
      mlir::func::FuncOp entry_function, Value thread_id, Value sequence,
      Value kv_head, Value output, Value partial_output,
      Value partial_maximum, Value partial_sum,
      Value completion_tickets) const {
    constexpr int64_t kReducerWidth = 32;
    constexpr int64_t kOutputElementsPerThread =
        kHeadSize / kReducerWidth;
    constexpr int64_t kMaxPartsPerLane = 4;
    TF_RET_CHECK(producer_descriptor_.fused_reducer &&
                 producer_descriptor_.num_segments <=
                     kReducerWidth * kMaxPartsPerLane);

    mlir::Type element_type = descriptor_.element_type == BF16
                                  ? builder.getBF16Type()
                                  : builder.getF16Type();
    auto f32_output_type = mlir::VectorType::get(
        {kOutputElementsPerThread}, builder.getF32Type());
    auto stored_output_type = mlir::VectorType::get(
        {kOutputElementsPerThread}, element_type);
    Value zero = F32Constant(builder, 0.0f);
    Value one = F32Constant(builder, 1.0f);
    Value minus_inf = NegativeInfinity(builder);
    // Every producer lane publishes its partial output before thread zero
    // releases the completion ticket. The last acq_rel atomic observes the
    // complete release sequence; an LDS broadcast then lets all four 32-lane
    // reducers consume the published partials.
    EmitWaitAll(builder);
    mlir::LLVM::FenceOp::create(builder,
                                mlir::LLVM::AtomicOrdering::release,
                                "agent-one-as");
    mlir::gpu::BarrierOp::create(builder);
    Value shared_previous = CreateSharedPointer(
        builder, builder.getI32Type(), kBlockThreads);
    Value thread_zero = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::eq, thread_id,
        IndexConstant(builder, 0));
    mlir::scf::IfOp increment = mlir::scf::IfOp::create(
        builder, mlir::TypeRange{builder.getI32Type()}, thread_zero,
        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(increment.thenBlock());
      Value ticket_base = Mul(
          builder,
          Add(builder,
              Mul(builder, sequence,
                  IndexConstant(builder, descriptor_.kv_heads)),
              kv_head),
          IndexConstant(builder, 2));
      Value previous_count = EmitBufferCompletionTicket(
          builder, entry_function.getLoc(), completion_tickets, ticket_base);
      mlir::scf::YieldOp::create(builder, previous_count);
      builder.setInsertionPointToStart(increment.elseBlock());
      Value not_last = mlir::arith::ConstantIntOp::create(
          builder, builder.getI32Type(), -1);
      mlir::scf::YieldOp::create(builder, mlir::ValueRange{not_last});
    }
    builder.setInsertionPointAfter(increment);
    StoreSharedPointer(builder, increment.getResult(0), shared_previous,
                       thread_id);
    mlir::gpu::BarrierOp::create(builder);
    mlir::LLVM::FenceOp::create(builder,
                                mlir::LLVM::AtomicOrdering::acquire,
                                "agent-one-as");
    Value previous = LoadSharedPointer(builder, shared_previous,
                                       IndexConstant(builder, 0),
                                       builder.getI32Type());
    Value is_last = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::eq, previous,
        mlir::arith::ConstantIntOp::create(
            builder, builder.getI32Type(),
            producer_descriptor_.num_segments - 1));

    mlir::scf::IfOp reduce = mlir::scf::IfOp::create(
        builder, mlir::TypeRange{output.getType()}, is_last,
        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(reduce.thenBlock());
      Value subgroup = Div(builder, thread_id, kReducerWidth);
      Value lane = Rem(builder, thread_id, kReducerWidth);
      Value query_head = Add(
          builder,
          Mul(builder, kv_head,
              IndexConstant(builder, descriptor_.gqa_group)),
          subgroup);
      auto subgroup_reduce = [&](Value value, bool maximum) {
        for (int64_t distance : {16, 8, 4, 2, 1}) {
          Value peer = mlir::gpu::ShuffleOp::create(
                           builder, value, distance, kReducerWidth,
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

      const int64_t parts_per_lane =
          (producer_descriptor_.num_segments + kReducerWidth - 1) /
          kReducerWidth;
      struct PartState {
        Value valid;
        Value sum;
        Value maximum;
      };
      auto load_part_state = [&](int64_t lane_part, int64_t cache_policy) {
        Value part = Add(
            builder, lane,
            IndexConstant(builder, lane_part * kReducerWidth));
        Value valid_index = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, part,
            IndexConstant(builder, producer_descriptor_.num_segments));
        Value safe_part = mlir::arith::SelectOp::create(
            builder, valid_index, part, IndexConstant(builder, 0));
        Value partial_state_index = Add(
            builder,
            Mul(builder,
                Add(builder,
                    Mul(builder, sequence,
                        IndexConstant(builder, descriptor_.query_heads)),
                    query_head),
                IndexConstant(builder, producer_descriptor_.num_segments)),
            safe_part);
        Value sum = EmitBufferLoad(builder, entry_function.getLoc(),
                                   partial_sum, partial_state_index,
                                   builder.getF32Type(), /*is_scalar=*/false,
                                   cache_policy);
        Value maximum = EmitBufferLoad(
            builder, entry_function.getLoc(), partial_maximum,
            partial_state_index, builder.getF32Type(), /*is_scalar=*/false,
            cache_policy);
        Value has_mass = mlir::arith::CmpFOp::create(
            builder, mlir::arith::CmpFPredicate::OGT, sum, zero);
        Value valid =
            mlir::arith::AndIOp::create(builder, valid_index, has_mass);
        return PartState{
            valid,
            mlir::arith::SelectOp::create(builder, valid, sum, zero),
            mlir::arith::SelectOp::create(builder, valid, maximum,
                                          minus_inf)};
      };

      std::vector<Value> part_valid;
      std::vector<Value> part_sum;
      std::vector<Value> part_maximum;
      part_valid.reserve(parts_per_lane);
      part_sum.reserve(parts_per_lane);
      part_maximum.reserve(parts_per_lane);
      Value local_max = minus_inf;
      for (int64_t lane_part = 0; lane_part < parts_per_lane; ++lane_part) {
        PartState state = load_part_state(lane_part, /*cache_policy=*/0);
        part_valid.push_back(state.valid);
        part_sum.push_back(state.sum);
        part_maximum.push_back(state.maximum);
        local_max = mlir::arith::MaxNumFOp::create(
            builder, local_max, state.maximum);
      }
      Value global_max = subgroup_reduce(local_max, /*maximum=*/true);

      std::vector<Value> part_weight;
      part_weight.reserve(parts_per_lane);
      Value local_sum = zero;
      for (int64_t lane_part = 0; lane_part < parts_per_lane; ++lane_part) {
        Value exponent = mlir::arith::SubFOp::create(
            builder, part_maximum[lane_part], global_max);
        Value candidate = mlir::ROCDL::ROCDLExp2::create(
            builder, builder.getF32Type(), exponent);
        Value scale = mlir::arith::SelectOp::create(
            builder, part_valid[lane_part], candidate, zero);
        part_weight.push_back(scale);
        local_sum = mlir::arith::AddFOp::create(
            builder, local_sum,
            mlir::arith::MulFOp::create(builder, part_sum[lane_part], scale));
      }
      Value global_sum = subgroup_reduce(local_sum, /*maximum=*/false);
      Value positive_sum = mlir::arith::CmpFOp::create(
          builder, mlir::arith::CmpFPredicate::OGT, global_sum, zero);
      Value safe_sum = mlir::arith::SelectOp::create(
          builder, positive_sum, global_sum, one);
      Value inverse_sum = mlir::ROCDL::ROCDLRcp::create(
          builder, builder.getF32Type(), safe_sum);
      for (Value& weight : part_weight) {
        weight = mlir::arith::MulFOp::create(builder, weight, inverse_sum);
      }

      Value output_column = Mul(
          builder, lane,
          IndexConstant(builder, kOutputElementsPerThread));
      Value result = mlir::vector::BroadcastOp::create(
          builder, f32_output_type, zero);
      Value subgroup_in_wave = Rem(builder, subgroup, 2);
      for (int64_t part = 0; part < producer_descriptor_.num_segments;
           ++part) {
        Value source_lane = Add(
            builder,
            Mul(builder, subgroup_in_wave,
                IndexConstant(builder, kReducerWidth)),
            IndexConstant(builder, part % kReducerWidth));
        Value source_lane_byte = mlir::arith::IndexCastOp::create(
            builder, builder.getI32Type(),
            Mul(builder, source_lane, IndexConstant(builder, 4)));
        Value weight_bits = mlir::arith::BitcastOp::create(
            builder, builder.getI32Type(),
            part_weight[part / kReducerWidth]);
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
                    IndexConstant(builder,
                                  producer_descriptor_.num_segments)),
                IndexConstant(builder, part)),
            IndexConstant(builder, kHeadSize));
        Value partial = EmitBufferLoad(
            builder, entry_function.getLoc(), partial_output,
            Add(builder, partial_base, output_column), f32_output_type);
        Value weight_vector = mlir::vector::BroadcastOp::create(
            builder, f32_output_type, weight);
        result = mlir::arith::AddFOp::create(
            builder, result,
            mlir::arith::MulFOp::create(builder, partial, weight_vector));
      }
      Value output_offset = Add(
          builder,
          Mul(builder,
              Add(builder,
                  Mul(builder, sequence,
                      IndexConstant(builder, descriptor_.query_heads)),
                  query_head),
              IndexConstant(builder, kHeadSize)),
          output_column);
      Value output_value = mlir::arith::TruncFOp::create(
          builder, stored_output_type, result);
      Value updated_output = EmitBufferStore(
          builder, entry_function.getLoc(), output_value, output,
          output_offset);
      mlir::scf::YieldOp::create(builder, updated_output);
      builder.setInsertionPointToStart(reduce.elseBlock());
      mlir::scf::YieldOp::create(builder, output);
    }
    builder.setInsertionPointAfter(reduce);
    mlir::func::ReturnOp::create(
        builder,
        mlir::ValueRange{reduce.getResult(0), partial_output, partial_maximum,
                         partial_sum, completion_tickets});
    return absl::OkStatus();
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function) const {
    const int64_t parameter_count = descriptor_.call->parent()->num_parameters();
    TF_RET_CHECK(entry_function.getNumArguments() ==
                 parameter_count +
                     (producer_descriptor_.fused_reducer ? 5 : 3));
    TF_RET_CHECK(descriptor_.query->opcode() == HloOpcode::kParameter);
    TF_RET_CHECK(descriptor_.key_cache->opcode() == HloOpcode::kParameter);
    TF_RET_CHECK(descriptor_.value_cache->opcode() == HloOpcode::kParameter);
    TF_RET_CHECK(descriptor_.block_table->opcode() == HloOpcode::kParameter);
    TF_RET_CHECK(descriptor_.used_k->opcode() == HloOpcode::kParameter ||
                 descriptor_.used_k->opcode() == HloOpcode::kConstant);
    TF_RET_CHECK(
        (descriptor_.element_type == BF16 || descriptor_.element_type == F16) &&
        descriptor_.query_heads == 4 * descriptor_.kv_heads &&
        descriptor_.gqa_group == 4 &&
        descriptor_.head_dimension == kHeadSize &&
        descriptor_.max_context >= 65536 &&
        descriptor_.max_context <= 262144 &&
        producer_descriptor_.num_segments >= 1 &&
        producer_descriptor_.segment_tokens >= kTileTokens &&
        producer_descriptor_.segment_tokens % kTileTokens == 0 &&
        unroll_factor_ >= 1 && unroll_factor_ <= 3);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());
    Value query = entry_function.getArgument(
        descriptor_.query->parameter_number());
    Value key_cache = entry_function.getArgument(
        descriptor_.key_cache->parameter_number());
    Value value_cache = entry_function.getArgument(
        descriptor_.value_cache->parameter_number());
    Value used_k;
    if (descriptor_.used_k->opcode() == HloOpcode::kParameter) {
      used_k = entry_function.getArgument(
          descriptor_.used_k->parameter_number());
    }
    Value block_table = entry_function.getArgument(
        descriptor_.block_table->parameter_number());
    Value output;
    const int64_t partial_output_index =
        parameter_count + (producer_descriptor_.fused_reducer ? 1 : 0);
    if (producer_descriptor_.fused_reducer) {
      output = entry_function.getArgument(parameter_count);
    }
    Value partial_output =
        entry_function.getArgument(partial_output_index + 0);
    Value partial_maximum =
        entry_function.getArgument(partial_output_index + 1);
    Value partial_sum = entry_function.getArgument(partial_output_index + 2);
    Value completion_tickets;
    if (producer_descriptor_.fused_reducer) {
      completion_tickets =
          entry_function.getArgument(partial_output_index + 3);
    }

    mlir::Type element_type = descriptor_.element_type == BF16
                                  ? builder.getBF16Type()
                                  : builder.getF16Type();
    auto v4_element = mlir::VectorType::get({4}, element_type);
    auto v8_element = mlir::VectorType::get({8}, element_type);
    auto v2_element = mlir::VectorType::get({2}, element_type);
    auto v1_i32 = mlir::VectorType::get({1}, builder.getI32Type());
    auto v4_i32 = mlir::VectorType::get({4}, builder.getI32Type());
    auto v4_f32 = mlir::VectorType::get({4}, builder.getF32Type());
    const char* atom_type_name =
        descriptor_.element_type == BF16
            ? "!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, "
              "(bf16,bf16)->f32>>"
            : "!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, "
              "(f16,f16)->f32>>";
    mlir::Type atom_type =
        mlir::parseType(atom_type_name, entry_function.getContext());
    TF_RET_CHECK(atom_type != nullptr);
    mlir::OperationState atom_state(entry_function.getLoc(),
                                    "fly.make_mma_atom");
    atom_state.addTypes(atom_type);
    Value atom = builder.create(atom_state)->getResult(0);
    auto mma = [&](Value lhs, Value rhs, Value accumulator) {
      mlir::OperationState state(entry_function.getLoc(),
                                 "fly.mma_atom_call_ssa");
      state.addOperands({atom, lhs, rhs, accumulator});
      state.addTypes(v4_f32);
      return builder.create(state)->getResult(0);
    };
    auto vector_from = [&](llvm::ArrayRef<Value> values, mlir::Type type) {
      return mlir::vector::FromElementsOp::create(
                 builder, mlir::cast<mlir::VectorType>(type), values)
          .getResult();
    };
    auto permute_half_pair = [&](Value high, Value low,
                                 int32_t selector) -> Value {
      Value selector_value = mlir::arith::ConstantIntOp::create(
          builder, builder.getI32Type(), selector);
      return mlir::LLVM::InlineAsmOp::create(
                 builder, builder.getI32Type(),
                 mlir::ValueRange{high, low, selector_value},
                 "v_perm_b32 $0, $1, $2, $3;\n", "=v,v,v,s",
                 /*has_side_effects=*/false,
                 /*is_align_stack=*/false,
                 mlir::LLVM::TailCallKind::None,
                 mlir::LLVM::AsmDialectAttr::get(
                     builder.getContext(), mlir::LLVM::AsmDialect::AD_ATT),
                 /*operand_attrs=*/mlir::ArrayAttr())
          .getResult(0);
    };
    Value zero_f32 = F32Constant(builder, 0.0f);
    Value minus_inf = NegativeInfinity(builder);
    Value zero_element =
        mlir::arith::TruncFOp::create(builder, element_type, zero_f32);
    Value zero8 =
        mlir::vector::BroadcastOp::create(builder, v8_element, zero_element);
    Value zero4 = mlir::vector::BroadcastOp::create(builder, v4_f32, zero_f32);
    Value scale4 = mlir::vector::BroadcastOp::create(
        builder, v4_f32,
        F32Constant(builder,
                    static_cast<float>(descriptor_.scale) * kLog2E));
    auto exp2_vector = [&](Value exponent) {
      llvm::SmallVector<Value, kMfmaOutputsPerLane> values;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        values.push_back(mlir::ROCDL::ROCDLExp2::create(
            builder, builder.getF32Type(),
            ExtractVectorElement(builder, exponent, component)));
      }
      return vector_from(values, v4_f32);
    };
    auto reduce_score_row = [&](Value value, bool maximum) {
      // FlyDSL's MFMA16 C layout is
      //   row = lane % 16,
      //   column = (lane / 16) * 4 + component.
      // Reduce the four register values first, then the four lane groups. This
      // is AITER's two-bpermute reduction, rather than reducing four vectors
      // across the unrelated row lanes.
      Value reduced = ExtractVectorElement(builder, value, 0);
      for (int64_t component = 1; component < kMfmaOutputsPerLane;
           ++component) {
        Value element = ExtractVectorElement(builder, value, component);
        reduced = maximum
                      ? mlir::arith::MaxNumFOp::create(builder, reduced, element)
                            .getResult()
                      : mlir::arith::AddFOp::create(builder, reduced, element)
                            .getResult();
      }
      for (int64_t distance : {16, 32}) {
        Value peer = mlir::gpu::ShuffleOp::create(
                         builder, reduced, distance, kWaveSize,
                         mlir::gpu::ShuffleMode::XOR)
                         .getShuffleResult();
        reduced = maximum
                      ? mlir::arith::MaxNumFOp::create(builder, reduced, peer)
                            .getResult()
                      : mlir::arith::AddFOp::create(builder, reduced, peer)
                            .getResult();
      }
      return reduced;
    };
    auto safe_maximum = [&](Value maximum) {
      Value valid = mlir::arith::CmpFOp::create(
          builder, mlir::arith::CmpFPredicate::OGT, maximum, minus_inf);
      return mlir::arith::SelectOp::create(builder, valid, maximum, zero_f32)
          .getResult();
    };
    auto exp2_scalar = [&](Value exponent) {
      return mlir::ROCDL::ROCDLExp2::create(builder, builder.getF32Type(),
                                            exponent)
          .getResult();
    };
    auto cache_offset = [&](Value physical_row, Value kv_head,
                            Value dimension) {
      return Add(
          builder,
          Mul(builder,
              Add(builder,
                  Mul(builder, physical_row,
                      IndexConstant(builder, descriptor_.kv_heads)),
                  kv_head),
              IndexConstant(builder, kHeadSize)),
          dimension);
    };
    auto k_swizzled_offset = [&](Value token, Value dimension) {
      Value swizzle = Mul(builder, Rem(builder, token, 8),
                           IndexConstant(builder, 8));
      Value physical_dimension =
          mlir::arith::XOrIOp::create(builder, dimension, swizzle);
      return Add(builder,
                 Mul(builder, token, IndexConstant(builder, kHeadSize)),
                 physical_dimension);
    };
    // Match Triton AMD's rotating_shared<vec=4, perPhase=4, maxPhase=4,
    // order=[token,dimension]> encoding for logical V[16,128]. The second XOR
    // term rotates each consecutive group of sixteen dimensions again so the
    // pattern does not repeat on the same LDS banks.
    auto v_rotating_offset = [&](Value token, Value dimension) {
      Value phase = Rem(builder, Div(builder, dimension, 4), 4);
      Value block = Rem(builder, Div(builder, dimension, 16), 4);
      Value rotation = Mul(
          builder, mlir::arith::XOrIOp::create(builder, phase, block),
          IndexConstant(builder, 4));
      Value physical_token =
          mlir::arith::XOrIOp::create(builder, token, rotation);
      return Add(builder,
                 Mul(builder, dimension, IndexConstant(builder, kTileTokens)),
                 physical_token);
    };
    auto store_rotating_value_pair = [&](Value first, Value second,
                                         Value token_base,
                                         Value dimension_base,
                                         Value shared_v) {
      Value first_words =
          mlir::vector::BitCastOp::create(builder, v4_i32, first);
      Value second_words =
          mlir::vector::BitCastOp::create(builder, v4_i32, second);
      for (int64_t word = 0; word < 4; ++word) {
        Value first_word = ExtractVectorElement(builder, first_words, word);
        Value second_word = ExtractVectorElement(builder, second_words, word);
        for (int64_t half = 0; half < 2; ++half) {
          const int32_t selector =
              half == 0 ? 0x05040100 : 0x07060302;
          Value packed =
              permute_half_pair(second_word, first_word, selector);
          Value packed_vector = mlir::vector::BitCastOp::create(
              builder, v2_element,
              vector_from(llvm::SmallVector<Value, 1>{packed}, v1_i32));
          Value dimension = Add(
              builder, dimension_base,
              IndexConstant(builder, 2 * word + half));
          StoreSharedPointer(builder, packed_vector, shared_v,
                             v_rotating_offset(token_base, dimension));
        }
      }
    };
    Value thread_id = EmitThreadId(builder, 0);
    Value wave = Div(builder, thread_id, kWaveSize);
    Value lane = Rem(builder, thread_id, kWaveSize);
    Value lane16 = Rem(builder, lane, kMfmaN);
    Value lane_group = Div(builder, lane, kMfmaN);
    Value sequence = EmitBlockId(builder, 0);
    Value kv_head = EmitBlockId(builder, 1);
    Value segment = EmitBlockId(builder, 2);
    Value segment_start = Mul(
        builder, segment,
        IndexConstant(builder, producer_descriptor_.segment_tokens));
    Value segment_end = Add(
        builder, segment_start,
        IndexConstant(builder, producer_descriptor_.segment_tokens));

    Value kv_len_i32;
    if (used_k) {
      kv_len_i32 = EmitBufferLoad(builder, entry_function.getLoc(), used_k,
                                  sequence, builder.getI32Type(),
                                  /*is_scalar=*/true);
    } else {
      TF_RET_CHECK(descriptor_.used_k->literal().shape().element_type() == S32);
      kv_len_i32 = mlir::arith::ConstantIntOp::create(
          builder, builder.getI32Type(),
          descriptor_.used_k->literal().Get<int32_t>({0}));
      for (int64_t i = 1; i < descriptor_.sequences; ++i) {
        Value is_sequence = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::eq, sequence,
            IndexConstant(builder, i));
        Value constant_length = mlir::arith::ConstantIntOp::create(
            builder, builder.getI32Type(),
            descriptor_.used_k->literal().Get<int32_t>({i}));
        kv_len_i32 = mlir::arith::SelectOp::create(
            builder, is_sequence, constant_length, kv_len_i32);
      }
    }
    Value kv_len = mlir::arith::IndexCastOp::create(
        builder, builder.getIndexType(), kv_len_i32);
    Value segment_valid = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::ult, segment_start, kv_len);
    Value remaining =
        mlir::arith::SubIOp::create(builder, kv_len, segment_start);
    Value nonnegative_remaining = mlir::arith::SelectOp::create(
        builder, segment_valid, remaining, IndexConstant(builder, 0));
    Value short_segment = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::ult, nonnegative_remaining,
        IndexConstant(builder, producer_descriptor_.segment_tokens));
    Value segment_token_count = mlir::arith::SelectOp::create(
        builder, short_segment, nonnegative_remaining,
        IndexConstant(builder, producer_descriptor_.segment_tokens));
    Value tiles_to_process = Mul(
        builder,
        Div(builder,
            Add(builder, segment_token_count,
                IndexConstant(builder, kTileTokens - 1)),
            kTileTokens),
        IndexConstant(builder, kTileTokens));

    Value query_row = lane16;
    Value query_row_valid = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::ult, query_row,
        IndexConstant(builder, descriptor_.gqa_group));
    Value query_head = Add(
        builder,
        Mul(builder, kv_head, IndexConstant(builder, descriptor_.gqa_group)),
        query_row);
    llvm::SmallVector<Value> query_fragments;
    query_fragments.reserve(kHeadSize / kMfmaK);
    for (int64_t k_step = 0; k_step < kHeadSize / kMfmaK; ++k_step) {
      Value q_offset = Add(
          builder,
          Mul(builder,
              Add(builder,
                  Mul(builder, sequence,
                      IndexConstant(builder, descriptor_.query_heads)),
                  query_head),
              IndexConstant(builder, kHeadSize)),
          Add(builder, IndexConstant(builder, k_step * kMfmaK),
              Mul(builder, lane_group,
                  IndexConstant(builder, kMfmaInputsPerLane))));
      Value safe_q_offset = mlir::arith::SelectOp::create(
          builder, query_row_valid, q_offset, IndexConstant(builder, 0));
      query_fragments.push_back(EmitBufferLoad(
          builder, entry_function.getLoc(), query, safe_q_offset, v4_element));
    }

    Value shared_k =
        CreateSharedPointer(builder, element_type, kTileTokens * kHeadSize);
    Value shared_v =
        CreateSharedPointer(builder, element_type, kTileTokens * kHeadSize);
    Value value_row_base = Add(
        builder, Mul(builder, wave, IndexConstant(builder, 8)),
        Mul(builder, lane_group, IndexConstant(builder, 2)));
    Value value_dimension_base =
        Mul(builder, lane16, IndexConstant(builder, kStageWidth));

    auto physical_page_for_tile = [&](Value logical_tile) {
      Value logical_page = Div(builder, logical_tile, descriptor_.page_size);
      Value table_offset = Add(
          builder,
          Mul(builder, sequence,
              IndexConstant(builder, descriptor_.pages_per_sequence)),
          logical_page);
      Value physical_page_i32 = EmitBufferLoad(
          builder, entry_function.getLoc(), block_table, table_offset,
          builder.getI32Type(), /*is_scalar=*/true);
      return mlir::arith::IndexCastOp::create(
          builder, builder.getIndexType(), physical_page_i32);
    };
    Value safe_first_tile = mlir::arith::SelectOp::create(
        builder, segment_valid, segment_start, IndexConstant(builder, 0));
    Value first_physical_page = physical_page_for_tile(safe_first_tile);
    Value first_page_slot = Rem(builder, safe_first_tile, descriptor_.page_size);
    Value first_physical_row = Add(
        builder,
        Mul(builder, first_physical_page,
            IndexConstant(builder, descriptor_.page_size)),
        first_page_slot);
    for (int64_t stage = 0; stage < kStageIterations; ++stage) {
      Value linear = Add(
          builder,
          Mul(builder, thread_id, IndexConstant(builder, kStageWidth)),
          IndexConstant(builder, stage * kBlockThreads * kStageWidth));
      Value token = Div(builder, linear, kHeadSize);
      Value dimension = Rem(builder, linear, kHeadSize);
      Value physical_row = Add(builder, first_physical_row, token);
      Value key = EmitBufferLoad(
          builder, entry_function.getLoc(), key_cache,
          cache_offset(physical_row, kv_head, dimension), v8_element,
          /*is_scalar=*/false, /*cache_policy=*/2);
      StoreSharedPointer(builder, key, shared_k,
                         k_swizzled_offset(token, dimension));
    }
    llvm::SmallVector<Value> first_values;
    first_values.reserve(kStageIterations);
    for (int64_t stage = 0; stage < kStageIterations; ++stage) {
      Value token =
          Add(builder, value_row_base, IndexConstant(builder, stage));
      first_values.push_back(EmitBufferLoad(
          builder, entry_function.getLoc(), value_cache,
          cache_offset(Add(builder, first_physical_row, token), kv_head,
                       value_dimension_base),
          v8_element, /*is_scalar=*/false, /*cache_policy=*/2));
    }
    TF_RET_CHECK(first_values.size() == 2);
    store_rotating_value_pair(first_values[0], first_values[1], value_row_base,
                              value_dimension_base, shared_v);
    mlir::gpu::BarrierOp::create(builder);

    llvm::SmallVector<Value> initial_state(kDBlocksPerWave, zero4);
    initial_state.append({minus_inf, zero_f32});
    mlir::scf::ForOp tile_loop = mlir::scf::ForOp::create(
        builder, IndexConstant(builder, 0), tiles_to_process,
        IndexConstant(builder, kTileTokens), initial_state,
        [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
    {
      mlir::OpBuilder::InsertionGuard loop_guard(builder);
      builder.setInsertionPointToStart(tile_loop.getBody());
      Value tile_base = tile_loop.getInductionVar();
      Value tile_start = Add(builder, segment_start, tile_base);
      llvm::SmallVector<Value> previous_outputs;
      for (int64_t d_block = 0; d_block < kDBlocksPerWave; ++d_block) {
        previous_outputs.push_back(tile_loop.getRegionIterArg(d_block));
      }
      Value previous_max = tile_loop.getRegionIterArg(kDBlocksPerWave);
      Value previous_sum = tile_loop.getRegionIterArg(kDBlocksPerWave + 1);

      Value next_tile_start =
          Add(builder, tile_start, IndexConstant(builder, kTileTokens));
      Value next_tile_valid = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, next_tile_start,
          Add(builder, segment_start, segment_token_count));
      // Issue all next-tile global reads before consuming the current shared
      // tile. Guard the uniform final iteration so every segment does not load
      // an unused K/V tile from page zero. The values remain in VGPRs across
      // current QK/softmax/PV work, preserving the rolling latency window.
      llvm::SmallVector<mlir::Type> prefetch_types(
          2 * kStageIterations, v8_element);
      mlir::scf::IfOp prefetch = mlir::scf::IfOp::create(
          builder, prefetch_types, next_tile_valid, /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard prefetch_guard(builder);
        builder.setInsertionPointToStart(prefetch.thenBlock());
        Value next_physical_page = physical_page_for_tile(next_tile_start);
        Value next_page_slot =
            Rem(builder, next_tile_start, descriptor_.page_size);
        Value next_physical_row = Add(
            builder,
            Mul(builder, next_physical_page,
                IndexConstant(builder, descriptor_.page_size)),
            next_page_slot);
        llvm::SmallVector<Value> prefetched;
        prefetched.reserve(2 * kStageIterations);
        for (int64_t stage = 0; stage < kStageIterations; ++stage) {
          Value linear = Add(
              builder,
              Mul(builder, thread_id, IndexConstant(builder, kStageWidth)),
              IndexConstant(builder, stage * kBlockThreads * kStageWidth));
          Value token = Div(builder, linear, kHeadSize);
          Value dimension = Rem(builder, linear, kHeadSize);
          prefetched.push_back(EmitBufferLoad(
              builder, entry_function.getLoc(), key_cache,
              cache_offset(Add(builder, next_physical_row, token), kv_head,
                           dimension),
              v8_element, /*is_scalar=*/false, /*cache_policy=*/2));
        }
        for (int64_t stage = 0; stage < kStageIterations; ++stage) {
          Value token =
              Add(builder, value_row_base, IndexConstant(builder, stage));
          prefetched.push_back(EmitBufferLoad(
              builder, entry_function.getLoc(), value_cache,
              cache_offset(Add(builder, next_physical_row, token), kv_head,
                           value_dimension_base),
              v8_element, /*is_scalar=*/false, /*cache_policy=*/2));
        }
        mlir::scf::YieldOp::create(builder, prefetched);
        builder.setInsertionPointToStart(prefetch.elseBlock());
        mlir::scf::YieldOp::create(
            builder, llvm::SmallVector<Value>(2 * kStageIterations, zero8));
      }
      builder.setInsertionPointAfter(prefetch);
      llvm::SmallVector<Value> next_keys;
      llvm::SmallVector<Value> next_values;
      for (int64_t stage = 0; stage < kStageIterations; ++stage) {
        next_keys.push_back(prefetch.getResult(stage));
        next_values.push_back(prefetch.getResult(kStageIterations + stage));
      }

      llvm::SmallVector<Value> key_fragments;
      key_fragments.reserve(kHeadSize / kMfmaK);
      for (int64_t k_step = 0; k_step < kHeadSize / kMfmaK; ++k_step) {
        Value dimension = Add(
            builder, IndexConstant(builder, k_step * kMfmaK),
            Mul(builder, lane_group,
                IndexConstant(builder, kMfmaInputsPerLane)));
        key_fragments.push_back(LoadSharedPointer(
            builder, shared_k, k_swizzled_offset(lane16, dimension),
            v4_element));
      }
      llvm::SmallVector<Value> value_fragments;
      value_fragments.reserve(kDBlocksPerWave);
      Value p_token_base =
          Mul(builder, lane_group, IndexConstant(builder, kMfmaInputsPerLane));
      for (int64_t local_d_block = 0; local_d_block < kDBlocksPerWave;
           ++local_d_block) {
        Value d_block = Add(
            builder,
            Mul(builder, wave, IndexConstant(builder, kDBlocksPerWave)),
            IndexConstant(builder, local_d_block));
        Value dimension =
            Add(builder,
                Mul(builder, d_block, IndexConstant(builder, kMfmaN)), lane16);
        value_fragments.push_back(LoadSharedPointer(
            builder, shared_v,
            v_rotating_offset(p_token_base, dimension), v4_element));
      }

      // All waves have captured the current shared fragments. K/V can now be
      // overwritten with the next tile while arithmetic consumes registers.
      mlir::gpu::BarrierOp::create(builder);
      for (int64_t stage = 0; stage < kStageIterations; ++stage) {
        Value linear = Add(
            builder,
            Mul(builder, thread_id, IndexConstant(builder, kStageWidth)),
            IndexConstant(builder, stage * kBlockThreads * kStageWidth));
        Value token = Div(builder, linear, kHeadSize);
        Value dimension = Rem(builder, linear, kHeadSize);
        StoreSharedPointer(builder, next_keys[stage], shared_k,
                           k_swizzled_offset(token, dimension));
      }
      TF_RET_CHECK(next_values.size() == 2);
      store_rotating_value_pair(next_values[0], next_values[1], value_row_base,
                                value_dimension_base, shared_v);

      Value score = zero4;
      for (int64_t k_step = 0; k_step < kHeadSize / kMfmaK; ++k_step) {
        // Form K * Q^T.  The numerical result is the transpose of the usual
        // Q * K^T score tile, and Fly's native MFMA16 C layout then places the
        // query row in lane16 while the four token columns occupy the register
        // components.  This is the source-swapped equivalent of AITER's
        // isTransposed MFMA layout and makes the score fragment directly
        // compatible with the following probability operand.
        score = mma(key_fragments[k_step], query_fragments[k_step], score);
      }
      score = mlir::arith::MulFOp::create(builder, score, scale4);
      llvm::SmallVector<Value, kMfmaOutputsPerLane> score_elements;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value logical_token = Add(
            builder, tile_start,
            Add(builder,
                Mul(builder, lane_group,
                    IndexConstant(builder, kMfmaOutputsPerLane)),
                IndexConstant(builder, component)));
        Value token_before_kv = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, logical_token, kv_len);
        Value token_before_segment = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, logical_token,
            segment_end);
        Value token_before_context = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, logical_token,
            IndexConstant(builder, descriptor_.max_context));
        Value token_valid = mlir::arith::AndIOp::create(
            builder, query_row_valid,
            mlir::arith::AndIOp::create(
                builder, token_before_kv,
                mlir::arith::AndIOp::create(builder, token_before_segment,
                                            token_before_context)));
        score_elements.push_back(mlir::arith::SelectOp::create(
            builder, token_valid,
            ExtractVectorElement(builder, score, component), minus_inf));
      }
      Value masked_score = vector_from(score_elements, v4_f32);

      Value tile_max = reduce_score_row(masked_score, /*maximum=*/true);
      Value new_max = mlir::arith::MaxNumFOp::create(
          builder, previous_max, tile_max);
      Value safe_new_max = safe_maximum(new_max);
      Value safe_new_max4 =
          mlir::vector::BroadcastOp::create(builder, v4_f32, safe_new_max);
      Value probability = exp2_vector(mlir::arith::SubFOp::create(
          builder, masked_score, safe_new_max4));
      // Invalid scores are already -inf, so exp2 produces an exact zero. This
      // is the same single-mask path as AITER and avoids four redundant
      // compare/select pairs after every exponential batch.
      Value tile_sum = reduce_score_row(probability, /*maximum=*/false);

      // QK and PV use compatible MFMA16 layouts. Narrow the four probabilities
      // in place and feed them directly to the PV atom, as AITER does; no LDS
      // transpose or second workgroup barrier is required.
      Value p_fragment =
          mlir::arith::TruncFOp::create(builder, v4_element, probability);
      llvm::SmallVector<Value> tile_outputs(kDBlocksPerWave, zero4);
      for (int64_t local_d_block = 0; local_d_block < kDBlocksPerWave;
           ++local_d_block) {
        // Form V^T * P^T.  The transposed mathematical result is laid out as
        // query=lane16 and four contiguous output dimensions per register,
        // preserving the scalar online-softmax state and permitting a direct
        // vector store without an LDS C-layout transpose.
        tile_outputs[local_d_block] =
            mma(value_fragments[local_d_block], p_fragment,
                tile_outputs[local_d_block]);
      }

      // The initial maximum is -inf, hence the first alpha is exactly zero;
      // no previous-sum guard is required. Subsequent empty tiles also retain
      // a zero contribution through the multiplication below.
      Value previous_weight = exp2_scalar(mlir::arith::SubFOp::create(
          builder, previous_max, safe_new_max));
      Value previous_weight4 =
          mlir::vector::BroadcastOp::create(builder, v4_f32, previous_weight);
      llvm::SmallVector<Value> next_state;
      for (int64_t local_d_block = 0; local_d_block < kDBlocksPerWave;
           ++local_d_block) {
        next_state.push_back(mlir::arith::AddFOp::create(
            builder,
            mlir::arith::MulFOp::create(
                builder, previous_outputs[local_d_block], previous_weight4),
            tile_outputs[local_d_block]));
      }
      next_state.append(
          {new_max,
           mlir::arith::AddFOp::create(
               builder,
               mlir::arith::MulFOp::create(builder, previous_sum,
                                           previous_weight),
               tile_sum)});
      mlir::scf::YieldOp::create(builder, next_state);
    }
    builder.setInsertionPointAfter(tile_loop);

    Value final_max = tile_loop.getResult(kDBlocksPerWave);
    Value final_sum = tile_loop.getResult(kDBlocksPerWave + 1);
    Value owns_state = mlir::arith::AndIOp::create(
        builder,
        mlir::arith::AndIOp::create(
            builder,
            mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::eq, wave,
                IndexConstant(builder, 0)),
            mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::eq, lane_group,
                IndexConstant(builder, 0))),
        query_row_valid);
    Value state_index = Add(
        builder,
        Mul(builder,
            Add(builder,
                Mul(builder, sequence,
                    IndexConstant(builder, descriptor_.query_heads)),
                query_head),
            IndexConstant(builder, producer_descriptor_.num_segments)),
        segment);
    mlir::scf::IfOp state_store = mlir::scf::IfOp::create(
        builder,
        mlir::TypeRange{partial_maximum.getType(), partial_sum.getType()},
        owns_state, /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(state_store.thenBlock());
      Value updated_maximum = EmitBufferStore(
          builder, entry_function.getLoc(), final_max, partial_maximum,
          state_index);
      Value updated_sum = EmitBufferStore(builder, entry_function.getLoc(),
                                          final_sum, partial_sum, state_index);
      mlir::scf::YieldOp::create(
          builder, mlir::ValueRange{updated_maximum, updated_sum});
      builder.setInsertionPointToStart(state_store.elseBlock());
      mlir::scf::YieldOp::create(
          builder, mlir::ValueRange{partial_maximum, partial_sum});
    }
    builder.setInsertionPointAfter(state_store);
    partial_maximum = state_store.getResult(0);
    partial_sum = state_store.getResult(1);

    // The PV accumulator already has a contiguous four-column fragment. Emit
    // one dwordx4 store per D16 block instead of four scalar branches/stores.
    mlir::scf::IfOp output_store = mlir::scf::IfOp::create(
        builder, mlir::TypeRange{partial_output.getType()}, query_row_valid,
        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(output_store.thenBlock());
      Value updated_output = partial_output;
      for (int64_t local_d_block = 0; local_d_block < kDBlocksPerWave;
           ++local_d_block) {
        Value dimension = Add(
            builder,
            IndexConstant(builder, local_d_block * kMfmaN),
            Add(builder,
                Mul(builder, wave, IndexConstant(builder, kWaveHeadSize)),
                Mul(builder, lane_group,
                    IndexConstant(builder, kMfmaOutputsPerLane))));
        Value output_offset = Add(
            builder,
            Mul(builder,
                Add(builder,
                    Mul(builder,
                        Add(builder,
                            Mul(builder, sequence,
                                IndexConstant(builder,
                                              descriptor_.query_heads)),
                            query_head),
                        IndexConstant(builder,
                                      producer_descriptor_.num_segments)),
                    segment),
                IndexConstant(builder, kHeadSize)),
            dimension);
        updated_output = EmitBufferStore(
            builder, entry_function.getLoc(),
            tile_loop.getResult(local_d_block), updated_output, output_offset);
      }
      mlir::scf::YieldOp::create(builder, updated_output);
      builder.setInsertionPointToStart(output_store.elseBlock());
      mlir::scf::YieldOp::create(builder, partial_output);
    }
    builder.setInsertionPointAfter(output_store);
    partial_output = output_store.getResult(0);
    if (producer_descriptor_.fused_reducer) {
      RETURN_IF_ERROR(EmitFusedReducer(
          builder, entry_function, thread_id, sequence, kv_head, output,
          partial_output, partial_maximum, partial_sum, completion_tickets));
    } else {
      mlir::func::ReturnOp::create(
          builder,
          mlir::ValueRange{partial_output, partial_maximum, partial_sum});
    }
    // Match Triton's staged static bodies.  The rolling K/V pipeline still
    // consumes one 16-token tile at a time, but grouping iterations amortizes
    // loop/address control and gives LLVM enough scope to interleave the next
    // global reads with multiple MFMA batches. XLA profiles factors 1/2/3.
    if (unroll_factor_ > 1 &&
        mlir::failed(
            mlir::loopUnrollByFactor(tile_loop, unroll_factor_))) {
      return absl::InternalError(
          "Failed to unroll cooperative Fly paged-attention loop.");
    }
    return absl::OkStatus();
  }

  FlyPagedAttentionSegmentedProducerDescriptor producer_descriptor_;
  FlyPagedAttentionDescriptor descriptor_;
  int64_t unroll_factor_;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

std::unique_ptr<MlirKernelEmitter>
CreateFlyXTilePagedAttentionOnlineEmitter(
    const HloFusionAnalysis& analysis) {
  if (std::optional<FlyPagedAttentionSegmentedProducerDescriptor> producer =
          GetFlyPagedAttentionSegmentedProducerDescriptor(analysis);
      producer.has_value() && producer->attention.max_context >= 65536 &&
      producer->attention.gqa_group == 4) {
    return std::make_unique<FlyXTilePagedAttentionCooperativeEmitter>(analysis);
  }
  return std::make_unique<FlyXTilePagedAttentionOnlineEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
