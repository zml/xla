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

#include "xla/backends/gpu/codegen/flydsl/xtile_paged_attention.h"

#include <array>
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
#include "xla/backends/gpu/codegen/flydsl/xtile_paged_attention_online.h"
#include "xla/backends/gpu/codegen/flydsl/xtile_paged_attention_reduce.h"
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
constexpr int64_t kNumWaves = 4;
constexpr int64_t kBlockThreads = kNumWaves * kWaveSize;
constexpr int64_t kTokensPerWave = 32;
constexpr int64_t kMfmaN = 16;
constexpr int64_t kMfmaK = 16;
constexpr int64_t kMfmaInputsPerLane = 4;
constexpr int64_t kMfmaOutputsPerLane = 4;
constexpr int64_t kHeadSize = 128;
constexpr int64_t kValueStageWidth = 8;
constexpr float kLog2E = 1.4426950408889634f;

Value IndexConstant(mlir::ImplicitLocOpBuilder& builder, int64_t value) {
  return mlir::arith::ConstantIndexOp::create(builder, value);
}

Value Add(mlir::ImplicitLocOpBuilder& builder, Value lhs, Value rhs) {
  return mlir::arith::AddIOp::create(builder, lhs, rhs);
}

Value Sub(mlir::ImplicitLocOpBuilder& builder, Value lhs, Value rhs) {
  return mlir::arith::SubIOp::create(builder, lhs, rhs);
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
                     mlir::Type result_type, bool is_scalar = false) {
  mlir::OperationState state(location, is_scalar ? "xla_gpu.scalar_buffer_load"
                                                 : "xla_gpu.buffer_load");
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

class FlyXTilePagedAttentionEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTilePagedAttentionEmitter(const HloFusionAnalysis& analysis)
      : descriptor_(*GetFlyPagedAttentionDescriptor(analysis)),
        launch_dimensions_(
            se::BlockDim(descriptor_.sequences, descriptor_.kv_heads, 1),
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
        "FlyXTilePagedAttentionEmitter builds its module directly.");
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function) const {
    const int64_t parameter_count = descriptor_.call->parent()->num_parameters();
    TF_RET_CHECK(entry_function.getNumArguments() == parameter_count + 1);
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
        descriptor_.gqa_group > 0 && descriptor_.gqa_group <= 16 &&
        descriptor_.head_dimension == kHeadSize &&
        descriptor_.max_context <= kNumWaves * kTokensPerWave &&
        descriptor_.max_context % kMfmaN == 0);

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

    mlir::Type element_type = descriptor_.element_type == BF16
                                  ? builder.getBF16Type()
                                  : builder.getF16Type();
    auto v4_element = mlir::VectorType::get({4}, element_type);
    auto v8_element = mlir::VectorType::get({8}, element_type);
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
    Value one_f32 = F32Constant(builder, 1.0f);
    Value minus_inf = NegativeInfinity(builder);
    Value zero_element =
        mlir::arith::TruncFOp::create(builder, element_type, zero_f32);
    Value zero4 = mlir::vector::BroadcastOp::create(builder, v4_f32, zero_f32);
    Value scale4 = mlir::vector::BroadcastOp::create(
        builder, v4_f32,
        F32Constant(builder, static_cast<float>(descriptor_.scale)));

    Value thread_id = EmitThreadId(builder, 0);
    Value wave = Div(builder, thread_id, kWaveSize);
    Value lane = Rem(builder, thread_id, kWaveSize);
    Value lane16 = Rem(builder, lane, kMfmaN);
    Value lane_group = Div(builder, lane, kMfmaN);
    Value sequence = EmitBlockId(builder, 0);
    Value kv_head = EmitBlockId(builder, 1);

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

    constexpr int64_t kValueElements = kNumWaves * kTokensPerWave * kHeadSize;
    const int64_t kProbabilityElements =
        kNumWaves * descriptor_.gqa_group * kTokensPerWave;
    const int64_t kScalarElements = kNumWaves * descriptor_.gqa_group;
    auto values_type =
        mlir::RankedTensorType::get({kValueElements}, element_type);
    auto probabilities_type =
        mlir::RankedTensorType::get({kProbabilityElements}, element_type);
    auto scalars_type =
        mlir::RankedTensorType::get({kScalarElements}, builder.getF32Type());
    Value shared_values = AllocateSharedOp::create(builder, values_type);
    Value shared_probabilities =
        AllocateSharedOp::create(builder, probabilities_type);
    Value shared_l = AllocateSharedOp::create(builder, scalars_type);
    Value shared_m = AllocateSharedOp::create(builder, scalars_type);

    auto sync_shared = [&]() {
      auto sync = SyncThreadsOp::create(
          builder,
          mlir::TypeRange{values_type, probabilities_type, scalars_type,
                          scalars_type},
          mlir::ValueRange{shared_values, shared_probabilities, shared_l,
                           shared_m});
      shared_values = sync.getResult(0);
      shared_probabilities = sync.getResult(1);
      shared_l = sync.getResult(2);
      shared_m = sync.getResult(3);
    };

    auto conditional_insert = [&](Value tensor, Value condition, Value value,
                                  Value index) {
      mlir::scf::IfOp insert = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{tensor.getType()}, condition,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(insert.thenBlock());
        Value updated = mlir::tensor::InsertOp::create(builder, value, tensor,
                                                       mlir::ValueRange{index});
        mlir::scf::YieldOp::create(builder, updated);
        builder.setInsertionPointToStart(insert.elseBlock());
        mlir::scf::YieldOp::create(builder, tensor);
      }
      builder.setInsertionPointAfter(insert);
      return insert.getResult(0);
    };

    auto cache_offset = [&](Value physical_page, Value page_slot,
                            Value dimension) {
      Value physical_row =
          Add(builder,
              Mul(builder, physical_page,
                  IndexConstant(builder, descriptor_.page_size)),
              page_slot);
      return Add(builder,
                 Mul(builder,
                     Add(builder,
                         Mul(builder, physical_row,
                             IndexConstant(builder, descriptor_.kv_heads)),
                         kv_head),
                     IndexConstant(builder, kHeadSize)),
                 dimension);
    };

    auto stage_cache = [&](Value cache, bool is_value) {
      const int64_t stage_iterations = descriptor_.max_context / kMfmaN;
      for (int64_t stage = 0; stage < stage_iterations; ++stage) {
        Value linear = Add(
            builder,
            Mul(builder, thread_id, IndexConstant(builder, kValueStageWidth)),
            IndexConstant(builder, stage * kBlockThreads * kValueStageWidth));
        Value token = Div(builder, linear, kHeadSize);
        Value dimension = Rem(builder, linear, kHeadSize);
        Value stage_start = IndexConstant(builder, stage * kMfmaN);
        Value stage_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, stage_start, kv_len);
        Value logical_page =
            IndexConstant(builder, stage * kMfmaN / descriptor_.page_size);
        Value safe_logical_page = mlir::arith::SelectOp::create(
            builder, stage_valid, logical_page, IndexConstant(builder, 0));
        Value table_offset =
            Add(builder,
                Mul(builder, sequence,
                    IndexConstant(builder, descriptor_.pages_per_sequence)),
                safe_logical_page);
        Value physical_page_i32 = EmitBufferLoad(
            builder, entry_function.getLoc(), block_table, table_offset,
            builder.getI32Type(), /*is_scalar=*/true);
        Value physical_page = mlir::arith::IndexCastOp::create(
            builder, builder.getIndexType(), physical_page_i32);
        Value page_slot = Add(
            builder,
            IndexConstant(builder, (stage * kMfmaN) % descriptor_.page_size),
            Sub(builder, token, stage_start));
        Value loaded = EmitBufferLoad(
            builder, entry_function.getLoc(), cache,
            cache_offset(physical_page, page_slot, dimension), v8_element);
        Value swizzle =
            is_value ? Mul(builder, Rem(builder, Div(builder, token, 4), 4),
                           IndexConstant(builder, 16))
                     : Mul(builder, Rem(builder, token, 8),
                           IndexConstant(builder, 8));
        Value shared_dimension =
            mlir::arith::XOrIOp::create(builder, dimension, swizzle);
        Value shared_offset =
            Add(builder, Mul(builder, token, IndexConstant(builder, kHeadSize)),
                shared_dimension);
        for (int64_t element = 0; element < kValueStageWidth; ++element) {
          shared_values = mlir::tensor::InsertOp::create(
              builder, ExtractVectorElement(builder, loaded, element),
              shared_values,
              mlir::ValueRange{Add(builder, shared_offset,
                                   IndexConstant(builder, element))});
        }
      }
    };

    // Phase 1: stage paged K and compute one 32-token score partition per
    // wave, exactly matching FlyDSL's short-decode specialization.
    stage_cache(key_cache, /*is_value=*/false);
    sync_shared();

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
      Value q_offset =
          Add(builder,
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

    Value tile_start =
        Mul(builder, wave, IndexConstant(builder, kTokensPerWave));
    Value query_row_base =
        Mul(builder, lane_group, IndexConstant(builder, kMfmaOutputsPerLane));
    std::array<Value, 2> masked_scores;
    for (int64_t token_half = 0; token_half < 2; ++token_half) {
      Value token = Add(
          builder, tile_start,
          Add(builder, IndexConstant(builder, token_half * kMfmaN), lane16));
      Value score = zero4;
      for (int64_t k_step = 0; k_step < query_fragments.size(); ++k_step) {
        Value dimension = Add(builder, IndexConstant(builder, k_step * kMfmaK),
                              Mul(builder, lane_group,
                                  IndexConstant(builder, kMfmaInputsPerLane)));
        Value swizzle =
            Mul(builder, Rem(builder, token, 8), IndexConstant(builder, 8));
        Value shared_dimension =
            mlir::arith::XOrIOp::create(builder, dimension, swizzle);
        Value shared_offset =
            Add(builder, Mul(builder, token, IndexConstant(builder, kHeadSize)),
                shared_dimension);
        llvm::SmallVector<Value, 4> k_elements;
        for (int64_t element = 0; element < kMfmaInputsPerLane; ++element) {
          k_elements.push_back(mlir::tensor::ExtractOp::create(
              builder, shared_values,
              mlir::ValueRange{Add(builder, shared_offset,
                                   IndexConstant(builder, element))}));
        }
        Value k_fragment = vector_from(k_elements, v4_element);
        score = mma(query_fragments[k_step], k_fragment, score);
      }
      score = mlir::arith::MulFOp::create(builder, score, scale4);
      Value token_valid = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, token, kv_len);
      llvm::SmallVector<Value, 4> score_elements;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value row =
            Add(builder, query_row_base, IndexConstant(builder, component));
        Value row_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, row,
            IndexConstant(builder, descriptor_.gqa_group));
        Value valid =
            mlir::arith::AndIOp::create(builder, token_valid, row_valid);
        score_elements.push_back(mlir::arith::SelectOp::create(
            builder, valid, ExtractVectorElement(builder, score, component),
            minus_inf));
      }
      masked_scores[token_half] = vector_from(score_elements, v4_f32);
    }

    auto reduce_vector = [&](Value value, bool is_maximum) {
      for (int64_t distance : {8, 4, 2, 1}) {
        llvm::SmallVector<Value, 4> reduced;
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

    Value local_max = mlir::arith::MaxNumFOp::create(builder, masked_scores[0],
                                                     masked_scores[1]);
    local_max = reduce_vector(local_max, /*is_maximum=*/true);
    std::array<Value, 2> probabilities;
    Value local_sum = zero4;
    Value log2e4 = mlir::vector::BroadcastOp::create(
        builder, v4_f32, F32Constant(builder, kLog2E));
    for (int64_t token_half = 0; token_half < 2; ++token_half) {
      Value token = Add(
          builder, tile_start,
          Add(builder, IndexConstant(builder, token_half * kMfmaN), lane16));
      Value token_valid = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, token, kv_len);
      Value exponent = mlir::arith::MulFOp::create(
          builder,
          mlir::arith::SubFOp::create(builder, masked_scores[token_half],
                                      local_max),
          log2e4);
      llvm::SmallVector<Value, 4> p_elements;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value candidate = mlir::ROCDL::ROCDLExp2::create(
            builder, builder.getF32Type(),
            ExtractVectorElement(builder, exponent, component));
        p_elements.push_back(mlir::arith::SelectOp::create(
            builder, token_valid, candidate, zero_f32));
      }
      probabilities[token_half] = vector_from(p_elements, v4_f32);
      local_sum = mlir::arith::AddFOp::create(builder, local_sum,
                                              probabilities[token_half]);
    }
    local_sum = reduce_vector(local_sum, /*is_maximum=*/false);

    for (int64_t component = 0; component < kMfmaOutputsPerLane; ++component) {
      Value row =
          Add(builder, query_row_base, IndexConstant(builder, component));
      Value row_valid = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, row,
          IndexConstant(builder, descriptor_.gqa_group));
      Value lane_owns_scalar = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, lane16,
          IndexConstant(builder, component));
      lane_owns_scalar =
          mlir::arith::AndIOp::create(builder, lane_owns_scalar, row_valid);
      Value scalar_slot =
          Add(builder,
              Mul(builder, wave, IndexConstant(builder, descriptor_.gqa_group)),
              row);
      shared_m = conditional_insert(
          shared_m, lane_owns_scalar,
          ExtractVectorElement(builder, local_max, component), scalar_slot);
      shared_l = conditional_insert(
          shared_l, lane_owns_scalar,
          ExtractVectorElement(builder, local_sum, component), scalar_slot);
      for (int64_t token_half = 0; token_half < 2; ++token_half) {
        Value p_slot = Add(
            builder,
            Mul(builder, scalar_slot, IndexConstant(builder, kTokensPerWave)),
            Add(builder, IndexConstant(builder, token_half * kMfmaN), lane16));
        Value p = mlir::arith::TruncFOp::create(
            builder, element_type,
            ExtractVectorElement(builder, probabilities[token_half],
                                 component));
        shared_probabilities =
            conditional_insert(shared_probabilities, row_valid, p, p_slot);
      }
    }
    sync_shared();

    // Phase 2: K is dead, so reuse its 32-KiB LDS tile for paged V.  Each
    // wave then owns two D16 output blocks and merges all four score
    // partitions without an FP32 partial-output spill.
    stage_cache(value_cache, /*is_value=*/true);
    sync_shared();

    std::array<Value, kNumWaves> partial_maxima;
    std::array<Value, kNumWaves> partial_sums;
    for (int64_t source_wave = 0; source_wave < kNumWaves; ++source_wave) {
      llvm::SmallVector<Value, 4> maxima;
      llvm::SmallVector<Value, 4> sums;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value row =
            Add(builder, query_row_base, IndexConstant(builder, component));
        Value row_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, row,
            IndexConstant(builder, descriptor_.gqa_group));
        Value safe_row = mlir::arith::SelectOp::create(
            builder, row_valid, row, IndexConstant(builder, 0));
        Value scalar_slot =
            Add(builder,
                IndexConstant(builder, source_wave * descriptor_.gqa_group),
                safe_row);
        Value maximum = mlir::tensor::ExtractOp::create(
            builder, shared_m, mlir::ValueRange{scalar_slot});
        Value sum = mlir::tensor::ExtractOp::create(
            builder, shared_l, mlir::ValueRange{scalar_slot});
        maxima.push_back(mlir::arith::SelectOp::create(builder, row_valid,
                                                       maximum, minus_inf));
        sums.push_back(
            mlir::arith::SelectOp::create(builder, row_valid, sum, zero_f32));
      }
      partial_maxima[source_wave] = vector_from(maxima, v4_f32);
      partial_sums[source_wave] = vector_from(sums, v4_f32);
    }

    Value global_max = partial_maxima[0];
    for (int64_t source_wave = 1; source_wave < kNumWaves; ++source_wave) {
      global_max = mlir::arith::MaxNumFOp::create(builder, global_max,
                                                  partial_maxima[source_wave]);
    }
    std::array<Value, kNumWaves> weights;
    Value denominator = zero4;
    for (int64_t source_wave = 0; source_wave < kNumWaves; ++source_wave) {
      Value exponent = mlir::arith::MulFOp::create(
          builder,
          mlir::arith::SubFOp::create(builder, partial_maxima[source_wave],
                                      global_max),
          log2e4);
      llvm::SmallVector<Value, 4> weight_elements;
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value candidate = mlir::ROCDL::ROCDLExp2::create(
            builder, builder.getF32Type(),
            ExtractVectorElement(builder, exponent, component));
        Value has_mass = mlir::arith::CmpFOp::create(
            builder, mlir::arith::CmpFPredicate::OGT,
            ExtractVectorElement(builder, partial_sums[source_wave], component),
            zero_f32);
        weight_elements.push_back(mlir::arith::SelectOp::create(
            builder, has_mass, candidate, zero_f32));
      }
      weights[source_wave] = vector_from(weight_elements, v4_f32);
      denominator = mlir::arith::AddFOp::create(
          builder, denominator,
          mlir::arith::MulFOp::create(builder, partial_sums[source_wave],
                                      weights[source_wave]));
    }
    llvm::SmallVector<Value, 4> inverse_elements;
    for (int64_t component = 0; component < kMfmaOutputsPerLane; ++component) {
      Value value = ExtractVectorElement(builder, denominator, component);
      Value positive = mlir::arith::CmpFOp::create(
          builder, mlir::arith::CmpFPredicate::OGT, value, zero_f32);
      Value safe =
          mlir::arith::SelectOp::create(builder, positive, value, one_f32);
      inverse_elements.push_back(
          mlir::ROCDL::ROCDLRcp::create(builder, builder.getF32Type(), safe));
    }
    Value inverse_denominator = vector_from(inverse_elements, v4_f32);

    constexpr int64_t kDBlocksPerWave = kHeadSize / (kNumWaves * kMfmaN);
    std::array<Value, kDBlocksPerWave> output_accumulators = {zero4, zero4};
    for (int64_t source_wave = 0; source_wave < kNumWaves; ++source_wave) {
      std::array<Value, kDBlocksPerWave> source_accumulators = {zero4, zero4};
      for (int64_t token_half = 0; token_half < 2; ++token_half) {
        Value p_row_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, lane16,
            IndexConstant(builder, descriptor_.gqa_group));
        Value safe_p_row = mlir::arith::SelectOp::create(
            builder, p_row_valid, lane16, IndexConstant(builder, 0));
        llvm::SmallVector<Value, 4> p_elements;
        for (int64_t k_lane = 0; k_lane < kMfmaInputsPerLane; ++k_lane) {
          const int64_t token_base = token_half * kMfmaN + k_lane;
          Value token_in_partition =
              Add(builder, IndexConstant(builder, token_base),
                  Mul(builder, lane_group,
                      IndexConstant(builder, kMfmaInputsPerLane)));
          Value p_slot =
              Add(builder,
                  Mul(builder,
                      Add(builder,
                          IndexConstant(builder,
                                        source_wave * descriptor_.gqa_group),
                          safe_p_row),
                      IndexConstant(builder, kTokensPerWave)),
                  token_in_partition);
          Value p = mlir::tensor::ExtractOp::create(
              builder, shared_probabilities, mlir::ValueRange{p_slot});
          p_elements.push_back(mlir::arith::SelectOp::create(
              builder, p_row_valid, p, zero_element));
        }
        Value p_fragment = vector_from(p_elements, v4_element);
        for (int64_t local_d_block = 0; local_d_block < kDBlocksPerWave;
             ++local_d_block) {
          Value d_block =
              Add(builder,
                  Mul(builder, wave, IndexConstant(builder, kDBlocksPerWave)),
                  IndexConstant(builder, local_d_block));
          Value dimension = Add(
              builder, Mul(builder, d_block, IndexConstant(builder, kMfmaN)),
              lane16);
          llvm::SmallVector<Value, 4> v_elements;
          for (int64_t k_lane = 0; k_lane < kMfmaInputsPerLane; ++k_lane) {
            Value token =
                Add(builder,
                    IndexConstant(builder, source_wave * kTokensPerWave +
                                               token_half * kMfmaN + k_lane),
                    Mul(builder, lane_group,
                        IndexConstant(builder, kMfmaInputsPerLane)));
            Value swizzle =
                Mul(builder, Rem(builder, Div(builder, token, 4), 4),
                    IndexConstant(builder, 16));
            Value shared_dimension =
                mlir::arith::XOrIOp::create(builder, dimension, swizzle);
            Value shared_offset = Add(
                builder, Mul(builder, token, IndexConstant(builder, kHeadSize)),
                shared_dimension);
            v_elements.push_back(mlir::tensor::ExtractOp::create(
                builder, shared_values, mlir::ValueRange{shared_offset}));
          }
          Value v_fragment = vector_from(v_elements, v4_element);
          source_accumulators[local_d_block] =
              mma(p_fragment, v_fragment, source_accumulators[local_d_block]);
        }
      }
      for (int64_t local_d_block = 0; local_d_block < kDBlocksPerWave;
           ++local_d_block) {
        output_accumulators[local_d_block] = mlir::arith::AddFOp::create(
            builder, output_accumulators[local_d_block],
            mlir::arith::MulFOp::create(builder,
                                        source_accumulators[local_d_block],
                                        weights[source_wave]));
      }
    }

    for (int64_t local_d_block = 0; local_d_block < kDBlocksPerWave;
         ++local_d_block) {
      Value d_block = Add(
          builder, Mul(builder, wave, IndexConstant(builder, kDBlocksPerWave)),
          IndexConstant(builder, local_d_block));
      Value dimension =
          Add(builder, Mul(builder, d_block, IndexConstant(builder, kMfmaN)),
              lane16);
      Value result = mlir::arith::MulFOp::create(
          builder, output_accumulators[local_d_block], inverse_denominator);
      for (int64_t component = 0; component < kMfmaOutputsPerLane;
           ++component) {
        Value row =
            Add(builder, query_row_base, IndexConstant(builder, component));
        Value row_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, row,
            IndexConstant(builder, descriptor_.gqa_group));
        Value output_head =
            Add(builder,
                Mul(builder, kv_head,
                    IndexConstant(builder, descriptor_.gqa_group)),
                row);
        Value output_offset =
            Add(builder,
                Mul(builder,
                    Add(builder,
                        Mul(builder, sequence,
                            IndexConstant(builder, descriptor_.query_heads)),
                        output_head),
                    IndexConstant(builder, kHeadSize)),
                dimension);
        Value output_value = mlir::arith::TruncFOp::create(
            builder, element_type,
            ExtractVectorElement(builder, result, component));
        mlir::scf::IfOp store = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{output.getType()}, row_valid,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
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

    mlir::func::ReturnOp::create(builder, output);
    return absl::OkStatus();
  }

  FlyPagedAttentionDescriptor descriptor_;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

std::unique_ptr<MlirKernelEmitter> CreateFlyXTilePagedAttentionEmitter(
    const HloFusionAnalysis& analysis) {
  if (GetFlyPagedAttentionSegmentedProducerDescriptor(analysis).has_value()) {
    return CreateFlyXTilePagedAttentionOnlineEmitter(analysis);
  }
  if (GetFlyPagedAttentionSegmentedReducerDescriptor(analysis).has_value()) {
    return CreateFlyXTilePagedAttentionSegmentedReducerEmitter(analysis);
  }
  if (GetFlyPagedAttentionDescriptor(analysis)->max_context >
      kNumWaves * kTokensPerWave) {
    return CreateFlyXTilePagedAttentionOnlineEmitter(analysis);
  }
  return std::make_unique<FlyXTilePagedAttentionEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
