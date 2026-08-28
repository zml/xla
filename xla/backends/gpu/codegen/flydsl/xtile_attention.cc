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

#include "xla/backends/gpu/codegen/flydsl/xtile_attention.h"

#include <algorithm>
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/flydsl/attention_support.h"
#include "xla/backends/gpu/codegen/flydsl/compiler.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
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

Value F32Constant(mlir::ImplicitLocOpBuilder& builder, float value) {
  return mlir::arith::ConstantFloatOp::create(builder, builder.getF32Type(),
                                              llvm::APFloat(value));
}

Value ExtractVectorElement(mlir::ImplicitLocOpBuilder& builder, Value vector,
                           int64_t index) {
  return mlir::vector::ExtractOp::create(builder, vector,
                                         llvm::SmallVector<int64_t>{index});
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

class FlyXTileAttentionEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileAttentionEmitter(const HloFusionAnalysis& analysis) {
    descriptor_ = *GetFlyAttentionDescriptor(analysis);
    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    if (config.output_tiles_size() == 1 &&
        config.output_tiles(0).sizes_size() == 4) {
      block_m_ = config.output_tiles(0).sizes(1);
    }
    num_warps_ = config.num_warps();
    const int64_t q_tiles = descriptor_.sequence / block_m_;
    const int64_t blocks = descriptor_.batch * descriptor_.heads * q_tiles;
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
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
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
        "FlyXTileAttentionEmitter builds its module directly.");
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function) const {
    const bool has_separate_key_value =
        descriptor_.key_value_parameter != descriptor_.qkv_parameter;
    TF_RET_CHECK(entry_function.getNumArguments() ==
                 (has_separate_key_value ? 3 : 2));
    TF_RET_CHECK(
        (descriptor_.element_type == BF16 || descriptor_.element_type == F16) &&
        descriptor_.sequence % 64 == 0 && block_m_ >= 32 &&
        descriptor_.sequence % block_m_ == 0 && block_m_ / 32 == num_warps_ &&
        num_warps_ <= 8 && descriptor_.key_value_heads > 0 &&
        descriptor_.heads % descriptor_.key_value_heads == 0 &&
        descriptor_.key_value_sequence % 64 == 0 &&
        descriptor_.qkv_parameter->opcode() == HloOpcode::kParameter &&
        descriptor_.key_value_parameter->opcode() == HloOpcode::kParameter);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());
    Value query = entry_function.getArgument(
        descriptor_.qkv_parameter->parameter_number());
    Value key_value = entry_function.getArgument(
        descriptor_.key_value_parameter->parameter_number());
    Value output =
        entry_function.getArgument(entry_function.getNumArguments() - 1);
    mlir::Type element_type = descriptor_.element_type == BF16
                                  ? builder.getBF16Type()
                                  : builder.getF16Type();
    auto v4_element = mlir::VectorType::get({4}, element_type);
    auto v8_element = mlir::VectorType::get({8}, element_type);
    auto v4_f32 = mlir::VectorType::get({4}, builder.getF32Type());
    auto v16_f32 = mlir::VectorType::get({16}, builder.getF32Type());

    // gfx942 FlyDSL attention uses MFMA32x32x8 with four input elements per
    // lane and sixteen FP32 accumulator elements per lane.
    const std::string type_name =
        descriptor_.element_type == BF16 ? "bf16" : "f16";
    const std::string atom_assembly =
        "!fly.mma_atom<!fly_rocdl.cdna3.mfma<32x32x8, (" + type_name + "," +
        type_name + ")->f32>>";
    mlir::Type atom_type =
        mlir::parseType(atom_assembly, entry_function.getContext());
    TF_RET_CHECK(atom_type != nullptr);
    mlir::OperationState atom_state(entry_function.getLoc(),
                                    "fly.make_mma_atom");
    atom_state.addTypes(atom_type);
    Value atom = builder.create(atom_state)->getResult(0);
    auto mma = [&](Value lhs, Value rhs, Value accumulator) {
      mlir::OperationState state(entry_function.getLoc(),
                                 "fly.mma_atom_call_ssa");
      state.addOperands({atom, lhs, rhs, accumulator});
      state.addTypes(v16_f32);
      return builder.create(state)->getResult(0);
    };

    Value thread_id = EmitThreadId(builder, 0);
    Value block_id = EmitBlockId(builder, 0);
    Value wave_id = Div(builder, thread_id, 64);
    Value lane_id = Rem(builder, thread_id, 64);
    Value lane_mod_32 = Rem(builder, lane_id, 32);
    Value lane_div_32 = Div(builder, lane_id, 32);
    const int64_t q_tiles = descriptor_.sequence / block_m_;
    Value head = Rem(builder, block_id, descriptor_.heads);
    Value batch_q_tile = Div(builder, block_id, descriptor_.heads);
    Value q_tile = Rem(builder, batch_q_tile, q_tiles);
    Value batch = Div(builder, batch_q_tile, q_tiles);
    Value q_start = Mul(builder, q_tile, IndexConstant(builder, block_m_));
    Value q_row =
        Add(builder, q_start,
            Add(builder, Mul(builder, wave_id, IndexConstant(builder, 32)),
                lane_mod_32));

    Value key_value_head =
        Div(builder, head, descriptor_.heads / descriptor_.key_value_heads);
    const int64_t packed_token_stride =
        (descriptor_.heads + 2 * descriptor_.key_value_heads) *
        descriptor_.head_dimension;
    const int64_t query_plane_stride =
        descriptor_.heads * descriptor_.head_dimension;
    const int64_t key_value_plane_stride =
        descriptor_.key_value_heads * descriptor_.head_dimension;
    auto query_index = [&](Value token, Value column) {
      const int64_t token_stride =
          has_separate_key_value ? query_plane_stride : packed_token_stride;
      Value token_base = Mul(
          builder,
          Add(builder,
              Mul(builder, batch, IndexConstant(builder, descriptor_.sequence)),
              token),
          IndexConstant(builder, token_stride));
      Value plane_base = Mul(
          builder, head, IndexConstant(builder, descriptor_.head_dimension));
      return Add(builder, Add(builder, token_base, plane_base), column);
    };
    auto key_value_index = [&](Value token, int64_t plane, Value column) {
      const int64_t token_stride = has_separate_key_value
                                       ? 2 * key_value_plane_stride
                                       : packed_token_stride;
      Value token_base =
          Mul(builder,
              Add(builder,
                  Mul(builder, batch,
                      IndexConstant(builder, descriptor_.key_value_sequence)),
                  token),
              IndexConstant(builder, token_stride));
      const int64_t plane_offset =
          has_separate_key_value
              ? (plane - 1) * key_value_plane_stride
              : query_plane_stride + (plane - 1) * key_value_plane_stride;
      Value plane_base =
          Add(builder,
              Mul(builder, key_value_head,
                  IndexConstant(builder, descriptor_.head_dimension)),
              IndexConstant(builder, plane_offset));
      return Add(builder, Add(builder, token_base, plane_base), column);
    };

    constexpr int64_t kBlockN = 64;
    const int64_t kStride = descriptor_.head_dimension + 4;
    constexpr int64_t kVtStride = kBlockN + 2;
    const int64_t kSharedElements = kBlockN * kStride;
    const int64_t vSharedElements = descriptor_.head_dimension * kVtStride;
    const int64_t vSharedBase = kSharedElements;
    auto shared_type = mlir::RankedTensorType::get(
        {kSharedElements + vSharedElements}, element_type);
    Value shared = AllocateSharedOp::create(builder, shared_type);

    int64_t swizzle_group = descriptor_.head_dimension / 16;
    swizzle_group &= -swizzle_group;
    auto k_shared_index = [&](Value row, Value column) {
      Value swizzle = Mul(builder, Rem(builder, row, swizzle_group),
                          IndexConstant(builder, 16));
      Value physical_column =
          mlir::arith::XOrIOp::create(builder, column, swizzle);
      return Add(builder, Mul(builder, row, IndexConstant(builder, kStride)),
                 physical_column);
    };

    auto zero_vector = [&](mlir::VectorType type) {
      Value zero = F32Constant(builder, 0.0f);
      return mlir::vector::BroadcastOp::create(builder, type, zero).getResult();
    };
    std::vector<Value> q_packs;
    q_packs.reserve(descriptor_.head_dimension / 8);
    const Shape& query_shape = descriptor_.qkv_parameter->shape();
    const bool query_is_f32 = query_shape.element_type() == F32;
    const int64_t query_split_k =
        query_shape.dimensions_size() == 3 ? query_shape.dimensions(0) : 1;
    const int64_t query_split_stride =
        descriptor_.batch * descriptor_.sequence * descriptor_.heads *
        descriptor_.head_dimension;
    for (int64_t k_step = 0; k_step < descriptor_.head_dimension / 8;
         ++k_step) {
      Value column = Add(builder, IndexConstant(builder, k_step * 8),
                         Mul(builder, lane_div_32, IndexConstant(builder, 4)));
      Value query_offset = query_index(q_row, column);
      Value q_pack =
          EmitBufferLoad(builder, entry_function.getLoc(), query, query_offset,
                         query_is_f32 ? v4_f32 : v4_element);
      for (int64_t split = 1; split < query_split_k; ++split) {
        Value split_offset =
            Add(builder, query_offset,
                IndexConstant(builder, split * query_split_stride));
        Value partial = EmitBufferLoad(builder, entry_function.getLoc(), query,
                                       split_offset, v4_f32);
        q_pack = mlir::arith::AddFOp::create(builder, q_pack, partial);
      }
      if (query_is_f32) {
        q_pack = mlir::arith::TruncFOp::create(builder, v4_element, q_pack);
      }
      q_packs.push_back(q_pack);
    }

    std::vector<Value> o_accumulators(descriptor_.head_dimension / 32,
                                      zero_vector(v16_f32));
    Value m_running = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(),
        llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(),
                              /*negative=*/true));
    Value l_running = F32Constant(builder, 0.0f);
    Value log2_scale = F32Constant(
        builder, static_cast<float>(descriptor_.scale * 1.4426950408889634));

    const int64_t block_threads = num_warps_ * 64;
    const int64_t vectors_per_tile = kBlockN * descriptor_.head_dimension / 8;
    TF_RET_CHECK(vectors_per_tile % block_threads == 0);
    const int64_t vectors_per_thread = vectors_per_tile / block_threads;

    for (int64_t kv_block = 0;
         kv_block < descriptor_.key_value_sequence / kBlockN; ++kv_block) {
      if (kv_block != 0) {
        shared =
            SyncThreadsOp::create(builder, mlir::TypeRange{shared_type}, shared)
                .getResult(0);
      }

      // Cooperative 128-bit global loads. K is stored row-major with the same
      // XOR16 swizzle as FlyDSL; V is transposed into [D,64+2] LDS so MFMA2
      // consumes a contiguous four-element K fragment.
      for (int64_t copy = 0; copy < vectors_per_thread; ++copy) {
        Value flat = Add(builder, thread_id,
                         IndexConstant(builder, copy * block_threads));
        Value row = Div(builder, flat, descriptor_.head_dimension / 8);
        Value column =
            Mul(builder, Rem(builder, flat, descriptor_.head_dimension / 8),
                IndexConstant(builder, 8));
        Value token =
            Add(builder, IndexConstant(builder, kv_block * kBlockN), row);
        Value k_vector = EmitBufferLoad(
            builder, entry_function.getLoc(), key_value,
            key_value_index(token, /*plane=*/1, column), v8_element);
        shared = mlir::vector::TransferWriteOp::create(
                     builder, k_vector, shared,
                     mlir::ValueRange{k_shared_index(row, column)},
                     llvm::ArrayRef<bool>{true})
                     .getResult();

        Value v_vector = EmitBufferLoad(
            builder, entry_function.getLoc(), key_value,
            key_value_index(token, /*plane=*/2, column), v8_element);
        for (int64_t element = 0; element < 8; ++element) {
          Value v_index =
              Add(builder, IndexConstant(builder, vSharedBase),
                  Add(builder,
                      Mul(builder,
                          Add(builder, column, IndexConstant(builder, element)),
                          IndexConstant(builder, kVtStride)),
                      row));
          shared = mlir::tensor::InsertOp::create(
              builder, ExtractVectorElement(builder, v_vector, element), shared,
              mlir::ValueRange{v_index});
        }
      }
      shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{shared_type}, shared)
              .getResult(0);

      Value scores_lo = zero_vector(v16_f32);
      Value scores_hi = zero_vector(v16_f32);
      for (int64_t k_step = 0; k_step < q_packs.size(); ++k_step) {
        Value column =
            Add(builder, IndexConstant(builder, k_step * 8),
                Mul(builder, lane_div_32, IndexConstant(builder, 4)));
        Value k_lo_index = k_shared_index(lane_mod_32, column);
        Value k_hi_index = k_shared_index(
            Add(builder, lane_mod_32, IndexConstant(builder, 32)), column);
        Value k_lo = mlir::vector::TransferReadOp::create(
            builder, v4_element, shared, mlir::ValueRange{k_lo_index},
            /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
        Value k_hi = mlir::vector::TransferReadOp::create(
            builder, v4_element, shared, mlir::ValueRange{k_hi_index},
            /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
        scores_lo = mma(k_lo, q_packs[k_step], scores_lo);
        scores_hi = mma(k_hi, q_packs[k_step], scores_hi);
      }

      llvm::SmallVector<Value, 16> score_values_lo;
      llvm::SmallVector<Value, 16> score_values_hi;
      Value local_max = m_running;
      for (int64_t element = 0; element < 16; ++element) {
        Value lo = ExtractVectorElement(builder, scores_lo, element);
        Value hi = ExtractVectorElement(builder, scores_hi, element);
        if (descriptor_.causal) {
          // MFMA32 distributes four adjacent key columns to each lane for
          // each of four K steps. The second accumulator covers columns
          // 32..63. Mask in registers before the online-softmax reduction so
          // the score matrix and predicate are never materialized.
          Value key_lo =
              Add(builder,
                  IndexConstant(builder, kv_block * kBlockN +
                                             (element / 4) * 8 + element % 4),
                  Mul(builder, lane_div_32, IndexConstant(builder, 4)));
          Value key_hi = Add(builder, key_lo, IndexConstant(builder, 32));
          Value lo_valid = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, key_lo, q_row);
          Value hi_valid = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, key_hi, q_row);
          Value negative_infinity = mlir::arith::ConstantFloatOp::create(
              builder, builder.getF32Type(),
              llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(),
                                    /*negative=*/true));
          lo = mlir::arith::SelectOp::create(builder, lo_valid, lo,
                                             negative_infinity);
          hi = mlir::arith::SelectOp::create(builder, hi_valid, hi,
                                             negative_infinity);
        }
        score_values_lo.push_back(lo);
        score_values_hi.push_back(hi);
        local_max = mlir::arith::MaxNumFOp::create(builder, local_max, lo);
        local_max = mlir::arith::MaxNumFOp::create(builder, local_max, hi);
      }
      Value peer_max = mlir::gpu::ShuffleOp::create(
                           builder, local_max, /*offset=*/32,
                           /*width=*/64, mlir::gpu::ShuffleMode::XOR)
                           .getShuffleResult();
      Value row_max =
          mlir::arith::MaxNumFOp::create(builder, local_max, peer_max);
      Value m_new = mlir::arith::MaxNumFOp::create(builder, m_running, row_max);
      Value correction_exponent = mlir::arith::MulFOp::create(
          builder, mlir::arith::SubFOp::create(builder, m_running, m_new),
          log2_scale);
      Value correction = mlir::ROCDL::ROCDLExp2::create(
          builder, builder.getF32Type(), correction_exponent);
      Value correction_vector =
          mlir::vector::BroadcastOp::create(builder, v16_f32, correction);
      for (Value& accumulator : o_accumulators) {
        accumulator = mlir::arith::MulFOp::create(builder, accumulator,
                                                  correction_vector);
      }

      Value negative_scaled_max = mlir::arith::NegFOp::create(
          builder, mlir::arith::MulFOp::create(builder, m_new, log2_scale));
      llvm::SmallVector<Value, 16> probabilities_lo;
      llvm::SmallVector<Value, 16> probabilities_hi;
      Value local_sum = F32Constant(builder, 0.0f);
      for (int64_t element = 0; element < 16; ++element) {
        Value lo_exponent = mlir::arith::AddFOp::create(
            builder,
            mlir::arith::MulFOp::create(builder, score_values_lo[element],
                                        log2_scale),
            negative_scaled_max);
        Value hi_exponent = mlir::arith::AddFOp::create(
            builder,
            mlir::arith::MulFOp::create(builder, score_values_hi[element],
                                        log2_scale),
            negative_scaled_max);
        Value p_lo = mlir::ROCDL::ROCDLExp2::create(
            builder, builder.getF32Type(), lo_exponent);
        Value p_hi = mlir::ROCDL::ROCDLExp2::create(
            builder, builder.getF32Type(), hi_exponent);
        probabilities_lo.push_back(p_lo);
        probabilities_hi.push_back(p_hi);
        local_sum = mlir::arith::AddFOp::create(builder, local_sum, p_lo);
        local_sum = mlir::arith::AddFOp::create(builder, local_sum, p_hi);
      }
      Value peer_sum = mlir::gpu::ShuffleOp::create(
                           builder, local_sum, /*offset=*/32,
                           /*width=*/64, mlir::gpu::ShuffleMode::XOR)
                           .getShuffleResult();
      Value tile_sum =
          mlir::arith::AddFOp::create(builder, local_sum, peer_sum);
      Value l_new = mlir::arith::AddFOp::create(
          builder, mlir::arith::MulFOp::create(builder, correction, l_running),
          tile_sum);

      llvm::SmallVector<Value, 4> p_packs_lo;
      llvm::SmallVector<Value, 4> p_packs_hi;
      for (int64_t p_step = 0; p_step < 4; ++p_step) {
        llvm::SmallVector<Value, 4> lo_elements;
        llvm::SmallVector<Value, 4> hi_elements;
        for (int64_t element = 0; element < 4; ++element) {
          lo_elements.push_back(probabilities_lo[p_step * 4 + element]);
          hi_elements.push_back(probabilities_hi[p_step * 4 + element]);
        }
        Value lo_f32 =
            mlir::vector::FromElementsOp::create(builder, v4_f32, lo_elements);
        Value hi_f32 =
            mlir::vector::FromElementsOp::create(builder, v4_f32, hi_elements);
        p_packs_lo.push_back(
            mlir::arith::TruncFOp::create(builder, v4_element, lo_f32));
        p_packs_hi.push_back(
            mlir::arith::TruncFOp::create(builder, v4_element, hi_f32));
      }

      for (int64_t d_chunk = 0; d_chunk < o_accumulators.size(); ++d_chunk) {
        for (int64_t p_step = 0; p_step < 4; ++p_step) {
          Value d_position =
              Add(builder, IndexConstant(builder, d_chunk * 32), lane_mod_32);
          Value k_column =
              Add(builder, IndexConstant(builder, p_step * 8),
                  Mul(builder, lane_div_32, IndexConstant(builder, 4)));
          Value v_lo_index = Add(
              builder, IndexConstant(builder, vSharedBase),
              Add(builder,
                  Mul(builder, d_position, IndexConstant(builder, kVtStride)),
                  k_column));
          Value v_hi_index =
              Add(builder, v_lo_index, IndexConstant(builder, 32));
          Value v_lo = mlir::vector::TransferReadOp::create(
              builder, v4_element, shared, mlir::ValueRange{v_lo_index},
              /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
          Value v_hi = mlir::vector::TransferReadOp::create(
              builder, v4_element, shared, mlir::ValueRange{v_hi_index},
              /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
          o_accumulators[d_chunk] =
              mma(v_lo, p_packs_lo[p_step], o_accumulators[d_chunk]);
          o_accumulators[d_chunk] =
              mma(v_hi, p_packs_hi[p_step], o_accumulators[d_chunk]);
        }
      }
      m_running = m_new;
      l_running = l_new;
    }

    Value inverse_sum =
        mlir::ROCDL::ROCDLRcp::create(builder, builder.getF32Type(), l_running);
    Value inverse_vector =
        mlir::vector::BroadcastOp::create(builder, v16_f32, inverse_sum);
    for (int64_t d_chunk = 0; d_chunk < o_accumulators.size(); ++d_chunk) {
      Value normalized = mlir::arith::MulFOp::create(
          builder, o_accumulators[d_chunk], inverse_vector);
      for (int64_t group = 0; group < 4; ++group) {
        llvm::SmallVector<int64_t, 4> mask = {group * 4, group * 4 + 1,
                                              group * 4 + 2, group * 4 + 3};
        Value output_f32 = mlir::vector::ShuffleOp::create(
            builder, v4_f32, normalized, normalized, mask);
        Value output_vector =
            mlir::arith::TruncFOp::create(builder, v4_element, output_f32);
        Value d_column =
            Add(builder, IndexConstant(builder, d_chunk * 32 + group * 8),
                Mul(builder, lane_div_32, IndexConstant(builder, 4)));
        Value output_index = Add(
            builder,
            Mul(builder,
                Add(builder,
                    Mul(builder,
                        Add(builder,
                            Mul(builder, batch,
                                IndexConstant(builder, descriptor_.sequence)),
                            q_row),
                        IndexConstant(builder, descriptor_.heads)),
                    head),
                IndexConstant(builder, descriptor_.head_dimension)),
            d_column);
        output = EmitBufferStore(builder, entry_function.getLoc(),
                                 output_vector, output, output_index);
      }
    }

    mlir::func::ReturnOp::create(builder, output);
    return absl::OkStatus();
  }

  FlyAttentionDescriptor descriptor_;
  int64_t block_m_ = 128;
  int64_t num_warps_ = 4;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileAttentionEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileAttentionEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
