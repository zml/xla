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

#include "xla/backends/gpu/codegen/flydsl/xtile_gemm.h"

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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla_data.pb.h"

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

Value ReadFirstLaneIndex(mlir::ImplicitLocOpBuilder& builder, Value value) {
  Value as_i32 =
      mlir::arith::IndexCastOp::create(builder, builder.getI32Type(), value);
  Value uniform = mlir::ROCDL::ReadfirstlaneOp::create(
      builder, builder.getI32Type(), as_i32);
  return mlir::arith::IndexCastOp::create(builder, builder.getIndexType(),
                                          uniform);
}

Value SwizzleXor16(mlir::ImplicitLocOpBuilder& builder, Value row, Value column,
                   int64_t stage_k) {
  // FlyDSL's swizzle_xor16 operates in bytes:
  //   col_bytes ^ ((row % (stage_bytes / 16)) * 16).
  // All columns here are BF16 element offsets, so express the same transform
  // in elements.
  Value swizzle =
      Mul(builder, Rem(builder, row, stage_k / 8), IndexConstant(builder, 8));
  return mlir::arith::XOrIOp::create(builder, column, swizzle);
}

Value ExtractTensor(mlir::ImplicitLocOpBuilder& builder, Value tensor,
                    Value row, Value column) {
  llvm::SmallVector<Value, 2> indices = {row, column};
  return mlir::tensor::ExtractOp::create(builder, tensor, indices);
}

void ScheduleGroup(mlir::ImplicitLocOpBuilder& builder,
                   mlir::ROCDL::SchedGroupMask mask, int64_t count) {
  if (count > 0) {
    mlir::ROCDL::SchedGroupBarrier::create(
        builder, mask, static_cast<uint32_t>(count), /*groupId=*/0);
  }
}

// FlyDSL code generation backend for XLA's xTile block-level GEMM schedule.
// The xTile contract supplies output tile sizes and launch parameters; this
// emitter lowers that scheduled GEMM directly to Fly/FlyROCDL operations.
class FlyXTileGemmEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileGemmEmitter(const HloFusionAnalysis& analysis) {
    const BlockLevelFusionConfig& xtile_config =
        analysis.fusion_backend_config().block_level_fusion_config();
    if (xtile_config.output_tiles_size() == 1 &&
        xtile_config.output_tiles(0).sizes_size() == 2) {
      block_m_ = xtile_config.output_tiles(0).sizes(0);
      block_n_ = xtile_config.output_tiles(0).sizes(1);
      num_warps_ = xtile_config.num_warps();
    }
    const FusionBackendConfig& fusion_config = analysis.fusion_backend_config();
    if (fusion_config.has_fly_gemm_config()) {
      use_mfma_32_ = fusion_config.fly_gemm_config().mfma_atom() ==
                     FlyGemmConfig::FLY_MFMA_32X32X8;
      prefetch_rhs_ = fusion_config.fly_gemm_config().prefetch_rhs();
      stage_output_ = fusion_config.fly_gemm_config().stage_output();
      schedule_instructions_ =
          fusion_config.fly_gemm_config().schedule_instructions();
      stage_rhs_ = fusion_config.fly_gemm_config().stage_rhs();
      async_lhs_ = fusion_config.fly_gemm_config().async_lhs();
    }
    const HloInstruction* dot = &analysis.fusion_root(0).instruction();
    m_ = dot->shape().dimensions(0);
    n_ = dot->shape().dimensions(1);
    k_ = dot->operand(0)->shape().dimensions(1);
    rhs_k_contiguous_ =
        dot->operand(1)->shape().has_layout() &&
        dot->operand(1)->shape().layout().minor_to_major(0) == 0;
    stage_k_ = k_ % 32 == 0 ? 32 : 16;
    absl::StatusOr<Tile> contraction_tile = dot->backend_config<Tile>();
    if (contraction_tile.ok() && contraction_tile->sizes_size() == 1 &&
        contraction_tile->sizes(0) >= 16 &&
        contraction_tile->sizes(0) % 16 == 0 &&
        k_ % contraction_tile->sizes(0) == 0) {
      stage_k_ = contraction_tile->sizes(0);
    }
    const int64_t blocks =
        ((m_ + block_m_ - 1) / block_m_) * ((n_ + block_n_ - 1) / block_n_);
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
        "FlyGemmEmitter builds its module directly.");
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function) const {
    TF_RET_CHECK(entry_function.getNumArguments() == 3);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());

    Value lhs = entry_function.getArgument(0);
    Value rhs = entry_function.getArgument(1);
    Value output = entry_function.getArgument(2);
    if (stage_rhs_) {
      constexpr int64_t kCopyElements = 2;
      const int64_t block_threads = num_warps_ * 64;
      TF_RET_CHECK(stage_k_ >= 32 && stage_k_ % 32 == 0 &&
                   block_m_ % 16 == 0 && block_n_ % 16 == 0 &&
                   !use_mfma_32_);
      TF_RET_CHECK((block_m_ * stage_k_ / kCopyElements) % block_threads ==
                       0 &&
                   (block_n_ * stage_k_ / kCopyElements) % block_threads ==
                       0);
      TF_RET_CHECK(2 * (block_m_ + block_n_) * stage_k_ * sizeof(uint16_t) <=
                   64 * 1024);
      TF_RET_CHECK(rhs_k_contiguous_);
      TF_RET_CHECK(m_ % block_m_ == 0 && n_ % block_n_ == 0);
    }
    if (async_lhs_) {
      TF_RET_CHECK(stage_k_ == 64 && block_m_ == 128 && block_n_ == 128 &&
                   (num_warps_ == 4 || num_warps_ == 8 ||
                    num_warps_ == 16) &&
                   !use_mfma_32_ && !stage_rhs_ && prefetch_rhs_ &&
                   rhs_k_contiguous_);
      TF_RET_CHECK(m_ % block_m_ == 0 && n_ % block_n_ == 0);
    }

    Value thread_id = EmitThreadId(builder, 0);
    Value block_id = EmitBlockId(builder, 0);
    Value wave_id = Div(builder, thread_id, 64);
    Value lane_id = Rem(builder, thread_id, 64);
    Value direct_copy_wave_offset;
    if (stage_rhs_ || async_lhs_) {
      direct_copy_wave_offset = ReadFirstLaneIndex(
          builder, Mul(builder, wave_id, IndexConstant(builder, 128)));
    }
    const bool use_mfma_32 = use_mfma_32_;
    const bool paired_mfma = stage_rhs_ || async_lhs_;
    const int64_t atom_m = use_mfma_32 ? 32 : 16;
    const int64_t atom_k = use_mfma_32 ? 8 : 16;
    const int64_t accumulator_elements = use_mfma_32 ? 16 : 4;
    Value lane_axis = Rem(builder, lane_id, atom_m);
    Value lane_group = Div(builder, lane_id, atom_m);

    const int64_t grid_n = (n_ + block_n_ - 1) / block_n_;
    Value block_m = Div(builder, block_id, grid_n);
    Value block_n = Rem(builder, block_id, grid_n);
    Value block_m_base =
        Mul(builder, block_m, IndexConstant(builder, block_m_));
    Value block_n_base =
        Mul(builder, block_n, IndexConstant(builder, block_n_));

    mlir::Type mma_atom_type = mlir::parseType(
        use_mfma_32 ? "!fly.mma_atom<!fly_rocdl.cdna3.mfma<32x32x8, "
                      "(bf16,bf16)->f32>>"
                    : "!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x16, "
                      "(bf16,bf16)->f32>>",
        entry_function.getContext());
    TF_RET_CHECK(mma_atom_type != nullptr);
    mlir::OperationState atom_state(entry_function.getLoc(),
                                    "fly.make_mma_atom");
    atom_state.addTypes(mma_atom_type);
    Value mma_atom = builder.create(atom_state)->getResult(0);

    mlir::VectorType input_vector_type =
        mlir::VectorType::get({4}, builder.getBF16Type());
    mlir::VectorType staged_input_vector_type =
        mlir::VectorType::get({8}, builder.getBF16Type());
    mlir::VectorType accumulator_type =
        mlir::VectorType::get({accumulator_elements}, builder.getF32Type());
    auto emit_buffer_load = [&](Value source, Value source_linear_index,
                                mlir::VectorType result_type) {
      mlir::OperationState load_state(entry_function.getLoc(),
                                      "xla_gpu.buffer_load");
      load_state.addOperands({source, source_linear_index});
      load_state.addTypes(result_type);
      return builder.create(load_state)->getResult(0);
    };
    Value zero = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(), llvm::APFloat(0.0f));
    Value zero_accumulator =
        mlir::vector::BroadcastOp::create(builder, accumulator_type, zero);
    const int64_t lhs_shared_elements = 2 * block_m_ * stage_k_;
    const int64_t rhs_shared_elements =
        stage_rhs_ ? 2 * block_n_ * stage_k_ : 0;
    const int64_t output_shared_elements =
        stage_output_ ? block_m_ * block_n_ : 0;
    auto lhs_shared_type = mlir::RankedTensorType::get(
        {std::max(lhs_shared_elements + rhs_shared_elements,
                  output_shared_elements)},
        builder.getBF16Type());
    Value lhs_shared = AllocateSharedOp::create(builder, lhs_shared_type);

    const int64_t tile_rows = block_m_ / atom_m;
    const int64_t tile_columns = block_n_ / atom_m;
    TF_RET_CHECK(tile_rows * tile_columns >= num_warps_);
    int64_t wave_grid_rows = 1;
    int64_t wave_grid_columns = num_warps_;
    int64_t best_fragment_loads =
        tile_rows * wave_grid_columns + tile_columns * wave_grid_rows;
    for (int64_t candidate_rows = 1; candidate_rows <= num_warps_;
         candidate_rows *= 2) {
      const int64_t candidate_columns = num_warps_ / candidate_rows;
      if (num_warps_ % candidate_rows != 0 || tile_rows % candidate_rows != 0 ||
          tile_columns % candidate_columns != 0) {
        continue;
      }
      const int64_t fragment_loads =
          tile_rows * candidate_columns + tile_columns * candidate_rows;
      if (fragment_loads < best_fragment_loads) {
        wave_grid_rows = candidate_rows;
        wave_grid_columns = candidate_columns;
        best_fragment_loads = fragment_loads;
      }
    }
    TF_RET_CHECK(tile_rows % wave_grid_rows == 0);
    TF_RET_CHECK(tile_columns % wave_grid_columns == 0);

    const int64_t wave_tile_rows = tile_rows / wave_grid_rows;
    const int64_t wave_tile_columns = tile_columns / wave_grid_columns;
    const int64_t accumulators_per_wave = wave_tile_rows * wave_tile_columns;
    Value wave_row = Div(builder, wave_id, wave_grid_columns);
    Value wave_column = Rem(builder, wave_id, wave_grid_columns);
    Value wave_row_offset =
        Mul(builder, wave_row, IndexConstant(builder, wave_tile_rows * atom_m));
    Value wave_column_offset =
        Mul(builder, wave_column,
            IndexConstant(builder, wave_tile_columns * atom_m));
    Value wave_row_base = Add(builder, block_m_base, wave_row_offset);
    Value wave_column_base = Add(builder, block_n_base, wave_column_offset);

    llvm::SmallVector<Value> initial_accumulators(accumulators_per_wave,
                                                  zero_accumulator);
    Value lower_bound = IndexConstant(builder, 0);
    Value upper_bound = IndexConstant(builder, k_);
    Value step = IndexConstant(builder, stage_k_);
    constexpr int64_t kLoadVectorWidth = 8;
    auto load_vector_type =
        mlir::VectorType::get({kLoadVectorWidth}, builder.getBF16Type());
    Value zero_load_vector = mlir::arith::ConstantOp::create(
        builder, load_vector_type, builder.getZeroAttr(load_vector_type));
    Value zero_input_vector = mlir::arith::ConstantOp::create(
        builder, input_vector_type, builder.getZeroAttr(input_vector_type));
    Value zero_staged_input_vector = mlir::arith::ConstantOp::create(
        builder, staged_input_vector_type,
        builder.getZeroAttr(staged_input_vector_type));
    Value load_step = IndexConstant(builder, num_warps_ * 64);
    const int64_t lhs_vectors_per_row = stage_k_ / kLoadVectorWidth;
    auto emit_lhs_register_stage = [&](Value destination, Value global_k,
                                       Value shared_stage) {
      mlir::scf::ForOp load_loop = mlir::scf::ForOp::create(
          builder, thread_id,
          IndexConstant(builder, block_m_ * lhs_vectors_per_row), load_step,
          mlir::ValueRange{destination},
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard load_guard(builder);
        builder.setInsertionPointToStart(load_loop.getBody());
        Value vector_index = load_loop.getInductionVar();
        Value shared_row = Div(builder, vector_index, lhs_vectors_per_row);
        Value shared_k =
            Mul(builder, Rem(builder, vector_index, lhs_vectors_per_row),
                IndexConstant(builder, kLoadVectorWidth));
        // Iterate over contiguous physical LDS addresses and invert the XOR
        // swizzle to find the corresponding logical global-memory column.
        Value logical_k = SwizzleXor16(builder, shared_row, shared_k, stage_k_);
        Value global_row = Add(builder, block_m_base, shared_row);
        Value loaded;
        if (m_ % block_m_ == 0) {
          if (async_lhs_) {
            Value source_linear_index = Add(
                builder, Mul(builder, global_row, IndexConstant(builder, k_)),
                Add(builder, global_k, logical_k));
            loaded =
                emit_buffer_load(lhs, source_linear_index, load_vector_type);
          } else {
            loaded = mlir::vector::TransferReadOp::create(
                builder, load_vector_type, lhs,
                mlir::ValueRange{global_row,
                                 Add(builder, global_k, logical_k)},
                /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
          }
        } else {
          Value row_in_bounds = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, global_row,
              IndexConstant(builder, m_));
          mlir::scf::IfOp load = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{load_vector_type}, row_in_bounds,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard if_guard(builder);
            builder.setInsertionPointToStart(load.thenBlock());
            Value in_bounds = mlir::vector::TransferReadOp::create(
                builder, load_vector_type, lhs,
                mlir::ValueRange{global_row, Add(builder, global_k, logical_k)},
                /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
            mlir::scf::YieldOp::create(builder, in_bounds);

            builder.setInsertionPointToStart(load.elseBlock());
            mlir::scf::YieldOp::create(builder, zero_load_vector);
          }
          builder.setInsertionPointAfter(load);
          loaded = load.getResult(0);
        }
        Value written = load_loop.getRegionIterArg(0);
        Value stage_base = Mul(builder, shared_stage,
                               IndexConstant(builder, block_m_ * stage_k_));
        Value row_base =
            Mul(builder, shared_row, IndexConstant(builder, stage_k_));
        written = mlir::vector::TransferWriteOp::create(
                      builder, loaded, written,
                      mlir::ValueRange{Add(
                          builder, Add(builder, stage_base, row_base),
                          shared_k)},
                      llvm::ArrayRef<bool>{true})
                      .getResult();
        mlir::scf::YieldOp::create(builder, written);
      }
      builder.setInsertionPointAfter(load_loop);
      return load_loop.getResult(0);
    };

    auto emit_lhs_direct_stage = [&](Value destination, Value global_k,
                                     Value shared_stage) {
      constexpr int64_t kCopyElements = 2;
      const int64_t copies = block_m_ * stage_k_ / kCopyElements;
      const int64_t block_threads = num_warps_ * 64;
      const int64_t copies_per_thread = copies / block_threads;
      Value copied = destination;
      for (int64_t copy = 0; copy < copies_per_thread; ++copy) {
        Value copy_index =
            Add(builder, thread_id,
                IndexConstant(builder, copy * block_threads));
        Value physical_linear =
            Mul(builder, copy_index, IndexConstant(builder, kCopyElements));
        Value shared_row = Div(builder, physical_linear, stage_k_);
        Value shared_k = Rem(builder, physical_linear, stage_k_);
        Value logical_k =
            SwizzleXor16(builder, shared_row, shared_k, stage_k_);
        Value destination_wave_base = Add(
            builder,
            Mul(builder, shared_stage,
                IndexConstant(builder, block_m_ * stage_k_)),
            Add(builder, direct_copy_wave_offset,
                IndexConstant(builder,
                              copy * block_threads * kCopyElements)));
        mlir::OperationState copy_state(
            entry_function.getLoc(),
            "xla_gpu.async_copy_global_to_shared");
        Value source_linear_index = Add(
            builder,
            Mul(builder, Add(builder, block_m_base, shared_row),
                IndexConstant(builder, k_)),
            Add(builder, global_k, logical_k));
        copy_state.addOperands({lhs, source_linear_index,
                                copied, destination_wave_base});
        copy_state.addTypes(destination.getType());
        copy_state.addAttribute("copy_bytes", builder.getI32IntegerAttr(4));
        copied = builder.create(copy_state)->getResult(0);
      }
      return copied;
    };

    auto emit_lhs_stage = [&](Value destination, Value global_k,
                              Value shared_stage) {
      if ((stage_rhs_ || async_lhs_) && m_ % block_m_ == 0) {
        return emit_lhs_direct_stage(destination, global_k, shared_stage);
      }
      return emit_lhs_register_stage(destination, global_k, shared_stage);
    };

    auto emit_rhs_lds_stage = [&](Value destination, Value global_k,
                                  Value shared_stage) {
      const int64_t rhs_shared_base = 2 * block_m_ * stage_k_;
      if (rhs_k_contiguous_) {
        constexpr int64_t kCopyElements = 2;
        const int64_t copies = block_n_ * stage_k_ / kCopyElements;
        const int64_t block_threads = num_warps_ * 64;
        const int64_t copies_per_thread = copies / block_threads;
        Value copied = destination;
        for (int64_t copy = 0; copy < copies_per_thread; ++copy) {
          Value copy_index =
              Add(builder, thread_id,
                  IndexConstant(builder, copy * block_threads));
          Value physical_linear =
              Mul(builder, copy_index, IndexConstant(builder, kCopyElements));
          Value shared_n = Div(builder, physical_linear, stage_k_);
          Value shared_k = Rem(builder, physical_linear, stage_k_);
          Value logical_k =
              SwizzleXor16(builder, shared_n, shared_k, stage_k_);
          Value destination_wave_base = Add(
              builder, IndexConstant(builder, rhs_shared_base),
              Add(builder,
                  Mul(builder, shared_stage,
                      IndexConstant(builder, block_n_ * stage_k_)),
                  Add(builder, direct_copy_wave_offset,
                      IndexConstant(builder,
                                    copy * block_threads * kCopyElements))));
          mlir::OperationState copy_state(
              entry_function.getLoc(),
              "xla_gpu.async_copy_global_to_shared");
          Value source_linear_index = Add(
              builder,
              Mul(builder, Add(builder, block_n_base, shared_n),
                  IndexConstant(builder, k_)),
              Add(builder, global_k, logical_k));
          copy_state.addOperands({rhs, source_linear_index,
                                  copied, destination_wave_base});
          copy_state.addTypes(destination.getType());
          copy_state.addAttribute("copy_bytes", builder.getI32IntegerAttr(4));
          copied = builder.create(copy_state)->getResult(0);
        }
        return copied;
      }
      const int64_t rhs_vectors = block_n_ * stage_k_ / kLoadVectorWidth;
      mlir::scf::ForOp load_loop = mlir::scf::ForOp::create(
          builder, thread_id, IndexConstant(builder, rhs_vectors), load_step,
          mlir::ValueRange{destination},
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard load_guard(builder);
        builder.setInsertionPointToStart(load_loop.getBody());
        Value vector_index = load_loop.getInductionVar();
        Value shared_n;
        Value shared_k;
        Value loaded;
        if (rhs_k_contiguous_) {
          const int64_t vectors_per_n = stage_k_ / kLoadVectorWidth;
          shared_n = Div(builder, vector_index, vectors_per_n);
          shared_k = Mul(builder, Rem(builder, vector_index, vectors_per_n),
                         IndexConstant(builder, kLoadVectorWidth));
          mlir::AffineMap k_vector_map = mlir::AffineMap::get(
              /*dimCount=*/2, /*symbolCount=*/0, builder.getAffineDimExpr(0),
              builder.getContext());
          loaded = mlir::vector::TransferReadOp::create(
              builder, load_vector_type, rhs,
              mlir::ValueRange{Add(builder, global_k, shared_k),
                               Add(builder, block_n_base, shared_n)},
              /*padding=*/std::nullopt, k_vector_map,
              llvm::ArrayRef<bool>{true});
        } else {
          const int64_t vectors_per_k = block_n_ / kLoadVectorWidth;
          shared_k = Div(builder, vector_index, vectors_per_k);
          Value shared_n_vector = Rem(builder, vector_index, vectors_per_k);
          shared_n = Mul(builder, shared_n_vector,
                         IndexConstant(builder, kLoadVectorWidth));
          loaded = mlir::vector::TransferReadOp::create(
              builder, load_vector_type, rhs,
              mlir::ValueRange{Add(builder, global_k, shared_k),
                               Add(builder, block_n_base, shared_n)},
              /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
        }

        Value written = load_loop.getRegionIterArg(0);
        for (int64_t element = 0; element < kLoadVectorWidth; ++element) {
          Value scalar = mlir::vector::ExtractOp::create(
              builder, loaded, llvm::SmallVector<int64_t>{element});
          Value element_n =
              rhs_k_contiguous_
                  ? shared_n
                  : Add(builder, shared_n, IndexConstant(builder, element));
          Value element_k =
              rhs_k_contiguous_
                  ? Add(builder, shared_k, IndexConstant(builder, element))
                  : shared_k;
          Value swizzled_k =
              SwizzleXor16(builder, element_n, element_k, stage_k_);
          Value stage_base = Mul(builder, shared_stage,
                                 IndexConstant(builder, block_n_ * stage_k_));
          Value row_base =
              Mul(builder, element_n, IndexConstant(builder, stage_k_));
          written = mlir::tensor::InsertOp::create(
              builder, scalar, written,
              Add(builder, IndexConstant(builder, rhs_shared_base),
                  Add(builder, stage_base,
                      Add(builder, row_base, swizzled_k))));
        }
        mlir::scf::YieldOp::create(builder, written);
      }
      builder.setInsertionPointAfter(load_loop);
      return load_loop.getResult(0);
    };

    auto emit_rhs_group = [&](Value global_k, int64_t k_offset) {
      llvm::SmallVector<Value> fragments;
      fragments.reserve(wave_tile_columns);
      Value group_k =
          Add(builder,
              Mul(builder, lane_group,
                  IndexConstant(builder, async_lhs_ ? 8 : 4)),
              IndexConstant(builder, k_offset));
      mlir::VectorType rhs_vector_type =
          async_lhs_ ? staged_input_vector_type : input_vector_type;
      for (int64_t tile_column = 0; tile_column < wave_tile_columns;
           ++tile_column) {
        Value column = Add(builder, wave_column_offset,
                           Add(builder, lane_axis,
                               IndexConstant(builder, tile_column * atom_m)));
        Value global_column = Add(builder, block_n_base, column);
        auto emit_rhs_vector = [&]() -> Value {
          if (rhs_k_contiguous_) {
            // A column-major [K,N] RHS is the native physical layout used by
            // FlyDSL GEMMs: the four K values consumed by one MFMA lane are a
            // single vector load. The explicit permutation map makes the
            // vector dimension advance source dimension K instead of N.
            mlir::AffineMap k_vector_map = mlir::AffineMap::get(
                /*dimCount=*/2, /*symbolCount=*/0, builder.getAffineDimExpr(0),
                builder.getContext());
            if (async_lhs_) {
              Value source_linear_index = Add(
                  builder,
                  Mul(builder, global_column, IndexConstant(builder, k_)),
                  Add(builder, global_k, group_k));
              return emit_buffer_load(rhs, source_linear_index,
                                      rhs_vector_type);
            }
            return mlir::vector::TransferReadOp::create(
                builder, rhs_vector_type, rhs,
                mlir::ValueRange{Add(builder, global_k, group_k),
                                 global_column},
                /*padding=*/std::nullopt, k_vector_map,
                llvm::ArrayRef<bool>{true});
          }
          llvm::SmallVector<Value, 8> elements;
          const int64_t vector_elements = async_lhs_ ? 8 : 4;
          for (int64_t element = 0; element < vector_elements; ++element) {
            Value k_index =
                Add(builder, group_k, IndexConstant(builder, element));
            elements.push_back(ExtractTensor(
                builder, rhs, Add(builder, global_k, k_index), global_column));
          }
          return mlir::vector::FromElementsOp::create(
              builder, rhs_vector_type, elements);
        };
        if (n_ % block_n_ == 0) {
          fragments.push_back(emit_rhs_vector());
        } else {
          Value column_in_bounds = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, global_column,
              IndexConstant(builder, n_));
          mlir::scf::IfOp load = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{rhs_vector_type}, column_in_bounds,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard if_guard(builder);
            builder.setInsertionPointToStart(load.thenBlock());
            Value loaded = emit_rhs_vector();
            mlir::scf::YieldOp::create(builder, loaded);
            builder.setInsertionPointToStart(load.elseBlock());
            mlir::scf::YieldOp::create(
                builder, async_lhs_ ? zero_staged_input_vector
                                    : zero_input_vector);
          }
          builder.setInsertionPointAfter(load);
          fragments.push_back(load.getResult(0));
        }
      }
      return fragments;
    };
    auto emit_rhs_stage = [&](Value global_k) {
      llvm::SmallVector<Value> fragments;
      const int64_t rhs_k_step = async_lhs_ ? 2 * atom_k : atom_k;
      fragments.reserve((stage_k_ / rhs_k_step) * wave_tile_columns);
      for (int64_t k_offset = 0; k_offset < stage_k_;
           k_offset += rhs_k_step) {
        llvm::SmallVector<Value> group = emit_rhs_group(global_k, k_offset);
        fragments.append(group);
      }
      return fragments;
    };
    auto emit_rhs_lds_group = [&](Value shared, Value shared_stage,
                                  int64_t k_offset) {
      const int64_t rhs_shared_base = 2 * block_m_ * stage_k_;
      llvm::SmallVector<Value> fragments;
      fragments.reserve(wave_tile_columns);
      Value group_k =
          Add(builder, Mul(builder, lane_group, IndexConstant(builder, 8)),
              IndexConstant(builder, k_offset));
      for (int64_t tile_column = 0; tile_column < wave_tile_columns;
           ++tile_column) {
        Value row = Add(builder, wave_column_offset,
                        Add(builder, lane_axis,
                            IndexConstant(builder, tile_column * atom_m)));
        Value swizzled_k = SwizzleXor16(builder, row, group_k, stage_k_);
        Value stage_base = Mul(builder, shared_stage,
                               IndexConstant(builder, block_n_ * stage_k_));
        Value shared_index =
            Add(builder, IndexConstant(builder, rhs_shared_base),
                Add(builder, stage_base,
                    Add(builder,
                        Mul(builder, row, IndexConstant(builder, stage_k_)),
                        swizzled_k)));
        fragments.push_back(mlir::vector::TransferReadOp::create(
            builder, staged_input_vector_type, shared,
            mlir::ValueRange{shared_index}, /*padding=*/std::nullopt,
            llvm::ArrayRef<bool>{true}));
      }
      return fragments;
    };

    // Match FlyDSL's two-stage HGEMM pipeline: retain the current RHS tile in
    // registers while issuing the next tile's VMEM loads. Larger K tiles would
    // double too many live fragments, so leave those to the streaming path.
    const bool prefetch_rhs = prefetch_rhs_ && !stage_rhs_;
    Value initial_shared = lhs_shared;
    if (stage_rhs_) {
      initial_shared =
          emit_rhs_lds_stage(initial_shared, lower_bound, lower_bound);
    }
    initial_shared =
        async_lhs_
            ? emit_lhs_register_stage(initial_shared, lower_bound, lower_bound)
            : emit_lhs_stage(initial_shared, lower_bound, lower_bound);
    llvm::SmallVector<Value> initial_loop_values = initial_accumulators;
    if (prefetch_rhs) {
      llvm::SmallVector<Value> initial_rhs = emit_rhs_stage(lower_bound);
      initial_loop_values.append(initial_rhs);
    }
    if ((stage_rhs_ || async_lhs_) && schedule_instructions_) {
      mlir::ROCDL::SchedBarrier::create(
          builder, mlir::ROCDL::SchedGroupMask::none);
    }
    initial_shared =
        SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                              initial_shared)
            .getResult(0);
    (void)initial_shared;
    auto emit_compute_tile =
        [&](Value staged_shared, Value current_stage, Value k_base,
            llvm::ArrayRef<Value> accumulators,
            llvm::ArrayRef<Value> prefetched_rhs_values) {
      llvm::SmallVector<Value> next_accumulators(accumulators);
      const int64_t compute_k_step = paired_mfma ? 2 * atom_k : atom_k;
      for (int64_t k_offset = 0; k_offset < stage_k_;
           k_offset += compute_k_step) {
        Value group_k =
            Add(builder,
                Mul(builder, lane_group,
                    IndexConstant(builder, paired_mfma ? 8 : 4)),
                IndexConstant(builder, k_offset));

        llvm::SmallVector<Value> rhs_vectors;
        auto load_rhs_vectors = [&] {
          rhs_vectors.reserve(wave_tile_columns);
          if (stage_rhs_) {
            rhs_vectors =
                emit_rhs_lds_group(staged_shared, current_stage, k_offset);
          } else if (prefetch_rhs) {
            const int64_t rhs_base =
                (k_offset / compute_k_step) * wave_tile_columns;
            for (int64_t tile_column = 0; tile_column < wave_tile_columns;
                 ++tile_column) {
              rhs_vectors.push_back(
                  prefetched_rhs_values[rhs_base + tile_column]);
            }
          } else {
            rhs_vectors = emit_rhs_group(k_base, k_offset);
          }
        };

        llvm::SmallVector<Value> lhs_vectors;
        auto load_lhs_vectors = [&] {
          lhs_vectors.reserve(wave_tile_rows);
          for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
            Value row = Add(builder, wave_row_offset,
                            Add(builder, lane_axis,
                                IndexConstant(builder, tile_row * atom_m)));
            Value swizzled_group_k =
                SwizzleXor16(builder, row, group_k, stage_k_);
            Value shared_index = Add(
                builder,
                Mul(builder, current_stage,
                    IndexConstant(builder, block_m_ * stage_k_)),
                Add(builder,
                    Mul(builder, row, IndexConstant(builder, stage_k_)),
                    swizzled_group_k));
            lhs_vectors.push_back(mlir::vector::TransferReadOp::create(
                builder,
                paired_mfma ? staged_input_vector_type : input_vector_type,
                staged_shared, mlir::ValueRange{shared_index},
                /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true}));
          }
        };
        if (stage_rhs_) {
          load_rhs_vectors();
          load_lhs_vectors();
        } else {
          load_lhs_vectors();
          load_rhs_vectors();
        }

        for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
          for (int64_t tile_column = 0; tile_column < wave_tile_columns;
               ++tile_column) {
            const int64_t accumulator_index =
                tile_row * wave_tile_columns + tile_column;
            const int64_t mfmas_per_fragment = paired_mfma ? 2 : 1;
            for (int64_t mfma = 0; mfma < mfmas_per_fragment; ++mfma) {
              Value lhs_fragment = lhs_vectors[tile_row];
              Value rhs_fragment = rhs_vectors[tile_column];
              if (paired_mfma) {
                const int64_t fragment_base = mfma * 4;
                llvm::SmallVector<int64_t, 4> fragment_mask = {
                    fragment_base, fragment_base + 1, fragment_base + 2,
                    fragment_base + 3};
                lhs_fragment = mlir::vector::ShuffleOp::create(
                    builder, input_vector_type, lhs_fragment, lhs_fragment,
                    fragment_mask);
                rhs_fragment = mlir::vector::ShuffleOp::create(
                    builder, input_vector_type, rhs_fragment, rhs_fragment,
                    fragment_mask);
              }
              mlir::OperationState mma_state(entry_function.getLoc(),
                                             "fly.mma_atom_call_ssa");
              mma_state.addOperands(
                  {mma_atom, lhs_fragment, rhs_fragment,
                   next_accumulators[accumulator_index]});
              mma_state.addTypes(accumulator_type);
              next_accumulators[accumulator_index] =
                  builder.create(mma_state)->getResult(0);
            }
          }
        }
      }
      return next_accumulators;
    };

    auto emit_hot_loop_schedule = [&] {
      if (schedule_instructions_) {
        // As in FlyDSL's hot-loop schedulers, place all scheduling groups at
        // the end of the region they control. LLVM's AMDGPU scheduler consumes
        // matching instructions bottom-up and interleaves each VMEM/LDS group
        // with the corresponding MFMA group.
        const int64_t k_groups =
            stage_k_ / (paired_mfma ? 2 * atom_k : atom_k);
        if (stage_rhs_) {
          constexpr int64_t kCopyElements = 2;
          const int64_t block_threads = num_warps_ * 64;
          const int64_t rhs_copies_per_thread =
              block_n_ * stage_k_ / kCopyElements / block_threads;
          const int64_t lhs_copies_per_thread =
              block_m_ * stage_k_ / kCopyElements / block_threads;
          for (int64_t copy = 0; copy < rhs_copies_per_thread; ++copy) {
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::vmem_read, 1);
          }
          for (int64_t copy = 0; copy < lhs_copies_per_thread; ++copy) {
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::vmem_read, 1);
          }
          for (int64_t group = 0; group < k_groups; ++group) {
            for (int64_t column = 0; column < wave_tile_columns; ++column) {
              ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::ds_read, 1);
            }
            for (int64_t row = 0; row < wave_tile_rows; ++row) {
              ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::ds_read, 1);
            }
            for (int64_t row = 0; row < wave_tile_rows; ++row) {
              ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::mfma_wmma,
                            wave_tile_columns);
            }
          }
          mlir::ROCDL::SchedBarrier::create(
              builder, mlir::ROCDL::SchedGroupMask::none);
        } else if (async_lhs_) {
          // FlyDSL's gfx942 B_TO_LDS=false scheduler first accounts for all
          // next-tile A DMA and B VGPR loads. It then pairs each group of A
          // LDS reads with the corresponding row of N-fragment MFMAs.
          constexpr int64_t kCopyElements = 2;
          const int64_t block_threads = num_warps_ * 64;
          const int64_t lhs_copies_per_thread =
              block_m_ * stage_k_ / kCopyElements / block_threads;
          const int64_t rhs_loads_per_thread =
              k_groups * wave_tile_columns;
          for (int64_t load = 0;
               load < lhs_copies_per_thread + rhs_loads_per_thread; ++load) {
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::vmem_read, 1);
          }
          for (int64_t group = 0; group < k_groups; ++group) {
            for (int64_t row = 0; row < wave_tile_rows; ++row) {
              ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::ds_read, 1);
            }
            for (int64_t row = 0; row < wave_tile_rows; ++row) {
              ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::mfma_wmma,
                            wave_tile_columns);
            }
          }
          mlir::ROCDL::SchedBarrier::create(
              builder, mlir::ROCDL::SchedGroupMask::none);
        } else {
          for (int64_t group = 0; group < k_groups; ++group) {
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::ds_read,
                          wave_tile_rows);
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::vmem_read,
                          wave_tile_columns);
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::mfma_wmma,
                          wave_tile_rows * wave_tile_columns);
          }
        }
      }
    };

    // FlyDSL peels the final K tile. Every main-loop iteration prefetches one
    // valid next tile, overlaps it with the current tile's MFMAs, and then
    // synchronizes. The peeled tail performs only the final LDS reads and
    // MFMAs—there is no wrapped prefetch and no trailing workgroup barrier.
    const bool faithful_fly_pipeline = stage_rhs_ || async_lhs_;
    Value loop_upper_bound =
        faithful_fly_pipeline ? IndexConstant(builder, k_ - stage_k_)
                              : upper_bound;
    mlir::scf::ForOp loop = mlir::scf::ForOp::create(
        builder, lower_bound, loop_upper_bound, step, initial_loop_values,
        [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      Value k_base = loop.getInductionVar();
      Value current_stage = Rem(builder, Div(builder, k_base, stage_k_), 2);
      Value next_stage = Rem(
          builder, Add(builder, current_stage, IndexConstant(builder, 1)), 2);
      Value next_k = Add(builder, k_base, step);
      Value prefetch_k =
          faithful_fly_pipeline ? next_k : Rem(builder, next_k, k_);
      Value staged_shared = lhs_shared;
      if (stage_rhs_) {
        staged_shared =
            emit_rhs_lds_stage(staged_shared, prefetch_k, next_stage);
      }
      staged_shared =
          emit_lhs_stage(staged_shared, prefetch_k, next_stage);
      llvm::SmallVector<Value> next_rhs;
      if (prefetch_rhs) {
        next_rhs = emit_rhs_stage(prefetch_k);
      }

      llvm::SmallVector<Value> current_accumulators;
      current_accumulators.reserve(accumulators_per_wave);
      for (int64_t index = 0; index < accumulators_per_wave; ++index) {
        current_accumulators.push_back(loop.getRegionIterArg(index));
      }
      llvm::SmallVector<Value> current_rhs;
      current_rhs.reserve(initial_loop_values.size() - accumulators_per_wave);
      for (int64_t index = accumulators_per_wave;
           index < initial_loop_values.size(); ++index) {
        current_rhs.push_back(loop.getRegionIterArg(index));
      }
      llvm::SmallVector<Value> next_accumulators =
          emit_compute_tile(staged_shared, current_stage, k_base,
                            current_accumulators, current_rhs);
      emit_hot_loop_schedule();
      Value synchronized_lhs =
          SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                staged_shared)
              .getResult(0);
      (void)synchronized_lhs;
      next_accumulators.append(next_rhs);
      mlir::scf::YieldOp::create(builder, next_accumulators);
    }

    builder.setInsertionPointAfter(loop);
    llvm::SmallVector<Value> final_accumulators;
    final_accumulators.reserve(accumulators_per_wave);
    if (faithful_fly_pipeline) {
      llvm::SmallVector<Value> loop_accumulators;
      loop_accumulators.reserve(accumulators_per_wave);
      for (int64_t index = 0; index < accumulators_per_wave; ++index) {
        loop_accumulators.push_back(loop.getResult(index));
      }
      llvm::SmallVector<Value> tail_rhs;
      for (int64_t index = accumulators_per_wave;
           index < initial_loop_values.size(); ++index) {
        tail_rhs.push_back(loop.getResult(index));
      }
      final_accumulators = emit_compute_tile(
          lhs_shared, IndexConstant(builder, (k_ / stage_k_ - 1) % 2),
          IndexConstant(builder, k_ - stage_k_), loop_accumulators,
          tail_rhs);
    } else {
      for (int64_t index = 0; index < accumulators_per_wave; ++index) {
        final_accumulators.push_back(loop.getResult(index));
      }
    }
    mlir::Type output_type =
        mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType();
    if (stage_output_) {
      TF_RET_CHECK(output_type.isBF16());
      TF_RET_CHECK(m_ % block_m_ == 0 && n_ % block_n_ == 0);
      TF_RET_CHECK(block_m_ * block_n_ * sizeof(uint16_t) <= 64 * 1024);
    }
    Value staged_output = lhs_shared;
    if (stage_output_) {
      // FlyDSL synchronizes all waves before reusing the A/B LDS allocation
      // for C. Without this barrier, one wave could overwrite a final-tile
      // fragment while another wave is still reading it.
      staged_output =
          SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                staged_output)
              .getResult(0);
    }
    for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
      for (int64_t tile_column = 0; tile_column < wave_tile_columns;
           ++tile_column) {
        const int64_t accumulator_index =
            tile_row * wave_tile_columns + tile_column;
        Value accumulator = final_accumulators[accumulator_index];
        for (int64_t element = 0; element < accumulator_elements; ++element) {
          Value result = mlir::vector::ExtractOp::create(
              builder, accumulator, llvm::SmallVector<int64_t>{element});
          if (output_type.isBF16()) {
            result =
                mlir::arith::TruncFOp::create(builder, output_type, result);
          }
          const int64_t element_row =
              use_mfma_32 ? (element / 4) * 8 + element % 4 : element;
          Value output_row = Add(
              builder, wave_row_base,
              Add(builder, Mul(builder, lane_group, IndexConstant(builder, 4)),
                  IndexConstant(builder, tile_row * atom_m + element_row)));
          Value output_column =
              Add(builder, wave_column_base,
                  Add(builder, lane_axis,
                      IndexConstant(builder, tile_column * atom_m)));
          llvm::SmallVector<Value, 2> indices = {output_row, output_column};
          if (stage_output_) {
            Value local_row = Add(
                builder, wave_row_offset,
                Add(builder,
                    Mul(builder, lane_group, IndexConstant(builder, 4)),
                    IndexConstant(builder, tile_row * atom_m + element_row)));
            Value local_column =
                Add(builder, wave_column_offset,
                    Add(builder, lane_axis,
                        IndexConstant(builder, tile_column * atom_m)));
            Value shared_index =
                Add(builder,
                    Mul(builder, local_row, IndexConstant(builder, block_n_)),
                    local_column);
            staged_output = mlir::tensor::InsertOp::create(
                builder, result, staged_output, shared_index);
          } else if (m_ % block_m_ == 0 && n_ % block_n_ == 0) {
            output = mlir::tensor::InsertOp::create(builder, result, output,
                                                    indices);
          } else {
            Value row_in_bounds = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ult, output_row,
                IndexConstant(builder, m_));
            Value column_in_bounds = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ult, output_column,
                IndexConstant(builder, n_));
            Value output_in_bounds = mlir::arith::AndIOp::create(
                builder, row_in_bounds, column_in_bounds);
            mlir::scf::IfOp store = mlir::scf::IfOp::create(
                builder, mlir::TypeRange{output.getType()}, output_in_bounds,
                /*withElseRegion=*/true);
            {
              mlir::OpBuilder::InsertionGuard if_guard(builder);
              builder.setInsertionPointToStart(store.thenBlock());
              Value updated = mlir::tensor::InsertOp::create(builder, result,
                                                             output, indices);
              mlir::scf::YieldOp::create(builder, updated);

              builder.setInsertionPointToStart(store.elseBlock());
              mlir::scf::YieldOp::create(builder, output);
            }
            builder.setInsertionPointAfter(store);
            output = store.getResult(0);
          }
        }
      }
    }

    if (stage_output_) {
      staged_output =
          SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                staged_output)
              .getResult(0);
      // FlyDSL GEMM epilogues use eight BF16 values per LDS/global transfer.
      constexpr int64_t kOutputVectorWidth = 8;
      auto output_vector_type =
          mlir::VectorType::get({kOutputVectorWidth}, builder.getBF16Type());
      const int64_t block_threads = num_warps_ * 64;
      const int64_t output_vectors = block_m_ * block_n_ / kOutputVectorWidth;
      mlir::scf::ForOp store_loop = mlir::scf::ForOp::create(
          builder, thread_id, IndexConstant(builder, output_vectors),
          IndexConstant(builder, block_threads), mlir::ValueRange{output},
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard store_guard(builder);
        builder.setInsertionPointToStart(store_loop.getBody());
        Value vector_index = store_loop.getInductionVar();
        Value local_linear = Mul(builder, vector_index,
                                 IndexConstant(builder, kOutputVectorWidth));
        Value local_row = Div(builder, local_linear, block_n_);
        Value local_column = Rem(builder, local_linear, block_n_);
        Value result = mlir::vector::TransferReadOp::create(
            builder, output_vector_type, staged_output,
            mlir::ValueRange{local_linear}, /*padding=*/std::nullopt,
            llvm::ArrayRef<bool>{true});
        Value updated =
            mlir::vector::TransferWriteOp::create(
                builder, result, store_loop.getRegionIterArg(0),
                mlir::ValueRange{Add(builder, block_m_base, local_row),
                                 Add(builder, block_n_base, local_column)},
                llvm::ArrayRef<bool>{true})
                .getResult();
        mlir::scf::YieldOp::create(builder, updated);
      }
      builder.setInsertionPointAfter(store_loop);
      output = store_loop.getResult(0);
    }

    mlir::func::ReturnOp::create(builder, output);
    return absl::OkStatus();
  }

  int64_t block_m_ = 0;
  int64_t block_n_ = 0;
  int64_t num_warps_ = 0;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  int64_t stage_k_;
  bool use_mfma_32_ = false;
  bool prefetch_rhs_ = false;
  bool stage_output_ = false;
  bool schedule_instructions_ = false;
  bool stage_rhs_ = false;
  bool async_lhs_ = false;
  bool rhs_k_contiguous_ = false;
  LaunchDimensions launch_dimensions_;
};

// A singleton GEMM dimension does not need an MFMA tile. Mapping M=1 across
// output columns and N=1 to one reduction wave per output row avoids doing the
// 16x16 padded work required by the generic emitter.
class FlyXTileGemvEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileGemvEmitter(const HloFusionAnalysis& analysis) {
    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    if (config.output_tiles_size() == 1 &&
        config.output_tiles(0).sizes_size() == 2) {
      block_m_ = config.output_tiles(0).sizes(0);
      block_n_ = config.output_tiles(0).sizes(1);
      num_warps_ = config.num_warps();
    }
    const HloInstruction* dot = &analysis.fusion_root(0).instruction();
    m_ = dot->shape().dimensions(0);
    n_ = dot->shape().dimensions(1);
    k_ = dot->operand(0)->shape().dimensions(1);
    const int64_t blocks =
        m_ == 1 ? (n_ + block_n_ - 1) / block_n_
                : (m_ + block_m_ - 1) / block_m_;
    launch_dimensions_ =
        LaunchDimensions(se::BlockDim(blocks, 1, 1),
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
    ASSIGN_OR_RETURN(
        mlir::func::FuncOp entry_function,
        emitters::EmitKernelApi(*module, fusion, buffer_assignment,
                                GetDefaultBufferAlignment(),
                                entry_function_name));
    SetBackendKind(&context, entry_function, BackendKind::kGpu);
    emitters::SetIndexDataLayout(*module, fusion);
    RETURN_IF_ERROR(EmitKernel(entry_function));
    return module;
  }

  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations&,
      const emitters::CallTargetProvider&, mlir::func::FuncOp,
      const HloFusionInstruction&) const override {
    return absl::UnimplementedError(
        "FlyGemvEmitter builds its module directly.");
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function) const {
    TF_RET_CHECK(entry_function.getNumArguments() == 3);
    TF_RET_CHECK(m_ == 1 || n_ == 1);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(),
                                       entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());
    Value lhs = entry_function.getArgument(0);
    Value rhs = entry_function.getArgument(1);
    Value output = entry_function.getArgument(2);
    Value thread_id = EmitThreadId(builder, 0);
    Value block_id = EmitBlockId(builder, 0);
    Value wave_id = Div(builder, thread_id, 64);
    Value lane_id = Rem(builder, thread_id, 64);
    Value zero = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(), llvm::APFloat(0.0f));

    auto emit_product = [&](Value row, Value column, Value k_start,
                            int64_t k_step) {
      mlir::scf::ForOp reduction = mlir::scf::ForOp::create(
          builder, k_start, IndexConstant(builder, k_),
          IndexConstant(builder, k_step), mlir::ValueRange{zero},
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(reduction.getBody());
        Value k_index = reduction.getInductionVar();
        Value a = ExtractTensor(builder, lhs, row, k_index);
        Value b = ExtractTensor(builder, rhs, k_index, column);
        a = mlir::arith::ExtFOp::create(builder, builder.getF32Type(), a);
        b = mlir::arith::ExtFOp::create(builder, builder.getF32Type(), b);
        Value product = mlir::arith::MulFOp::create(builder, a, b);
        Value sum = mlir::arith::AddFOp::create(
            builder, reduction.getRegionIterArg(0), product);
        mlir::scf::YieldOp::create(builder, sum);
      }
      builder.setInsertionPointAfter(reduction);
      return reduction.getResult(0);
    };

    auto convert_output = [&](Value value) {
      mlir::Type output_type =
          mlir::cast<mlir::RankedTensorType>(output.getType())
              .getElementType();
      if (output_type.isBF16()) {
        return mlir::arith::TruncFOp::create(builder, output_type, value)
            .getResult();
      }
      return value;
    };

    if (m_ == 1) {
      const int64_t threads = num_warps_ * 64;
      Value column_base =
          Mul(builder, block_id, IndexConstant(builder, block_n_));
      mlir::scf::ForOp columns = mlir::scf::ForOp::create(
          builder, thread_id, IndexConstant(builder, block_n_),
          IndexConstant(builder, threads), mlir::ValueRange{output},
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(columns.getBody());
        Value column = Add(builder, column_base, columns.getInductionVar());
        Value sum = emit_product(IndexConstant(builder, 0), column,
                                 IndexConstant(builder, 0), /*k_step=*/1);
        Value result = convert_output(sum);
        Value updated = mlir::tensor::InsertOp::create(
            builder, result, columns.getRegionIterArg(0),
            mlir::ValueRange{IndexConstant(builder, 0), column});
        mlir::scf::YieldOp::create(builder, updated);
      }
      builder.setInsertionPointAfter(columns);
      mlir::func::ReturnOp::create(builder, columns.getResult(0));
      return absl::OkStatus();
    }

    Value row_base =
        Mul(builder, block_id, IndexConstant(builder, block_m_));
    mlir::scf::ForOp rows = mlir::scf::ForOp::create(
        builder, wave_id, IndexConstant(builder, block_m_),
        IndexConstant(builder, num_warps_), mlir::ValueRange{output},
        [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(rows.getBody());
      Value row = Add(builder, row_base, rows.getInductionVar());
      Value sum = emit_product(row, IndexConstant(builder, 0), lane_id,
                               /*k_step=*/64);
      for (int32_t distance : {32, 16, 8, 4, 2, 1}) {
        Value shuffled =
            mlir::gpu::ShuffleOp::create(
                builder, sum, distance, /*width=*/64,
                mlir::gpu::ShuffleMode::DOWN)
                .getShuffleResult();
        sum = mlir::arith::AddFOp::create(builder, sum, shuffled);
      }
      Value is_lane_zero = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, lane_id,
          IndexConstant(builder, 0));
      mlir::scf::IfOp store = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{output.getType()}, is_lane_zero,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard if_guard(builder);
        builder.setInsertionPointToStart(store.thenBlock());
        Value result = convert_output(sum);
        Value updated = mlir::tensor::InsertOp::create(
            builder, result, rows.getRegionIterArg(0),
            mlir::ValueRange{row, IndexConstant(builder, 0)});
        mlir::scf::YieldOp::create(builder, updated);
        builder.setInsertionPointToStart(store.elseBlock());
        mlir::scf::YieldOp::create(builder, rows.getRegionIterArg(0));
      }
      builder.setInsertionPointAfter(store);
      mlir::scf::YieldOp::create(builder, store.getResult(0));
    }
    builder.setInsertionPointAfter(rows);
    mlir::func::ReturnOp::create(builder, rows.getResult(0));
    return absl::OkStatus();
  }

  int64_t block_m_ = 0;
  int64_t block_n_ = 0;
  int64_t num_warps_ = 0;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileGemmEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileGemmEmitter>(analysis);
}

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileGemvEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileGemvEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
