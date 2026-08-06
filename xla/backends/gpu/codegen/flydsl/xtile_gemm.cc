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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

Value Sub(mlir::ImplicitLocOpBuilder& builder, Value lhs, Value rhs) {
  return mlir::arith::SubIOp::create(builder, lhs, rhs);
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

Value SwizzleTritonVec4(mlir::ImplicitLocOpBuilder& builder, Value row,
                        Value column) {
  // Triton's gfx942 MFMA32 GEMM uses
  //   swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8}>.
  // `column` is a BF16 element offset, so the phase is four elements wide.
  Value phase = Rem(builder, Div(builder, row, 2), 8);
  return mlir::arith::XOrIOp::create(
      builder, column, Mul(builder, phase, IndexConstant(builder, 4)));
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
        (xtile_config.output_tiles(0).sizes_size() == 2 ||
         xtile_config.output_tiles(0).sizes_size() == 3)) {
      const int64_t tile_rank = xtile_config.output_tiles(0).sizes_size();
      block_m_ = xtile_config.output_tiles(0).sizes(tile_rank - 2);
      block_n_ = xtile_config.output_tiles(0).sizes(tile_rank - 1);
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
      preload_lds_fragments_ =
          fusion_config.fly_gemm_config().preload_lds_fragments();
      single_buffer_lds_ = fusion_config.fly_gemm_config().single_buffer_lds();
      direct_to_vgpr_ = fusion_config.fly_gemm_config().direct_to_vgpr();
      rolling_refill_ = fusion_config.fly_gemm_config().rolling_refill();
      local_split_k_ = fusion_config.fly_gemm_config().local_split_k();
    }
    const HloInstruction* dot = &analysis.fusion_root(0).instruction();
    global_split_k_ = dot->shape().dimensions_size() == 3;
    split_k_batches_ = global_split_k_ ? dot->shape().dimensions(0) : 1;
    m_ = dot->shape().dimensions(global_split_k_ ? 1 : 0);
    n_ = dot->shape().dimensions(global_split_k_ ? 2 : 1);
    k_ = dot->operand(0)->shape().dimensions(
        dot->dot_dimension_numbers().lhs_contracting_dimensions(0));
    rhs_contracting_dimension_ =
        dot->dot_dimension_numbers().rhs_contracting_dimensions(0);
    rhs_k_contiguous_ =
        dot->operand(1)->shape().has_layout() &&
        dot->operand(1)->shape().layout().minor_to_major(0) ==
            rhs_contracting_dimension_;
    stage_k_ = k_ % 32 == 0 ? 32 : 16;
    absl::StatusOr<Tile> contraction_tile = dot->backend_config<Tile>();
    if (contraction_tile.ok() && contraction_tile->sizes_size() == 1 &&
        contraction_tile->sizes(0) >= 16 &&
        contraction_tile->sizes(0) % 16 == 0 &&
        k_ % contraction_tile->sizes(0) == 0) {
      stage_k_ = contraction_tile->sizes(0);
    }
    const int64_t blocks = split_k_batches_ *
                           ((m_ + block_m_ - 1) / block_m_) *
                           ((n_ + block_n_ - 1) / block_n_);
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
    Value global_split_k_id;
    auto rhs_indices = [&](Value k, Value n) {
      llvm::SmallVector<Value, 3> indices;
      if (global_split_k_) {
        indices.assign({n, global_split_k_id, k});
        return indices;
      }
      if (rhs_contracting_dimension_ == 0) {
        indices.assign({k, n});
      } else {
        indices.assign({n, k});
      }
      return indices;
    };
    if (stage_rhs_) {
      constexpr int64_t kCopyElements = 2;
      const int64_t block_threads = num_warps_ * 64;
      TF_RET_CHECK(stage_k_ >= 32 && stage_k_ % 32 == 0 && block_m_ % 16 == 0 &&
                   block_n_ % 16 == 0 && (!use_mfma_32_ || single_buffer_lds_));
      TF_RET_CHECK((block_m_ * stage_k_ / kCopyElements) % block_threads == 0 &&
                   (block_n_ * stage_k_ / kCopyElements) % block_threads == 0);
      const int64_t lds_stages = single_buffer_lds_ ? 1 : 2;
      const bool tensile_double_buffer =
          !single_buffer_lds_ && direct_to_vgpr_ && !use_mfma_32_ &&
          num_warps_ == 4 && stage_k_ == 64 &&
          ((block_m_ == 256 && block_n_ == 224) ||
           (block_m_ == 224 && block_n_ == 256));
      // Tensile stages only the non-DirectToVgpr operand. Its 224x64 payload
      // uses seven padded 32-row blocks (31.5 KiB) and a 32 KiB ping-pong
      // stride, for an exact 65,024-byte allocation.
      constexpr int64_t kTensileStageStrideElements = 16 * 1024;
      constexpr int64_t kTensilePhysicalStageElements = 7 * 2304;
      const int64_t staged_lds_bytes =
          tensile_double_buffer
              ? (kTensileStageStrideElements +
                 kTensilePhysicalStageElements) *
                    sizeof(uint16_t)
              : lds_stages * (block_m_ + block_n_) * stage_k_ *
                    sizeof(uint16_t);
      TF_RET_CHECK(staged_lds_bytes <= 64 * 1024);
      TF_RET_CHECK(rhs_k_contiguous_);
      // XLA's global split-K and Fly's wave-local split-K are composable: the
      // former selects an input/output batch while the latter reduces two
      // wave partitions into that batch's FP32 partial. Staged-output and
      // async-LHS variants still have no rank-3 epilogue.
      TF_RET_CHECK(!global_split_k_ || (!stage_output_ && !async_lhs_));
      const bool tensile_n_edge_tile =
          (single_buffer_lds_ || tensile_double_buffer) && !use_mfma_32_ &&
          num_warps_ == 4 &&
          stage_k_ == 64 && block_m_ == 256 && block_n_ == 224 && n_ % 16 == 0;
      const bool tensile_m_edge_tile =
          (single_buffer_lds_ || tensile_double_buffer) && !use_mfma_32_ &&
          num_warps_ == 4 && stage_k_ == 64 && block_m_ == 224 &&
          block_n_ == 256 && m_ % 16 == 0;
      TF_RET_CHECK((m_ % block_m_ == 0 && n_ % block_n_ == 0) ||
                   (global_split_k_ && m_ % block_m_ == 0 && n_ % 16 == 0) ||
                   (m_ % block_m_ == 0 && tensile_n_edge_tile) ||
                   (n_ % block_n_ == 0 && tensile_m_edge_tile));
    }
    if (preload_lds_fragments_) {
      const bool native_double_buffer =
          !single_buffer_lds_ && !use_mfma_32_ && stage_k_ == 64 &&
          block_m_ == block_n_ && (block_m_ == 64 || block_m_ == 128);
      const bool triton_single_buffer =
          single_buffer_lds_ && use_mfma_32_ && num_warps_ == 4 &&
          ((stage_k_ == 32 && ((block_m_ == 128 && block_n_ == 256) ||
                               (block_m_ == 256 && block_n_ == 128))) ||
           (stage_k_ == 64 && block_m_ == 128 && block_n_ == 128));
      const bool tensile_single_buffer =
          single_buffer_lds_ && !use_mfma_32_ && num_warps_ == 4 &&
          stage_k_ == 64 &&
          ((block_m_ == 256 && block_n_ == 224) ||
           (block_m_ == 224 && block_n_ == 256));
      const bool tensile_double_buffer =
          !single_buffer_lds_ && direct_to_vgpr_ && !use_mfma_32_ &&
          num_warps_ == 4 && stage_k_ == 64 &&
          ((block_m_ == 256 && block_n_ == 224) ||
           (block_m_ == 224 && block_n_ == 256));
      const bool small_grid_single_buffer =
          single_buffer_lds_ && !use_mfma_32_ && num_warps_ == 4 &&
          stage_k_ == 128 && block_m_ == 128 &&
          (block_n_ == 64 || block_n_ == 96);
      TF_RET_CHECK(stage_rhs_ && !async_lhs_ &&
                   (native_double_buffer || triton_single_buffer ||
                    tensile_single_buffer || tensile_double_buffer ||
                    small_grid_single_buffer));
    }
    TF_RET_CHECK(!single_buffer_lds_ || preload_lds_fragments_);
    if (async_lhs_) {
      TF_RET_CHECK(stage_k_ == 64 && block_m_ == 128 && block_n_ == 128 &&
                   (num_warps_ == 4 || num_warps_ == 8 || num_warps_ == 16) &&
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
    const bool triton_vec4_lds = single_buffer_lds_ && use_mfma_32 &&
                                 stage_k_ == 32 && block_m_ == 128 &&
                                 block_n_ == 256 && num_warps_ == 4;
    const bool tensile_wide_tile = !use_mfma_32 && stage_k_ == 64 &&
                                   ((block_m_ == 256 && block_n_ == 224) ||
                                    (block_m_ == 224 && block_n_ == 256)) &&
                                   num_warps_ == 4;
    const bool small_grid_single_buffer =
        single_buffer_lds_ && !use_mfma_32 && stage_k_ == 128 &&
        block_m_ == 128 && (block_n_ == 64 || block_n_ == 96) &&
        num_warps_ == 4;
    const bool small_grid_rolling_refill =
        small_grid_single_buffer && rolling_refill_;
    const bool tensile_transposed_tile =
        tensile_wide_tile && block_m_ == 224 && block_n_ == 256;
    const bool source_swap_mfma = tensile_wide_tile || local_split_k_;
    TF_RET_CHECK(!rolling_refill_ || small_grid_single_buffer);
    TF_RET_CHECK(!local_split_k_ ||
                 (small_grid_single_buffer && stage_rhs_ &&
                  preload_lds_fragments_ && !stage_output_ &&
                  stage_k_ % 2 == 0 && k_ % stage_k_ == 0 &&
                  k_ >= 2 * stage_k_));
    // hipBLASLt's matching MT256x224x64 solution uses DirectToVgprA. Keep the
    // wave-local 256-wide operand in registers in either output orientation.
    const bool tensile_direct_lhs =
        direct_to_vgpr_ && tensile_wide_tile && !tensile_transposed_tile;
    const bool tensile_direct_rhs =
        direct_to_vgpr_ && tensile_wide_tile && tensile_transposed_tile;
    const bool tensile_direct_operand =
        tensile_direct_lhs || tensile_direct_rhs;
    const bool tensile_double_buffer =
        tensile_direct_operand && !single_buffer_lds_;
    const int64_t atom_m = use_mfma_32 ? 32 : 16;
    const int64_t atom_k = use_mfma_32 ? 8 : 16;
    const int64_t accumulator_elements = use_mfma_32 ? 16 : 4;
    Value lane_axis = Rem(builder, lane_id, atom_m);
    Value lane_group = Div(builder, lane_id, atom_m);
    const int64_t local_split_factor = local_split_k_ ? 2 : 1;
    const int64_t output_waves = num_warps_ / local_split_factor;
    const int64_t compute_stage_k = stage_k_ / local_split_factor;
    Value output_wave_id =
        local_split_k_ ? Rem(builder, wave_id, output_waves) : wave_id;
    Value local_split_id =
        local_split_k_ ? Div(builder, wave_id, output_waves)
                       : IndexConstant(builder, 0);
    Value local_split_k_offset =
        Mul(builder, local_split_id, IndexConstant(builder, compute_stage_k));
    constexpr int64_t kLocalSplitRowsPerBlock = 64;
    constexpr int64_t kLocalSplitRowsPerPad = 4;
    constexpr int64_t kLocalSplitPadElements = 16;
    auto local_split_physical_row = [&](Value logical_row) {
      // Tensile's LocalSplit layout transposes each logical 4x16 row tile:
      //   logical = tile * 16 + lane
      //   physical = lane * 4 + tile.
      // This makes the four fragments consumed by a wave adjacent in LDS.
      Value row_block = Div(builder, logical_row, kLocalSplitRowsPerBlock);
      Value row_in_block = Rem(builder, logical_row, kLocalSplitRowsPerBlock);
      Value tile = Div(builder, row_in_block, atom_m);
      Value lane = Rem(builder, row_in_block, atom_m);
      return Add(
          builder,
          Mul(builder, row_block,
              IndexConstant(builder, kLocalSplitRowsPerBlock)),
          Add(builder,
              Mul(builder, lane,
                  IndexConstant(builder, kLocalSplitRowsPerPad)),
              tile));
    };
    auto local_split_logical_row = [&](Value physical_row) {
      Value row_block = Div(builder, physical_row, kLocalSplitRowsPerBlock);
      Value row_in_block = Rem(builder, physical_row, kLocalSplitRowsPerBlock);
      Value lane = Div(builder, row_in_block, kLocalSplitRowsPerPad);
      Value tile = Rem(builder, row_in_block, kLocalSplitRowsPerPad);
      return Add(
          builder,
          Mul(builder, row_block,
              IndexConstant(builder, kLocalSplitRowsPerBlock)),
          Add(builder, Mul(builder, tile, IndexConstant(builder, atom_m)),
              lane));
    };
    auto add_local_split_lds_padding = [&](Value physical_element) {
      // Insert 32 bytes after every 1 KiB (four 128-wide BF16 rows).
      // A 256-thread dwordx4 refill therefore advances by 4224 bytes,
      // matching the native Tensile kernel exactly.
      return Add(
          builder, physical_element,
          Mul(builder, Div(builder, physical_element,
                           kLocalSplitRowsPerPad * stage_k_),
              IndexConstant(builder, kLocalSplitPadElements)));
    };

    const int64_t grid_m = (m_ + block_m_ - 1) / block_m_;
    const int64_t grid_n = (n_ + block_n_ - 1) / block_n_;
    const int64_t output_grid = grid_m * grid_n;
    global_split_k_id = global_split_k_
                            ? Div(builder, block_id, output_grid)
                            : IndexConstant(builder, 0);
    Value output_block_id = global_split_k_
                                ? Rem(builder, block_id, output_grid)
                                : block_id;
    if (tensile_direct_lhs && output_grid >= 8) {
      // Match Tensile's WGMXCC=8 remap on the serialized workgroup id before
      // recovering the two-dimensional tile coordinates. MI300X has 304 CUs,
      // so complete 304-workgroup chunks are transposed as an 8x38 grid.
      constexpr int64_t kWorkgroupMappingXcc = 8;
      constexpr int64_t kMi300xComputeUnits = 304;
      Value xcc_group = Div(builder, output_block_id, kMi300xComputeUnits);
      Value xcc_group_base =
          Mul(builder, xcc_group, IndexConstant(builder, kMi300xComputeUnits));
      Value xcc_group_offset = Sub(builder, output_block_id, xcc_group_base);
      Value xcc_group_size = mlir::arith::SelectOp::create(
          builder,
          mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult,
              Sub(builder, IndexConstant(builder, output_grid),
                  xcc_group_base),
              IndexConstant(builder, kMi300xComputeUnits)),
          Sub(builder, IndexConstant(builder, output_grid), xcc_group_base),
          IndexConstant(builder, kMi300xComputeUnits));
      Value mapped_range = Mul(
          builder, Div(builder, xcc_group_size, kWorkgroupMappingXcc),
          IndexConstant(builder, kWorkgroupMappingXcc));
      Value mapped_offset = Add(
          builder, Div(builder, xcc_group_offset, kWorkgroupMappingXcc),
          Mul(builder,
              Rem(builder, xcc_group_offset, kWorkgroupMappingXcc),
              Div(builder, xcc_group_size, kWorkgroupMappingXcc)));
      output_block_id = mlir::arith::SelectOp::create(
          builder,
          mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, xcc_group_offset,
              mapped_range),
          Add(builder, xcc_group_base, mapped_offset), output_block_id);
    }
    Value block_m;
    Value block_n;
    if (tensile_direct_rhs) {
      // The matching hipBLASLt WGM4 tile is transposed relative to this one:
      // its 256-wide DirectToVgprA dimension is XLA's N dimension. Apply the
      // corresponding grouping to four neighboring N tiles here. The final
      // group may contain fewer than four N tiles.
      constexpr int64_t kWorkgroupMappingN = 4;
      Value group_id =
          Div(builder, output_block_id, kWorkgroupMappingN * grid_m);
      Value first_block_n =
          Mul(builder, group_id,
              IndexConstant(builder, kWorkgroupMappingN));
      Value remaining_n =
          Sub(builder, IndexConstant(builder, grid_n), first_block_n);
      Value group_n = mlir::arith::SelectOp::create(
          builder,
          mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, remaining_n,
              IndexConstant(builder, kWorkgroupMappingN)),
          remaining_n, IndexConstant(builder, kWorkgroupMappingN));
      Value block_in_group =
          Rem(builder, output_block_id, kWorkgroupMappingN * grid_m);
      block_m =
          mlir::arith::DivUIOp::create(builder, block_in_group, group_n);
      block_n = Add(builder, first_block_n,
                    mlir::arith::RemUIOp::create(builder, block_in_group,
                                                group_n));
    } else if (tensile_direct_lhs) {
      // Apply the selected Tensile solution's positive WGM=8 after WGMXCC has
      // remapped the serialized dispatch id.
      constexpr int64_t kWorkgroupMappingN = 8;
      Value group_id =
          Div(builder, output_block_id, grid_m * kWorkgroupMappingN);
      Value first_block_n =
          Mul(builder, group_id, IndexConstant(builder, kWorkgroupMappingN));
      Value remaining_n =
          Sub(builder, IndexConstant(builder, grid_n), first_block_n);
      Value group_n = mlir::arith::SelectOp::create(
          builder,
          mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, remaining_n,
              IndexConstant(builder, kWorkgroupMappingN)),
          remaining_n, IndexConstant(builder, kWorkgroupMappingN));
      Value block_in_group =
          Rem(builder, output_block_id, grid_m * kWorkgroupMappingN);
      block_m = mlir::arith::DivUIOp::create(builder, block_in_group, group_n);
      block_n = Add(builder, first_block_n,
                    mlir::arith::RemUIOp::create(builder, block_in_group,
                                                group_n));
    } else {
      block_m = Div(builder, output_block_id, grid_n);
      block_n = Rem(builder, output_block_id, grid_n);
    }
    Value block_m_base =
        Mul(builder, block_m, IndexConstant(builder, block_m_));
    Value block_n_base =
        Mul(builder, block_n, IndexConstant(builder, block_n_));
    const int64_t input_row_stride = split_k_batches_ * k_;
    Value input_batch_offset =
        Mul(builder, global_split_k_id, IndexConstant(builder, k_));
    auto input_linear_index = [&](Value row, Value k) {
      return Add(builder,
                 Mul(builder, row,
                     IndexConstant(builder, input_row_stride)),
                 Add(builder, input_batch_offset, k));
    };
    auto output_indices = [&](Value row, Value column) {
      llvm::SmallVector<Value, 3> indices;
      if (global_split_k_) {
        indices.assign({global_split_k_id, row, column});
      } else {
        indices.assign({row, column});
      }
      return indices;
    };

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
    auto emit_buffer_store = [&](Value value, Value destination,
                                 Value destination_linear_index) {
      mlir::OperationState store_state(entry_function.getLoc(),
                                       "xla_gpu.buffer_store");
      store_state.addOperands(
          {value, destination, destination_linear_index});
      store_state.addTypes(destination.getType());
      store_state.addAttribute("cache_policy", builder.getI32IntegerAttr(0));
      return builder.create(store_state)->getResult(0);
    };
    auto emit_split_buffer_load = [&](Value source, Value vector_linear_index,
                                      Value scalar_linear_index,
                                      mlir::VectorType result_type) {
      mlir::OperationState load_state(entry_function.getLoc(),
                                      "xla_gpu.split_buffer_load");
      load_state.addOperands(
          {source, vector_linear_index, scalar_linear_index});
      load_state.addTypes(result_type);
      return builder.create(load_state)->getResult(0);
    };
    auto emit_wait_vmcnt = [&](int64_t remaining) {
      CHECK_GE(remaining, 0);
      CHECK_LE(remaining, 63);
      mlir::OperationState wait_state(entry_function.getLoc(),
                                      "rocdl.s.waitcnt");
      // Keep the non-VM counters at their inactive maxima. gfx9 encodes the
      // low four VM counter bits in [3:0] and the upper two in [15:14].
      const int64_t bitfield =
          0x0f70 | (remaining & 0x0f) | ((remaining & 0x30) << 10);
      wait_state.addAttribute("bitfield", builder.getI32IntegerAttr(bitfield));
      builder.create(wait_state);
    };
    auto emit_wait_lgkmcnt = [&](int64_t remaining) {
      CHECK_GE(remaining, 0);
      CHECK_LE(remaining, 15);
      mlir::OperationState wait_state(entry_function.getLoc(),
                                      "rocdl.s.waitcnt");
      // Keep vmcnt/expcnt inactive while selecting the four-bit LGKM counter.
      const int64_t bitfield = 0xc07f | (remaining << 8);
      wait_state.addAttribute("bitfield", builder.getI32IntegerAttr(bitfield));
      builder.create(wait_state);
    };
    Value zero = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(), llvm::APFloat(0.0f));
    Value zero_accumulator =
        mlir::vector::BroadcastOp::create(builder, accumulator_type, zero);
    const int64_t pipeline_stages = single_buffer_lds_ ? 1 : 2;
    const int64_t lhs_stage_elements =
        block_m_ * stage_k_ +
        (local_split_k_ ? block_m_ / kLocalSplitRowsPerPad *
                              kLocalSplitPadElements
                        : 0);
    constexpr int64_t kTensileStageStrideElements = 16 * 1024;
    constexpr int64_t kTensilePhysicalBlockElements = 2304;
    constexpr int64_t kTensilePhysicalStageElements =
        7 * kTensilePhysicalBlockElements;
    const int64_t rhs_stage_elements =
        (tensile_double_buffer ? kTensileStageStrideElements
                               : block_n_ * stage_k_) +
        (local_split_k_ ? block_n_ / kLocalSplitRowsPerPad *
                              kLocalSplitPadElements
                        : 0);
    const int64_t lhs_shared_elements =
        tensile_direct_lhs
            ? 0
            : (tensile_double_buffer && tensile_direct_rhs
                   ? kTensileStageStrideElements +
                         kTensilePhysicalStageElements
                   : pipeline_stages * lhs_stage_elements);
    const int64_t rhs_shared_elements =
        stage_rhs_ && !tensile_direct_rhs
            ? (tensile_double_buffer
                   ? kTensileStageStrideElements +
                         kTensilePhysicalStageElements
                   : pipeline_stages * rhs_stage_elements)
            : 0;
    const int64_t output_shared_elements =
        stage_output_ ? block_m_ * block_n_ : 0;
    const int64_t local_split_shared_elements =
        local_split_k_ ? 2 * block_m_ * block_n_ : 0;
    auto lhs_shared_type = mlir::RankedTensorType::get(
        {std::max({lhs_shared_elements + rhs_shared_elements,
                   output_shared_elements, local_split_shared_elements})},
        builder.getBF16Type());
    Value lhs_shared = AllocateSharedOp::create(builder, lhs_shared_type);

    const int64_t tile_rows = block_m_ / atom_m;
    const int64_t tile_columns = block_n_ / atom_m;
    TF_RET_CHECK(tile_rows * tile_columns >= output_waves);
    int64_t wave_grid_rows = 1;
    int64_t wave_grid_columns = output_waves;
    int64_t best_fragment_loads =
        tile_rows * wave_grid_columns + tile_columns * wave_grid_rows;
    for (int64_t candidate_rows = 1; candidate_rows <= output_waves;
         candidate_rows *= 2) {
      const int64_t candidate_columns = output_waves / candidate_rows;
      if (output_waves % candidate_rows != 0 ||
          tile_rows % candidate_rows != 0 ||
          tile_columns % candidate_columns != 0) {
        continue;
      }
      const int64_t fragment_loads =
          tile_rows * candidate_columns + tile_columns * candidate_rows;
      if (fragment_loads < best_fragment_loads ||
          (triton_vec4_lds && fragment_loads == best_fragment_loads &&
           candidate_rows == 2 && candidate_columns == 2)) {
        wave_grid_rows = candidate_rows;
        wave_grid_columns = candidate_columns;
        best_fragment_loads = fragment_loads;
      }
    }
    if (tensile_wide_tile) {
      wave_grid_rows = tensile_transposed_tile ? 1 : 4;
      wave_grid_columns = tensile_transposed_tile ? 4 : 1;
    }
    TF_RET_CHECK(tile_rows % wave_grid_rows == 0);
    TF_RET_CHECK(tile_columns % wave_grid_columns == 0);

    const int64_t wave_tile_rows = tile_rows / wave_grid_rows;
    const int64_t wave_tile_columns = tile_columns / wave_grid_columns;
    const int64_t accumulators_per_wave = wave_tile_rows * wave_tile_columns;
    auto get_accumulator_index = [&](int64_t tile_row, int64_t tile_column) {
      // DirectToVgprA holds one staged-B fragment while visiting four A rows.
      // Keep those four accumulators adjacent, as Tensile does, instead of
      // separating them by the full 14-column wave tile in the AGPR file.
      return tensile_direct_lhs
                 ? tile_column * wave_tile_rows + tile_row
                 : tile_row * wave_tile_columns + tile_column;
    };
    Value wave_row = Div(builder, output_wave_id, wave_grid_columns);
    Value wave_column = Rem(builder, output_wave_id, wave_grid_columns);
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
    auto tensile_physical_k = [&](Value logical_k) {
      if (!tensile_direct_lhs || global_split_k_) {
        return logical_k;
      }
      Value in_bounds = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, logical_k, upper_bound);
      // A K offset of exactly k_ aliases the following row in a flat buffer.
      // Send speculative modulo-pipeline loads beyond the complete tensor so
      // the buffer descriptor suppresses them without a branch or extra live
      // vector value.
      Value out_of_bounds =
          IndexConstant(builder, std::max(m_, n_) * k_);
      return mlir::arith::SelectOp::create(builder, in_bounds, logical_k,
                                           out_of_bounds)
          .getResult();
    };
    constexpr int64_t load_vector_width = 8;
    auto load_vector_type =
        mlir::VectorType::get({load_vector_width}, builder.getBF16Type());
    auto lds_vector_type = mlir::VectorType::get({4}, builder.getBF16Type());
    Value zero_load_vector = mlir::arith::ConstantOp::create(
        builder, load_vector_type, builder.getZeroAttr(load_vector_type));
    Value zero_input_vector = mlir::arith::ConstantOp::create(
        builder, input_vector_type, builder.getZeroAttr(input_vector_type));
    Value zero_staged_input_vector = mlir::arith::ConstantOp::create(
        builder, staged_input_vector_type,
        builder.getZeroAttr(staged_input_vector_type));
    Value load_step = IndexConstant(builder, num_warps_ * 64);
    const int64_t lhs_vectors_per_row = stage_k_ / load_vector_width;
    auto emit_lhs_register_vector = [&](Value global_k, int64_t copy) {
      const int64_t block_threads = num_warps_ * 64;
      Value vector_index =
          Add(builder, thread_id, IndexConstant(builder, copy * block_threads));
      Value shared_row = Div(builder, vector_index, lhs_vectors_per_row);
      Value shared_k =
          Mul(builder, Rem(builder, vector_index, lhs_vectors_per_row),
              IndexConstant(builder, load_vector_width));
      if (tensile_double_buffer && tensile_direct_rhs) {
        // Symmetric DirectToVgprB layout: adjacent physical rows alternate
        // between the low and high 16-row tiles, matching the staged-RHS
        // permutation used by the 256x224 orientation.
        Value physical_row = Div(builder, thread_id, lhs_vectors_per_row);
        Value lane = Div(builder, physical_row, 2);
        Value tile = Rem(builder, physical_row, 2);
        shared_row = Add(
            builder, IndexConstant(builder, copy * 32),
            Add(builder, Mul(builder, tile, IndexConstant(builder, 16)),
                lane));
        shared_k = Mul(
            builder, Rem(builder, thread_id, lhs_vectors_per_row),
            IndexConstant(builder, load_vector_width));
      }
      Value logical_row = local_split_k_
                              ? local_split_logical_row(shared_row)
                              : shared_row;
      Value logical_k =
          (triton_vec4_lds || local_split_k_ || tensile_double_buffer)
              ? shared_k
              : SwizzleXor16(builder, shared_row, shared_k, stage_k_);
      if (tensile_direct_rhs || local_split_k_) {
        Value vector_linear_index = input_linear_index(
            Add(builder, block_m_base, logical_row), logical_k);
        return emit_split_buffer_load(
            lhs, vector_linear_index, ReadFirstLaneIndex(builder, global_k),
            load_vector_type);
      }
      Value source_linear_index = input_linear_index(
          Add(builder, block_m_base, shared_row),
          Add(builder, global_k, logical_k));
      return emit_buffer_load(lhs, source_linear_index, load_vector_type);
    };
    auto emit_lhs_register_vectors = [&](Value global_k) {
      const int64_t block_threads = num_warps_ * 64;
      const int64_t vector_count = block_m_ * stage_k_ / load_vector_width;
      llvm::SmallVector<Value> vectors;
      vectors.reserve(vector_count / block_threads);
      for (int64_t copy = 0; copy < vector_count / block_threads; ++copy) {
        vectors.push_back(emit_lhs_register_vector(global_k, copy));
      }
      return vectors;
    };
    const int64_t rhs_vectors_per_row = stage_k_ / load_vector_width;
    auto emit_rhs_register_vector = [&](Value global_k, int64_t copy) {
      const int64_t block_threads = num_warps_ * 64;
      Value vector_index =
          Add(builder, thread_id, IndexConstant(builder, copy * block_threads));
      Value shared_row = Div(builder, vector_index, rhs_vectors_per_row);
      Value shared_k =
          Mul(builder, Rem(builder, vector_index, rhs_vectors_per_row),
              IndexConstant(builder, load_vector_width));
      if (tensile_double_buffer) {
        // Tensile reads each 32-row global block in row order. The low/high
        // 16-row interleave required by our existing MFMA fragment layout is
        // applied to the LDS destination instead of the HBM source below.
        Value physical_row = Div(builder, thread_id, rhs_vectors_per_row);
        shared_row = Add(
            builder,
            IndexConstant(builder, copy * 32),
            physical_row);
        shared_k = Mul(
            builder, Rem(builder, thread_id, rhs_vectors_per_row),
            IndexConstant(builder, load_vector_width));
      }
      Value logical_row = local_split_k_
                              ? local_split_logical_row(shared_row)
                              : shared_row;
      Value logical_k =
          (triton_vec4_lds || local_split_k_ || tensile_double_buffer)
              ? shared_k
              : SwizzleXor16(builder, shared_row, shared_k, stage_k_);
      if (tensile_direct_lhs || local_split_k_) {
        Value vector_linear_index = input_linear_index(
            Add(builder, block_n_base, logical_row), logical_k);
        return emit_split_buffer_load(
            rhs, vector_linear_index,
            ReadFirstLaneIndex(builder, tensile_physical_k(global_k)),
            load_vector_type);
      }
      Value source_linear_index = input_linear_index(
          Add(builder, block_n_base, shared_row),
          Add(builder, global_k, logical_k));
      return emit_buffer_load(rhs, source_linear_index, load_vector_type);
    };
    auto emit_rhs_register_vectors = [&](Value global_k) {
      const int64_t block_threads = num_warps_ * 64;
      const int64_t vector_count = block_n_ * stage_k_ / load_vector_width;
      llvm::SmallVector<Value> vectors;
      vectors.reserve(vector_count / block_threads);
      for (int64_t copy = 0; copy < vector_count / block_threads; ++copy) {
        vectors.push_back(emit_rhs_register_vector(global_k, copy));
      }
      return vectors;
    };
    auto emit_lhs_register_stores = [&](Value destination,
                                        llvm::ArrayRef<Value> vectors,
                                        int64_t copy_begin, int64_t copy_end) {
      const int64_t block_threads = num_warps_ * 64;
      Value stored = destination;
      if (triton_vec4_lds) {
        CHECK_GE(copy_begin, 0);
        CHECK_LE(copy_end, vectors.size());
        CHECK_EQ((copy_end - copy_begin) % 2, 0);
        constexpr int64_t kLdsVectorWidth = 4;
        const int64_t rows_per_copy = block_threads / lhs_vectors_per_row;
        for (int64_t half = 0; half < 2; ++half) {
          llvm::SmallVector<int64_t, 4> half_mask = {
              half * kLdsVectorWidth + 0, half * kLdsVectorWidth + 1,
              half * kLdsVectorWidth + 2, half * kLdsVectorWidth + 3};
          for (int64_t copy = copy_begin; copy < copy_end; copy += 2) {
            Value vector_index =
                Add(builder, thread_id,
                    IndexConstant(builder, copy * block_threads));
            Value shared_row = Div(builder, vector_index, lhs_vectors_per_row);
            Value logical_k =
                Mul(builder, Rem(builder, vector_index, lhs_vectors_per_row),
                    IndexConstant(builder, load_vector_width));
            Value physical_k = SwizzleTritonVec4(
                builder, shared_row,
                Add(builder, logical_k,
                    IndexConstant(builder, half * kLdsVectorWidth)));
            Value first_index =
                Add(builder,
                    Mul(builder, shared_row, IndexConstant(builder, stage_k_)),
                    physical_k);
            for (int64_t member = 0; member < 2; ++member) {
              Value fragment = mlir::vector::ShuffleOp::create(
                  builder, lds_vector_type, vectors[copy + member],
                  vectors[copy + member], half_mask);
              Value shared_index =
                  member == 0
                      ? first_index
                      : Add(builder, first_index,
                            IndexConstant(builder, rows_per_copy * stage_k_));
              stored =
                  mlir::vector::TransferWriteOp::create(
                      builder, fragment, stored, mlir::ValueRange{shared_index},
                      llvm::ArrayRef<bool>{true})
                      .getResult();
            }
          }
        }
        return stored;
      }
      for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
        Value vector_index = Add(builder, thread_id,
                                 IndexConstant(builder, copy * block_threads));
        Value physical_linear = Mul(builder, vector_index,
                                    IndexConstant(builder, load_vector_width));
        if (local_split_k_) {
          physical_linear = add_local_split_lds_padding(physical_linear);
        }
        stored =
            mlir::vector::TransferWriteOp::create(
                builder, vectors[copy], stored,
                mlir::ValueRange{physical_linear}, llvm::ArrayRef<bool>{true})
                .getResult();
      }
      return stored;
    };
    auto emit_lhs_register_stores_at_stage =
        [&](Value destination, llvm::ArrayRef<Value> vectors,
            int64_t copy_begin, int64_t copy_end, Value shared_stage) {
          if (!tensile_double_buffer) {
            return emit_lhs_register_stores(destination, vectors, copy_begin,
                                            copy_end);
          }
          CHECK_GE(copy_begin, 0);
          CHECK_LE(copy_end, vectors.size());
          Value stored = destination;
          Value stage_base = Mul(
              builder, shared_stage,
              IndexConstant(builder, kTensileStageStrideElements));
          Value thread_offset = Add(
              builder,
              Mul(builder, thread_id,
                  IndexConstant(builder, load_vector_width)),
              Mul(builder, Div(builder, thread_id, 16),
                  IndexConstant(builder, 16)));
          for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
            Value shared_index =
                Add(builder, stage_base,
                    Add(builder, thread_offset,
                        IndexConstant(
                            builder,
                            copy * kTensilePhysicalBlockElements)));
            stored = mlir::vector::TransferWriteOp::create(
                         builder, vectors[copy], stored,
                         mlir::ValueRange{shared_index},
                         llvm::ArrayRef<bool>{true})
                         .getResult();
          }
          return stored;
        };
    auto emit_lhs_register_store_word_at_stage =
        [&](Value destination, Value vector, int64_t copy, int64_t word,
            Value shared_stage) {
          CHECK(tensile_double_buffer);
          CHECK_GE(copy, 0);
          CHECK_LT(copy, 7);
          CHECK_GE(word, 0);
          CHECK_LT(word, 4);
          Value stage_base = Mul(
              builder, shared_stage,
              IndexConstant(builder, kTensileStageStrideElements));
          Value thread_offset = Add(
              builder,
              Mul(builder, thread_id,
                  IndexConstant(builder, load_vector_width)),
              Mul(builder, Div(builder, thread_id, 16),
                  IndexConstant(builder, 16)));
          Value shared_index =
              Add(builder, stage_base,
                  Add(builder, thread_offset,
                      IndexConstant(
                          builder,
                          copy * kTensilePhysicalBlockElements + word * 2)));
          auto word_type =
              mlir::VectorType::get({2}, builder.getBF16Type());
          llvm::SmallVector<int64_t, 2> word_mask = {word * 2, word * 2 + 1};
          Value word_value = mlir::vector::ShuffleOp::create(
              builder, word_type, vector, vector, word_mask);
          return mlir::vector::TransferWriteOp::create(
                     builder, word_value, destination,
                     mlir::ValueRange{shared_index},
                     llvm::ArrayRef<bool>{true})
              .getResult();
        };
    auto emit_rhs_register_stores = [&](Value destination,
                                        llvm::ArrayRef<Value> vectors,
                                        int64_t copy_begin, int64_t copy_end) {
      const int64_t block_threads = num_warps_ * 64;
      const int64_t rhs_shared_base = lhs_shared_elements;
      Value stored = destination;
      if (triton_vec4_lds) {
        CHECK_GE(copy_begin, 0);
        CHECK_LE(copy_end, vectors.size());
        CHECK_EQ((copy_end - copy_begin) % 2, 0);
        constexpr int64_t kLdsVectorWidth = 4;
        const int64_t rhs_vectors_per_row = stage_k_ / load_vector_width;
        const int64_t rows_per_copy = block_threads / rhs_vectors_per_row;
        for (int64_t half = 0; half < 2; ++half) {
          llvm::SmallVector<int64_t, 4> half_mask = {
              half * kLdsVectorWidth + 0, half * kLdsVectorWidth + 1,
              half * kLdsVectorWidth + 2, half * kLdsVectorWidth + 3};
          for (int64_t copy = copy_begin; copy < copy_end; copy += 2) {
            Value vector_index =
                Add(builder, thread_id,
                    IndexConstant(builder, copy * block_threads));
            Value shared_row = Div(builder, vector_index, rhs_vectors_per_row);
            Value logical_k =
                Mul(builder, Rem(builder, vector_index, rhs_vectors_per_row),
                    IndexConstant(builder, load_vector_width));
            Value physical_k = SwizzleTritonVec4(
                builder, shared_row,
                Add(builder, logical_k,
                    IndexConstant(builder, half * kLdsVectorWidth)));
            Value first_index = Add(
                builder, IndexConstant(builder, rhs_shared_base),
                Add(builder,
                    Mul(builder, shared_row, IndexConstant(builder, stage_k_)),
                    physical_k));
            for (int64_t member = 0; member < 2; ++member) {
              Value fragment = mlir::vector::ShuffleOp::create(
                  builder, lds_vector_type, vectors[copy + member],
                  vectors[copy + member], half_mask);
              Value shared_index =
                  member == 0
                      ? first_index
                      : Add(builder, first_index,
                            IndexConstant(builder, rows_per_copy * stage_k_));
              stored =
                  mlir::vector::TransferWriteOp::create(
                      builder, fragment, stored, mlir::ValueRange{shared_index},
                      llvm::ArrayRef<bool>{true})
                      .getResult();
            }
          }
        }
        return stored;
      }
      for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
        Value vector_index = Add(builder, thread_id,
                                 IndexConstant(builder, copy * block_threads));
        Value physical_linear = Mul(builder, vector_index,
                                    IndexConstant(builder, load_vector_width));
        if (local_split_k_) {
          physical_linear = add_local_split_lds_padding(physical_linear);
        }
        physical_linear =
            Add(builder, IndexConstant(builder, rhs_shared_base),
                physical_linear);
        stored =
            mlir::vector::TransferWriteOp::create(
                builder, vectors[copy], stored,
                mlir::ValueRange{physical_linear}, llvm::ArrayRef<bool>{true})
                .getResult();
      }
      return stored;
    };
    auto emit_rhs_register_stores_at_stage =
        [&](Value destination, llvm::ArrayRef<Value> vectors,
            int64_t copy_begin, int64_t copy_end, Value shared_stage) {
          if (!tensile_double_buffer) {
            return emit_rhs_register_stores(destination, vectors, copy_begin,
                                            copy_end);
          }
          CHECK_GE(copy_begin, 0);
          CHECK_LE(copy_end, vectors.size());
          const int64_t rhs_shared_base = lhs_shared_elements;
          Value stored = destination;
          Value stage_base = Mul(
              builder, shared_stage,
              IndexConstant(builder, kTensileStageStrideElements));
          Value store_thread_id = thread_id;
          if (tensile_direct_lhs) {
            Value source_row = Div(builder, thread_id, rhs_vectors_per_row);
            Value low_half = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ult, source_row,
                IndexConstant(builder, 16));
            Value interleaved_row = mlir::arith::SelectOp::create(
                builder, low_half,
                Mul(builder, source_row, IndexConstant(builder, 2)),
                Add(builder,
                    Mul(builder,
                        Sub(builder, source_row, IndexConstant(builder, 16)),
                        IndexConstant(builder, 2)),
                    IndexConstant(builder, 1)));
            store_thread_id = Add(
                builder,
                Mul(builder, interleaved_row,
                    IndexConstant(builder, rhs_vectors_per_row)),
                Rem(builder, thread_id, rhs_vectors_per_row));
          }
          Value thread_offset = Add(
              builder,
              Mul(builder, store_thread_id,
                  IndexConstant(builder, load_vector_width)),
              Mul(builder, Div(builder, store_thread_id, 16),
                  IndexConstant(builder, 16)));
          for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
            Value shared_index =
                Add(builder, IndexConstant(builder, rhs_shared_base),
                    Add(builder, stage_base,
                        Add(builder, thread_offset,
                            IndexConstant(
                                builder,
                                copy * kTensilePhysicalBlockElements))));
            stored = mlir::vector::TransferWriteOp::create(
                         builder, vectors[copy], stored,
                         mlir::ValueRange{shared_index},
                         llvm::ArrayRef<bool>{true})
                         .getResult();
          }
          return stored;
        };
    auto emit_rhs_register_store_word_at_stage =
        [&](Value destination, Value vector, int64_t copy, int64_t word,
            Value shared_stage) {
          CHECK(tensile_double_buffer);
          CHECK_GE(copy, 0);
          CHECK_LT(copy, 7);
          CHECK_GE(word, 0);
          CHECK_LT(word, 4);
          const int64_t rhs_shared_base = lhs_shared_elements;
          Value stage_base = Mul(
              builder, shared_stage,
              IndexConstant(builder, kTensileStageStrideElements));
          Value store_thread_id = thread_id;
          if (tensile_direct_lhs) {
            Value source_row = Div(builder, thread_id, rhs_vectors_per_row);
            Value low_half = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ult, source_row,
                IndexConstant(builder, 16));
            Value interleaved_row = mlir::arith::SelectOp::create(
                builder, low_half,
                Mul(builder, source_row, IndexConstant(builder, 2)),
                Add(builder,
                    Mul(builder,
                        Sub(builder, source_row, IndexConstant(builder, 16)),
                        IndexConstant(builder, 2)),
                    IndexConstant(builder, 1)));
            store_thread_id = Add(
                builder,
                Mul(builder, interleaved_row,
                    IndexConstant(builder, rhs_vectors_per_row)),
                Rem(builder, thread_id, rhs_vectors_per_row));
          }
          Value thread_offset = Add(
              builder,
              Mul(builder, store_thread_id,
                  IndexConstant(builder, load_vector_width)),
              Mul(builder, Div(builder, store_thread_id, 16),
                  IndexConstant(builder, 16)));
          Value shared_index =
              Add(builder, IndexConstant(builder, rhs_shared_base),
                  Add(builder, stage_base,
                      Add(builder, thread_offset,
                          IndexConstant(
                              builder,
                              copy * kTensilePhysicalBlockElements +
                                  word * 2))));
          auto word_type =
              mlir::VectorType::get({2}, builder.getBF16Type());
          llvm::SmallVector<int64_t, 2> word_mask = {word * 2, word * 2 + 1};
          Value word_value = mlir::vector::ShuffleOp::create(
              builder, word_type, vector, vector, word_mask);
          return mlir::vector::TransferWriteOp::create(
                     builder, word_value, destination,
                     mlir::ValueRange{shared_index},
                     llvm::ArrayRef<bool>{true})
              .getResult();
        };
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
                IndexConstant(builder, load_vector_width));
        // Iterate over contiguous physical LDS addresses and invert the XOR
        // swizzle to find the corresponding logical global-memory column.
        Value logical_k = SwizzleXor16(builder, shared_row, shared_k, stage_k_);
        Value global_row = Add(builder, block_m_base, shared_row);
        Value loaded;
        if (m_ % block_m_ == 0) {
          if (async_lhs_) {
            Value source_linear_index = input_linear_index(
                global_row, Add(builder, global_k, logical_k));
            loaded =
                emit_buffer_load(lhs, source_linear_index, load_vector_type);
          } else {
            loaded = mlir::vector::TransferReadOp::create(
                builder, load_vector_type, lhs,
                mlir::ValueRange{global_row, Add(builder, global_k, logical_k)},
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
        written =
            mlir::vector::TransferWriteOp::create(
                builder, loaded, written,
                mlir::ValueRange{
                    Add(builder, Add(builder, stage_base, row_base), shared_k)},
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
        Value copy_index = Add(builder, thread_id,
                               IndexConstant(builder, copy * block_threads));
        Value physical_linear =
            Mul(builder, copy_index, IndexConstant(builder, kCopyElements));
        Value shared_row = Div(builder, physical_linear, stage_k_);
        Value shared_k = Rem(builder, physical_linear, stage_k_);
        Value logical_k = SwizzleXor16(builder, shared_row, shared_k, stage_k_);
        Value destination_wave_base = Add(
            builder,
            Mul(builder, shared_stage,
                IndexConstant(builder, block_m_ * stage_k_)),
            Add(builder, direct_copy_wave_offset,
                IndexConstant(builder, copy * block_threads * kCopyElements)));
        mlir::OperationState copy_state(entry_function.getLoc(),
                                        "xla_gpu.async_copy_global_to_shared");
        Value source_linear_index = input_linear_index(
            Add(builder, block_m_base, shared_row),
            Add(builder, global_k, logical_k));
        copy_state.addOperands(
            {lhs, source_linear_index, copied, destination_wave_base});
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
      const int64_t rhs_shared_base = lhs_shared_elements;
      if (rhs_k_contiguous_) {
        constexpr int64_t kCopyElements = 2;
        const int64_t copies = block_n_ * stage_k_ / kCopyElements;
        const int64_t block_threads = num_warps_ * 64;
        const int64_t copies_per_thread = copies / block_threads;
        Value copied = destination;
        for (int64_t copy = 0; copy < copies_per_thread; ++copy) {
          Value copy_index = Add(builder, thread_id,
                                 IndexConstant(builder, copy * block_threads));
          Value physical_linear =
              Mul(builder, copy_index, IndexConstant(builder, kCopyElements));
          Value shared_n = Div(builder, physical_linear, stage_k_);
          Value shared_k = Rem(builder, physical_linear, stage_k_);
          Value logical_k = SwizzleXor16(builder, shared_n, shared_k, stage_k_);
          Value destination_wave_base =
              Add(builder, IndexConstant(builder, rhs_shared_base),
                  Add(builder,
                      Mul(builder, shared_stage,
                          IndexConstant(builder, block_n_ * stage_k_)),
                      Add(builder, direct_copy_wave_offset,
                          IndexConstant(
                              builder, copy * block_threads * kCopyElements))));
          mlir::OperationState copy_state(
              entry_function.getLoc(), "xla_gpu.async_copy_global_to_shared");
          Value source_linear_index = input_linear_index(
              Add(builder, block_n_base, shared_n),
              Add(builder, global_k, logical_k));
          copy_state.addOperands(
              {rhs, source_linear_index, copied, destination_wave_base});
          copy_state.addTypes(destination.getType());
          copy_state.addAttribute("copy_bytes", builder.getI32IntegerAttr(4));
          copied = builder.create(copy_state)->getResult(0);
        }
        return copied;
      }
      const int64_t rhs_vectors = block_n_ * stage_k_ / load_vector_width;
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
          const int64_t vectors_per_n = stage_k_ / load_vector_width;
          shared_n = Div(builder, vector_index, vectors_per_n);
          shared_k = Mul(builder, Rem(builder, vector_index, vectors_per_n),
                         IndexConstant(builder, load_vector_width));
          mlir::AffineMap k_vector_map = mlir::AffineMap::get(
              /*dimCount=*/2, /*symbolCount=*/0,
              builder.getAffineDimExpr(rhs_contracting_dimension_),
              builder.getContext());
          llvm::SmallVector<Value, 2> indices = rhs_indices(
              Add(builder, global_k, shared_k),
              Add(builder, block_n_base, shared_n));
          loaded = mlir::vector::TransferReadOp::create(
              builder, load_vector_type, rhs, indices,
              /*padding=*/std::nullopt, k_vector_map,
              llvm::ArrayRef<bool>{true});
        } else {
          const int64_t vectors_per_k = block_n_ / load_vector_width;
          shared_k = Div(builder, vector_index, vectors_per_k);
          Value shared_n_vector = Rem(builder, vector_index, vectors_per_k);
          shared_n = Mul(builder, shared_n_vector,
                         IndexConstant(builder, load_vector_width));
          mlir::AffineMap n_vector_map = mlir::AffineMap::get(
              /*dimCount=*/2, /*symbolCount=*/0,
              builder.getAffineDimExpr(1 - rhs_contracting_dimension_),
              builder.getContext());
          llvm::SmallVector<Value, 2> indices = rhs_indices(
              Add(builder, global_k, shared_k),
              Add(builder, block_n_base, shared_n));
          loaded = mlir::vector::TransferReadOp::create(
              builder, load_vector_type, rhs, indices,
              /*padding=*/std::nullopt, n_vector_map,
              llvm::ArrayRef<bool>{true});
        }

        Value written = load_loop.getRegionIterArg(0);
        for (int64_t element = 0; element < load_vector_width; ++element) {
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
      Value group_k = Add(
          builder,
          Mul(builder, lane_group, IndexConstant(builder, async_lhs_ ? 8 : 4)),
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
            // A K-contiguous RHS is the native physical layout used by
            // FlyDSL GEMMs. It can be represented either as column-major
            // [K,N] contracting dimension 0 or row-major [N,K] contracting
            // dimension 1; both make one lane's K values a vector load.
            mlir::AffineMap k_vector_map = mlir::AffineMap::get(
                /*dimCount=*/2, /*symbolCount=*/0,
                builder.getAffineDimExpr(rhs_contracting_dimension_),
                builder.getContext());
            if (async_lhs_) {
              Value source_linear_index = input_linear_index(
                  global_column, Add(builder, global_k, group_k));
              return emit_buffer_load(rhs, source_linear_index,
                                      rhs_vector_type);
            }
            llvm::SmallVector<Value, 2> indices = rhs_indices(
                Add(builder, global_k, group_k), global_column);
            return mlir::vector::TransferReadOp::create(
                builder, rhs_vector_type, rhs, indices,
                /*padding=*/std::nullopt, k_vector_map,
                llvm::ArrayRef<bool>{true});
          }
          llvm::SmallVector<Value, 8> elements;
          const int64_t vector_elements = async_lhs_ ? 8 : 4;
          for (int64_t element = 0; element < vector_elements; ++element) {
            Value k_index =
                Add(builder, group_k, IndexConstant(builder, element));
            llvm::SmallVector<Value, 2> indices = rhs_indices(
                Add(builder, global_k, k_index), global_column);
            elements.push_back(
                mlir::tensor::ExtractOp::create(builder, rhs, indices));
          }
          return mlir::vector::FromElementsOp::create(builder, rhs_vector_type,
                                                      elements);
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
            mlir::scf::YieldOp::create(builder, async_lhs_
                                                    ? zero_staged_input_vector
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
      for (int64_t k_offset = 0; k_offset < stage_k_; k_offset += rhs_k_step) {
        llvm::SmallVector<Value> group = emit_rhs_group(global_k, k_offset);
        fragments.append(group);
      }
      return fragments;
    };
    auto emit_rhs_lds_fragment = [&](Value shared, Value shared_stage,
                                     int64_t k_offset,
                                     int64_t tile_column) -> Value {
      CHECK(!triton_vec4_lds);
      const int64_t rhs_shared_base = lhs_shared_elements;
      Value group_k =
          Add(builder, Mul(builder, lane_group, IndexConstant(builder, 8)),
              Add(builder, local_split_k_offset,
                  IndexConstant(builder, k_offset)));
      if (tensile_double_buffer) {
        constexpr int64_t kPaddedRowStride = 2 * 64 + 16;
        Value dynamic_offset =
            Add(builder,
                Mul(builder, lane_axis,
                    IndexConstant(builder, kPaddedRowStride)),
                group_k);
        const int64_t static_offset =
            (tile_column / 2) * kTensilePhysicalBlockElements +
            (tile_column % 2) * 64;
        Value stage_base = Mul(
            builder, shared_stage,
            IndexConstant(builder, kTensileStageStrideElements));
        Value shared_index =
            Add(builder, IndexConstant(builder, rhs_shared_base),
                Add(builder, stage_base,
                    Add(builder, dynamic_offset,
                        IndexConstant(builder, static_offset))));
        return mlir::vector::TransferReadOp::create(
            builder, staged_input_vector_type, shared,
            mlir::ValueRange{shared_index}, /*padding=*/std::nullopt,
            llvm::ArrayRef<bool>{true});
      }
      Value row = Add(builder, wave_column_offset,
                      Add(builder, lane_axis,
                          IndexConstant(builder, tile_column * atom_m)));
      Value physical_row =
          local_split_k_ ? local_split_physical_row(row) : row;
      Value physical_k = local_split_k_
                             ? group_k
                             : SwizzleXor16(builder, row, group_k, stage_k_);
      Value stage_base = Mul(builder, shared_stage,
                             IndexConstant(builder, rhs_stage_elements));
      Value physical_element =
          Add(builder,
              Mul(builder, physical_row, IndexConstant(builder, stage_k_)),
              physical_k);
      if (local_split_k_) {
        physical_element = add_local_split_lds_padding(physical_element);
      }
      Value shared_index = Add(
          builder, IndexConstant(builder, rhs_shared_base),
          Add(builder, stage_base, physical_element));
      return mlir::vector::TransferReadOp::create(
          builder, staged_input_vector_type, shared,
          mlir::ValueRange{shared_index}, /*padding=*/std::nullopt,
          llvm::ArrayRef<bool>{true});
    };
    auto emit_rhs_lds_group = [&](Value shared, Value shared_stage,
                                  int64_t k_offset) {
      const int64_t rhs_shared_base = lhs_shared_elements;
      llvm::SmallVector<Value> fragments;
      fragments.reserve(wave_tile_columns);
      Value group_k =
          Add(builder, Mul(builder, lane_group, IndexConstant(builder, 8)),
              Add(builder, local_split_k_offset,
                  IndexConstant(builder, k_offset)));
      if (triton_vec4_lds) {
        CHECK_EQ(wave_tile_columns % 2, 0);
        llvm::SmallVector<Value> low_fragments(wave_tile_columns);
        llvm::SmallVector<Value> high_fragments(wave_tile_columns);
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             tile_column += 2) {
          Value row = Add(builder, wave_column_offset,
                          Add(builder, lane_axis,
                              IndexConstant(builder, tile_column * atom_m)));
          for (int64_t half = 0; half < 2; ++half) {
            Value physical_k = SwizzleTritonVec4(
                builder, row,
                Add(builder, group_k, IndexConstant(builder, half * 4)));
            Value stage_base = Mul(builder, shared_stage,
                                   IndexConstant(builder, block_n_ * stage_k_));
            Value first_index =
                Add(builder, IndexConstant(builder, rhs_shared_base),
                    Add(builder, stage_base,
                        Add(builder,
                            Mul(builder, row, IndexConstant(builder, stage_k_)),
                            physical_k)));
            Value first = mlir::vector::TransferReadOp::create(
                builder, lds_vector_type, shared, mlir::ValueRange{first_index},
                /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
            Value second_index = Add(builder, first_index,
                                     IndexConstant(builder, atom_m * stage_k_));
            Value second = mlir::vector::TransferReadOp::create(
                builder, lds_vector_type, shared,
                mlir::ValueRange{second_index}, /*padding=*/std::nullopt,
                llvm::ArrayRef<bool>{true});
            (half == 0 ? low_fragments : high_fragments)[tile_column] = first;
            (half == 0 ? low_fragments : high_fragments)[tile_column + 1] =
                second;
          }
        }
        fragments.append(low_fragments);
        fragments.append(high_fragments);
        return fragments;
      }
      for (int64_t tile_column = 0; tile_column < wave_tile_columns;
           ++tile_column) {
        fragments.push_back(emit_rhs_lds_fragment(
            shared, shared_stage, k_offset, tile_column));
      }
      return fragments;
    };
    auto emit_lhs_lds_fragment = [&](Value shared, Value shared_stage,
                                     int64_t k_offset,
                                     int64_t tile_row) -> Value {
      CHECK(!triton_vec4_lds);
      Value group_k =
          Add(builder, Mul(builder, lane_group, IndexConstant(builder, 8)),
              Add(builder, local_split_k_offset,
                  IndexConstant(builder, k_offset)));
      if (tensile_double_buffer && tensile_direct_rhs) {
        constexpr int64_t kPaddedRowStride = 2 * 64 + 16;
        Value dynamic_offset =
            Add(builder,
                Mul(builder, lane_axis,
                    IndexConstant(builder, kPaddedRowStride)),
                group_k);
        const int64_t static_offset =
            (tile_row / 2) * kTensilePhysicalBlockElements +
            (tile_row % 2) * 64;
        Value stage_base = Mul(
            builder, shared_stage,
            IndexConstant(builder, kTensileStageStrideElements));
        Value shared_index =
            Add(builder, stage_base,
                Add(builder, dynamic_offset,
                    IndexConstant(builder, static_offset)));
        return mlir::vector::TransferReadOp::create(
            builder, staged_input_vector_type, shared,
            mlir::ValueRange{shared_index}, /*padding=*/std::nullopt,
            llvm::ArrayRef<bool>{true});
      }
      Value row = Add(
          builder, wave_row_offset,
          Add(builder, lane_axis, IndexConstant(builder, tile_row * atom_m)));
      Value physical_row =
          local_split_k_ ? local_split_physical_row(row) : row;
      Value physical_k = local_split_k_
                             ? group_k
                             : SwizzleXor16(builder, row, group_k, stage_k_);
      Value stage_base = Mul(builder, shared_stage,
                             IndexConstant(builder, lhs_stage_elements));
      Value physical_element =
          Add(builder,
              Mul(builder, physical_row, IndexConstant(builder, stage_k_)),
              physical_k);
      if (local_split_k_) {
        physical_element = add_local_split_lds_padding(physical_element);
      }
      Value shared_index = Add(builder, stage_base, physical_element);
      return mlir::vector::TransferReadOp::create(
          builder, staged_input_vector_type, shared,
          mlir::ValueRange{shared_index}, /*padding=*/std::nullopt,
          llvm::ArrayRef<bool>{true});
    };
    auto emit_lhs_lds_group = [&](Value shared, Value shared_stage,
                                  int64_t k_offset) {
      llvm::SmallVector<Value> fragments;
      fragments.reserve(wave_tile_rows);
      Value group_k =
          Add(builder, Mul(builder, lane_group, IndexConstant(builder, 8)),
              Add(builder, local_split_k_offset,
                  IndexConstant(builder, k_offset)));
      if (triton_vec4_lds) {
        CHECK_EQ(wave_tile_rows % 2, 0);
        llvm::SmallVector<Value> low_fragments(wave_tile_rows);
        llvm::SmallVector<Value> high_fragments(wave_tile_rows);
        for (int64_t tile_row = 0; tile_row < wave_tile_rows; tile_row += 2) {
          Value row = Add(builder, wave_row_offset,
                          Add(builder, lane_axis,
                              IndexConstant(builder, tile_row * atom_m)));
          for (int64_t half = 0; half < 2; ++half) {
            Value physical_k = SwizzleTritonVec4(
                builder, row,
                Add(builder, group_k, IndexConstant(builder, half * 4)));
            Value stage_base = Mul(builder, shared_stage,
                                   IndexConstant(builder, block_m_ * stage_k_));
            Value first_index =
                Add(builder, stage_base,
                    Add(builder,
                        Mul(builder, row, IndexConstant(builder, stage_k_)),
                        physical_k));
            Value first = mlir::vector::TransferReadOp::create(
                builder, lds_vector_type, shared, mlir::ValueRange{first_index},
                /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
            Value second_index = Add(builder, first_index,
                                     IndexConstant(builder, atom_m * stage_k_));
            Value second = mlir::vector::TransferReadOp::create(
                builder, lds_vector_type, shared,
                mlir::ValueRange{second_index}, /*padding=*/std::nullopt,
                llvm::ArrayRef<bool>{true});
            (half == 0 ? low_fragments : high_fragments)[tile_row] = first;
            (half == 0 ? low_fragments : high_fragments)[tile_row + 1] = second;
          }
        }
        fragments.append(low_fragments);
        fragments.append(high_fragments);
        return fragments;
      }
      for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
        fragments.push_back(
            emit_lhs_lds_fragment(shared, shared_stage, k_offset, tile_row));
      }
      return fragments;
    };
    auto emit_lhs_lds_fragments = [&](Value shared, Value shared_stage) {
      llvm::SmallVector<Value> fragments;
      const int64_t k_groups = compute_stage_k / (2 * atom_k);
      fragments.reserve(k_groups * wave_tile_rows);
      for (int64_t group = 0; group < k_groups; ++group) {
        llvm::SmallVector<Value> values =
            emit_lhs_lds_group(shared, shared_stage, group * 2 * atom_k);
        fragments.append(values);
      }
      return fragments;
    };
    auto emit_rhs_lds_fragments = [&](Value shared, Value shared_stage) {
      llvm::SmallVector<Value> fragments;
      const int64_t k_groups = compute_stage_k / (2 * atom_k);
      fragments.reserve(k_groups * wave_tile_columns);
      for (int64_t group = 0; group < k_groups; ++group) {
        llvm::SmallVector<Value> values =
            emit_rhs_lds_group(shared, shared_stage, group * 2 * atom_k);
        fragments.append(values);
      }
      return fragments;
    };
    auto emit_tensile_direct_rhs_fragment = [&](Value global_k,
                                                int64_t fragment) {
      CHECK(tensile_direct_rhs);
      const int64_t k_groups = stage_k_ / (2 * atom_k);
      CHECK_GE(fragment, 0);
      CHECK_LT(fragment, k_groups * wave_tile_columns);
      const int64_t group = fragment / wave_tile_columns;
      const int64_t tile_column = fragment % wave_tile_columns;
      Value row = Add(builder, wave_column_offset,
                      Add(builder, lane_axis,
                          IndexConstant(builder, tile_column * atom_m)));
      Value vector_linear_index = input_linear_index(
          Add(builder, block_n_base, row),
          Mul(builder, lane_group, IndexConstant(builder, 8)));
      Value scalar_linear_index = ReadFirstLaneIndex(
          builder,
          Add(builder, global_k,
              IndexConstant(builder, group * 2 * atom_k)));
      return emit_split_buffer_load(rhs, vector_linear_index,
                                    scalar_linear_index,
                                    staged_input_vector_type);
    };
    auto emit_tensile_direct_rhs_fragments = [&](Value global_k) {
      const int64_t fragment_count =
          stage_k_ / (2 * atom_k) * wave_tile_columns;
      llvm::SmallVector<Value> fragments;
      fragments.reserve(fragment_count);
      for (int64_t fragment = 0; fragment < fragment_count; ++fragment) {
        fragments.push_back(
            emit_tensile_direct_rhs_fragment(global_k, fragment));
      }
      return fragments;
    };
    auto emit_tensile_direct_lhs_fragment = [&](Value global_k,
                                                int64_t fragment) {
      CHECK(tensile_direct_lhs);
      const int64_t k_groups = stage_k_ / (2 * atom_k);
      CHECK_GE(fragment, 0);
      CHECK_LT(fragment, k_groups * wave_tile_rows);
      const int64_t group = fragment / wave_tile_rows;
      const int64_t tile_row = fragment % wave_tile_rows;
      Value row = Add(
          builder, wave_row_offset,
          Add(builder,
              Mul(builder, lane_axis,
                  IndexConstant(builder, wave_tile_rows)),
              IndexConstant(builder, tile_row)));
      Value vector_linear_index = input_linear_index(
          Add(builder, block_m_base, row),
          Mul(builder, lane_group, IndexConstant(builder, 8)));
      Value scalar_linear_index = ReadFirstLaneIndex(
          builder,
          Add(builder, tensile_physical_k(global_k),
              IndexConstant(builder, group * 2 * atom_k)));
      return emit_split_buffer_load(lhs, vector_linear_index,
                                    scalar_linear_index,
                                    staged_input_vector_type);
    };
    auto emit_tensile_direct_lhs_fragments = [&](Value global_k) {
      const int64_t fragment_count = stage_k_ / (2 * atom_k) * wave_tile_rows;
      llvm::SmallVector<Value> fragments;
      fragments.reserve(fragment_count);
      for (int64_t fragment = 0; fragment < fragment_count; ++fragment) {
        fragments.push_back(
            emit_tensile_direct_lhs_fragment(global_k, fragment));
      }
      return fragments;
    };

    // Match FlyDSL's two-stage HGEMM pipeline: retain the current RHS tile in
    // registers while issuing the next tile's VMEM loads. Larger K tiles would
    // double too many live fragments, so leave those to the streaming path.
    const bool prefetch_rhs = prefetch_rhs_ && !stage_rhs_;
    const bool load_rhs_on_demand =
        single_buffer_lds_ && block_m_ == 128 && block_n_ == 128;
    Value initial_shared = lhs_shared;
    if (single_buffer_lds_ || tensile_double_buffer) {
      llvm::SmallVector<Value> initial_rhs_registers;
      if (!tensile_direct_rhs) {
        initial_rhs_registers = emit_rhs_register_vectors(lower_bound);
      }
      llvm::SmallVector<Value> initial_lhs_registers;
      if (!tensile_direct_lhs) {
        initial_lhs_registers = emit_lhs_register_vectors(lower_bound);
      }
      emit_wait_vmcnt(0);
      if (!tensile_direct_rhs) {
        initial_shared = emit_rhs_register_stores_at_stage(
            initial_shared, initial_rhs_registers, 0,
            initial_rhs_registers.size(), lower_bound);
      }
      if (!tensile_direct_lhs) {
        initial_shared = emit_lhs_register_stores_at_stage(
            initial_shared, initial_lhs_registers, 0,
            initial_lhs_registers.size(), lower_bound);
      }
    } else if (stage_rhs_) {
      initial_shared =
          emit_rhs_lds_stage(initial_shared, lower_bound, lower_bound);
    }
    if (!single_buffer_lds_ && !tensile_double_buffer &&
        !tensile_direct_lhs) {
      initial_shared =
          async_lhs_ ? emit_lhs_register_stage(initial_shared, lower_bound,
                                               lower_bound)
                     : emit_lhs_stage(initial_shared, lower_bound, lower_bound);
    }
    llvm::SmallVector<Value> initial_loop_values = initial_accumulators;
    if (prefetch_rhs) {
      llvm::SmallVector<Value> initial_rhs = emit_rhs_stage(lower_bound);
      initial_loop_values.append(initial_rhs);
    }
    if ((stage_rhs_ || async_lhs_) && schedule_instructions_) {
      mlir::ROCDL::SchedBarrier::create(builder,
                                        mlir::ROCDL::SchedGroupMask::none);
    }
    if (stage_rhs_ || async_lhs_) {
      // The CDNA global-to-LDS copy increments vmcnt. A workgroup barrier does
      // not retire those VMEM operations, so wait before any wave consumes the
      // newly populated stage.
      emit_wait_vmcnt(0);
    }
    initial_shared =
        SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                              initial_shared)
            .getResult(0);
    if (preload_lds_fragments_ && !load_rhs_on_demand) {
      llvm::SmallVector<Value> initial_rhs =
          tensile_direct_lhs
              ? emit_tensile_direct_lhs_fragments(lower_bound)
              : (tensile_direct_rhs
                     ? emit_tensile_direct_rhs_fragments(lower_bound)
                     : (local_split_k_
                            ? emit_rhs_lds_group(initial_shared, lower_bound, 0)
                            : emit_rhs_lds_fragments(initial_shared,
                                                     lower_bound)));
      initial_loop_values.append(initial_rhs);
      if (local_split_k_) {
        llvm::SmallVector<Value> initial_lhs =
            emit_lhs_lds_group(initial_shared, lower_bound, 0);
        initial_loop_values.append(initial_lhs);
      }
      mlir::ROCDL::SchedBarrier::create(builder,
                                        mlir::ROCDL::SchedGroupMask::none);
    }
    const int64_t preloaded_fragment_end = initial_loop_values.size();
    int64_t prefetched_rhs_loop_index = -1;
    int64_t prefetched_lhs_loop_index = -1;
    int64_t prefetched_rhs_count = 0;
    int64_t prefetched_lhs_count = 0;
    if (tensile_wide_tile || local_split_k_) {
      // Keep a full global-memory tile one iteration ahead of LDS. The hot
      // loop consumes this buffer and refills it under the current MFMAs.
      llvm::SmallVector<Value> prefetched_rhs;
      if (!tensile_direct_rhs) {
        prefetched_rhs = emit_rhs_register_vectors(step);
      }
      llvm::SmallVector<Value> prefetched_lhs;
      if (!tensile_direct_lhs) {
        prefetched_lhs = emit_lhs_register_vectors(step);
      }
      prefetched_rhs_loop_index = initial_loop_values.size();
      prefetched_rhs_count = prefetched_rhs.size();
      initial_loop_values.append(prefetched_rhs);
      prefetched_lhs_loop_index = initial_loop_values.size();
      prefetched_lhs_count = prefetched_lhs.size();
      initial_loop_values.append(prefetched_lhs);
    }
    int64_t preloaded_staged_group_loop_index = -1;
    int64_t preloaded_staged_group_count = 0;
    if (tensile_direct_operand) {
      // Enter the modulo loop with the first staged K32 group resident in
      // VGPRs, as Tensile does. Each half produces the corresponding group
      // for its successor at the closing refill barrier.
      llvm::SmallVector<Value> preloaded_staged_group =
          tensile_direct_lhs
              ? emit_rhs_lds_group(initial_shared, lower_bound, 0)
              : emit_lhs_lds_group(initial_shared, lower_bound, 0);
      preloaded_staged_group_loop_index = initial_loop_values.size();
      preloaded_staged_group_count = preloaded_staged_group.size();
      initial_loop_values.append(preloaded_staged_group);
      mlir::ROCDL::SchedBarrier::create(
          builder, mlir::ROCDL::SchedGroupMask::none);
    }
    int64_t shared_loop_index = -1;
    if (single_buffer_lds_ || stage_rhs_ || async_lhs_) {
      // The LDS tensor is an SSA value. Carry the staged tile through the loop
      // explicitly so fragment reads retain their dependency on the stores
      // and barriers that produced the current stage.
      shared_loop_index = initial_loop_values.size();
      initial_loop_values.push_back(initial_shared);
    }
    auto emit_compute_tile = [&](Value staged_shared, Value current_stage,
                                 Value k_base,
                                 llvm::ArrayRef<Value> accumulators,
                                 llvm::ArrayRef<Value> prefetched_rhs_values) {
      llvm::SmallVector<Value> next_accumulators(accumulators);
      const int64_t compute_k_step = paired_mfma ? 2 * atom_k : atom_k;
      for (int64_t k_offset = 0; k_offset < stage_k_;
           k_offset += compute_k_step) {
        Value group_k = Add(builder,
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
          if (paired_mfma) {
            lhs_vectors =
                emit_lhs_lds_group(staged_shared, current_stage, k_offset);
          } else {
            lhs_vectors.reserve(wave_tile_rows);
            for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
              Value row = Add(builder, wave_row_offset,
                              Add(builder, lane_axis,
                                  IndexConstant(builder, tile_row * atom_m)));
              Value swizzled_group_k =
                  SwizzleXor16(builder, row, group_k, stage_k_);
              Value shared_index =
                  Add(builder,
                      Mul(builder, current_stage,
                          IndexConstant(builder, block_m_ * stage_k_)),
                      Add(builder,
                          Mul(builder, row, IndexConstant(builder, stage_k_)),
                          swizzled_group_k));
              lhs_vectors.push_back(mlir::vector::TransferReadOp::create(
                  builder, input_vector_type, staged_shared,
                  mlir::ValueRange{shared_index}, /*padding=*/std::nullopt,
                  llvm::ArrayRef<bool>{true}));
            }
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
                get_accumulator_index(tile_row, tile_column);
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
              mma_state.addOperands({mma_atom, lhs_fragment, rhs_fragment,
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

    auto emit_compute_preloaded = [&](llvm::ArrayRef<Value> lhs_fragments,
                                      llvm::ArrayRef<Value> rhs_fragments,
                                      llvm::ArrayRef<Value> accumulators,
                                      auto&& before_mfma) {
      llvm::SmallVector<Value> next_accumulators(accumulators);
      const int64_t k_groups = compute_stage_k / (2 * atom_k);
      int64_t mfma_index = 0;
      if (triton_vec4_lds) {
        for (int64_t group = 0; group < k_groups; ++group) {
          const int64_t lhs_group_base = group * 2 * wave_tile_rows;
          const int64_t rhs_group_base = group * 2 * wave_tile_columns;
          for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
            for (int64_t tile_column = 0; tile_column < wave_tile_columns;
                 ++tile_column) {
              const int64_t accumulator_index =
                  get_accumulator_index(tile_row, tile_column);
              for (int64_t mfma = 0; mfma < 2; ++mfma) {
                before_mfma(mfma_index++);
                Value lhs_mfma =
                    lhs_fragments[lhs_group_base + mfma * wave_tile_rows +
                                  tile_row];
                Value rhs_mfma =
                    rhs_fragments[rhs_group_base + mfma * wave_tile_columns +
                                  tile_column];
                mlir::OperationState mma_state(entry_function.getLoc(),
                                               "fly.mma_atom_call_ssa");
                mma_state.addOperands({mma_atom, lhs_mfma, rhs_mfma,
                                       next_accumulators[accumulator_index]});
                mma_state.addTypes(accumulator_type);
                next_accumulators[accumulator_index] =
                    builder.create(mma_state)->getResult(0);
              }
            }
          }
        }
        return next_accumulators;
      }
      constexpr int64_t kMfmasPerFragment = 2;
      for (int64_t group = 0; group < k_groups; ++group) {
        for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
          Value lhs_fragment = lhs_fragments[group * wave_tile_rows + tile_row];
          for (int64_t tile_column = 0; tile_column < wave_tile_columns;
               ++tile_column) {
            Value rhs_fragment =
                rhs_fragments[group * wave_tile_columns + tile_column];
            const int64_t accumulator_index =
                get_accumulator_index(tile_row, tile_column);
            for (int64_t mfma = 0; mfma < kMfmasPerFragment; ++mfma) {
              before_mfma(mfma_index++);
              const int64_t fragment_base = mfma * 4;
              llvm::SmallVector<int64_t, 4> fragment_mask = {
                  fragment_base, fragment_base + 1, fragment_base + 2,
                  fragment_base + 3};
              Value lhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, lhs_fragment, lhs_fragment,
                  fragment_mask);
              Value rhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, rhs_fragment, rhs_fragment,
                  fragment_mask);
              mlir::OperationState mma_state(entry_function.getLoc(),
                                             "fly.mma_atom_call_ssa");
              // The native wide tile computes the transposed MFMA view. This
              // maps each lane's four accumulator elements to adjacent N
              // columns, enabling packed stores into XLA's row-major output.
              mma_state.addOperands(
                  source_swap_mfma
                      ? mlir::ValueRange{mma_atom, rhs_mfma, lhs_mfma,
                                         next_accumulators[accumulator_index]}
                      : mlir::ValueRange{mma_atom, lhs_mfma, rhs_mfma,
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

    auto emit_compute_local_split =
        [&](Value shared, Value shared_stage,
            llvm::ArrayRef<Value> first_lhs_fragments,
            llvm::ArrayRef<Value> first_rhs_fragments,
            llvm::ArrayRef<Value> accumulators,
            auto&& before_mfma,
            auto&& after_mfma) -> llvm::SmallVector<Value> {
      CHECK(local_split_k_);
      CHECK_EQ(compute_stage_k, 64);
      CHECK_EQ(wave_tile_rows, 4);
      CHECK_EQ(wave_tile_columns, 4);
      CHECK_EQ(first_lhs_fragments.size(), wave_tile_rows);
      CHECK_EQ(first_rhs_fragments.size(), wave_tile_columns);

      llvm::SmallVector<Value> next_accumulators(accumulators);
      int64_t mfma_index = 0;
      constexpr int64_t kMfmasPerFragment = 2;
      constexpr int64_t kGroups = 2;
      llvm::SmallVector<Value> second_lhs_fragments(wave_tile_rows);
      llvm::SmallVector<Value> second_rhs_fragments(wave_tile_columns);
      auto emit_scheduled_read = [&](auto&& emit_read) {
        mlir::ROCDL::SchedBarrier::create(
            builder, mlir::ROCDL::SchedGroupMask::none);
        Value value = emit_read();
        mlir::ROCDL::SchedBarrier::create(
            builder, mlir::ROCDL::SchedGroupMask::none);
        return value;
      };
      for (int64_t group = 0; group < kGroups; ++group) {
        llvm::SmallVector<Value> lhs_fragments;
        llvm::SmallVector<Value> rhs_fragments;
        if (group == 0) {
          lhs_fragments.append(first_lhs_fragments.begin(),
                               first_lhs_fragments.end());
          rhs_fragments.append(first_rhs_fragments.begin(),
                               first_rhs_fragments.end());
        } else {
          lhs_fragments = std::move(second_lhs_fragments);
          rhs_fragments = std::move(second_rhs_fragments);
        }
        // SourceSwap traverses N first in Tensile: one MFMA half visits all
        // sixteen independent accumulators before returning to their second
        // half. This avoids serial back-to-back updates to one accumulator.
        for (int64_t mfma = 0; mfma < kMfmasPerFragment; ++mfma) {
          for (int64_t tile_column = 0; tile_column < wave_tile_columns;
               ++tile_column) {
            for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
              const int64_t accumulator_index =
                  get_accumulator_index(tile_row, tile_column);
              const int64_t current_mfma = mfma_index++;
              before_mfma(current_mfma);
              const int64_t fragment_base = mfma * 4;
              llvm::SmallVector<int64_t, 4> fragment_mask = {
                  fragment_base, fragment_base + 1, fragment_base + 2,
                  fragment_base + 3};
              Value lhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, lhs_fragments[tile_row],
                  lhs_fragments[tile_row], fragment_mask);
              Value rhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, rhs_fragments[tile_column],
                  rhs_fragments[tile_column], fragment_mask);
              mlir::OperationState mma_state(entry_function.getLoc(),
                                             "fly.mma_atom_call_ssa");
              mma_state.addOperands(
                  {mma_atom, rhs_mfma, lhs_mfma,
                   next_accumulators[accumulator_index]});
              mma_state.addTypes(accumulator_type);
              next_accumulators[accumulator_index] =
                  builder.create(mma_state)->getResult(0);
              if (group == 0) {
                // Exact eight-read K32 cadence from Tensile's LocalSplitU=2
                // loop. Each read is anchored immediately after its paired
                // MFMA so LLVM preserves the intended lgkmcnt progression.
                switch (current_mfma) {
                  case 0:
                    second_rhs_fragments[0] = emit_scheduled_read([&] {
                      return emit_rhs_lds_fragment(
                          shared, shared_stage, 2 * atom_k, 0);
                    });
                    break;
                  case 1:
                    second_lhs_fragments[0] = emit_scheduled_read([&] {
                      return emit_lhs_lds_fragment(
                          shared, shared_stage, 2 * atom_k, 0);
                    });
                    break;
                  case 2:
                    second_rhs_fragments[1] = emit_scheduled_read([&] {
                      return emit_rhs_lds_fragment(
                          shared, shared_stage, 2 * atom_k, 1);
                    });
                    break;
                  case 3:
                    second_rhs_fragments[2] = emit_scheduled_read([&] {
                      return emit_rhs_lds_fragment(
                          shared, shared_stage, 2 * atom_k, 2);
                    });
                    break;
                  case 6:
                    second_rhs_fragments[3] = emit_scheduled_read([&] {
                      return emit_rhs_lds_fragment(
                          shared, shared_stage, 2 * atom_k, 3);
                    });
                    break;
                  case 7:
                    second_lhs_fragments[1] = emit_scheduled_read([&] {
                      return emit_lhs_lds_fragment(
                          shared, shared_stage, 2 * atom_k, 1);
                    });
                    break;
                  case 8:
                    second_lhs_fragments[2] = emit_scheduled_read([&] {
                      return emit_lhs_lds_fragment(
                          shared, shared_stage, 2 * atom_k, 2);
                    });
                    break;
                  case 9:
                    second_lhs_fragments[3] = emit_scheduled_read([&] {
                      return emit_lhs_lds_fragment(
                          shared, shared_stage, 2 * atom_k, 3);
                    });
                    break;
                  default:
                    break;
                }
              }
              after_mfma(current_mfma);
            }
          }
        }
      }
      CHECK_EQ(mfma_index, 64);
      return next_accumulators;
    };

    auto emit_hot_loop_schedule = [&] {
      if (schedule_instructions_ ||
          (preload_lds_fragments_ && !single_buffer_lds_)) {
        // As in FlyDSL's hot-loop schedulers, place all scheduling groups at
        // the end of the region they control. LLVM's AMDGPU scheduler consumes
        // matching instructions bottom-up and interleaves each VMEM/LDS group
        // with the corresponding MFMA group.
        const int64_t k_groups =
            compute_stage_k / (paired_mfma ? 2 * atom_k : atom_k);
        if (preload_lds_fragments_) {
          if (local_split_k_) {
            // The LocalSplit path emits the K32 LDS/MFMA pipeline explicitly.
            // Do not let the generic preloaded-fragment schedule collect its
            // two groups back into one homogeneous LDS run.
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
            return;
          }
          if (tensile_wide_tile) {
            // This path emits the native wait/write/load/MFMA modulo schedule
            // directly. Keep LLVM from regrouping it into homogeneous runs.
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
            return;
          }
          // Exact B_TO_LDS schedule from FlyDSL's splitk_hgemm: all current A
          // fragments are read before next-tile DMA, then each A/B direct copy
          // is paired with two independent MFMAs.
          const int64_t block_threads = num_warps_ * 64;
          const int64_t copy_elements =
              single_buffer_lds_ ? load_vector_width : 2;
          const int64_t lhs_copies_per_thread =
              block_m_ * stage_k_ / copy_elements / block_threads;
          const int64_t rhs_copies_per_thread =
              block_n_ * stage_k_ / copy_elements / block_threads;
          const int64_t fragment_reads =
              k_groups *
              (wave_tile_rows + (load_rhs_on_demand ? wave_tile_columns : 0));
          int64_t remaining_mfmas =
              k_groups * wave_tile_rows * wave_tile_columns * 2;
          const int64_t total_copies =
              lhs_copies_per_thread + rhs_copies_per_thread;
          const int64_t mfmas_per_copy =
              single_buffer_lds_
                  ? (remaining_mfmas + total_copies - 1) / total_copies
                  : 2;
          for (int64_t load = 0; load < fragment_reads; ++load) {
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::ds_read, 1);
          }
          for (int64_t copy = 0; copy < total_copies; ++copy) {
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::vmem_read, 1);
            const int64_t scheduled_mfmas =
                std::min<int64_t>(mfmas_per_copy, remaining_mfmas);
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::mfma_wmma,
                          scheduled_mfmas);
            remaining_mfmas -= scheduled_mfmas;
          }
          for (int64_t mfma = 0; mfma < remaining_mfmas; ++mfma) {
            ScheduleGroup(builder, mlir::ROCDL::SchedGroupMask::mfma_wmma, 1);
          }
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
        } else if (stage_rhs_) {
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
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
        } else if (async_lhs_) {
          // FlyDSL's gfx942 B_TO_LDS=false scheduler first accounts for all
          // next-tile A DMA and B VGPR loads. It then pairs each group of A
          // LDS reads with the corresponding row of N-fragment MFMAs.
          constexpr int64_t kCopyElements = 2;
          const int64_t block_threads = num_warps_ * 64;
          const int64_t lhs_copies_per_thread =
              block_m_ * stage_k_ / kCopyElements / block_threads;
          const int64_t rhs_loads_per_thread = k_groups * wave_tile_columns;
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
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
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

        struct DirectHalfState {
          llvm::SmallVector<Value> accumulators;
          llvm::SmallVector<Value> direct;
          llvm::SmallVector<Value> staged_registers;
          llvm::SmallVector<Value> preloaded_staged_group;
          Value shared;
        };

        auto emit_direct_half =
            [&](Value half_k, Value half_shared, Value half_stage,
                Value refill_stage,
                llvm::ArrayRef<Value> half_accumulators,
                llvm::ArrayRef<Value> half_direct,
                llvm::ArrayRef<Value> half_staged_registers,
                llvm::ArrayRef<Value> half_preloaded_staged_group,
                bool prefetch_next_direct, bool refill_next_staged,
                bool load_future_staged,
                bool direct_has_trailing_staged_loads,
                bool preload_next_staged_group) -> DirectHalfState {
          llvm::SmallVector<Value> future_direct(half_direct);
          llvm::SmallVector<Value> future_staged =
              load_future_staged
                  ? llvm::SmallVector<Value>()
                  : llvm::SmallVector<Value>(half_staged_registers);
          llvm::SmallVector<Value> preloaded_next_staged_group =
              preload_next_staged_group
                  ? llvm::SmallVector<Value>()
                  : llvm::SmallVector<Value>(half_preloaded_staged_group);
          Value prefetch_k = Add(builder, half_k, step);
          Value future_k = Add(builder, prefetch_k, step);
          Value refilled_shared = half_shared;

          auto emit_staged_lds_batch = [&](Value source_shared,
                                           Value source_stage, int64_t group,
                                           int64_t fragment_begin,
                                           int64_t fragment_end) {
            llvm::SmallVector<Value> fragments;
            fragments.reserve(fragment_end - fragment_begin);
            for (int64_t fragment = fragment_begin;
                 fragment < fragment_end; ++fragment) {
              fragments.push_back(
                  tensile_direct_lhs
                      ? emit_rhs_lds_fragment(source_shared, source_stage,
                                              group * 2 * atom_k, fragment)
                      : emit_lhs_lds_fragment(source_shared, source_stage,
                                              group * 2 * atom_k, fragment));
            }
            return fragments;
          };

          llvm::SmallVector<Value> pipelined_staged_group;

          auto before_mfma = [&](int64_t mfma_index) {
            // Exact DirectToVgprA modulo schedule from the dispatched
            // hipBLASLt MT256x224x64 kernel.  The eight next-tile RHS vectors
            // are issued as four pairs while the current MFMAs run.
            if (prefetch_next_direct) {
              constexpr std::array<int64_t, 4> kDirectLoadPositions = {
                  35, 57, 79, 101};
              for (int64_t pair = 0; pair < kDirectLoadPositions.size();
                   ++pair) {
                if (mfma_index != kDirectLoadPositions[pair]) {
                  continue;
                }
                mlir::ROCDL::SchedBarrier::create(
                    builder, mlir::ROCDL::SchedGroupMask::none);
                const int64_t direct_group_width =
                    tensile_direct_lhs ? wave_tile_rows : wave_tile_columns;
                future_direct[pair] =
                    tensile_direct_lhs
                        ? emit_tensile_direct_lhs_fragment(prefetch_k, pair)
                        : emit_tensile_direct_rhs_fragment(prefetch_k, pair);
                future_direct[pair + direct_group_width] =
                    tensile_direct_lhs
                        ? emit_tensile_direct_lhs_fragment(
                              prefetch_k, pair + direct_group_width)
                        : emit_tensile_direct_rhs_fragment(
                              prefetch_k, pair + direct_group_width);
                mlir::ROCDL::SchedBarrier::create(
                    builder, mlir::ROCDL::SchedGroupMask::none);
                return;
              }
            }

            // Issue the second staged K32 group under the first group's MFMAs.
            // This mirrors Tensile's DirectToVgpr modulo schedule for either
            // operand orientation and prevents a bulk LDS wait at the group
            // boundary.
            constexpr std::array<int64_t, 14> kStagedGroupReadPositions = {
                1,  2,  3,  4,  7,  8,  9,
                10, 13, 14, 15, 16, 19, 20};
            for (int64_t fragment = 0;
                 fragment < kStagedGroupReadPositions.size(); ++fragment) {
              if (mfma_index != kStagedGroupReadPositions[fragment]) {
                continue;
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              llvm::SmallVector<Value> values = emit_staged_lds_batch(
                  half_shared, half_stage, /*group=*/1, fragment, fragment + 1);
              pipelined_staged_group.append(values);
              if (mfma_index == 4) {
                // Four new group-1 reads are now in flight. Retiring to four
                // also completes every older group-0 read, exactly as the
                // Tensile schedule does, so later group-0 MFMAs need no
                // fragment-by-fragment waits.
                emit_wait_lgkmcnt(4);
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }

            // Every current LDS fragment has retired before the single
            // buffer is overwritten.  The workgroup barrier at MFMA 192
            // closes all seven writes and opens the next tile for reading.
            if (mfma_index == 112) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              emit_wait_lgkmcnt(0);
              if (!tensile_double_buffer &&
                  half_preloaded_staged_group.empty()) {
                refilled_shared = SyncThreadsOp::create(
                                      builder, mlir::TypeRange{lhs_shared_type},
                                      refilled_shared)
                                      .getResult(0);
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }

            constexpr std::array<int64_t, 7> kLdsRefillPositions = {
                122, 134, 145, 156, 167, 178, 189};
            for (int64_t copy = 0; copy < kLdsRefillPositions.size(); ++copy) {
              if (mfma_index != kLdsRefillPositions[copy]) {
                continue;
              }
              if (!refill_next_staged ||
                  (tensile_double_buffer && copy == 5)) {
                continue;
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              emit_wait_vmcnt(load_future_staged ? 14 : 14 - copy);
              if (tensile_direct_lhs) {
                refilled_shared = emit_rhs_register_stores_at_stage(
                    refilled_shared, half_staged_registers, copy, copy + 1,
                    refill_stage);
              } else {
                refilled_shared = emit_lhs_register_stores_at_stage(
                    refilled_shared, half_staged_registers, copy, copy + 1,
                    refill_stage);
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }

            // Keep each replacement VMEM load late in the gap before the next
            // LDS write. This separates the two memory pipelines while leaving
            // the same 15-operation VMEM window as the native schedule. Copy 5
            // has a four-word LDS write in the double-buffered path and keeps
            // its dedicated replacement at MFMA 182 below.
            constexpr std::array<int64_t, 6> kFutureStagedLoadPositions = {
                131, 143, 154, 165, 176, 191};
            constexpr std::array<int64_t, 6> kFutureStagedLoadCopies = {
                0, 1, 2, 3, 4, 6};
            if (load_future_staged) {
              for (int64_t index = 0;
                   index < kFutureStagedLoadPositions.size(); ++index) {
                if (mfma_index != kFutureStagedLoadPositions[index]) {
                  continue;
                }
                const int64_t copy = kFutureStagedLoadCopies[index];
                mlir::ROCDL::SchedBarrier::create(
                    builder, mlir::ROCDL::SchedGroupMask::none);
                future_staged.push_back(
                    tensile_direct_lhs
                        ? emit_rhs_register_vector(future_k, copy)
                        : emit_lhs_register_vector(future_k, copy));
                mlir::ROCDL::SchedBarrier::create(
                    builder, mlir::ROCDL::SchedGroupMask::none);
                return;
              }
            }
            if (load_future_staged && !tensile_double_buffer &&
                mfma_index == 187) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              future_staged.push_back(
                  tensile_direct_lhs
                      ? emit_rhs_register_vector(future_k, /*copy=*/5)
                      : emit_lhs_register_vector(future_k, /*copy=*/5));
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }

            if (refill_next_staged && tensile_double_buffer &&
                mfma_index >= 178 &&
                mfma_index <= 181) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              if (mfma_index == 178) {
                emit_wait_vmcnt(load_future_staged ? 14 : 9);
              }
              refilled_shared =
                  tensile_direct_lhs
                      ? emit_rhs_register_store_word_at_stage(
                            refilled_shared, half_staged_registers[5],
                            /*copy=*/5, /*word=*/mfma_index - 178,
                            refill_stage)
                      : emit_lhs_register_store_word_at_stage(
                            refilled_shared, half_staged_registers[5],
                            /*copy=*/5, /*word=*/mfma_index - 178,
                            refill_stage);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }
            if (refill_next_staged && load_future_staged &&
                tensile_double_buffer && mfma_index == 182) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              future_staged.push_back(
                  tensile_direct_lhs
                      ? emit_rhs_register_vector(future_k, /*copy=*/5)
                      : emit_lhs_register_vector(future_k, /*copy=*/5));
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }

            if (refill_next_staged && mfma_index == 192) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              refilled_shared = SyncThreadsOp::create(
                                    builder, mlir::TypeRange{lhs_shared_type},
                                    refilled_shared)
                                    .getResult(0);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }

            // Tensile distributes the 14 next-iteration group-0 reads over
            // the 32 MFMAs after the refill barrier instead of making all 14
            // fragments live at once. Keep the same instruction positions.
            if (preload_next_staged_group) {
              constexpr std::array<int64_t, 14> kNextLdsReadPositions = {
                  193, 194, 195, 196, 199, 200, 201,
                  202, 205, 206, 207, 208, 211, 212};
              for (int64_t fragment = 0;
                   fragment < kNextLdsReadPositions.size(); ++fragment) {
                if (mfma_index != kNextLdsReadPositions[fragment]) {
                  continue;
                }
                mlir::ROCDL::SchedBarrier::create(
                    builder, mlir::ROCDL::SchedGroupMask::none);
                llvm::SmallVector<Value> values = emit_staged_lds_batch(
                    refilled_shared, refill_stage, /*group=*/0, fragment,
                    fragment + 1);
                preloaded_next_staged_group.append(values);
                mlir::ROCDL::SchedBarrier::create(
                    builder, mlir::ROCDL::SchedGroupMask::none);
                return;
              }
            }
          };

          // Eight current direct vectors precede the seven LDS payloads in
          // VMEM order.  Retire only the direct operand here; the payloads are
          // advanced one at a time by the modulo refill schedule.
          emit_wait_vmcnt(direct_has_trailing_staged_loads
                              ? half_staged_registers.size()
                              : 0);
          llvm::SmallVector<Value> next_accumulators(half_accumulators);
          int64_t mfma_index = 0;
          const int64_t k_groups = stage_k_ / (2 * atom_k);
          for (int64_t group = 0; group < k_groups; ++group) {
            // Tensile pipelines the second K32 LDS group under the first
            // group's MFMAs. This both hides LDS latency and guarantees every
            // read has retired before the single-buffer refill starts.
            llvm::SmallVector<Value> lhs_group;
            if (tensile_direct_lhs) {
              lhs_group = llvm::SmallVector<Value>(
                  half_direct.slice(group * wave_tile_rows, wave_tile_rows));
            } else if (group == 0 && !half_preloaded_staged_group.empty()) {
              lhs_group = llvm::SmallVector<Value>(half_preloaded_staged_group);
            } else if (group == 1 && !pipelined_staged_group.empty()) {
              CHECK_EQ(pipelined_staged_group.size(), wave_tile_rows);
              lhs_group = std::move(pipelined_staged_group);
            } else {
              lhs_group = emit_lhs_lds_group(half_shared, lower_bound,
                                             group * 2 * atom_k);
            }
            CHECK_EQ(lhs_group.size(), wave_tile_rows);
            llvm::SmallVector<Value> rhs_group;
            if (tensile_direct_rhs) {
              rhs_group = llvm::SmallVector<Value>(half_direct.slice(
                  group * wave_tile_columns, wave_tile_columns));
            } else if (group == 0 && !half_preloaded_staged_group.empty()) {
              rhs_group = llvm::SmallVector<Value>(half_preloaded_staged_group);
            } else if (group == 1 && !pipelined_staged_group.empty()) {
              CHECK_EQ(pipelined_staged_group.size(), wave_tile_columns);
              rhs_group = std::move(pipelined_staged_group);
            } else {
              rhs_group = emit_rhs_lds_group(half_shared, half_stage,
                                             group * 2 * atom_k);
            }
            llvm::SmallVector<Value> batch_lhs;
            llvm::SmallVector<Value> batch_rhs;
            llvm::SmallVector<int64_t> batch_accumulator_indices;
            auto flush_mfma_batch = [&] {
              const int64_t batch_size = batch_lhs.size();
              if (batch_size == 0) {
                return;
              }
              llvm::SmallVector<mlir::Type> output_types(batch_size,
                                                         accumulator_type);
              mlir::Type output_type = mlir::LLVM::LLVMStructType::getLiteral(
                  builder.getContext(), output_types);
              llvm::SmallVector<Value> operands;
              operands.reserve(3 * batch_size);
              for (int64_t index = 0; index < batch_size; ++index) {
                operands.push_back(batch_lhs[index]);
                operands.push_back(batch_rhs[index]);
              }
              for (int64_t accumulator_index : batch_accumulator_indices) {
                operands.push_back(next_accumulators[accumulator_index]);
              }

              std::string assembly;
              std::string constraints;
              auto append_constraint = [&](const std::string& value) {
                if (!constraints.empty()) {
                  constraints += ',';
                }
                constraints += value;
              };
              for (int64_t index = 0; index < batch_size; ++index) {
                append_constraint("=a");
              }
              for (int64_t index = 0; index < 2 * batch_size; ++index) {
                append_constraint("v");
              }
              for (int64_t index = 0; index < batch_size; ++index) {
                append_constraint(std::to_string(index));
                assembly +=
                    "v_mfma_f32_16x16x16_bf16 $" + std::to_string(index) +
                    ", $" + std::to_string(batch_size + 2 * index) + ", $" +
                    std::to_string(batch_size + 2 * index + 1) + ", $" +
                    std::to_string(3 * batch_size + index) + ";\n";
              }
              auto asm_dialect = mlir::LLVM::AsmDialectAttr::get(
                  builder.getContext(), mlir::LLVM::AsmDialect::AD_ATT);
              Value result = mlir::LLVM::InlineAsmOp::create(
                                 builder, output_type, operands, assembly,
                                 constraints,
                                 /*has_side_effects=*/false,
                                 /*is_align_stack=*/false,
                                 mlir::LLVM::TailCallKind::None, asm_dialect,
                                 /*operand_attrs=*/mlir::ArrayAttr())
                                 .getResult(0);
              for (int64_t index = 0; index < batch_size; ++index) {
                next_accumulators[batch_accumulator_indices[index]] =
                    mlir::LLVM::ExtractValueOp::create(builder, result, index);
              }
              batch_lhs.clear();
              batch_rhs.clear();
              batch_accumulator_indices.clear();
            };
            auto append_mfma = [&](int64_t tile_row, int64_t tile_column,
                                   int64_t mfma) {
              constexpr std::array<int64_t, 23> kMfmaScheduleBoundaries = {
                  35,  57,  79,  101, 112, 122, 131, 134,
                  143, 145, 154, 156, 165, 167, 176, 178,
                  179, 180, 181, 182, 189, 191, 192};
              constexpr std::array<int64_t, 28> kStagedLdsBoundaries = {
                  1,   2,   3,   4,   7,   8,   9,   10,  13,  14,
                  15,  16,  19,  20,  193, 194, 195, 196, 199, 200,
                  201, 202, 205, 206, 207, 208, 211, 212};
              const bool is_schedule_boundary =
                  std::find(kMfmaScheduleBoundaries.begin(),
                            kMfmaScheduleBoundaries.end(), mfma_index) !=
                      kMfmaScheduleBoundaries.end() ||
                  std::find(kStagedLdsBoundaries.begin(),
                            kStagedLdsBoundaries.end(), mfma_index) !=
                      kStagedLdsBoundaries.end() ||
                  (!tensile_double_buffer && mfma_index == 187);
              if (is_schedule_boundary) {
                flush_mfma_batch();
              }
              before_mfma(mfma_index++);
              const int64_t fragment_base = mfma * 4;
              llvm::SmallVector<int64_t, 4> fragment_mask = {
                  fragment_base, fragment_base + 1, fragment_base + 2,
                  fragment_base + 3};
              Value lhs_fragment = lhs_group[tile_row];
              Value rhs_fragment = rhs_group[tile_column];
              Value lhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, lhs_fragment, lhs_fragment,
                  fragment_mask);
              Value rhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, rhs_fragment, rhs_fragment,
                  fragment_mask);
              batch_lhs.push_back(source_swap_mfma ? rhs_mfma : lhs_mfma);
              batch_rhs.push_back(source_swap_mfma ? lhs_mfma : rhs_mfma);
              batch_accumulator_indices.push_back(
                  get_accumulator_index(tile_row, tile_column));
            };

            if (tensile_direct_lhs) {
              // Tensile's MIWaveTile 4x14 order holds one staged-B fragment
              // fixed while cycling the four DirectToVgprA fragments. Besides
              // matching its MFMA schedule, this lets one LDS wait cover four
              // MFMAs instead of retiring a new staged fragment per MFMA.
              for (int64_t mfma = 0; mfma < 2; ++mfma) {
                for (int64_t tile_column = 0;
                     tile_column < wave_tile_columns; ++tile_column) {
                  for (int64_t tile_row = 0; tile_row < wave_tile_rows;
                       ++tile_row) {
                    append_mfma(tile_row, tile_column, mfma);
                  }
                  flush_mfma_batch();
                }
              }
            } else {
              // One row per tied-accumulator assembly batch stays below the
              // six-wave VGPR occupancy threshold on gfx942.
              for (int64_t tile_row = 0; tile_row < wave_tile_rows;
                   ++tile_row) {
                for (int64_t mfma = 0; mfma < 2; ++mfma) {
                  for (int64_t tile_column = 0;
                       tile_column < wave_tile_columns; ++tile_column) {
                    append_mfma(tile_row, tile_column, mfma);
                  }
                  flush_mfma_batch();
                }
              }
            }
          }
          CHECK_EQ(mfma_index, 224);
          CHECK_EQ(future_direct.size(), half_direct.size());
          CHECK_EQ(future_staged.size(), half_staged_registers.size());
          emit_hot_loop_schedule();
          return DirectHalfState{
              std::move(next_accumulators), std::move(future_direct),
              std::move(future_staged), std::move(preloaded_next_staged_group),
              refilled_shared};
        };

    // FlyDSL peels the final K tile. Every main-loop iteration prefetches one
    // valid next tile, overlaps it with the current tile's MFMAs, and then
    // synchronizes. The peeled tail performs only the final LDS reads and
    // MFMAs—there is no wrapped prefetch and no trailing workgroup barrier.
    const bool faithful_fly_pipeline = stage_rhs_ || async_lhs_;
    // DirectToVgpr ping-pongs two physical register banks.  Keep two K tiles
    // in one SCF iteration so the second half returns both the direct operand
    // and the one-tile-ahead LDS payload to their original banks before the
    // loop backedge.  This is also the cadence of Tensile's unrolled hot loop.
    Value loop_upper_bound =
        tensile_direct_operand
            ? IndexConstant(
                  builder,
                  k_ - (((k_ / stage_k_) % 2) ? stage_k_ : 2 * stage_k_))
            : (faithful_fly_pipeline ? IndexConstant(builder, k_ - stage_k_)
                                     : upper_bound);
    Value loop_step =
        tensile_direct_operand ? IndexConstant(builder, 2 * stage_k_) : step;
    mlir::scf::ForOp loop = mlir::scf::ForOp::create(
        builder, lower_bound, loop_upper_bound, loop_step, initial_loop_values,
        [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      Value k_base = loop.getInductionVar();
      Value current_stage =
          Rem(builder, Div(builder, k_base, stage_k_), pipeline_stages);
      Value next_stage =
          Rem(builder, Add(builder, current_stage, IndexConstant(builder, 1)),
              pipeline_stages);
      Value next_k = Add(builder, k_base, step);
      Value prefetch_k =
          faithful_fly_pipeline ? next_k : Rem(builder, next_k, k_);
      Value staged_shared = shared_loop_index >= 0
                                ? loop.getRegionIterArg(shared_loop_index)
                                : lhs_shared;
      if (tensile_direct_operand) {
        llvm::SmallVector<Value> current_accumulators;
        current_accumulators.reserve(accumulators_per_wave);
        for (int64_t index = 0; index < accumulators_per_wave; ++index) {
          current_accumulators.push_back(loop.getRegionIterArg(index));
        }
        llvm::SmallVector<Value> current_direct;
        for (int64_t index = accumulators_per_wave;
             index < preloaded_fragment_end; ++index) {
          current_direct.push_back(loop.getRegionIterArg(index));
        }
        llvm::SmallVector<Value> next_staged_registers;
        const int64_t prefetched_staged_count =
            tensile_direct_lhs ? prefetched_rhs_count : prefetched_lhs_count;
        const int64_t prefetched_staged_loop_index =
            tensile_direct_lhs ? prefetched_rhs_loop_index
                               : prefetched_lhs_loop_index;
        for (int64_t index = 0; index < prefetched_staged_count; ++index) {
          next_staged_registers.push_back(
              loop.getRegionIterArg(prefetched_staged_loop_index + index));
        }
        llvm::SmallVector<Value> current_preloaded_staged_group;
        for (int64_t index = 0; index < preloaded_staged_group_count; ++index) {
          current_preloaded_staged_group.push_back(loop.getRegionIterArg(
              preloaded_staged_group_loop_index + index));
        }
        DirectHalfState first = emit_direct_half(
            k_base, staged_shared, current_stage, next_stage,
            current_accumulators, current_direct, next_staged_registers,
            current_preloaded_staged_group,
            /*prefetch_next_direct=*/true,
            /*refill_next_staged=*/true,
            /*load_future_staged=*/true,
            /*direct_has_trailing_staged_loads=*/true,
            /*preload_next_staged_group=*/true);
        DirectHalfState second = emit_direct_half(
            Add(builder, k_base, step), first.shared, next_stage,
            current_stage, first.accumulators, first.direct,
            first.staged_registers, first.preloaded_staged_group,
            /*prefetch_next_direct=*/true,
            /*refill_next_staged=*/true,
            /*load_future_staged=*/true,
            /*direct_has_trailing_staged_loads=*/true,
            /*preload_next_staged_group=*/true);

        llvm::SmallVector<Value> next_loop_values =
            std::move(second.accumulators);
        next_loop_values.append(second.direct);
        next_loop_values.append(second.staged_registers);
        next_loop_values.append(second.preloaded_staged_group);
        next_loop_values.push_back(second.shared);
        mlir::scf::YieldOp::create(builder, next_loop_values);
      } else {
        llvm::SmallVector<Value> current_lhs;
        if (preload_lds_fragments_) {
          if (local_split_k_) {
            const int64_t lhs_begin =
                accumulators_per_wave + wave_tile_columns;
            for (int64_t index = 0; index < wave_tile_rows; ++index) {
              current_lhs.push_back(
                  loop.getRegionIterArg(lhs_begin + index));
            }
          } else {
            current_lhs =
                emit_lhs_lds_fragments(staged_shared, current_stage);
          }
        }
        llvm::SmallVector<Value> next_lhs_registers;
        llvm::SmallVector<Value> next_rhs_registers;
        if (single_buffer_lds_) {
          if (tensile_wide_tile || local_split_k_) {
            for (int64_t index = 0; index < prefetched_rhs_count; ++index) {
              next_rhs_registers.push_back(
                  loop.getRegionIterArg(prefetched_rhs_loop_index + index));
            }
            for (int64_t index = 0; index < prefetched_lhs_count; ++index) {
              next_lhs_registers.push_back(
                  loop.getRegionIterArg(prefetched_lhs_loop_index + index));
            }
          } else if (small_grid_rolling_refill) {
            // Keep a short rolling VMEM window instead of holding the entire
            // next A+B tile live across the current tile's MFMAs.
            constexpr int64_t kPrefetchDepth = 8;
            const int64_t block_threads = num_warps_ * 64;
            const int64_t rhs_copies =
                block_n_ * stage_k_ / load_vector_width / block_threads;
            const int64_t lhs_copies =
                block_m_ * stage_k_ / load_vector_width / block_threads;
            next_rhs_registers.resize(rhs_copies);
            next_lhs_registers.resize(lhs_copies);
            for (int64_t copy = 0;
                 copy < std::min(kPrefetchDepth, rhs_copies + lhs_copies);
                 ++copy) {
              if (copy < rhs_copies) {
                next_rhs_registers[copy] =
                    emit_rhs_register_vector(prefetch_k, copy);
              } else {
                const int64_t lhs_copy = copy - rhs_copies;
                next_lhs_registers[lhs_copy] =
                    emit_lhs_register_vector(prefetch_k, lhs_copy);
              }
            }
          } else {
            next_rhs_registers = emit_rhs_register_vectors(prefetch_k);
            next_lhs_registers = emit_lhs_register_vectors(prefetch_k);
          }
        } else if (stage_rhs_) {
          staged_shared =
              emit_rhs_lds_stage(staged_shared, prefetch_k, next_stage);
        }
        if (!single_buffer_lds_) {
          staged_shared = emit_lhs_stage(staged_shared, prefetch_k, next_stage);
        }
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
        const int64_t fragment_state_end = preloaded_fragment_end;
        if (local_split_k_) {
          for (int64_t index = 0; index < wave_tile_columns; ++index) {
            current_rhs.push_back(
                loop.getRegionIterArg(accumulators_per_wave + index));
          }
        } else if (load_rhs_on_demand) {
          current_rhs = emit_rhs_lds_fragments(staged_shared, current_stage);
        } else {
          current_rhs.reserve(fragment_state_end - accumulators_per_wave);
          for (int64_t index = accumulators_per_wave;
               index < fragment_state_end; ++index) {
            current_rhs.push_back(loop.getRegionIterArg(index));
          }
        }
        if (small_grid_single_buffer && !local_split_k_) {
          // The complete current A/B tile is now resident in VGPRs. Make that
          // point uniform across the four waves before any wave starts
          // overwriting the single LDS allocation with the next tile.
          mlir::ROCDL::SchedBarrier::create(
              builder, mlir::ROCDL::SchedGroupMask::none);
          staged_shared =
              SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                    staged_shared)
                  .getResult(0);
          mlir::ROCDL::SchedBarrier::create(
              builder, mlir::ROCDL::SchedGroupMask::none);
        }
        llvm::SmallVector<Value> future_rhs_registers;
        if (tensile_direct_rhs || local_split_k_) {
          future_rhs_registers.resize(current_rhs.size());
        }
        llvm::SmallVector<Value> future_lhs_registers;
        if (local_split_k_) {
          future_rhs_registers.resize(next_rhs_registers.size());
          future_lhs_registers.resize(next_lhs_registers.size());
        }
        if (local_split_k_) {
          next_rhs.resize(wave_tile_columns);
        }
        llvm::SmallVector<Value> next_local_lhs;
        if (local_split_k_) {
          next_local_lhs.resize(wave_tile_rows);
        }
        Value future_k = Add(builder, prefetch_k, step);
        // The last LocalSplit replacement is deliberately speculative and is
        // not consumed. Keeping the steady-state PGR2 cadence through the
        // final loop iteration is faster than branching into a second tail.
        const int64_t refill_copies =
            next_rhs_registers.size() + next_lhs_registers.size();
        const int64_t small_grid_mfmas =
            compute_stage_k / atom_k * wave_tile_rows * wave_tile_columns;
        auto before_mfma = [&](int64_t mfma_index) {
          if (!tensile_wide_tile && !small_grid_single_buffer) {
            return;
          }
          if (local_split_k_) {
            // The second K32 group's LDS reads are in flight under the first
            // group's MFMAs. Open the overwrite window after enough
            // independent work to retire them, then follow Tensile's compact
            // store/load cadence.
            if (mfma_index == 21) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              staged_shared =
                  SyncThreadsOp::create(
                      builder, mlir::TypeRange{lhs_shared_type}, staged_shared)
                      .getResult(0);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
            }
            constexpr std::array<int64_t, 12> kRefillPositions = {
                22, 24, 26, 28, 31, 33, 35, 37, 40, 42, 44, 47};
            for (int64_t copy = 0; copy < refill_copies; ++copy) {
              CHECK_LT(copy, kRefillPositions.size());
              if (mfma_index != kRefillPositions[copy]) {
                continue;
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              emit_wait_vmcnt(refill_copies - 1);
              if (copy < next_rhs_registers.size()) {
                staged_shared = emit_rhs_register_stores(
                    staged_shared, next_rhs_registers, copy, copy + 1);
              } else {
                const int64_t lhs_copy = copy - next_rhs_registers.size();
                staged_shared = emit_lhs_register_stores(
                    staged_shared, next_lhs_registers, lhs_copy,
                    lhs_copy + 1);
              }
              // Reuse the retired payload's logical register slot for the
              // tile after the one just stored. Keeping twelve loads in
              // flight at a constant vmcnt(11) is hipBLASLt's PGR2 cadence.
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              if (copy < next_rhs_registers.size()) {
                future_rhs_registers[copy] =
                    emit_rhs_register_vector(future_k, copy);
              } else {
                const int64_t lhs_copy = copy - next_rhs_registers.size();
                future_lhs_registers[lhs_copy] =
                    emit_lhs_register_vector(future_k, lhs_copy);
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              break;
            }
            if (mfma_index == 50) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              staged_shared =
                  SyncThreadsOp::create(
                      builder, mlir::TypeRange{lhs_shared_type}, staged_shared)
                      .getResult(0);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
            }
            return;
          }
          if (small_grid_rolling_refill) {
            constexpr int64_t kPrefetchDepth = 8;
            // Spread each dwordx4 refill payload uniformly across the
            // independent MFMAs. Retire the oldest outstanding load,
            // store it to LDS, and issue one replacement load.
            for (int64_t copy = 0; copy < refill_copies; ++copy) {
              const int64_t position =
                  (copy + 1) * small_grid_mfmas / (refill_copies + 1);
              if (mfma_index != position) {
                continue;
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              emit_wait_vmcnt(
                  std::min(kPrefetchDepth, refill_copies - copy) - 1);
              if (copy < next_rhs_registers.size()) {
                staged_shared = emit_rhs_register_stores(
                    staged_shared, next_rhs_registers, copy, copy + 1);
              } else {
                const int64_t lhs_copy = copy - next_rhs_registers.size();
                staged_shared = emit_lhs_register_stores(
                    staged_shared, next_lhs_registers, lhs_copy,
                    lhs_copy + 1);
              }
              const int64_t future_copy = copy + kPrefetchDepth;
              if (future_copy < refill_copies) {
                if (future_copy < next_rhs_registers.size()) {
                  next_rhs_registers[future_copy] =
                      emit_rhs_register_vector(prefetch_k, future_copy);
                } else {
                  const int64_t lhs_copy =
                      future_copy - next_rhs_registers.size();
                  next_lhs_registers[lhs_copy] =
                      emit_lhs_register_vector(prefetch_k, lhs_copy);
                }
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }
            if (mfma_index == small_grid_mfmas - 1) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              staged_shared =
                  SyncThreadsOp::create(
                      builder, mlir::TypeRange{lhs_shared_type}, staged_shared)
                      .getResult(0);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
            }
            return;
          }
          if (small_grid_single_buffer) {
            // Spread the next A+B tile's payloads across the independent MFMAs
            // and progressively retire VMEM so each store also ends that
            // payload's VGPR lifetime.
            for (int64_t copy = 0; copy < refill_copies; ++copy) {
              if (mfma_index != 4 + copy * 5) {
                continue;
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              emit_wait_vmcnt(refill_copies - copy - 1);
              if (copy < next_rhs_registers.size()) {
                staged_shared = emit_rhs_register_stores(
                    staged_shared, next_rhs_registers, copy, copy + 1);
              } else {
                const int64_t lhs_copy = copy - next_rhs_registers.size();
                staged_shared = emit_lhs_register_stores(
                    staged_shared, next_lhs_registers, lhs_copy,
                    lhs_copy + 1);
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }
            if (mfma_index == 63) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              staged_shared =
                  SyncThreadsOp::create(
                      builder, mlir::TypeRange{lhs_shared_type}, staged_shared)
                      .getResult(0);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
            }
            return;
          }
          if (tensile_direct_rhs) {
            // Exact DirectToVgprA modulo schedule from the dispatched
            // hipBLASLt MT256x224x64 kernel.  The eight next-tile RHS vectors
            // are issued as four pairs while the current MFMAs run.
            constexpr std::array<int64_t, 4> kDirectLoadPositions = {35, 57, 79,
                                                                     101};
            for (int64_t pair = 0; pair < kDirectLoadPositions.size(); ++pair) {
              if (mfma_index != kDirectLoadPositions[pair]) {
                continue;
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              future_rhs_registers[pair] =
                  emit_tensile_direct_rhs_fragment(prefetch_k, pair);
              future_rhs_registers[pair + wave_tile_columns] =
                  emit_tensile_direct_rhs_fragment(prefetch_k,
                                                   pair + wave_tile_columns);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }
            // All current LDS fragments have been consumed before the first
            // overwrite.  Tensile deliberately uses no opening workgroup
            // barrier here; the sole barrier closes the refill window.
            if (mfma_index == 112) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              emit_wait_lgkmcnt(0);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }

            constexpr std::array<int64_t, 7> kLdsRefillPositions = {
                122, 134, 145, 156, 167, 178, 189};
            for (int64_t copy = 0; copy < kLdsRefillPositions.size(); ++copy) {
              if (mfma_index != kLdsRefillPositions[copy]) {
                continue;
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              emit_wait_vmcnt(14);
              staged_shared = emit_lhs_register_stores(
                  staged_shared, next_lhs_registers, copy, copy + 1);
              future_lhs_registers.push_back(
                  emit_lhs_register_vector(future_k, copy));
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }

            if (mfma_index == 192) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              staged_shared =
                  SyncThreadsOp::create(
                      builder, mlir::TypeRange{lhs_shared_type}, staged_shared)
                      .getResult(0);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
            }
            return;
          }
          // Reproduce the barrier window in hipBLASLt's 256x224x64 solution.
          // It executes 59 independent MFMAs after issuing the current tile's
          // LDS reads, then opens the single-buffer refill window. The closing
          // barrier is after MFMA 201, leaving 23 independent MFMAs in flight.
          if (mfma_index == 59 || mfma_index == 201) {
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
            staged_shared =
                SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                      staged_shared)
                    .getResult(0);
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
          }
          int64_t copy = -1;
          for (int64_t candidate = 0; candidate < refill_copies; ++candidate) {
            // Each dwordx4 operation combines two adjacent dwordx2 operations
            // from Tensile's 30-copy sequence. Its native refill positions are
            // floor(60 + copy * 143 / 30) MFMAs into the modulo loop.
            if (mfma_index == 60 + ((2 * candidate + 1) * 143) / 30) {
              copy = candidate;
              break;
            }
          }
          if (copy < 0) {
            return;
          }
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
          // One new 16-byte load replaces every consumed register vector, so a
          // constant vmcnt(14) advances exactly the oldest of 15 outstanding
          // operations.
          emit_wait_vmcnt(refill_copies - 1);
          if (copy < next_rhs_registers.size()) {
            staged_shared = emit_rhs_register_stores(
                staged_shared, next_rhs_registers, copy, copy + 1);
            future_rhs_registers.push_back(
                emit_rhs_register_vector(future_k, copy));
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
            return;
          }
          const int64_t lhs_copy = copy - next_rhs_registers.size();
          staged_shared = emit_lhs_register_stores(
              staged_shared, next_lhs_registers, lhs_copy, lhs_copy + 1);
          future_lhs_registers.push_back(
              emit_lhs_register_vector(future_k, lhs_copy));
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
        };
        auto after_mfma = [&](int64_t mfma_index) {
          if (!local_split_k_) {
            return;
          }
          auto emit_scheduled_read = [&](auto&& emit_read) {
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
            Value value = emit_read();
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
            return value;
          };
          switch (mfma_index) {
            case 50:
              next_rhs[0] = emit_scheduled_read([&] {
                return emit_rhs_lds_fragment(staged_shared, next_stage, 0, 0);
              });
              break;
            case 51:
              next_local_lhs[0] = emit_scheduled_read([&] {
                return emit_lhs_lds_fragment(staged_shared, next_stage, 0, 0);
              });
              break;
            case 52:
              next_rhs[1] = emit_scheduled_read([&] {
                return emit_rhs_lds_fragment(staged_shared, next_stage, 0, 1);
              });
              break;
            case 53:
              next_rhs[2] = emit_scheduled_read([&] {
                return emit_rhs_lds_fragment(staged_shared, next_stage, 0, 2);
              });
              break;
            case 56:
              next_rhs[3] = emit_scheduled_read([&] {
                return emit_rhs_lds_fragment(staged_shared, next_stage, 0, 3);
              });
              break;
            case 57:
              next_local_lhs[1] = emit_scheduled_read([&] {
                return emit_lhs_lds_fragment(staged_shared, next_stage, 0, 1);
              });
              break;
            case 58:
              next_local_lhs[2] = emit_scheduled_read([&] {
                return emit_lhs_lds_fragment(staged_shared, next_stage, 0, 2);
              });
              break;
            case 59:
              next_local_lhs[3] = emit_scheduled_read([&] {
                return emit_lhs_lds_fragment(staged_shared, next_stage, 0, 3);
              });
              break;
            default:
              break;
          }
        };
        if (tensile_direct_rhs) {
          // Eight current RHS loads precede the seven one-tile-ahead LDS
          // payloads in VMEM order.  Retire just the direct operand here and
          // leave all seven payloads outstanding for the refill schedule.
          emit_wait_vmcnt(next_lhs_registers.size());
        }
        llvm::SmallVector<Value> next_accumulators =
            local_split_k_
                ? emit_compute_local_split(
                      staged_shared, current_stage, current_lhs, current_rhs,
                      current_accumulators, before_mfma, after_mfma)
                : (preload_lds_fragments_
                       ? emit_compute_preloaded(
                             current_lhs, current_rhs, current_accumulators,
                             before_mfma)
                       : emit_compute_tile(staged_shared, current_stage, k_base,
                                           current_accumulators, current_rhs));
        if (tensile_wide_tile || local_split_k_) {
          CHECK_EQ(future_rhs_registers.size(),
                   tensile_direct_rhs ? current_rhs.size()
                                      : next_rhs_registers.size());
          CHECK_EQ(future_lhs_registers.size(), next_lhs_registers.size());
          if (!tensile_direct_rhs || local_split_k_) {
            next_rhs_registers = std::move(future_rhs_registers);
          }
          next_lhs_registers = std::move(future_lhs_registers);
        }
        emit_hot_loop_schedule();
        Value synchronized_lhs = staged_shared;
        if (!tensile_wide_tile && !small_grid_single_buffer) {
          if (stage_rhs_ || async_lhs_) {
            emit_wait_vmcnt(0);
          }
          synchronized_lhs =
              SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                    staged_shared)
                  .getResult(0);
        }
        if (single_buffer_lds_) {
          if (tensile_wide_tile) {
            // The refill was issued above and the preceding synchronization is
            // the closing barrier for those writes.
          } else if (small_grid_single_buffer) {
            // The modulo schedule issued the refill and its closing barrier.
          } else if (triton_vec4_lds) {
            CHECK_EQ(next_rhs_registers.size(), 4);
            CHECK_EQ(next_lhs_registers.size(), 2);
            emit_wait_vmcnt(4);
            synchronized_lhs = emit_rhs_register_stores(
                synchronized_lhs, next_rhs_registers, 0, 2);
            emit_wait_vmcnt(2);
            synchronized_lhs = emit_rhs_register_stores(
                synchronized_lhs, next_rhs_registers, 2, 4);
            emit_wait_vmcnt(0);
            synchronized_lhs = emit_lhs_register_stores(
                synchronized_lhs, next_lhs_registers, 0, 2);
          } else {
            emit_wait_vmcnt(0);
            synchronized_lhs =
                emit_rhs_register_stores(synchronized_lhs, next_rhs_registers,
                                         0, next_rhs_registers.size());
            synchronized_lhs =
                emit_lhs_register_stores(synchronized_lhs, next_lhs_registers,
                                         0, next_lhs_registers.size());
          }
          if (!tensile_wide_tile && !small_grid_single_buffer) {
            synchronized_lhs =
                SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                      synchronized_lhs)
                    .getResult(0);
          }
        }
        if (preload_lds_fragments_ && !load_rhs_on_demand &&
            !local_split_k_) {
          next_rhs = tensile_direct_rhs
                         ? std::move(future_rhs_registers)
                         : emit_rhs_lds_fragments(synchronized_lhs,
                                                  next_stage);
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
        }
        next_accumulators.append(next_rhs);
        if (local_split_k_) {
          next_accumulators.append(next_local_lhs);
        }
        if (shared_loop_index >= 0) {
          if (tensile_wide_tile || local_split_k_) {
            next_accumulators.append(next_rhs_registers);
            next_accumulators.append(next_lhs_registers);
          }
          next_accumulators.push_back(synchronized_lhs);
        }
        mlir::scf::YieldOp::create(builder, next_accumulators);
      }
    }

    builder.setInsertionPointAfter(loop);
    llvm::SmallVector<Value> final_accumulators;
    final_accumulators.reserve(accumulators_per_wave);
    if (tensile_direct_operand) {
      llvm::SmallVector<Value> loop_accumulators;
      loop_accumulators.reserve(accumulators_per_wave);
      for (int64_t index = 0; index < accumulators_per_wave; ++index) {
        loop_accumulators.push_back(loop.getResult(index));
      }
      if ((k_ / stage_k_) % 2 == 0) {
        llvm::SmallVector<Value> tail_direct;
        for (int64_t index = accumulators_per_wave;
             index < preloaded_fragment_end; ++index) {
          tail_direct.push_back(loop.getResult(index));
        }
        llvm::SmallVector<Value> tail_staged_registers;
        const int64_t prefetched_staged_count =
            tensile_direct_lhs ? prefetched_rhs_count : prefetched_lhs_count;
        const int64_t prefetched_staged_loop_index =
            tensile_direct_lhs ? prefetched_rhs_loop_index
                               : prefetched_lhs_loop_index;
        for (int64_t index = 0; index < prefetched_staged_count; ++index) {
          tail_staged_registers.push_back(
              loop.getResult(prefetched_staged_loop_index + index));
        }
        llvm::SmallVector<Value> tail_preloaded_staged_group;
        for (int64_t index = 0; index < preloaded_staged_group_count; ++index) {
          tail_preloaded_staged_group.push_back(
              loop.getResult(preloaded_staged_group_loop_index + index));
        }
        Value tail_shared = loop.getResult(shared_loop_index);
        const int64_t first_tail_tile = k_ / stage_k_ - 2;
        Value first_tail_k = IndexConstant(builder, k_ - 2 * stage_k_);
        Value first_tail_stage =
            IndexConstant(builder, first_tail_tile % pipeline_stages);
        Value second_tail_stage = IndexConstant(
            builder, (first_tail_tile + 1) % pipeline_stages);
        DirectHalfState first = emit_direct_half(
            first_tail_k, tail_shared, first_tail_stage, second_tail_stage,
            loop_accumulators, tail_direct, tail_staged_registers,
            tail_preloaded_staged_group,
            /*prefetch_next_direct=*/true,
            /*refill_next_staged=*/true,
            /*load_future_staged=*/false,
            /*direct_has_trailing_staged_loads=*/true,
            /*preload_next_staged_group=*/true);
        DirectHalfState second = emit_direct_half(
            IndexConstant(builder, k_ - stage_k_), first.shared,
            second_tail_stage, first_tail_stage, first.accumulators,
            first.direct, first.staged_registers,
            first.preloaded_staged_group,
            /*prefetch_next_direct=*/false,
            /*refill_next_staged=*/false,
            /*load_future_staged=*/false,
            /*direct_has_trailing_staged_loads=*/false,
            /*preload_next_staged_group=*/false);
        final_accumulators = std::move(second.accumulators);
      } else {
        llvm::SmallVector<Value> tail_direct;
        for (int64_t index = accumulators_per_wave;
             index < preloaded_fragment_end; ++index) {
          tail_direct.push_back(loop.getResult(index));
        }
        Value tail_shared = loop.getResult(shared_loop_index);
        llvm::SmallVector<Value> tail_lhs =
            tensile_direct_lhs
                ? llvm::SmallVector<Value>(tail_direct)
                : emit_lhs_lds_fragments(tail_shared, lower_bound);
        llvm::SmallVector<Value> tail_rhs =
            tensile_direct_rhs
                ? llvm::SmallVector<Value>(tail_direct)
                : emit_rhs_lds_fragments(tail_shared, lower_bound);
        // The loop's second half leaves the final direct operand followed by
        // one unused LDS payload in the VMEM queue.  Retire only the former.
        emit_wait_vmcnt(tensile_direct_lhs ? prefetched_rhs_count
                                           : prefetched_lhs_count);
        final_accumulators = emit_compute_preloaded(
            tail_lhs, tail_rhs, loop_accumulators, [](int64_t) {});
      }
    } else if (faithful_fly_pipeline) {
      llvm::SmallVector<Value> loop_accumulators;
      loop_accumulators.reserve(accumulators_per_wave);
      for (int64_t index = 0; index < accumulators_per_wave; ++index) {
        loop_accumulators.push_back(loop.getResult(index));
      }
      if (local_split_k_) {
        llvm::SmallVector<Value> tail_rhs;
        llvm::SmallVector<Value> tail_lhs;
        for (int64_t index = 0; index < wave_tile_columns; ++index) {
          tail_rhs.push_back(
              loop.getResult(accumulators_per_wave + index));
        }
        for (int64_t index = 0; index < wave_tile_rows; ++index) {
          tail_lhs.push_back(loop.getResult(
              accumulators_per_wave + wave_tile_columns + index));
        }
        final_accumulators = emit_compute_local_split(
            loop.getResult(shared_loop_index), lower_bound, tail_lhs, tail_rhs,
            loop_accumulators, [](int64_t) {}, [](int64_t) {});
      } else {
        llvm::SmallVector<Value> tail_rhs;
        const int64_t fragment_state_end = preloaded_fragment_end;
        for (int64_t index = accumulators_per_wave;
             index < fragment_state_end; ++index) {
          tail_rhs.push_back(loop.getResult(index));
        }
        Value tail_stage =
            IndexConstant(builder, (k_ / stage_k_ - 1) % pipeline_stages);
        if (preload_lds_fragments_) {
          Value tail_shared = loop.getResult(shared_loop_index);
          llvm::SmallVector<Value> tail_lhs =
              emit_lhs_lds_fragments(tail_shared, tail_stage);
          if (load_rhs_on_demand) {
            tail_rhs = emit_rhs_lds_fragments(tail_shared, tail_stage);
          }
          final_accumulators = emit_compute_preloaded(
              tail_lhs, tail_rhs, loop_accumulators, [](int64_t) {});
        } else {
          final_accumulators = emit_compute_tile(
              lhs_shared, tail_stage, IndexConstant(builder, k_ - stage_k_),
              loop_accumulators, tail_rhs);
        }
      }
    } else {
      for (int64_t index = 0; index < accumulators_per_wave; ++index) {
        final_accumulators.push_back(loop.getResult(index));
      }
    }
    mlir::Type output_type =
        mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType();
    if (local_split_k_) {
      // Each wave retains the half of its accumulator tile that it will write
      // and spills the other half for its partner K partition. All four waves
      // therefore share the epilogue while scratch traffic and footprint stay
      // half that of spilling both complete partitions.
      TF_RET_CHECK((output_type.isBF16() || output_type.isF32()) &&
                   accumulator_elements == 4 && accumulators_per_wave == 16 &&
                   m_ % block_m_ == 0 && n_ % block_n_ == 0);
      Value reduction_shared = loop.getResult(shared_loop_index);
      reduction_shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                reduction_shared)
              .getResult(0);

      auto scratch_vector_type =
          mlir::VectorType::get({2 * accumulator_elements},
                                builder.getBF16Type());
      constexpr int64_t kEpilogueAccumulatorsPerWave = 8;
      auto scratch_index = [&](Value source_wave, int64_t slot) {
        // Keep lanes contiguous within one accumulator slot. The previous
        // lane-major layout advanced adjacent lanes by 128 bytes, mapping all
        // dwordx4 operations onto the same LDS bank group.
        Value wave_slot = Add(
            builder,
            Mul(builder, source_wave,
                IndexConstant(builder, kEpilogueAccumulatorsPerWave)),
            IndexConstant(builder, slot));
        return Mul(
            builder,
            Add(builder, Mul(builder, wave_slot, IndexConstant(builder, 64)),
                lane_id),
            IndexConstant(builder, 2 * accumulator_elements));
      };
      Value is_lower_split = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, local_split_id,
          IndexConstant(builder, 0));
      mlir::scf::IfOp spill = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{lhs_shared_type}, is_lower_split,
          /*withElseRegion=*/true);
      auto spill_accumulators = [&](Value stored, int64_t accumulator_base) {
        for (int64_t slot = 0; slot < kEpilogueAccumulatorsPerWave; ++slot) {
          Value partial_bits = mlir::vector::BitCastOp::create(
              builder, scratch_vector_type,
              final_accumulators[accumulator_base + slot]);
          Value partial_index = scratch_index(wave_id, slot);
          stored = mlir::vector::TransferWriteOp::create(
                       builder, partial_bits, stored,
                       mlir::ValueRange{partial_index},
                       llvm::ArrayRef<bool>{true})
                       .getResult();
        }
        mlir::scf::YieldOp::create(builder, stored);
      };
      {
        mlir::OpBuilder::InsertionGuard spill_guard(builder);
        builder.setInsertionPointToStart(spill.thenBlock());
        // The lower K partition writes the upper half of its tile.
        spill_accumulators(reduction_shared,
                           kEpilogueAccumulatorsPerWave);
        builder.setInsertionPointToStart(spill.elseBlock());
        // The upper K partition writes the lower half of its tile.
        spill_accumulators(reduction_shared, 0);
      }
      builder.setInsertionPointAfter(spill);
      reduction_shared = spill.getResult(0);
      reduction_shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                reduction_shared)
              .getResult(0);

      auto packed_output_type =
          mlir::VectorType::get({accumulator_elements}, output_type);
      mlir::scf::IfOp reduce = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{output.getType()}, is_lower_split,
          /*withElseRegion=*/true);
      auto reduce_accumulators = [&](Value stored_output,
                                     int64_t accumulator_base,
                                     Value partner_wave) {
        for (int64_t slot = 0; slot < kEpilogueAccumulatorsPerWave; ++slot) {
          Value partial_index = scratch_index(partner_wave, slot);
          Value partial_bits = mlir::vector::TransferReadOp::create(
              builder, scratch_vector_type, reduction_shared,
              mlir::ValueRange{partial_index}, /*padding=*/std::nullopt,
              llvm::ArrayRef<bool>{true});
          Value partner_partial = mlir::vector::BitCastOp::create(
              builder, accumulator_type, partial_bits);
          const int64_t accumulator_index = accumulator_base + slot;
          Value reduced = mlir::arith::AddFOp::create(
              builder, final_accumulators[accumulator_index], partner_partial);
          Value packed =
              output_type.isBF16()
                  ? mlir::arith::TruncFOp::create(builder, packed_output_type,
                                                  reduced)
                        .getResult()
                  : reduced;
          const int64_t tile_row =
              accumulator_index / wave_tile_columns;
          const int64_t tile_column =
              accumulator_index % wave_tile_columns;
          Value output_row = Add(
              builder, wave_row_base,
              Add(builder, lane_axis,
                  IndexConstant(builder, tile_row * atom_m)));
          Value output_column = Add(
              builder, wave_column_base,
              Add(builder,
                  Mul(builder, lane_group, IndexConstant(builder, 4)),
                  IndexConstant(builder, tile_column * atom_m)));
          llvm::SmallVector<Value, 3> indices =
              output_indices(output_row, output_column);
          stored_output =
              mlir::vector::TransferWriteOp::create(
                  builder, packed, stored_output, indices,
                  llvm::ArrayRef<bool>{true})
                  .getResult();
        }
        mlir::scf::YieldOp::create(builder, stored_output);
      };
      {
        mlir::OpBuilder::InsertionGuard reduce_guard(builder);
        builder.setInsertionPointToStart(reduce.thenBlock());
        Value upper_partner =
            Add(builder, output_wave_id, IndexConstant(builder, output_waves));
        reduce_accumulators(output, /*accumulator_base=*/0, upper_partner);
        builder.setInsertionPointToStart(reduce.elseBlock());
        reduce_accumulators(output,
                            /*accumulator_base=*/
                                kEpilogueAccumulatorsPerWave,
                            /*partner_wave=*/output_wave_id);
      }
      builder.setInsertionPointAfter(reduce);
      mlir::func::ReturnOp::create(builder, reduce.getResult(0));
      return absl::OkStatus();
    }
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
    Value output_before_local_split = output;
    std::optional<mlir::scf::IfOp> local_split_store;
    if (local_split_k_) {
      Value is_lower_split = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, local_split_id,
          IndexConstant(builder, 0));
      local_split_store.emplace(mlir::scf::IfOp::create(
          builder, mlir::TypeRange{output.getType()}, is_lower_split,
          /*withElseRegion=*/true));
      builder.setInsertionPointToStart(local_split_store->thenBlock());
    }
    if (source_swap_mfma) {
      TF_RET_CHECK(!stage_output_ && n_ % 16 == 0 && m_ % 16 == 0);
      TF_RET_CHECK(accumulator_elements == 4);
      auto packed_output_type =
          mlir::VectorType::get({accumulator_elements}, output_type);
      auto pack_accumulator_bf16 = [&](Value accumulator) -> Value {
        mlir::Type i32_type = builder.getI32Type();
        auto packed_i32_type = mlir::VectorType::get({2}, i32_type);
        Value shift_16 =
            mlir::arith::ConstantIntOp::create(builder, i32_type, 16);
        Value one = mlir::arith::ConstantIntOp::create(builder, i32_type, 1);
        Value rounding_bias =
            mlir::arith::ConstantIntOp::create(builder, i32_type, 0x7fff);
        Value quiet_nan_bit =
            mlir::arith::ConstantIntOp::create(builder, i32_type, 0x00400000);
        Value absolute_value_mask = mlir::arith::ConstantIntOp::create(
            builder, i32_type, 0x7fffffff);
        Value positive_infinity = mlir::arith::ConstantIntOp::create(
            builder, i32_type, 0x7f800000);
        llvm::SmallVector<Value, 4> bits;
        llvm::SmallVector<Value, 4> is_nan;
        for (int64_t element = 0; element < accumulator_elements; ++element) {
          Value value = mlir::vector::ExtractOp::create(
              builder, accumulator, llvm::SmallVector<int64_t>{element});
          Value value_bits =
              mlir::arith::BitcastOp::create(builder, i32_type, value);
          bits.push_back(value_bits);
          // Classify NaNs from the integer representation. Keeping both the
          // rounding and NaN paths in the integer domain lets LLVM materialize
          // each AGPR exactly once in a VGPR. A floating-point unordered
          // compare otherwise causes a second v_accvgpr_read for every result.
          Value magnitude = mlir::arith::AndIOp::create(
              builder, value_bits, absolute_value_mask);
          is_nan.push_back(mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ugt, magnitude,
              positive_infinity));
        }

        llvm::SmallVector<Value, 4> rounded_bits;
        for (int64_t element = 0; element < accumulator_elements; ++element) {
          Value retained_lsb = mlir::arith::AndIOp::create(
              builder,
              mlir::arith::ShRUIOp::create(builder, bits[element], shift_16),
              one);
          Value rounded = mlir::arith::AddIOp::create(
              builder,
              mlir::arith::AddIOp::create(builder, bits[element],
                                          rounding_bias),
              retained_lsb);
          Value quiet_nan =
              mlir::arith::OrIOp::create(builder, bits[element], quiet_nan_bit);
          rounded_bits.push_back(mlir::arith::SelectOp::create(
              builder, is_nan[element], quiet_nan, rounded));
        }

        llvm::SmallVector<Value, 2> packed_words;
        Value permute_mask = mlir::arith::ConstantIntOp::create(
            builder, i32_type, 0x07060302);
        for (int64_t element = 0; element < accumulator_elements;
             element += 2) {
          packed_words.push_back(
              mlir::LLVM::InlineAsmOp::create(
                  builder, i32_type,
                  mlir::ValueRange{rounded_bits[element + 1],
                                   rounded_bits[element], permute_mask},
                  "v_perm_b32 $0, $1, $2, $3;\n", "=v,v,v,s",
                  /*has_side_effects=*/false,
                  /*is_align_stack=*/false,
                  mlir::LLVM::TailCallKind::None,
                  mlir::LLVM::AsmDialectAttr::get(
                      builder.getContext(), mlir::LLVM::AsmDialect::AD_ATT),
                  /*operand_attrs=*/mlir::ArrayAttr())
                  .getResult(0));
        }
        Value packed_words_vector = mlir::vector::FromElementsOp::create(
            builder, packed_i32_type, packed_words);
        return mlir::vector::BitCastOp::create(builder, packed_output_type,
                                               packed_words_vector);
      };
      auto read_accumulators_to_vgprs =
          [&](llvm::ArrayRef<Value> accumulators) {
            const int64_t value_count =
                accumulators.size() * accumulator_elements;
            llvm::SmallVector<mlir::Type> output_types(
                value_count, builder.getF32Type());
            mlir::Type output_type = mlir::LLVM::LLVMStructType::getLiteral(
                builder.getContext(), output_types);
            llvm::SmallVector<Value> operands;
            operands.reserve(value_count);
            for (Value accumulator : accumulators) {
              for (int64_t element = 0; element < accumulator_elements;
                   ++element) {
                operands.push_back(mlir::vector::ExtractOp::create(
                    builder, accumulator,
                    llvm::SmallVector<int64_t>{element}));
              }
            }

            std::string assembly;
            std::string constraints;
            auto append_constraint = [&](const std::string& value) {
              if (!constraints.empty()) constraints += ',';
              constraints += value;
            };
            for (int64_t index = 0; index < value_count; ++index) {
              append_constraint("=v");
            }
            for (int64_t index = 0; index < value_count; ++index) {
              append_constraint("a");
              assembly += "v_accvgpr_read_b32 $" + std::to_string(index) +
                          ", $" + std::to_string(value_count + index) +
                          ";\n";
            }
            auto asm_dialect = mlir::LLVM::AsmDialectAttr::get(
                builder.getContext(), mlir::LLVM::AsmDialect::AD_ATT);
            Value result =
                mlir::LLVM::InlineAsmOp::create(
                    builder, output_type, operands, assembly, constraints,
                    /*has_side_effects=*/false,
                    /*is_align_stack=*/false, mlir::LLVM::TailCallKind::None,
                    asm_dialect, /*operand_attrs=*/mlir::ArrayAttr())
                    .getResult(0);
            llvm::SmallVector<Value> values;
            values.reserve(value_count);
            for (int64_t index = 0; index < value_count; ++index) {
              values.push_back(mlir::LLVM::ExtractValueOp::create(
                  builder, result, index));
            }
            return values;
          };
      auto pack_vgpr_bf16 = [&](llvm::ArrayRef<Value> values) -> Value {
        CHECK_EQ(values.size(), accumulator_elements);
        mlir::Type i32_type = builder.getI32Type();
        auto packed_i32_type = mlir::VectorType::get({2}, i32_type);
        Value shift_16 =
            mlir::arith::ConstantIntOp::create(builder, i32_type, 16);
        Value one = mlir::arith::ConstantIntOp::create(builder, i32_type, 1);
        Value rounding_bias =
            mlir::arith::ConstantIntOp::create(builder, i32_type, 0x7fff);
        Value canonical_nan = mlir::arith::ConstantIntOp::create(
            builder, i32_type, static_cast<int32_t>(0x7fff0000u));
        llvm::SmallVector<Value, 4> rounded_bits;
        for (Value value : values) {
          Value bits =
              mlir::arith::BitcastOp::create(builder, i32_type, value);
          Value retained_lsb = mlir::arith::AndIOp::create(
              builder,
              mlir::arith::ShRUIOp::create(builder, bits, shift_16), one);
          Value rounded = mlir::arith::AddIOp::create(
              builder,
              mlir::arith::AddIOp::create(builder, bits, rounding_bias),
              retained_lsb);
          Value is_nan = mlir::arith::CmpFOp::create(
              builder, mlir::arith::CmpFPredicate::UNO, value, value);
          rounded_bits.push_back(mlir::arith::SelectOp::create(
              builder, is_nan, canonical_nan, rounded));
        }

        llvm::SmallVector<Value, 2> packed_words;
        Value permute_mask = mlir::arith::ConstantIntOp::create(
            builder, i32_type, 0x07060302);
        for (int64_t element = 0; element < accumulator_elements;
             element += 2) {
          packed_words.push_back(
              mlir::LLVM::InlineAsmOp::create(
                  builder, i32_type,
                  mlir::ValueRange{rounded_bits[element + 1],
                                   rounded_bits[element], permute_mask},
                  "v_perm_b32 $0, $1, $2, $3;\n", "=v,v,v,s",
                  /*has_side_effects=*/false,
                  /*is_align_stack=*/false,
                  mlir::LLVM::TailCallKind::None,
                  mlir::LLVM::AsmDialectAttr::get(
                      builder.getContext(), mlir::LLVM::AsmDialect::AD_ATT),
                  /*operand_attrs=*/mlir::ArrayAttr())
                  .getResult(0));
        }
        Value packed_words_vector = mlir::vector::FromElementsOp::create(
            builder, packed_i32_type, packed_words);
        return mlir::vector::BitCastOp::create(builder, packed_output_type,
                                               packed_words_vector);
      };
      for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
        Value output_row =
            tensile_direct_lhs
                ? Add(builder, wave_row_base,
                      Add(builder,
                          Mul(builder, lane_axis,
                              IndexConstant(builder, wave_tile_rows)),
                          IndexConstant(builder, tile_row)))
                : Add(builder, wave_row_base,
                      Add(builder, lane_axis,
                          IndexConstant(builder, tile_row * atom_m)));
        llvm::SmallVector<Value> row_accumulator_values;
        if (tensile_direct_lhs && output_type.isBF16()) {
          llvm::SmallVector<Value> row_accumulators;
          row_accumulators.reserve(wave_tile_columns);
          for (int64_t tile_column = 0;
               tile_column < wave_tile_columns; ++tile_column) {
            row_accumulators.push_back(final_accumulators[
                get_accumulator_index(tile_row, tile_column)]);
          }
          row_accumulator_values =
              read_accumulators_to_vgprs(row_accumulators);
        }
        auto emit_row_stores = [&](Value destination) {
          Value updated = destination;
          for (int64_t tile_column = 0; tile_column < wave_tile_columns;
               ++tile_column) {
            const int64_t accumulator_index =
                get_accumulator_index(tile_row, tile_column);
            Value packed_result = final_accumulators[accumulator_index];
            if (output_type.isBF16()) {
              packed_result = row_accumulator_values.empty()
                                  ? pack_accumulator_bf16(packed_result)
                                  : pack_vgpr_bf16(llvm::ArrayRef<Value>(
                                        row_accumulator_values)
                                                        .slice(
                                                            tile_column *
                                                                accumulator_elements,
                                                            accumulator_elements));
            }
            Value output_column =
                Add(builder, wave_column_base,
                    Add(builder,
                        Mul(builder, lane_group, IndexConstant(builder, 4)),
                        IndexConstant(builder, tile_column * atom_m)));
            if (tensile_direct_lhs && n_ % block_n_ != 0) {
              Value output_linear_index =
                  Add(builder,
                      Mul(builder, output_row, IndexConstant(builder, n_)),
                      output_column);
              // N is a multiple of the 16-column MFMA atom. The prefix that
              // also exists in the final partial macro-tile is therefore
              // always safe; only the remaining column fragments need an
              // explicit row-bound mask. Allocation bounds alone are not
              // sufficient because an invalid column can alias the next row.
              if (tile_column * atom_m >= n_ % block_n_) {
                Value column_in_bounds = mlir::arith::CmpIOp::create(
                    builder, mlir::arith::CmpIPredicate::ult, output_column,
                    IndexConstant(builder, n_));
                output_linear_index = mlir::arith::SelectOp::create(
                    builder, column_in_bounds, output_linear_index,
                    IndexConstant(builder, 1LL << 30));
              }
              Value stored_result = packed_result;
              if (output_type.isBF16()) {
                stored_result = mlir::vector::BitCastOp::create(
                    builder, mlir::VectorType::get({2}, builder.getI32Type()),
                    stored_result);
              }
              updated = emit_buffer_store(stored_result, updated,
                                          output_linear_index);
            } else {
              updated = mlir::vector::TransferWriteOp::create(
                            builder, packed_result, updated,
                            mlir::ValueRange{output_row, output_column},
                            llvm::ArrayRef<bool>{true})
                            .getResult();
            }
          }
          return updated;
        };
        if (m_ % block_m_ == 0) {
          output = emit_row_stores(output);
        } else {
          Value row_in_bounds = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, output_row,
              IndexConstant(builder, m_));
          mlir::scf::IfOp store = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{output.getType()}, row_in_bounds,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard if_guard(builder);
            builder.setInsertionPointToStart(store.thenBlock());
            mlir::scf::YieldOp::create(builder, emit_row_stores(output));
            builder.setInsertionPointToStart(store.elseBlock());
            mlir::scf::YieldOp::create(builder, output);
          }
          builder.setInsertionPointAfter(store);
          output = store.getResult(0);
        }
      }
    } else {
      for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             ++tile_column) {
          const int64_t accumulator_index =
              get_accumulator_index(tile_row, tile_column);
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
                Add(builder,
                    Mul(builder, lane_group, IndexConstant(builder, 4)),
                    IndexConstant(builder, tile_row * atom_m + element_row)));
            Value output_column =
                Add(builder, wave_column_base,
                    Add(builder, lane_axis,
                        IndexConstant(builder, tile_column * atom_m)));
            llvm::SmallVector<Value, 3> indices =
                output_indices(output_row, output_column);
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
    }
    if (local_split_store.has_value()) {
      mlir::scf::YieldOp::create(builder, output);
      builder.setInsertionPointToStart(local_split_store->elseBlock());
      mlir::scf::YieldOp::create(builder, output_before_local_split);
      builder.setInsertionPointAfter(*local_split_store);
      output = local_split_store->getResult(0);
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
  bool preload_lds_fragments_ = false;
  bool single_buffer_lds_ = false;
  bool direct_to_vgpr_ = false;
  bool rolling_refill_ = false;
  bool local_split_k_ = false;
  bool global_split_k_ = false;
  int64_t split_k_batches_ = 1;
  int64_t rhs_contracting_dimension_ = 0;
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
    rhs_contracting_dimension_ =
        dot->dot_dimension_numbers().rhs_contracting_dimensions(0);
    const int64_t blocks = m_ == 1 ? (n_ + block_n_ - 1) / block_n_
                                   : (m_ + block_m_ - 1) / block_m_;
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
        "FlyGemvEmitter builds its module directly.");
  }

  absl::Status EmitKernel(mlir::func::FuncOp entry_function) const {
    TF_RET_CHECK(entry_function.getNumArguments() == 3);
    TF_RET_CHECK(m_ == 1 || n_ == 1);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
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
        Value b = rhs_contracting_dimension_ == 0
                      ? ExtractTensor(builder, rhs, k_index, column)
                      : ExtractTensor(builder, rhs, column, k_index);
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
          mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType();
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

    Value row_base = Mul(builder, block_id, IndexConstant(builder, block_m_));
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
            mlir::gpu::ShuffleOp::create(builder, sum, distance, /*width=*/64,
                                         mlir::gpu::ShuffleMode::DOWN)
                .getShuffleResult();
        sum = mlir::arith::AddFOp::create(builder, sum, shuffled);
      }
      Value is_lane_zero =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::eq,
                                      lane_id, IndexConstant(builder, 0));
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
  int64_t rhs_contracting_dimension_ = 0;
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
