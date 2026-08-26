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

#include "xla/backends/gpu/codegen/flydsl/xtile_scan.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "xla/backends/gpu/codegen/flydsl/scan_support.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using mlir::Value;

constexpr int64_t kWaveSize = 64;

int64_t ElementBits(PrimitiveType type) {
  return type == F16 || type == BF16 ? 16 : 32;
}

class FlyXTileScanEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileScanEmitter(const HloFusionAnalysis& analysis)
      : descriptor_(*GetFlyScanDescriptor(analysis)) {
    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    num_warps_ = config.num_warps();
    CHECK_GT(num_warps_, 0);
    CHECK_LE(num_warps_, 16);
    launch_dimensions_ = LaunchDimensions(
        se::BlockDim((descriptor_.rows + num_warps_ - 1) / num_warps_, 1, 1),
        se::ThreadDim(num_warps_ * kWaveSize, 1, 1));
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
  struct BufferState {
    mlir::Type pointer_type;
    mlir::Type memref_type;
    mlir::Type offset_type;
    Value pointer;
    Value layout;
    Value copy_atom;
    Value zero_vector;
  };

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateMLIRModule(
      mlir::MLIRContext& context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment*) const override {
    context.getOrLoadDialect<mlir::fly::FlyDialect>();
    context.getOrLoadDialect<mlir::fly_rocdl::FlyROCDLDialect>();
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();

    mlir::OpBuilder module_builder(&context);
    mlir::Location location =
        mlir::NameLoc::get(module_builder.getStringAttr(fusion.name()));
    mlir::OwningOpRef<mlir::ModuleOp> module =
        llvm_ir::CreateMlirModuleOp(location);
    module.get()->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                          module_builder.getUnitAttr());

    module_builder.setInsertionPointToStart(module->getBody());
    mlir::gpu::GPUModuleOp gpu_module = mlir::gpu::GPUModuleOp::create(
        module_builder, location, "fly_scan_kernels");
    module_builder.setInsertionPointToStart(
        &gpu_module.getBodyRegion().front());

    mlir::Type element_type = emitters::PrimitiveTypeToMlirType(
        descriptor_.element_type, module_builder);
    mlir::fly::AddressSpaceAttr global_address =
        mlir::fly::AddressSpaceAttr::get(&context,
                                         mlir::fly::AddressSpace::Global);
    mlir::Type pointer_type =
        mlir::fly::PointerType::get(element_type, global_address);
    mlir::FunctionType function_type = mlir::FunctionType::get(
        &context, mlir::TypeRange{pointer_type, pointer_type},
        /*results=*/mlir::TypeRange{});
    mlir::gpu::GPUFuncOp kernel = mlir::gpu::GPUFuncOp::create(
        module_builder, location, entry_function_name, function_type);
    kernel.setKernelAttr(module_builder.getUnitAttr());
    kernel.addEntryBlock();

    RETURN_IF_ERROR(EmitKernel(kernel));
    return module;
  }

  absl::Status EmitEntryFunction(const emitters::PartitionedComputations&,
                                 const emitters::CallTargetProvider&,
                                 mlir::func::FuncOp,
                                 const HloFusionInstruction&) const override {
    return absl::UnimplementedError(
        "FlyXTileScanEmitter builds a native gpu.func module.");
  }

  Value I64(mlir::ImplicitLocOpBuilder& builder, int64_t value) const {
    return mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                              value);
  }

  Value Zero(mlir::ImplicitLocOpBuilder& builder) const {
    if (descriptor_.element_type == S32 || descriptor_.element_type == U32) {
      return mlir::arith::ConstantIntOp::create(builder, builder.getI32Type(),
                                                0);
    }
    return mlir::arith::ConstantFloatOp::create(builder, builder.getF32Type(),
                                                llvm::APFloat(0.0f));
  }

  Value Combine(mlir::ImplicitLocOpBuilder& builder, Value lhs,
                Value rhs) const {
    if (descriptor_.element_type == S32 || descriptor_.element_type == U32) {
      return mlir::arith::AddIOp::create(builder, lhs, rhs);
    }
    Value result = mlir::arith::AddFOp::create(builder, lhs, rhs);
    if (descriptor_.element_type == F16 || descriptor_.element_type == BF16) {
      mlir::Type storage_type =
          emitters::PrimitiveTypeToMlirType(descriptor_.element_type, builder);
      result = mlir::arith::TruncFOp::create(builder, storage_type, result);
      result =
          mlir::arith::ExtFOp::create(builder, builder.getF32Type(), result);
    }
    return result;
  }

  Value ToAccumulator(mlir::ImplicitLocOpBuilder& builder, Value value) const {
    if (descriptor_.element_type == F16 || descriptor_.element_type == BF16) {
      return mlir::arith::ExtFOp::create(builder, builder.getF32Type(), value);
    }
    return value;
  }

  Value ToStorage(mlir::ImplicitLocOpBuilder& builder, Value value) const {
    if (descriptor_.element_type == F16 || descriptor_.element_type == BF16) {
      return mlir::arith::TruncFOp::create(
          builder,
          emitters::PrimitiveTypeToMlirType(descriptor_.element_type, builder),
          value);
    }
    return value;
  }

  BufferState CreateBufferState(mlir::ImplicitLocOpBuilder& builder,
                                Value argument) const {
    mlir::MLIRContext* context = builder.getContext();
    mlir::Type element_type =
        emitters::PrimitiveTypeToMlirType(descriptor_.element_type, builder);
    auto vector_type = mlir::VectorType::get({1}, element_type);
    auto shape_attr = mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
    auto tuple_type = mlir::fly::IntTupleType::get(shape_attr);
    Value shape = mlir::fly::MakeIntTupleOp::create(builder, tuple_type,
                                                    mlir::ValueRange{});
    Value stride = mlir::fly::MakeIntTupleOp::create(builder, tuple_type,
                                                     mlir::ValueRange{});
    auto layout_type = mlir::fly::LayoutType::get(shape_attr, shape_attr);
    Value layout =
        mlir::fly::MakeLayoutOp::create(builder, layout_type, shape, stride);
    auto copy_atom_type = mlir::fly::CopyAtomType::get(
        mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(
            context, ElementBits(descriptor_.element_type),
            /*cacheModifier=*/0),
        ElementBits(descriptor_.element_type));
    Value copy_atom = mlir::fly::MakeCopyAtomOp::create(
        builder, copy_atom_type, ElementBits(descriptor_.element_type));

    auto address = mlir::fly_rocdl::BufferDescAddressAttr::get(context);
    auto pointer_type = mlir::fly::PointerType::get(element_type, address);
    Value descriptor_stride =
        mlir::arith::ConstantIntOp::create(builder, builder.getI16Type(), 0);
    Value descriptor_extent = mlir::arith::ConstantIntOp::create(
        builder, builder.getI64Type(),
        descriptor_.rows * descriptor_.row_length *
            (ElementBits(descriptor_.element_type) / 8));
    Value descriptor_flags = mlir::arith::ConstantIntOp::create(
        builder, builder.getI32Type(), 0x27000);
    Value pointer = mlir::fly::MakePtrOp::create(
        builder, pointer_type,
        mlir::ValueRange{argument, descriptor_stride, descriptor_extent,
                         descriptor_flags},
        /*dictAttrs=*/nullptr);
    auto memref_type = mlir::fly::MemRefType::get(
        element_type, pointer_type.getAddressSpace(), layout_type.getAttr());
    auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
        context, /*width=*/32, /*divisibility=*/1);
    auto offset_type = mlir::fly::IntTupleType::get(offset_attr);
    mlir::Attribute zero =
        descriptor_.element_type == S32 || descriptor_.element_type == U32
            ? static_cast<mlir::Attribute>(builder.getI32IntegerAttr(0))
            : builder.getFloatAttr(element_type, 0.0);
    Value zero_vector = mlir::arith::ConstantOp::create(
        builder, vector_type, mlir::DenseElementsAttr::get(vector_type, zero));
    return BufferState{pointer_type, memref_type, offset_type, pointer,
                       layout,       copy_atom,   zero_vector};
  }

  Value OffsetView(mlir::ImplicitLocOpBuilder& builder,
                   const BufferState& state, Value element_offset) const {
    Value offset_i32 = mlir::arith::TruncIOp::create(
        builder, builder.getI32Type(), element_offset);
    Value offset_tuple = mlir::fly::MakeIntTupleOp::create(
        builder, state.offset_type, mlir::ValueRange{offset_i32});
    Value advanced = mlir::fly::AddOffsetOp::create(
        builder, state.pointer_type, state.pointer, offset_tuple);
    return mlir::fly::MakeViewOp::create(builder, state.memref_type, advanced,
                                         state.layout);
  }

  Value Load(mlir::ImplicitLocOpBuilder& builder, const BufferState& state,
             Value element_offset, Value predicate) const {
    Value view = OffsetView(builder, state, element_offset);
    Value vector = mlir::fly::CopyAtomCallSSA::create(
                       builder, mlir::TypeRange{state.zero_vector.getType()},
                       state.copy_atom, view, state.zero_vector, predicate)
                       .getResult(0);
    return mlir::vector::ExtractOp::create(builder, vector, 0);
  }

  void Store(mlir::ImplicitLocOpBuilder& builder, const BufferState& state,
             Value element_offset, Value predicate, Value value) const {
    Value vector = mlir::vector::FromElementsOp::create(
        builder, state.zero_vector.getType(), mlir::ValueRange{value});
    Value view = OffsetView(builder, state, element_offset);
    mlir::fly::CopyAtomCallSSA::create(
        builder, mlir::TypeRange{}, state.copy_atom, vector, view, predicate);
  }

  absl::Status EmitKernel(mlir::gpu::GPUFuncOp kernel) const {
    TF_RET_CHECK(kernel.getNumArguments() == 2);
    mlir::ImplicitLocOpBuilder builder(kernel.getLoc(), kernel);
    builder.setInsertionPointToStart(&kernel.getBody().front());

    Value thread_id =
        mlir::gpu::ThreadIdOp::create(builder, mlir::gpu::Dimension::x);
    Value block_id =
        mlir::gpu::BlockIdOp::create(builder, mlir::gpu::Dimension::x);
    Value thread = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(), thread_id);
    Value block = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(), block_id);
    Value lane =
        mlir::arith::RemUIOp::create(builder, thread, I64(builder, kWaveSize));
    Value wave =
        mlir::arith::DivUIOp::create(builder, thread, I64(builder, kWaveSize));
    Value row = mlir::arith::AddIOp::create(
        builder,
        mlir::arith::MulIOp::create(builder, block, I64(builder, num_warps_)),
        wave);
    Value row_valid =
        mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::ult,
                                    row, I64(builder, descriptor_.rows));
    Value row_base = mlir::arith::MulIOp::create(
        builder, row, I64(builder, descriptor_.row_length));

    BufferState input = CreateBufferState(builder, kernel.getArgument(0));
    BufferState output = CreateBufferState(builder, kernel.getArgument(1));
    Value carry = Zero(builder);
    const int64_t stripes =
        (descriptor_.row_length + kWaveSize - 1) / kWaveSize;
    auto emit_stripe = [&](Value stripe, Value incoming_carry) {
      Value logical = mlir::arith::AddIOp::create(
          builder, lane,
          mlir::arith::MulIOp::create(builder, stripe,
                                      I64(builder, kWaveSize)));
      Value lane_valid = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, logical,
          I64(builder, descriptor_.row_length));
      Value valid = mlir::arith::AndIOp::create(builder, row_valid, lane_valid);
      Value physical_column = logical;
      if (descriptor_.is_reverse) {
        physical_column = mlir::arith::SubIOp::create(
            builder, I64(builder, descriptor_.row_length - 1), logical);
      }
      Value element_offset =
          mlir::arith::AddIOp::create(builder, row_base, physical_column);
      Value value =
          ToAccumulator(builder, Load(builder, input, element_offset, valid));

      Value prefix = value;
      for (int64_t distance : {1, 2, 4, 8, 16, 32}) {
        Value shuffled =
            mlir::gpu::ShuffleOp::create(builder, prefix, distance, kWaveSize,
                                         mlir::gpu::ShuffleMode::UP)
                .getShuffleResult();
        Value has_predecessor = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::uge, lane,
            I64(builder, distance));
        Value combined = Combine(builder, prefix, shuffled);
        prefix = mlir::arith::SelectOp::create(builder, has_predecessor,
                                               combined, prefix);
      }
      prefix = Combine(builder, incoming_carry, prefix);
      Store(builder, output, element_offset, valid, ToStorage(builder, prefix));
      return mlir::gpu::ShuffleOp::create(builder, prefix, /*offset=*/63,
                                          /*width=*/kWaveSize,
                                          mlir::gpu::ShuffleMode::IDX)
          .getShuffleResult();
    };
    constexpr int64_t kMaxUnrolledStripes = 16;
    if (stripes <= kMaxUnrolledStripes) {
      for (int64_t stripe = 0; stripe < stripes; ++stripe) {
        carry = emit_stripe(I64(builder, stripe), carry);
      }
    } else {
      mlir::scf::ForOp stripe_loop = mlir::scf::ForOp::create(
          builder, I64(builder, 0), I64(builder, stripes), I64(builder, 1),
          mlir::ValueRange{carry},
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(stripe_loop.getBody());
        Value next_carry = emit_stripe(stripe_loop.getInductionVar(),
                                       stripe_loop.getRegionIterArg(0));
        mlir::scf::YieldOp::create(builder, next_carry);
      }
      builder.setInsertionPointAfter(stripe_loop);
    }
    mlir::gpu::ReturnOp::create(builder);
    return absl::OkStatus();
  }

  FlyScanDescriptor descriptor_;
  int64_t num_warps_ = 0;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileScanEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileScanEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
