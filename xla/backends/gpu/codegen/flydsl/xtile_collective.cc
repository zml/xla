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

#include "xla/backends/gpu/codegen/flydsl/xtile_collective.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using ::xla::se::gpu::AllReduceStrategy;
using mlir::Value;

constexpr int64_t kTransactionBytes = 16;
constexpr int64_t kFlyCollectiveThreads = 512;
constexpr int64_t kFlyCollectiveMaxBlocks = 32;

class FlyXTileCollectiveEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileCollectiveEmitter(const HloFusionAnalysis& analysis,
                                     CollectiveEpilogue producer,
                                     CollectiveEpilogue epilogue)
      : producer_(std::move(producer)), epilogue_(std::move(epilogue)) {
    const HloInstruction& root = analysis.fusion_root(0).instruction();
    collective_opcode_ = root.opcode();
    if (const auto* all_reduce = DynCast<HloAllReduceInstruction>(&root)) {
      CHECK_EQ(all_reduce->operand_count(), 1);
      element_type_ = all_reduce->shape().element_type();
      num_elements_ = ShapeUtil::ElementsIn(all_reduce->shape());
      output_elements_ = num_elements_;
      world_size_ = all_reduce->device_list()->num_devices_per_group();
      reduction_opcode_ = all_reduce->called_computations()
                              .front()
                              ->root_instruction()
                              ->opcode();
      const int64_t element_bytes = primitive_util::ByteWidth(element_type_);
      strategy_ = GetAllReduceStrategy(num_elements_ * element_bytes,
                                       /*is_multimem_enabled=*/false);
    } else {
      const auto* all_gather = Cast<HloAllGatherInstruction>(&root);
      CHECK_EQ(all_gather->operand_count(), 1);
      CHECK(ShapeUtil::IsEffectivelyMostMajorDimension(
          all_gather->operand(0)->shape(),
          all_gather->all_gather_dimension()));
      element_type_ = all_gather->operand(0)->shape().element_type();
      num_elements_ = ShapeUtil::ElementsIn(all_gather->operand(0)->shape());
      output_elements_ = ShapeUtil::ElementsIn(all_gather->shape());
      world_size_ = all_gather->device_list()->num_devices_per_group();
      CHECK_EQ(output_elements_, num_elements_ * world_size_);
      strategy_ = AllReduceStrategy::kOneShot;
    }
    CHECK_GT(world_size_, 0);
    CHECK_LE(world_size_, se::gpu::kMaxNumAllReduceInputPtrs);

    const int64_t element_bytes = primitive_util::ByteWidth(element_type_);
    vector_lanes_ = std::max<int64_t>(1, kTransactionBytes / element_bytes);
    // Small BF16 reductions are latency-bound. Match Triton's four-element
    // per-thread schedule here: it halves each thread's dependent reduction
    // chain and gives every one of the 32 workgroups useful work.
    if (collective_opcode_ == HloOpcode::kAllReduce &&
        element_type_ == BF16 &&
        num_elements_ * element_bytes <= 1024 * 1024) {
      vector_lanes_ = 4;
    }
    io_vector_lanes_ =
        std::max<int64_t>(vector_lanes_, kTransactionBytes / element_bytes);
    segment_elements_ =
        (num_elements_ + world_size_ - 1) / world_size_;
    // AllReduceLaunchDimensions is shared with kernels that process four
    // elements per thread. This emitter instead performs one 16-byte vector
    // transaction per thread, which is eight elements for BF16. Derive the
    // launch from the actual vector width; otherwise half of the one-shot BF16
    // blocks are data-idle but still pay the inter-GPU signal barrier.
    const int64_t work_elements =
        strategy_ == AllReduceStrategy::kTwoShot ? segment_elements_
                                                 : num_elements_;
    const int64_t wave_size = WarpSize(analysis.device_info());
    const int64_t logical_threads =
        (work_elements + vector_lanes_ - 1) / vector_lanes_;
    const int64_t total_threads =
        ((logical_threads + wave_size - 1) / wave_size) * wave_size;
    // All-gather performs eight peer copies after its local publish on the
    // common eight-rank path. Spreading the same useful threads over smaller
    // workgroups lets MI300X schedule more independent stripes while retaining
    // 16-byte vector transactions. The 32-workgroup cap balances CU occupancy
    // against the cost and launch-skew sensitivity of cross-rank barriers.
    const int64_t max_threads_per_block =
        collective_opcode_ == HloOpcode::kAllGather
            ? kFlyCollectiveThreads / 2
            : kFlyCollectiveThreads;
    const int64_t threads_per_block = std::min<int64_t>(
        max_threads_per_block,
        llvm::bit_ceil(static_cast<uint64_t>(total_threads)));
    const int64_t blocks = std::min<int64_t>(
        kFlyCollectiveMaxBlocks,
        (total_threads + threads_per_block - 1) / threads_per_block);
    launch_dimensions_ = LaunchDimensions(blocks, threads_per_block);
    full_vectors_only_ =
        num_elements_ % vector_lanes_ == 0 &&
        (strategy_ != AllReduceStrategy::kTwoShot ||
         segment_elements_ % vector_lanes_ == 0);
    // Follow FlyDSL's native collective schedule on the performance-critical
    // eight-rank BF16 sum path: one wave loads each peer, then wave zero
    // reduces the values through LDS. All waves must execute the same number
    // of barrier-containing loop iterations, so retain the generic exact-tail
    // implementation when the pack count does not divide the launch evenly.
    if (collective_opcode_ == HloOpcode::kAllReduce && full_vectors_only_ &&
        element_type_ == BF16 &&
        reduction_opcode_ == HloOpcode::kAdd && world_size_ == 8 &&
        strategy_ == AllReduceStrategy::kTwoShot &&
        num_elements_ * element_bytes >= 2 * 1024 * 1024) {
      const int64_t threads_per_rank = kFlyCollectiveThreads / world_size_;
      const int64_t work_packs = segment_elements_ / vector_lanes_;
      const int64_t blocks = std::min<int64_t>(
          kFlyCollectiveMaxBlocks,
          (work_packs + threads_per_rank - 1) / threads_per_rank);
      if (blocks > 0 &&
          work_packs % (blocks * threads_per_rank) == 0) {
        use_lds_algorithm_ = true;
        launch_dimensions_ =
            LaunchDimensions(blocks, kFlyCollectiveThreads);
      }
    }
    scratch_elements_per_buffer_ =
        ((num_elements_ * element_bytes + kXlaAllocatedBufferAlignBytes - 1) /
         kXlaAllocatedBufferAlignBytes) *
        kXlaAllocatedBufferAlignBytes / element_bytes;
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
      const BufferAssignment*) const override {
    context.getOrLoadDialect<mlir::fly::FlyDialect>();
    context.getOrLoadDialect<mlir::fly_rocdl::FlyROCDLDialect>();
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
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
        module_builder, location, "fly_collective_kernels");
    module_builder.setInsertionPointToStart(
        &gpu_module.getBodyRegion().front());

    mlir::Type storage_type = StorageType(module_builder);
    auto global_address = mlir::fly::AddressSpaceAttr::get(
        &context, mlir::fly::AddressSpace::Global);
    mlir::Type data_pointer =
        mlir::fly::PointerType::get(storage_type, global_address);
    mlir::Type table_pointer =
        mlir::fly::PointerType::get(module_builder.getI64Type(),
                                    global_address);
    llvm::SmallVector<mlir::Type> argument_types{data_pointer, data_pointer};
    for (int64_t input = 0;
         input < producer_.buffer_count + epilogue_.buffer_count; ++input) {
      argument_types.push_back(data_pointer);
    }
    argument_types.append({module_builder.getI32Type(),
                           module_builder.getI32Type(), table_pointer,
                           table_pointer});
    mlir::FunctionType function_type = mlir::FunctionType::get(
        &context, argument_types, /*results=*/mlir::TypeRange{});
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
        "FlyXTileCollectiveEmitter builds a native gpu.func module.");
  }

  mlir::Type StorageType(mlir::Builder& builder) const {
    switch (element_type_) {
      case F16:
        return builder.getF16Type();
      case F32:
        return builder.getF32Type();
      case BF16:
        return builder.getBF16Type();
      case F64:
        return builder.getF64Type();
      case S32:
        return builder.getI32Type();
      case S64:
        return builder.getI64Type();
      case S8:
        return builder.getI8Type();
      case S16:
        return builder.getI16Type();
      case PRED:
        // XLA's device ABI stores predicates as bytes.
        return builder.getI8Type();
      default:
        LOG(FATAL) << "Unsupported Fly collective element type: "
                   << PrimitiveType_Name(element_type_);
    }
  }

  Value I32(mlir::ImplicitLocOpBuilder& builder, int32_t value) const {
    return mlir::arith::ConstantIntOp::create(builder, builder.getI32Type(),
                                              value);
  }

  Value I64(mlir::ImplicitLocOpBuilder& builder, int64_t value) const {
    return mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                              value);
  }

  Value ToI64(mlir::ImplicitLocOpBuilder& builder, Value value) const {
    if (value.getType().isInteger(64)) return value;
    if (value.getType().isIndex()) {
      return mlir::arith::IndexCastOp::create(builder, builder.getI64Type(),
                                              value);
    }
    return mlir::arith::ExtUIOp::create(builder, builder.getI64Type(), value);
  }

  Value AddOffset(mlir::ImplicitLocOpBuilder& builder, Value pointer,
                  Value element_offset) const {
    element_offset = ToI64(builder, element_offset);
    Value offset_i32 = mlir::arith::TruncIOp::create(
        builder, builder.getI32Type(), element_offset);
    auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
        builder.getContext(), /*width=*/32, /*divisibility=*/1);
    auto offset_type = mlir::fly::IntTupleType::get(offset_attr);
    Value offset = mlir::fly::MakeIntTupleOp::create(
        builder, offset_type, mlir::ValueRange{offset_i32});
    return mlir::fly::AddOffsetOp::create(builder, pointer.getType(), pointer,
                                          offset);
  }

  Value PointerAddressFromTable(mlir::ImplicitLocOpBuilder& builder,
                                Value table, Value rank) const {
    return mlir::fly::PtrLoadOp::create(
        builder, builder.getI64Type(), AddOffset(builder, table, rank));
  }

  Value PointerFromTable(mlir::ImplicitLocOpBuilder& builder, Value table,
                         Value rank, mlir::Type pointee_type) const {
    Value address = PointerAddressFromTable(builder, table, rank);
    auto global_address = mlir::fly::AddressSpaceAttr::get(
        builder.getContext(), mlir::fly::AddressSpace::Global);
    auto pointer_type =
        mlir::fly::PointerType::get(pointee_type, global_address);
    return mlir::fly::IntToPtrOp::create(builder, pointer_type, address);
  }

  Value BufferPointer(mlir::ImplicitLocOpBuilder& builder, Value base,
                      int64_t allocation_bytes) const {
    auto buffer_address = mlir::fly_rocdl::BufferDescAddressAttr::get(
        builder.getContext());
    auto pointer_type = mlir::fly::PointerType::get(StorageType(builder),
                                                    buffer_address);
    Value stride = mlir::arith::ConstantIntOp::create(
        builder, builder.getI16Type(), 0);
    Value extent = I64(builder, allocation_bytes);
    // CDNA buffer descriptor: DATA_FORMAT=7 and NUM_FORMAT=4.
    Value flags = I32(builder, 0x27000);
    return mlir::fly::MakePtrOp::create(
        builder, pointer_type, mlir::ValueRange{base, stride, extent, flags},
        /*dictAttrs=*/nullptr);
  }

  Value SharedPointer(mlir::ImplicitLocOpBuilder& builder,
                      int64_t buffers) const {
    mlir::MLIRContext* context = builder.getContext();
    constexpr int64_t kAlignment = 16;
    auto shared_address = mlir::fly::AddressSpaceAttr::get(
        context, mlir::fly::AddressSpace::Shared);
    auto alignment = mlir::fly::AlignAttr::get(context, kAlignment);
    auto raw_pointer_type = mlir::fly::PointerType::get(
        builder.getI8Type(), shared_address, alignment);
    auto allocation = builder.getDictionaryAttr(
        {builder.getNamedAttr("allocAlign",
                              builder.getI64IntegerAttr(kAlignment)),
         builder.getNamedAttr(
             "allocBytes",
             builder.getI64IntegerAttr(buffers * kFlyCollectiveThreads *
                                       kTransactionBytes))});
    Value raw_pointer = mlir::fly::MakePtrOp::create(
        builder, raw_pointer_type, mlir::ValueRange{}, allocation);
    auto pointer_type = mlir::fly::PointerType::get(
        StorageType(builder), shared_address, alignment);
    return mlir::fly::RecastIterOp::create(builder, pointer_type,
                                           raw_pointer);
  }

  Value ScratchPointer(mlir::ImplicitLocOpBuilder& builder, Value table,
                       Value rank, Value buffer_offset) const {
    Value pointer;
    if (full_vectors_only_) {
      Value address = PointerAddressFromTable(builder, table, rank);
      auto global_address = mlir::fly::AddressSpaceAttr::get(
          builder.getContext(), mlir::fly::AddressSpace::Global);
      auto global_pointer_type =
          mlir::fly::PointerType::get(StorageType(builder), global_address);
      Value base = mlir::fly::IntToPtrOp::create(builder, global_pointer_type,
                                                 address);
      pointer = BufferPointer(
          builder, base,
          2 * scratch_elements_per_buffer_ *
              primitive_util::ByteWidth(element_type_));
    } else {
      pointer =
          PointerFromTable(builder, table, rank, StorageType(builder));
    }
    return AddOffset(builder, pointer, buffer_offset);
  }

  Value ZeroVector(mlir::ImplicitLocOpBuilder& builder,
                   mlir::VectorType vector_type) const {
    mlir::Attribute zero = builder.getZeroAttr(vector_type.getElementType());
    return mlir::arith::ConstantOp::create(
        builder, vector_type,
        mlir::DenseElementsAttr::get(vector_type, zero));
  }

  Value LoadVector(mlir::ImplicitLocOpBuilder& builder, Value pointer,
                   Value element_offset, Value limit,
                   mlir::VectorType vector_type) const {
    if (full_vectors_only_) {
      return mlir::fly::PtrLoadOp::create(
          builder, vector_type, AddOffset(builder, pointer, element_offset));
    }
    Value end = mlir::arith::AddIOp::create(
        builder, ToI64(builder, element_offset), I64(builder, vector_lanes_));
    Value full = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::ule, end, ToI64(builder, limit));
    mlir::scf::IfOp load = mlir::scf::IfOp::create(
        builder, mlir::TypeRange{vector_type}, full,
        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(load.thenBlock());
      Value full_value = mlir::fly::PtrLoadOp::create(
          builder, vector_type, AddOffset(builder, pointer, element_offset));
      mlir::scf::YieldOp::create(builder, full_value);

      builder.setInsertionPointToStart(load.elseBlock());
      Value value = ZeroVector(builder, vector_type);
      for (int64_t lane = 0; lane < vector_lanes_; ++lane) {
        Value lane_offset = mlir::arith::AddIOp::create(
            builder, ToI64(builder, element_offset), I64(builder, lane));
        Value valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, lane_offset,
            ToI64(builder, limit));
        mlir::scf::IfOp lane_load = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{vector_type}, valid,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard lane_guard(builder);
          builder.setInsertionPointToStart(lane_load.thenBlock());
          Value scalar = mlir::fly::PtrLoadOp::create(
              builder, vector_type.getElementType(),
              AddOffset(builder, pointer, lane_offset));
          Value inserted = mlir::vector::InsertOp::create(
              builder, scalar, value, mlir::ArrayRef<int64_t>{lane});
          mlir::scf::YieldOp::create(builder, inserted);
          builder.setInsertionPointToStart(lane_load.elseBlock());
          mlir::scf::YieldOp::create(builder, value);
        }
        builder.setInsertionPointAfter(lane_load);
        value = lane_load.getResult(0);
      }
      mlir::scf::YieldOp::create(builder, value);
    }
    builder.setInsertionPointAfter(load);
    return load.getResult(0);
  }

  void StoreVector(mlir::ImplicitLocOpBuilder& builder, Value value,
                   Value pointer, Value element_offset, Value limit) const {
    if (full_vectors_only_) {
      mlir::fly::PtrStoreOp::create(
          builder, value, AddOffset(builder, pointer, element_offset));
      return;
    }
    Value end = mlir::arith::AddIOp::create(
        builder, ToI64(builder, element_offset), I64(builder, vector_lanes_));
    Value full = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::ule, end, ToI64(builder, limit));
    mlir::scf::IfOp store = mlir::scf::IfOp::create(
        builder, mlir::TypeRange{}, full, /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(store.thenBlock());
      mlir::fly::PtrStoreOp::create(
          builder, value, AddOffset(builder, pointer, element_offset));

      builder.setInsertionPointToStart(store.elseBlock());
      for (int64_t lane = 0; lane < vector_lanes_; ++lane) {
        Value lane_offset = mlir::arith::AddIOp::create(
            builder, ToI64(builder, element_offset), I64(builder, lane));
        Value valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, lane_offset,
            ToI64(builder, limit));
        mlir::scf::IfOp lane_store = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{}, valid, /*withElseRegion=*/false);
        {
          mlir::OpBuilder::InsertionGuard lane_guard(builder);
          builder.setInsertionPointToStart(lane_store.thenBlock());
          Value scalar = mlir::vector::ExtractOp::create(builder, value, lane);
          mlir::fly::PtrStoreOp::create(
              builder, scalar, AddOffset(builder, pointer, lane_offset));
        }
        builder.setInsertionPointAfter(lane_store);
      }
    }
    builder.setInsertionPointAfter(store);
  }

  Value SplatInteger(mlir::ImplicitLocOpBuilder& builder,
                     mlir::VectorType type, int64_t value) const {
    return mlir::arith::ConstantOp::create(
        builder, type,
        mlir::DenseElementsAttr::get(
            type, builder.getIntegerAttr(type.getElementType(), value)));
  }

  Value SplatFloat(mlir::ImplicitLocOpBuilder& builder, mlir::VectorType type,
                   double value) const {
    return mlir::arith::ConstantOp::create(
        builder, type,
        mlir::DenseElementsAttr::get(
            type, builder.getFloatAttr(type.getElementType(), value)));
  }

  Value Bf16PairToF32(mlir::ImplicitLocOpBuilder& builder, Value pair) const {
    auto i16_type = mlir::VectorType::get({2}, builder.getI16Type());
    auto i32_type = mlir::VectorType::get({2}, builder.getI32Type());
    auto f32_type = mlir::VectorType::get({2}, builder.getF32Type());
    Value bits = mlir::arith::BitcastOp::create(builder, i16_type, pair);
    bits = mlir::arith::ExtUIOp::create(builder, i32_type, bits);
    bits = mlir::arith::ShLIOp::create(builder, bits,
                                       SplatInteger(builder, i32_type, 16));
    return mlir::arith::BitcastOp::create(builder, f32_type, bits);
  }

  Value RoundF32PairToBf16(mlir::ImplicitLocOpBuilder& builder,
                           Value value) const {
    auto i16_type = mlir::VectorType::get({2}, builder.getI16Type());
    auto i32_type = mlir::VectorType::get({2}, builder.getI32Type());
    auto bf16_type = mlir::VectorType::get({2}, builder.getBF16Type());
    Value bits = mlir::arith::BitcastOp::create(builder, i32_type, value);
    Value shift = SplatInteger(builder, i32_type, 16);
    Value retained_lsb = mlir::arith::AndIOp::create(
        builder, mlir::arith::ShRUIOp::create(builder, bits, shift),
        SplatInteger(builder, i32_type, 1));
    Value rounded = mlir::arith::AddIOp::create(
        builder,
        mlir::arith::AddIOp::create(builder, bits,
                                    SplatInteger(builder, i32_type, 0x7fff)),
        retained_lsb);
    rounded = mlir::arith::ShRUIOp::create(builder, rounded, shift);
    Value narrowed = mlir::arith::TruncIOp::create(builder, i16_type, rounded);
    Value result = mlir::arith::BitcastOp::create(builder, bf16_type, narrowed);
    Value is_nan = mlir::arith::CmpFOp::create(
        builder, mlir::arith::CmpFPredicate::UNO, value, value);
    Value canonical_nan = mlir::arith::BitcastOp::create(
        builder, bf16_type, SplatInteger(builder, i16_type, 0x7fc0));
    return mlir::arith::SelectOp::create(builder, is_nan, canonical_nan,
                                         result);
  }

  Value AssemblePairs(mlir::ImplicitLocOpBuilder& builder,
                      llvm::SmallVector<Value> pairs) const {
    CHECK(!pairs.empty());
    while (pairs.size() > 1) {
      CHECK_EQ(pairs.size() % 2, 0);
      llvm::SmallVector<Value> combined;
      combined.reserve(pairs.size() / 2);
      for (int64_t pair = 0; pair < pairs.size(); pair += 2) {
        auto input_type = mlir::cast<mlir::VectorType>(pairs[pair].getType());
        const int64_t input_elements = input_type.getNumElements();
        auto result_type = mlir::VectorType::get({2 * input_elements},
                                                 input_type.getElementType());
        llvm::SmallVector<int64_t> mask;
        mask.reserve(2 * input_elements);
        for (int64_t element = 0; element < 2 * input_elements; ++element) {
          mask.push_back(element);
        }
        combined.push_back(mlir::vector::ShuffleOp::create(
            builder, result_type, pairs[pair], pairs[pair + 1], mask));
      }
      pairs = std::move(combined);
    }
    return pairs.front();
  }

  Value PairwiseBf16Add(mlir::ImplicitLocOpBuilder& builder, Value lhs,
                        Value rhs) const {
    auto vector_type = mlir::cast<mlir::VectorType>(lhs.getType());
    const int64_t vector_width = vector_type.getNumElements();
    CHECK_EQ(vector_width % 2, 0);
    auto bf16_pair_type = mlir::VectorType::get({2}, builder.getBF16Type());
    llvm::SmallVector<Value> pairs;
    pairs.reserve(vector_width / 2);
    for (int64_t pair = 0; pair < vector_width / 2; ++pair) {
      llvm::SmallVector<int64_t, 2> mask{2 * pair, 2 * pair + 1};
      Value lhs_pair = mlir::vector::ShuffleOp::create(
          builder, bf16_pair_type, lhs, lhs, mask);
      Value rhs_pair = mlir::vector::ShuffleOp::create(
          builder, bf16_pair_type, rhs, rhs, mask);
      lhs_pair = Bf16PairToF32(builder, lhs_pair);
      rhs_pair = Bf16PairToF32(builder, rhs_pair);
      Value sum = mlir::arith::AddFOp::create(builder, lhs_pair, rhs_pair);
      pairs.push_back(RoundF32PairToBf16(builder, sum));
    }
    return AssemblePairs(builder, std::move(pairs));
  }

  Value Bf16VectorToF32(mlir::ImplicitLocOpBuilder& builder,
                        Value value) const {
    auto vector_type = mlir::cast<mlir::VectorType>(value.getType());
    const int64_t vector_width = vector_type.getNumElements();
    CHECK_EQ(vector_width % 2, 0);
    auto bf16_pair_type = mlir::VectorType::get({2}, builder.getBF16Type());
    llvm::SmallVector<Value> pairs;
    pairs.reserve(vector_width / 2);
    for (int64_t pair = 0; pair < vector_width / 2; ++pair) {
      llvm::SmallVector<int64_t, 2> mask{2 * pair, 2 * pair + 1};
      pairs.push_back(Bf16PairToF32(
          builder, mlir::vector::ShuffleOp::create(
                       builder, bf16_pair_type, value, value, mask)));
    }
    return AssemblePairs(builder, std::move(pairs));
  }

  Value RoundF32VectorToBf16(mlir::ImplicitLocOpBuilder& builder,
                             Value value) const {
    auto vector_type = mlir::cast<mlir::VectorType>(value.getType());
    const int64_t vector_width = vector_type.getNumElements();
    CHECK_EQ(vector_width % 2, 0);
    auto f32_pair_type = mlir::VectorType::get({2}, builder.getF32Type());
    llvm::SmallVector<Value> pairs;
    pairs.reserve(vector_width / 2);
    for (int64_t pair = 0; pair < vector_width / 2; ++pair) {
      llvm::SmallVector<int64_t, 2> mask{2 * pair, 2 * pair + 1};
      pairs.push_back(RoundF32PairToBf16(
          builder, mlir::vector::ShuffleOp::create(
                       builder, f32_pair_type, value, value, mask)));
    }
    return AssemblePairs(builder, std::move(pairs));
  }

  Value Combine(mlir::ImplicitLocOpBuilder& builder, Value lhs,
                Value rhs) const {
    const bool floating = element_type_ == F32 || element_type_ == BF16 ||
                          element_type_ == F64;
    switch (reduction_opcode_) {
      case HloOpcode::kAdd:
        if (element_type_ == BF16) {
          return PairwiseBf16Add(builder, lhs, rhs);
        }
        return floating ? static_cast<Value>(
                              mlir::arith::AddFOp::create(builder, lhs, rhs))
                        : static_cast<Value>(
                              mlir::arith::AddIOp::create(builder, lhs, rhs));
      case HloOpcode::kMultiply:
        return floating ? static_cast<Value>(
                              mlir::arith::MulFOp::create(builder, lhs, rhs))
                        : static_cast<Value>(
                              mlir::arith::MulIOp::create(builder, lhs, rhs));
      case HloOpcode::kMaximum:
        return floating
                   ? static_cast<Value>(mlir::arith::MaximumFOp::create(
                         builder, lhs, rhs))
                   : static_cast<Value>(
                         mlir::arith::MaxSIOp::create(builder, lhs, rhs));
      case HloOpcode::kMinimum:
        return floating
                   ? static_cast<Value>(mlir::arith::MinimumFOp::create(
                         builder, lhs, rhs))
                   : static_cast<Value>(
                         mlir::arith::MinSIOp::create(builder, lhs, rhs));
      case HloOpcode::kOr:
        return mlir::arith::OrIOp::create(builder, lhs, rhs);
      case HloOpcode::kAnd:
        return mlir::arith::AndIOp::create(builder, lhs, rhs);
      default:
        LOG(FATAL) << "Unsupported Fly collective reduction: "
                   << HloOpcodeString(reduction_opcode_);
    }
  }

  Value ApplyElementwise(mlir::ImplicitLocOpBuilder& builder,
                         const CollectiveEpilogue& program, Value value,
                         llvm::ArrayRef<Value> inputs, Value offset,
                         Value limit) const {
    if (program.empty()) {
      return value;
    }
    CHECK_EQ(inputs.size(), program.buffer_count);
    auto bf16_type = mlir::cast<mlir::VectorType>(value.getType());
    auto f32_type = mlir::VectorType::get(bf16_type.getShape(),
                                          builder.getF32Type());
    Value accumulator = Bf16VectorToF32(builder, value);
    for (const CollectiveEpilogueStep& step : program.steps) {
      Value operand;
      if (step.uses_buffer()) {
        CHECK_LT(step.buffer_index, inputs.size());
        operand = Bf16VectorToF32(
            builder, LoadVector(builder, inputs[step.buffer_index], offset,
                                limit, bf16_type));
      } else {
        operand = SplatFloat(builder, f32_type, step.scalar_value);
      }
      Value lhs = step.accumulator_is_lhs ? accumulator : operand;
      Value rhs = step.accumulator_is_lhs ? operand : accumulator;
      switch (step.opcode) {
        case HloOpcode::kAdd:
          accumulator = mlir::arith::AddFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kSubtract:
          accumulator = mlir::arith::SubFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kMultiply:
          accumulator = mlir::arith::MulFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kDivide:
          accumulator = mlir::arith::DivFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kMaximum:
          accumulator = mlir::arith::MaximumFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kMinimum:
          accumulator = mlir::arith::MinimumFOp::create(builder, lhs, rhs);
          break;
        default:
          LOG(FATAL) << "Unsupported Fly collective epilogue operation: "
                     << HloOpcodeString(step.opcode);
      }
    }
    return RoundF32VectorToBf16(builder, accumulator);
  }

  Value ApplyProducer(mlir::ImplicitLocOpBuilder& builder, Value value,
                      llvm::ArrayRef<Value> inputs, Value offset,
                      Value limit) const {
    return ApplyElementwise(builder, producer_, value, inputs, offset, limit);
  }

  Value ApplyEpilogue(mlir::ImplicitLocOpBuilder& builder, Value value,
                      llvm::ArrayRef<Value> inputs, Value offset,
                      Value limit) const {
    return ApplyElementwise(builder, epilogue_, value, inputs, offset, limit);
  }

  void EmitSync(mlir::ImplicitLocOpBuilder& builder, Value signal_table,
                Value rank, Value signal_value) const {
    // Match XLA Triton's protocol: a workgroup barrier orders all data writes,
    // one thread per peer performs a system-scope release, then spins on the
    // corresponding local flag with a system-scope acquire.
    mlir::gpu::BarrierOp::create(builder);
    Value thread_index = mlir::gpu::ThreadIdOp::create(
        builder, mlir::gpu::Dimension::x);
    Value thread = mlir::arith::IndexCastOp::create(
        builder, builder.getI32Type(), thread_index);
    Value participates = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::ult, thread,
        I32(builder, world_size_));
    mlir::scf::IfOp sync = mlir::scf::IfOp::create(
        builder, mlir::TypeRange{}, participates, /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(sync.thenBlock());
      Value block_index = mlir::gpu::BlockIdOp::create(
          builder, mlir::gpu::Dimension::x);
      Value block = mlir::arith::IndexCastOp::create(
          builder, builder.getI32Type(), block_index);
      Value block_base = mlir::arith::MulIOp::create(
          builder, block, I32(builder, world_size_));

      Value peer_signal = PointerFromTable(
          builder, signal_table, thread, builder.getI32Type());
      Value write_offset =
          mlir::arith::AddIOp::create(builder, block_base, rank);
      Value write_pointer = AddOffset(builder, peer_signal, write_offset);
      auto llvm_global_pointer = mlir::LLVM::LLVMPointerType::get(
          builder.getContext(), /*addressSpace=*/1);
      Value llvm_write_pointer = mlir::fly::ToLLVMPtrOp::create(
          builder, llvm_global_pointer, write_pointer,
          /*llvm_address_space=*/1);
      mlir::LLVM::StoreOp::create(
          builder, signal_value, llvm_write_pointer,
          /*alignment=*/4, /*isVolatile=*/false, /*isNonTemporal=*/false,
          /*isInvariantGroup=*/false, mlir::LLVM::AtomicOrdering::release,
          /*syncscope=*/"");

      Value local_signal = PointerFromTable(
          builder, signal_table, rank, builder.getI32Type());
      Value read_offset =
          mlir::arith::AddIOp::create(builder, block_base, thread);
      Value read_pointer = AddOffset(builder, local_signal, read_offset);
      Value llvm_read_pointer = mlir::fly::ToLLVMPtrOp::create(
          builder, llvm_global_pointer, read_pointer,
          /*llvm_address_space=*/1);
      mlir::scf::WhileOp::create(
          builder, mlir::TypeRange{}, mlir::ValueRange{},
          [&](mlir::OpBuilder& op_builder, mlir::Location location,
              mlir::ValueRange) {
            mlir::ImplicitLocOpBuilder loop_builder(location, op_builder);
            Value observed = mlir::LLVM::LoadOp::create(
                loop_builder, loop_builder.getI32Type(), llvm_read_pointer,
                /*alignment=*/4, /*isVolatile=*/false,
                /*isNonTemporal=*/false, /*isInvariant=*/false,
                /*isInvariantGroup=*/false,
                mlir::LLVM::AtomicOrdering::acquire,
                /*syncscope=*/"");
            Value waiting = mlir::arith::CmpIOp::create(
                loop_builder, mlir::arith::CmpIPredicate::ult, observed,
                signal_value);
            mlir::scf::ConditionOp::create(loop_builder, waiting,
                                           mlir::ValueRange{});
          },
          [&](mlir::OpBuilder& op_builder, mlir::Location location,
              mlir::ValueRange) {
            mlir::scf::YieldOp::create(op_builder, location);
          });
    }
    builder.setInsertionPointAfter(sync);
    mlir::gpu::BarrierOp::create(builder);
  }

  struct RankGroupIndex {
    Value thread;
    Value group;
    Value lane;
    Value first_pack;
    Value pack_stride;
  };

  RankGroupIndex GetRankGroupIndex(
      mlir::ImplicitLocOpBuilder& builder) const {
    const int64_t threads_per_rank =
        kFlyCollectiveThreads / world_size_;
    Value thread = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(),
        mlir::gpu::ThreadIdOp::create(builder, mlir::gpu::Dimension::x));
    Value block = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(),
        mlir::gpu::BlockIdOp::create(builder, mlir::gpu::Dimension::x));
    Value group = mlir::arith::DivUIOp::create(
        builder, thread, I64(builder, threads_per_rank));
    Value lane = mlir::arith::RemUIOp::create(
        builder, thread, I64(builder, threads_per_rank));
    Value first_pack = mlir::arith::AddIOp::create(
        builder,
        mlir::arith::MulIOp::create(builder, block,
                                    I64(builder, threads_per_rank)),
        lane);
    Value pack_stride =
        I64(builder, launch_dimensions_.num_blocks() * threads_per_rank);
    return {thread, group, lane, first_pack, pack_stride};
  }

  void EmitWideCopy(mlir::ImplicitLocOpBuilder& builder,
                    llvm::ArrayRef<Value> producer_inputs, Value input,
                    Value local_scratch) const {
    const int64_t tile_elements =
        num_elements_ / launch_dimensions_.num_blocks();
    auto vector_type = mlir::VectorType::get(
        {io_vector_lanes_}, StorageType(builder));
    Value thread = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(),
        mlir::gpu::ThreadIdOp::create(builder, mlir::gpu::Dimension::x));
    Value block = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(),
        mlir::gpu::BlockIdOp::create(builder, mlir::gpu::Dimension::x));
    Value tile_base = mlir::arith::MulIOp::create(
        builder, block, I64(builder, tile_elements));
    Value first = mlir::arith::AddIOp::create(
        builder, tile_base,
        mlir::arith::MulIOp::create(builder, thread,
                                    I64(builder, io_vector_lanes_)));
    Value limit = mlir::arith::AddIOp::create(
        builder, tile_base, I64(builder, tile_elements));
    Value stride = I64(builder, launch_dimensions_.num_threads_per_block() *
                                    io_vector_lanes_);
    mlir::scf::ForOp copy = mlir::scf::ForOp::create(
        builder, first, limit, stride);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(copy.getBody());
      Value offset = copy.getInductionVar();
      StoreVector(builder,
                  ApplyProducer(
                      builder,
                      LoadVector(builder, input, offset, limit, vector_type),
                      producer_inputs, offset, limit),
                  local_scratch, offset, limit);
    }
    builder.setInsertionPointAfter(copy);
  }

  void EmitWideGather(mlir::ImplicitLocOpBuilder& builder,
                      llvm::ArrayRef<Value> epilogue_inputs, Value output,
                      Value scratch_table, Value buffer_offset) const {
    const int64_t tile_elements =
        num_elements_ / launch_dimensions_.num_blocks();
    const int64_t subtile_elements = tile_elements / world_size_;
    auto vector_type = mlir::VectorType::get(
        {io_vector_lanes_}, StorageType(builder));
    Value thread = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(),
        mlir::gpu::ThreadIdOp::create(builder, mlir::gpu::Dimension::x));
    Value block = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(),
        mlir::gpu::BlockIdOp::create(builder, mlir::gpu::Dimension::x));
    Value first = mlir::arith::MulIOp::create(
        builder, thread, I64(builder, io_vector_lanes_));
    Value stride = I64(builder,
                       launch_dimensions_.num_threads_per_block() *
                                    io_vector_lanes_);
    Value tile_base = mlir::arith::MulIOp::create(
        builder, block, I64(builder, tile_elements));
    mlir::scf::ForOp gather =
        mlir::scf::ForOp::create(builder, first, I64(builder, tile_elements),
                                 stride);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(gather.getBody());
      Value local_offset = gather.getInductionVar();
      Value source_rank = mlir::arith::DivUIOp::create(
          builder, local_offset, I64(builder, subtile_elements));
      Value source =
          ScratchPointer(builder, scratch_table, source_rank, buffer_offset);
      Value offset = mlir::arith::AddIOp::create(
          builder, tile_base, local_offset);
      Value value = LoadVector(builder, source, offset,
                               I64(builder, num_elements_), vector_type);
      StoreVector(builder,
                  ApplyEpilogue(builder, value, epilogue_inputs, offset,
                                I64(builder, num_elements_)),
                  output, offset, I64(builder, num_elements_));
    }
    builder.setInsertionPointAfter(gather);
  }

  void EmitFlyLdsReduce(mlir::ImplicitLocOpBuilder& builder, Value output,
                        Value output_base, Value scratch_table,
                        Value buffer_offset, int64_t work_elements) const {
    auto vector_type = mlir::VectorType::get(
        {vector_lanes_}, StorageType(builder));
    const int64_t work_packs = work_elements / vector_lanes_;
    RankGroupIndex index = GetRankGroupIndex(builder);
    Value peer_scratch =
        ScratchPointer(builder, scratch_table, index.group, buffer_offset);
    // FlyDSL alternates two 8 KiB LDS halves. The barrier following each load
    // also waits for wave zero to finish reading the opposite half from the
    // previous iteration, avoiding a second barrier per iteration.
    Value shared = SharedPointer(builder, /*buffers=*/2);
    mlir::scf::ForOp reduce = mlir::scf::ForOp::create(
        builder, index.first_pack, I64(builder, work_packs),
        index.pack_stride);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(reduce.getBody());
      Value pack = reduce.getInductionVar();
      Value iteration = mlir::arith::DivUIOp::create(
          builder, pack, index.pack_stride);
      Value parity = mlir::arith::AndIOp::create(
          builder, iteration, I64(builder, 1));
      Value shared_base = mlir::arith::MulIOp::create(
          builder, parity, I64(builder, kFlyCollectiveThreads));
      Value shared_slot = mlir::arith::AddIOp::create(
          builder, shared_base, index.thread);
      Value shared_offset = mlir::arith::MulIOp::create(
          builder, shared_slot, I64(builder, vector_lanes_));
      Value element_offset = mlir::arith::AddIOp::create(
          builder, output_base,
          mlir::arith::MulIOp::create(builder, pack,
                                      I64(builder, vector_lanes_)));
      Value raw = mlir::fly::PtrLoadOp::create(
          builder, vector_type,
          AddOffset(builder, peer_scratch, element_offset));
      mlir::fly::PtrStoreOp::create(
          builder, raw, AddOffset(builder, shared, shared_offset));
      mlir::gpu::BarrierOp::create(builder);

      Value is_reducer = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, index.group,
          I64(builder, 0));
      mlir::scf::IfOp write = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{}, is_reducer,
          /*withElseRegion=*/false);
      {
        mlir::OpBuilder::InsertionGuard write_guard(builder);
        builder.setInsertionPointToStart(write.thenBlock());
        Value lane_slot = mlir::arith::AddIOp::create(
            builder, shared_base, index.lane);
        Value lane_offset = mlir::arith::MulIOp::create(
            builder, lane_slot, I64(builder, vector_lanes_));
        Value accumulator = mlir::fly::PtrLoadOp::create(
            builder, vector_type,
            AddOffset(builder, shared, lane_offset));
        for (int64_t peer = 1; peer < world_size_; ++peer) {
          Value peer_slot = mlir::arith::AddIOp::create(
              builder, lane_slot,
              I64(builder, peer *
                               (kFlyCollectiveThreads / world_size_)));
          Value peer_offset = mlir::arith::MulIOp::create(
              builder, peer_slot, I64(builder, vector_lanes_));
          Value peer_value = mlir::fly::PtrLoadOp::create(
              builder, vector_type,
              AddOffset(builder, shared, peer_offset));
          accumulator = Combine(builder, accumulator, peer_value);
        }
        mlir::fly::PtrStoreOp::create(
            builder, accumulator,
            AddOffset(builder, output, element_offset));
      }
    }
    builder.setInsertionPointAfter(reduce);
  }

  void EmitFlySegmentCopy(mlir::ImplicitLocOpBuilder& builder,
                          llvm::ArrayRef<Value> producer_inputs, Value input,
                          Value local_scratch) const {
    auto vector_type = mlir::VectorType::get(
        {vector_lanes_}, StorageType(builder));
    const int64_t segment_packs = segment_elements_ / vector_lanes_;
    RankGroupIndex index = GetRankGroupIndex(builder);
    mlir::scf::ForOp copy = mlir::scf::ForOp::create(
        builder, index.first_pack, I64(builder, segment_packs),
        index.pack_stride);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(copy.getBody());
      Value segment_pack_base = mlir::arith::MulIOp::create(
          builder, index.group, I64(builder, segment_packs));
      Value pack = mlir::arith::AddIOp::create(
          builder, segment_pack_base, copy.getInductionVar());
      Value offset = mlir::arith::MulIOp::create(
          builder, pack, I64(builder, vector_lanes_));
      Value value = mlir::fly::PtrLoadOp::create(
          builder, vector_type, AddOffset(builder, input, offset));
      mlir::fly::PtrStoreOp::create(
          builder,
          ApplyProducer(builder, value, producer_inputs, offset,
                        I64(builder, num_elements_)),
          AddOffset(builder, local_scratch, offset));
    }
    builder.setInsertionPointAfter(copy);
  }

  void EmitFlyGather(mlir::ImplicitLocOpBuilder& builder,
                     llvm::ArrayRef<Value> epilogue_inputs, Value output,
                     Value rank, Value scratch_table,
                     Value buffer_offset) const {
    auto vector_type = mlir::VectorType::get(
        {vector_lanes_}, StorageType(builder));
    const int64_t segment_packs = segment_elements_ / vector_lanes_;
    RankGroupIndex index = GetRankGroupIndex(builder);
    Value source_rank = mlir::arith::AndIOp::create(
        builder,
        mlir::arith::AddIOp::create(builder, ToI64(builder, rank),
                                    index.group),
        I64(builder, world_size_ - 1));
    Value source =
        ScratchPointer(builder, scratch_table, source_rank, buffer_offset);
    Value segment_pack_base = mlir::arith::MulIOp::create(
        builder, source_rank, I64(builder, segment_packs));
    mlir::scf::ForOp gather = mlir::scf::ForOp::create(
        builder, index.first_pack, I64(builder, segment_packs),
        index.pack_stride);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(gather.getBody());
      Value pack = mlir::arith::AddIOp::create(
          builder, segment_pack_base, gather.getInductionVar());
      Value offset = mlir::arith::MulIOp::create(
          builder, pack, I64(builder, vector_lanes_));
      Value value = mlir::fly::PtrLoadOp::create(
          builder, vector_type, AddOffset(builder, source, offset));
      mlir::fly::PtrStoreOp::create(
          builder,
          ApplyEpilogue(builder, value, epilogue_inputs, offset,
                        I64(builder, num_elements_)),
          AddOffset(builder, output, offset));
    }
    builder.setInsertionPointAfter(gather);
  }

  void EmitOneShot(mlir::ImplicitLocOpBuilder& builder, Value input,
                   llvm::ArrayRef<Value> producer_inputs,
                   llvm::ArrayRef<Value> epilogue_inputs, Value output,
                   Value rank, Value signal_value, Value signal_table,
                   Value scratch_table,
                   Value buffer_offset) const {
    auto vector_type = mlir::VectorType::get(
        {vector_lanes_}, StorageType(builder));
    Value limit = I64(builder, num_elements_);
    Value local_scratch =
        ScratchPointer(builder, scratch_table, rank, buffer_offset);

    llvm::SmallVector<Value> peer_scratch;
    peer_scratch.reserve(world_size_);
    for (int64_t peer = 0; peer < world_size_; ++peer) {
      peer_scratch.push_back(ScratchPointer(
          builder, scratch_table, I32(builder, peer), buffer_offset));
    }

    Value first = LinearOffset(builder);
    Value stride = LinearStride(builder);
    mlir::scf::ForOp copy =
        mlir::scf::ForOp::create(builder, first, limit, stride);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(copy.getBody());
      Value offset = copy.getInductionVar();
      StoreVector(builder,
                  ApplyProducer(
                      builder,
                      LoadVector(builder, input, offset, limit, vector_type),
                      producer_inputs, offset, limit),
                  local_scratch, offset, limit);
    }
    builder.setInsertionPointAfter(copy);
    EmitSync(builder, signal_table, rank, signal_value);

    mlir::scf::ForOp reduce =
        mlir::scf::ForOp::create(builder, first, limit, stride);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(reduce.getBody());
      Value offset = reduce.getInductionVar();
      Value accumulator = LoadVector(builder, peer_scratch[0], offset, limit,
                                     vector_type);
      for (int64_t peer = 1; peer < world_size_; ++peer) {
        accumulator = Combine(
            builder, accumulator,
            LoadVector(builder, peer_scratch[peer], offset, limit,
                       vector_type));
      }
      StoreVector(builder,
                  ApplyEpilogue(builder, accumulator, epilogue_inputs, offset,
                                limit),
                  output, offset, limit);
    }
    builder.setInsertionPointAfter(reduce);
  }

  // Each workgroup owns a disjoint contiguous stripe of the local input. It
  // publishes that stripe once, synchronizes with the workgroup having the
  // same block id on every peer, then copies the corresponding peer stripes
  // to their rank-ordered output chunks. Unlike an output-tiled formulation,
  // this does not replicate the local publish phase once per source rank.
  void EmitAllGather(mlir::ImplicitLocOpBuilder& builder, Value input,
                     Value output, Value rank, Value signal_value,
                     Value signal_table, Value scratch_table,
                     Value buffer_offset) const {
    auto vector_type = mlir::VectorType::get(
        {vector_lanes_}, StorageType(builder));
    Value input_limit = I64(builder, num_elements_);
    Value output_limit = I64(builder, output_elements_);
    Value local_scratch =
        ScratchPointer(builder, scratch_table, rank, buffer_offset);
    Value first = LinearOffset(builder);
    Value stride = LinearStride(builder);

    mlir::scf::ForOp publish =
        mlir::scf::ForOp::create(builder, first, input_limit, stride);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(publish.getBody());
      Value offset = publish.getInductionVar();
      StoreVector(builder,
                  LoadVector(builder, input, offset, input_limit, vector_type),
                  local_scratch, offset, input_limit);
    }
    builder.setInsertionPointAfter(publish);
    EmitSync(builder, signal_table, rank, signal_value);

    for (int64_t peer = 0; peer < world_size_; ++peer) {
      Value peer_scratch = ScratchPointer(
          builder, scratch_table, I32(builder, peer), buffer_offset);
      Value output_base = I64(builder, peer * num_elements_);
      mlir::scf::ForOp gather =
          mlir::scf::ForOp::create(builder, first, input_limit, stride);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(gather.getBody());
        Value input_offset = gather.getInductionVar();
        Value output_offset = mlir::arith::AddIOp::create(
            builder, output_base, input_offset);
        StoreVector(
            builder,
            LoadVector(builder, peer_scratch, input_offset, input_limit,
                       vector_type),
            output, output_offset, output_limit);
      }
      builder.setInsertionPointAfter(gather);
    }
  }

  void EmitTwoShot(mlir::ImplicitLocOpBuilder& builder, Value input,
                   llvm::ArrayRef<Value> producer_inputs,
                   llvm::ArrayRef<Value> epilogue_inputs, Value output,
                   Value rank, Value signal_value, Value signal_table,
                   Value scratch_table,
                   Value buffer_offset) const {
    auto vector_type = mlir::VectorType::get(
        {vector_lanes_}, StorageType(builder));
    Value full_limit = I64(builder, num_elements_);
    Value segment_limit = I64(builder, segment_elements_);
    Value local_scratch =
        ScratchPointer(builder, scratch_table, rank, buffer_offset);

    if (use_lds_algorithm_) {
      EmitFlySegmentCopy(builder, producer_inputs, input, local_scratch);
      EmitSync(builder, signal_table, rank, signal_value);
      Value rank_base = mlir::arith::MulIOp::create(
          builder, ToI64(builder, rank), I64(builder, segment_elements_));
      EmitFlyLdsReduce(builder, local_scratch, rank_base, scratch_table,
                       buffer_offset, segment_elements_);
      Value next_signal = mlir::arith::AddIOp::create(
          builder, signal_value, I32(builder, 1));
      EmitSync(builder, signal_table, rank, next_signal);
      EmitFlyGather(builder, epilogue_inputs, output, rank, scratch_table,
                    buffer_offset);
      return;
    }

    llvm::SmallVector<Value> peer_scratch;
    peer_scratch.reserve(world_size_);
    for (int64_t peer = 0; peer < world_size_; ++peer) {
      peer_scratch.push_back(ScratchPointer(
          builder, scratch_table, I32(builder, peer), buffer_offset));
    }

    Value first = LinearOffset(builder);
    Value stride = LinearStride(builder);
    const bool use_wide_io =
        full_vectors_only_ && io_vector_lanes_ > vector_lanes_ &&
        num_elements_ % launch_dimensions_.num_blocks() == 0 &&
        (num_elements_ / launch_dimensions_.num_blocks()) % world_size_ == 0;
    if (use_wide_io) {
      EmitWideCopy(builder, producer_inputs, input, local_scratch);
    } else {
      mlir::scf::ForOp copy =
          mlir::scf::ForOp::create(builder, first, segment_limit, stride);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(copy.getBody());
        Value local_offset = copy.getInductionVar();
        for (int64_t segment = 0; segment < world_size_; ++segment) {
          Value offset = mlir::arith::AddIOp::create(
              builder, local_offset,
              I64(builder, segment * segment_elements_));
          Value segment_end = I64(
              builder,
              std::min(num_elements_, (segment + 1) * segment_elements_));
          StoreVector(builder,
                      ApplyProducer(
                          builder,
                          LoadVector(builder, input, offset, segment_end,
                                     vector_type),
                          producer_inputs, offset, segment_end),
                      local_scratch, offset, segment_end);
        }
      }
      builder.setInsertionPointAfter(copy);
    }
    EmitSync(builder, signal_table, rank, signal_value);

    Value rank_i64 = ToI64(builder, rank);
    Value reduce_first = first;
    Value reduce_limit = segment_limit;
    Value reduce_stride = stride;
    Value rank_base;
    Value rank_end;
    if (use_wide_io) {
      const int64_t tile_elements =
          num_elements_ / launch_dimensions_.num_blocks();
      const int64_t subtile_elements = tile_elements / world_size_;
      Value thread = mlir::arith::IndexCastOp::create(
          builder, builder.getI64Type(),
          mlir::gpu::ThreadIdOp::create(builder, mlir::gpu::Dimension::x));
      Value block = mlir::arith::IndexCastOp::create(
          builder, builder.getI64Type(),
          mlir::gpu::BlockIdOp::create(builder, mlir::gpu::Dimension::x));
      rank_base = mlir::arith::AddIOp::create(
          builder,
          mlir::arith::MulIOp::create(builder, block,
                                      I64(builder, tile_elements)),
          mlir::arith::MulIOp::create(builder, rank_i64,
                                      I64(builder, subtile_elements)));
      rank_end = mlir::arith::AddIOp::create(
          builder, rank_base, I64(builder, subtile_elements));
      reduce_first = mlir::arith::MulIOp::create(
          builder, thread, I64(builder, vector_lanes_));
      reduce_limit = I64(builder, subtile_elements);
      reduce_stride = I64(
          builder, launch_dimensions_.num_threads_per_block() * vector_lanes_);
    } else {
      rank_base = mlir::arith::MulIOp::create(
          builder, rank_i64, I64(builder, segment_elements_));
      rank_end = mlir::arith::MinUIOp::create(
          builder,
          mlir::arith::AddIOp::create(builder, rank_base, segment_limit),
          full_limit);
    }
    mlir::scf::ForOp reduce =
        mlir::scf::ForOp::create(builder, reduce_first, reduce_limit,
                                 reduce_stride);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(reduce.getBody());
      Value offset = mlir::arith::AddIOp::create(
          builder, rank_base, reduce.getInductionVar());
      Value accumulator = LoadVector(builder, peer_scratch[0], offset,
                                     rank_end, vector_type);
      for (int64_t peer = 1; peer < world_size_; ++peer) {
        accumulator = Combine(
            builder, accumulator,
            LoadVector(builder, peer_scratch[peer], offset, rank_end,
                       vector_type));
      }
      StoreVector(builder, accumulator, local_scratch, offset, rank_end);
    }
    builder.setInsertionPointAfter(reduce);

    Value next_signal = mlir::arith::AddIOp::create(
        builder, signal_value, I32(builder, 1));
    EmitSync(builder, signal_table, rank, next_signal);

    if (use_wide_io) {
      EmitWideGather(builder, epilogue_inputs, output, scratch_table,
                     buffer_offset);
    } else {
      mlir::scf::ForOp gather =
          mlir::scf::ForOp::create(builder, first, segment_limit, stride);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(gather.getBody());
        Value local_offset = gather.getInductionVar();
        for (int64_t peer = 0; peer < world_size_; ++peer) {
          Value offset = mlir::arith::AddIOp::create(
              builder, local_offset,
              I64(builder, peer * segment_elements_));
          Value segment_end = I64(
              builder,
              std::min(num_elements_, (peer + 1) * segment_elements_));
          Value value = LoadVector(builder, peer_scratch[peer], offset,
                                   segment_end, vector_type);
          StoreVector(builder,
                      ApplyEpilogue(builder, value, epilogue_inputs, offset,
                                    segment_end),
                      output, offset, segment_end);
        }
      }
      builder.setInsertionPointAfter(gather);
    }
  }

  Value LinearOffset(mlir::ImplicitLocOpBuilder& builder) const {
    Value thread = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(),
        mlir::gpu::ThreadIdOp::create(builder, mlir::gpu::Dimension::x));
    Value block = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(),
        mlir::gpu::BlockIdOp::create(builder, mlir::gpu::Dimension::x));
    Value linear_thread = mlir::arith::AddIOp::create(
        builder,
        mlir::arith::MulIOp::create(
            builder, block,
            I64(builder, launch_dimensions_.num_threads_per_block())),
        thread);
    return mlir::arith::MulIOp::create(
        builder, linear_thread, I64(builder, vector_lanes_));
  }

  Value LinearStride(mlir::ImplicitLocOpBuilder& builder) const {
    return I64(builder, launch_dimensions_.num_blocks() *
                            launch_dimensions_.num_threads_per_block() *
                            vector_lanes_);
  }

  absl::Status EmitKernel(mlir::gpu::GPUFuncOp kernel) const {
    TF_RET_CHECK(kernel.getNumArguments() ==
                 6 + producer_.buffer_count + epilogue_.buffer_count);
    mlir::ImplicitLocOpBuilder builder(kernel.getLoc(), kernel);
    builder.setInsertionPointToStart(&kernel.getBody().front());
    int64_t argument = 0;
    Value input = kernel.getArgument(argument++);
    Value output = kernel.getArgument(argument++);
    llvm::SmallVector<Value> producer_inputs;
    producer_inputs.reserve(producer_.buffer_count);
    for (int64_t producer_input = 0;
         producer_input < producer_.buffer_count; ++producer_input) {
      producer_inputs.push_back(kernel.getArgument(argument++));
    }
    llvm::SmallVector<Value> epilogue_inputs;
    epilogue_inputs.reserve(epilogue_.buffer_count);
    for (int64_t epilogue_input = 0;
         epilogue_input < epilogue_.buffer_count; ++epilogue_input) {
      epilogue_inputs.push_back(kernel.getArgument(argument++));
    }
    Value rank = kernel.getArgument(argument++);
    Value signal_value = kernel.getArgument(argument++);
    Value signal_table = kernel.getArgument(argument++);
    Value scratch_table = kernel.getArgument(argument++);

    if (full_vectors_only_) {
      const int64_t input_allocation_bytes =
          num_elements_ * primitive_util::ByteWidth(element_type_);
      const int64_t output_allocation_bytes =
          output_elements_ * primitive_util::ByteWidth(element_type_);
      input = BufferPointer(builder, input, input_allocation_bytes);
      for (Value& producer_input : producer_inputs) {
        producer_input =
            BufferPointer(builder, producer_input, input_allocation_bytes);
      }
      for (Value& epilogue_input : epilogue_inputs) {
        epilogue_input =
            BufferPointer(builder, epilogue_input, output_allocation_bytes);
      }
      output = BufferPointer(builder, output, output_allocation_bytes);
    }

    Value buffer_signal = signal_value;
    if (strategy_ == AllReduceStrategy::kTwoShot) {
      buffer_signal = mlir::arith::ShRUIOp::create(
          builder, signal_value, I32(builder, 1));
    }
    Value buffer_index = mlir::arith::AndIOp::create(
        builder, buffer_signal, I32(builder, 1));
    Value buffer_offset = mlir::arith::MulIOp::create(
        builder, ToI64(builder, buffer_index),
        I64(builder, scratch_elements_per_buffer_));

    if (collective_opcode_ == HloOpcode::kAllGather) {
      TF_RET_CHECK(producer_inputs.empty() && epilogue_inputs.empty());
      EmitAllGather(builder, input, output, rank, signal_value, signal_table,
                    scratch_table, buffer_offset);
      mlir::gpu::ReturnOp::create(builder);
      return absl::OkStatus();
    }

    switch (strategy_) {
      case AllReduceStrategy::kOneShot:
        EmitOneShot(builder, input, producer_inputs, epilogue_inputs, output,
                    rank, signal_value, signal_table, scratch_table,
                    buffer_offset);
        break;
      case AllReduceStrategy::kTwoShot:
        EmitTwoShot(builder, input, producer_inputs, epilogue_inputs, output,
                    rank, signal_value, signal_table, scratch_table,
                    buffer_offset);
        break;
      case AllReduceStrategy::kMultimem:
        return absl::UnimplementedError(
            "Fly collective emitter does not support multimem strategy.");
    }
    mlir::gpu::ReturnOp::create(builder);
    return absl::OkStatus();
  }

  PrimitiveType element_type_ = PRIMITIVE_TYPE_INVALID;
  HloOpcode collective_opcode_ = HloOpcode::kAllReduce;
  HloOpcode reduction_opcode_ = HloOpcode::kAdd;
  int64_t num_elements_ = 0;
  int64_t output_elements_ = 0;
  int64_t world_size_ = 0;
  int64_t vector_lanes_ = 0;
  int64_t io_vector_lanes_ = 0;
  int64_t segment_elements_ = 0;
  int64_t scratch_elements_per_buffer_ = 0;
  bool full_vectors_only_ = false;
  bool use_lds_algorithm_ = false;
  CollectiveEpilogue producer_;
  CollectiveEpilogue epilogue_;
  AllReduceStrategy strategy_ = AllReduceStrategy::kOneShot;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileCollectiveEmitter(
    const HloFusionAnalysis& analysis, CollectiveEpilogue producer,
    CollectiveEpilogue epilogue) {
  return std::make_unique<FlyXTileCollectiveEmitter>(
      analysis, std::move(producer), std::move(epilogue));
}

}  // namespace xla::gpu::flydsl
