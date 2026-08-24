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

#include "xla/backends/gpu/codegen/flydsl/xtile_elementwise.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using mlir::Value;

bool IsSupportedType(PrimitiveType type) {
  return type == F16 || type == BF16 || type == F32;
}

bool IsSupportedValueType(PrimitiveType type) {
  return IsSupportedType(type) || type == PRED;
}

int64_t ElementBits(PrimitiveType type) {
  return type == F32 ? 32 : 16;
}

bool HasSamePhysicalDimensions(const Shape& lhs, const Shape& rhs) {
  return ShapeUtil::EqualIgnoringElementType(lhs, rhs) &&
         LayoutUtil::Equal(lhs.layout(), rhs.layout());
}

bool IsSupportedElementwiseGraph(
    const HloInstruction* instruction, const Shape& output_shape,
    absl::flat_hash_set<const HloInstruction*>& visited) {
  if (!visited.insert(instruction).second) {
    return true;
  }
  switch (instruction->opcode()) {
    case HloOpcode::kParameter:
      // Using one copy atom for every kernel argument keeps this emitter small.
      // Mixed-precision temporaries are supported, but external buffers must
      // use the output element type.
      return instruction->shape().element_type() ==
                 output_shape.element_type() &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape);
    case HloOpcode::kConstant:
      return ShapeUtil::IsScalar(instruction->shape()) &&
             IsSupportedType(instruction->shape().element_type());
    case HloOpcode::kBroadcast:
      return instruction->operand_count() == 1 &&
             ShapeUtil::IsScalar(instruction->operand(0)->shape()) &&
             IsSupportedType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kConvert:
      return instruction->operand_count() == 1 &&
             IsSupportedType(instruction->shape().element_type()) &&
             IsSupportedType(instruction->operand(0)->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kAbs:
    case HloOpcode::kCopy:
    case HloOpcode::kExp:
    case HloOpcode::kLog:
    case HloOpcode::kNegate:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh:
      return IsSupportedType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 1 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
      return IsSupportedType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 2 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited);
    case HloOpcode::kClamp:
      return IsSupportedType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 3 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(2), output_shape,
                                         visited);
    case HloOpcode::kCompare:
      return instruction->shape().element_type() == PRED &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 2 &&
             Cast<const HloCompareInstruction>(instruction)->type() ==
                 Comparison::Type::kFloat &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited);
    case HloOpcode::kSelect:
      return IsSupportedType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 3 &&
             instruction->operand(0)->shape().element_type() == PRED &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(2), output_shape,
                                         visited);
    default:
      return false;
  }
}

class FlyXTileElementwiseEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileElementwiseEmitter(const HloFusionAnalysis& analysis)
      : elements_(ShapeUtil::ElementsIn(analysis.first_result_shape())),
        element_type_(analysis.first_result_shape().element_type()),
        output_count_(analysis.fusion_root_count()) {
    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    CHECK_EQ(config.output_tiles_size(), 1);
    CHECK_GT(config.output_tiles(0).sizes_size(), 0);
    vector_size_bits_ = config.vector_size_bits() == 0
                            ? 64
                            : config.vector_size_bits();
    CHECK(vector_size_bits_ == 64 || vector_size_bits_ == 128);
    CHECK_EQ(vector_size_bits_ % ElementBits(element_type_), 0);
    vector_width_ = vector_size_bits_ / ElementBits(element_type_);
    vectors_per_thread_ = config.output_tiles(0).sizes(0);
    threads_ = config.num_warps() * 64;
    CHECK_GT(vectors_per_thread_, 0);
    CHECK_GT(threads_, 0);
    const int64_t elements_per_block =
        threads_ * vectors_per_thread_ * vector_width_;
    launch_dimensions_ = LaunchDimensions(
        se::BlockDim((elements_ + elements_per_block - 1) /
                         elements_per_block,
                     1, 1),
        se::ThreadDim(threads_, 1, 1));
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
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::ub::UBDialect>();

    mlir::OpBuilder module_builder(&context);
    mlir::Location location =
        mlir::NameLoc::get(module_builder.getStringAttr(fusion.name()));
    mlir::OwningOpRef<mlir::ModuleOp> module =
        llvm_ir::CreateMlirModuleOp(location);
    module.get()->setAttr(
        mlir::gpu::GPUDialect::getContainerModuleAttrName(),
        module_builder.getUnitAttr());

    module_builder.setInsertionPointToStart(module->getBody());
    mlir::gpu::GPUModuleOp gpu_module = mlir::gpu::GPUModuleOp::create(
        module_builder, location, "fly_elementwise_kernels");
    module_builder.setInsertionPointToStart(
        &gpu_module.getBodyRegion().front());

    mlir::Type element_mlir_type =
        emitters::PrimitiveTypeToMlirType(element_type_, module_builder);
    mlir::fly::AddressSpaceAttr global_address =
        mlir::fly::AddressSpaceAttr::get(&context,
                                         mlir::fly::AddressSpace::Global);
    mlir::fly::PointerType pointer_type =
        mlir::fly::PointerType::get(element_mlir_type, global_address);
    llvm::SmallVector<mlir::Type> argument_types(
        fusion.operand_count() + output_count_, pointer_type);
    mlir::FunctionType function_type = mlir::FunctionType::get(
        &context, argument_types, /*results=*/mlir::TypeRange{});
    mlir::gpu::GPUFuncOp kernel = mlir::gpu::GPUFuncOp::create(
        module_builder, location, entry_function_name, function_type);
    kernel.setKernelAttr(module_builder.getUnitAttr());
    kernel.addEntryBlock();

    RETURN_IF_ERROR(EmitKernel(kernel, fusion));
    return module;
  }

  absl::Status EmitEntryFunction(const emitters::PartitionedComputations&,
                                 const emitters::CallTargetProvider&,
                                 mlir::func::FuncOp,
                                 const HloFusionInstruction&) const override {
    return absl::UnimplementedError(
        "FlyXTileElementwiseEmitter builds a native gpu.func module.");
  }

  Value SplatInteger(mlir::ImplicitLocOpBuilder& builder,
                     mlir::VectorType type, int64_t value) const {
    return mlir::arith::ConstantOp::create(
        builder, type,
        mlir::DenseElementsAttr::get(
            type, builder.getIntegerAttr(type.getElementType(), value)));
  }

  Value Bf16PairToF32(mlir::ImplicitLocOpBuilder& builder,
                      Value pair) const {
    auto i16_type = mlir::VectorType::get({2}, builder.getI16Type());
    auto i32_type = mlir::VectorType::get({2}, builder.getI32Type());
    auto f32_type = mlir::VectorType::get({2}, builder.getF32Type());
    Value bits = mlir::arith::BitcastOp::create(builder, i16_type, pair);
    bits = mlir::arith::ExtUIOp::create(builder, i32_type, bits);
    bits = mlir::arith::ShLIOp::create(
        builder, bits, SplatInteger(builder, i32_type, 16));
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
        mlir::arith::AddIOp::create(
            builder, bits, SplatInteger(builder, i32_type, 0x7fff)),
        retained_lsb);
    rounded = mlir::arith::ShRUIOp::create(builder, rounded, shift);
    Value narrowed =
        mlir::arith::TruncIOp::create(builder, i16_type, rounded);
    Value result =
        mlir::arith::BitcastOp::create(builder, bf16_type, narrowed);
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
        auto input_type =
            mlir::cast<mlir::VectorType>(pairs[pair].getType());
        const int64_t input_elements = input_type.getNumElements();
        auto result_type = mlir::VectorType::get(
            {2 * input_elements}, input_type.getElementType());
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

  absl::StatusOr<Value> EmitConvert(mlir::ImplicitLocOpBuilder& builder,
                                    PrimitiveType source_type,
                                    PrimitiveType destination_type,
                                    Value operand) const {
    if (source_type == destination_type) {
      return operand;
    }
    auto operand_vector_type = mlir::cast<mlir::VectorType>(operand.getType());
    const int64_t vector_width = operand_vector_type.getNumElements();
    auto destination_vector_type = mlir::VectorType::get(
        {vector_width},
        emitters::PrimitiveTypeToMlirType(destination_type, builder));
    if (source_type == BF16 && destination_type == F32 &&
        vector_width % 2 == 0) {
      auto bf16_pair_type =
          mlir::VectorType::get({2}, builder.getBF16Type());
      llvm::SmallVector<Value> pairs;
      pairs.reserve(vector_width / 2);
      for (int64_t pair = 0; pair < vector_width / 2; ++pair) {
        llvm::SmallVector<int64_t, 2> mask{2 * pair, 2 * pair + 1};
        Value source_pair = mlir::vector::ShuffleOp::create(
            builder, bf16_pair_type, operand, operand, mask);
        pairs.push_back(Bf16PairToF32(builder, source_pair));
      }
      return AssemblePairs(builder, std::move(pairs));
    }
    if (source_type == F32 && destination_type == BF16 &&
        vector_width % 2 == 0) {
      auto f32_pair_type =
          mlir::VectorType::get({2}, builder.getF32Type());
      llvm::SmallVector<Value> pairs;
      pairs.reserve(vector_width / 2);
      for (int64_t pair = 0; pair < vector_width / 2; ++pair) {
        llvm::SmallVector<int64_t, 2> mask{2 * pair, 2 * pair + 1};
        Value source_pair = mlir::vector::ShuffleOp::create(
            builder, f32_pair_type, operand, operand, mask);
        pairs.push_back(RoundF32PairToBf16(builder, source_pair));
      }
      return AssemblePairs(builder, std::move(pairs));
    }
    if (ElementBits(source_type) < ElementBits(destination_type)) {
      return mlir::arith::ExtFOp::create(builder, destination_vector_type,
                                         operand)
          .getResult();
    }
    return mlir::arith::TruncFOp::create(builder, destination_vector_type,
                                         operand)
        .getResult();
  }

  absl::StatusOr<Value> EmitPairwiseBf16Binary(
      mlir::ImplicitLocOpBuilder& builder, HloOpcode opcode, Value lhs,
      Value rhs) const {
    const int64_t vector_width =
        mlir::cast<mlir::VectorType>(lhs.getType()).getNumElements();
    if (vector_width % 2 != 0) {
      auto f32_type =
          mlir::VectorType::get({vector_width}, builder.getF32Type());
      auto bf16_type =
          mlir::VectorType::get({vector_width}, builder.getBF16Type());
      lhs = mlir::arith::ExtFOp::create(builder, f32_type, lhs);
      rhs = mlir::arith::ExtFOp::create(builder, f32_type, rhs);
      Value computed;
      switch (opcode) {
        case HloOpcode::kAdd:
          computed = mlir::arith::AddFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kSubtract:
          computed = mlir::arith::SubFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kMultiply:
          computed = mlir::arith::MulFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kDivide:
          computed = mlir::arith::DivFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kMaximum:
          computed = mlir::arith::MaximumFOp::create(builder, lhs, rhs);
          break;
        case HloOpcode::kMinimum:
          computed = mlir::arith::MinimumFOp::create(builder, lhs, rhs);
          break;
        default:
          return absl::InternalError("Unexpected pairwise BF16 opcode.");
      }
      return mlir::arith::TruncFOp::create(builder, bf16_type, computed)
          .getResult();
    }
    auto bf16_pair_type =
        mlir::VectorType::get({2}, builder.getBF16Type());
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
      Value computed;
      switch (opcode) {
        case HloOpcode::kAdd:
          computed = mlir::arith::AddFOp::create(builder, lhs_pair, rhs_pair);
          break;
        case HloOpcode::kSubtract:
          computed = mlir::arith::SubFOp::create(builder, lhs_pair, rhs_pair);
          break;
        case HloOpcode::kMultiply:
          computed = mlir::arith::MulFOp::create(builder, lhs_pair, rhs_pair);
          break;
        case HloOpcode::kDivide:
          computed = mlir::arith::DivFOp::create(builder, lhs_pair, rhs_pair);
          break;
        case HloOpcode::kMaximum:
          computed =
              mlir::arith::MaximumFOp::create(builder, lhs_pair, rhs_pair);
          break;
        case HloOpcode::kMinimum:
          computed =
              mlir::arith::MinimumFOp::create(builder, lhs_pair, rhs_pair);
          break;
        default:
          return absl::InternalError("Unexpected pairwise BF16 opcode.");
      }
      pairs.push_back(RoundF32PairToBf16(builder, computed));
    }
    return AssemblePairs(builder, std::move(pairs));
  }

  absl::StatusOr<mlir::arith::CmpFPredicate> GetComparePredicate(
      ComparisonDirection direction) const {
    switch (direction) {
      case ComparisonDirection::kEq:
        return mlir::arith::CmpFPredicate::OEQ;
      case ComparisonDirection::kNe:
        return mlir::arith::CmpFPredicate::UNE;
      case ComparisonDirection::kGe:
        return mlir::arith::CmpFPredicate::OGE;
      case ComparisonDirection::kGt:
        return mlir::arith::CmpFPredicate::OGT;
      case ComparisonDirection::kLe:
        return mlir::arith::CmpFPredicate::OLE;
      case ComparisonDirection::kLt:
        return mlir::arith::CmpFPredicate::OLT;
    }
    return absl::InvalidArgumentError(
        "Unsupported floating-point comparison direction.");
  }

  absl::StatusOr<Value> EmitScalarizedUnaryMath(
      mlir::ImplicitLocOpBuilder& builder, HloOpcode opcode,
      Value operand) const {
    auto operand_type = mlir::dyn_cast<mlir::VectorType>(operand.getType());
    TF_RET_CHECK(operand_type && operand_type.getRank() == 1 &&
                 operand_type.getElementType().isF32());
    const int64_t vector_width = operand_type.getNumElements();

    // MathToLLVM's AMDGPU patterns lower scalar transcendental operations to
    // the appropriate LLVM intrinsic or OCML call. They intentionally do not
    // legalize vector forms such as math.exp vector<4xf32>, so expose each lane
    // here while preserving vectorized memory operations and arithmetic around
    // the transcendental.
    llvm::SmallVector<Value> results;
    results.reserve(vector_width);
    for (int64_t lane = 0; lane < vector_width; ++lane) {
      Value scalar =
          mlir::vector::ExtractOp::create(builder, operand, lane).getResult();
      switch (opcode) {
        case HloOpcode::kExp:
          scalar = mlir::math::ExpOp::create(builder, scalar);
          break;
        case HloOpcode::kLog:
          scalar = mlir::math::LogOp::create(builder, scalar);
          break;
        case HloOpcode::kRsqrt:
          scalar = mlir::math::RsqrtOp::create(builder, scalar);
          break;
        case HloOpcode::kSqrt:
          scalar = mlir::math::SqrtOp::create(builder, scalar);
          break;
        case HloOpcode::kTanh:
          scalar = mlir::math::TanhOp::create(builder, scalar);
          break;
        default:
          return absl::InternalError(
              "Unexpected Fly elementwise unary opcode.");
      }
      results.push_back(scalar);
    }
    return mlir::vector::FromElementsOp::create(builder, operand_type, results)
        .getResult();
  }

  absl::StatusOr<Value> EmitVector(
      mlir::ImplicitLocOpBuilder& builder,
      llvm::ArrayRef<Value> argument_pointers,
      const HloInstruction* instruction, Value element_offset, Value predicate,
      Value copy_atom, Value vector_layout,
      int64_t vector_width,
      absl::flat_hash_map<const HloInstruction*, Value>& cache) const {
    auto existing = cache.find(instruction);
    if (existing != cache.end()) {
      return existing->second;
    }

    const PrimitiveType instruction_type =
        instruction->shape().element_type();
    TF_RET_CHECK(IsSupportedValueType(instruction_type));
    auto vector_type = mlir::VectorType::get(
        {vector_width},
        emitters::PrimitiveTypeToMlirType(instruction_type, builder));
    Value result;
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        Value pointer = argument_pointers[instruction->parameter_number()];
        auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
            builder.getContext(), /*width=*/32,
            /*divisibility=*/vector_width);
        auto offset_type = mlir::fly::IntTupleType::get(offset_attr);
        Value element_offset_i32 = mlir::arith::TruncIOp::create(
            builder, builder.getI32Type(), element_offset);
        Value offset_tuple = mlir::fly::MakeIntTupleOp::create(
            builder, offset_type, mlir::ValueRange{element_offset_i32});
        Value advanced = mlir::fly::AddOffsetOp::create(
            builder, pointer.getType(), pointer, offset_tuple);
        auto layout_type =
            mlir::cast<mlir::fly::LayoutType>(vector_layout.getType());
        auto memref_type = mlir::fly::MemRefType::get(
            vector_type.getElementType(),
            mlir::cast<mlir::fly::PointerType>(pointer.getType())
                .getAddressSpace(),
            layout_type.getAttr());
        Value view = mlir::fly::MakeViewOp::create(
            builder, memref_type, advanced, vector_layout);
        Value poison = mlir::ub::PoisonOp::create(builder, vector_type);
        mlir::fly::CopyAtomCallSSA load = mlir::fly::CopyAtomCallSSA::create(
            builder, mlir::TypeRange{vector_type}, copy_atom, view, poison,
            predicate);
        result = load.getResult(0);
        break;
      }
      case HloOpcode::kConstant: {
        std::optional<double> value = instruction->literal().GetAsDouble({});
        TF_RET_CHECK(value.has_value());
        mlir::Attribute scalar =
            builder.getFloatAttr(vector_type.getElementType(), *value);
        result = mlir::arith::ConstantOp::create(
            builder, vector_type,
            mlir::DenseElementsAttr::get(vector_type, scalar));
        break;
      }
      case HloOpcode::kBroadcast: {
        TF_ASSIGN_OR_RETURN(
            result,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        break;
      }
      case HloOpcode::kConvert: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            result,
            EmitConvert(builder,
                        instruction->operand(0)->shape().element_type(),
                        instruction_type, operand));
        break;
      }
      case HloOpcode::kCopy: {
        TF_ASSIGN_OR_RETURN(
            result,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        break;
      }
      case HloOpcode::kAbs:
      case HloOpcode::kNegate: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        if (instruction->opcode() == HloOpcode::kAbs) {
          result = mlir::math::AbsFOp::create(builder, operand);
        } else if (instruction_type != F32) {
          auto compute_type =
              mlir::VectorType::get({vector_width}, builder.getF32Type());
          operand =
              mlir::arith::ExtFOp::create(builder, compute_type, operand);
          operand = mlir::arith::NegFOp::create(builder, operand);
          result =
              mlir::arith::TruncFOp::create(builder, vector_type, operand);
        } else {
          result = mlir::arith::NegFOp::create(builder, operand);
        }
        break;
      }
      case HloOpcode::kExp:
      case HloOpcode::kLog:
      case HloOpcode::kRsqrt:
      case HloOpcode::kSqrt:
      case HloOpcode::kTanh: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        if (instruction_type != F32) {
          TF_ASSIGN_OR_RETURN(
              operand,
              EmitConvert(builder, instruction_type, F32, operand));
        }
        TF_ASSIGN_OR_RETURN(
            result, EmitScalarizedUnaryMath(builder, instruction->opcode(),
                                            operand));
        if (instruction_type != F32) {
          TF_ASSIGN_OR_RETURN(
              result,
              EmitConvert(builder, F32, instruction_type, result));
        }
        break;
      }
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum: {
        TF_ASSIGN_OR_RETURN(
            Value lhs,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value rhs,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        if (instruction_type == BF16) {
          TF_ASSIGN_OR_RETURN(
              result,
              EmitPairwiseBf16Binary(builder, instruction->opcode(), lhs,
                                     rhs));
          break;
        }
        if (instruction_type != F32) {
          auto compute_type =
              mlir::VectorType::get({vector_width}, builder.getF32Type());
          lhs = mlir::arith::ExtFOp::create(builder, compute_type, lhs);
          rhs = mlir::arith::ExtFOp::create(builder, compute_type, rhs);
        }
        Value computed;
        switch (instruction->opcode()) {
          case HloOpcode::kAdd:
            computed = mlir::arith::AddFOp::create(builder, lhs, rhs);
            break;
          case HloOpcode::kSubtract:
            computed = mlir::arith::SubFOp::create(builder, lhs, rhs);
            break;
          case HloOpcode::kMultiply:
            computed = mlir::arith::MulFOp::create(builder, lhs, rhs);
            break;
          case HloOpcode::kDivide:
            computed = mlir::arith::DivFOp::create(builder, lhs, rhs);
            break;
          case HloOpcode::kMaximum:
            computed = mlir::arith::MaximumFOp::create(builder, lhs, rhs);
            break;
          case HloOpcode::kMinimum:
            computed = mlir::arith::MinimumFOp::create(builder, lhs, rhs);
            break;
          default:
            return absl::InternalError("Unexpected Fly elementwise opcode.");
        }
        result = instruction_type == F32
                     ? computed
                     : mlir::arith::TruncFOp::create(builder, vector_type,
                                                     computed)
                           .getResult();
        break;
      }
      case HloOpcode::kClamp: {
        TF_ASSIGN_OR_RETURN(
            Value lower,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value upper,
            EmitVector(builder, argument_pointers, instruction->operand(2),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        if (instruction_type == BF16) {
          TF_ASSIGN_OR_RETURN(
              operand, EmitPairwiseBf16Binary(
                           builder, HloOpcode::kMaximum, lower, operand));
          TF_ASSIGN_OR_RETURN(
              result, EmitPairwiseBf16Binary(
                          builder, HloOpcode::kMinimum, operand, upper));
        } else {
          if (instruction_type != F32) {
            TF_ASSIGN_OR_RETURN(
                lower, EmitConvert(builder, instruction_type, F32, lower));
            TF_ASSIGN_OR_RETURN(
                operand, EmitConvert(builder, instruction_type, F32, operand));
            TF_ASSIGN_OR_RETURN(
                upper, EmitConvert(builder, instruction_type, F32, upper));
          }
          result = mlir::arith::MinimumFOp::create(
              builder,
              mlir::arith::MaximumFOp::create(builder, lower, operand), upper);
          if (instruction_type != F32) {
            TF_ASSIGN_OR_RETURN(
                result,
                EmitConvert(builder, F32, instruction_type, result));
          }
        }
        break;
      }
      case HloOpcode::kCompare: {
        TF_ASSIGN_OR_RETURN(
            Value lhs,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value rhs,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        const PrimitiveType operand_type =
            instruction->operand(0)->shape().element_type();
        if (operand_type != F32) {
          TF_ASSIGN_OR_RETURN(lhs,
                              EmitConvert(builder, operand_type, F32, lhs));
          TF_ASSIGN_OR_RETURN(rhs,
                              EmitConvert(builder, operand_type, F32, rhs));
        }
        TF_ASSIGN_OR_RETURN(
            mlir::arith::CmpFPredicate compare_predicate,
            GetComparePredicate(instruction->comparison_direction()));
        result = mlir::arith::CmpFOp::create(builder, compare_predicate, lhs,
                                             rhs);
        break;
      }
      case HloOpcode::kSelect: {
        TF_ASSIGN_OR_RETURN(
            Value condition,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value on_true,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value on_false,
            EmitVector(builder, argument_pointers, instruction->operand(2),
                       element_offset, predicate, copy_atom, vector_layout,
                       vector_width, cache));
        result = mlir::arith::SelectOp::create(builder, condition, on_true,
                                               on_false);
        break;
      }
      default:
        return absl::InvalidArgumentError(
            "Unsupported opcode in native Fly elementwise fusion.");
    }
    cache[instruction] = result;
    return result;
  }

  absl::Status EmitKernel(mlir::gpu::GPUFuncOp kernel,
                          const HloFusionInstruction& fusion) const {
    TF_RET_CHECK(kernel.getNumArguments() ==
                 fusion.operand_count() + output_count_);
    const HloInstruction* fusion_root = fusion.fused_expression_root();
    TF_RET_CHECK(fusion_root != nullptr);
    llvm::SmallVector<const HloInstruction*> roots;
    if (fusion_root->opcode() == HloOpcode::kTuple) {
      roots.append(fusion_root->operands().begin(),
                   fusion_root->operands().end());
    } else {
      roots.push_back(fusion_root);
    }
    TF_RET_CHECK(roots.size() == output_count_);

    mlir::ImplicitLocOpBuilder builder(kernel.getLoc(), kernel);
    builder.setInsertionPointToStart(&kernel.getBody().front());
    Value thread_id = mlir::gpu::ThreadIdOp::create(
        builder, mlir::gpu::Dimension::x);
    Value block_id =
        mlir::gpu::BlockIdOp::create(builder, mlir::gpu::Dimension::x);
    Value thread_i64 = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(), thread_id);
    Value block_i64 = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(), block_id);
    Value vector_width = mlir::arith::ConstantIntOp::create(
        builder, builder.getI64Type(), vector_width_);
    Value elements_per_block = mlir::arith::ConstantIntOp::create(
        builder, builder.getI64Type(),
        threads_ * vectors_per_thread_ * vector_width_);
    Value thread_stride = mlir::arith::ConstantIntOp::create(
        builder, builder.getI64Type(), threads_ * vector_width_);
    Value block_base =
        mlir::arith::MulIOp::create(builder, block_i64, elements_per_block);
    Value thread_base =
        mlir::arith::MulIOp::create(builder, thread_i64, vector_width);
    Value base = mlir::arith::AddIOp::create(builder, block_base, thread_base);
    const int64_t tail_elements = elements_ % vector_width_;
    const int64_t full_elements = elements_ - tail_elements;
    Value element_count = mlir::arith::ConstantIntOp::create(
        builder, builder.getI64Type(), full_elements);

    mlir::MLIRContext* context = builder.getContext();
    auto shape_attr =
        mlir::fly::IntTupleAttr::getLeafStatic(context, vector_width_);
    auto stride_attr = mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
    auto shape_type = mlir::fly::IntTupleType::get(shape_attr);
    auto stride_type = mlir::fly::IntTupleType::get(stride_attr);
    Value shape = mlir::fly::MakeIntTupleOp::create(
        builder, shape_type, mlir::ValueRange{});
    Value stride = mlir::fly::MakeIntTupleOp::create(
        builder, stride_type, mlir::ValueRange{});
    auto layout_type = mlir::fly::LayoutType::get(shape_attr, stride_attr);
    Value vector_layout = mlir::fly::MakeLayoutOp::create(
        builder, layout_type, shape, stride);
    auto copy_atom_type = mlir::fly::CopyAtomType::get(
        mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context,
                                                        vector_size_bits_,
                                                        /*cacheModifier=*/0),
        ElementBits(element_type_));
    Value copy_atom = mlir::fly::MakeCopyAtomOp::create(
        builder, copy_atom_type, ElementBits(element_type_));

    auto buffer_address =
        mlir::fly_rocdl::BufferDescAddressAttr::get(context);
    auto buffer_pointer_type = mlir::fly::PointerType::get(
        emitters::PrimitiveTypeToMlirType(element_type_, builder),
        buffer_address);
    Value descriptor_stride = mlir::arith::ConstantIntOp::create(
        builder, builder.getI16Type(), 0);
    Value descriptor_extent = mlir::arith::ConstantIntOp::create(
        builder, builder.getI64Type(),
        elements_ * (ElementBits(element_type_) / 8));
    Value descriptor_flags = mlir::arith::ConstantIntOp::create(
        builder, builder.getI32Type(), 0x27000);
    llvm::SmallVector<Value> argument_pointers;
    argument_pointers.reserve(kernel.getNumArguments());
    for (Value pointer : kernel.getArguments()) {
      argument_pointers.push_back(mlir::fly::MakePtrOp::create(
          builder, buffer_pointer_type,
          mlir::ValueRange{pointer, descriptor_stride, descriptor_extent,
                           descriptor_flags},
          /*dictAttrs=*/nullptr));
    }

    auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
        context, /*width=*/32, /*divisibility=*/vector_width_);
    auto offset_type = mlir::fly::IntTupleType::get(offset_attr);

    for (int64_t vector = 0;
         full_elements != 0 && vector < vectors_per_thread_; ++vector) {
      Value vector_offset = base;
      if (vector != 0) {
        Value vector_index = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), vector);
        Value relative = mlir::arith::MulIOp::create(
            builder, vector_index, thread_stride);
        vector_offset =
            mlir::arith::AddIOp::create(builder, base, relative);
      }
      Value in_bounds = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, vector_offset,
          element_count);
      absl::flat_hash_map<const HloInstruction*, Value> cache;
      llvm::SmallVector<Value> results;
      results.reserve(output_count_);
      for (const HloInstruction* root : roots) {
        TF_ASSIGN_OR_RETURN(
            Value result,
            EmitVector(builder, argument_pointers, root, vector_offset,
                       in_bounds, copy_atom, vector_layout, vector_width_,
                       cache));
        results.push_back(result);
      }

      Value vector_offset_i32 = mlir::arith::TruncIOp::create(
          builder, builder.getI32Type(), vector_offset);
      Value offset_tuple = mlir::fly::MakeIntTupleOp::create(
          builder, offset_type, mlir::ValueRange{vector_offset_i32});
      for (auto [output_index, result] : llvm::enumerate(results)) {
        Value output_pointer =
            argument_pointers[fusion.operand_count() + output_index];
        auto output_pointer_type =
            mlir::cast<mlir::fly::PointerType>(output_pointer.getType());
        auto output_memref_type = mlir::fly::MemRefType::get(
            output_pointer_type.getElemTy(),
            output_pointer_type.getAddressSpace(), layout_type.getAttr());
        Value advanced_output = mlir::fly::AddOffsetOp::create(
            builder, output_pointer_type, output_pointer, offset_tuple);
        Value output_view = mlir::fly::MakeViewOp::create(
            builder, output_memref_type, advanced_output, vector_layout);
        mlir::fly::CopyAtomCallSSA::create(
            builder, mlir::TypeRange{}, copy_atom, result, output_view,
            in_bounds);
      }
    }
    if (tail_elements != 0) {
      auto scalar_shape_attr =
          mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
      auto scalar_shape_type =
          mlir::fly::IntTupleType::get(scalar_shape_attr);
      Value scalar_shape = mlir::fly::MakeIntTupleOp::create(
          builder, scalar_shape_type, mlir::ValueRange{});
      Value scalar_stride = mlir::fly::MakeIntTupleOp::create(
          builder, scalar_shape_type, mlir::ValueRange{});
      auto scalar_layout_type = mlir::fly::LayoutType::get(
          scalar_shape_attr, scalar_shape_attr);
      Value scalar_layout = mlir::fly::MakeLayoutOp::create(
          builder, scalar_layout_type, scalar_shape, scalar_stride);
      const int64_t element_bits = ElementBits(element_type_);
      auto scalar_copy_atom_type = mlir::fly::CopyAtomType::get(
          mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(
              context, element_bits, /*cacheModifier=*/0),
          element_bits);
      Value scalar_copy_atom = mlir::fly::MakeCopyAtomOp::create(
          builder, scalar_copy_atom_type, element_bits);
      Value block_zero = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, block_i64,
          mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                             0));
      Value tail_lane = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, thread_i64,
          mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), tail_elements));
      Value tail_valid =
          mlir::arith::AndIOp::create(builder, block_zero, tail_lane);
      mlir::scf::IfOp tail = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{}, tail_valid,
          /*withElseRegion=*/false);
      {
        mlir::OpBuilder::InsertionGuard tail_guard(builder);
        builder.setInsertionPointToStart(tail.thenBlock());
        Value tail_offset = mlir::arith::AddIOp::create(
            builder,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               full_elements),
            thread_i64);
        Value true_predicate = mlir::arith::ConstantIntOp::create(
            builder, builder.getI1Type(), 1);
        absl::flat_hash_map<const HloInstruction*, Value> cache;
        llvm::SmallVector<Value> results;
        results.reserve(output_count_);
        for (const HloInstruction* root : roots) {
          TF_ASSIGN_OR_RETURN(
              Value result,
              EmitVector(builder, argument_pointers, root, tail_offset,
                         true_predicate, scalar_copy_atom, scalar_layout,
                         /*vector_width=*/1, cache));
          results.push_back(result);
        }

        auto scalar_offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
            context, /*width=*/32, /*divisibility=*/1);
        auto scalar_offset_type =
            mlir::fly::IntTupleType::get(scalar_offset_attr);
        Value tail_offset_i32 = mlir::arith::TruncIOp::create(
            builder, builder.getI32Type(), tail_offset);
        Value tail_offset_tuple = mlir::fly::MakeIntTupleOp::create(
            builder, scalar_offset_type, mlir::ValueRange{tail_offset_i32});
        for (auto [output_index, result] : llvm::enumerate(results)) {
          Value output_pointer =
              argument_pointers[fusion.operand_count() + output_index];
          auto output_pointer_type =
              mlir::cast<mlir::fly::PointerType>(output_pointer.getType());
          auto output_memref_type = mlir::fly::MemRefType::get(
              output_pointer_type.getElemTy(),
              output_pointer_type.getAddressSpace(),
              scalar_layout_type.getAttr());
          Value advanced_output = mlir::fly::AddOffsetOp::create(
              builder, output_pointer_type, output_pointer, tail_offset_tuple);
          Value output_view = mlir::fly::MakeViewOp::create(
              builder, output_memref_type, advanced_output, scalar_layout);
          mlir::fly::CopyAtomCallSSA::create(
              builder, mlir::TypeRange{}, scalar_copy_atom, result,
              output_view, true_predicate);
        }
      }
      builder.setInsertionPointAfter(tail);
    }
    mlir::gpu::ReturnOp::create(builder);
    return absl::OkStatus();
  }

  int64_t elements_ = 0;
  PrimitiveType element_type_ = PRIMITIVE_TYPE_INVALID;
  int64_t output_count_ = 0;
  int64_t vector_size_bits_ = 0;
  int64_t vector_width_ = 0;
  int64_t vectors_per_thread_ = 1;
  int64_t threads_ = 0;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

bool IsFlyXTileElementwiseFusion(const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() == 0) {
    return false;
  }
  const Shape& output_shape = analysis.fusion_root(0).shape();
  if (!output_shape.IsArray() || !output_shape.has_layout() ||
      !IsSupportedType(output_shape.element_type())) {
    return false;
  }
  const int64_t elements = ShapeUtil::ElementsIn(output_shape);
  const int64_t element_bytes = ElementBits(output_shape.element_type()) / 8;
  if (elements == 0 ||
      elements > std::numeric_limits<uint32_t>::max() / element_bytes) {
    return false;
  }
  absl::flat_hash_set<const HloInstruction*> visited;
  for (int64_t root_index = 0; root_index < analysis.fusion_root_count();
       ++root_index) {
    const HloInstruction& root =
        analysis.fusion_root(root_index).instruction();
    if (root.shape().element_type() != output_shape.element_type() ||
        !HasSamePhysicalDimensions(root.shape(), output_shape) ||
        !IsSupportedElementwiseGraph(&root, output_shape, visited)) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileElementwiseEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileElementwiseEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
