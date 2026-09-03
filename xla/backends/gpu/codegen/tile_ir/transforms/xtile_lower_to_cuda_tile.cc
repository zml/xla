#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "xla/backends/gpu/codegen/tile_ir/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::gpu::tile_ir {

#define GEN_PASS_DEF_XTILELOWERTOCUDATILEPASS
#include "xla/backends/gpu/codegen/tile_ir/transforms/passes.h.inc"

namespace {

namespace ct = ::mlir::cuda_tile;
namespace ma = ::mlir::arith;

using ::mlir::failure;
using ::mlir::success;
using ::mlir::Value;

// The entry parameter's alignment, not the allocation's: the kernel gets
// base plus slice offset, and an overstated div_by faults in cuLaunchKernel.
constexpr int64_t kBufferAlignmentBytes = 16;

ct::TileType ScalarI32(mlir::MLIRContext* ctx) {
  return ct::TileType::get({}, mlir::IntegerType::get(ctx, 32));
}

mlir::DenseTypedElementsAttr Splat(mlir::Type tile_type,
                                   mlir::Attribute value) {
  auto shaped = mlir::cast<mlir::ShapedType>(tile_type);
  mlir::Type element_type = shaped.getElementType();
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(value);
      integer && integer.getType() != element_type) {
    value = mlir::IntegerAttr::get(
        element_type, integer.getValue().sextOrTrunc(
                          element_type.getIntOrFloatBitWidth()));
  } else if (auto flt = mlir::dyn_cast<mlir::FloatAttr>(value);
             flt && flt.getType() != element_type) {
    value = mlir::FloatAttr::get(element_type, flt.getValueAsDouble());
  }
  return mlir::cast<mlir::DenseTypedElementsAttr>(
      mlir::DenseElementsAttr::get(shaped, value));
}

absl::StatusOr<llvm::SmallVector<int64_t>> PhysicalStrides(
    mlir::MemRefType memref) {
  ABSL_ASSIGN_OR_RETURN(llvm::SmallVector<int64_t> minor_to_major,
                   ::xla::xtile::GetPermutationMinorToMajor(memref));
  llvm::SmallVector<int64_t> strides(memref.getRank(), 1);
  int64_t running = 1;
  for (int64_t dim : minor_to_major) {
    strides[dim] = running;
    running *= memref.getDimSize(dim);
  }
  return strides;
}

bool IsRhsScaleTranspose(mlir::stablehlo::TransposeOp op) {
  if (!op.getResult().hasOneUse()) return false;
  auto dot = mlir::dyn_cast<::xla::xtile::DotScaledOp>(
      *op.getResult().getUsers().begin());
  if (!dot || dot.getRhsScale() != op.getResult()) return false;

  auto dims = mlir::dyn_cast_or_null<mlir::stablehlo::DotDimensionNumbersAttr>(
      dot.getDotDimensionNumbersAttr());
  if (!dims || dims.getRhsContractingDimensions().empty() ||
      dims.getRhsContractingDimensions()[0] != 0) {
    return false;
  }

  llvm::ArrayRef<int64_t> permutation = op.getPermutation();
  int64_t rank = permutation.size();
  if (rank < 2) return false;
  for (int64_t i = 0; i < rank - 2; ++i) {
    if (permutation[i] != i) return false;
  }
  return permutation[rank - 2] == rank - 1 && permutation[rank - 1] == rank - 2;
}

class KernelConverter {
 public:
  KernelConverter(::xla::xtile::EntryFuncOp entry, ct::EntryOp target,
                  mlir::ImplicitLocOpBuilder& builder)
      : entry_(entry), target_(target), b_(builder) {}

  mlir::LogicalResult Run();

 private:
  mlir::Type ConvertType(mlir::Type type);
  mlir::LogicalResult ConvertRegion(mlir::Region& from, mlir::Region& to);
  mlir::LogicalResult ConvertOp(mlir::Operation* op);

  mlir::LogicalResult ConvertExtract(::xla::xtile::ExtractTileOp op);
  mlir::LogicalResult ConvertInsert(::xla::xtile::InsertTileOp op);
  mlir::LogicalResult ConvertDotScaled(::xla::xtile::DotScaledOp op);
  mlir::LogicalResult ConvertFor(mlir::scf::ForOp op);
  mlir::LogicalResult ConvertConstant(ma::ConstantOp op);
  mlir::LogicalResult ConvertReduce(mlir::stablehlo::ReduceOp op);

  absl::StatusOr<Value> GetPartitionView(Value buffer,
                                         llvm::ArrayRef<int64_t> tile_shape);

  llvm::SmallVector<Value> GetTileIndex(mlir::ValueRange offsets,
                                        llvm::ArrayRef<int64_t> tile_shape);

  Value MaybeUnpack(Value tile, mlir::Type elem_type, int64_t packed_dim);

  Value RestoreScaleType(Value scale);

  Value Transpose2D(Value tile);

  ::xla::xtile::EntryFuncOp entry_;
  ct::EntryOp target_;
  mlir::ImplicitLocOpBuilder& b_;
  mlir::IRMapping map_;
  Value token_;
  llvm::DenseMap<std::pair<void*, mlir::Attribute>, Value> partition_views_;
  llvm::DenseMap<void*, Value> tensor_views_;
};

mlir::Type KernelConverter::ConvertType(mlir::Type type) {
  if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    return ct::TileType::get(tensor.getShape(), tensor.getElementType());
  }
  if (mlir::isa<mlir::IndexType>(type)) {
    return ScalarI32(type.getContext());
  }
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return ct::TileType::get({}, integer);
  }
  if (auto flt = mlir::dyn_cast<mlir::FloatType>(type)) {
    return ct::TileType::get({}, flt);
  }
  return nullptr;
}

absl::StatusOr<Value> KernelConverter::GetPartitionView(
    Value buffer, llvm::ArrayRef<int64_t> tile_shape) {
  auto memref = mlir::cast<mlir::MemRefType>(buffer.getType());
  mlir::MLIRContext* ctx = memref.getContext();

  Value& tensor_view = tensor_views_[buffer.getAsOpaquePointer()];
  if (!tensor_view) {
    llvm::SmallVector<int64_t> shape(memref.getShape());
    llvm::SmallVector<int64_t> strides;
    if (shape.empty()) {
      shape.push_back(1);
      strides.push_back(1);
    } else {
      ABSL_ASSIGN_OR_RETURN(strides, PhysicalStrides(memref));
    }
    auto view_type = ct::TensorViewType::get(ctx, memref.getElementType(),
                                             shape, strides);
    mlir::OpBuilder::InsertionGuard guard(b_);
    b_.setInsertionPointToStart(&target_.getBody().front());
  // div_by is a buffer size here, not an address alignment: more than the
  // scalar's width reads past the allocation.
    int64_t num_elements = 1;
    for (int64_t d : memref.getShape()) num_elements *= d;
    const int64_t byte_size =
        (num_elements * memref.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    int64_t alignment = kBufferAlignmentBytes;
    while (alignment > 1 && alignment > byte_size) alignment /= 2;
    Value base = ct::AssumeOp::create(
        b_, map_.lookup(buffer),
        ct::DivByAttr::get(ctx, alignment, /*every=*/std::nullopt,
                           /*along=*/std::nullopt));
    tensor_view = ct::MakeTensorViewOp::create(b_, view_type, base,
                                               /*dynamicShape=*/mlir::ValueRange{},
                                               /*dynamicStrides=*/mlir::ValueRange{});
  }

  llvm::SmallVector<int32_t> tile_shape_i32(tile_shape.begin(),
                                            tile_shape.end());
  if (tile_shape_i32.empty()) tile_shape_i32.push_back(1);
  auto tile_shape_attr = mlir::DenseI32ArrayAttr::get(ctx, tile_shape_i32);
  Value& partition_view =
      partition_views_[{buffer.getAsOpaquePointer(), tile_shape_attr}];
  if (!partition_view) {
    llvm::SmallVector<int32_t> dim_map(tile_shape_i32.size());
    for (int32_t i = 0; i < dim_map.size(); ++i) dim_map[i] = i;
    auto partition_type = ct::PartitionViewType::get(
        ctx, tile_shape_attr,
        mlir::cast<ct::TensorViewType>(tensor_view.getType()), dim_map,
        ct::PaddingValueAttr::get(ctx, ct::PaddingValue::zero));
    mlir::OpBuilder::InsertionGuard guard(b_);
    b_.setInsertionPointAfterValue(tensor_view);
    partition_view =
        ct::MakePartitionViewOp::create(b_, partition_type, tensor_view);
  }
  return partition_view;
}

llvm::SmallVector<Value> KernelConverter::GetTileIndex(
    mlir::ValueRange offsets, llvm::ArrayRef<int64_t> tile_shape) {
  llvm::SmallVector<Value> index;
  index.reserve(offsets.size());
  for (auto [offset, extent] : llvm::zip(offsets, tile_shape)) {
    if (auto mul = offset.getDefiningOp<ma::MulIOp>()) {
      mlir::IntegerAttr factor;
      if (mlir::matchPattern(mul.getRhs(), mlir::m_Constant(&factor)) &&
          factor.getInt() == extent) {
        index.push_back(map_.lookup(mul.getLhs()));
        continue;
      }
    }
    if (extent == 1) {
      index.push_back(map_.lookup(offset));
      continue;
    }
    ct::TileType i32 = ScalarI32(b_.getContext());
    Value divisor = ct::ConstantOp::create(
        b_, i32, Splat(i32, b_.getI32IntegerAttr(extent)));
    index.push_back(ct::DivIOp::create(b_, map_.lookup(offset), divisor,
                                       ct::Signedness::Signed));
  }
  return index;
}

Value KernelConverter::MaybeUnpack(Value tile, mlir::Type elem_type,
                                   int64_t packed_dim) {
  auto tile_type = mlir::cast<ct::TileType>(tile.getType());
  if (tile_type.getElementType() == elem_type) return tile;

  const bool packed_is_major = packed_dim != tile_type.getRank() - 1;
  if (packed_is_major) {
    if (tile_type.getRank() != 2) return Value();
    tile = Transpose2D(tile);
    tile_type = mlir::cast<ct::TileType>(tile.getType());
  }

  int64_t packed_elements = tile_type.getNumElements();
  int64_t ratio = 8 / elem_type.getIntOrFloatBitWidth();
  auto flat_i8 = ct::TileType::get({packed_elements}, tile_type.getElementType());
  auto flat_unpacked = ct::TileType::get({packed_elements * ratio}, elem_type);

  llvm::SmallVector<int64_t> logical_shape(tile_type.getShape());
  logical_shape.back() *= ratio;
  auto logical = ct::TileType::get(logical_shape, elem_type);

  Value flat = ct::ReshapeOp::create(b_, flat_i8, tile);
  Value unpacked = ct::UnpackOp::create(b_, flat_unpacked, flat);
  Value result = ct::ReshapeOp::create(b_, logical, unpacked);
  return packed_is_major ? Transpose2D(result) : result;
}

mlir::LogicalResult KernelConverter::ConvertExtract(
    ::xla::xtile::ExtractTileOp op) {
  for (int64_t stride : op.getStrides()) {
    if (stride != 1 && stride != 0) {
      return op.emitOpError("non-unit tile strides are not supported by the "
                            "CUDA Tile IR backend");
    }
  }
  absl::StatusOr<Value> view =
      GetPartitionView(op.getSource(), op.getFullTileShape());
  if (!view.ok()) return op.emitOpError(view.status().message());

  mlir::Type result = ConvertType(op.getType());
  if (!result) return op.emitOpError("unsupported result type");

  auto result_tile = mlir::cast<ct::TileType>(result);
  const bool rank0 =
      mlir::cast<mlir::MemRefType>(op.getSource().getType()).getRank() == 0;
  llvm::SmallVector<Value> index =
      GetTileIndex(op.getOffsets(), op.getFullTileShape());
  if (rank0) {
    ct::TileType i32 = ScalarI32(b_.getContext());
    index.push_back(ct::ConstantOp::create(
        b_, i32, Splat(i32, b_.getI32IntegerAttr(0))));
  }
  mlir::Type load_type =
      rank0 ? ct::TileType::get({1}, result_tile.getElementType()) : result;

  auto load = ct::LoadViewTkoOp::create(
      b_, load_type, ct::TokenType::get(b_.getContext()),
      ct::MemoryOrderingSemanticsAttr::get(b_.getContext(),
                                           ct::MemoryOrderingSemantics::WEAK),
      /*memory_scope=*/ct::MemoryScopeAttr{}, *view, index, token_,
      /*optimization_hints=*/ct::OptimizationHintsAttr{});
  Value tile = load.getTile();
  if (rank0) tile = ct::ReshapeOp::create(b_, result, tile);
  map_.map(op.getResult(), tile);
  return success();
}

mlir::LogicalResult KernelConverter::ConvertInsert(
    ::xla::xtile::InsertTileOp op) {
  absl::StatusOr<Value> view =
      GetPartitionView(op.getDestination(), op.getFullTileShape());
  if (!view.ok()) return op.emitOpError(view.status().message());

  Value source = map_.lookup(op.getSource());
  auto source_tile = mlir::cast<ct::TileType>(source.getType());
  llvm::SmallVector<Value> index =
      GetTileIndex(op.getOffsets(), op.getFullTileShape());
  if (mlir::cast<mlir::MemRefType>(op.getDestination().getType()).getRank() ==
      0) {
    source = ct::ReshapeOp::create(
        b_, ct::TileType::get({1}, source_tile.getElementType()), source);
    ct::TileType i32 = ScalarI32(b_.getContext());
    index.push_back(ct::ConstantOp::create(
        b_, i32, Splat(i32, b_.getI32IntegerAttr(0))));
  }

  ct::StoreViewTkoOp::create(
      b_, ct::TokenType::get(b_.getContext()),
      ct::MemoryOrderingSemanticsAttr::get(b_.getContext(),
                                           ct::MemoryOrderingSemantics::WEAK),
      /*memory_scope=*/ct::MemoryScopeAttr{}, source, *view, index, token_,
      /*optimization_hints=*/ct::OptimizationHintsAttr{});
  return success();
}

Value KernelConverter::RestoreScaleType(Value scale) {
  auto type = mlir::cast<ct::TileType>(scale.getType());
  if (!type.getElementType().isInteger(8)) return scale;
  auto typed = ct::TileType::get(
      type.getShape(), mlir::Float8E8M0FNUType::get(b_.getContext()));
  return ct::BitcastOp::create(b_, typed, scale);
}

Value KernelConverter::Transpose2D(Value tile) {
  auto type = mlir::cast<ct::TileType>(tile.getType());
  llvm::SmallVector<int64_t> swapped(type.getShape().rbegin(),
                                     type.getShape().rend());
  return ct::PermuteOp::create(
      b_, ct::TileType::get(swapped, type.getElementType()), tile,
      b_.getDenseI32ArrayAttr({1, 0}));
}

mlir::LogicalResult KernelConverter::ConvertDotScaled(
    ::xla::xtile::DotScaledOp op) {
  if (!op.getLhsScale() || !op.getRhsScale()) {
    return op.emitOpError("mmaf_scaled requires both scale operands");
  }
  auto add = mlir::dyn_cast_or_null<ma::AddFOp>(
      op.getResult().hasOneUse() ? *op.getResult().getUsers().begin()
                                 : nullptr);
  if (!add) {
    return op.emitOpError(
        "expected the dot to feed an arith.addf carrying the accumulator");
  }
  Value accumulator = add.getLhs() == op.getResult() ? add.getRhs()
                                                     : add.getLhs();

  auto dims = mlir::cast<mlir::stablehlo::DotDimensionNumbersAttr>(
      op.getDotDimensionNumbersAttr());
  int64_t lhs_c = dims.getLhsContractingDimensions()[0];
  int64_t rhs_c = dims.getRhsContractingDimensions()[0];
  Value lhs = MaybeUnpack(map_.lookup(op.getLhs()),
                          op.getLhsElemTypeAttr().getValue(),
                          op.getLhsKPack() ? lhs_c : 1 - lhs_c);
  Value rhs = MaybeUnpack(map_.lookup(op.getRhs()),
                          op.getRhsElemTypeAttr().getValue(),
                          op.getRhsKPack() ? rhs_c : 1 - rhs_c);
  if (!lhs || !rhs) {
    return op.emitOpError(
        "sub-byte operand is packed along a non-minor dimension, which the "
        "CUDA Tile IR backend cannot unpack in registers");
  }

  if (lhs_c != 1) lhs = Transpose2D(lhs);
  if (rhs_c != 0) rhs = Transpose2D(rhs);

  Value lhs_scale = RestoreScaleType(map_.lookup(op.getLhsScale()));
  Value rhs_scale = RestoreScaleType(map_.lookup(op.getRhsScale()));

  Value result = ct::MmaFScaledOp::create(b_, map_.lookup(accumulator).getType(),
                                          lhs, rhs, map_.lookup(accumulator),
                                          lhs_scale, rhs_scale);
  map_.map(add.getResult(), result);
  return success();
}

mlir::LogicalResult KernelConverter::ConvertFor(mlir::scf::ForOp op) {
  llvm::SmallVector<Value> inits;
  for (Value init : op.getInitArgs()) inits.push_back(map_.lookup(init));

  auto loop = ct::ForOp::create(b_, map_.lookup(op.getLowerBound()),
                                map_.lookup(op.getUpperBound()),
                                map_.lookup(op.getStep()), inits);

  map_.map(op.getInductionVar(), loop.getBody()->getArgument(0));
  for (auto [old_arg, new_arg] :
       llvm::zip(op.getRegionIterArgs(), loop.getBody()->getArguments().drop_front())) {
    map_.map(old_arg, new_arg);
  }
  for (auto [old_result, new_result] :
       llvm::zip(op.getResults(), loop.getResults())) {
    map_.map(old_result, new_result);
  }

  mlir::OpBuilder::InsertionGuard guard(b_);
  b_.setInsertionPointToStart(loop.getBody());
  for (mlir::Operation& nested : op.getBody()->without_terminator()) {
    if (failed(ConvertOp(&nested))) return failure();
  }
  llvm::SmallVector<Value> yielded;
  for (Value value : op.getBody()->getTerminator()->getOperands()) {
    yielded.push_back(map_.lookup(value));
  }
  ct::ContinueOp::create(b_, yielded);
  return success();
}

mlir::LogicalResult KernelConverter::ConvertConstant(ma::ConstantOp op) {
  mlir::Type type = ConvertType(op.getType());
  if (!type) return op.emitOpError("unsupported constant type");

  if (auto splat = mlir::dyn_cast<mlir::SplatElementsAttr>(op.getValue())) {
    map_.map(op.getResult(),
             ct::ConstantOp::create(
                 b_, type,
                 Splat(type, splat.getSplatValue<mlir::Attribute>())));
    return success();
  }
  if (!mlir::isa<mlir::ElementsAttr>(op.getValue())) {
    map_.map(op.getResult(),
             ct::ConstantOp::create(b_, type, Splat(type, op.getValue())));
    return success();
  }
  return op.emitOpError("only splat and scalar constants are supported");
}

mlir::LogicalResult KernelConverter::ConvertReduce(
    mlir::stablehlo::ReduceOp op) {
  if (op.getInputs().size() != 1 || op.getDimensions().size() != 1) {
    return op.emitOpError(
        "only single-operand, single-dimension reductions are supported");
  }
  mlir::Type result = ConvertType(op.getResult(0).getType());
  if (!result) return op.emitOpError("unsupported reduce result type");
  mlir::Type element = mlir::cast<ct::TileType>(result).getElementType();
  if (!mlir::isa<mlir::FloatType>(element)) {
    return op.emitOpError("only float reductions are supported");
  }

  mlir::Attribute identity;
  if (auto init = op.getInitValues()[0].getDefiningOp<ma::ConstantOp>()) {
    if (auto splat = mlir::dyn_cast<mlir::SplatElementsAttr>(init.getValue())) {
      identity = splat.getSplatValue<mlir::Attribute>();
    } else if (!mlir::isa<mlir::ElementsAttr>(init.getValue())) {
      identity = init.getValue();
    }
  }
  if (!identity) return op.emitOpError("reduce init value is not a constant");

  mlir::Block& body = op.getBody().front();
  if (!llvm::hasSingleElement(body.without_terminator())) {
    return op.emitOpError("reduce body must be a single combiner op");
  }
  mlir::Operation& combiner = *body.begin();

  auto reduce = ct::ReduceOp::create(
      b_, mlir::TypeRange{result},
      mlir::ValueRange{map_.lookup(op.getInputs()[0])},
      static_cast<uint32_t>(op.getDimensions()[0]),
      b_.getArrayAttr({identity}));

  auto scalar = ct::TileType::get({}, element);
  mlir::OpBuilder::InsertionGuard guard(b_);
  mlir::Block* block =
      b_.createBlock(&reduce.getBody(), {}, {scalar, scalar},
                     {b_.getLoc(), b_.getLoc()});
  Value lhs = block->getArgument(0);
  Value rhs = block->getArgument(1);
  Value combined;
  if (mlir::isa<mlir::stablehlo::AddOp>(combiner)) {
    combined = ct::AddFOp::create(b_, lhs, rhs, ct::RoundingMode::NEAREST_EVEN);
  } else if (mlir::isa<mlir::stablehlo::MulOp>(combiner)) {
    combined = ct::MulFOp::create(b_, lhs, rhs, ct::RoundingMode::NEAREST_EVEN);
  } else if (mlir::isa<mlir::stablehlo::MaxOp>(combiner)) {
    combined = ct::MaxFOp::create(b_, lhs, rhs, /*propagate_nan=*/true,
                                  /*flush_to_zero=*/false);
  } else if (mlir::isa<mlir::stablehlo::MinOp>(combiner)) {
    combined = ct::MinFOp::create(b_, lhs, rhs, /*propagate_nan=*/true,
                                  /*flush_to_zero=*/false);
  } else {
    return combiner.emitOpError("unsupported reduce combiner");
  }
  ct::YieldOp::create(b_, combined);

  map_.map(op.getResult(0), reduce.getResult(0));
  return success();
}

mlir::LogicalResult KernelConverter::ConvertOp(mlir::Operation* op) {
  if (auto extract = mlir::dyn_cast<::xla::xtile::ExtractTileOp>(op)) {
    return ConvertExtract(extract);
  }
  if (auto insert = mlir::dyn_cast<::xla::xtile::InsertTileOp>(op)) {
    return ConvertInsert(insert);
  }
  if (auto dot = mlir::dyn_cast<::xla::xtile::DotScaledOp>(op)) {
    return ConvertDotScaled(dot);
  }
  if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
    return ConvertFor(loop);
  }
  if (auto constant = mlir::dyn_cast<ma::ConstantOp>(op)) {
    return ConvertConstant(constant);
  }
  if (mlir::isa<ma::AddFOp>(op) && map_.contains(op->getResult(0))) {
    return success();
  }
  if (auto reduce = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
    return ConvertReduce(reduce);
  }
  if (auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    mlir::Type result = ConvertType(reshape.getType());
    if (!result) return op->emitOpError("unsupported reshape result type");
    map_.map(op->getResult(0),
             ct::ReshapeOp::create(b_, result,
                                   map_.lookup(reshape.getOperand())));
    return success();
  }
  if (auto broadcast = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
    mlir::Type result = ConvertType(broadcast.getType());
    if (!result) return op->emitOpError("unsupported broadcast result type");
    auto result_type = mlir::cast<ct::TileType>(result);
    auto source = mlir::cast<ct::TileType>(
        map_.lookup(broadcast.getOperand()).getType());
    llvm::ArrayRef<int64_t> dims = broadcast.getBroadcastDimensions();
    if (dims.size() != source.getRank()) {
      return op->emitOpError(
          "broadcast_dimensions must have one entry per source dimension");
    }
    for (int64_t i = 1; i < dims.size(); ++i) {
      if (dims[i] <= dims[i - 1]) {
        return op->emitOpError(
            "broadcast_dimensions must be strictly increasing");
      }
    }
    llvm::SmallVector<int64_t> ranked_shape(result_type.getRank(), 1);
    for (int64_t i = 0; i < dims.size(); ++i) {
      ranked_shape[dims[i]] = source.getShape()[i];
    }
    Value ranked = ct::ReshapeOp::create(
        b_, ct::TileType::get(ranked_shape, source.getElementType()),
        map_.lookup(broadcast.getOperand()));
    map_.map(op->getResult(0), ct::BroadcastOp::create(b_, result, ranked));
    return success();
  }
  if (auto convert = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    mlir::Type result = ConvertType(convert.getType());
    if (!result || !mlir::isa<mlir::FloatType>(
                       mlir::cast<ct::TileType>(result).getElementType())) {
      return op->emitOpError("only float conversions are supported");
    }
    map_.map(op->getResult(0),
             ct::FToFOp::create(b_, result, map_.lookup(convert.getOperand()),
                                ct::RoundingMode::NEAREST_EVEN));
    return success();
  }
  if (mlir::isa<mlir::stablehlo::MulOp, mlir::stablehlo::AddOp,
                mlir::stablehlo::SubtractOp, mlir::stablehlo::DivOp,
                mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp>(op)) {
    Value lhs = map_.lookup(op->getOperand(0));
    Value rhs = map_.lookup(op->getOperand(1));
    Value result;
    if (mlir::isa<mlir::stablehlo::MulOp>(op)) {
      result = ct::MulFOp::create(b_, lhs, rhs, ct::RoundingMode::NEAREST_EVEN);
    } else if (mlir::isa<mlir::stablehlo::AddOp>(op)) {
      result = ct::AddFOp::create(b_, lhs, rhs, ct::RoundingMode::NEAREST_EVEN);
    } else if (mlir::isa<mlir::stablehlo::SubtractOp>(op)) {
      result = ct::SubFOp::create(b_, lhs, rhs, ct::RoundingMode::NEAREST_EVEN);
    } else if (mlir::isa<mlir::stablehlo::DivOp>(op)) {
      result = ct::DivFOp::create(b_, lhs, rhs, ct::RoundingMode::NEAREST_EVEN);
    } else if (mlir::isa<mlir::stablehlo::MaxOp>(op)) {
      result = ct::MaxFOp::create(b_, lhs, rhs, /*propagate_nan=*/true,
                                  /*flush_to_zero=*/false);
    } else {
      result = ct::MinFOp::create(b_, lhs, rhs, /*propagate_nan=*/true,
                                  /*flush_to_zero=*/false);
    }
    map_.map(op->getResult(0), result);
    return success();
  }
  if (mlir::isa<ma::IndexCastOp, ma::IndexCastUIOp>(op)) {
    map_.map(op->getResult(0), map_.lookup(op->getOperand(0)));
    return success();
  }
  if (mlir::isa<ma::AddIOp, ma::SubIOp, ma::MulIOp, ma::DivSIOp, ma::DivUIOp,
                ma::RemSIOp, ma::RemUIOp>(op)) {
    Value lhs = map_.lookup(op->getOperand(0));
    Value rhs = map_.lookup(op->getOperand(1));
    Value result;
    if (mlir::isa<ma::AddIOp>(op)) {
      result = ct::AddIOp::create(b_, lhs, rhs);
    } else if (mlir::isa<ma::SubIOp>(op)) {
      result = ct::SubIOp::create(b_, lhs, rhs);
    } else if (mlir::isa<ma::MulIOp>(op)) {
      result = ct::MulIOp::create(b_, lhs, rhs);
    } else if (mlir::isa<ma::DivSIOp>(op)) {
      result = ct::DivIOp::create(b_, lhs, rhs, ct::Signedness::Signed);
    } else if (mlir::isa<ma::DivUIOp>(op)) {
      result = ct::DivIOp::create(b_, lhs, rhs, ct::Signedness::Unsigned);
    } else if (mlir::isa<ma::RemSIOp>(op)) {
      result = ct::RemIOp::create(b_, lhs, rhs, ct::Signedness::Signed);
    } else {
      result = ct::RemIOp::create(b_, lhs, rhs, ct::Signedness::Unsigned);
    }
    map_.map(op->getResult(0), result);
    return success();
  }
  if (auto transpose = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op)) {
    if (IsRhsScaleTranspose(transpose)) {
      map_.map(transpose.getResult(), map_.lookup(transpose.getOperand()));
      return success();
    }
    mlir::Type result = ConvertType(transpose.getType());
    if (!result) return op->emitOpError("unsupported transpose result type");
    llvm::SmallVector<int32_t> permutation(transpose.getPermutation().begin(),
                                           transpose.getPermutation().end());
    map_.map(transpose.getResult(),
             ct::PermuteOp::create(b_, result,
                                   map_.lookup(transpose.getOperand()),
                                   b_.getDenseI32ArrayAttr(permutation)));
    return success();
  }
  if (mlir::isa<::xla::xtile::EntryFuncReturnOp>(op)) {
    ct::ReturnOp::create(b_);
    return success();
  }
  return op->emitOpError(
      "has no CUDA Tile IR equivalent; this fusion should have stayed on the "
      "Triton rung");
}

mlir::LogicalResult KernelConverter::Run() {
  mlir::Block& body = target_.getBody().front();
  b_.setInsertionPointToStart(&body);

  token_ = ct::MakeTokenOp::create(b_, ct::TokenType::get(b_.getContext()));

  auto block_id = ct::GetTileBlockIdOp::create(b_, ScalarI32(b_.getContext()),
                                               ScalarI32(b_.getContext()),
                                               ScalarI32(b_.getContext()));
  map_.map(entry_.getProgramId(), block_id.getBlockIdX());

  for (auto [old_arg, new_arg] :
       llvm::zip(entry_.getBufferArgs(), body.getArguments())) {
    map_.map(old_arg, new_arg);
  }
  for (auto [old_arg, new_arg] :
       llvm::zip(entry_.getOpaqueArgs(),
                 body.getArguments().drop_front(
                     entry_.getBufferArgs().size()))) {
    map_.map(old_arg, new_arg);
  }

  for (mlir::Operation& op : entry_.getBody().front()) {
    if (failed(ConvertOp(&op))) return failure();
  }
  return success();
}

class XTileLowerToCudaTilePass
    : public impl::XTileLowerToCudaTilePassBase<XTileLowerToCudaTilePass> {
 public:
  using XTileLowerToCudaTilePassBase::XTileLowerToCudaTilePassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* ctx = &getContext();

    ::xla::xtile::EntryFuncOp entry;
    module.walk([&](::xla::xtile::EntryFuncOp op) { entry = op; });
    if (!entry) {
      module.emitError("no xtile.entry_func to lower");
      return signalPassFailure();
    }

    mlir::ImplicitLocOpBuilder b(entry.getLoc(), ctx);
    b.setInsertionPointToStart(module.getBody());
    auto tile_module = ct::ModuleOp::create(b, "kernels");
    b.setInsertionPointToStart(&tile_module.getBody().front());

    llvm::SmallVector<mlir::Type> arg_types;
    for (mlir::Value arg : entry.getBufferArgs()) {
      auto memref = mlir::cast<mlir::MemRefType>(arg.getType());
      arg_types.push_back(ct::TileType::get(
          {}, ct::PointerType::get(ctx, memref.getElementType())));
    }
    for (mlir::Value arg : entry.getOpaqueArgs()) {
      arg_types.push_back(ct::TileType::get({}, arg.getType()));
    }

    auto target = ct::EntryOp::create(b, entry.getName(),
                                      b.getFunctionType(arg_types, {}),
                                      /*arg_attrs=*/mlir::ArrayAttr{},
                                      /*res_attrs=*/mlir::ArrayAttr{},
                                      /*optimization_hints=*/
                                      ct::OptimizationHintsAttr{});
    target.getBody().emplaceBlock().addArguments(
        arg_types, llvm::SmallVector<mlir::Location>(arg_types.size(),
                                                     entry.getLoc()));

    KernelConverter converter(entry, target, b);
    if (failed(converter.Run())) {
      tile_module.erase();
      return signalPassFailure();
    }
    entry.erase();
  }
};

}  // namespace

bool IsLowerableToCudaTile(mlir::Operation* module) {
  int64_t num_dots = 0;
  module->walk([&](::xla::xtile::DotScaledOp) { ++num_dots; });
  if (num_dots > 1) return false;

  bool lowerable = true;
  module->walk([&](::xla::xtile::DotScaledOp dot) {
    if (dot.getLhsElemTypeAttr().getValue() !=
        dot.getRhsElemTypeAttr().getValue()) {
      lowerable = false;
      return mlir::WalkResult::interrupt();
    }
    auto dims = mlir::cast<mlir::stablehlo::DotDimensionNumbersAttr>(
        dot.getDotDimensionNumbersAttr());
    if (!dims.getLhsBatchingDimensions().empty() ||
        !dims.getRhsBatchingDimensions().empty()) {
      lowerable = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!lowerable) return false;

  module->walk([&](mlir::Operation* op) {
    Value buffer;
    if (auto extract = mlir::dyn_cast<::xla::xtile::ExtractTileOp>(op)) {
      buffer = extract.getSource();
    } else if (auto insert = mlir::dyn_cast<::xla::xtile::InsertTileOp>(op)) {
      buffer = insert.getDestination();
    }
    if (buffer && !mlir::isa<mlir::BlockArgument>(buffer)) {
      lowerable = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!lowerable) return false;

  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<::xla::xtile::ExtractTileOp, ::xla::xtile::InsertTileOp,
                  ::xla::xtile::DotScaledOp, ::xla::xtile::EntryFuncOp,
                  ::xla::xtile::EntryFuncReturnOp, mlir::scf::ForOp,
                  mlir::scf::YieldOp, ma::ConstantOp, ma::AddFOp, ma::AddIOp,
                  ma::SubIOp, ma::MulIOp, ma::DivSIOp, ma::DivUIOp, ma::RemSIOp,
                  ma::RemUIOp, ma::IndexCastOp, ma::IndexCastUIOp,
                  mlir::ModuleOp, ::xla::ApplyIndexingOp,
                  mlir::stablehlo::BroadcastInDimOp,
                  mlir::stablehlo::ConvertOp, mlir::stablehlo::MulOp,
                  mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                  mlir::stablehlo::DivOp, mlir::stablehlo::MaxOp,
                  mlir::stablehlo::MinOp, mlir::stablehlo::ReshapeOp,
                  mlir::stablehlo::ReduceOp, mlir::stablehlo::ReturnOp>(op)) {
      return mlir::WalkResult::advance();
    }
    if (mlir::isa<mlir::stablehlo::TransposeOp>(op)) {
      return mlir::WalkResult::advance();
    }
    LOG_FIRST_N(WARNING, 20)
        << "CUDA Tile IR declines a fusion: no lowering for '"
        << op->getName().getStringRef().str() << "'.";
    lowerable = false;
    return mlir::WalkResult::interrupt();
  });
  return lowerable;
}

}  // namespace xla::gpu::tile_ir
