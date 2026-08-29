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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::flydsl {
namespace {

using mlir::Value;

bool IsSupportedFloatingType(PrimitiveType type) {
  return type == F16 || type == BF16 || type == F32 || type == F64;
}

bool IsFp8Type(PrimitiveType type) {
  return type == F8E4M3FN || type == F8E5M2 || type == F8E4M3FNUZ ||
         type == F8E5M2FNUZ;
}

bool IsS4Type(PrimitiveType type) { return type == S4; }

bool IsFloatingValueType(PrimitiveType type) {
  return IsSupportedFloatingType(type) || IsFp8Type(type);
}

bool IsSupportedSignedIntegerType(PrimitiveType type) {
  return type == S8 || type == S16 || type == S32 || type == S64;
}

bool IsLowPrecisionFloatingType(PrimitiveType type) {
  return type == F16 || type == BF16;
}

bool IsSupportedType(PrimitiveType type) {
  return IsSupportedFloatingType(type) || IsSupportedSignedIntegerType(type);
}

bool IsSupportedValueType(PrimitiveType type) {
  return IsSupportedType(type) || IsFp8Type(type) || IsS4Type(type) ||
         type == PRED;
}

bool IsSupportedExternalType(PrimitiveType type) {
  return IsSupportedValueType(type);
}

bool IsSupportedConversion(PrimitiveType source, PrimitiveType destination) {
  if (!IsSupportedValueType(source) || !IsSupportedValueType(destination)) {
    return false;
  }
  // S4 is byte-packed at the device ABI boundary. Keep logical values in
  // signed byte vectors inside the kernel, then unpack inputs and pack outputs
  // at the buffer boundary.
  if (IsS4Type(source)) {
    return !IsS4Type(destination) &&
           (IsSupportedSignedIntegerType(destination) ||
            IsSupportedFloatingType(destination));
  }
  if (IsS4Type(destination)) {
    return source == PRED || IsSupportedSignedIntegerType(source) ||
           IsSupportedFloatingType(source);
  }
  if (!IsFp8Type(source) && !IsFp8Type(destination)) {
    return true;
  }
  // Match XLA's Triton FP8 conversion grammar. FP8 values may convert among
  // FP8 formats or to/from the 16/32-bit floating compute types, but not
  // directly to predicates, integers, or F64.
  return (IsFp8Type(source) && IsFp8Type(destination)) ||
         ((IsFp8Type(source) || IsFp8Type(destination)) &&
          (source == F16 || source == BF16 || source == F32 ||
           destination == F16 || destination == BF16 || destination == F32));
}

int64_t ElementBits(PrimitiveType type) {
  switch (type) {
    case S4:
      return 4;
    case PRED:
    case S8:
    case F8E4M3FN:
    case F8E5M2:
    case F8E4M3FNUZ:
    case F8E5M2FNUZ:
      return 8;
    case F16:
    case BF16:
    case S16:
      return 16;
    case F32:
    case S32:
      return 32;
    case F64:
    case S64:
      return 64;
    default:
      return 0;
  }
}

mlir::Type StorageElementType(PrimitiveType type, mlir::OpBuilder& builder) {
  // XLA's device ABI stores predicates and every FP8 format in one byte.
  // Keep FP8 vectors byte-backed inside the native kernel as well: MLIR's
  // generic GPU type converter does not legalize vectors of FNUZ types, while
  // scalar FP8 values are correctly lowered by XLA's AMD conversion pass.
  if (type == PRED || IsFp8Type(type) || IsS4Type(type)) {
    return builder.getI8Type();
  }
  return emitters::PrimitiveTypeToMlirType(type, builder);
}

bool HasSamePhysicalDimensions(const Shape& lhs, const Shape& rhs) {
  if (lhs.element_type() == S4 || rhs.element_type() == S4) {
    return Shape::Equal().IgnoreElementType().IgnoreElementSizeInLayout()(lhs,
                                                                          rhs);
  }
  return ShapeUtil::EqualIgnoringElementType(lhs, rhs) &&
         LayoutUtil::Equal(lhs.layout(), rhs.layout());
}

bool HasSamePackedPhysicalDimensions(const Shape& lhs, const Shape& rhs) {
  return Shape::Equal().IgnoreElementType().IgnoreElementSizeInLayout()(lhs,
                                                                        rhs);
}

bool HasSameFlatElements(const Shape& lhs, const Shape& rhs) {
  return lhs.IsArray() && rhs.IsArray() && lhs.has_layout() &&
         rhs.has_layout() &&
         ShapeUtil::ElementsIn(lhs) == ShapeUtil::ElementsIn(rhs);
}

bool HasSamePhysicalBytes(const Shape& lhs, const Shape& rhs) {
  return lhs.IsArray() && rhs.IsArray() && lhs.has_layout() &&
         rhs.has_layout() &&
         ShapeUtil::ByteSizeOfElements(lhs) ==
             ShapeUtil::ByteSizeOfElements(rhs);
}

bool IsSupportedPhysicalView(const HloInstruction* instruction) {
  if (instruction->operand_count() != 1 ||
      !IsSupportedValueType(instruction->shape().element_type()) ||
      !IsSupportedValueType(instruction->operand(0)->shape().element_type()) ||
      !instruction->shape().layout().tiles().empty() ||
      !instruction->operand(0)->shape().layout().tiles().empty()) {
    return false;
  }
  const PrimitiveType result_type = instruction->shape().element_type();
  const PrimitiveType operand_type =
      instruction->operand(0)->shape().element_type();
  switch (instruction->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
      return HasSamePhysicalBytes(instruction->shape(),
                                  instruction->operand(0)->shape()) &&
             (result_type == operand_type ||
              (result_type != PRED && operand_type != PRED));
    case HloOpcode::kReshape:
      return result_type == operand_type &&
             HasSameFlatElements(instruction->shape(),
                                 instruction->operand(0)->shape()) &&
             ShapeUtil::ReshapeIsBitcast(instruction->operand(0)->shape(),
                                         instruction->shape());
    case HloOpcode::kTranspose:
      return result_type == operand_type &&
             HasSameFlatElements(instruction->shape(),
                                 instruction->operand(0)->shape()) &&
             ShapeUtil::TransposeIsBitcast(instruction->operand(0)->shape(),
                                           instruction->shape(),
                                           instruction->dimensions());
    default:
      return false;
  }
}

struct DataMovingTranspose {
  std::vector<int64_t> output_dimensions;
  std::vector<int64_t> output_to_input_dimensions;
  std::vector<int64_t> input_strides;
};

std::optional<DataMovingTranspose> GetDataMovingTranspose(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kTranspose ||
      instruction->operand_count() != 1 ||
      IsSupportedPhysicalView(instruction)) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& output = instruction->shape();
  const int64_t rank = output.dimensions_size();
  if (!input.IsArray() || !output.IsArray() || !input.has_layout() ||
      !output.has_layout() || rank == 0 ||
      input.dimensions_size() != rank ||
      input.element_type() != output.element_type() ||
      input.element_type() == S4 ||
      !IsSupportedValueType(output.element_type()) ||
      !input.layout().tiles().empty() || !output.layout().tiles().empty() ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      instruction->dimensions().size() != rank ||
      ShapeUtil::ElementsIn(input) != ShapeUtil::ElementsIn(output)) {
    return std::nullopt;
  }

  DataMovingTranspose transpose;
  transpose.output_dimensions.assign(output.dimensions().begin(),
                                      output.dimensions().end());
  transpose.output_to_input_dimensions.assign(
      instruction->dimensions().begin(), instruction->dimensions().end());
  transpose.input_strides.resize(rank);
  std::vector<bool> seen_input_dimension(rank, false);
  int64_t input_stride = 1;
  for (int64_t input_dimension = rank - 1; input_dimension >= 0;
       --input_dimension) {
    if (input.dimensions(input_dimension) <= 0) {
      return std::nullopt;
    }
    transpose.input_strides[input_dimension] = input_stride;
    input_stride *= input.dimensions(input_dimension);
  }
  for (int64_t output_dimension = 0; output_dimension < rank;
       ++output_dimension) {
    const int64_t input_dimension =
        transpose.output_to_input_dimensions[output_dimension];
    if (input_dimension < 0 || input_dimension >= rank ||
        seen_input_dimension[input_dimension] ||
        output.dimensions(output_dimension) !=
            input.dimensions(input_dimension)) {
      return std::nullopt;
    }
    seen_input_dimension[input_dimension] = true;
  }
  return transpose;
}

std::optional<int64_t> IotaPhysicalStride(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kIota ||
      !instruction->shape().IsArray() || !instruction->shape().has_layout() ||
      !instruction->shape().layout().tiles().empty()) {
    return std::nullopt;
  }
  const int64_t iota_dimension =
      Cast<const HloIotaInstruction>(instruction)->iota_dimension();
  if (iota_dimension < 0 ||
      iota_dimension >= instruction->shape().dimensions_size()) {
    return std::nullopt;
  }
  int64_t stride = 1;
  for (int64_t dimension : instruction->shape().layout().minor_to_major()) {
    if (dimension == iota_dimension) {
      return stride;
    }
    const int64_t dimension_size = instruction->shape().dimensions(dimension);
    if (dimension_size != 0 &&
        stride > std::numeric_limits<int64_t>::max() / dimension_size) {
      return std::nullopt;
    }
    stride *= dimension_size;
  }
  return std::nullopt;
}

std::optional<int64_t> ContiguousSliceBase(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kSlice ||
      instruction->operand_count() != 1) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& output = instruction->shape();
  const int64_t rank = input.dimensions_size();
  if (!input.IsArray() || !output.IsArray() || !input.has_layout() ||
      !output.has_layout() || input.element_type() != output.element_type() ||
      output.dimensions_size() != rank ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      instruction->slice_starts().size() != rank ||
      instruction->slice_limits().size() != rank ||
      instruction->slice_strides().size() != rank ||
      ShapeUtil::ElementsIn(output) == 0) {
    return std::nullopt;
  }

  int64_t base = 0;
  int64_t last = 0;
  int64_t stride = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    const int64_t start = instruction->slice_starts(dimension);
    const int64_t limit = instruction->slice_limits(dimension);
    if (instruction->slice_strides(dimension) != 1 || start < 0 ||
        limit <= start || limit > input.dimensions(dimension) ||
        limit - start != output.dimensions(dimension)) {
      return std::nullopt;
    }
    base += start * stride;
    last += (limit - 1) * stride;
    stride *= input.dimensions(dimension);
  }
  if (last - base + 1 != ShapeUtil::ElementsIn(output)) {
    return std::nullopt;
  }
  return base;
}

struct RectangularSlice {
  int64_t input_base = 0;
  // Physical input-element stride for one logical output coordinate in every
  // dimension.  This includes both the physical row-major input stride and
  // the HLO slice stride.
  std::vector<int64_t> input_strides;
  std::vector<int64_t> output_dimensions;
  int64_t input_minor_dimension = 0;
  int64_t input_minor_start = 0;
  // Nonzero when flattening the output preserves one affine input stride.
  // This is common for channel subsampling such as [..., 0:N:2].
  int64_t flat_input_stride = 0;
};

std::optional<RectangularSlice> GetRectangularSlice(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kSlice ||
      instruction->operand_count() != 1) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& output = instruction->shape();
  if (!input.IsArray() || !output.IsArray() || !input.has_layout() ||
      !output.has_layout() || input.element_type() != output.element_type() ||
      input.dimensions_size() < 1 ||
      output.dimensions_size() != input.dimensions_size() ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      instruction->slice_starts().size() != input.dimensions_size() ||
      instruction->slice_limits().size() != input.dimensions_size() ||
      instruction->slice_strides().size() != input.dimensions_size() ||
      ShapeUtil::ElementsIn(output) == 0) {
    return std::nullopt;
  }
  const int64_t rank = input.dimensions_size();
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    const int64_t start = instruction->slice_starts(dimension);
    const int64_t limit = instruction->slice_limits(dimension);
    const int64_t slice_stride = instruction->slice_strides(dimension);
    if (slice_stride <= 0 || start < 0 || limit <= start ||
        limit > input.dimensions(dimension) ||
        (limit - start + slice_stride - 1) / slice_stride !=
            output.dimensions(dimension)) {
      return std::nullopt;
    }
  }
  RectangularSlice slice;
  slice.input_strides.resize(rank);
  slice.output_dimensions.assign(output.dimensions().begin(),
                                 output.dimensions().end());
  slice.input_minor_dimension = input.dimensions(rank - 1);
  slice.input_minor_start = instruction->slice_starts(rank - 1);
  int64_t input_stride = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    slice.input_strides[dimension] =
        input_stride * instruction->slice_strides(dimension);
    slice.input_base += instruction->slice_starts(dimension) * input_stride;
    input_stride *= input.dimensions(dimension);
  }
  int64_t expected_input_stride = slice.input_strides.back();
  bool has_flat_input_stride = true;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    has_flat_input_stride &=
        slice.input_strides[dimension] == expected_input_stride;
    expected_input_stride *= slice.output_dimensions[dimension];
  }
  if (has_flat_input_stride) {
    slice.flat_input_stride = slice.input_strides.back();
  }
  return slice;
}

struct DynamicSlice {
  std::vector<int64_t> input_strides;
  std::vector<int64_t> output_dimensions;
  std::vector<int64_t> start_limits;
};

struct RowGather {
  std::vector<int64_t> indexed_dimension_sizes;
  int64_t index_components = 0;
  int64_t input_row_elements = 0;
  int64_t row_elements = 0;
};

struct StridedAxisGather {
  int64_t prefix_elements = 0;
  int64_t index_rows = 0;
  int64_t indexed_dimension_size = 0;
  int64_t suffix_elements = 0;
};

enum class ScatterUpdateKind { kOverwrite, kAdd, kMaximum, kMinimum };

struct OverwriteRowScatter {
  const HloScatterInstruction* scatter = nullptr;
  std::vector<int64_t> indexed_dimension_sizes;
  std::vector<int64_t> indexed_dimension_start_limits;
  std::vector<int64_t> output_dimension_sizes;
  std::vector<int64_t> update_window_dimension_sizes;
  int64_t index_components = 0;
  int64_t update_rows = 0;
  int64_t output_row_elements = 0;
  int64_t row_elements = 0;
  int64_t contiguous_elements = 0;
  bool has_contiguous_window = true;
  ScatterUpdateKind update_kind = ScatterUpdateKind::kOverwrite;
  bool requires_atomic_operation = false;
};

std::optional<RowGather> GetRowGather(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kGather ||
      instruction->operand_count() != 2) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& indices = instruction->operand(1)->shape();
  const Shape& output = instruction->shape();
  const int64_t input_rank = input.dimensions_size();
  if (!input.IsArray() || !indices.IsArray() || !output.IsArray() ||
      !input.has_layout() || !indices.has_layout() || !output.has_layout() ||
      input_rank == 0 ||
      input.element_type() != output.element_type() ||
      input.element_type() == S4 ||
      !IsSupportedValueType(input.element_type()) ||
      (indices.element_type() != S32 && indices.element_type() != S64) ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(indices.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      ShapeUtil::ElementsIn(input) <= 0 ||
      ShapeUtil::ElementsIn(indices) <= 0 ||
      ShapeUtil::ElementsIn(output) <= 0 ||
      ShapeUtil::ByteSizeOfElements(input) >
          std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }

  const HloGatherInstruction* gather =
      Cast<const HloGatherInstruction>(instruction);
  const GatherDimensionNumbers& dimensions =
      gather->gather_dimension_numbers();
  const int64_t index_components = dimensions.start_index_map_size();
  if (index_components <= 0 || index_components > input_rank ||
      dimensions.operand_batching_dims_size() != 0 ||
      dimensions.start_indices_batching_dims_size() != 0 ||
      gather->gather_slice_sizes().size() != input_rank) {
    return std::nullopt;
  }
  for (int64_t dimension = 0; dimension < input_rank; ++dimension) {
    if (input.dimensions(dimension) <= 0 ||
        gather->gather_slice_sizes()[dimension] <= 0 ||
        gather->gather_slice_sizes()[dimension] >
            input.dimensions(dimension) ||
        (dimension < index_components &&
         gather->gather_slice_sizes()[dimension] != 1) ||
        (dimension >= index_components && dimension != input_rank - 1 &&
         gather->gather_slice_sizes()[dimension] !=
             input.dimensions(dimension)) ||
        (dimension < index_components &&
         dimensions.start_index_map(dimension) != dimension)) {
      return std::nullopt;
    }
  }
  if (index_components < input_rank - 1 &&
      gather->gather_slice_sizes().back() != input.dimensions(input_rank - 1)) {
    // A partial minor dimension is contiguous only when it is the entire
    // trailing payload. With additional trailing dimensions, each logical row
    // would require a separate input stride.
    return std::nullopt;
  }

  const int64_t indices_rank = indices.dimensions_size();
  int64_t batch_rank = 0;
  if (dimensions.index_vector_dim() == indices_rank) {
    if (index_components != 1) {
      return std::nullopt;
    }
    batch_rank = indices_rank;
  } else if (indices_rank > 0 &&
             dimensions.index_vector_dim() == indices_rank - 1 &&
             indices.dimensions(indices_rank - 1) == index_components) {
    batch_rank = indices_rank - 1;
  } else {
    return std::nullopt;
  }

  const int64_t trailing_rank = input_rank - index_components;
  const bool collapsed_form =
      dimensions.collapsed_slice_dims_size() == index_components &&
      dimensions.offset_dims_size() == trailing_rank &&
      output.dimensions_size() == batch_rank + trailing_rank;
  const bool explicit_unit_form =
      dimensions.collapsed_slice_dims_size() == 0 &&
      dimensions.offset_dims_size() == input_rank &&
      output.dimensions_size() == batch_rank + input_rank;
  if (!collapsed_form && !explicit_unit_form) {
    return std::nullopt;
  }
  for (int64_t dimension = 0; dimension < dimensions.offset_dims_size();
       ++dimension) {
    if (dimensions.offset_dims(dimension) != batch_rank + dimension) {
      return std::nullopt;
    }
  }
  if (collapsed_form) {
    for (int64_t dimension = 0; dimension < index_components; ++dimension) {
      if (dimensions.collapsed_slice_dims(dimension) != dimension) {
        return std::nullopt;
      }
    }
  }
  for (int64_t dimension = 0; dimension < batch_rank; ++dimension) {
    if (indices.dimensions(dimension) <= 0 ||
        output.dimensions(dimension) != indices.dimensions(dimension)) {
      return std::nullopt;
    }
  }
  const int64_t output_slice_rank =
      explicit_unit_form ? input_rank : trailing_rank;
  for (int64_t dimension = 0; dimension < output_slice_rank; ++dimension) {
    const int64_t input_dimension =
        explicit_unit_form ? dimension : index_components + dimension;
    const int64_t expected = gather->gather_slice_sizes()[input_dimension];
    if (output.dimensions(batch_rank + dimension) != expected) {
      return std::nullopt;
    }
  }

  int64_t output_rows = 1;
  for (int64_t dimension = 0; dimension < batch_rank; ++dimension) {
    output_rows *= indices.dimensions(dimension);
  }
  int64_t input_row_elements = 1;
  int64_t row_elements = 1;
  for (int64_t dimension = index_components; dimension < input_rank;
       ++dimension) {
    input_row_elements *= input.dimensions(dimension);
    row_elements *= gather->gather_slice_sizes()[dimension];
  }
  if (ShapeUtil::ElementsIn(indices) != output_rows * index_components ||
      ShapeUtil::ElementsIn(output) != output_rows * row_elements) {
    return std::nullopt;
  }

  return RowGather{
      /*indexed_dimension_sizes=*/std::vector<int64_t>(
          input.dimensions().begin(),
          input.dimensions().begin() + index_components),
      /*index_components=*/index_components,
      /*input_row_elements=*/input_row_elements,
      /*row_elements=*/row_elements};
}

std::optional<StridedAxisGather> GetStridedAxisGather(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kGather ||
      instruction->operand_count() != 2) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& indices = instruction->operand(1)->shape();
  const Shape& output = instruction->shape();
  const int64_t input_rank = input.dimensions_size();
  if (!input.IsArray() || !indices.IsArray() || !output.IsArray() ||
      !input.has_layout() || !indices.has_layout() || !output.has_layout() ||
      input_rank < 2 || input.element_type() != output.element_type() ||
      input.element_type() == S4 ||
      !IsSupportedValueType(input.element_type()) ||
      (indices.element_type() != S32 && indices.element_type() != S64) ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(indices.layout()) ||
      ShapeUtil::ElementsIn(input) <= 0 ||
      ShapeUtil::ElementsIn(indices) <= 0 ||
      ShapeUtil::ElementsIn(output) <= 0 ||
      ShapeUtil::ByteSizeOfElements(input) >
          std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }

  const HloGatherInstruction* gather =
      Cast<const HloGatherInstruction>(instruction);
  const GatherDimensionNumbers& dimensions =
      gather->gather_dimension_numbers();
  if (dimensions.start_index_map_size() != 1 ||
      dimensions.operand_batching_dims_size() != 0 ||
      dimensions.start_indices_batching_dims_size() != 0 ||
      gather->gather_slice_sizes().size() != input_rank) {
    return std::nullopt;
  }
  const int64_t axis = dimensions.start_index_map(0);
  if (axis <= 0 || axis >= input_rank || input.dimensions(axis) <= 0) {
    return std::nullopt;
  }
  for (int64_t dimension = 0; dimension < input_rank; ++dimension) {
    if (input.dimensions(dimension) <= 0 ||
        gather->gather_slice_sizes()[dimension] !=
            (dimension == axis ? 1 : input.dimensions(dimension))) {
      return std::nullopt;
    }
  }

  const int64_t indices_rank = indices.dimensions_size();
  int64_t batch_rank = 0;
  if (dimensions.index_vector_dim() == indices_rank) {
    batch_rank = indices_rank;
  } else if (indices_rank > 0 &&
             dimensions.index_vector_dim() == indices_rank - 1 &&
             indices.dimensions(indices_rank - 1) == 1) {
    batch_rank = indices_rank - 1;
  } else {
    return std::nullopt;
  }

  const bool collapsed_form =
      dimensions.collapsed_slice_dims_size() == 1 &&
      dimensions.collapsed_slice_dims(0) == axis &&
      dimensions.offset_dims_size() == input_rank - 1 &&
      output.dimensions_size() == input_rank - 1 + batch_rank;
  const bool explicit_unit_form =
      dimensions.collapsed_slice_dims_size() == 0 &&
      dimensions.offset_dims_size() == input_rank &&
      output.dimensions_size() == input_rank + batch_rank;
  if (!collapsed_form && !explicit_unit_form) {
    return std::nullopt;
  }

  const auto has_inserted_batch_order = [&] {
    if (!LayoutUtil::IsMonotonicWithDim0Major(output.layout())) {
      return false;
    }
    int64_t offset_index = 0;
    for (int64_t output_dimension = 0;
         output_dimension < output.dimensions_size(); ++output_dimension) {
      const bool is_batch_dimension =
          output_dimension >= axis && output_dimension < axis + batch_rank;
      if (is_batch_dimension) {
        if (output.dimensions(output_dimension) !=
            indices.dimensions(output_dimension - axis)) {
          return false;
        }
        continue;
      }
      if (offset_index >= dimensions.offset_dims_size() ||
          dimensions.offset_dims(offset_index) != output_dimension) {
        return false;
      }
      const int64_t input_dimension =
          output_dimension < axis
              ? output_dimension
              : output_dimension - batch_rank +
                    (collapsed_form ? 1 : 0);
      const int64_t expected =
          input_dimension == axis ? 1 : input.dimensions(input_dimension);
      if (output.dimensions(output_dimension) != expected) {
        return false;
      }
      ++offset_index;
    }
    return offset_index == dimensions.offset_dims_size();
  };
  const auto has_canonical_batch_first_order = [&] {
    for (int64_t offset = 0; offset < dimensions.offset_dims_size();
         ++offset) {
      if (dimensions.offset_dims(offset) != batch_rank + offset) {
        return false;
      }
    }
    for (int64_t batch = 0; batch < batch_rank; ++batch) {
      if (output.dimensions(batch) != indices.dimensions(batch)) {
        return false;
      }
    }
    auto output_dimension_for_input = [&](int64_t input_dimension) {
      return batch_rank + input_dimension -
             (collapsed_form && input_dimension > axis ? 1 : 0);
    };
    for (int64_t input_dimension = 0; input_dimension < input_rank;
         ++input_dimension) {
      if (collapsed_form && input_dimension == axis) {
        continue;
      }
      const int64_t output_dimension =
          output_dimension_for_input(input_dimension);
      const int64_t expected =
          input_dimension == axis ? 1 : input.dimensions(input_dimension);
      if (output.dimensions(output_dimension) != expected) {
        return false;
      }
    }

    llvm::SmallVector<int64_t> expected_minor_to_major;
    for (int64_t input_dimension = input_rank - 1;
         input_dimension > axis; --input_dimension) {
      expected_minor_to_major.push_back(
          output_dimension_for_input(input_dimension));
    }
    if (explicit_unit_form) {
      expected_minor_to_major.push_back(
          output_dimension_for_input(axis));
    }
    for (int64_t batch = batch_rank - 1; batch >= 0; --batch) {
      expected_minor_to_major.push_back(batch);
    }
    for (int64_t input_dimension = axis - 1; input_dimension >= 0;
         --input_dimension) {
      expected_minor_to_major.push_back(
          output_dimension_for_input(input_dimension));
    }
    if (output.layout().minor_to_major_size() !=
        expected_minor_to_major.size()) {
      return false;
    }
    for (auto [position, dimension] :
         llvm::enumerate(expected_minor_to_major)) {
      if (output.layout().minor_to_major(position) != dimension) {
        return false;
      }
    }
    return true;
  };
  if (!has_inserted_batch_order() && !has_canonical_batch_first_order()) {
    return std::nullopt;
  }

  int64_t prefix_elements = 1;
  for (int64_t dimension = 0; dimension < axis; ++dimension) {
    prefix_elements *= input.dimensions(dimension);
  }
  int64_t index_rows = 1;
  for (int64_t dimension = 0; dimension < batch_rank; ++dimension) {
    if (indices.dimensions(dimension) <= 0) {
      return std::nullopt;
    }
    index_rows *= indices.dimensions(dimension);
  }
  int64_t suffix_elements = 1;
  for (int64_t dimension = axis + 1; dimension < input_rank; ++dimension) {
    suffix_elements *= input.dimensions(dimension);
  }
  if (ShapeUtil::ElementsIn(indices) != index_rows ||
      ShapeUtil::ElementsIn(output) !=
          prefix_elements * index_rows * suffix_elements) {
    return std::nullopt;
  }
  return StridedAxisGather{
      /*prefix_elements=*/prefix_elements,
      /*index_rows=*/index_rows,
      /*indexed_dimension_size=*/input.dimensions(axis),
      /*suffix_elements=*/suffix_elements};
}

std::optional<OverwriteRowScatter> GetOverwriteRowScatter(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kScatter ||
      instruction->operand_count() != 3) {
    return std::nullopt;
  }
  const HloScatterInstruction* scatter =
      Cast<const HloScatterInstruction>(instruction);
  if (scatter->scatter_operand_count() != 1 ||
      scatter->operand(0)->opcode() != HloOpcode::kParameter) {
    return std::nullopt;
  }

  const HloComputation* update_computation = scatter->to_apply();
  const HloInstruction* update_root =
      update_computation == nullptr ? nullptr
                                    : update_computation->root_instruction();
  if (update_root == nullptr || update_computation->num_parameters() != 2) {
    return std::nullopt;
  }
  ScatterUpdateKind update_kind;
  if (update_root->opcode() == HloOpcode::kParameter &&
      update_root->parameter_number() == 1) {
    update_kind = ScatterUpdateKind::kOverwrite;
  } else {
    const HloInstruction* combiner = update_root;
    while (combiner->opcode() == HloOpcode::kConvert &&
           combiner->operand_count() == 1) {
      combiner = combiner->operand(0);
    }
    auto parameter_number_through_converts =
        [](const HloInstruction* value) -> std::optional<int64_t> {
      while (value->opcode() == HloOpcode::kConvert &&
             value->operand_count() == 1) {
        value = value->operand(0);
      }
      if (value->opcode() != HloOpcode::kParameter) {
        return std::nullopt;
      }
      return value->parameter_number();
    };
    std::optional<int64_t> lhs_parameter;
    std::optional<int64_t> rhs_parameter;
    if ((combiner->opcode() == HloOpcode::kAdd ||
         combiner->opcode() == HloOpcode::kMaximum ||
         combiner->opcode() == HloOpcode::kMinimum) &&
        combiner->operand_count() == 2) {
      lhs_parameter =
          parameter_number_through_converts(combiner->operand(0));
      rhs_parameter =
          parameter_number_through_converts(combiner->operand(1));
    }
    if (!lhs_parameter.has_value() || !rhs_parameter.has_value() ||
        *lhs_parameter == *rhs_parameter) {
      return std::nullopt;
    }
    switch (combiner->opcode()) {
      case HloOpcode::kAdd:
        update_kind = ScatterUpdateKind::kAdd;
        break;
      case HloOpcode::kMaximum:
        update_kind = ScatterUpdateKind::kMaximum;
        break;
      case HloOpcode::kMinimum:
        update_kind = ScatterUpdateKind::kMinimum;
        break;
      default:
        return std::nullopt;
    }
  }

  const Shape& output = scatter->shape();
  const Shape& base = scatter->operand(0)->shape();
  const Shape& indices = scatter->operand(1)->shape();
  const Shape& updates = scatter->operand(2)->shape();
  const int64_t output_rank = output.dimensions_size();
  const int64_t index_components =
      indices.dimensions_size() == 2 ? indices.dimensions(1) : 0;
  if (!output.IsArray() || !base.IsArray() || !indices.IsArray() ||
      !updates.IsArray() || !output.has_layout() || !base.has_layout() ||
      !indices.has_layout() || !updates.has_layout() || output_rank < 1 ||
      base.element_type() != output.element_type() ||
      updates.element_type() != output.element_type() ||
      output.element_type() == S4 || output.element_type() == PRED ||
      !IsSupportedValueType(output.element_type()) ||
      (indices.element_type() != S32 && indices.element_type() != S64) ||
      !ShapeUtil::Equal(base, output) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(indices.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(updates.layout()) ||
      indices.dimensions_size() != 2 || indices.dimensions(0) <= 0 ||
      index_components <= 0 || index_components > output_rank ||
      updates.dimensions_size() != output_rank + 1 ||
      updates.dimensions(0) != indices.dimensions(0) ||
      ShapeUtil::ElementsIn(output) <= 0 ||
      ShapeUtil::ElementsIn(updates) <= 0 ||
      ShapeUtil::ByteSizeOfElements(output) >
          std::numeric_limits<uint32_t>::max() ||
      ShapeUtil::ByteSizeOfElements(updates) >
          std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  // The universal atomic lowering maps these directly to signed LLVM
  // atomicrmw max/min. Keep the initial contract deliberately narrow until
  // floating NaN/signed-zero semantics and additional integer widths are
  // proven equivalent to HLO on every supported AMD target.
  if ((update_kind == ScatterUpdateKind::kMaximum ||
       update_kind == ScatterUpdateKind::kMinimum) &&
      output.element_type() != S32) {
    return std::nullopt;
  }
  for (int64_t dimension = 0; dimension < output_rank; ++dimension) {
    if (output.dimensions(dimension) <= 0) {
      return std::nullopt;
    }
    const int64_t update_size = updates.dimensions(dimension + 1);
    if (update_size <= 0 || update_size > output.dimensions(dimension)) {
      return std::nullopt;
    }
  }

  const ScatterDimensionNumbers& dimensions =
      scatter->scatter_dimension_numbers();
  if (dimensions.index_vector_dim() != 1 ||
      dimensions.update_window_dims_size() != output_rank ||
      dimensions.inserted_window_dims_size() != 0 ||
      dimensions.scatter_dims_to_operand_dims_size() != index_components ||
      dimensions.input_batching_dims_size() != 0 ||
      dimensions.scatter_indices_batching_dims_size() != 0) {
    return std::nullopt;
  }
  for (int64_t dimension = 0; dimension < output_rank; ++dimension) {
    if (dimensions.update_window_dims(dimension) != dimension + 1) {
      return std::nullopt;
    }
  }
  for (int64_t component = 0; component < index_components; ++component) {
    if (dimensions.scatter_dims_to_operand_dims(component) != component) {
      return std::nullopt;
    }
  }

  std::vector<int64_t> indexed_dimension_sizes;
  indexed_dimension_sizes.reserve(index_components);
  std::vector<int64_t> indexed_dimension_start_limits;
  indexed_dimension_start_limits.reserve(index_components);
  int64_t output_rows = 1;
  for (int64_t component = 0; component < index_components; ++component) {
    indexed_dimension_sizes.push_back(output.dimensions(component));
    indexed_dimension_start_limits.push_back(
        output.dimensions(component) - updates.dimensions(component + 1) + 1);
    output_rows *= output.dimensions(component);
  }
  const int64_t output_row_elements =
      ShapeUtil::ElementsIn(output) / output_rows;
  const int64_t row_elements =
      ShapeUtil::ElementsIn(updates) / indices.dimensions(0);
  if (output_row_elements <= 0 || row_elements <= 0 ||
      ShapeUtil::ElementsIn(updates) !=
          indices.dimensions(0) * row_elements) {
    return std::nullopt;
  }
  std::vector<int64_t> output_dimension_sizes(output.dimensions().begin(),
                                               output.dimensions().end());
  std::vector<int64_t> update_window_dimension_sizes;
  update_window_dimension_sizes.reserve(output_rank);
  int64_t first_non_unit_dimension = output_rank;
  for (int64_t dimension = 0; dimension < output_rank; ++dimension) {
    const int64_t update_size = updates.dimensions(dimension + 1);
    update_window_dimension_sizes.push_back(update_size);
    if (update_size != 1 && first_non_unit_dimension == output_rank) {
      first_non_unit_dimension = dimension;
    }
  }
  bool has_contiguous_window = true;
  for (int64_t dimension = first_non_unit_dimension + 1;
       dimension < output_rank; ++dimension) {
    if (update_window_dimension_sizes[dimension] !=
        output_dimension_sizes[dimension]) {
      has_contiguous_window = false;
      break;
    }
  }
  return OverwriteRowScatter{
      /*scatter=*/scatter,
      /*indexed_dimension_sizes=*/std::move(indexed_dimension_sizes),
      /*indexed_dimension_start_limits=*/
          std::move(indexed_dimension_start_limits),
      /*output_dimension_sizes=*/std::move(output_dimension_sizes),
      /*update_window_dimension_sizes=*/
          std::move(update_window_dimension_sizes),
      /*index_components=*/index_components,
      /*update_rows=*/indices.dimensions(0),
      /*output_row_elements=*/output_row_elements,
      /*row_elements=*/row_elements,
      /*contiguous_elements=*/updates.dimensions(output_rank),
      /*has_contiguous_window=*/has_contiguous_window,
      /*update_kind=*/update_kind,
      /*requires_atomic_operation=*/
          update_kind != ScatterUpdateKind::kOverwrite ||
              !scatter->unique_indices()};
}

bool IsSupportedGather(const HloInstruction* instruction) {
  return GetRowGather(instruction).has_value() ||
         GetStridedAxisGather(instruction).has_value();
}

std::optional<DynamicSlice> GetDynamicSlice(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDynamicSlice) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& output = instruction->shape();
  const int64_t rank = input.dimensions_size();
  if (!input.IsArray() || !output.IsArray() || !input.has_layout() ||
      !output.has_layout() || input.element_type() != output.element_type() ||
      rank == 0 || output.dimensions_size() != rank ||
      instruction->operand_count() != rank + 1 ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      ShapeUtil::ElementsIn(output) == 0) {
    return std::nullopt;
  }
  DynamicSlice slice;
  slice.input_strides.resize(rank);
  slice.output_dimensions.assign(output.dimensions().begin(),
                                 output.dimensions().end());
  slice.start_limits.resize(rank);
  int64_t input_stride = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    if (output.dimensions(dimension) <= 0 ||
        output.dimensions(dimension) > input.dimensions(dimension)) {
      return std::nullopt;
    }
    const HloInstruction* start = instruction->operand(dimension + 1);
    if (!ShapeUtil::IsScalar(start->shape()) ||
        (start->shape().element_type() != S32 &&
         start->shape().element_type() != S64)) {
      return std::nullopt;
    }
    slice.input_strides[dimension] = input_stride;
    slice.start_limits[dimension] =
        input.dimensions(dimension) - output.dimensions(dimension);
    input_stride *= input.dimensions(dimension);
  }
  return slice;
}

struct DynamicUpdateSlice {
  std::vector<int64_t> input_dimensions;
  std::vector<int64_t> update_dimensions;
  std::vector<int64_t> update_strides;
  std::vector<int64_t> start_limits;
};

std::optional<DynamicUpdateSlice> GetDynamicUpdateSlice(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDynamicUpdateSlice) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& update = instruction->operand(1)->shape();
  const Shape& output = instruction->shape();
  const int64_t rank = input.dimensions_size();
  if (!input.IsArray() || !update.IsArray() || !output.IsArray() ||
      !input.has_layout() || !update.has_layout() || !output.has_layout() ||
      input.element_type() != output.element_type() ||
      update.element_type() != output.element_type() || rank == 0 ||
      update.dimensions_size() != rank || output.dimensions_size() != rank ||
      instruction->operand_count() != rank + 2 ||
      !HasSamePhysicalDimensions(input, output) ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(update.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      ShapeUtil::ElementsIn(output) == 0 ||
      ShapeUtil::ElementsIn(update) == 0) {
    return std::nullopt;
  }
  DynamicUpdateSlice descriptor;
  descriptor.input_dimensions.assign(input.dimensions().begin(),
                                     input.dimensions().end());
  descriptor.update_dimensions.assign(update.dimensions().begin(),
                                      update.dimensions().end());
  descriptor.update_strides.resize(rank);
  descriptor.start_limits.resize(rank);
  int64_t update_stride = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    if (update.dimensions(dimension) > input.dimensions(dimension)) {
      return std::nullopt;
    }
    const HloInstruction* start = instruction->operand(dimension + 2);
    if (!ShapeUtil::IsScalar(start->shape()) ||
        (start->shape().element_type() != S32 &&
         start->shape().element_type() != S64)) {
      return std::nullopt;
    }
    descriptor.update_strides[dimension] = update_stride;
    descriptor.start_limits[dimension] =
        input.dimensions(dimension) - update.dimensions(dimension);
    update_stride *= update.dimensions(dimension);
  }
  return descriptor;
}

bool IsFlatReverse(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kReverse ||
      instruction->operand_count() != 1) {
    return false;
  }
  const Shape& shape = instruction->shape();
  if (!shape.IsArray() || !shape.has_layout() || shape.dimensions_size() == 0 ||
      instruction->operand(0)->shape().element_type() != shape.element_type() ||
      !HasSamePhysicalDimensions(instruction->operand(0)->shape(), shape) ||
      !LayoutUtil::IsMonotonicWithDim0Major(shape.layout()) ||
      instruction->dimensions().size() != shape.dimensions_size()) {
    return false;
  }
  for (int64_t dimension = 0; dimension < shape.dimensions_size();
       ++dimension) {
    if (instruction->dimensions(dimension) != dimension) {
      return false;
    }
  }
  return true;
}

struct PartialReverse {
  std::vector<int64_t> dimensions;
  std::vector<int64_t> strides;
  std::vector<bool> reversed_dimensions;
};

std::optional<PartialReverse> GetPartialReverse(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kReverse ||
      instruction->operand_count() != 1) {
    return std::nullopt;
  }
  const Shape& shape = instruction->shape();
  const int64_t rank = shape.dimensions_size();
  if (!shape.IsArray() || !shape.has_layout() || rank == 0 ||
      instruction->operand(0)->shape().element_type() != shape.element_type() ||
      !HasSamePhysicalDimensions(instruction->operand(0)->shape(), shape) ||
      !LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return std::nullopt;
  }
  PartialReverse reverse;
  reverse.dimensions.assign(shape.dimensions().begin(),
                            shape.dimensions().end());
  reverse.strides.resize(rank);
  reverse.reversed_dimensions.resize(rank, false);
  for (int64_t dimension : instruction->dimensions()) {
    if (dimension < 0 || dimension >= rank ||
        reverse.reversed_dimensions[dimension]) {
      return std::nullopt;
    }
    reverse.reversed_dimensions[dimension] = true;
  }
  int64_t stride = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    reverse.strides[dimension] = stride;
    stride *= shape.dimensions(dimension);
  }
  return reverse;
}

struct FlatPadInterval {
  int64_t input_begin;
  int64_t input_end;
};

std::optional<FlatPadInterval> GetFlatPadInterval(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kPad ||
      instruction->operand_count() != 2) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& output = instruction->shape();
  const PaddingConfig& padding = instruction->padding_config();
  const int64_t rank = input.dimensions_size();
  if (!input.IsArray() || !output.IsArray() || !input.has_layout() ||
      !output.has_layout() || input.element_type() != output.element_type() ||
      output.dimensions_size() != rank || padding.dimensions_size() != rank ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      ShapeUtil::ElementsIn(input) == 0 || ShapeUtil::ElementsIn(output) == 0) {
    return std::nullopt;
  }

  int64_t input_begin = 0;
  int64_t input_last = 0;
  int64_t output_stride = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    const auto& dimension_padding = padding.dimensions(dimension);
    const int64_t low = dimension_padding.edge_padding_low();
    const int64_t high = dimension_padding.edge_padding_high();
    if (low < 0 || high < 0 || dimension_padding.interior_padding() != 0 ||
        output.dimensions(dimension) !=
            low + input.dimensions(dimension) + high) {
      return std::nullopt;
    }
    input_begin += low * output_stride;
    input_last += (low + input.dimensions(dimension) - 1) * output_stride;
    output_stride *= output.dimensions(dimension);
  }
  const int64_t input_elements = ShapeUtil::ElementsIn(input);
  if (input_last - input_begin + 1 != input_elements) {
    return std::nullopt;
  }
  return FlatPadInterval{input_begin, input_begin + input_elements};
}

struct RectangularPad {
  std::vector<int64_t> input_dimensions;
  std::vector<int64_t> input_strides;
  std::vector<int64_t> output_dimensions;
  std::vector<int64_t> begins;
  std::vector<int64_t> span_ends;
  std::vector<int64_t> steps;
};

std::optional<RectangularPad> GetRectangularPad(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kPad ||
      instruction->operand_count() != 2) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& output = instruction->shape();
  const PaddingConfig& padding = instruction->padding_config();
  const int64_t rank = input.dimensions_size();
  if (!input.IsArray() || !output.IsArray() || !input.has_layout() ||
      !output.has_layout() || input.element_type() != output.element_type() ||
      rank == 0 || output.dimensions_size() != rank ||
      padding.dimensions_size() != rank ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      ShapeUtil::ElementsIn(input) == 0 || ShapeUtil::ElementsIn(output) == 0) {
    return std::nullopt;
  }
  RectangularPad pad;
  pad.input_dimensions.assign(input.dimensions().begin(),
                              input.dimensions().end());
  pad.output_dimensions.assign(output.dimensions().begin(),
                               output.dimensions().end());
  pad.input_strides.resize(rank);
  pad.begins.resize(rank);
  pad.span_ends.resize(rank);
  pad.steps.resize(rank);
  int64_t input_stride = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    const auto& config = padding.dimensions(dimension);
    const int64_t low = config.edge_padding_low();
    const int64_t high = config.edge_padding_high();
    const int64_t interior = config.interior_padding();
    const int64_t expected = low + input.dimensions(dimension) +
                             (input.dimensions(dimension) - 1) * interior +
                             high;
    if (interior < 0 || output.dimensions(dimension) != expected) {
      return std::nullopt;
    }
    pad.input_strides[dimension] = input_stride;
    pad.begins[dimension] = low;
    pad.steps[dimension] = interior + 1;
    pad.span_ends[dimension] =
        low + (input.dimensions(dimension) - 1) * (interior + 1) + 1;
    input_stride *= input.dimensions(dimension);
  }
  return pad;
}

struct ReduceWindowDescriptor {
  std::vector<int64_t> input_dimensions;
  std::vector<int64_t> input_strides;
  std::vector<int64_t> output_dimensions;
  std::vector<int64_t> window_sizes;
  std::vector<int64_t> window_strides;
  std::vector<int64_t> window_dilations;
  std::vector<int64_t> base_dilations;
  std::vector<int64_t> padding_low;
  int64_t window_elements = 0;
  HloOpcode reducer = HloOpcode::kAdd;
};

std::optional<ReduceWindowDescriptor> GetReduceWindowDescriptor(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kReduceWindow ||
      instruction->operand_count() != 2) {
    return std::nullopt;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& init = instruction->operand(1)->shape();
  const Shape& output = instruction->shape();
  const int64_t rank = input.dimensions_size();
  const Window& window =
      Cast<const HloReduceWindowInstruction>(instruction)->window();
  if (!input.IsArray() || !output.IsArray() || !input.has_layout() ||
      !output.has_layout() || rank == 0 || output.dimensions_size() != rank ||
      window.dimensions_size() != rank ||
      input.element_type() != output.element_type() ||
      !IsSupportedType(output.element_type()) || !ShapeUtil::IsScalar(init) ||
      init.element_type() != output.element_type() ||
      instruction->operand(1)->opcode() != HloOpcode::kConstant ||
      !LayoutUtil::IsMonotonicWithDim0Major(input.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(output.layout()) ||
      ShapeUtil::ElementsIn(input) == 0 || ShapeUtil::ElementsIn(output) == 0) {
    return std::nullopt;
  }

  const HloComputation* reducer = instruction->to_apply();
  if (reducer == nullptr || reducer->num_parameters() != 2 ||
      reducer->root_instruction()->operand_count() != 2 ||
      !ShapeUtil::IsScalar(reducer->root_instruction()->shape()) ||
      reducer->root_instruction()->shape().element_type() !=
          output.element_type()) {
    return std::nullopt;
  }
  const HloOpcode reducer_opcode = reducer->root_instruction()->opcode();
  if (reducer_opcode != HloOpcode::kAdd &&
      reducer_opcode != HloOpcode::kMaximum &&
      reducer_opcode != HloOpcode::kMinimum) {
    return std::nullopt;
  }

  ReduceWindowDescriptor descriptor;
  descriptor.input_dimensions.assign(input.dimensions().begin(),
                                     input.dimensions().end());
  descriptor.output_dimensions.assign(output.dimensions().begin(),
                                      output.dimensions().end());
  descriptor.input_strides.resize(rank);
  descriptor.window_sizes.resize(rank);
  descriptor.window_strides.resize(rank);
  descriptor.window_dilations.resize(rank);
  descriptor.base_dilations.resize(rank);
  descriptor.padding_low.resize(rank);
  descriptor.reducer = reducer_opcode;
  descriptor.window_elements = 1;
  int64_t input_stride = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    const WindowDimension& window_dimension = window.dimensions(dimension);
    if (window_dimension.size() <= 0 || window_dimension.stride() <= 0 ||
        window_dimension.window_dilation() <= 0 ||
        window_dimension.base_dilation() <= 0 ||
        descriptor.window_elements > 128 / window_dimension.size()) {
      return std::nullopt;
    }
    descriptor.input_strides[dimension] = input_stride;
    descriptor.window_sizes[dimension] = window_dimension.size();
    descriptor.window_strides[dimension] = window_dimension.stride();
    descriptor.window_dilations[dimension] = window_dimension.window_dilation();
    descriptor.base_dilations[dimension] = window_dimension.base_dilation();
    descriptor.padding_low[dimension] = window_dimension.padding_low();
    descriptor.window_elements *= window_dimension.size();
    input_stride *= input.dimensions(dimension);
  }
  return descriptor;
}

struct StridedReductionDescriptor {
  int64_t reduction_size;
  int64_t inner_elements;
  int64_t input_parameter_number;
  PrimitiveType type;
  HloOpcode reducer;
  const HloComputation* reducer_computation;
  bool direct_binary_reducer;
  double floating_init_value;
  int64_t integer_init_value;
};

bool IsSupportedElementwiseGraph(
    const HloInstruction* instruction, const Shape& output_shape,
    absl::flat_hash_set<const HloInstruction*>& visited);

std::optional<double> GetScalarConstantAsDouble(
    const HloInstruction* instruction) {
  if (!ShapeUtil::IsScalar(instruction->shape())) {
    return std::nullopt;
  }
  if (instruction->opcode() == HloOpcode::kConstant) {
    if (IsSupportedFloatingType(instruction->shape().element_type())) {
      return instruction->literal().GetAsDouble({});
    }
    switch (instruction->shape().element_type()) {
      case S8:
        return instruction->literal().GetFirstElement<int8_t>();
      case S16:
        return instruction->literal().GetFirstElement<int16_t>();
      case S32:
        return instruction->literal().GetFirstElement<int32_t>();
      default:
        return std::nullopt;
    }
  }
  if (instruction->opcode() == HloOpcode::kConvert &&
      instruction->operand_count() == 1) {
    return GetScalarConstantAsDouble(instruction->operand(0));
  }
  return std::nullopt;
}

bool IsSplatGraph(const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
      return ShapeUtil::IsScalar(instruction->shape());
    case HloOpcode::kBroadcast:
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kCopy:
      return instruction->operand_count() == 1 &&
             IsSplatGraph(instruction->operand(0));
    case HloOpcode::kReduce:
      return instruction->operand_count() == 2 &&
             IsSplatGraph(instruction->operand(0)) &&
             IsSplatGraph(instruction->operand(1));
    default:
      if (!instruction->IsElementwise() || instruction->operand_count() == 0) {
        return false;
      }
      return absl::c_all_of(instruction->operands(), IsSplatGraph);
  }
}

std::optional<StridedReductionDescriptor> GetStridedReductionDescriptor(
    const HloInstruction* instruction, const Shape& output_shape) {
  if (instruction->opcode() != HloOpcode::kReduce ||
      instruction->operand_count() != 2 ||
      !HasSameFlatElements(instruction->shape(), output_shape) ||
      instruction->dimensions().size() != 1) {
    return std::nullopt;
  }
  const HloInstruction* input = instruction->operand(0);
  const HloInstruction* init = instruction->operand(1);
  const PrimitiveType reduction_type = instruction->shape().element_type();
  const int64_t input_rank = input->shape().dimensions_size();
  const int64_t reduction_dimension = instruction->dimensions(0);
  if ((reduction_type != F32 && reduction_type != S32) ||
      input->shape().element_type() != reduction_type ||
      input_rank != instruction->shape().dimensions_size() + 1 ||
      reduction_dimension < 0 || reduction_dimension >= input_rank ||
      input->shape().dimensions(reduction_dimension) < 1 ||
      !input->shape().has_layout() || !instruction->shape().has_layout() ||
      !LayoutUtil::IsMonotonicWithDim0Major(input->shape().layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(instruction->shape().layout()) ||
      !ShapeUtil::IsScalar(init->shape()) ||
      init->shape().element_type() != reduction_type) {
    return std::nullopt;
  }
  const std::optional<double> init_value = GetScalarConstantAsDouble(init);
  if (!init_value.has_value()) {
    return std::nullopt;
  }
  for (int64_t input_dimension = 0, output_dimension = 0;
       input_dimension < input_rank; ++input_dimension) {
    if (input_dimension == reduction_dimension) {
      continue;
    }
    if (input->shape().dimensions(input_dimension) !=
        instruction->shape().dimensions(output_dimension++)) {
      return std::nullopt;
    }
  }
  const HloComputation* reducer = instruction->to_apply();
  if (reducer == nullptr || reducer->num_parameters() != 2 ||
      !ShapeUtil::IsScalar(reducer->root_instruction()->shape()) ||
      reducer->root_instruction()->shape().element_type() != reduction_type) {
    return std::nullopt;
  }
  const HloInstruction* reducer_root = reducer->root_instruction();
  const HloOpcode reducer_opcode = reducer->root_instruction()->opcode();
  const bool direct_binary_reducer =
      reducer_root->operand_count() == 2 &&
      reducer_root->operand(0)->opcode() == HloOpcode::kParameter &&
      reducer_root->operand(1)->opcode() == HloOpcode::kParameter &&
      reducer_root->operand(0)->parameter_number() !=
          reducer_root->operand(1)->parameter_number() &&
      (reducer_opcode == HloOpcode::kAdd ||
       (reduction_type == F32 &&
        (reducer_opcode == HloOpcode::kMaximum ||
         reducer_opcode == HloOpcode::kMinimum)));
  if (!direct_binary_reducer) {
    // General reducer DAGs are evaluated in source order below. Keep them on
    // the straight-line path so non-associative reducers are never silently
    // reassociated by the eight-way unrolled reduction schedule.
    if (reduction_type != F32 ||
        input->shape().dimensions(reduction_dimension) > 32) {
      return std::nullopt;
    }
    absl::flat_hash_set<const HloInstruction*> reducer_visited;
    if (!IsSupportedElementwiseGraph(reducer_root, reducer_root->shape(),
                                     reducer_visited)) {
      return std::nullopt;
    }
  }
  if (reduction_dimension == input_rank - 1 &&
      instruction->user_count() == 0 && !IsSplatGraph(input) &&
      direct_binary_reducer) {
    return std::nullopt;
  }
  int64_t inner_elements = 1;
  for (int64_t dimension = reduction_dimension + 1; dimension < input_rank;
       ++dimension) {
    inner_elements *= input->shape().dimensions(dimension);
  }
  const HloInstruction* parameter_view = input;
  while (parameter_view->opcode() != HloOpcode::kParameter &&
         parameter_view->operand_count() == 1) {
    parameter_view = parameter_view->operand(0);
  }
  const int64_t parameter_number =
      parameter_view->opcode() == HloOpcode::kParameter
          ? parameter_view->parameter_number()
          : -1;
  return StridedReductionDescriptor{
      input->shape().dimensions(reduction_dimension), inner_elements,
      parameter_number, reduction_type, reducer_opcode, reducer,
      direct_binary_reducer, reduction_type == F32 ? *init_value : 0.0,
      reduction_type == S32 ? static_cast<int64_t>(*init_value) : 0};
}

bool IsStripedLoopInvariantBroadcast(const HloInstruction* instruction,
                                     int64_t thread_stride,
                                     int64_t vectors_per_thread) {
  if (instruction->opcode() != HloOpcode::kBroadcast ||
      instruction->operand_count() != 1 ||
      ShapeUtil::IsScalar(instruction->operand(0)->shape()) ||
      vectors_per_thread <= 1) {
    return false;
  }
  const Shape& input = instruction->operand(0)->shape();
  const Shape& output = instruction->shape();
  const int64_t input_rank = input.dimensions_size();
  const int64_t output_rank = output.dimensions_size();
  if (output_rank != input_rank + 1 ||
      instruction->dimensions().size() != input_rank) {
    return false;
  }

  int64_t omitted_dimension = -1;
  for (int64_t output_dimension = 0, input_dimension = 0;
       output_dimension < output_rank; ++output_dimension) {
    if (input_dimension < input_rank &&
        instruction->dimensions(input_dimension) == output_dimension) {
      ++input_dimension;
      continue;
    }
    if (omitted_dimension != -1) {
      return false;
    }
    omitted_dimension = output_dimension;
  }
  if (omitted_dimension == -1) {
    return false;
  }

  int64_t inner_elements = 1;
  for (int64_t dimension = omitted_dimension + 1; dimension < output_rank;
       ++dimension) {
    inner_elements *= output.dimensions(dimension);
  }
  if (thread_stride % inner_elements != 0) {
    return false;
  }
  const int64_t omitted_step = thread_stride / inner_elements;
  const int64_t striped_span = omitted_step * vectors_per_thread;
  return omitted_step > 0 &&
         output.dimensions(omitted_dimension) % striped_span == 0;
}

bool IsSupportedElementwiseGraph(
    const HloInstruction* instruction, const Shape& output_shape,
    absl::flat_hash_set<const HloInstruction*>& visited) {
  if (!visited.insert(instruction).second) {
    return true;
  }
  switch (instruction->opcode()) {
    case HloOpcode::kIota:
      return instruction->operand_count() == 0 &&
             IsSupportedType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             IotaPhysicalStride(instruction).has_value();
    case HloOpcode::kParameter:
      return IsSupportedExternalType(instruction->shape().element_type()) &&
             (instruction->shape().element_type() == S4 ||
                      output_shape.element_type() == S4
                  ? HasSamePackedPhysicalDimensions(instruction->shape(),
                                                    output_shape)
                  : HasSamePhysicalDimensions(instruction->shape(),
                                              output_shape));
    case HloOpcode::kConstant:
      return ShapeUtil::ElementsIn(instruction->shape()) == 1 &&
             IsSupportedValueType(instruction->shape().element_type());
    case HloOpcode::kBroadcast:
      if (instruction->operand_count() != 1 ||
          !IsSupportedValueType(instruction->shape().element_type()) ||
          !HasSamePhysicalDimensions(instruction->shape(), output_shape)) {
        return false;
      }
      if (ShapeUtil::IsScalar(instruction->operand(0)->shape())) {
        return IsSupportedElementwiseGraph(
            instruction->operand(0), instruction->operand(0)->shape(), visited);
      }
      if (instruction->operand(0)->shape().element_type() !=
              instruction->shape().element_type() ||
          !instruction->operand(0)->shape().has_layout() ||
          !LayoutUtil::IsMonotonicWithDim0Major(
              instruction->operand(0)->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(
              instruction->shape().layout()) ||
          instruction->operand(0)->shape().dimensions_size() >=
              instruction->shape().dimensions_size() ||
          ShapeUtil::ElementsIn(instruction->operand(0)->shape()) == 0) {
        return false;
      }
      {
        const int64_t input_rank =
            instruction->operand(0)->shape().dimensions_size();
        if (instruction->dimensions().size() != input_rank) {
          return false;
        }
        int64_t previous_output_dimension = -1;
        for (int64_t input_dimension = 0; input_dimension < input_rank;
             ++input_dimension) {
          const int64_t output_dimension =
              instruction->dimensions(input_dimension);
          if (output_dimension <= previous_output_dimension ||
              output_dimension >= instruction->shape().dimensions_size() ||
              instruction->operand(0)->shape().dimensions(input_dimension) !=
                  instruction->shape().dimensions(output_dimension)) {
            return false;
          }
          previous_output_dimension = output_dimension;
        }
      }
      return IsSupportedElementwiseGraph(
          instruction->operand(0), instruction->operand(0)->shape(), visited);
    case HloOpcode::kConvert:
      return instruction->operand_count() == 1 &&
             IsSupportedConversion(
                 instruction->operand(0)->shape().element_type(),
                 instruction->shape().element_type()) &&
             HasSameFlatElements(instruction->shape(), output_shape) &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kReshape:
      return IsSupportedPhysicalView(instruction) &&
             HasSameFlatElements(instruction->shape(), output_shape) &&
             IsSupportedElementwiseGraph(instruction->operand(0),
                                         instruction->operand(0)->shape(),
                                         visited);
    case HloOpcode::kTranspose:
      return (IsSupportedPhysicalView(instruction) ||
              GetDataMovingTranspose(instruction).has_value()) &&
             HasSameFlatElements(instruction->shape(), output_shape) &&
             IsSupportedElementwiseGraph(instruction->operand(0),
                                         instruction->operand(0)->shape(),
                                         visited);
    case HloOpcode::kSlice:
      return IsSupportedValueType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             (ContiguousSliceBase(instruction).has_value() ||
              GetRectangularSlice(instruction).has_value()) &&
             IsSupportedElementwiseGraph(instruction->operand(0),
                                         instruction->operand(0)->shape(),
                                         visited);
    case HloOpcode::kDynamicSlice: {
      if (!IsSupportedValueType(instruction->shape().element_type()) ||
          !HasSameFlatElements(instruction->shape(), output_shape) ||
          !GetDynamicSlice(instruction).has_value() ||
          !IsSupportedElementwiseGraph(instruction->operand(0),
                                       instruction->operand(0)->shape(),
                                       visited)) {
        return false;
      }
      for (int64_t operand = 1; operand < instruction->operand_count();
           ++operand) {
        if (!IsSupportedElementwiseGraph(
                instruction->operand(operand),
                instruction->operand(operand)->shape(), visited)) {
          return false;
        }
      }
      return true;
    }
    case HloOpcode::kGather:
      return HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             IsSupportedGather(instruction) &&
             IsSupportedElementwiseGraph(instruction->operand(0),
                                         instruction->operand(0)->shape(),
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1),
                                         instruction->operand(1)->shape(),
                                         visited);
    case HloOpcode::kDynamicUpdateSlice: {
      if (!IsSupportedType(instruction->shape().element_type()) ||
          !HasSameFlatElements(instruction->shape(), output_shape) ||
          !GetDynamicUpdateSlice(instruction).has_value() ||
          !IsSupportedElementwiseGraph(instruction->operand(0),
                                       instruction->operand(0)->shape(),
                                       visited) ||
          !IsSupportedElementwiseGraph(instruction->operand(1),
                                       instruction->operand(1)->shape(),
                                       visited)) {
        return false;
      }
      for (int64_t operand = 2; operand < instruction->operand_count();
           ++operand) {
        if (!IsSupportedElementwiseGraph(
                instruction->operand(operand),
                instruction->operand(operand)->shape(), visited)) {
          return false;
        }
      }
      return true;
    }
    case HloOpcode::kReverse:
      return IsSupportedValueType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             (IsFlatReverse(instruction) ||
              GetPartialReverse(instruction).has_value()) &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kPad:
      return IsSupportedValueType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             (GetFlatPadInterval(instruction).has_value() ||
              GetRectangularPad(instruction).has_value()) &&
             ShapeUtil::IsScalar(instruction->operand(1)->shape()) &&
             instruction->operand(1)->opcode() == HloOpcode::kConstant &&
             instruction->operand(1)->shape().element_type() ==
                 instruction->shape().element_type() &&
             IsSupportedElementwiseGraph(instruction->operand(0),
                                         instruction->operand(0)->shape(),
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited);
    case HloOpcode::kConcatenate: {
      const int64_t rank = instruction->shape().dimensions_size();
      if (rank == 0 || instruction->dimensions().size() != 1 ||
          instruction->dimensions(0) < 0 ||
          instruction->dimensions(0) >= rank ||
          !IsSupportedValueType(instruction->shape().element_type()) ||
          !HasSamePhysicalDimensions(instruction->shape(), output_shape) ||
          instruction->operand_count() < 2) {
        return false;
      }
      const int64_t concatenate_dimension = instruction->dimensions(0);
      int64_t concatenated_extent = 0;
      for (const HloInstruction* operand : instruction->operands()) {
        if (operand->shape().dimensions_size() != rank ||
            operand->shape().element_type() !=
                instruction->shape().element_type() ||
            !operand->shape().has_layout() ||
            !LayoutUtil::IsMonotonicWithDim0Major(operand->shape().layout()) ||
            !IsSupportedElementwiseGraph(operand, operand->shape(), visited)) {
          return false;
        }
        for (int64_t dimension = 0; dimension < rank; ++dimension) {
          if (dimension != concatenate_dimension &&
              operand->shape().dimensions(dimension) !=
                  instruction->shape().dimensions(dimension)) {
            return false;
          }
        }
        concatenated_extent +=
            operand->shape().dimensions(concatenate_dimension);
      }
      return concatenated_extent ==
             instruction->shape().dimensions(concatenate_dimension);
    }
    case HloOpcode::kReduce:
      return GetStridedReductionDescriptor(instruction, output_shape)
                 .has_value() &&
             IsSupportedElementwiseGraph(instruction->operand(0),
                                         instruction->operand(0)->shape(),
                                         visited);
    case HloOpcode::kReduceWindow:
      return HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             GetReduceWindowDescriptor(instruction).has_value() &&
             IsSupportedElementwiseGraph(instruction->operand(0),
                                         instruction->operand(0)->shape(),
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited);
    case HloOpcode::kCopy:
      return IsSupportedValueType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 1 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kAbs:
      return (IsSupportedType(instruction->shape().element_type()) ||
              IsFp8Type(instruction->shape().element_type())) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 1 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kNegate:
      return IsSupportedType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 1 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kNot:
      return (IsSupportedSignedIntegerType(
                  instruction->shape().element_type()) ||
              instruction->shape().element_type() == PRED) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 1 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kAcos:
    case HloOpcode::kAcosh:
    case HloOpcode::kAsin:
    case HloOpcode::kAsinh:
    case HloOpcode::kAtanh:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kCos:
    case HloOpcode::kCosh:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kSinh:
    case HloOpcode::kSqrt:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
      return IsSupportedFloatingType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 1 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited);
    case HloOpcode::kReducePrecision:
      return IsFloatingValueType(instruction->shape().element_type()) &&
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
    case HloOpcode::kRemainder:
      return (IsSupportedType(instruction->shape().element_type()) ||
              ((instruction->opcode() == HloOpcode::kAdd ||
                instruction->opcode() == HloOpcode::kMultiply ||
                instruction->opcode() == HloOpcode::kMaximum ||
                instruction->opcode() == HloOpcode::kMinimum) &&
               instruction->shape().element_type() == PRED)) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 2 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited);
    case HloOpcode::kAtan2:
    case HloOpcode::kPower:
      return IsSupportedFloatingType(instruction->shape().element_type()) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 2 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited);
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
      return (IsSupportedSignedIntegerType(
                  instruction->shape().element_type()) ||
              instruction->shape().element_type() == PRED) &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             instruction->operand_count() == 2 &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited);
    case HloOpcode::kClamp:
      return (IsSupportedType(instruction->shape().element_type()) ||
              instruction->shape().element_type() == PRED) &&
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
             (Cast<const HloCompareInstruction>(instruction)->type() ==
                  Comparison::Type::kFloat ||
              Cast<const HloCompareInstruction>(instruction)->type() ==
                  Comparison::Type::kSigned ||
              (instruction->operand(0)->shape().element_type() == PRED &&
               Cast<const HloCompareInstruction>(instruction)->type() ==
                   Comparison::Type::kUnsigned)) &&
             IsSupportedElementwiseGraph(instruction->operand(0), output_shape,
                                         visited) &&
             IsSupportedElementwiseGraph(instruction->operand(1), output_shape,
                                         visited);
    case HloOpcode::kSelect:
      return (IsSupportedType(instruction->shape().element_type()) ||
              IsFp8Type(instruction->shape().element_type()) ||
              instruction->shape().element_type() == PRED) &&
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

bool HasSupportedS4InputGraph(const HloFusionAnalysis& analysis) {
  bool has_s4_input = false;
  bool supported = true;
  bool has_offset_remapping = false;
  absl::flat_hash_set<const HloInstruction*> checked_s4_consumers;
  std::function<bool(const HloInstruction*)> has_supported_s4_consumers =
      [&](const HloInstruction* instruction) {
        if (!checked_s4_consumers.insert(instruction).second) {
          return true;
        }
        if (instruction->users().empty()) {
          return false;
        }
        for (const HloInstruction* user : instruction->users()) {
          if (user->opcode() == HloOpcode::kConvert &&
              user->operand_count() == 1 && user->operand(0) == instruction &&
              user->shape().element_type() != S4) {
            continue;
          }
          const bool is_s4_view =
              user->operand(0) == instruction &&
              user->shape().element_type() == S4 &&
              (((user->opcode() == HloOpcode::kBitcast ||
                 user->opcode() == HloOpcode::kReshape ||
                 user->opcode() == HloOpcode::kTranspose) &&
                IsSupportedPhysicalView(user)) ||
               (user->opcode() == HloOpcode::kSlice &&
                (ContiguousSliceBase(user).has_value() ||
                 GetRectangularSlice(user).has_value())) ||
               (user->opcode() == HloOpcode::kDynamicSlice &&
                GetDynamicSlice(user).has_value()) ||
               (user->opcode() == HloOpcode::kReverse &&
                (IsFlatReverse(user) || GetPartialReverse(user).has_value())));
          if (!is_s4_view || !has_supported_s4_consumers(user)) {
            return false;
          }
        }
        return true;
      };
  absl::flat_hash_set<const HloInstruction*> visited;
  std::function<void(const HloInstruction*)> visit =
      [&](const HloInstruction* instruction) {
        if (!visited.insert(instruction).second) {
          return;
        }
        if (IsS4Type(instruction->shape().element_type()) &&
            instruction->opcode() == HloOpcode::kParameter) {
          has_s4_input = true;
          supported &= has_supported_s4_consumers(instruction);
        }
        switch (instruction->opcode()) {
          case HloOpcode::kReduce:
            has_offset_remapping = true;
            break;
          default:
            break;
        }
        for (const HloInstruction* operand : instruction->operands()) {
          visit(operand);
        }
      };
  for (int64_t root_index = 0; root_index < analysis.fusion_root_count();
       ++root_index) {
    visit(&analysis.fusion_root(root_index).instruction());
  }
  if (!has_s4_input) {
    return true;
  }
  // Direct elementwise consumers preserve the logical S4 offset. Bitcasts,
  // bitcast reshapes, and effective transposes preserve the physical flat
  // offset, while static and dynamic slices are handled by the nibble-aware
  // packed loader below.
  return supported && !has_offset_remapping;
}

bool HasPotentiallyOddS4Offset(const HloFusionAnalysis& analysis) {
  absl::flat_hash_map<const HloInstruction*, bool> depends_on_s4;
  std::function<bool(const HloInstruction*)> depends =
      [&](const HloInstruction* instruction) {
        auto [it, inserted] = depends_on_s4.try_emplace(instruction, false);
        if (!inserted) {
          return it->second;
        }
        bool result = instruction->opcode() == HloOpcode::kParameter &&
                      instruction->shape().element_type() == S4;
        for (const HloInstruction* operand : instruction->operands()) {
          result |= depends(operand);
        }
        // Recursive insertions can rehash a flat_hash_map and invalidate `it`.
        // Look the instruction up again before publishing the memoized result.
        // Large fusion DAGs otherwise corrupt the allocator while constructing
        // the native elementwise emitter.
        depends_on_s4[instruction] = result;
        return result;
      };
  absl::flat_hash_set<const HloInstruction*> visited;
  std::function<bool(const HloInstruction*)> visit =
      [&](const HloInstruction* instruction) {
        if (!visited.insert(instruction).second) {
          return false;
        }
        const bool remaps_offsets =
            instruction->opcode() == HloOpcode::kBitcastConvert ||
            instruction->opcode() == HloOpcode::kSlice ||
            instruction->opcode() == HloOpcode::kDynamicSlice ||
            instruction->opcode() == HloOpcode::kDynamicUpdateSlice ||
            instruction->opcode() == HloOpcode::kReverse ||
            instruction->opcode() == HloOpcode::kPad ||
            instruction->opcode() == HloOpcode::kConcatenate ||
            instruction->opcode() == HloOpcode::kReduceWindow ||
            (instruction->opcode() == HloOpcode::kBroadcast &&
             !ShapeUtil::IsScalar(instruction->operand(0)->shape()));
        if (remaps_offsets && depends(instruction)) {
          return true;
        }
        for (const HloInstruction* operand : instruction->operands()) {
          if (visit(operand)) {
            return true;
          }
        }
        return false;
      };
  for (int64_t root_index = 0; root_index < analysis.fusion_root_count();
       ++root_index) {
    if (visit(&analysis.fusion_root(root_index).instruction())) {
      return true;
    }
  }
  return false;
}

std::vector<int64_t> ScalarIndexParameterNumbers(
    const HloFusionAdaptor& fusion) {
  absl::flat_hash_set<const HloInstruction*> index_nodes;
  std::function<void(const HloInstruction*)> collect =
      [&](const HloInstruction* instruction) {
        if (!index_nodes.insert(instruction).second) {
          return;
        }
        for (const HloInstruction* operand : instruction->operands()) {
          collect(operand);
        }
      };
  fusion.ForEach([&](HloInstructionAdaptor adaptor) {
    const HloInstruction* instruction = &adaptor.instruction();
    if (instruction->opcode() == HloOpcode::kGather &&
        instruction->operand_count() == 2 &&
        IsSupportedGather(instruction)) {
      collect(instruction->operand(1));
    } else if (GetOverwriteRowScatter(instruction).has_value()) {
      collect(instruction->operand(1));
    }
  });

  auto is_index_only = [&](const HloInstruction* parameter) {
    absl::flat_hash_set<const HloInstruction*> visited;
    std::function<bool(const HloInstruction*)> visit_users =
        [&](const HloInstruction* instruction) {
          if (!visited.insert(instruction).second) {
            return true;
          }
          for (const HloInstruction* user : instruction->users()) {
            if (user->opcode() == HloOpcode::kGather) {
              if (user->operand_count() != 2 || user->operand(1) != instruction ||
                  !IsSupportedGather(user)) {
                return false;
              }
              continue;
            }
            if (user->opcode() == HloOpcode::kScatter) {
              if (user->operand_count() != 3 || user->operand(1) != instruction ||
                  !GetOverwriteRowScatter(user).has_value()) {
                return false;
              }
              continue;
            }
            if (!index_nodes.contains(user) || !visit_users(user)) {
              return false;
            }
          }
          return !instruction->users().empty();
        };
    return index_nodes.contains(parameter) && visit_users(parameter);
  };

  std::vector<int64_t> parameter_numbers;
  // HloFusionAdaptor intentionally hides fusion parameters from ForEach().
  // They are still present in the original gather-index expression, so find
  // them in the index DAG collected above.
  for (const HloInstruction* parameter : index_nodes) {
    if (parameter->opcode() == HloOpcode::kParameter &&
        is_index_only(parameter)) {
      parameter_numbers.push_back(parameter->parameter_number());
    }
  }
  std::sort(parameter_numbers.begin(), parameter_numbers.end());
  return parameter_numbers;
}

class FlyXTileElementwiseEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileElementwiseEmitter(const HloFusionAnalysis& analysis)
      : elements_(GetFlyXTileElementwiseElementCount(analysis)),
        element_type_(analysis.first_result_shape().element_type()),
        output_count_(analysis.fusion_root_count()),
        scalar_index_parameter_numbers_(
            GetFlyXTileScalarIndexParameterNumbers(analysis)),
        s4_may_start_odd_(HasPotentiallyOddS4Offset(analysis)),
        cache_modifier_(GetFlyXTileMemoryPolicy(analysis) ==
                                FlyXTileMemoryPolicy::kNonTemporal
                            ? 2
                            : 0) {
    if (analysis.fusion_root_count() == 1) {
      overwrite_row_scatter_ = GetOverwriteRowScatter(
          &analysis.fusion_root(0).instruction());
    }
    output_elements_.reserve(output_count_);
    for (int64_t root_index = 0; root_index < output_count_; ++root_index) {
      // Overwrite scatters traverse their update domain, not every element of
      // the larger destination buffer. Preserve that specialized launch
      // domain so single-output scatter kernels stay on their existing path.
      output_elements_.push_back(overwrite_row_scatter_.has_value()
                                     ? elements_
                                     : ShapeUtil::ElementsIn(
                                           analysis.fusion_root(root_index)
                                               .shape()));
    }
    uniform_output_elements_ = absl::c_all_of(
        output_elements_,
        [&](int64_t output_elements) { return output_elements == elements_; });
    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    // Fly's elementwise emitter uses one shared flat output domain for every
    // root.  Fly's autotuner encodes that domain with one output tile, while
    // XLA's symbolic tiler records one equivalent tile per tuple result during
    // priority fusion.  Accept both representations; the generated kernel
    // already evaluates and stores every root in the same traversal.
    CHECK(config.output_tiles_size() == 1 ||
          config.output_tiles_size() == output_count_);
    CHECK_GT(config.output_tiles(0).sizes_size(), 0);
    vector_size_bits_ =
        config.vector_size_bits() == 0 ? 64 : config.vector_size_bits();
    CHECK(vector_size_bits_ == 16 || vector_size_bits_ == 32 ||
          vector_size_bits_ == 64 || vector_size_bits_ == 128);
    CHECK_EQ(vector_size_bits_ % ElementBits(element_type_), 0);
    vector_width_ = vector_size_bits_ / ElementBits(element_type_);
    vectors_per_thread_ = config.output_tiles(0).sizes(0);
    for (int64_t output = 1; output < config.output_tiles_size(); ++output) {
      CHECK_GT(config.output_tiles(output).sizes_size(), 0);
      CHECK_EQ(config.output_tiles(output).sizes(0), vectors_per_thread_);
    }
    threads_ = config.num_warps() * 64;
    CHECK_GT(vectors_per_thread_, 0);
    CHECK_GT(threads_, 0);
    const int64_t elements_per_block =
        threads_ * vectors_per_thread_ * vector_width_;
    if (output_count_ == 1 && !overwrite_row_scatter_.has_value()) {
      cooperative_strided_reduction_ =
          GetStridedReductionDescriptor(&analysis.fusion_root(0).instruction(),
                                        analysis.fusion_root(0).shape());
    }
    if (cooperative_strided_reduction_.has_value() && config.num_warps() == 4 &&
        vector_width_ == 4 && vectors_per_thread_ == 1 &&
        cooperative_strided_reduction_->input_parameter_number >= 0 &&
        cooperative_strided_reduction_->type == F32 &&
        cooperative_strided_reduction_->reducer == HloOpcode::kAdd &&
        cooperative_strided_reduction_->floating_init_value == 0.0 &&
        cooperative_strided_reduction_->reduction_size % 4 == 0 &&
        cooperative_strided_reduction_->inner_elements % (64 * vector_width_) ==
            0) {
      const int64_t output_elements_per_block = 64 * vector_width_;
      launch_dimensions_ = LaunchDimensions(
          se::BlockDim(elements_ / output_elements_per_block, 1, 1),
          se::ThreadDim(threads_, 1, 1));
      return;
    }
    cooperative_strided_reduction_.reset();
    launch_dimensions_ = LaunchDimensions(
        se::BlockDim((elements_ + elements_per_block - 1) / elements_per_block,
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
    module.get()->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                          module_builder.getUnitAttr());

    module_builder.setInsertionPointToStart(module->getBody());
    mlir::gpu::GPUModuleOp gpu_module = mlir::gpu::GPUModuleOp::create(
        module_builder, location, "fly_elementwise_kernels");
    module_builder.setInsertionPointToStart(
        &gpu_module.getBodyRegion().front());

    mlir::fly::AddressSpaceAttr global_address =
        mlir::fly::AddressSpaceAttr::get(&context,
                                         mlir::fly::AddressSpace::Global);
    llvm::SmallVector<mlir::Type> argument_types;
    argument_types.reserve(fusion.operand_count() + output_count_);
    for (const HloInstruction* operand : fusion.operands()) {
      argument_types.push_back(mlir::fly::PointerType::get(
          StorageElementType(operand->shape().element_type(), module_builder),
          global_address));
    }
    const HloInstruction* fusion_root = fusion.fused_expression_root();
    TF_RET_CHECK(fusion_root != nullptr);
    if (fusion_root->opcode() == HloOpcode::kTuple) {
      for (const HloInstruction* root : fusion_root->operands()) {
        argument_types.push_back(mlir::fly::PointerType::get(
            StorageElementType(root->shape().element_type(), module_builder),
            global_address));
      }
    } else {
      argument_types.push_back(mlir::fly::PointerType::get(
          StorageElementType(fusion_root->shape().element_type(),
                             module_builder),
          global_address));
    }
    TF_RET_CHECK(argument_types.size() ==
                 fusion.operand_count() + output_count_);
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

  Value SplatInteger(mlir::ImplicitLocOpBuilder& builder, mlir::VectorType type,
                     int64_t value) const {
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

  Value ClampFloatToS4(mlir::ImplicitLocOpBuilder& builder, Value value) const {
    auto source_type = mlir::cast<mlir::VectorType>(value.getType());
    auto result_type =
        mlir::VectorType::get(source_type.getShape(), builder.getI8Type());
    Value result = mlir::arith::FPToSIOp::create(builder, result_type, value);
    result = mlir::arith::SelectOp::create(
        builder,
        mlir::arith::CmpFOp::create(builder, mlir::arith::CmpFPredicate::OLE,
                                    value,
                                    SplatFloat(builder, source_type, -8)),
        SplatInteger(builder, result_type, -8), result);
    result = mlir::arith::SelectOp::create(
        builder,
        mlir::arith::CmpFOp::create(builder, mlir::arith::CmpFPredicate::OGE,
                                    value, SplatFloat(builder, source_type, 7)),
        SplatInteger(builder, result_type, 7), result);
    return mlir::arith::SelectOp::create(
        builder,
        mlir::arith::CmpFOp::create(builder, mlir::arith::CmpFPredicate::UNO,
                                    value, value),
        SplatInteger(builder, result_type, 0), result);
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

  Value SignedI8ToBf16(mlir::ImplicitLocOpBuilder& builder, Value value) const {
    auto input_type = mlir::cast<mlir::VectorType>(value.getType());
    auto f32_type =
        mlir::VectorType::get(input_type.getShape(), builder.getF32Type());
    auto i32_type =
        mlir::VectorType::get(input_type.getShape(), builder.getI32Type());
    auto i16_type =
        mlir::VectorType::get(input_type.getShape(), builder.getI16Type());
    auto bf16_type =
        mlir::VectorType::get(input_type.getShape(), builder.getBF16Type());
    Value converted = mlir::arith::SIToFPOp::create(builder, f32_type, value);
    Value bits = mlir::arith::BitcastOp::create(builder, i32_type, converted);
    bits = mlir::arith::ShRUIOp::create(builder, bits,
                                        SplatInteger(builder, i32_type, 16));
    Value high_half = mlir::arith::TruncIOp::create(builder, i16_type, bits);
    // Signed bytes are exactly representable in BF16. Extracting the high
    // half of their FP32 representation avoids LLVM's generic BF16 rounding
    // expansion, matching the optimized native Fly S4 GEMM path.
    return mlir::arith::BitcastOp::create(builder, bf16_type, high_half);
  }

  Value PackS4Values(mlir::ImplicitLocOpBuilder& builder,
                     Value logical_values) const {
    auto logical_type = mlir::cast<mlir::VectorType>(logical_values.getType());
    const int64_t logical_width = logical_type.getNumElements();
    CHECK(logical_type.getElementType().isInteger(8));
    CHECK_EQ(logical_width % 2, 0);
    auto packed_type = mlir::VectorType::get({logical_width / 2},
                                             logical_type.getElementType());
    llvm::SmallVector<int64_t> low_mask;
    llvm::SmallVector<int64_t> high_mask;
    low_mask.reserve(logical_width / 2);
    high_mask.reserve(logical_width / 2);
    for (int64_t element = 0; element < logical_width; element += 2) {
      low_mask.push_back(element);
      high_mask.push_back(element + 1);
    }
    Value low = mlir::vector::ShuffleOp::create(
        builder, packed_type, logical_values, logical_values, low_mask);
    Value high = mlir::vector::ShuffleOp::create(
        builder, packed_type, logical_values, logical_values, high_mask);
    Value nibble_mask = SplatInteger(builder, packed_type, 0xf);
    low = mlir::arith::AndIOp::create(builder, low, nibble_mask);
    high = mlir::arith::AndIOp::create(builder, high, nibble_mask);
    high = mlir::arith::ShLIOp::create(builder, high,
                                       SplatInteger(builder, packed_type, 4));
    return mlir::arith::OrIOp::create(builder, low, high);
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
        {vector_width}, destination_type == PRED
                            ? mlir::Type(builder.getI1Type())
                            : StorageElementType(destination_type, builder));
    if (destination_type == PRED) {
      if (IsSupportedFloatingType(source_type)) {
        return mlir::arith::CmpFOp::create(
                   builder, mlir::arith::CmpFPredicate::UNE, operand,
                   mlir::arith::ConstantOp::create(
                       builder, operand_vector_type,
                       mlir::DenseElementsAttr::get(
                           operand_vector_type,
                           builder.getFloatAttr(
                               operand_vector_type.getElementType(), 0.0))))
            .getResult();
      }
      return mlir::arith::CmpIOp::create(
                 builder, mlir::arith::CmpIPredicate::ne, operand,
                 mlir::arith::ConstantOp::create(
                     builder, operand_vector_type,
                     mlir::DenseElementsAttr::get(
                         operand_vector_type,
                         builder.getIntegerAttr(
                             operand_vector_type.getElementType(), 0))))
          .getResult();
    }
    if (source_type == PRED) {
      if (IsSupportedFloatingType(destination_type)) {
        return mlir::arith::UIToFPOp::create(builder, destination_vector_type,
                                             operand)
            .getResult();
      }
      return mlir::arith::ExtUIOp::create(builder, destination_vector_type,
                                          operand)
          .getResult();
    }
    if (destination_type == S4) {
      if (IsSupportedFloatingType(source_type)) {
        return ClampFloatToS4(builder, operand);
      }
      TF_RET_CHECK(IsSupportedSignedIntegerType(source_type));
      TF_RET_CHECK(operand_vector_type.getElementType().isInteger());
      if (operand_vector_type.getElementType().getIntOrFloatBitWidth() == 8) {
        return operand;
      }
      return mlir::arith::TruncIOp::create(builder, destination_vector_type,
                                           operand)
          .getResult();
    }
    if (source_type == S4) {
      TF_RET_CHECK(operand_vector_type.getElementType().isInteger(8));
      if (destination_type == S8) {
        return operand;
      }
      if (IsSupportedSignedIntegerType(destination_type)) {
        return mlir::arith::ExtSIOp::create(builder, destination_vector_type,
                                            operand)
            .getResult();
      }
      if (destination_type == BF16) {
        return SignedI8ToBf16(builder, operand);
      }
      auto f32_type =
          mlir::VectorType::get({vector_width}, builder.getF32Type());
      Value f32 = mlir::arith::SIToFPOp::create(builder, f32_type, operand);
      if (destination_type == F32) {
        return f32;
      }
      if (destination_type == F16) {
        return mlir::arith::TruncFOp::create(builder, destination_vector_type,
                                             f32)
            .getResult();
      }
      return mlir::arith::ExtFOp::create(builder, destination_vector_type, f32)
          .getResult();
    }
    if (source_type == S8 && destination_type == BF16) {
      return SignedI8ToBf16(builder, operand);
    }
    if (IsFp8Type(source_type) || IsFp8Type(destination_type)) {
      const bool can_vectorize_fp8 =
          vector_width > 1 && llvm::isPowerOf2_64(vector_width);
      Value logical_operand = operand;
      if (IsFp8Type(source_type) && can_vectorize_fp8) {
        auto logical_source_vector_type = mlir::VectorType::get(
            {vector_width},
            emitters::PrimitiveTypeToMlirType(source_type, builder));
        logical_operand = mlir::arith::BitcastOp::create(
            builder, logical_source_vector_type, operand);
      }
      llvm::SmallVector<Value> converted;
      converted.reserve(vector_width);
      mlir::Type logical_destination_type =
          IsFp8Type(destination_type)
              ? emitters::PrimitiveTypeToMlirType(destination_type, builder)
              : destination_vector_type.getElementType();
      for (int64_t lane = 0; lane < vector_width; ++lane) {
        Value value =
            mlir::vector::ExtractOp::create(builder, logical_operand, lane);
        if (IsFp8Type(source_type) && !can_vectorize_fp8) {
          value = mlir::arith::BitcastOp::create(
              builder, emitters::PrimitiveTypeToMlirType(source_type, builder),
              value);
        }
        if (IsFp8Type(source_type) && IsFp8Type(destination_type)) {
          value =
              mlir::arith::ExtFOp::create(builder, builder.getF32Type(), value);
          value = mlir::arith::TruncFOp::create(
              builder, logical_destination_type, value);
        } else if (IsFp8Type(source_type)) {
          value = mlir::arith::ExtFOp::create(builder, logical_destination_type,
                                              value);
        } else {
          value = mlir::arith::TruncFOp::create(
              builder, logical_destination_type, value);
        }
        if (IsFp8Type(destination_type) && !can_vectorize_fp8) {
          value = mlir::arith::BitcastOp::create(builder, builder.getI8Type(),
                                                 value);
        }
        converted.push_back(value);
      }

      // Keep the logical FP8 vector intact until XLA's AMD float-conversion
      // pass has seen the complete extract/extend and truncate/insert chains.
      // It can then lower adjacent lanes with the two-result gfx942 packed
      // conversion intrinsics. Byte backing is restored immediately after the
      // logical conversion, before generic GPU type conversion.
      mlir::VectorType assembled_type = destination_vector_type;
      if (IsFp8Type(destination_type) && can_vectorize_fp8) {
        assembled_type =
            mlir::VectorType::get({vector_width}, logical_destination_type);
      }
      Value result = mlir::ub::PoisonOp::create(builder, assembled_type);
      for (auto [lane, value] : llvm::enumerate(converted)) {
        result = mlir::vector::InsertOp::create(builder, value, result, lane);
      }
      if (IsFp8Type(destination_type) && can_vectorize_fp8) {
        result = mlir::arith::BitcastOp::create(
            builder, destination_vector_type, result);
      }
      return result;
    }
    if (source_type == BF16 && destination_type == F32 &&
        vector_width % 2 == 0) {
      auto bf16_pair_type = mlir::VectorType::get({2}, builder.getBF16Type());
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
      auto f32_pair_type = mlir::VectorType::get({2}, builder.getF32Type());
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
    const bool source_is_float = IsFloatingValueType(source_type);
    const bool destination_is_float = IsFloatingValueType(destination_type);
    if (source_is_float && destination_is_float &&
        ElementBits(source_type) < ElementBits(destination_type)) {
      return mlir::arith::ExtFOp::create(builder, destination_vector_type,
                                         operand)
          .getResult();
    }
    if (source_is_float && destination_is_float) {
      return mlir::arith::TruncFOp::create(builder, destination_vector_type,
                                           operand)
          .getResult();
    }
    if (!source_is_float && !destination_is_float &&
        ElementBits(source_type) < ElementBits(destination_type)) {
      return mlir::arith::ExtSIOp::create(builder, destination_vector_type,
                                          operand)
          .getResult();
    }
    if (!source_is_float && !destination_is_float) {
      return mlir::arith::TruncIOp::create(builder, destination_vector_type,
                                           operand)
          .getResult();
    }
    if (!source_is_float) {
      return mlir::arith::SIToFPOp::create(builder, destination_vector_type,
                                           operand)
          .getResult();
    }
    return mlir::arith::FPToSIOp::create(builder, destination_vector_type,
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
    auto bf16_pair_type = mlir::VectorType::get({2}, builder.getBF16Type());
    llvm::SmallVector<Value> pairs;
    pairs.reserve(vector_width / 2);
    for (int64_t pair = 0; pair < vector_width / 2; ++pair) {
      llvm::SmallVector<int64_t, 2> mask{2 * pair, 2 * pair + 1};
      Value lhs_pair = mlir::vector::ShuffleOp::create(builder, bf16_pair_type,
                                                       lhs, lhs, mask);
      Value rhs_pair = mlir::vector::ShuffleOp::create(builder, bf16_pair_type,
                                                       rhs, rhs, mask);
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

  absl::StatusOr<Value> EmitReduceWindowBinary(
      mlir::ImplicitLocOpBuilder& builder, PrimitiveType type, HloOpcode opcode,
      Value lhs, Value rhs) const {
    const int64_t vector_width =
        mlir::cast<mlir::VectorType>(lhs.getType()).getNumElements();
    if (type == BF16) {
      return EmitPairwiseBf16Binary(builder, opcode, lhs, rhs);
    }
    auto result_type = mlir::cast<mlir::VectorType>(lhs.getType());
    if (type != F32) {
      auto compute_type =
          mlir::VectorType::get({vector_width}, builder.getF32Type());
      lhs = mlir::arith::ExtFOp::create(builder, compute_type, lhs);
      rhs = mlir::arith::ExtFOp::create(builder, compute_type, rhs);
    }
    Value computed;
    switch (opcode) {
      case HloOpcode::kAdd:
        computed = mlir::arith::AddFOp::create(builder, lhs, rhs);
        break;
      case HloOpcode::kMaximum:
        computed = mlir::arith::MaximumFOp::create(builder, lhs, rhs);
        break;
      case HloOpcode::kMinimum:
        computed = mlir::arith::MinimumFOp::create(builder, lhs, rhs);
        break;
      default:
        return absl::InternalError("Unexpected reduce-window reducer.");
    }
    return type == F32
               ? computed
               : mlir::arith::TruncFOp::create(builder, result_type, computed)
                     .getResult();
  }

  absl::StatusOr<Value> EmitStridedReductionBinary(
      mlir::ImplicitLocOpBuilder& builder, PrimitiveType type, HloOpcode opcode,
      Value lhs, Value rhs) const {
    if (type == F32) {
      return EmitReduceWindowBinary(builder, type, opcode, lhs, rhs);
    }
    if (type == S32 && opcode == HloOpcode::kAdd) {
      return mlir::arith::AddIOp::create(builder, lhs, rhs).getResult();
    }
    return absl::InternalError("Unexpected strided-reduction reducer.");
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

  absl::StatusOr<mlir::arith::CmpIPredicate> GetSignedComparePredicate(
      ComparisonDirection direction) const {
    switch (direction) {
      case ComparisonDirection::kEq:
        return mlir::arith::CmpIPredicate::eq;
      case ComparisonDirection::kNe:
        return mlir::arith::CmpIPredicate::ne;
      case ComparisonDirection::kGe:
        return mlir::arith::CmpIPredicate::sge;
      case ComparisonDirection::kGt:
        return mlir::arith::CmpIPredicate::sgt;
      case ComparisonDirection::kLe:
        return mlir::arith::CmpIPredicate::sle;
      case ComparisonDirection::kLt:
        return mlir::arith::CmpIPredicate::slt;
    }
    return absl::InvalidArgumentError(
        "Unsupported signed-integer comparison direction.");
  }

  absl::StatusOr<mlir::arith::CmpIPredicate> GetUnsignedComparePredicate(
      ComparisonDirection direction) const {
    switch (direction) {
      case ComparisonDirection::kEq:
        return mlir::arith::CmpIPredicate::eq;
      case ComparisonDirection::kNe:
        return mlir::arith::CmpIPredicate::ne;
      case ComparisonDirection::kGe:
        return mlir::arith::CmpIPredicate::uge;
      case ComparisonDirection::kGt:
        return mlir::arith::CmpIPredicate::ugt;
      case ComparisonDirection::kLe:
        return mlir::arith::CmpIPredicate::ule;
      case ComparisonDirection::kLt:
        return mlir::arith::CmpIPredicate::ult;
    }
    return absl::InvalidArgumentError(
        "Unsupported unsigned-integer comparison direction.");
  }

  absl::StatusOr<Value> EmitScalarizedUnaryMath(
      mlir::ImplicitLocOpBuilder& builder, HloOpcode opcode,
      Value operand) const {
    auto operand_type = mlir::dyn_cast<mlir::VectorType>(operand.getType());
    TF_RET_CHECK(operand_type && operand_type.getRank() == 1 &&
                 mlir::isa<mlir::FloatType>(operand_type.getElementType()));
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
        case HloOpcode::kAcos:
          scalar = mlir::math::AcosOp::create(builder, scalar);
          break;
        case HloOpcode::kAcosh:
          scalar = mlir::math::AcoshOp::create(builder, scalar);
          break;
        case HloOpcode::kAsin:
          scalar = mlir::math::AsinOp::create(builder, scalar);
          break;
        case HloOpcode::kAsinh:
          scalar = mlir::math::AsinhOp::create(builder, scalar);
          break;
        case HloOpcode::kAtanh:
          scalar = mlir::math::AtanhOp::create(builder, scalar);
          break;
        case HloOpcode::kCbrt:
          scalar = mlir::math::CbrtOp::create(builder, scalar);
          break;
        case HloOpcode::kCeil:
          scalar = mlir::math::CeilOp::create(builder, scalar);
          break;
        case HloOpcode::kCos:
          scalar = mlir::math::CosOp::create(builder, scalar);
          break;
        case HloOpcode::kCosh:
          scalar = mlir::math::CoshOp::create(builder, scalar);
          break;
        case HloOpcode::kErf:
          scalar = mlir::math::ErfOp::create(builder, scalar);
          break;
        case HloOpcode::kExp:
          scalar = mlir::math::ExpOp::create(builder, scalar);
          break;
        case HloOpcode::kExpm1:
          scalar = mlir::math::ExpM1Op::create(builder, scalar);
          break;
        case HloOpcode::kFloor:
          scalar = mlir::math::FloorOp::create(builder, scalar);
          break;
        case HloOpcode::kLog:
          scalar = mlir::math::LogOp::create(builder, scalar);
          break;
        case HloOpcode::kLog1p:
          scalar = mlir::math::Log1pOp::create(builder, scalar);
          break;
        case HloOpcode::kRoundNearestEven:
          scalar = mlir::math::RoundEvenOp::create(builder, scalar);
          break;
        case HloOpcode::kRsqrt:
          scalar = mlir::math::RsqrtOp::create(builder, scalar);
          break;
        case HloOpcode::kSin:
          scalar = mlir::math::SinOp::create(builder, scalar);
          break;
        case HloOpcode::kSinh:
          scalar = mlir::math::SinhOp::create(builder, scalar);
          break;
        case HloOpcode::kSqrt:
          scalar = mlir::math::SqrtOp::create(builder, scalar);
          break;
        case HloOpcode::kTan:
          scalar = mlir::math::TanOp::create(builder, scalar);
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

  absl::StatusOr<Value> EmitScalarizedBinaryMath(
      mlir::ImplicitLocOpBuilder& builder, HloOpcode opcode, Value lhs,
      Value rhs) const {
    auto lhs_type = mlir::dyn_cast<mlir::VectorType>(lhs.getType());
    auto rhs_type = mlir::dyn_cast<mlir::VectorType>(rhs.getType());
    TF_RET_CHECK(lhs_type && rhs_type && lhs_type == rhs_type &&
                 lhs_type.getRank() == 1 &&
                 mlir::isa<mlir::FloatType>(lhs_type.getElementType()));
    const int64_t vector_width = lhs_type.getNumElements();

    llvm::SmallVector<Value> results;
    results.reserve(vector_width);
    for (int64_t lane = 0; lane < vector_width; ++lane) {
      Value lhs_scalar =
          mlir::vector::ExtractOp::create(builder, lhs, lane).getResult();
      Value rhs_scalar =
          mlir::vector::ExtractOp::create(builder, rhs, lane).getResult();
      Value scalar;
      switch (opcode) {
        case HloOpcode::kAtan2:
          scalar = mlir::math::Atan2Op::create(builder, lhs_scalar, rhs_scalar);
          break;
        case HloOpcode::kPower:
          scalar = mlir::math::PowFOp::create(builder, lhs_scalar, rhs_scalar);
          break;
        case HloOpcode::kRemainder:
          scalar = mlir::arith::RemFOp::create(builder, lhs_scalar, rhs_scalar);
          break;
        default:
          return absl::InternalError(
              "Unexpected Fly elementwise binary math opcode.");
      }
      results.push_back(scalar);
    }
    return mlir::vector::FromElementsOp::create(builder, lhs_type, results)
        .getResult();
  }

  absl::StatusOr<Value> EmitScalarizedReducePrecision(
      mlir::ImplicitLocOpBuilder& builder, const HloInstruction* instruction,
      Value operand) const {
    auto operand_type = mlir::dyn_cast<mlir::VectorType>(operand.getType());
    TF_RET_CHECK(operand_type && operand_type.getRank() == 1 &&
                 mlir::isa<mlir::FloatType>(operand_type.getElementType()));
    const int64_t vector_width = operand_type.getNumElements();

    mlir::mhlo::ReducePrecisionOp::Properties properties;
    properties.exponent_bits =
        builder.getI32IntegerAttr(instruction->exponent_bits());
    properties.mantissa_bits =
        builder.getI32IntegerAttr(instruction->mantissa_bits());
    llvm::SmallVector<Value> results;
    results.reserve(vector_width);
    for (int64_t lane = 0; lane < vector_width; ++lane) {
      Value scalar =
          mlir::vector::ExtractOp::create(builder, operand, lane).getResult();
      scalar = mlir::mhlo::MhloOpToStdScalarOp::mapOpOfType<
          mlir::mhlo::ReducePrecisionOp>(
          builder.getLoc(), scalar.getType(), {scalar.getType()},
          mlir::mhlo::ReducePrecisionOp::Adaptor(scalar, nullptr, properties),
          /*attributes=*/{}, &builder);
      results.push_back(scalar);
    }
    return mlir::vector::FromElementsOp::create(builder, operand_type, results)
        .getResult();
  }

  struct ScalarCopyContext {
    Value layout;
    absl::flat_hash_map<PrimitiveType, Value> copy_atoms;
  };

  ScalarCopyContext CreateScalarCopyContext(
      mlir::ImplicitLocOpBuilder& builder,
      const absl::flat_hash_map<PrimitiveType, Value>& copy_atoms) const {
    return CreateCopyContext(builder, copy_atoms, /*width=*/1);
  }

  ScalarCopyContext CreateCopyContext(
      mlir::ImplicitLocOpBuilder& builder,
      const absl::flat_hash_map<PrimitiveType, Value>& copy_atoms,
      int64_t width) const {
    mlir::MLIRContext* context = builder.getContext();
    auto shape_attr = mlir::fly::IntTupleAttr::getLeafStatic(context, width);
    auto stride_attr = mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
    auto shape_type = mlir::fly::IntTupleType::get(shape_attr);
    auto stride_type = mlir::fly::IntTupleType::get(stride_attr);
    Value shape = mlir::fly::MakeIntTupleOp::create(builder, shape_type,
                                                    mlir::ValueRange{});
    Value stride = mlir::fly::MakeIntTupleOp::create(builder, stride_type,
                                                     mlir::ValueRange{});
    auto layout_type = mlir::fly::LayoutType::get(shape_attr, stride_attr);
    ScalarCopyContext scalar_copies;
    scalar_copies.layout =
        mlir::fly::MakeLayoutOp::create(builder, layout_type, shape, stride);
    for (const auto& entry : copy_atoms) {
      PrimitiveType type = entry.first;
      const int64_t element_bits = ElementBits(type);
      const int64_t copy_bits = width * element_bits;
      CHECK(copy_bits == 8 || copy_bits == 16 || copy_bits == 32 ||
            copy_bits == 64 || copy_bits == 128);
      auto scalar_copy_atom_type = mlir::fly::CopyAtomType::get(
          mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context, copy_bits,
                                                          /*cacheModifier=*/0),
          element_bits);
      scalar_copies.copy_atoms[type] = mlir::fly::MakeCopyAtomOp::create(
          builder, scalar_copy_atom_type, element_bits);
    }
    return scalar_copies;
  }

  ScalarCopyContext CreateCompatibleCopyContext(
      mlir::ImplicitLocOpBuilder& builder, int64_t width) const {
    mlir::MLIRContext* context = builder.getContext();
    auto shape_attr = mlir::fly::IntTupleAttr::getLeafStatic(context, width);
    auto stride_attr = mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
    auto shape_type = mlir::fly::IntTupleType::get(shape_attr);
    auto stride_type = mlir::fly::IntTupleType::get(stride_attr);
    Value shape = mlir::fly::MakeIntTupleOp::create(builder, shape_type,
                                                    mlir::ValueRange{});
    Value stride = mlir::fly::MakeIntTupleOp::create(builder, stride_type,
                                                     mlir::ValueRange{});
    auto layout_type = mlir::fly::LayoutType::get(shape_attr, stride_attr);
    ScalarCopyContext copies;
    copies.layout =
        mlir::fly::MakeLayoutOp::create(builder, layout_type, shape, stride);
    for (PrimitiveType type : {PRED, S8, S16, S32, S64, F8E4M3FN, F8E5M2,
                               F8E4M3FNUZ, F8E5M2FNUZ, F16, BF16, F32, F64}) {
      const int64_t element_bits = ElementBits(type);
      const int64_t copy_bits = width * element_bits;
      if (copy_bits != 8 && copy_bits != 16 && copy_bits != 32 &&
          copy_bits != 64 && copy_bits != 128) {
        continue;
      }
      auto copy_atom_type = mlir::fly::CopyAtomType::get(
          mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context, copy_bits,
                                                          /*cacheModifier=*/0),
          element_bits);
      copies.copy_atoms[type] = mlir::fly::MakeCopyAtomOp::create(
          builder, copy_atom_type, element_bits);
    }
    return copies;
  }

  absl::StatusOr<Value> EmitClampedSignedStart(
      mlir::ImplicitLocOpBuilder& builder,
      llvm::ArrayRef<Value> argument_pointers, const HloInstruction* start,
      int64_t limit, const ScalarCopyContext& scalar_copies,
      absl::flat_hash_map<const HloInstruction*, Value>& raw_start_cache)
      const {
    Value raw;
    auto existing = raw_start_cache.find(start);
    if (existing != raw_start_cache.end()) {
      raw = existing->second;
    } else if (start->opcode() == HloOpcode::kConstant) {
      const int64_t value = start->shape().element_type() == S32
                                ? start->literal().GetFirstElement<int32_t>()
                                : start->literal().GetFirstElement<int64_t>();
      raw = mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               value);
      raw_start_cache[start] = raw;
    } else {
      Value thread_id =
          mlir::gpu::ThreadIdOp::create(builder, mlir::gpu::Dimension::x);
      Value thread_i32 = mlir::arith::IndexCastOp::create(
          builder, builder.getI32Type(), thread_id);
      Value wave_lane =
          mlir::arith::AndIOp::create(builder, thread_i32,
                                      mlir::arith::ConstantIntOp::create(
                                          builder, builder.getI32Type(), 63));
      Value wave_leader = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, wave_lane,
          mlir::arith::ConstantIntOp::create(builder, builder.getI32Type(), 0));
      Value zero_offset =
          mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(), 0);
      absl::flat_hash_map<const HloInstruction*, Value> start_cache;
      TF_ASSIGN_OR_RETURN(
          Value loaded,
          EmitVector(builder, argument_pointers, start, zero_offset,
                     wave_leader, scalar_copies.copy_atoms,
                     scalar_copies.layout, /*vector_width=*/1, start_cache));
      Value scalar = mlir::vector::ExtractOp::create(builder, loaded, 0);
      if (start->shape().element_type() == S32) {
        Value uniform = mlir::ROCDL::ReadfirstlaneOp::create(
            builder, builder.getI32Type(), scalar);
        raw = mlir::arith::ExtSIOp::create(builder, builder.getI64Type(),
                                           uniform);
      } else {
        Value low = mlir::arith::TruncIOp::create(builder, builder.getI32Type(),
                                                  scalar);
        Value high = mlir::arith::ShRUIOp::create(
            builder, scalar,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               32));
        high =
            mlir::arith::TruncIOp::create(builder, builder.getI32Type(), high);
        low = mlir::ROCDL::ReadfirstlaneOp::create(builder,
                                                   builder.getI32Type(), low);
        high = mlir::ROCDL::ReadfirstlaneOp::create(builder,
                                                    builder.getI32Type(), high);
        Value low_i64 =
            mlir::arith::ExtUIOp::create(builder, builder.getI64Type(), low);
        Value high_i64 =
            mlir::arith::ExtUIOp::create(builder, builder.getI64Type(), high);
        high_i64 =
            mlir::arith::ShLIOp::create(builder, high_i64,
                                        mlir::arith::ConstantIntOp::create(
                                            builder, builder.getI64Type(), 32));
        raw = mlir::arith::OrIOp::create(builder, low_i64, high_i64);
      }
      raw_start_cache[start] = raw;
    }

    Value zero =
        mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(), 0);
    Value upper = mlir::arith::ConstantIntOp::create(
        builder, builder.getI64Type(), limit);
    Value below_zero = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::slt, raw, zero);
    Value nonnegative =
        mlir::arith::SelectOp::create(builder, below_zero, zero, raw);
    Value above_limit = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::sgt, nonnegative, upper);
    return mlir::arith::SelectOp::create(builder, above_limit, upper,
                                         nonnegative)
        .getResult();
  }

  absl::StatusOr<Value> EmitS4Parameter(mlir::ImplicitLocOpBuilder& builder,
                                        llvm::ArrayRef<Value> argument_pointers,
                                        const HloInstruction* instruction,
                                        Value element_offset, Value predicate,
                                        int64_t vector_width) const {
    TF_RET_CHECK(instruction->opcode() == HloOpcode::kParameter &&
                 instruction->shape().element_type() == S4);
    TF_RET_CHECK(vector_width == 1 || vector_width % 2 == 0);
    const int64_t packed_width = std::max<int64_t>(1, vector_width / 2);

    mlir::MLIRContext* context = builder.getContext();
    Value pointer = argument_pointers[instruction->parameter_number()];
    auto pointer_type = mlir::cast<mlir::fly::PointerType>(pointer.getType());
    TF_RET_CHECK(pointer_type.getElemTy().isInteger(8));
    Value one =
        mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(), 1);
    Value byte_offset =
        mlir::arith::ShRUIOp::create(builder, element_offset, one);
    auto load_bytes = [&](Value load_byte_offset, int64_t load_width,
                          Value load_predicate,
                          int32_t cache_modifier) -> Value {
      const int64_t copy_bits = load_width * 8;
      CHECK(copy_bits == 8 || copy_bits == 16 || copy_bits == 32 ||
            copy_bits == 64 || copy_bits == 128);
      auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
          context, /*width=*/32,
          /*divisibility=*/s4_may_start_odd_ ? 1 : load_width);
      auto offset_type = mlir::fly::IntTupleType::get(offset_attr);
      Value load_byte_offset_i32 = mlir::arith::TruncIOp::create(
          builder, builder.getI32Type(), load_byte_offset);
      Value offset_tuple = mlir::fly::MakeIntTupleOp::create(
          builder, offset_type, mlir::ValueRange{load_byte_offset_i32});
      Value advanced = mlir::fly::AddOffsetOp::create(builder, pointer_type,
                                                      pointer, offset_tuple);

      auto shape_attr =
          mlir::fly::IntTupleAttr::getLeafStatic(context, load_width);
      auto stride_attr = mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
      auto shape_type = mlir::fly::IntTupleType::get(shape_attr);
      auto stride_type = mlir::fly::IntTupleType::get(stride_attr);
      Value shape = mlir::fly::MakeIntTupleOp::create(builder, shape_type,
                                                      mlir::ValueRange{});
      Value stride = mlir::fly::MakeIntTupleOp::create(builder, stride_type,
                                                       mlir::ValueRange{});
      auto layout_type = mlir::fly::LayoutType::get(shape_attr, stride_attr);
      Value layout =
          mlir::fly::MakeLayoutOp::create(builder, layout_type, shape, stride);
      auto memref_type = mlir::fly::MemRefType::get(
          builder.getI8Type(), pointer_type.getAddressSpace(),
          layout_type.getAttr());
      Value view =
          mlir::fly::MakeViewOp::create(builder, memref_type, advanced, layout);
      auto copy_atom_type = mlir::fly::CopyAtomType::get(
          mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context, copy_bits,
                                                          cache_modifier),
          /*elementBits=*/8);
      Value copy_atom = mlir::fly::MakeCopyAtomOp::create(
          builder, copy_atom_type, /*elementBits=*/8);
      auto packed_type =
          mlir::VectorType::get({load_width}, builder.getI8Type());
      Value poison = mlir::ub::PoisonOp::create(builder, packed_type);
      return mlir::fly::CopyAtomCallSSA::create(
                 builder, mlir::TypeRange{packed_type}, copy_atom, view, poison,
                 load_predicate)
          .getResult(0);
    };
    Value packed = load_bytes(byte_offset, packed_width, predicate,
                              vector_width == 1 ? 0 : cache_modifier_);

    if (vector_width == 1) {
      Value byte = mlir::vector::ExtractOp::create(builder, packed, 0);
      Value shift =
          mlir::arith::ConstantIntOp::create(builder, builder.getI8Type(), 4);
      Value low = mlir::arith::ShLIOp::create(builder, byte, shift);
      low = mlir::arith::ShRSIOp::create(builder, low, shift);
      Value high = mlir::arith::ShRSIOp::create(builder, byte, shift);
      Value odd = mlir::arith::AndIOp::create(builder, element_offset, one);
      Value use_high = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ne, odd,
          mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(), 0));
      Value nibble =
          mlir::arith::SelectOp::create(builder, use_high, high, low);
      auto result_type = mlir::VectorType::get({1}, builder.getI8Type());
      return mlir::vector::FromElementsOp::create(builder, result_type,
                                                  mlir::ValueRange{nibble})
          .getResult();
    }

    auto packed_type =
        mlir::VectorType::get({packed_width}, builder.getI8Type());
    Value shift = SplatInteger(builder, packed_type, 4);
    Value low = mlir::arith::ShLIOp::create(builder, packed, shift);
    low = mlir::arith::ShRSIOp::create(builder, low, shift);
    Value high = mlir::arith::ShRSIOp::create(builder, packed, shift);
    llvm::SmallVector<int64_t> interleave;
    interleave.reserve(vector_width);
    for (int64_t element = 0; element < packed_width; ++element) {
      interleave.push_back(element);
      interleave.push_back(packed_width + element);
    }
    auto result_type =
        mlir::VectorType::get({vector_width}, builder.getI8Type());
    Value even_result = mlir::vector::ShuffleOp::create(builder, result_type,
                                                        low, high, interleave);
    if (!s4_may_start_odd_) {
      return even_result;
    }

    Value odd = mlir::arith::AndIOp::create(builder, element_offset, one);
    Value use_high = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::ne, odd,
        mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(), 0));
    Value trailing_offset = mlir::arith::AddIOp::create(
        builder, byte_offset,
        mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                           packed_width));
    Value trailing_predicate =
        mlir::arith::AndIOp::create(builder, predicate, use_high);
    Value trailing_vector =
        load_bytes(trailing_offset, /*load_width=*/1, trailing_predicate,
                   /*cache_modifier=*/0);
    Value trailing =
        mlir::vector::ExtractOp::create(builder, trailing_vector, 0);
    llvm::SmallVector<Value> next_elements;
    next_elements.reserve(packed_width);
    for (int64_t element = 1; element < packed_width; ++element) {
      next_elements.push_back(
          mlir::vector::ExtractOp::create(builder, packed, element));
    }
    next_elements.push_back(trailing);
    Value next = mlir::vector::FromElementsOp::create(builder, packed_type,
                                                      next_elements);
    Value low_next = mlir::arith::ShLIOp::create(builder, next, shift);
    low_next = mlir::arith::ShRSIOp::create(builder, low_next, shift);
    llvm::SmallVector<int64_t> odd_interleave;
    odd_interleave.reserve(vector_width);
    for (int64_t element = 0; element < packed_width; ++element) {
      odd_interleave.push_back(element);
      odd_interleave.push_back(packed_width + element);
    }
    Value odd_result = mlir::vector::ShuffleOp::create(
        builder, result_type, high, low_next, odd_interleave);
    return mlir::arith::SelectOp::create(builder, use_high, odd_result,
                                         even_result)
        .getResult();
  }

  absl::StatusOr<Value> EmitVector(
      mlir::ImplicitLocOpBuilder& builder,
      llvm::ArrayRef<Value> argument_pointers,
      const HloInstruction* instruction, Value element_offset, Value predicate,
      const absl::flat_hash_map<PrimitiveType, Value>& copy_atoms,
      Value vector_layout, int64_t vector_width,
      absl::flat_hash_map<const HloInstruction*, Value>& cache) const {
    auto existing = cache.find(instruction);
    if (existing != cache.end()) {
      return existing->second;
    }

    const PrimitiveType instruction_type = instruction->shape().element_type();
    TF_RET_CHECK(IsSupportedValueType(instruction_type));
    auto vector_type = mlir::VectorType::get(
        {vector_width}, instruction_type == PRED
                            ? mlir::Type(builder.getI1Type())
                            : StorageElementType(instruction_type, builder));
    Value result;
    switch (instruction->opcode()) {
      case HloOpcode::kIota: {
        const std::optional<int64_t> physical_stride =
            IotaPhysicalStride(instruction);
        TF_RET_CHECK(physical_stride.has_value());
        const int64_t iota_dimension =
            Cast<const HloIotaInstruction>(instruction)->iota_dimension();
        const int64_t dimension_size =
            instruction->shape().dimensions(iota_dimension);
        auto index_vector_type =
            mlir::VectorType::get({vector_width}, builder.getI32Type());
        Value base_scalar = mlir::arith::TruncIOp::create(
            builder, builder.getI32Type(), element_offset);
        Value base = mlir::vector::BroadcastOp::create(
            builder, index_vector_type, base_scalar);
        llvm::SmallVector<int32_t> lane_offsets;
        lane_offsets.reserve(vector_width);
        for (int32_t lane = 0; lane < vector_width; ++lane) {
          lane_offsets.push_back(lane);
        }
        Value coordinate = mlir::arith::AddIOp::create(
            builder, base,
            mlir::arith::ConstantOp::create(
                builder, index_vector_type,
                mlir::DenseIntElementsAttr::get(index_vector_type,
                                                lane_offsets)));
        auto splat_index = [&](int64_t value) {
          return mlir::arith::ConstantOp::create(
                     builder, index_vector_type,
                     mlir::DenseElementsAttr::get(
                         index_vector_type, builder.getI32IntegerAttr(
                                                static_cast<int32_t>(value))))
              .getResult();
        };
        if (*physical_stride != 1) {
          coordinate = mlir::arith::DivUIOp::create(
              builder, coordinate, splat_index(*physical_stride));
        }
        if (dimension_size <= 1) {
          coordinate = splat_index(0);
        } else if (ShapeUtil::ElementsIn(instruction->shape()) /
                       dimension_size !=
                   *physical_stride) {
          coordinate = mlir::arith::RemUIOp::create(
              builder, coordinate, splat_index(dimension_size));
        }
        if (instruction_type == S32) {
          result = coordinate;
        } else if (instruction_type == S64) {
          result =
              mlir::arith::ExtUIOp::create(builder, vector_type, coordinate);
        } else if (IsSupportedSignedIntegerType(instruction_type)) {
          result =
              mlir::arith::TruncIOp::create(builder, vector_type, coordinate);
        } else {
          result =
              mlir::arith::UIToFPOp::create(builder, vector_type, coordinate);
        }
        break;
      }
      case HloOpcode::kParameter: {
        if (instruction_type == S4) {
          TF_ASSIGN_OR_RETURN(
              result, EmitS4Parameter(builder, argument_pointers, instruction,
                                      element_offset, predicate, vector_width));
          break;
        }
        auto copy_atom = copy_atoms.find(instruction_type);
        if (copy_atom == copy_atoms.end()) {
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          int64_t emitted = 0;
          while (emitted < vector_width) {
            int64_t chunk_width = std::min<int64_t>(
                vector_width - emitted, 128 / ElementBits(instruction_type));
            while (chunk_width * ElementBits(instruction_type) != 8 &&
                   chunk_width * ElementBits(instruction_type) != 16 &&
                   chunk_width * ElementBits(instruction_type) != 32 &&
                   chunk_width * ElementBits(instruction_type) != 64 &&
                   chunk_width * ElementBits(instruction_type) != 128) {
              --chunk_width;
            }
            TF_RET_CHECK(chunk_width > 0);
            Value chunk_offset = element_offset;
            if (emitted != 0) {
              chunk_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), emitted));
            }
            ScalarCopyContext chunk_copies =
                CreateCompatibleCopyContext(builder, chunk_width);
            absl::flat_hash_map<const HloInstruction*, Value> chunk_cache;
            TF_ASSIGN_OR_RETURN(
                Value chunk,
                EmitVector(builder, argument_pointers, instruction,
                           chunk_offset, predicate, chunk_copies.copy_atoms,
                           chunk_copies.layout, chunk_width, chunk_cache));
            for (int64_t lane = 0; lane < chunk_width; ++lane) {
              lanes.push_back(
                  mlir::vector::ExtractOp::create(builder, chunk, lane));
            }
            emitted += chunk_width;
          }
          result =
              mlir::vector::FromElementsOp::create(builder, vector_type, lanes);
          break;
        }
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
        auto storage_vector_type =
            instruction_type == PRED
                ? mlir::VectorType::get({vector_width}, builder.getI8Type())
                : vector_type;
        auto memref_type = mlir::fly::MemRefType::get(
            storage_vector_type.getElementType(),
            mlir::cast<mlir::fly::PointerType>(pointer.getType())
                .getAddressSpace(),
            layout_type.getAttr());
        Value view = mlir::fly::MakeViewOp::create(builder, memref_type,
                                                   advanced, vector_layout);
        Value poison = mlir::ub::PoisonOp::create(builder, storage_vector_type);
        mlir::fly::CopyAtomCallSSA load = mlir::fly::CopyAtomCallSSA::create(
            builder, mlir::TypeRange{storage_vector_type}, copy_atom->second,
            view, poison, predicate);
        result = load.getResult(0);
        if (instruction_type == PRED) {
          result = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ne, result,
              mlir::arith::ConstantOp::create(
                  builder, storage_vector_type,
                  mlir::DenseElementsAttr::get(
                      storage_vector_type,
                      builder.getIntegerAttr(builder.getI8Type(), 0))));
        }
        break;
      }
      case HloOpcode::kConstant: {
        mlir::Attribute scalar;
        if (IsFp8Type(instruction_type)) {
          const uint8_t value = *static_cast<const uint8_t*>(
              instruction->literal().untyped_data());
          scalar = builder.getIntegerAttr(builder.getI8Type(), value);
        } else if (IsSupportedFloatingType(instruction_type)) {
          std::vector<int64_t> first_index(
              instruction->shape().dimensions_size(), 0);
          std::optional<double> value =
              instruction->literal().GetAsDouble(first_index);
          TF_RET_CHECK(value.has_value());
          scalar = builder.getFloatAttr(vector_type.getElementType(), *value);
        } else {
          int64_t value;
          switch (instruction_type) {
            case PRED:
              value = instruction->literal().GetFirstElement<bool>();
              break;
            case S8:
              value = instruction->literal().GetFirstElement<int8_t>();
              break;
            case S16:
              value = instruction->literal().GetFirstElement<int16_t>();
              break;
            case S32:
              value = instruction->literal().GetFirstElement<int32_t>();
              break;
            case S64:
              value = instruction->literal().GetFirstElement<int64_t>();
              break;
            default:
              return absl::InvalidArgumentError(
                  "Unsupported Fly elementwise constant type.");
          }
          scalar = builder.getIntegerAttr(vector_type.getElementType(), value);
        }
        result = mlir::arith::ConstantOp::create(
            builder, vector_type,
            mlir::DenseElementsAttr::get(vector_type, scalar));
        break;
      }
      case HloOpcode::kBroadcast: {
        if (ShapeUtil::IsScalar(instruction->operand(0)->shape())) {
          ScalarCopyContext scalar_copies =
              CreateScalarCopyContext(builder, copy_atoms);
          Value zero_offset = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
          TF_ASSIGN_OR_RETURN(
              Value scalar_vector,
              EmitVector(builder, argument_pointers, instruction->operand(0),
                         zero_offset, predicate, scalar_copies.copy_atoms,
                         scalar_copies.layout, /*vector_width=*/1,
                         operand_cache));
          Value scalar =
              mlir::vector::ExtractOp::create(builder, scalar_vector, 0);
          result =
              mlir::vector::BroadcastOp::create(builder, vector_type, scalar);
          break;
        }
        {
          const int64_t input_rank =
              instruction->operand(0)->shape().dimensions_size();
          const int64_t output_rank = instruction->shape().dimensions_size();
          bool is_leading_broadcast = true;
          bool is_trailing_broadcast = true;
          for (int64_t dimension = 0; dimension < input_rank; ++dimension) {
            is_leading_broadcast &=
                instruction->dimensions(dimension) == dimension;
            is_trailing_broadcast &= instruction->dimensions(dimension) ==
                                     output_rank - input_rank + dimension;
          }
          if (is_leading_broadcast) {
            const int64_t repeat =
                ShapeUtil::ElementsIn(instruction->shape()) /
                ShapeUtil::ElementsIn(instruction->operand(0)->shape());
            Value repeat_value = mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(), repeat);
            auto input_offset = [&](Value output_offset) {
              return mlir::arith::DivUIOp::create(builder, output_offset,
                                                  repeat_value)
                  .getResult();
            };
            Value first_input_offset = input_offset(element_offset);
            Value last_output_offset = element_offset;
            if (vector_width != 1) {
              last_output_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), vector_width - 1));
            }
            Value last_input_offset = input_offset(last_output_offset);
            Value reuses_one_input = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::eq, first_input_offset,
                last_input_offset);
            mlir::scf::IfOp select = mlir::scf::IfOp::create(
                builder, mlir::TypeRange{vector_type}, reuses_one_input,
                /*withElseRegion=*/true);
            {
              mlir::OpBuilder::InsertionGuard guard(builder);
              builder.setInsertionPointToStart(select.thenBlock());
              ScalarCopyContext scalar_copies =
                  CreateScalarCopyContext(builder, copy_atoms);
              absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
              TF_ASSIGN_OR_RETURN(
                  Value scalar_vector,
                  EmitVector(
                      builder, argument_pointers, instruction->operand(0),
                      first_input_offset, predicate, scalar_copies.copy_atoms,
                      scalar_copies.layout, /*vector_width=*/1, operand_cache));
              Value scalar =
                  mlir::vector::ExtractOp::create(builder, scalar_vector, 0);
              Value vector = mlir::vector::BroadcastOp::create(
                  builder, vector_type, scalar);
              mlir::scf::YieldOp::create(builder, vector);

              builder.setInsertionPointToStart(select.elseBlock());
              ScalarCopyContext boundary_copies =
                  CreateScalarCopyContext(builder, copy_atoms);
              llvm::SmallVector<Value> lanes;
              lanes.reserve(vector_width);
              for (int64_t lane = 0; lane < vector_width; ++lane) {
                Value lane_offset = element_offset;
                if (lane != 0) {
                  lane_offset = mlir::arith::AddIOp::create(
                      builder, element_offset,
                      mlir::arith::ConstantIntOp::create(
                          builder, builder.getI64Type(), lane));
                }
                Value scalar_input_offset = input_offset(lane_offset);
                absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
                TF_ASSIGN_OR_RETURN(
                    Value scalar,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(0), scalar_input_offset,
                               predicate, boundary_copies.copy_atoms,
                               boundary_copies.layout, /*vector_width=*/1,
                               operand_cache));
                lanes.push_back(
                    mlir::vector::ExtractOp::create(builder, scalar, 0));
              }
              Value boundary = mlir::vector::FromElementsOp::create(
                  builder, vector_type, lanes);
              mlir::scf::YieldOp::create(builder, boundary);
            }
            builder.setInsertionPointAfter(select);
            result = select.getResult(0);
            break;
          }
          if (!is_trailing_broadcast) {
            auto input_offset = [&](Value output_offset) {
              Value mapped = mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(), 0);
              int64_t input_stride = 1;
              for (int64_t input_dimension = input_rank - 1;
                   input_dimension >= 0; --input_dimension) {
                const int64_t output_dimension =
                    instruction->dimensions(input_dimension);
                int64_t output_stride = 1;
                for (int64_t dimension = output_dimension + 1;
                     dimension < output_rank; ++dimension) {
                  output_stride *= instruction->shape().dimensions(dimension);
                }
                Value coordinate = output_offset;
                if (output_stride != 1) {
                  coordinate = mlir::arith::DivUIOp::create(
                      builder, coordinate,
                      mlir::arith::ConstantIntOp::create(
                          builder, builder.getI64Type(), output_stride));
                }
                const int64_t output_extent =
                    instruction->shape().dimensions(output_dimension);
                if (output_extent != 1) {
                  coordinate = mlir::arith::RemUIOp::create(
                      builder, coordinate,
                      mlir::arith::ConstantIntOp::create(
                          builder, builder.getI64Type(), output_extent));
                } else {
                  coordinate = mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), 0);
                }
                if (input_stride != 1) {
                  coordinate = mlir::arith::MulIOp::create(
                      builder, coordinate,
                      mlir::arith::ConstantIntOp::create(
                          builder, builder.getI64Type(), input_stride));
                }
                mapped =
                    mlir::arith::AddIOp::create(builder, mapped, coordinate);
                input_stride *= instruction->operand(0)->shape().dimensions(
                    input_dimension);
              }
              return mapped;
            };

            const int64_t output_minor_extent =
                instruction->shape().dimensions(output_rank - 1);
            Value minor_offset = mlir::arith::RemUIOp::create(
                builder, element_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), output_minor_extent));
            Value minor_end = mlir::arith::AddIOp::create(
                builder, minor_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), vector_width));
            Value fits_in_minor = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ule, minor_end,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), output_minor_extent));
            mlir::scf::IfOp select = mlir::scf::IfOp::create(
                builder, mlir::TypeRange{vector_type}, fits_in_minor,
                /*withElseRegion=*/true);
            {
              mlir::OpBuilder::InsertionGuard guard(builder);
              builder.setInsertionPointToStart(select.thenBlock());
              Value first_input_offset = input_offset(element_offset);
              absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
              if (instruction->dimensions(input_rank - 1) == output_rank - 1) {
                TF_ASSIGN_OR_RETURN(
                    Value vector,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(0), first_input_offset,
                               predicate, copy_atoms, vector_layout,
                               vector_width, operand_cache));
                mlir::scf::YieldOp::create(builder, vector);
              } else {
                ScalarCopyContext scalar_copies =
                    CreateScalarCopyContext(builder, copy_atoms);
                TF_ASSIGN_OR_RETURN(
                    Value scalar_vector,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(0), first_input_offset,
                               predicate, scalar_copies.copy_atoms,
                               scalar_copies.layout, /*vector_width=*/1,
                               operand_cache));
                Value scalar =
                    mlir::vector::ExtractOp::create(builder, scalar_vector, 0);
                Value vector = mlir::vector::BroadcastOp::create(
                    builder, vector_type, scalar);
                mlir::scf::YieldOp::create(builder, vector);
              }

              builder.setInsertionPointToStart(select.elseBlock());
              ScalarCopyContext scalar_copies =
                  CreateScalarCopyContext(builder, copy_atoms);
              llvm::SmallVector<Value> lanes;
              lanes.reserve(vector_width);
              for (int64_t lane = 0; lane < vector_width; ++lane) {
                Value lane_offset = element_offset;
                if (lane != 0) {
                  lane_offset = mlir::arith::AddIOp::create(
                      builder, element_offset,
                      mlir::arith::ConstantIntOp::create(
                          builder, builder.getI64Type(), lane));
                }
                Value scalar_input_offset = input_offset(lane_offset);
                absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
                TF_ASSIGN_OR_RETURN(
                    Value scalar,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(0), scalar_input_offset,
                               predicate, scalar_copies.copy_atoms,
                               scalar_copies.layout, /*vector_width=*/1,
                               operand_cache));
                lanes.push_back(
                    mlir::vector::ExtractOp::create(builder, scalar, 0));
              }
              Value boundary = mlir::vector::FromElementsOp::create(
                  builder, vector_type, lanes);
              mlir::scf::YieldOp::create(builder, boundary);
            }
            builder.setInsertionPointAfter(select);
            result = select.getResult(0);
            break;
          }
          const int64_t period =
              ShapeUtil::ElementsIn(instruction->operand(0)->shape());
          Value period_value = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), period);
          Value input_offset = mlir::arith::RemUIOp::create(
              builder, element_offset, period_value);
          Value vector_end = mlir::arith::AddIOp::create(
              builder, input_offset,
              mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                                 vector_width));
          Value fits_in_period = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, vector_end,
              period_value);
          mlir::scf::IfOp select = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{vector_type}, fits_in_period,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(select.thenBlock());
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value vector,
                EmitVector(builder, argument_pointers, instruction->operand(0),
                           input_offset, predicate, copy_atoms, vector_layout,
                           vector_width, operand_cache));
            mlir::scf::YieldOp::create(builder, vector);

            builder.setInsertionPointToStart(select.elseBlock());
            ScalarCopyContext scalar_copies =
                CreateScalarCopyContext(builder, copy_atoms);
            llvm::SmallVector<Value> lanes;
            lanes.reserve(vector_width);
            for (int64_t lane = 0; lane < vector_width; ++lane) {
              Value lane_offset = element_offset;
              if (lane != 0) {
                lane_offset = mlir::arith::AddIOp::create(
                    builder, element_offset,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(), lane));
              }
              Value scalar_input_offset = mlir::arith::RemUIOp::create(
                  builder, lane_offset, period_value);
              absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
              TF_ASSIGN_OR_RETURN(
                  Value scalar,
                  EmitVector(
                      builder, argument_pointers, instruction->operand(0),
                      scalar_input_offset, predicate, scalar_copies.copy_atoms,
                      scalar_copies.layout, /*vector_width=*/1, operand_cache));
              lanes.push_back(
                  mlir::vector::ExtractOp::create(builder, scalar, 0));
            }
            Value boundary = mlir::vector::FromElementsOp::create(
                builder, vector_type, lanes);
            mlir::scf::YieldOp::create(builder, boundary);
          }
          builder.setInsertionPointAfter(select);
          result = select.getResult(0);
          break;
        }
      }
      case HloOpcode::kBitcast:
      case HloOpcode::kBitcastConvert:
      case HloOpcode::kReshape: {
        TF_RET_CHECK(IsSupportedPhysicalView(instruction));
        const HloInstruction* operand = instruction->operand(0);
        const PrimitiveType operand_type = operand->shape().element_type();
        if (instruction_type != operand_type) {
          const int64_t result_bits = ElementBits(instruction_type);
          const int64_t operand_bits = ElementBits(operand_type);
          const int64_t result_vector_bits = vector_width * result_bits;
          Value operand_offset = element_offset;
          if (result_bits > operand_bits) {
            operand_offset = mlir::arith::MulIOp::create(
                builder, element_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), result_bits / operand_bits));
          } else if (operand_bits > result_bits) {
            operand_offset = mlir::arith::DivUIOp::create(
                builder, element_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), operand_bits / result_bits));
          }

          if (result_vector_bits < operand_bits) {
            // The ordinary scalar tail can address one packed component of a
            // wider source value. Bitcast the containing source lane and
            // select the exact little-endian component without reading past
            // either buffer.
            TF_RET_CHECK(vector_width == 1 && operand_bits % result_bits == 0);
            ScalarCopyContext operand_copies =
                CreateCompatibleCopyContext(builder, /*width=*/1);
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value source,
                EmitVector(builder, argument_pointers, operand, operand_offset,
                           predicate, operand_copies.copy_atoms,
                           operand_copies.layout,
                           /*vector_width=*/1, operand_cache));
            const int64_t components = operand_bits / result_bits;
            auto packed_type = mlir::VectorType::get(
                {components}, vector_type.getElementType());
            Value packed =
                mlir::vector::BitCastOp::create(builder, packed_type, source);
            Value component = mlir::arith::RemUIOp::create(
                builder, element_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), components));
            Value selected =
                mlir::vector::ExtractOp::create(builder, packed, 0);
            for (int64_t index = 1; index < components; ++index) {
              Value is_index = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::eq, component,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), index));
              Value candidate =
                  mlir::vector::ExtractOp::create(builder, packed, index);
              selected = mlir::arith::SelectOp::create(builder, is_index,
                                                       candidate, selected);
            }
            result = mlir::vector::FromElementsOp::create(
                builder, vector_type, mlir::ValueRange{selected});
            break;
          }

          TF_RET_CHECK(result_vector_bits % operand_bits == 0);
          const int64_t operand_width = result_vector_bits / operand_bits;
          ScalarCopyContext operand_copies =
              CreateCompatibleCopyContext(builder, operand_width);
          absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
          TF_ASSIGN_OR_RETURN(
              Value source,
              EmitVector(builder, argument_pointers, operand, operand_offset,
                         predicate, operand_copies.copy_atoms,
                         operand_copies.layout, operand_width, operand_cache));
          result =
              mlir::vector::BitCastOp::create(builder, vector_type, source);
          break;
        }
        absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
        TF_ASSIGN_OR_RETURN(
            result, EmitVector(builder, argument_pointers, operand,
                               element_offset, predicate, copy_atoms,
                               vector_layout, vector_width, operand_cache));
        break;
      }
      case HloOpcode::kTranspose: {
        if (IsSupportedPhysicalView(instruction)) {
          absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
          TF_ASSIGN_OR_RETURN(
              result,
              EmitVector(builder, argument_pointers, instruction->operand(0),
                         element_offset, predicate, copy_atoms, vector_layout,
                         vector_width, operand_cache));
          break;
        }
        std::optional<DataMovingTranspose> transpose =
            GetDataMovingTranspose(instruction);
        TF_RET_CHECK(transpose.has_value());
        const int64_t rank = transpose->output_dimensions.size();
        auto map_input_offset = [&](Value output_offset) {
          Value remaining = output_offset;
          Value input_offset = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          for (int64_t output_dimension = rank - 1; output_dimension >= 0;
               --output_dimension) {
            Value coordinate = remaining;
            if (output_dimension != 0) {
              Value dimension_size = mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  transpose->output_dimensions[output_dimension]);
              coordinate = mlir::arith::RemUIOp::create(builder, remaining,
                                                        dimension_size);
              remaining = mlir::arith::DivUIOp::create(builder, remaining,
                                                       dimension_size);
            }
            const int64_t input_dimension =
                transpose->output_to_input_dimensions[output_dimension];
            const int64_t input_stride =
                transpose->input_strides[input_dimension];
            if (input_stride != 1) {
              coordinate = mlir::arith::MulIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), input_stride));
            }
            input_offset =
                mlir::arith::AddIOp::create(builder, input_offset, coordinate);
          }
          return input_offset;
        };
        auto emit_scalar_lanes = [&]() -> absl::StatusOr<Value> {
          ScalarCopyContext scalar_copies =
              CreateScalarCopyContext(builder, copy_atoms);
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lane_offset = element_offset;
            if (lane != 0) {
              lane_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), lane));
            }
            Value input_offset = map_input_offset(lane_offset);
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value scalar,
                EmitVector(builder, argument_pointers, instruction->operand(0),
                           input_offset, predicate,
                           scalar_copies.copy_atoms, scalar_copies.layout,
                           /*vector_width=*/1, operand_cache));
            lanes.push_back(
                mlir::vector::ExtractOp::create(builder, scalar, 0));
          }
          return mlir::vector::FromElementsOp::create(builder, vector_type,
                                                       lanes);
        };

        const int64_t input_minor_stride = transpose->input_strides[
            transpose->output_to_input_dimensions.back()];
        const int64_t input_span =
            (vector_width - 1) * input_minor_stride + 1;
        int64_t packed_width = 1;
        while (packed_width < input_span) {
          packed_width *= 2;
        }
        const bool can_load_packed_span =
            packed_width * ElementBits(instruction_type) <= 128;
        Value output_row_elements = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(),
            transpose->output_dimensions.back());
        Value output_column = mlir::arith::RemUIOp::create(
            builder, element_offset, output_row_elements);
        Value vector_column_end = mlir::arith::AddIOp::create(
            builder, output_column,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               vector_width));
        Value fits_in_row = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule, vector_column_end,
            output_row_elements);
        mlir::scf::IfOp select = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{vector_type}, fits_in_row,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          Value input_offset = map_input_offset(element_offset);
          Value vector;
          if (input_minor_stride == 1) {
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                vector,
                EmitVector(builder, argument_pointers, instruction->operand(0),
                           input_offset, predicate, copy_atoms, vector_layout,
                           vector_width, operand_cache));
          } else if (can_load_packed_span) {
            ScalarCopyContext packed_copies =
                CreateCompatibleCopyContext(builder, packed_width);
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value packed,
                EmitVector(builder, argument_pointers, instruction->operand(0),
                           input_offset, predicate, packed_copies.copy_atoms,
                           packed_copies.layout, packed_width, operand_cache));
            llvm::SmallVector<int64_t> selected_lanes;
            selected_lanes.reserve(vector_width);
            for (int64_t lane = 0; lane < vector_width; ++lane) {
              selected_lanes.push_back(lane * input_minor_stride);
            }
            vector = mlir::vector::ShuffleOp::create(
                builder, vector_type, packed, packed, selected_lanes);
          } else {
            TF_ASSIGN_OR_RETURN(vector, emit_scalar_lanes());
          }
          mlir::scf::YieldOp::create(builder, vector);

          builder.setInsertionPointToStart(select.elseBlock());
          TF_ASSIGN_OR_RETURN(Value boundary, emit_scalar_lanes());
          mlir::scf::YieldOp::create(builder, boundary);
        }
        builder.setInsertionPointAfter(select);
        result = select.getResult(0);
        break;
      }
      case HloOpcode::kSlice: {
        std::optional<int64_t> slice_base = ContiguousSliceBase(instruction);
        if (slice_base.has_value()) {
          Value input_offset = element_offset;
          if (*slice_base != 0) {
            input_offset = mlir::arith::AddIOp::create(
                builder, element_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), *slice_base));
          }
          absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
          TF_ASSIGN_OR_RETURN(
              result,
              EmitVector(builder, argument_pointers, instruction->operand(0),
                         input_offset, predicate, copy_atoms, vector_layout,
                         vector_width, operand_cache));
          break;
        }
        std::optional<RectangularSlice> slice =
            GetRectangularSlice(instruction);
        TF_RET_CHECK(slice.has_value());
        Value input_base = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), slice->input_base);
        auto map_input_offset = [&](Value output_offset) {
          if (slice->flat_input_stride != 0) {
            Value input_offset = output_offset;
            if (slice->flat_input_stride != 1) {
              input_offset = mlir::arith::MulIOp::create(
                  builder, input_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(),
                      slice->flat_input_stride));
            }
            if (slice->input_base != 0) {
              input_offset = mlir::arith::AddIOp::create(
                  builder, input_offset, input_base);
            }
            return input_offset;
          }
          Value remaining = output_offset;
          Value input_offset = input_base;
          const int64_t rank = slice->output_dimensions.size();
          for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
            Value coordinate = remaining;
            if (dimension != 0) {
              Value dimension_size = mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  slice->output_dimensions[dimension]);
              coordinate = mlir::arith::RemUIOp::create(builder, remaining,
                                                        dimension_size);
              remaining = mlir::arith::DivUIOp::create(builder, remaining,
                                                       dimension_size);
            }
            if (slice->input_strides[dimension] != 1) {
              coordinate = mlir::arith::MulIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(),
                      slice->input_strides[dimension]));
            }
            input_offset =
                mlir::arith::AddIOp::create(builder, input_offset, coordinate);
          }
          return input_offset;
        };
        auto emit_scalar_lanes = [&]() -> absl::StatusOr<Value> {
          ScalarCopyContext scalar_copies =
              CreateScalarCopyContext(builder, copy_atoms);
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lane_offset = element_offset;
            if (lane != 0) {
              lane_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), lane));
            }
            Value scalar_input_offset = map_input_offset(lane_offset);
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value scalar,
                EmitVector(builder, argument_pointers, instruction->operand(0),
                           scalar_input_offset, predicate,
                           scalar_copies.copy_atoms, scalar_copies.layout,
                           /*vector_width=*/1, operand_cache));
            lanes.push_back(
                mlir::vector::ExtractOp::create(builder, scalar, 0));
          }
          return mlir::vector::FromElementsOp::create(builder, vector_type,
                                                       lanes);
        };
        const int64_t minor_stride = slice->input_strides.back();
        const int64_t input_span =
            (vector_width - 1) * minor_stride + 1;
        int64_t packed_width = 1;
        while (packed_width < input_span) {
          packed_width *= 2;
        }
        const bool can_load_packed_span =
            packed_width * ElementBits(instruction->shape().element_type()) <=
            128;
        Value output_row_elements = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), slice->output_dimensions.back());
        Value output_column = mlir::arith::RemUIOp::create(
            builder, element_offset, output_row_elements);
        Value vector_column_end = mlir::arith::AddIOp::create(
            builder, output_column,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               vector_width));
        Value fits_in_row = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule, vector_column_end,
            output_row_elements);
        if (minor_stride != 1 && can_load_packed_span) {
          Value input_column = output_column;
          if (minor_stride != 1) {
            input_column = mlir::arith::MulIOp::create(
                builder, input_column,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), minor_stride));
          }
          if (slice->input_minor_start != 0) {
            input_column = mlir::arith::AddIOp::create(
                builder, input_column,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), slice->input_minor_start));
          }
          Value packed_column_end = mlir::arith::AddIOp::create(
              builder, input_column,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(), packed_width));
          Value fits_input_row = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, packed_column_end,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  slice->input_minor_dimension));
          fits_in_row = mlir::arith::AndIOp::create(builder, fits_in_row,
                                                    fits_input_row);
        }
        mlir::scf::IfOp select = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{vector_type}, fits_in_row,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          Value input_offset = map_input_offset(element_offset);
          Value vector;
          if (minor_stride == 1) {
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                vector,
                EmitVector(builder, argument_pointers,
                           instruction->operand(0), input_offset, predicate,
                           copy_atoms, vector_layout, vector_width,
                           operand_cache));
          } else if (can_load_packed_span) {
            ScalarCopyContext packed_copies =
                CreateCompatibleCopyContext(builder, packed_width);
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value packed,
                EmitVector(builder, argument_pointers,
                           instruction->operand(0), input_offset, predicate,
                           packed_copies.copy_atoms, packed_copies.layout,
                           packed_width, operand_cache));
            llvm::SmallVector<int64_t> selected_lanes;
            selected_lanes.reserve(vector_width);
            for (int64_t lane = 0; lane < vector_width; ++lane) {
              selected_lanes.push_back(lane * minor_stride);
            }
            vector = mlir::vector::ShuffleOp::create(
                builder, vector_type, packed, packed, selected_lanes);
          } else {
            TF_ASSIGN_OR_RETURN(vector, emit_scalar_lanes());
          }
          mlir::scf::YieldOp::create(builder, vector);

          builder.setInsertionPointToStart(select.elseBlock());
          TF_ASSIGN_OR_RETURN(Value boundary, emit_scalar_lanes());
          mlir::scf::YieldOp::create(builder, boundary);
        }
        builder.setInsertionPointAfter(select);
        result = select.getResult(0);
        break;
      }
      case HloOpcode::kDynamicSlice: {
        std::optional<DynamicSlice> slice = GetDynamicSlice(instruction);
        TF_RET_CHECK(slice.has_value());
        ScalarCopyContext scalar_copies =
            CreateScalarCopyContext(builder, copy_atoms);
        absl::flat_hash_map<const HloInstruction*, Value> raw_start_cache;
        llvm::SmallVector<Value> starts;
        starts.reserve(slice->start_limits.size());
        const int64_t rank = slice->start_limits.size();
        for (int64_t dimension = 0; dimension < rank; ++dimension) {
          TF_ASSIGN_OR_RETURN(
              Value start,
              EmitClampedSignedStart(builder, argument_pointers,
                                     instruction->operand(dimension + 1),
                                     slice->start_limits[dimension],
                                     scalar_copies, raw_start_cache));
          starts.push_back(start);
        }
        Value output_columns = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), slice->output_dimensions.back());
        auto map_input_offset = [&](Value output_offset) {
          Value remaining = output_offset;
          Value input_offset = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
            Value coordinate = remaining;
            if (dimension != 0) {
              Value dimension_size = mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  slice->output_dimensions[dimension]);
              coordinate = mlir::arith::RemUIOp::create(builder, remaining,
                                                        dimension_size);
              remaining = mlir::arith::DivUIOp::create(builder, remaining,
                                                       dimension_size);
            }
            coordinate = mlir::arith::AddIOp::create(builder, coordinate,
                                                     starts[dimension]);
            if (slice->input_strides[dimension] != 1) {
              coordinate = mlir::arith::MulIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(),
                      slice->input_strides[dimension]));
            }
            input_offset =
                mlir::arith::AddIOp::create(builder, input_offset, coordinate);
          }
          return input_offset;
        };
        Value output_column = mlir::arith::RemUIOp::create(
            builder, element_offset, output_columns);
        Value vector_column_end = mlir::arith::AddIOp::create(
            builder, output_column,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               vector_width));
        Value fits_in_row = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule, vector_column_end,
            output_columns);
        mlir::scf::IfOp select = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{vector_type}, fits_in_row,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          Value input_offset = map_input_offset(element_offset);
          absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
          TF_ASSIGN_OR_RETURN(
              Value vector,
              EmitVector(builder, argument_pointers, instruction->operand(0),
                         input_offset, predicate, copy_atoms, vector_layout,
                         vector_width, operand_cache));
          mlir::scf::YieldOp::create(builder, vector);

          builder.setInsertionPointToStart(select.elseBlock());
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lane_offset = element_offset;
            if (lane != 0) {
              lane_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), lane));
            }
            Value scalar_input_offset = map_input_offset(lane_offset);
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value scalar,
                EmitVector(builder, argument_pointers, instruction->operand(0),
                           scalar_input_offset, predicate,
                           scalar_copies.copy_atoms, scalar_copies.layout,
                           /*vector_width=*/1, operand_cache));
            lanes.push_back(
                mlir::vector::ExtractOp::create(builder, scalar, 0));
          }
          Value boundary =
              mlir::vector::FromElementsOp::create(builder, vector_type, lanes);
          mlir::scf::YieldOp::create(builder, boundary);
        }
        builder.setInsertionPointAfter(select);
        result = select.getResult(0);
        break;
      }
      case HloOpcode::kGather: {
        std::optional<RowGather> row_gather = GetRowGather(instruction);
        std::optional<StridedAxisGather> axis_gather =
            GetStridedAxisGather(instruction);
        TF_RET_CHECK(row_gather.has_value() || axis_gather.has_value());
        ScalarCopyContext scalar_copies =
            CreateScalarCopyContext(builder, copy_atoms);
        const int64_t contiguous_elements =
            row_gather ? row_gather->row_elements
                       : axis_gather->suffix_elements;
        Value row_elements = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), contiguous_elements);
        Value input_row_elements = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(),
            row_gather ? row_gather->input_row_elements
                       : axis_gather->suffix_elements);
        auto map_input_offset =
            [&](Value output_offset) -> absl::StatusOr<Value> {
          Value output_row = mlir::arith::DivUIOp::create(
              builder, output_offset, row_elements);
          Value output_column = mlir::arith::RemUIOp::create(
              builder, output_offset, row_elements);
          Value zero = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          if (axis_gather) {
            Value index_offset = zero;
            Value prefix = output_row;
            if (axis_gather->index_rows != 1) {
              Value index_rows = mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(), axis_gather->index_rows);
              index_offset = mlir::arith::RemUIOp::create(
                  builder, output_row, index_rows);
              prefix = mlir::arith::DivUIOp::create(builder, output_row,
                                                    index_rows);
            }
            absl::flat_hash_map<const HloInstruction*, Value> index_cache;
            TF_ASSIGN_OR_RETURN(
                Value index_vector,
                EmitVector(builder, argument_pointers,
                           instruction->operand(1), index_offset, predicate,
                           scalar_copies.copy_atoms, scalar_copies.layout,
                           /*vector_width=*/1, index_cache));
            Value index =
                mlir::vector::ExtractOp::create(builder, index_vector, 0);
            if (instruction->operand(1)->shape().element_type() == S32) {
              index = mlir::arith::ExtSIOp::create(
                  builder, builder.getI64Type(), index);
            }
            Value upper = mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(),
                axis_gather->indexed_dimension_size - 1);
            Value below_zero = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::slt, index, zero);
            index = mlir::arith::SelectOp::create(builder, below_zero, zero,
                                                  index);
            Value above_upper = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::sgt, index, upper);
            index = mlir::arith::SelectOp::create(builder, above_upper, upper,
                                                  index);
            Value input_row = mlir::arith::AddIOp::create(
                builder,
                mlir::arith::MulIOp::create(
                    builder, prefix,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(),
                        axis_gather->indexed_dimension_size)),
                index);
            Value input_offset = mlir::arith::MulIOp::create(
                builder, input_row, input_row_elements);
            return mlir::arith::AddIOp::create(builder, input_offset,
                                               output_column)
                .getResult();
          }

          Value index_base = output_row;
          if (row_gather->index_components != 1) {
            index_base = mlir::arith::MulIOp::create(
                builder, output_row,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(),
                    row_gather->index_components));
          }
          Value input_row = zero;
          for (int64_t component = 0;
               component < row_gather->index_components;
               ++component) {
            Value index_offset = index_base;
            if (component != 0) {
              index_offset = mlir::arith::AddIOp::create(
                  builder, index_base,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), component));
            }
            absl::flat_hash_map<const HloInstruction*, Value> index_cache;
            TF_ASSIGN_OR_RETURN(
                Value index_vector,
                EmitVector(builder, argument_pointers,
                           instruction->operand(1), index_offset, predicate,
                           scalar_copies.copy_atoms, scalar_copies.layout,
                           /*vector_width=*/1, index_cache));
            Value index =
                mlir::vector::ExtractOp::create(builder, index_vector, 0);
            if (instruction->operand(1)->shape().element_type() == S32) {
              index = mlir::arith::ExtSIOp::create(
                  builder, builder.getI64Type(), index);
            }
            Value dimension_size = mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(),
                row_gather->indexed_dimension_sizes[component]);
            Value upper = mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(),
                row_gather->indexed_dimension_sizes[component] - 1);
            Value below_zero = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::slt, index, zero);
            index = mlir::arith::SelectOp::create(builder, below_zero, zero,
                                                  index);
            Value above_upper = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::sgt, index, upper);
            index = mlir::arith::SelectOp::create(builder, above_upper, upper,
                                                  index);
            input_row = mlir::arith::AddIOp::create(
                builder,
                mlir::arith::MulIOp::create(builder, input_row,
                                            dimension_size),
                index);
          }
          Value input_offset =
              mlir::arith::MulIOp::create(builder, input_row,
                                          input_row_elements);
          return mlir::arith::AddIOp::create(builder, input_offset,
                                             output_column)
              .getResult();
        };

        Value output_column = mlir::arith::RemUIOp::create(
            builder, element_offset, row_elements);
        Value vector_column_end = mlir::arith::AddIOp::create(
            builder, output_column,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               vector_width));
        Value fits_in_row = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule, vector_column_end,
            row_elements);
        mlir::scf::IfOp select = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{vector_type}, fits_in_row,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          TF_ASSIGN_OR_RETURN(Value input_offset,
                              map_input_offset(element_offset));
          absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
          TF_ASSIGN_OR_RETURN(
              Value vector,
              EmitVector(builder, argument_pointers, instruction->operand(0),
                         input_offset, predicate, copy_atoms, vector_layout,
                         vector_width, operand_cache));
          mlir::scf::YieldOp::create(builder, vector);

          builder.setInsertionPointToStart(select.elseBlock());
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lane_offset = element_offset;
            if (lane != 0) {
              lane_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), lane));
            }
            TF_ASSIGN_OR_RETURN(Value input_offset,
                                map_input_offset(lane_offset));
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value scalar,
                EmitVector(builder, argument_pointers,
                           instruction->operand(0), input_offset, predicate,
                           scalar_copies.copy_atoms, scalar_copies.layout,
                           /*vector_width=*/1, operand_cache));
            lanes.push_back(
                mlir::vector::ExtractOp::create(builder, scalar, 0));
          }
          Value boundary =
              mlir::vector::FromElementsOp::create(builder, vector_type, lanes);
          mlir::scf::YieldOp::create(builder, boundary);
        }
        builder.setInsertionPointAfter(select);
        result = select.getResult(0);
        break;
      }
      case HloOpcode::kDynamicUpdateSlice: {
        std::optional<DynamicUpdateSlice> update =
            GetDynamicUpdateSlice(instruction);
        TF_RET_CHECK(update.has_value());
        const int64_t rank = update->input_dimensions.size();
        ScalarCopyContext scalar_copies =
            CreateScalarCopyContext(builder, copy_atoms);
        absl::flat_hash_map<const HloInstruction*, Value> raw_start_cache;
        llvm::SmallVector<Value> starts;
        llvm::SmallVector<Value> ends;
        starts.reserve(rank);
        ends.reserve(rank);
        for (int64_t dimension = 0; dimension < rank; ++dimension) {
          TF_ASSIGN_OR_RETURN(
              Value start,
              EmitClampedSignedStart(builder, argument_pointers,
                                     instruction->operand(dimension + 2),
                                     update->start_limits[dimension],
                                     scalar_copies, raw_start_cache));
          starts.push_back(start);
          ends.push_back(mlir::arith::AddIOp::create(
              builder, start,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  update->update_dimensions[dimension])));
        }
        Value input_columns = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), update->input_dimensions.back());
        auto decode_input_coordinates = [&](Value flat_offset) {
          llvm::SmallVector<Value> coordinates(rank);
          Value remaining = flat_offset;
          for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
            Value coordinate = remaining;
            if (dimension != 0) {
              Value dimension_size = mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  update->input_dimensions[dimension]);
              coordinate = mlir::arith::RemUIOp::create(builder, remaining,
                                                        dimension_size);
              remaining = mlir::arith::DivUIOp::create(builder, remaining,
                                                       dimension_size);
            }
            coordinates[dimension] = coordinate;
          }
          return coordinates;
        };
        auto is_inside_update = [&](llvm::ArrayRef<Value> coordinates,
                                    int64_t dimensions) {
          Value inside = mlir::arith::ConstantIntOp::create(
              builder, builder.getI1Type(), 1);
          for (int64_t dimension = 0; dimension < dimensions; ++dimension) {
            Value after_begin = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::uge,
                coordinates[dimension], starts[dimension]);
            Value before_end = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ult,
                coordinates[dimension], ends[dimension]);
            inside = mlir::arith::AndIOp::create(
                builder, inside,
                mlir::arith::AndIOp::create(builder, after_begin, before_end));
          }
          return inside;
        };
        auto map_update_offset = [&](llvm::ArrayRef<Value> coordinates) {
          Value update_offset = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          for (int64_t dimension = 0; dimension < rank; ++dimension) {
            Value coordinate = mlir::arith::SubIOp::create(
                builder, coordinates[dimension], starts[dimension]);
            if (update->update_strides[dimension] != 1) {
              coordinate = mlir::arith::MulIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(),
                      update->update_strides[dimension]));
            }
            update_offset =
                mlir::arith::AddIOp::create(builder, update_offset, coordinate);
          }
          return update_offset;
        };

        llvm::SmallVector<Value> output_coordinates =
            decode_input_coordinates(element_offset);
        Value output_column = output_coordinates.back();
        Value vector_column_end = mlir::arith::AddIOp::create(
            builder, output_column,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               vector_width));
        Value fits_in_row = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule, vector_column_end,
            input_columns);
        Value outer_is_update = is_inside_update(output_coordinates, rank - 1);
        Value starts_in_update = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::uge, output_column,
            starts.back());
        Value ends_in_update = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule, vector_column_end,
            ends.back());
        Value is_update = mlir::arith::AndIOp::create(
            builder, fits_in_row,
            mlir::arith::AndIOp::create(
                builder, outer_is_update,
                mlir::arith::AndIOp::create(builder, starts_in_update,
                                            ends_in_update)));
        mlir::scf::IfOp select = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{vector_type}, is_update,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          Value update_offset = map_update_offset(output_coordinates);
          absl::flat_hash_map<const HloInstruction*, Value> update_cache;
          TF_ASSIGN_OR_RETURN(
              Value update_value,
              EmitVector(builder, argument_pointers, instruction->operand(1),
                         update_offset, predicate, copy_atoms, vector_layout,
                         vector_width, update_cache));
          mlir::scf::YieldOp::create(builder, update_value);

          builder.setInsertionPointToStart(select.elseBlock());
          Value before_update = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, vector_column_end,
              starts.back());
          Value after_update = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::uge, output_column,
              ends.back());
          Value outer_is_base =
              mlir::arith::XOrIOp::create(builder, outer_is_update,
                                          mlir::arith::ConstantIntOp::create(
                                              builder, builder.getI1Type(), 1));
          Value columns_are_base =
              mlir::arith::OrIOp::create(builder, before_update, after_update);
          Value is_base = mlir::arith::AndIOp::create(
              builder, fits_in_row,
              mlir::arith::OrIOp::create(builder, outer_is_base,
                                         columns_are_base));
          mlir::scf::IfOp base_or_boundary = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{vector_type}, is_base,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard base_guard(builder);
            builder.setInsertionPointToStart(base_or_boundary.thenBlock());
            absl::flat_hash_map<const HloInstruction*, Value> base_cache;
            TF_ASSIGN_OR_RETURN(
                Value base_value,
                EmitVector(builder, argument_pointers, instruction->operand(0),
                           element_offset, predicate, copy_atoms, vector_layout,
                           vector_width, base_cache));
            mlir::scf::YieldOp::create(builder, base_value);

            builder.setInsertionPointToStart(base_or_boundary.elseBlock());
            auto scalar_vector_type = mlir::VectorType::get(
                {1},
                emitters::PrimitiveTypeToMlirType(instruction_type, builder));
            llvm::SmallVector<Value> lanes;
            lanes.reserve(vector_width);
            for (int64_t lane = 0; lane < vector_width; ++lane) {
              Value lane_offset = element_offset;
              if (lane != 0) {
                lane_offset = mlir::arith::AddIOp::create(
                    builder, element_offset,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(), lane));
              }
              llvm::SmallVector<Value> lane_coordinates =
                  decode_input_coordinates(lane_offset);
              Value lane_is_update = is_inside_update(lane_coordinates, rank);
              mlir::scf::IfOp lane_select = mlir::scf::IfOp::create(
                  builder, mlir::TypeRange{scalar_vector_type}, lane_is_update,
                  /*withElseRegion=*/true);
              {
                mlir::OpBuilder::InsertionGuard lane_guard(builder);
                builder.setInsertionPointToStart(lane_select.thenBlock());
                Value update_offset = map_update_offset(lane_coordinates);
                absl::flat_hash_map<const HloInstruction*, Value> update_cache;
                TF_ASSIGN_OR_RETURN(
                    Value update_lane,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(1), update_offset,
                               predicate, scalar_copies.copy_atoms,
                               scalar_copies.layout, /*vector_width=*/1,
                               update_cache));
                mlir::scf::YieldOp::create(builder, update_lane);

                builder.setInsertionPointToStart(lane_select.elseBlock());
                absl::flat_hash_map<const HloInstruction*, Value> base_cache;
                TF_ASSIGN_OR_RETURN(
                    Value base_lane,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(0), lane_offset, predicate,
                               scalar_copies.copy_atoms, scalar_copies.layout,
                               /*vector_width=*/1, base_cache));
                mlir::scf::YieldOp::create(builder, base_lane);
              }
              builder.setInsertionPointAfter(lane_select);
              lanes.push_back(mlir::vector::ExtractOp::create(
                  builder, lane_select.getResult(0), 0));
            }
            Value boundary = mlir::vector::FromElementsOp::create(
                builder, vector_type, lanes);
            mlir::scf::YieldOp::create(builder, boundary);
          }
          builder.setInsertionPointAfter(base_or_boundary);
          mlir::scf::YieldOp::create(builder, base_or_boundary.getResult(0));
        }
        builder.setInsertionPointAfter(select);
        result = select.getResult(0);
        break;
      }
      case HloOpcode::kReverse: {
        if (IsFlatReverse(instruction)) {
          const int64_t elements = ShapeUtil::ElementsIn(instruction->shape());
          Value input_offset = mlir::arith::SubIOp::create(
              builder,
              mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                                 elements - vector_width),
              element_offset);
          absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
          TF_ASSIGN_OR_RETURN(
              Value forward,
              EmitVector(builder, argument_pointers, instruction->operand(0),
                         input_offset, predicate, copy_atoms, vector_layout,
                         vector_width, operand_cache));
          llvm::SmallVector<int64_t> reversed_lanes;
          reversed_lanes.reserve(vector_width);
          for (int64_t lane = vector_width - 1; lane >= 0; --lane) {
            reversed_lanes.push_back(lane);
          }
          result = mlir::vector::ShuffleOp::create(
              builder, vector_type, forward, forward, reversed_lanes);
          break;
        }

        std::optional<PartialReverse> reverse = GetPartialReverse(instruction);
        TF_RET_CHECK(reverse.has_value());
        const int64_t rank = reverse->dimensions.size();
        Value columns = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), reverse->dimensions.back());
        auto map_input_offset = [&](Value output_offset) {
          Value remaining = output_offset;
          Value input_offset = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
            Value coordinate = remaining;
            if (dimension != 0) {
              Value dimension_size = mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  reverse->dimensions[dimension]);
              coordinate = mlir::arith::RemUIOp::create(builder, remaining,
                                                        dimension_size);
              remaining = mlir::arith::DivUIOp::create(builder, remaining,
                                                       dimension_size);
            }
            if (reverse->reversed_dimensions[dimension]) {
              coordinate = mlir::arith::SubIOp::create(
                  builder,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(),
                      reverse->dimensions[dimension] - 1),
                  coordinate);
            }
            if (reverse->strides[dimension] != 1) {
              coordinate = mlir::arith::MulIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(),
                      reverse->strides[dimension]));
            }
            input_offset =
                mlir::arith::AddIOp::create(builder, input_offset, coordinate);
          }
          return input_offset;
        };
        Value output_column =
            mlir::arith::RemUIOp::create(builder, element_offset, columns);
        Value vector_column_end = mlir::arith::AddIOp::create(
            builder, output_column,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               vector_width));
        Value fits_in_row = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule, vector_column_end,
            columns);
        mlir::scf::IfOp select = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{vector_type}, fits_in_row,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          Value input_offset = element_offset;
          if (reverse->reversed_dimensions.back() && vector_width != 1) {
            input_offset = mlir::arith::AddIOp::create(
                builder, element_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), vector_width - 1));
          }
          input_offset = map_input_offset(input_offset);
          absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
          TF_ASSIGN_OR_RETURN(
              Value vector,
              EmitVector(builder, argument_pointers, instruction->operand(0),
                         input_offset, predicate, copy_atoms, vector_layout,
                         vector_width, operand_cache));
          if (reverse->reversed_dimensions.back() && vector_width != 1) {
            llvm::SmallVector<int64_t> reversed_lanes;
            reversed_lanes.reserve(vector_width);
            for (int64_t lane = vector_width - 1; lane >= 0; --lane) {
              reversed_lanes.push_back(lane);
            }
            vector = mlir::vector::ShuffleOp::create(
                builder, vector_type, vector, vector, reversed_lanes);
          }
          mlir::scf::YieldOp::create(builder, vector);

          builder.setInsertionPointToStart(select.elseBlock());
          ScalarCopyContext scalar_copies =
              CreateScalarCopyContext(builder, copy_atoms);
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lane_offset = element_offset;
            if (lane != 0) {
              lane_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), lane));
            }
            Value scalar_input_offset = map_input_offset(lane_offset);
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value scalar,
                EmitVector(builder, argument_pointers, instruction->operand(0),
                           scalar_input_offset, predicate,
                           scalar_copies.copy_atoms, scalar_copies.layout,
                           /*vector_width=*/1, operand_cache));
            lanes.push_back(
                mlir::vector::ExtractOp::create(builder, scalar, 0));
          }
          Value boundary =
              mlir::vector::FromElementsOp::create(builder, vector_type, lanes);
          mlir::scf::YieldOp::create(builder, boundary);
        }
        builder.setInsertionPointAfter(select);
        result = select.getResult(0);
        break;
      }
      case HloOpcode::kPad: {
        std::optional<FlatPadInterval> interval =
            GetFlatPadInterval(instruction);
        if (!interval.has_value()) {
          std::optional<RectangularPad> pad = GetRectangularPad(instruction);
          TF_RET_CHECK(pad.has_value());
          const int64_t rank = pad->input_dimensions.size();
          Value output_columns = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), pad->output_dimensions.back());
          llvm::SmallVector<Value> begins;
          llvm::SmallVector<Value> span_ends;
          llvm::SmallVector<Value> steps;
          begins.reserve(rank);
          span_ends.reserve(rank);
          steps.reserve(rank);
          for (int64_t dimension = 0; dimension < rank; ++dimension) {
            begins.push_back(mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(), pad->begins[dimension]));
            span_ends.push_back(mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(), pad->span_ends[dimension]));
            steps.push_back(mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(), pad->steps[dimension]));
          }
          auto decode_output_coordinates = [&](Value flat_offset) {
            llvm::SmallVector<Value> coordinates(rank);
            Value remaining = flat_offset;
            for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
              Value coordinate = remaining;
              if (dimension != 0) {
                Value dimension_size = mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(),
                    pad->output_dimensions[dimension]);
                coordinate = mlir::arith::RemUIOp::create(builder, remaining,
                                                          dimension_size);
                remaining = mlir::arith::DivUIOp::create(builder, remaining,
                                                         dimension_size);
              }
              coordinates[dimension] = coordinate;
            }
            return coordinates;
          };
          auto map_input_offset = [&](llvm::ArrayRef<Value> coordinates) {
            Value input_offset = mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(), 0);
            for (int64_t dimension = 0; dimension < rank; ++dimension) {
              Value coordinate = mlir::arith::SubIOp::create(
                  builder, coordinates[dimension], begins[dimension]);
              coordinate = mlir::arith::DivUIOp::create(builder, coordinate,
                                                        steps[dimension]);
              if (pad->input_strides[dimension] != 1) {
                coordinate = mlir::arith::MulIOp::create(
                    builder, coordinate,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(),
                        pad->input_strides[dimension]));
              }
              input_offset = mlir::arith::AddIOp::create(builder, input_offset,
                                                         coordinate);
            }
            return input_offset;
          };
          auto is_in_input_dimension = [&](Value coordinate,
                                           int64_t dimension) {
            Value starts_in_input = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::sge, coordinate,
                begins[dimension]);
            Value ends_in_input = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::slt, coordinate,
                span_ends[dimension]);
            Value relative = mlir::arith::SubIOp::create(builder, coordinate,
                                                         begins[dimension]);
            Value remainder = mlir::arith::RemUIOp::create(builder, relative,
                                                           steps[dimension]);
            Value aligned = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::eq, remainder,
                mlir::arith::ConstantIntOp::create(builder,
                                                   builder.getI64Type(), 0));
            return mlir::arith::AndIOp::create(
                       builder, aligned,
                       mlir::arith::AndIOp::create(builder, starts_in_input,
                                                   ends_in_input))
                .getResult();
          };
          auto is_in_input_dimensions = [&](llvm::ArrayRef<Value> coordinates,
                                            int64_t dimensions) {
            Value is_input = mlir::arith::ConstantIntOp::create(
                builder, builder.getI1Type(), 1);
            for (int64_t dimension = 0; dimension < dimensions; ++dimension) {
              is_input = mlir::arith::AndIOp::create(
                  builder, is_input,
                  is_in_input_dimension(coordinates[dimension], dimension));
            }
            return is_input;
          };

          llvm::SmallVector<Value> output_coordinates =
              decode_output_coordinates(element_offset);
          Value output_column = output_coordinates.back();
          Value vector_column_end = mlir::arith::AddIOp::create(
              builder, output_column,
              mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                                 vector_width));
          Value fits_in_row = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, vector_column_end,
              output_columns);
          Value outer_is_input =
              is_in_input_dimensions(output_coordinates, rank - 1);
          Value columns_are_input = mlir::arith::ConstantIntOp::create(
              builder, builder.getI1Type(), 0);
          if (pad->steps.back() == 1) {
            Value starts_in_input_column = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::sge, output_column,
                begins.back());
            Value ends_in_input_column = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::sle, vector_column_end,
                span_ends.back());
            columns_are_input = mlir::arith::AndIOp::create(
                builder, starts_in_input_column, ends_in_input_column);
          }
          Value is_input = mlir::arith::AndIOp::create(
              builder, fits_in_row,
              mlir::arith::AndIOp::create(builder, outer_is_input,
                                          columns_are_input));

          auto emit_padding_or_boundary = [&]() -> absl::StatusOr<Value> {
            Value before_input_columns = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::sle, vector_column_end,
                begins.back());
            Value after_input_columns = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::sge, output_column,
                span_ends.back());
            Value outside_input_columns = mlir::arith::OrIOp::create(
                builder, before_input_columns, after_input_columns);
            Value outer_is_padding = mlir::arith::XOrIOp::create(
                builder, outer_is_input,
                mlir::arith::ConstantIntOp::create(builder, builder.getI1Type(),
                                                   1));
            Value is_padding = mlir::arith::AndIOp::create(
                builder, fits_in_row,
                mlir::arith::OrIOp::create(builder, outer_is_padding,
                                           outside_input_columns));
            mlir::scf::IfOp padding_or_boundary = mlir::scf::IfOp::create(
                builder, mlir::TypeRange{vector_type}, is_padding,
                /*withElseRegion=*/true);
            {
              mlir::OpBuilder::InsertionGuard padding_guard(builder);
              builder.setInsertionPointToStart(padding_or_boundary.thenBlock());
              absl::flat_hash_map<const HloInstruction*, Value> padding_cache;
              TF_ASSIGN_OR_RETURN(
                  Value padding_value,
                  EmitVector(builder, argument_pointers,
                             instruction->operand(1), element_offset, predicate,
                             copy_atoms, vector_layout, vector_width,
                             padding_cache));
              mlir::scf::YieldOp::create(builder, padding_value);

              builder.setInsertionPointToStart(padding_or_boundary.elseBlock());
              ScalarCopyContext scalar_copies =
                  CreateScalarCopyContext(builder, copy_atoms);
              auto scalar_vector_type = mlir::VectorType::get(
                  {1},
                  emitters::PrimitiveTypeToMlirType(instruction_type, builder));
              llvm::SmallVector<Value> lanes;
              lanes.reserve(vector_width);
              for (int64_t lane = 0; lane < vector_width; ++lane) {
                Value lane_offset = element_offset;
                if (lane != 0) {
                  lane_offset = mlir::arith::AddIOp::create(
                      builder, element_offset,
                      mlir::arith::ConstantIntOp::create(
                          builder, builder.getI64Type(), lane));
                }
                llvm::SmallVector<Value> lane_coordinates =
                    decode_output_coordinates(lane_offset);
                Value lane_is_input =
                    is_in_input_dimensions(lane_coordinates, rank);
                mlir::scf::IfOp lane_select = mlir::scf::IfOp::create(
                    builder, mlir::TypeRange{scalar_vector_type}, lane_is_input,
                    /*withElseRegion=*/true);
                {
                  mlir::OpBuilder::InsertionGuard lane_guard(builder);
                  builder.setInsertionPointToStart(lane_select.thenBlock());
                  Value scalar_input_offset =
                      map_input_offset(lane_coordinates);
                  absl::flat_hash_map<const HloInstruction*, Value> input_cache;
                  TF_ASSIGN_OR_RETURN(
                      Value input_lane,
                      EmitVector(builder, argument_pointers,
                                 instruction->operand(0), scalar_input_offset,
                                 predicate, scalar_copies.copy_atoms,
                                 scalar_copies.layout, /*vector_width=*/1,
                                 input_cache));
                  mlir::scf::YieldOp::create(builder, input_lane);

                  builder.setInsertionPointToStart(lane_select.elseBlock());
                  absl::flat_hash_map<const HloInstruction*, Value>
                      padding_cache;
                  TF_ASSIGN_OR_RETURN(
                      Value padding_lane,
                      EmitVector(builder, argument_pointers,
                                 instruction->operand(1), lane_offset,
                                 predicate, scalar_copies.copy_atoms,
                                 scalar_copies.layout, /*vector_width=*/1,
                                 padding_cache));
                  mlir::scf::YieldOp::create(builder, padding_lane);
                }
                builder.setInsertionPointAfter(lane_select);
                lanes.push_back(mlir::vector::ExtractOp::create(
                    builder, lane_select.getResult(0), 0));
              }
              Value boundary = mlir::vector::FromElementsOp::create(
                  builder, vector_type, lanes);
              mlir::scf::YieldOp::create(builder, boundary);
            }
            builder.setInsertionPointAfter(padding_or_boundary);
            return padding_or_boundary.getResult(0);
          };

          mlir::scf::IfOp select = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{vector_type}, is_input,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(select.thenBlock());
            Value input_offset = map_input_offset(output_coordinates);
            absl::flat_hash_map<const HloInstruction*, Value> input_cache;
            TF_ASSIGN_OR_RETURN(
                Value input_value,
                EmitVector(builder, argument_pointers, instruction->operand(0),
                           input_offset, predicate, copy_atoms, vector_layout,
                           vector_width, input_cache));
            mlir::scf::YieldOp::create(builder, input_value);

            builder.setInsertionPointToStart(select.elseBlock());
            if (pad->steps.back() > 1 &&
                vector_width % pad->steps.back() == 0) {
              Value starts_in_input_column =
                  is_in_input_dimension(output_column, rank - 1);
              Value ends_in_input_span = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::sle, vector_column_end,
                  span_ends.back());
              Value is_packed_input = mlir::arith::AndIOp::create(
                  builder, fits_in_row,
                  mlir::arith::AndIOp::create(
                      builder, outer_is_input,
                      mlir::arith::AndIOp::create(builder,
                                                  starts_in_input_column,
                                                  ends_in_input_span)));
              mlir::scf::IfOp packed_or_other = mlir::scf::IfOp::create(
                  builder, mlir::TypeRange{vector_type}, is_packed_input,
                  /*withElseRegion=*/true);
              {
                mlir::OpBuilder::InsertionGuard packed_guard(builder);
                builder.setInsertionPointToStart(packed_or_other.thenBlock());
                const int64_t packed_width = vector_width / pad->steps.back();
                ScalarCopyContext packed_copies =
                    CreateCopyContext(builder, copy_atoms, packed_width);
                Value input_offset = map_input_offset(output_coordinates);
                absl::flat_hash_map<const HloInstruction*, Value> input_cache;
                TF_ASSIGN_OR_RETURN(
                    Value input_value,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(0), input_offset, predicate,
                               packed_copies.copy_atoms, packed_copies.layout,
                               packed_width, input_cache));
                absl::flat_hash_map<const HloInstruction*, Value> padding_cache;
                TF_ASSIGN_OR_RETURN(
                    Value padding_value,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(1), element_offset,
                               predicate, copy_atoms, vector_layout,
                               vector_width, padding_cache));
                llvm::SmallVector<Value> lanes;
                lanes.reserve(vector_width);
                for (int64_t lane = 0; lane < vector_width; ++lane) {
                  Value source = lane % pad->steps.back() == 0 ? input_value
                                                               : padding_value;
                  int64_t source_lane = lane % pad->steps.back() == 0
                                            ? lane / pad->steps.back()
                                            : lane;
                  lanes.push_back(mlir::vector::ExtractOp::create(
                      builder, source, source_lane));
                }
                Value packed = mlir::vector::FromElementsOp::create(
                    builder, vector_type, lanes);
                mlir::scf::YieldOp::create(builder, packed);

                builder.setInsertionPointToStart(packed_or_other.elseBlock());
                TF_ASSIGN_OR_RETURN(Value other, emit_padding_or_boundary());
                mlir::scf::YieldOp::create(builder, other);
              }
              builder.setInsertionPointAfter(packed_or_other);
              mlir::scf::YieldOp::create(builder, packed_or_other.getResult(0));
            } else {
              TF_ASSIGN_OR_RETURN(Value padding_or_boundary,
                                  emit_padding_or_boundary());
              mlir::scf::YieldOp::create(builder, padding_or_boundary);
            }
          }
          builder.setInsertionPointAfter(select);
          result = select.getResult(0);
          break;
        }
        TF_RET_CHECK(interval.has_value());
        Value input_begin = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), interval->input_begin);
        Value input_end = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), interval->input_end);
        Value vector_end = mlir::arith::AddIOp::create(
            builder, element_offset,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               vector_width));
        Value starts_in_input = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::uge, element_offset,
            input_begin);
        Value ends_in_input = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule, vector_end, input_end);
        Value is_input = mlir::arith::AndIOp::create(builder, starts_in_input,
                                                     ends_in_input);
        mlir::scf::IfOp select = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{vector_type}, is_input,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          Value input_offset =
              mlir::arith::SubIOp::create(builder, element_offset, input_begin);
          absl::flat_hash_map<const HloInstruction*, Value> input_cache;
          TF_ASSIGN_OR_RETURN(
              Value input_value,
              EmitVector(builder, argument_pointers, instruction->operand(0),
                         input_offset, predicate, copy_atoms, vector_layout,
                         vector_width, input_cache));
          mlir::scf::YieldOp::create(builder, input_value);

          builder.setInsertionPointToStart(select.elseBlock());
          Value before_input = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, vector_end,
              input_begin);
          Value after_input = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::uge, element_offset,
              input_end);
          Value is_padding =
              mlir::arith::OrIOp::create(builder, before_input, after_input);
          mlir::scf::IfOp padding_or_boundary = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{vector_type}, is_padding,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard padding_guard(builder);
            builder.setInsertionPointToStart(padding_or_boundary.thenBlock());
            absl::flat_hash_map<const HloInstruction*, Value> padding_cache;
            TF_ASSIGN_OR_RETURN(
                Value padding_value,
                EmitVector(builder, argument_pointers, instruction->operand(1),
                           element_offset, predicate, copy_atoms, vector_layout,
                           vector_width, padding_cache));
            mlir::scf::YieldOp::create(builder, padding_value);

            builder.setInsertionPointToStart(padding_or_boundary.elseBlock());
            ScalarCopyContext scalar_copies =
                CreateScalarCopyContext(builder, copy_atoms);
            auto scalar_vector_type = mlir::VectorType::get(
                {1},
                emitters::PrimitiveTypeToMlirType(instruction_type, builder));
            llvm::SmallVector<Value> lanes;
            lanes.reserve(vector_width);
            for (int64_t lane = 0; lane < vector_width; ++lane) {
              Value lane_offset = element_offset;
              if (lane != 0) {
                lane_offset = mlir::arith::AddIOp::create(
                    builder, element_offset,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(), lane));
              }
              Value lane_starts_in_input = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::uge, lane_offset,
                  input_begin);
              Value lane_ends_in_input = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::ult, lane_offset,
                  input_end);
              Value lane_is_input = mlir::arith::AndIOp::create(
                  builder, lane_starts_in_input, lane_ends_in_input);
              mlir::scf::IfOp lane_select = mlir::scf::IfOp::create(
                  builder, mlir::TypeRange{scalar_vector_type}, lane_is_input,
                  /*withElseRegion=*/true);
              {
                mlir::OpBuilder::InsertionGuard lane_guard(builder);
                builder.setInsertionPointToStart(lane_select.thenBlock());
                Value scalar_input_offset = mlir::arith::SubIOp::create(
                    builder, lane_offset, input_begin);
                absl::flat_hash_map<const HloInstruction*, Value> input_cache;
                TF_ASSIGN_OR_RETURN(
                    Value input_lane,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(0), scalar_input_offset,
                               predicate, scalar_copies.copy_atoms,
                               scalar_copies.layout, /*vector_width=*/1,
                               input_cache));
                mlir::scf::YieldOp::create(builder, input_lane);
                builder.setInsertionPointToStart(lane_select.elseBlock());
                absl::flat_hash_map<const HloInstruction*, Value> padding_cache;
                TF_ASSIGN_OR_RETURN(
                    Value padding_lane,
                    EmitVector(builder, argument_pointers,
                               instruction->operand(1), lane_offset, predicate,
                               scalar_copies.copy_atoms, scalar_copies.layout,
                               /*vector_width=*/1, padding_cache));
                mlir::scf::YieldOp::create(builder, padding_lane);
              }
              builder.setInsertionPointAfter(lane_select);
              lanes.push_back(mlir::vector::ExtractOp::create(
                  builder, lane_select.getResult(0), 0));
            }
            Value boundary = mlir::vector::FromElementsOp::create(
                builder, vector_type, lanes);
            mlir::scf::YieldOp::create(builder, boundary);
          }
          builder.setInsertionPointAfter(padding_or_boundary);
          mlir::scf::YieldOp::create(builder, padding_or_boundary.getResult(0));
        }
        builder.setInsertionPointAfter(select);
        result = select.getResult(0);
        break;
      }
      case HloOpcode::kConcatenate: {
        const auto* concatenate = Cast<HloConcatenateInstruction>(instruction);
        const int64_t concatenate_dimension =
            concatenate->concatenate_dimension();
        if (concatenate_dimension != 0) {
          // A non-leading concatenate interleaves operand fragments in the
          // flattened physical buffer. Select predicated scalar loads instead
          // of carrying Fly buffer descriptors through nested scf.if regions;
          // FlyToROCDL can then lower every descriptor at its use site, and
          // only the selected raw-buffer transaction executes.
          int64_t inner_elements = 1;
          for (int64_t dimension = concatenate_dimension + 1;
               dimension < instruction->shape().dimensions_size();
               ++dimension) {
            inner_elements *= instruction->shape().dimensions(dimension);
          }
          const int64_t output_outer_elements =
              instruction->shape().dimensions(concatenate_dimension) *
              inner_elements;
          ScalarCopyContext scalar_copies =
              CreateScalarCopyContext(builder, copy_atoms);
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lane_offset = element_offset;
            if (lane != 0) {
              lane_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), lane));
            }
            Value outer_index = mlir::arith::DivUIOp::create(
                builder, lane_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), output_outer_elements));
            Value within_outer = mlir::arith::RemUIOp::create(
                builder, lane_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), output_outer_elements));
            Value selected;
            int64_t operand_start = 0;
            for (const HloInstruction* operand : instruction->operands()) {
              const int64_t operand_elements =
                  operand->shape().dimensions(concatenate_dimension) *
                  inner_elements;
              const int64_t operand_end = operand_start + operand_elements;
              Value below_end = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::ult, within_outer,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), operand_end));
              Value belongs = below_end;
              if (operand_start != 0) {
                Value at_or_above_start = mlir::arith::CmpIOp::create(
                    builder, mlir::arith::CmpIPredicate::uge, within_outer,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(), operand_start));
                belongs = mlir::arith::AndIOp::create(
                    builder, at_or_above_start, below_end);
              }
              Value within_operand = within_outer;
              if (operand_start != 0) {
                within_operand = mlir::arith::SubIOp::create(
                    builder, within_outer,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(), operand_start));
              }
              Value operand_offset = mlir::arith::AddIOp::create(
                  builder,
                  mlir::arith::MulIOp::create(
                      builder, outer_index,
                      mlir::arith::ConstantIntOp::create(
                          builder, builder.getI64Type(), operand_elements)),
                  within_operand);
              Value load_predicate = mlir::arith::AndIOp::create(
                  builder, predicate, belongs);
              absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
              TF_ASSIGN_OR_RETURN(
                  Value loaded,
                  EmitVector(builder, argument_pointers, operand,
                             operand_offset, load_predicate,
                             scalar_copies.copy_atoms, scalar_copies.layout,
                             /*vector_width=*/1, operand_cache));
              Value scalar =
                  mlir::vector::ExtractOp::create(builder, loaded, 0);
              selected = selected ? mlir::arith::SelectOp::create(
                                        builder, belongs, scalar, selected)
                                        .getResult()
                                  : scalar;
              operand_start = operand_end;
            }
            lanes.push_back(selected);
          }
          result = mlir::vector::FromElementsOp::create(builder, vector_type,
                                                        lanes);
          break;
        }
        ScalarCopyContext scalar_copies =
            CreateScalarCopyContext(builder, copy_atoms);

        // The AMD GPU control-flow conversion cannot carry vector<i1> block
        // arguments. Predicate buffers are byte-backed at the XLA ABI, so
        // retain that i8 storage representation through the concatenate's
        // nested scf.if regions and recover logical predicates only after the
        // selected fragment leaves control flow.
        const bool predicate_concatenate = instruction_type == PRED;
        mlir::Type control_flow_element_type =
            predicate_concatenate
                ? mlir::Type(builder.getI8Type())
                : emitters::PrimitiveTypeToMlirType(instruction_type, builder);
        auto control_flow_vector_type = mlir::VectorType::get(
            {vector_width}, control_flow_element_type);
        auto scalar_vector_type =
            mlir::VectorType::get({1}, control_flow_element_type);
        auto to_control_flow_value = [&](Value value) -> Value {
          if (!predicate_concatenate) {
            return value;
          }
          auto value_type = mlir::cast<mlir::VectorType>(value.getType());
          auto storage_type =
              mlir::VectorType::get(value_type.getShape(), builder.getI8Type());
          return mlir::arith::ExtUIOp::create(builder, storage_type, value);
        };
        auto from_control_flow_value = [&](Value value) -> Value {
          if (!predicate_concatenate) {
            return value;
          }
          auto value_type = mlir::cast<mlir::VectorType>(value.getType());
          Value zero = mlir::arith::ConstantOp::create(
              builder, value_type,
              mlir::DenseElementsAttr::get(
                  value_type,
                  builder.getIntegerAttr(value_type.getElementType(), 0)));
          return mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ne, value, zero);
        };
        std::function<absl::StatusOr<Value>(int64_t, int64_t, Value)>
            emit_scalar_operand =
                [&](int64_t operand_index, int64_t operand_start,
                    Value scalar_offset) -> absl::StatusOr<Value> {
          const HloInstruction* operand = instruction->operand(operand_index);
          Value operand_offset = scalar_offset;
          if (operand_start != 0) {
            operand_offset = mlir::arith::SubIOp::create(
                builder, scalar_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), operand_start));
          }
          if (operand_index + 1 == instruction->operand_count()) {
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value selected,
                EmitVector(builder, argument_pointers, operand, operand_offset,
                           predicate, scalar_copies.copy_atoms,
                           scalar_copies.layout, /*vector_width=*/1,
                           operand_cache));
            return to_control_flow_value(selected);
          }
          const int64_t operand_end =
              operand_start + ShapeUtil::ElementsIn(operand->shape());
          Value belongs_to_operand = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, scalar_offset,
              mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                                 operand_end));
          mlir::scf::IfOp select = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{scalar_vector_type}, belongs_to_operand,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(select.thenBlock());
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value selected,
                EmitVector(builder, argument_pointers, operand, operand_offset,
                           predicate, scalar_copies.copy_atoms,
                           scalar_copies.layout, /*vector_width=*/1,
                           operand_cache));
            mlir::scf::YieldOp::create(builder,
                                       to_control_flow_value(selected));
            builder.setInsertionPointToStart(select.elseBlock());
            TF_ASSIGN_OR_RETURN(
                Value remaining,
                emit_scalar_operand(operand_index + 1, operand_end,
                                    scalar_offset));
            mlir::scf::YieldOp::create(builder, remaining);
          }
          builder.setInsertionPointAfter(select);
          return select.getResult(0);
        };

        auto emit_boundary = [&]() -> absl::StatusOr<Value> {
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lane_offset = element_offset;
            if (lane != 0) {
              lane_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), lane));
            }
            TF_ASSIGN_OR_RETURN(
                Value scalar,
                emit_scalar_operand(/*operand_index=*/0, /*operand_start=*/0,
                                    lane_offset));
            lanes.push_back(
                mlir::vector::ExtractOp::create(builder, scalar, 0));
          }
          return mlir::vector::FromElementsOp::create(
                     builder, control_flow_vector_type, lanes)
              .getResult();
        };

        std::function<absl::StatusOr<Value>(int64_t, int64_t)> emit_operand =
            [&](int64_t operand_index,
                int64_t operand_start) -> absl::StatusOr<Value> {
          const HloInstruction* operand = instruction->operand(operand_index);
          Value operand_offset = element_offset;
          if (operand_start != 0) {
            operand_offset = mlir::arith::SubIOp::create(
                builder, element_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), operand_start));
          }
          if (operand_index + 1 == instruction->operand_count()) {
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value selected,
                EmitVector(builder, argument_pointers, operand, operand_offset,
                           predicate, copy_atoms, vector_layout, vector_width,
                           operand_cache));
            return to_control_flow_value(selected);
          }

          const int64_t operand_end =
              operand_start + ShapeUtil::ElementsIn(operand->shape());
          Value vector_end = mlir::arith::AddIOp::create(
              builder, element_offset,
              mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                                 vector_width));
          Value fits_in_operand = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, vector_end,
              mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                                 operand_end));
          mlir::scf::IfOp select = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{control_flow_vector_type},
              fits_in_operand,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(select.thenBlock());
            absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
            TF_ASSIGN_OR_RETURN(
                Value selected,
                EmitVector(builder, argument_pointers, operand, operand_offset,
                           predicate, copy_atoms, vector_layout, vector_width,
                           operand_cache));
            mlir::scf::YieldOp::create(builder,
                                       to_control_flow_value(selected));
            builder.setInsertionPointToStart(select.elseBlock());
            Value starts_in_operand = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ult, element_offset,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), operand_end));
            mlir::scf::IfOp boundary = mlir::scf::IfOp::create(
                builder, mlir::TypeRange{control_flow_vector_type},
                starts_in_operand,
                /*withElseRegion=*/true);
            {
              mlir::OpBuilder::InsertionGuard boundary_guard(builder);
              builder.setInsertionPointToStart(boundary.thenBlock());
              TF_ASSIGN_OR_RETURN(Value boundary_value, emit_boundary());
              mlir::scf::YieldOp::create(builder, boundary_value);
              builder.setInsertionPointToStart(boundary.elseBlock());
              TF_ASSIGN_OR_RETURN(Value remaining,
                                  emit_operand(operand_index + 1, operand_end));
              mlir::scf::YieldOp::create(builder, remaining);
            }
            builder.setInsertionPointAfter(boundary);
            mlir::scf::YieldOp::create(builder, boundary.getResult(0));
          }
          builder.setInsertionPointAfter(select);
          return select.getResult(0);
        };
        TF_ASSIGN_OR_RETURN(result, emit_operand(/*operand_index=*/0,
                                                 /*operand_start=*/0));
        result = from_control_flow_value(result);
        break;
      }
      case HloOpcode::kReduceWindow: {
        std::optional<ReduceWindowDescriptor> descriptor =
            GetReduceWindowDescriptor(instruction);
        TF_RET_CHECK(descriptor.has_value());
        const int64_t rank = descriptor->input_dimensions.size();
        TF_RET_CHECK(rank > 0);

        TF_ASSIGN_OR_RETURN(
            result,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));

        auto decode_output_coordinates = [&](Value flat_offset) {
          llvm::SmallVector<Value> coordinates(rank);
          Value remaining = flat_offset;
          for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
            Value coordinate = remaining;
            if (dimension != 0) {
              Value extent = mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  descriptor->output_dimensions[dimension]);
              coordinate =
                  mlir::arith::RemUIOp::create(builder, remaining, extent);
              remaining =
                  mlir::arith::DivUIOp::create(builder, remaining, extent);
            }
            coordinates[dimension] = coordinate;
          }
          return coordinates;
        };

        auto map_input = [&](llvm::ArrayRef<Value> output_coordinates,
                             llvm::ArrayRef<int64_t> window_coordinates,
                             int64_t minor_extent) {
          Value input_offset = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          Value valid = mlir::arith::ConstantIntOp::create(
              builder, builder.getI1Type(), 1);
          for (int64_t dimension = 0; dimension < rank; ++dimension) {
            Value coordinate = output_coordinates[dimension];
            if (descriptor->window_strides[dimension] != 1) {
              coordinate = mlir::arith::MulIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(),
                      descriptor->window_strides[dimension]));
            }
            const int64_t window_offset =
                window_coordinates[dimension] *
                descriptor->window_dilations[dimension];
            if (window_offset != 0) {
              coordinate = mlir::arith::AddIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), window_offset));
            }
            if (descriptor->padding_low[dimension] != 0) {
              coordinate = mlir::arith::SubIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(),
                      descriptor->padding_low[dimension]));
            }
            if (descriptor->base_dilations[dimension] != 1) {
              Value dilation = mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  descriptor->base_dilations[dimension]);
              Value remainder =
                  mlir::arith::RemSIOp::create(builder, coordinate, dilation);
              Value divisible = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::eq, remainder,
                  mlir::arith::ConstantIntOp::create(builder,
                                                     builder.getI64Type(), 0));
              valid = mlir::arith::AndIOp::create(builder, valid, divisible);
              coordinate =
                  mlir::arith::DivSIOp::create(builder, coordinate, dilation);
            }
            Value nonnegative = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::sge, coordinate,
                mlir::arith::ConstantIntOp::create(builder,
                                                   builder.getI64Type(), 0));
            Value coordinate_end = coordinate;
            if (dimension == rank - 1 && minor_extent != 1) {
              coordinate_end = mlir::arith::AddIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), minor_extent - 1));
            }
            Value below_extent = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::slt, coordinate_end,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(),
                    descriptor->input_dimensions[dimension]));
            valid = mlir::arith::AndIOp::create(
                builder, valid,
                mlir::arith::AndIOp::create(builder, nonnegative,
                                            below_extent));
            if (descriptor->input_strides[dimension] != 1) {
              coordinate = mlir::arith::MulIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(),
                      descriptor->input_strides[dimension]));
            }
            input_offset =
                mlir::arith::AddIOp::create(builder, input_offset, coordinate);
          }
          return std::pair<Value, Value>{input_offset, valid};
        };

        llvm::SmallVector<Value> output_coordinates =
            decode_output_coordinates(element_offset);
        Value output_minor_end = mlir::arith::AddIOp::create(
            builder, output_coordinates.back(),
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               vector_width));
        Value stays_in_output_row = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule, output_minor_end,
            mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(),
                descriptor->output_dimensions.back()));
        const bool contiguous_minor = descriptor->window_strides.back() == 1 &&
                                      descriptor->base_dilations.back() == 1;
        bool minor_sliding_window = contiguous_minor;
        for (int64_t dimension = 0; dimension + 1 < rank; ++dimension) {
          minor_sliding_window &= descriptor->window_sizes[dimension] == 1;
        }
        ScalarCopyContext scalar_copies =
            CreateScalarCopyContext(builder, copy_atoms);

        llvm::SmallVector<llvm::SmallVector<int64_t>> window_points;
        window_points.reserve(descriptor->window_elements);
        for (int64_t window_linear = 0;
             window_linear < descriptor->window_elements; ++window_linear) {
          llvm::SmallVector<int64_t> window_coordinates(rank);
          int64_t remaining = window_linear;
          for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
            window_coordinates[dimension] =
                remaining % descriptor->window_sizes[dimension];
            remaining /= descriptor->window_sizes[dimension];
          }
          window_points.push_back(std::move(window_coordinates));
        }

        auto emit_scalar_window = [&]() -> absl::StatusOr<Value> {
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lane_offset = element_offset;
            if (lane != 0) {
              lane_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), lane));
            }
            llvm::SmallVector<Value> lane_coordinates =
                decode_output_coordinates(lane_offset);
            Value accumulator = mlir::vector::ShuffleOp::create(
                builder,
                mlir::VectorType::get({1}, emitters::PrimitiveTypeToMlirType(
                                               instruction_type, builder)),
                result, result, llvm::SmallVector<int64_t>{lane});
            for (const llvm::SmallVector<int64_t>& window_coordinates :
                 window_points) {
              auto [input_offset, input_valid] =
                  map_input(lane_coordinates, window_coordinates,
                            /*minor_extent=*/1);
              Value lane_valid =
                  mlir::arith::AndIOp::create(builder, predicate, input_valid);
              absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
              TF_ASSIGN_OR_RETURN(
                  Value input_scalar,
                  EmitVector(builder, argument_pointers,
                             instruction->operand(0), input_offset, lane_valid,
                             scalar_copies.copy_atoms, scalar_copies.layout,
                             /*vector_width=*/1, operand_cache));
              TF_ASSIGN_OR_RETURN(
                  Value combined,
                  EmitReduceWindowBinary(builder, instruction_type,
                                         descriptor->reducer, accumulator,
                                         input_scalar));
              accumulator = mlir::arith::SelectOp::create(
                  builder, lane_valid, combined, accumulator);
            }
            lanes.push_back(
                mlir::vector::ExtractOp::create(builder, accumulator, 0));
          }
          return mlir::vector::FromElementsOp::create(builder, vector_type,
                                                      lanes)
              .getResult();
        };

        if (!contiguous_minor) {
          TF_ASSIGN_OR_RETURN(result, emit_scalar_window());
          break;
        }

        llvm::SmallVector<Value> vector_input_offsets;
        vector_input_offsets.reserve(window_points.size());
        Value all_vector_valid = mlir::arith::AndIOp::create(
            builder, predicate, stays_in_output_row);
        for (const llvm::SmallVector<int64_t>& window_coordinates :
             window_points) {
          auto [input_offset, input_valid] =
              map_input(output_coordinates, window_coordinates, vector_width);
          vector_input_offsets.push_back(input_offset);
          all_vector_valid = mlir::arith::AndIOp::create(
              builder, all_vector_valid, input_valid);
        }
        const int64_t maximum_minor_shift =
            (descriptor->window_sizes.back() - 1) *
            descriptor->window_dilations.back();
        const int64_t sliding_chunks =
            (vector_width + maximum_minor_shift - 1) / vector_width + 1;
        if (minor_sliding_window) {
          Value loaded_end =
              mlir::arith::AddIOp::create(builder, vector_input_offsets.front(),
                                          mlir::arith::ConstantIntOp::create(
                                              builder, builder.getI64Type(),
                                              sliding_chunks * vector_width));
          Value chunks_in_allocation = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, loaded_end,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  ShapeUtil::ElementsIn(instruction->operand(0)->shape())));
          all_vector_valid = mlir::arith::AndIOp::create(
              builder, all_vector_valid, chunks_in_allocation);
        }

        // Test the complete window once. Interior vectors then execute a
        // straight-line sequence of contiguous copies and reducer operations;
        // only a vector touching any padding, dilation hole, or physical row
        // edge enters the scalar boundary path.
        mlir::scf::IfOp select = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{vector_type}, all_vector_valid,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          Value accumulator = result;
          if (minor_sliding_window) {
            // Load the union of all overlapping minor-window vectors once,
            // then form each shifted reducer operand with register shuffles.
            // A W-wide stride-one window needs ceil((V + W - 1) / V)
            // transactions instead of W transactions for V output lanes.
            llvm::SmallVector<Value> chunks;
            chunks.reserve(sliding_chunks);
            for (int64_t chunk = 0; chunk < sliding_chunks; ++chunk) {
              Value input_offset = vector_input_offsets.front();
              if (chunk != 0) {
                input_offset = mlir::arith::AddIOp::create(
                    builder, input_offset,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(), chunk * vector_width));
              }
              absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
              TF_ASSIGN_OR_RETURN(
                  Value input_vector,
                  EmitVector(builder, argument_pointers,
                             instruction->operand(0), input_offset,
                             all_vector_valid, copy_atoms, vector_layout,
                             vector_width, operand_cache));
              chunks.push_back(input_vector);
            }
            for (int64_t point = 0; point < window_points.size(); ++point) {
              const int64_t shift = window_points[point].back() *
                                    descriptor->window_dilations.back();
              const int64_t chunk = shift / vector_width;
              const int64_t lane = shift % vector_width;
              Value input_vector = chunks[chunk];
              if (lane != 0) {
                llvm::SmallVector<int64_t> mask;
                mask.reserve(vector_width);
                for (int64_t element = 0; element < vector_width; ++element) {
                  mask.push_back(lane + element);
                }
                input_vector = mlir::vector::ShuffleOp::create(
                    builder, vector_type, chunks[chunk], chunks[chunk + 1],
                    mask);
              }
              TF_ASSIGN_OR_RETURN(accumulator, EmitReduceWindowBinary(
                                                   builder, instruction_type,
                                                   descriptor->reducer,
                                                   accumulator, input_vector));
            }
          } else {
            for (Value input_offset : vector_input_offsets) {
              absl::flat_hash_map<const HloInstruction*, Value> operand_cache;
              TF_ASSIGN_OR_RETURN(
                  Value input_vector,
                  EmitVector(builder, argument_pointers,
                             instruction->operand(0), input_offset,
                             all_vector_valid, copy_atoms, vector_layout,
                             vector_width, operand_cache));
              TF_ASSIGN_OR_RETURN(accumulator, EmitReduceWindowBinary(
                                                   builder, instruction_type,
                                                   descriptor->reducer,
                                                   accumulator, input_vector));
            }
          }
          mlir::scf::YieldOp::create(builder, accumulator);

          builder.setInsertionPointToStart(select.elseBlock());
          TF_ASSIGN_OR_RETURN(Value boundary, emit_scalar_window());
          mlir::scf::YieldOp::create(builder, boundary);
        }
        builder.setInsertionPointAfter(select);
        result = select.getResult(0);
        break;
      }
      case HloOpcode::kReduce: {
        const HloInstruction* input = instruction->operand(0);
        std::optional<StridedReductionDescriptor> descriptor =
            GetStridedReductionDescriptor(instruction, instruction->shape());
        TF_RET_CHECK(descriptor.has_value());
        const int64_t reduction_size = descriptor->reduction_size;
        const int64_t inner_elements = descriptor->inner_elements;
        Value initial =
            descriptor->type == F32
                ? SplatFloat(builder, vector_type,
                             descriptor->floating_init_value)
                : SplatInteger(builder, vector_type,
                               descriptor->integer_init_value);
        auto combine = [&](Value lhs, Value rhs) -> absl::StatusOr<Value> {
          if (descriptor->direct_binary_reducer) {
            return EmitStridedReductionBinary(builder, instruction_type,
                                              descriptor->reducer, lhs, rhs);
          }
          const HloComputation* reducer = descriptor->reducer_computation;
          TF_RET_CHECK(reducer != nullptr && reducer->num_parameters() == 2);
          absl::flat_hash_map<const HloInstruction*, Value> reducer_cache;
          reducer_cache.emplace(reducer->parameter_instruction(0), lhs);
          reducer_cache.emplace(reducer->parameter_instruction(1), rhs);
          Value zero_offset = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          return EmitVector(builder, argument_pointers,
                            reducer->root_instruction(), zero_offset,
                            predicate, copy_atoms, vector_layout, vector_width,
                            reducer_cache);
        };
        if (descriptor->direct_binary_reducer && IsSplatGraph(input)) {
          Value zero_offset = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          absl::flat_hash_map<const HloInstruction*, Value> splat_cache;
          TF_ASSIGN_OR_RETURN(
              Value splat,
              EmitVector(builder, argument_pointers, input, zero_offset,
                         predicate, copy_atoms, vector_layout, vector_width,
                         splat_cache));
          if (descriptor->reducer == HloOpcode::kAdd) {
            Value count = descriptor->type == F32
                              ? SplatFloat(builder, vector_type,
                                           static_cast<double>(reduction_size))
                              : SplatInteger(builder, vector_type,
                                             reduction_size);
            splat = descriptor->type == F32
                        ? mlir::arith::MulFOp::create(builder, splat, count)
                              .getResult()
                        : mlir::arith::MulIOp::create(builder, splat, count)
                              .getResult();
          }
          TF_ASSIGN_OR_RETURN(result, combine(initial, splat));
          break;
        }
        Value inner = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), inner_elements);
        Value input_outer_stride = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), reduction_size * inner_elements);
        auto source_base = [&](Value output_offset) {
          Value outer =
              mlir::arith::DivUIOp::create(builder, output_offset, inner);
          Value inner_offset =
              mlir::arith::RemUIOp::create(builder, output_offset, inner);
          Value outer_base =
              mlir::arith::MulIOp::create(builder, outer, input_outer_stride);
          return mlir::arith::AddIOp::create(builder, outer_base, inner_offset)
              .getResult();
        };
        auto source_offset = [&](Value base, Value part) {
          Value part_base = mlir::arith::MulIOp::create(builder, part, inner);
          return mlir::arith::AddIOp::create(builder, base, part_base)
              .getResult();
        };
        Value vector_source_base = source_base(element_offset);
        auto emit_partial = [&](Value part) -> absl::StatusOr<Value> {
          if (inner_elements % vector_width == 0) {
            Value part_offset = source_offset(vector_source_base, part);
            absl::flat_hash_map<const HloInstruction*, Value> part_cache;
            return EmitVector(builder, argument_pointers, input, part_offset,
                              predicate, copy_atoms, vector_layout,
                              vector_width, part_cache);
          }
          ScalarCopyContext scalar_copies =
              CreateScalarCopyContext(builder, copy_atoms);
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lane_output_offset = element_offset;
            if (lane != 0) {
              lane_output_offset = mlir::arith::AddIOp::create(
                  builder, element_offset,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), lane));
            }
            Value lane_input_offset =
                source_offset(source_base(lane_output_offset), part);
            absl::flat_hash_map<const HloInstruction*, Value> lane_cache;
            TF_ASSIGN_OR_RETURN(
                Value scalar,
                EmitVector(builder, argument_pointers, input, lane_input_offset,
                           predicate, scalar_copies.copy_atoms,
                           scalar_copies.layout,
                           /*vector_width=*/1, lane_cache));
            lanes.push_back(
                mlir::vector::ExtractOp::create(builder, scalar, 0));
          }
          return mlir::vector::FromElementsOp::create(builder, vector_type,
                                                      lanes)
              .getResult();
        };
        // Small split/leading dimensions benefit from straight-line loads:
        // LLVM can overlap their independent VMEM operations and there is no
        // loop-control dependency in the hot path. Keep a real loop for large
        // extents so general leading reductions do not grow code linearly.
        if (reduction_size <= 32) {
          result = initial;
          for (int64_t part = 0; part < reduction_size; ++part) {
            Value part_value = mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(), part);
            TF_ASSIGN_OR_RETURN(Value partial, emit_partial(part_value));
            TF_ASSIGN_OR_RETURN(result, combine(result, partial));
          }
          break;
        }
        constexpr int64_t kLoopUnroll = 8;
        const int64_t loop_end = reduction_size - reduction_size % kLoopUnroll;
        double neutral_value = 0.0;
        if (descriptor->reducer == HloOpcode::kMaximum) {
          neutral_value = -std::numeric_limits<double>::infinity();
        } else if (descriptor->reducer == HloOpcode::kMinimum) {
          neutral_value = std::numeric_limits<double>::infinity();
        }
        Value neutral =
            descriptor->type == F32
                ? SplatFloat(builder, vector_type, neutral_value)
                : SplatInteger(builder, vector_type, 0);
        llvm::SmallVector<Value> initial_accumulators(kLoopUnroll, neutral);
        mlir::scf::ForOp reduction = mlir::scf::ForOp::create(
            builder,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               0),
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               loop_end),
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               kLoopUnroll),
            initial_accumulators,
            [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(reduction.getBody());
          llvm::SmallVector<Value> accumulators;
          accumulators.reserve(kLoopUnroll);
          for (int64_t part = 0; part < kLoopUnroll; ++part) {
            Value part_value = reduction.getInductionVar();
            if (part != 0) {
              part_value = mlir::arith::AddIOp::create(
                  builder, part_value,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), part));
            }
            TF_ASSIGN_OR_RETURN(Value partial, emit_partial(part_value));
            TF_ASSIGN_OR_RETURN(
                Value accumulated,
                combine(reduction.getRegionIterArg(part), partial));
            accumulators.push_back(accumulated);
          }
          mlir::scf::YieldOp::create(builder, accumulators);
        }
        builder.setInsertionPointAfter(reduction);
        result = initial;
        for (int64_t accumulator = 1; accumulator < kLoopUnroll;
             ++accumulator) {
          TF_ASSIGN_OR_RETURN(result,
                              combine(result, reduction.getResult(accumulator)));
        }
        TF_ASSIGN_OR_RETURN(result,
                            combine(result, reduction.getResult(0)));
        for (int64_t part = loop_end; part < reduction_size; ++part) {
          Value part_value = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), part);
          TF_ASSIGN_OR_RETURN(Value partial, emit_partial(part_value));
          TF_ASSIGN_OR_RETURN(result, combine(result, partial));
        }
        break;
      }
      case HloOpcode::kConvert: {
        const HloInstruction* source = instruction->operand(0);
        if (instruction_type == PRED && source->opcode() == HloOpcode::kAdd &&
            IsSupportedSignedIntegerType(source->shape().element_type())) {
          const HloInstruction* converted_predicate = nullptr;
          const HloInstruction* integer_operand = nullptr;
          for (int64_t operand_index = 0; operand_index < 2; ++operand_index) {
            const HloInstruction* candidate = source->operand(operand_index);
            if (candidate->opcode() == HloOpcode::kConvert &&
                candidate->operand(0)->shape().element_type() == PRED) {
              converted_predicate = candidate;
              integer_operand = source->operand(1 - operand_index);
              break;
            }
          }
          if (converted_predicate != nullptr &&
              integer_operand->shape().element_type() ==
                  source->shape().element_type()) {
            // x + zext(pred) != 0 is x != (pred ? -1 : 0). Exposing that
            // identity avoids materializing the zext, add, and per-lane
            // negation that LLVM otherwise leaves in this common predicate
            // conversion pattern.
            TF_ASSIGN_OR_RETURN(
                Value condition,
                EmitVector(builder, argument_pointers,
                           converted_predicate->operand(0), element_offset,
                           predicate, copy_atoms, vector_layout, vector_width,
                           cache));
            TF_ASSIGN_OR_RETURN(
                Value integer,
                EmitVector(builder, argument_pointers, integer_operand,
                           element_offset, predicate, copy_atoms, vector_layout,
                           vector_width, cache));
            auto integer_type = mlir::cast<mlir::VectorType>(integer.getType());
            Value zero = mlir::arith::ConstantOp::create(
                builder, integer_type,
                mlir::DenseElementsAttr::get(
                    integer_type,
                    builder.getIntegerAttr(integer_type.getElementType(), 0)));
            Value negative_one = mlir::arith::ConstantOp::create(
                builder, integer_type,
                mlir::DenseElementsAttr::get(
                    integer_type,
                    builder.getIntegerAttr(integer_type.getElementType(), -1)));
            Value threshold = mlir::arith::SelectOp::create(builder, condition,
                                                            negative_one, zero);
            result = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ne, integer, threshold);
            break;
          }
        }
        TF_ASSIGN_OR_RETURN(Value operand,
                            EmitVector(builder, argument_pointers, source,
                                       element_offset, predicate, copy_atoms,
                                       vector_layout, vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            result, EmitConvert(builder,
                                instruction->operand(0)->shape().element_type(),
                                instruction_type, operand));
        break;
      }
      case HloOpcode::kCopy: {
        TF_ASSIGN_OR_RETURN(
            result,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        break;
      }
      case HloOpcode::kAbs:
      case HloOpcode::kNegate:
      case HloOpcode::kNot: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        if (IsSupportedSignedIntegerType(instruction_type)) {
          if (instruction->opcode() == HloOpcode::kNot) {
            result = mlir::arith::XOrIOp::create(
                builder, operand, SplatInteger(builder, vector_type, -1));
          } else {
            Value zero = SplatInteger(builder, vector_type, 0);
            Value negated = mlir::arith::SubIOp::create(builder, zero, operand);
            if (instruction->opcode() == HloOpcode::kAbs) {
              Value negative = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::slt, operand, zero);
              result = mlir::arith::SelectOp::create(builder, negative, negated,
                                                     operand);
            } else {
              result = negated;
            }
          }
        } else if (instruction_type == PRED) {
          TF_RET_CHECK(instruction->opcode() == HloOpcode::kNot);
          result = mlir::arith::XOrIOp::create(
              builder, operand, SplatInteger(builder, vector_type, 1));
        } else if (instruction->opcode() == HloOpcode::kAbs) {
          if (IsFp8Type(instruction_type)) {
            result = mlir::arith::AndIOp::create(
                builder, operand, SplatInteger(builder, vector_type, 0x7f));
          } else {
            result = mlir::math::AbsFOp::create(builder, operand);
          }
        } else if (IsLowPrecisionFloatingType(instruction_type)) {
          auto compute_type =
              mlir::VectorType::get({vector_width}, builder.getF32Type());
          operand = mlir::arith::ExtFOp::create(builder, compute_type, operand);
          operand = mlir::arith::NegFOp::create(builder, operand);
          result = mlir::arith::TruncFOp::create(builder, vector_type, operand);
        } else {
          result = mlir::arith::NegFOp::create(builder, operand);
        }
        break;
      }
      case HloOpcode::kAcos:
      case HloOpcode::kAcosh:
      case HloOpcode::kAsin:
      case HloOpcode::kAsinh:
      case HloOpcode::kAtanh:
      case HloOpcode::kCbrt:
      case HloOpcode::kCeil:
      case HloOpcode::kCos:
      case HloOpcode::kCosh:
      case HloOpcode::kErf:
      case HloOpcode::kExp:
      case HloOpcode::kExpm1:
      case HloOpcode::kFloor:
      case HloOpcode::kLog:
      case HloOpcode::kLog1p:
      case HloOpcode::kRoundNearestEven:
      case HloOpcode::kRsqrt:
      case HloOpcode::kSin:
      case HloOpcode::kSinh:
      case HloOpcode::kSqrt:
      case HloOpcode::kTan:
      case HloOpcode::kTanh: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        if (IsLowPrecisionFloatingType(instruction_type)) {
          TF_ASSIGN_OR_RETURN(
              operand, EmitConvert(builder, instruction_type, F32, operand));
        }
        TF_ASSIGN_OR_RETURN(
            result,
            EmitScalarizedUnaryMath(builder, instruction->opcode(), operand));
        if (IsLowPrecisionFloatingType(instruction_type)) {
          TF_ASSIGN_OR_RETURN(
              result, EmitConvert(builder, F32, instruction_type, result));
        }
        break;
      }
      case HloOpcode::kReducePrecision: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        if (IsFp8Type(instruction_type)) {
          TF_ASSIGN_OR_RETURN(
              operand, EmitConvert(builder, instruction_type, F32, operand));
          TF_ASSIGN_OR_RETURN(result, EmitScalarizedReducePrecision(
                                          builder, instruction, operand));
          TF_ASSIGN_OR_RETURN(
              result, EmitConvert(builder, F32, instruction_type, result));
        } else {
          TF_ASSIGN_OR_RETURN(result, EmitScalarizedReducePrecision(
                                          builder, instruction, operand));
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
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value rhs,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        if (instruction_type == PRED) {
          switch (instruction->opcode()) {
            case HloOpcode::kAdd:
              result = mlir::arith::XOrIOp::create(builder, lhs, rhs);
              break;
            case HloOpcode::kMultiply:
            case HloOpcode::kMinimum:
              result = mlir::arith::AndIOp::create(builder, lhs, rhs);
              break;
            case HloOpcode::kMaximum:
              result = mlir::arith::OrIOp::create(builder, lhs, rhs);
              break;
            default:
              return absl::InternalError(
                  "Unexpected Fly predicate elementwise opcode.");
          }
          break;
        }
        if (IsSupportedSignedIntegerType(instruction_type)) {
          switch (instruction->opcode()) {
            case HloOpcode::kAdd:
              result = mlir::arith::AddIOp::create(builder, lhs, rhs);
              break;
            case HloOpcode::kSubtract:
              result = mlir::arith::SubIOp::create(builder, lhs, rhs);
              break;
            case HloOpcode::kMultiply:
              result = mlir::arith::MulIOp::create(builder, lhs, rhs);
              break;
            case HloOpcode::kDivide:
              result = mlir::arith::DivSIOp::create(builder, lhs, rhs);
              break;
            case HloOpcode::kMaximum:
              result = mlir::arith::MaxSIOp::create(builder, lhs, rhs);
              break;
            case HloOpcode::kMinimum:
              result = mlir::arith::MinSIOp::create(builder, lhs, rhs);
              break;
            default:
              return absl::InternalError(
                  "Unexpected Fly integer elementwise opcode.");
          }
          break;
        }
        if (instruction_type == BF16) {
          TF_ASSIGN_OR_RETURN(
              result,
              EmitPairwiseBf16Binary(builder, instruction->opcode(), lhs, rhs));
          break;
        }
        if (IsLowPrecisionFloatingType(instruction_type)) {
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
        result =
            IsLowPrecisionFloatingType(instruction_type)
                ? mlir::arith::TruncFOp::create(builder, vector_type, computed)
                      .getResult()
                : computed;
        break;
      }
      case HloOpcode::kAtan2:
      case HloOpcode::kPower:
      case HloOpcode::kRemainder: {
        TF_ASSIGN_OR_RETURN(
            Value lhs,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value rhs,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        if (IsSupportedSignedIntegerType(instruction_type)) {
          TF_RET_CHECK(instruction->opcode() == HloOpcode::kRemainder);
          result = mlir::arith::RemSIOp::create(builder, lhs, rhs);
          break;
        }
        if (IsLowPrecisionFloatingType(instruction_type)) {
          TF_ASSIGN_OR_RETURN(lhs,
                              EmitConvert(builder, instruction_type, F32, lhs));
          TF_ASSIGN_OR_RETURN(rhs,
                              EmitConvert(builder, instruction_type, F32, rhs));
        }
        TF_ASSIGN_OR_RETURN(
            result,
            EmitScalarizedBinaryMath(builder, instruction->opcode(), lhs, rhs));
        if (IsLowPrecisionFloatingType(instruction_type)) {
          TF_ASSIGN_OR_RETURN(
              result, EmitConvert(builder, F32, instruction_type, result));
        }
        break;
      }
      case HloOpcode::kAnd:
      case HloOpcode::kOr:
      case HloOpcode::kXor: {
        const HloInstruction* lhs_instruction = instruction->operand(0);
        const HloInstruction* rhs_instruction = instruction->operand(1);
        const bool lhs_is_not = lhs_instruction->opcode() == HloOpcode::kNot;
        const bool rhs_is_not = rhs_instruction->opcode() == HloOpcode::kNot;
        if (instruction->opcode() == HloOpcode::kXor &&
            (lhs_is_not || rhs_is_not)) {
          // LLVM's AMDGPU backend recognizes scalar xor(xor(a, b), -1) as
          // v_xnor_b32, but misses the equivalent vector pattern. Expose the
          // lanes here and bypass the intermediate vector not. Two inverted
          // operands cancel because (~a) xor (~b) == a xor b.
          if (lhs_is_not) {
            lhs_instruction = lhs_instruction->operand(0);
          }
          if (rhs_is_not) {
            rhs_instruction = rhs_instruction->operand(0);
          }
          TF_ASSIGN_OR_RETURN(
              Value lhs, EmitVector(builder, argument_pointers, lhs_instruction,
                                    element_offset, predicate, copy_atoms,
                                    vector_layout, vector_width, cache));
          TF_ASSIGN_OR_RETURN(
              Value rhs, EmitVector(builder, argument_pointers, rhs_instruction,
                                    element_offset, predicate, copy_atoms,
                                    vector_layout, vector_width, cache));
          llvm::SmallVector<Value> lanes;
          lanes.reserve(vector_width);
          for (int64_t lane = 0; lane < vector_width; ++lane) {
            Value lhs_lane =
                mlir::vector::ExtractOp::create(builder, lhs, lane);
            Value rhs_lane =
                mlir::vector::ExtractOp::create(builder, rhs, lane);
            Value lane_result =
                mlir::arith::XOrIOp::create(builder, lhs_lane, rhs_lane);
            if (lhs_is_not != rhs_is_not) {
              Value all_ones = mlir::arith::ConstantOp::create(
                  builder, lane_result.getType(),
                  builder.getIntegerAttr(lane_result.getType(), -1));
              lane_result =
                  mlir::arith::XOrIOp::create(builder, lane_result, all_ones);
            }
            lanes.push_back(lane_result);
          }
          result =
              mlir::vector::FromElementsOp::create(builder, vector_type, lanes);
          break;
        }
        TF_ASSIGN_OR_RETURN(
            Value lhs, EmitVector(builder, argument_pointers, lhs_instruction,
                                  element_offset, predicate, copy_atoms,
                                  vector_layout, vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value rhs, EmitVector(builder, argument_pointers, rhs_instruction,
                                  element_offset, predicate, copy_atoms,
                                  vector_layout, vector_width, cache));
        switch (instruction->opcode()) {
          case HloOpcode::kAnd:
            result = mlir::arith::AndIOp::create(builder, lhs, rhs);
            break;
          case HloOpcode::kOr:
            result = mlir::arith::OrIOp::create(builder, lhs, rhs);
            break;
          case HloOpcode::kXor:
            result = mlir::arith::XOrIOp::create(builder, lhs, rhs);
            break;
          default:
            return absl::InternalError(
                "Unexpected Fly integer logical opcode.");
        }
        break;
      }
      case HloOpcode::kClamp: {
        TF_ASSIGN_OR_RETURN(
            Value lower,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value upper,
            EmitVector(builder, argument_pointers, instruction->operand(2),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        if (IsSupportedSignedIntegerType(instruction_type)) {
          result = mlir::arith::MinSIOp::create(
              builder, mlir::arith::MaxSIOp::create(builder, lower, operand),
              upper);
        } else if (instruction_type == PRED) {
          result = mlir::arith::AndIOp::create(
              builder, mlir::arith::OrIOp::create(builder, lower, operand),
              upper);
        } else if (instruction_type == BF16) {
          TF_ASSIGN_OR_RETURN(
              operand, EmitPairwiseBf16Binary(builder, HloOpcode::kMaximum,
                                              lower, operand));
          TF_ASSIGN_OR_RETURN(
              result, EmitPairwiseBf16Binary(builder, HloOpcode::kMinimum,
                                             operand, upper));
        } else {
          if (IsLowPrecisionFloatingType(instruction_type)) {
            TF_ASSIGN_OR_RETURN(
                lower, EmitConvert(builder, instruction_type, F32, lower));
            TF_ASSIGN_OR_RETURN(
                operand, EmitConvert(builder, instruction_type, F32, operand));
            TF_ASSIGN_OR_RETURN(
                upper, EmitConvert(builder, instruction_type, F32, upper));
          }
          result = mlir::arith::MinimumFOp::create(
              builder, mlir::arith::MaximumFOp::create(builder, lower, operand),
              upper);
          if (IsLowPrecisionFloatingType(instruction_type)) {
            TF_ASSIGN_OR_RETURN(
                result, EmitConvert(builder, F32, instruction_type, result));
          }
        }
        break;
      }
      case HloOpcode::kCompare: {
        TF_ASSIGN_OR_RETURN(
            Value lhs,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value rhs,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        const PrimitiveType operand_type =
            instruction->operand(0)->shape().element_type();
        if (IsLowPrecisionFloatingType(operand_type)) {
          TF_ASSIGN_OR_RETURN(lhs,
                              EmitConvert(builder, operand_type, F32, lhs));
          TF_ASSIGN_OR_RETURN(rhs,
                              EmitConvert(builder, operand_type, F32, rhs));
        }
        if (operand_type == PRED) {
          TF_ASSIGN_OR_RETURN(
              mlir::arith::CmpIPredicate compare_predicate,
              GetUnsignedComparePredicate(instruction->comparison_direction()));
          result =
              mlir::arith::CmpIOp::create(builder, compare_predicate, lhs, rhs);
        } else if (IsSupportedSignedIntegerType(operand_type)) {
          TF_ASSIGN_OR_RETURN(
              mlir::arith::CmpIPredicate compare_predicate,
              GetSignedComparePredicate(instruction->comparison_direction()));
          result =
              mlir::arith::CmpIOp::create(builder, compare_predicate, lhs, rhs);
        } else {
          TF_ASSIGN_OR_RETURN(
              mlir::arith::CmpFPredicate compare_predicate,
              GetComparePredicate(instruction->comparison_direction()));
          result =
              mlir::arith::CmpFOp::create(builder, compare_predicate, lhs, rhs);
        }
        break;
      }
      case HloOpcode::kSelect: {
        TF_ASSIGN_OR_RETURN(
            Value condition,
            EmitVector(builder, argument_pointers, instruction->operand(0),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value on_true,
            EmitVector(builder, argument_pointers, instruction->operand(1),
                       element_offset, predicate, copy_atoms, vector_layout,
                       vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value on_false,
            EmitVector(builder, argument_pointers, instruction->operand(2),
                       element_offset, predicate, copy_atoms, vector_layout,
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

  absl::Status EmitCooperativeStridedReductionKernel(
      mlir::gpu::GPUFuncOp kernel, const HloFusionInstruction& fusion,
      const StridedReductionDescriptor& descriptor) const {
    TF_RET_CHECK(output_count_ == 1 && vector_width_ == 4 &&
                 vectors_per_thread_ == 1 && threads_ == 256 &&
                 descriptor.type == F32 &&
                 descriptor.reduction_size % 4 == 0 &&
                 descriptor.inner_elements % (64 * vector_width_) == 0 &&
                 descriptor.input_parameter_number >= 0 &&
                 descriptor.input_parameter_number < fusion.operand_count());
    const HloInstruction* root = fusion.fused_expression_root();
    TF_RET_CHECK(root != nullptr && root->opcode() == HloOpcode::kReduce &&
                 root->shape().element_type() == F32 &&
                 root->operand(0)->shape().element_type() == F32);

    mlir::ImplicitLocOpBuilder builder(kernel.getLoc(), kernel);
    builder.setInsertionPointToStart(&kernel.getBody().front());
    mlir::MLIRContext* context = builder.getContext();
    Value thread_id =
        mlir::gpu::ThreadIdOp::create(builder, mlir::gpu::Dimension::x);
    Value block_id =
        mlir::gpu::BlockIdOp::create(builder, mlir::gpu::Dimension::x);
    Value thread_i64 = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(), thread_id);
    Value block_i64 = mlir::arith::IndexCastOp::create(
        builder, builder.getI64Type(), block_id);
    auto i64_constant = [&](int64_t value) -> Value {
      return mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                                value);
    };
    Value wave_id =
        mlir::arith::DivUIOp::create(builder, thread_i64, i64_constant(64));
    Value lane_id =
        mlir::arith::RemUIOp::create(builder, thread_i64, i64_constant(64));

    constexpr int64_t kOutputElementsPerBlock = 64 * 4;
    Value output_block_offset = mlir::arith::MulIOp::create(
        builder, block_i64, i64_constant(kOutputElementsPerBlock));
    Value outer = mlir::arith::DivUIOp::create(
        builder, output_block_offset, i64_constant(descriptor.inner_elements));
    Value inner_block_offset = mlir::arith::RemUIOp::create(
        builder, output_block_offset, i64_constant(descriptor.inner_elements));
    Value lane_offset = mlir::arith::MulIOp::create(
        builder, lane_id, i64_constant(vector_width_));
    Value output_offset =
        mlir::arith::AddIOp::create(builder, output_block_offset, lane_offset);
    Value input_outer_offset = mlir::arith::MulIOp::create(
        builder, outer,
        i64_constant(descriptor.reduction_size * descriptor.inner_elements));
    Value input_inner_offset =
        mlir::arith::AddIOp::create(builder, inner_block_offset, lane_offset);
    Value input_base = mlir::arith::AddIOp::create(builder, input_outer_offset,
                                                   input_inner_offset);

    auto buffer_address = mlir::fly_rocdl::BufferDescAddressAttr::get(context);
    Value descriptor_stride =
        mlir::arith::ConstantIntOp::create(builder, builder.getI16Type(), 0);
    Value descriptor_flags = mlir::arith::ConstantIntOp::create(
        builder, builder.getI32Type(), 0x27000);
    auto make_buffer_pointer = [&](Value pointer,
                                   int64_t byte_extent) -> Value {
      auto buffer_pointer_type =
          mlir::fly::PointerType::get(builder.getF32Type(), buffer_address);
      Value descriptor_extent = mlir::arith::ConstantIntOp::create(
          builder, builder.getI64Type(), byte_extent);
      return mlir::fly::MakePtrOp::create(
          builder, buffer_pointer_type,
          mlir::ValueRange{pointer, descriptor_stride, descriptor_extent,
                           descriptor_flags},
          /*dictAttrs=*/nullptr);
    };
    Value input_pointer = make_buffer_pointer(
        kernel.getArgument(descriptor.input_parameter_number),
        ShapeUtil::ByteSizeOfElements(root->operand(0)->shape()));
    Value output_pointer =
        make_buffer_pointer(kernel.getArgument(fusion.operand_count()),
                            ShapeUtil::ByteSizeOfElements(root->shape()));

    auto shape_attr =
        mlir::fly::IntTupleAttr::getLeafStatic(context, vector_width_);
    auto stride_attr = mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
    auto shape_type = mlir::fly::IntTupleType::get(shape_attr);
    auto stride_type = mlir::fly::IntTupleType::get(stride_attr);
    Value shape = mlir::fly::MakeIntTupleOp::create(builder, shape_type,
                                                    mlir::ValueRange{});
    Value stride = mlir::fly::MakeIntTupleOp::create(builder, stride_type,
                                                     mlir::ValueRange{});
    auto layout_type = mlir::fly::LayoutType::get(shape_attr, stride_attr);
    Value layout =
        mlir::fly::MakeLayoutOp::create(builder, layout_type, shape, stride);
    auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
        context, /*width=*/32, /*divisibility=*/vector_width_);
    auto offset_type = mlir::fly::IntTupleType::get(offset_attr);
    auto copy_atom_type = mlir::fly::CopyAtomType::get(
        mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context, /*bits=*/128,
                                                        cache_modifier_),
        /*elementBits=*/32);
    Value copy_atom = mlir::fly::MakeCopyAtomOp::create(builder, copy_atom_type,
                                                        /*elementBits=*/32);
    auto vector_type =
        mlir::VectorType::get({vector_width_}, builder.getF32Type());
    auto make_view = [&](Value pointer, Value element_offset) -> Value {
      Value offset_i32 = mlir::arith::TruncIOp::create(
          builder, builder.getI32Type(), element_offset);
      Value offset_tuple = mlir::fly::MakeIntTupleOp::create(
          builder, offset_type, mlir::ValueRange{offset_i32});
      Value advanced = mlir::fly::AddOffsetOp::create(
          builder, pointer.getType(), pointer, offset_tuple);
      auto memref_type = mlir::fly::MemRefType::get(
          builder.getF32Type(),
          mlir::cast<mlir::fly::PointerType>(pointer.getType())
              .getAddressSpace(),
          layout_type.getAttr());
      return mlir::fly::MakeViewOp::create(builder, memref_type, advanced,
                                           layout);
    };

    Value accumulator = mlir::arith::ConstantOp::create(
        builder, vector_type, builder.getZeroAttr(vector_type));
    Value true_predicate =
        mlir::arith::ConstantIntOp::create(builder, builder.getI1Type(), 1);
    for (int64_t iteration = 0; iteration < descriptor.reduction_size / 4;
         ++iteration) {
      Value reduction_part = wave_id;
      if (iteration != 0) {
        reduction_part = mlir::arith::AddIOp::create(
            builder, wave_id, i64_constant(iteration * 4));
      }
      Value input_offset = mlir::arith::AddIOp::create(
          builder, input_base,
          mlir::arith::MulIOp::create(builder, reduction_part,
                                      i64_constant(descriptor.inner_elements)));
      Value input_view = make_view(input_pointer, input_offset);
      Value poison = mlir::ub::PoisonOp::create(builder, vector_type);
      Value loaded = mlir::fly::CopyAtomCallSSA::create(
                         builder, mlir::TypeRange{vector_type}, copy_atom,
                         input_view, poison, true_predicate)
                         .getResult(0);
      accumulator = mlir::arith::AddFOp::create(builder, accumulator, loaded);
    }

    auto partials_type = mlir::RankedTensorType::get(
        {4 * kOutputElementsPerBlock}, builder.getF32Type());
    Value partials = AllocateSharedOp::create(builder, partials_type);
    Value partial_base = mlir::arith::AddIOp::create(
        builder,
        mlir::arith::MulIOp::create(builder, wave_id,
                                    i64_constant(kOutputElementsPerBlock)),
        lane_offset);
    for (int64_t lane = 0; lane < vector_width_; ++lane) {
      Value scalar =
          mlir::vector::ExtractOp::create(builder, accumulator, lane);
      partials = mlir::tensor::InsertOp::create(
          builder, scalar, partials,
          mlir::ValueRange{mlir::arith::AddIOp::create(builder, partial_base,
                                                       i64_constant(lane))});
    }
    partials =
        SyncThreadsOp::create(builder, mlir::TypeRange{partials_type}, partials)
            .getResult(0);

    Value is_wave_zero = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::eq, wave_id, i64_constant(0));
    mlir::scf::IfOp store =
        mlir::scf::IfOp::create(builder, mlir::TypeRange{}, is_wave_zero,
                                /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(store.thenBlock());
      llvm::SmallVector<Value> results;
      results.reserve(vector_width_);
      for (int64_t lane = 0; lane < vector_width_; ++lane) {
        Value local_index = mlir::arith::AddIOp::create(builder, lane_offset,
                                                        i64_constant(lane));
        Value sum = mlir::tensor::ExtractOp::create(
            builder, partials, mlir::ValueRange{local_index});
        for (int64_t source_wave = 1; source_wave < 4; ++source_wave) {
          Value source_index = mlir::arith::AddIOp::create(
              builder, local_index,
              i64_constant(source_wave * kOutputElementsPerBlock));
          Value next = mlir::tensor::ExtractOp::create(
              builder, partials, mlir::ValueRange{source_index});
          sum = mlir::arith::AddFOp::create(builder, sum, next);
        }
        results.push_back(sum);
      }
      Value result =
          mlir::vector::FromElementsOp::create(builder, vector_type, results);
      Value output_view = make_view(output_pointer, output_offset);
      mlir::fly::CopyAtomCallSSA::create(builder, mlir::TypeRange{}, copy_atom,
                                         result, output_view, true_predicate);
    }
    builder.setInsertionPointAfter(store);
    mlir::gpu::ReturnOp::create(builder);
    return absl::OkStatus();
  }

  absl::Status EmitKernel(mlir::gpu::GPUFuncOp kernel,
                          const HloFusionInstruction& fusion) const {
    if (cooperative_strided_reduction_.has_value()) {
      return EmitCooperativeStridedReductionKernel(
          kernel, fusion, *cooperative_strided_reduction_);
    }
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
    Value thread_id =
        mlir::gpu::ThreadIdOp::create(builder, mlir::gpu::Dimension::x);
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
    Value shape = mlir::fly::MakeIntTupleOp::create(builder, shape_type,
                                                    mlir::ValueRange{});
    Value stride = mlir::fly::MakeIntTupleOp::create(builder, stride_type,
                                                     mlir::ValueRange{});
    auto layout_type = mlir::fly::LayoutType::get(shape_attr, stride_attr);
    Value vector_layout =
        mlir::fly::MakeLayoutOp::create(builder, layout_type, shape, stride);
    mlir::fly::LayoutType s4_output_layout_type;
    mlir::fly::IntTupleType s4_output_offset_type;
    Value s4_output_layout;
    Value s4_output_copy_atom;
    if (element_type_ == S4) {
      TF_RET_CHECK(vector_width_ % 2 == 0);
      const int64_t packed_width = vector_width_ / 2;
      auto packed_shape_attr =
          mlir::fly::IntTupleAttr::getLeafStatic(context, packed_width);
      auto packed_shape_type = mlir::fly::IntTupleType::get(packed_shape_attr);
      Value packed_shape = mlir::fly::MakeIntTupleOp::create(
          builder, packed_shape_type, mlir::ValueRange{});
      Value packed_stride = mlir::fly::MakeIntTupleOp::create(
          builder, stride_type, mlir::ValueRange{});
      s4_output_layout_type =
          mlir::fly::LayoutType::get(packed_shape_attr, stride_attr);
      s4_output_layout = mlir::fly::MakeLayoutOp::create(
          builder, s4_output_layout_type, packed_shape, packed_stride);
      auto packed_offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
          context, /*width=*/32, /*divisibility=*/packed_width);
      s4_output_offset_type = mlir::fly::IntTupleType::get(packed_offset_attr);
      auto copy_atom_type = mlir::fly::CopyAtomType::get(
          mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(
              context, packed_width * 8, cache_modifier_),
          /*elementBits=*/8);
      s4_output_copy_atom = mlir::fly::MakeCopyAtomOp::create(
          builder, copy_atom_type, /*elementBits=*/8);
    }
    absl::flat_hash_set<PrimitiveType> external_types;
    absl::flat_hash_set<PrimitiveType> vector_external_types;
    for (auto [operand_index, operand] : llvm::enumerate(fusion.operands())) {
      const PrimitiveType type = operand->shape().element_type();
      external_types.insert(type);
      // Scalar integral operands are dynamic slice/update indices. They are
      // loaded once per wave through a scalar copy context, so they must not
      // inflate the transaction width used for the vectorized data path.
      if (!ShapeUtil::IsScalar(operand->shape()) &&
          !std::binary_search(scalar_index_parameter_numbers_.begin(),
                              scalar_index_parameter_numbers_.end(),
                              operand_index)) {
        vector_external_types.insert(type);
      }
    }
    for (const HloInstruction* root : roots) {
      const PrimitiveType type = root->shape().element_type();
      external_types.insert(type);
      vector_external_types.insert(type);
    }
    absl::flat_hash_map<PrimitiveType, Value> copy_atoms;
    for (PrimitiveType type : external_types) {
      TF_RET_CHECK(IsSupportedExternalType(type));
      if (type == S4) {
        continue;
      }
      const int64_t copy_width =
          vector_external_types.contains(type) ? vector_width_ : int64_t{1};
      const int64_t copy_bits = copy_width * ElementBits(type);
      if (copy_bits != 8 && copy_bits != 16 && copy_bits != 32 &&
          copy_bits != 64 && copy_bits != 128) {
        continue;
      }
      auto copy_atom_type = mlir::fly::CopyAtomType::get(
          mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context, copy_bits,
                                                          cache_modifier_),
          ElementBits(type));
      copy_atoms[type] = mlir::fly::MakeCopyAtomOp::create(
          builder, copy_atom_type, ElementBits(type));
    }

    auto buffer_address = mlir::fly_rocdl::BufferDescAddressAttr::get(context);
    Value descriptor_stride =
        mlir::arith::ConstantIntOp::create(builder, builder.getI16Type(), 0);
    Value descriptor_flags = mlir::arith::ConstantIntOp::create(
        builder, builder.getI32Type(), 0x27000);
    llvm::SmallVector<const Shape*> argument_shapes;
    argument_shapes.reserve(kernel.getNumArguments());
    for (const HloInstruction* operand : fusion.operands()) {
      argument_shapes.push_back(&operand->shape());
    }
    for (const HloInstruction* root : roots) {
      argument_shapes.push_back(&root->shape());
    }
    TF_RET_CHECK(argument_shapes.size() == kernel.getNumArguments());
    llvm::SmallVector<Value> argument_pointers;
    argument_pointers.reserve(kernel.getNumArguments());
    for (auto [index, pointer] : llvm::enumerate(kernel.getArguments())) {
      const Shape& shape = *argument_shapes[index];
      const PrimitiveType type = shape.element_type();
      auto buffer_pointer_type = mlir::fly::PointerType::get(
          StorageElementType(type, builder), buffer_address);
      Value descriptor_extent = mlir::arith::ConstantIntOp::create(
          builder, builder.getI64Type(), ShapeUtil::ByteSizeOfElements(shape));
      argument_pointers.push_back(mlir::fly::MakePtrOp::create(
          builder, buffer_pointer_type,
          mlir::ValueRange{pointer, descriptor_stride, descriptor_extent,
                           descriptor_flags},
          /*dictAttrs=*/nullptr));
    }

    auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
        context, /*width=*/32, /*divisibility=*/vector_width_);
    auto offset_type = mlir::fly::IntTupleType::get(offset_attr);
    auto scatter_offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
        context, /*width=*/32, /*divisibility=*/1);
    auto scatter_offset_type =
        mlir::fly::IntTupleType::get(scatter_offset_attr);
    std::optional<ScalarCopyContext> scatter_scalar_copies;
    Value scatter_atomic_copy_atom;
    Value scatter_packed_atomic_copy_atom;
    const bool has_packed_bf16_scatter_add =
        overwrite_row_scatter_.has_value() &&
        overwrite_row_scatter_->update_kind == ScatterUpdateKind::kAdd &&
        element_type_ == BF16 && vector_width_ == 2;
    const bool use_paired_bf16_scatter_add =
        has_packed_bf16_scatter_add &&
        overwrite_row_scatter_->contiguous_elements % 2 == 0 &&
        overwrite_row_scatter_->output_dimension_sizes.back() % 2 == 0 &&
        overwrite_row_scatter_->index_components <
            overwrite_row_scatter_->output_dimension_sizes.size();
    if (overwrite_row_scatter_.has_value()) {
      scatter_scalar_copies.emplace(
          CreateScalarCopyContext(builder, copy_atoms));
      if (overwrite_row_scatter_->requires_atomic_operation) {
        const int64_t element_bits = ElementBits(element_type_);
        mlir::fly::AtomicOp atomic_operation;
        switch (overwrite_row_scatter_->update_kind) {
          case ScatterUpdateKind::kOverwrite:
            atomic_operation = mlir::fly::AtomicOp::Exchange;
            break;
          case ScatterUpdateKind::kAdd:
            atomic_operation = mlir::fly::AtomicOp::Add;
            break;
          case ScatterUpdateKind::kMaximum:
            atomic_operation = mlir::fly::AtomicOp::Max;
            break;
          case ScatterUpdateKind::kMinimum:
            atomic_operation = mlir::fly::AtomicOp::Min;
            break;
        }
        auto atomic_op =
            mlir::fly::AtomicOpAttr::get(context, atomic_operation);
        auto atomic_type = mlir::fly::CopyOpUniversalAtomicType::get(
            atomic_op, StorageElementType(element_type_, builder),
            "agent-one-as");
        auto copy_atom_type =
            mlir::fly::CopyAtomType::get(atomic_type, element_bits);
        scatter_atomic_copy_atom = mlir::fly::MakeCopyAtomOp::create(
            builder, copy_atom_type, element_bits);
        if (has_packed_bf16_scatter_add) {
          auto packed_type = mlir::VectorType::get(
              {2}, StorageElementType(element_type_, builder));
          auto packed_atomic_type = mlir::fly::CopyOpUniversalAtomicType::get(
              atomic_op, packed_type, "agent-one-as");
          auto packed_copy_atom_type = mlir::fly::CopyAtomType::get(
              packed_atomic_type, element_bits);
          scatter_packed_atomic_copy_atom =
              mlir::fly::MakeCopyAtomOp::create(
                  builder, packed_copy_atom_type, element_bits);
        }
      }
    }

    auto emit_scatter_indices =
        [&](Value update_offset, Value predicate)
        -> absl::StatusOr<llvm::SmallVector<Value>> {
      TF_RET_CHECK(overwrite_row_scatter_.has_value() &&
                   scatter_scalar_copies.has_value());
      Value update_row = mlir::arith::DivUIOp::create(
          builder, update_offset,
          mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(),
              overwrite_row_scatter_->row_elements));
      Value first_index_offset = mlir::arith::MulIOp::create(
          builder, update_row,
          mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(),
              overwrite_row_scatter_->index_components));
      llvm::SmallVector<Value> indices;
      indices.reserve(overwrite_row_scatter_->index_components);
      for (int64_t component = 0;
           component < overwrite_row_scatter_->index_components;
           ++component) {
        Value index_offset = first_index_offset;
        if (component != 0) {
          index_offset = mlir::arith::AddIOp::create(
              builder, first_index_offset,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(), component));
        }
        absl::flat_hash_map<const HloInstruction*, Value> index_cache;
        TF_ASSIGN_OR_RETURN(
            Value index_vector,
            EmitVector(builder, argument_pointers,
                       overwrite_row_scatter_->scatter->operand(1),
                       index_offset, predicate,
                       scatter_scalar_copies->copy_atoms,
                       scatter_scalar_copies->layout, /*vector_width=*/1,
                       index_cache));
        Value index =
            mlir::vector::ExtractOp::create(builder, index_vector, 0);
        if (overwrite_row_scatter_->scatter->operand(1)
                ->shape()
                .element_type() == S32) {
          index = mlir::arith::ExtSIOp::create(builder, builder.getI64Type(),
                                               index);
        }
        Value zero = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), 0);
        indices.push_back(
            mlir::arith::SelectOp::create(builder, predicate, index, zero));
      }
      return indices;
    };
    auto scatter_indices_are_valid =
        [&](llvm::ArrayRef<Value> indices) -> Value {
      Value valid = mlir::arith::ConstantIntOp::create(
          builder, builder.getI1Type(), 1);
      Value zero = mlir::arith::ConstantIntOp::create(
          builder, builder.getI64Type(), 0);
      for (int64_t component = 0;
           component < overwrite_row_scatter_->index_components;
           ++component) {
        Value lower = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::sge, indices[component],
            zero);
        Value upper = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::slt, indices[component],
            mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(),
                overwrite_row_scatter_
                    ->indexed_dimension_start_limits[component]));
        valid = mlir::arith::AndIOp::create(
            builder, valid,
            mlir::arith::AndIOp::create(builder, lower, upper));
      }
      return valid;
    };
    auto scatter_output_offset = [&](Value update_offset,
                                     llvm::ArrayRef<Value> indices) -> Value {
      Value column = mlir::arith::RemUIOp::create(
          builder, update_offset,
          mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(),
              overwrite_row_scatter_->row_elements));
      if (!overwrite_row_scatter_->has_contiguous_window) {
        int64_t update_stride = overwrite_row_scatter_->row_elements;
        int64_t output_stride = ShapeUtil::ElementsIn(
            overwrite_row_scatter_->scatter->shape());
        Value output_offset = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), 0);
        for (int64_t dimension = 0;
             dimension <
             overwrite_row_scatter_->output_dimension_sizes.size();
             ++dimension) {
          update_stride /= overwrite_row_scatter_
                               ->update_window_dimension_sizes[dimension];
          output_stride /=
              overwrite_row_scatter_->output_dimension_sizes[dimension];
          Value coordinate = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 0);
          const int64_t update_size = overwrite_row_scatter_
                                          ->update_window_dimension_sizes
                                              [dimension];
          if (update_size != 1) {
            coordinate = column;
            if (update_stride != 1) {
              coordinate = mlir::arith::DivUIOp::create(
                  builder, coordinate,
                  mlir::arith::ConstantIntOp::create(
                      builder, builder.getI64Type(), update_stride));
            }
            coordinate = mlir::arith::RemUIOp::create(
                builder, coordinate,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), update_size));
          }
          if (dimension < overwrite_row_scatter_->index_components) {
            coordinate = mlir::arith::AddIOp::create(
                builder, indices[dimension], coordinate);
          }
          if (output_stride != 1) {
            coordinate = mlir::arith::MulIOp::create(
                builder, coordinate,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(), output_stride));
          }
          output_offset = mlir::arith::AddIOp::create(builder, output_offset,
                                                       coordinate);
        }
        return output_offset;
      }
      Value output_row = indices.front();
      for (int64_t component = 1;
           component < overwrite_row_scatter_->index_components;
           ++component) {
        output_row = mlir::arith::AddIOp::create(
            builder,
            mlir::arith::MulIOp::create(
                builder, output_row,
                mlir::arith::ConstantIntOp::create(
                    builder, builder.getI64Type(),
                    overwrite_row_scatter_
                        ->indexed_dimension_sizes[component])),
            indices[component]);
      }
      return mlir::arith::AddIOp::create(
          builder,
          mlir::arith::MulIOp::create(
              builder, output_row,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  overwrite_row_scatter_->output_row_elements)),
          column);
    };
    auto emit_scalar_scatter_store =
        [&](Value scalar, Value update_offset,
            Value predicate) -> absl::Status {
      TF_RET_CHECK(overwrite_row_scatter_.has_value() &&
                   scatter_scalar_copies.has_value());
      TF_ASSIGN_OR_RETURN(llvm::SmallVector<Value> indices,
                          emit_scatter_indices(update_offset, predicate));
      Value store_predicate = mlir::arith::AndIOp::create(
          builder, predicate, scatter_indices_are_valid(indices));
      Value output_offset = scatter_output_offset(update_offset, indices);
      if (has_packed_bf16_scatter_add) {
        Value one = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), 1);
        Value pair_offset = mlir::arith::AndIOp::create(
            builder, output_offset,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               -2));
        Value high_lane = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ne,
            mlir::arith::AndIOp::create(builder, output_offset, one),
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               0));
        Value pair_offset_i32 = mlir::arith::TruncIOp::create(
            builder, builder.getI32Type(), pair_offset);
        Value pair_offset_tuple = mlir::fly::MakeIntTupleOp::create(
            builder, scatter_offset_type,
            mlir::ValueRange{pair_offset_i32});
        Value output_pointer =
            kernel.getArgument(fusion.operand_count());
        auto output_pointer_type = mlir::cast<mlir::fly::PointerType>(
            output_pointer.getType());
        auto output_memref_type = mlir::fly::MemRefType::get(
            output_pointer_type.getElemTy(),
            output_pointer_type.getAddressSpace(), layout_type.getAttr());
        Value advanced_output = mlir::fly::AddOffsetOp::create(
            builder, output_pointer_type, output_pointer, pair_offset_tuple);
        Value output_view = mlir::fly::MakeViewOp::create(
            builder, output_memref_type, advanced_output, vector_layout);
        Value zero = mlir::arith::ConstantFloatOp::create(
            builder, builder.getBF16Type(),
            llvm::APFloat::getZero(llvm::APFloat::BFloat()));
        auto pair_type = mlir::VectorType::get({2}, builder.getBF16Type());
        Value low_pair = mlir::vector::FromElementsOp::create(
            builder, pair_type, mlir::ValueRange{scalar, zero});
        Value high_pair = mlir::vector::FromElementsOp::create(
            builder, pair_type, mlir::ValueRange{zero, scalar});
        Value packed_value = mlir::arith::SelectOp::create(
            builder, high_lane, high_pair, low_pair);
        mlir::fly::CopyAtomCallSSA::create(
            builder, mlir::TypeRange{}, scatter_packed_atomic_copy_atom,
            packed_value, output_view, store_predicate);
        return absl::OkStatus();
      }
      Value output_offset_i32 = mlir::arith::TruncIOp::create(
          builder, builder.getI32Type(), output_offset);
      Value output_offset_tuple = mlir::fly::MakeIntTupleOp::create(
          builder, scatter_offset_type,
          mlir::ValueRange{output_offset_i32});
      Value output_pointer =
          overwrite_row_scatter_->requires_atomic_operation
              ? kernel.getArgument(fusion.operand_count())
              : argument_pointers[fusion.operand_count()];
      auto output_pointer_type =
          mlir::cast<mlir::fly::PointerType>(output_pointer.getType());
      auto scalar_layout_type = mlir::cast<mlir::fly::LayoutType>(
          scatter_scalar_copies->layout.getType());
      auto output_memref_type = mlir::fly::MemRefType::get(
          output_pointer_type.getElemTy(), output_pointer_type.getAddressSpace(),
          scalar_layout_type.getAttr());
      Value advanced_output = mlir::fly::AddOffsetOp::create(
          builder, output_pointer_type, output_pointer, output_offset_tuple);
      Value output_view = mlir::fly::MakeViewOp::create(
          builder, output_memref_type, advanced_output,
          scatter_scalar_copies->layout);
      auto scalar_vector_type = mlir::VectorType::get(
          {1}, StorageElementType(element_type_, builder));
      Value scalar_vector = mlir::vector::FromElementsOp::create(
          builder, scalar_vector_type, mlir::ValueRange{scalar});
      Value store_value =
          overwrite_row_scatter_->requires_atomic_operation ? scalar
                                                            : scalar_vector;
      mlir::fly::CopyAtomCallSSA::create(
          builder, mlir::TypeRange{},
          overwrite_row_scatter_->requires_atomic_operation
              ? scatter_atomic_copy_atom
              : scatter_scalar_copies->copy_atoms.at(element_type_),
          store_value,
          output_view, store_predicate);
      return absl::OkStatus();
    };

    llvm::SmallVector<const HloInstruction*> striped_invariant_broadcasts;
    if (uniform_output_elements_ && vectors_per_thread_ > 1 &&
        !overwrite_row_scatter_.has_value()) {
      for (const HloInstruction* instruction :
           fusion.fused_instructions_computation()->instructions()) {
        if (IsStripedLoopInvariantBroadcast(
                instruction, threads_ * vector_width_, vectors_per_thread_)) {
          striped_invariant_broadcasts.push_back(instruction);
        }
      }
    }
    absl::flat_hash_map<const HloInstruction*, Value> striped_invariant_cache;

    for (int64_t vector = 0; full_elements != 0 && vector < vectors_per_thread_;
         ++vector) {
      Value vector_offset = base;
      if (vector != 0) {
        Value vector_index = mlir::arith::ConstantIntOp::create(
            builder, builder.getI64Type(), vector);
        Value relative =
            mlir::arith::MulIOp::create(builder, vector_index, thread_stride);
        vector_offset = mlir::arith::AddIOp::create(builder, base, relative);
      }
      Value in_bounds =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::ult,
                                      vector_offset, element_count);
      llvm::SmallVector<Value> results;
      llvm::SmallVector<Value> result_predicates;
      results.reserve(output_count_);
      result_predicates.reserve(output_count_);
      absl::flat_hash_map<const HloInstruction*, Value> shared_cache =
          striped_invariant_cache;
      for (auto [output_index, root] : llvm::enumerate(roots)) {
        Value output_in_bounds = in_bounds;
        if (!uniform_output_elements_) {
          const int64_t output_full_elements =
              output_elements_[output_index] -
              output_elements_[output_index] % vector_width_;
          output_in_bounds = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, vector_offset,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(), output_full_elements));
        }
        const HloInstruction* expression =
            overwrite_row_scatter_.has_value() ? root->operand(2) : root;
        absl::flat_hash_map<const HloInstruction*, Value> private_cache;
        absl::flat_hash_map<const HloInstruction*, Value>& cache =
            uniform_output_elements_ ? shared_cache : private_cache;
        TF_ASSIGN_OR_RETURN(
            Value result,
            EmitVector(builder, argument_pointers, expression, vector_offset,
                       output_in_bounds, copy_atoms, vector_layout,
                       vector_width_, cache));
        results.push_back(result);
        result_predicates.push_back(output_in_bounds);
      }
      if (vector == 0) {
        for (const HloInstruction* broadcast : striped_invariant_broadcasts) {
          if (auto it = shared_cache.find(broadcast);
              it != shared_cache.end()) {
            striped_invariant_cache.insert(*it);
          }
        }
      }

      Value vector_offset_i32 = mlir::arith::TruncIOp::create(
          builder, builder.getI32Type(), vector_offset);
      Value offset_tuple = mlir::fly::MakeIntTupleOp::create(
          builder, offset_type, mlir::ValueRange{vector_offset_i32});
      for (auto [output_index, result] : llvm::enumerate(results)) {
        const PrimitiveType output_type =
            roots[output_index]->shape().element_type();
        Value output_in_bounds = result_predicates[output_index];
        if (overwrite_row_scatter_.has_value()) {
          TF_RET_CHECK(output_index == 0 && output_type == element_type_);
          if (overwrite_row_scatter_->requires_atomic_operation) {
            if (use_paired_bf16_scatter_add) {
              TF_ASSIGN_OR_RETURN(
                  llvm::SmallVector<Value> indices,
                  emit_scatter_indices(vector_offset, in_bounds));
              Value store_predicate = mlir::arith::AndIOp::create(
                  builder, in_bounds, scatter_indices_are_valid(indices));
              Value output_offset =
                  scatter_output_offset(vector_offset, indices);
              Value output_offset_i32 = mlir::arith::TruncIOp::create(
                  builder, builder.getI32Type(), output_offset);
              Value output_offset_tuple = mlir::fly::MakeIntTupleOp::create(
                  builder, scatter_offset_type,
                  mlir::ValueRange{output_offset_i32});
              Value output_pointer =
                  kernel.getArgument(fusion.operand_count());
              auto output_pointer_type =
                  mlir::cast<mlir::fly::PointerType>(
                      output_pointer.getType());
              auto output_memref_type = mlir::fly::MemRefType::get(
                  output_pointer_type.getElemTy(),
                  output_pointer_type.getAddressSpace(),
                  layout_type.getAttr());
              Value advanced_output = mlir::fly::AddOffsetOp::create(
                  builder, output_pointer_type, output_pointer,
                  output_offset_tuple);
              Value output_view = mlir::fly::MakeViewOp::create(
                  builder, output_memref_type, advanced_output,
                  vector_layout);
              mlir::fly::CopyAtomCallSSA::create(
                  builder, mlir::TypeRange{},
                  scatter_packed_atomic_copy_atom, result, output_view,
                  store_predicate);
              continue;
            }
            for (int64_t lane = 0; lane < vector_width_; ++lane) {
              Value lane_offset = vector_offset;
              if (lane != 0) {
                lane_offset = mlir::arith::AddIOp::create(
                    builder, vector_offset,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(), lane));
              }
              Value lane_value =
                  mlir::vector::ExtractOp::create(builder, result, lane);
              RETURN_IF_ERROR(emit_scalar_scatter_store(
                  lane_value, lane_offset, in_bounds));
            }
            continue;
          }
          Value column = mlir::arith::RemUIOp::create(
              builder, vector_offset,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  overwrite_row_scatter_->contiguous_elements));
          Value vector_end = mlir::arith::AddIOp::create(
              builder, column,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(), vector_width_));
          Value fits_in_row = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule, vector_end,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(),
                  overwrite_row_scatter_->contiguous_elements));
          TF_ASSIGN_OR_RETURN(llvm::SmallVector<Value> indices,
                              emit_scatter_indices(vector_offset, in_bounds));
          Value vector_store_predicate = mlir::arith::AndIOp::create(
              builder, in_bounds,
              mlir::arith::AndIOp::create(
                  builder, fits_in_row,
                  scatter_indices_are_valid(indices)));
          Value output_offset = scatter_output_offset(vector_offset, indices);
          Value output_offset_i32 = mlir::arith::TruncIOp::create(
              builder, builder.getI32Type(), output_offset);
          Value output_offset_tuple = mlir::fly::MakeIntTupleOp::create(
              builder, scatter_offset_type,
              mlir::ValueRange{output_offset_i32});
          Value output_pointer =
              argument_pointers[fusion.operand_count()];
          auto output_pointer_type =
              mlir::cast<mlir::fly::PointerType>(output_pointer.getType());
          auto output_memref_type = mlir::fly::MemRefType::get(
              output_pointer_type.getElemTy(),
              output_pointer_type.getAddressSpace(), layout_type.getAttr());
          Value advanced_output = mlir::fly::AddOffsetOp::create(
              builder, output_pointer_type, output_pointer,
              output_offset_tuple);
          Value output_view = mlir::fly::MakeViewOp::create(
              builder, output_memref_type, advanced_output, vector_layout);
          mlir::fly::CopyAtomCallSSA::create(
              builder, mlir::TypeRange{}, copy_atoms.at(output_type), result,
              output_view, vector_store_predicate);

          Value crosses_row = mlir::arith::XOrIOp::create(
              builder, fits_in_row,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI1Type(), 1));
          Value needs_scalar_store = mlir::arith::AndIOp::create(
              builder, in_bounds, crosses_row);
          mlir::scf::IfOp boundary = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{}, needs_scalar_store,
              /*withElseRegion=*/false);
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(boundary.thenBlock());
            Value true_predicate = mlir::arith::ConstantIntOp::create(
                builder, builder.getI1Type(), 1);
            for (int64_t lane = 0; lane < vector_width_; ++lane) {
              Value lane_offset = vector_offset;
              if (lane != 0) {
                lane_offset = mlir::arith::AddIOp::create(
                    builder, vector_offset,
                    mlir::arith::ConstantIntOp::create(
                        builder, builder.getI64Type(), lane));
              }
              Value lane_value =
                  mlir::vector::ExtractOp::create(builder, result, lane);
              RETURN_IF_ERROR(emit_scalar_scatter_store(
                  lane_value, lane_offset, true_predicate));
            }
          }
          builder.setInsertionPointAfter(boundary);
          continue;
        }
        if (output_type == S4) {
          result = PackS4Values(builder, result);
          Value one = mlir::arith::ConstantIntOp::create(
              builder, builder.getI64Type(), 1);
          Value byte_offset =
              mlir::arith::ShRUIOp::create(builder, vector_offset, one);
          Value byte_offset_i32 = mlir::arith::TruncIOp::create(
              builder, builder.getI32Type(), byte_offset);
          Value packed_offset_tuple = mlir::fly::MakeIntTupleOp::create(
              builder, s4_output_offset_type,
              mlir::ValueRange{byte_offset_i32});
          Value output_pointer =
              argument_pointers[fusion.operand_count() + output_index];
          auto output_pointer_type =
              mlir::cast<mlir::fly::PointerType>(output_pointer.getType());
          auto output_memref_type =
              mlir::fly::MemRefType::get(output_pointer_type.getElemTy(),
                                         output_pointer_type.getAddressSpace(),
                                         s4_output_layout_type.getAttr());
          Value advanced_output = mlir::fly::AddOffsetOp::create(
              builder, output_pointer_type, output_pointer,
              packed_offset_tuple);
          Value output_view = mlir::fly::MakeViewOp::create(
              builder, output_memref_type, advanced_output, s4_output_layout);
          mlir::fly::CopyAtomCallSSA::create(builder, mlir::TypeRange{},
                                             s4_output_copy_atom, result,
                                             output_view, output_in_bounds);
          continue;
        }
        if (output_type == PRED) {
          auto predicate_type = mlir::cast<mlir::VectorType>(result.getType());
          auto storage_type = mlir::VectorType::get(predicate_type.getShape(),
                                                    builder.getI8Type());
          result = mlir::arith::ExtUIOp::create(builder, storage_type, result);
        }
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
        mlir::fly::CopyAtomCallSSA::create(builder, mlir::TypeRange{},
                                           copy_atoms.at(output_type), result,
                                           output_view, output_in_bounds);
      }
    }
    if (!uniform_output_elements_) {
      // Tuple roots can cover different prefix lengths of the same flat launch
      // domain. Vectorize each root up to its own full-vector boundary above,
      // then let the first few lanes of block zero finish that root's scalar
      // tail. Keeping the tails independent avoids evaluating or storing a
      // shorter output with the larger root's predicate.
      auto scalar_shape_attr =
          mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
      auto scalar_shape_type =
          mlir::fly::IntTupleType::get(scalar_shape_attr);
      Value scalar_shape = mlir::fly::MakeIntTupleOp::create(
          builder, scalar_shape_type, mlir::ValueRange{});
      Value scalar_stride = mlir::fly::MakeIntTupleOp::create(
          builder, scalar_shape_type, mlir::ValueRange{});
      auto scalar_layout_type =
          mlir::fly::LayoutType::get(scalar_shape_attr, scalar_shape_attr);
      Value scalar_layout = mlir::fly::MakeLayoutOp::create(
          builder, scalar_layout_type, scalar_shape, scalar_stride);
      absl::flat_hash_map<PrimitiveType, Value> scalar_copy_atoms;
      for (PrimitiveType type : external_types) {
        if (type == S4) {
          continue;
        }
        const int64_t element_bits = ElementBits(type);
        auto scalar_copy_atom_type = mlir::fly::CopyAtomType::get(
            mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(
                context, element_bits, /*cacheModifier=*/0),
            element_bits);
        scalar_copy_atoms[type] = mlir::fly::MakeCopyAtomOp::create(
            builder, scalar_copy_atom_type, element_bits);
      }
      auto scalar_offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
          context, /*width=*/32, /*divisibility=*/1);
      auto scalar_offset_type =
          mlir::fly::IntTupleType::get(scalar_offset_attr);
      Value block_zero = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, block_i64,
          mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(), 0));
      for (auto [output_index, root] : llvm::enumerate(roots)) {
        const int64_t output_tail_elements =
            output_elements_[output_index] % vector_width_;
        if (output_tail_elements == 0) {
          continue;
        }
        const int64_t output_full_elements =
            output_elements_[output_index] - output_tail_elements;
        Value tail_lane = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, thread_i64,
            mlir::arith::ConstantIntOp::create(
                builder, builder.getI64Type(), output_tail_elements));
        Value tail_valid =
            mlir::arith::AndIOp::create(builder, block_zero, tail_lane);
        mlir::scf::IfOp tail =
            mlir::scf::IfOp::create(builder, mlir::TypeRange{}, tail_valid,
                                    /*withElseRegion=*/false);
        {
          mlir::OpBuilder::InsertionGuard tail_guard(builder);
          builder.setInsertionPointToStart(tail.thenBlock());
          Value tail_offset = mlir::arith::AddIOp::create(
              builder,
              mlir::arith::ConstantIntOp::create(
                  builder, builder.getI64Type(), output_full_elements),
              thread_i64);
          Value true_predicate = mlir::arith::ConstantIntOp::create(
              builder, builder.getI1Type(), 1);
          absl::flat_hash_map<const HloInstruction*, Value> cache;
          TF_ASSIGN_OR_RETURN(
              Value result,
              EmitVector(builder, argument_pointers, root, tail_offset,
                         true_predicate, scalar_copy_atoms, scalar_layout,
                         /*vector_width=*/1, cache));
          const PrimitiveType output_type = root->shape().element_type();
          if (output_type == PRED) {
            auto predicate_type =
                mlir::cast<mlir::VectorType>(result.getType());
            auto storage_type = mlir::VectorType::get(predicate_type.getShape(),
                                                      builder.getI8Type());
            result =
                mlir::arith::ExtUIOp::create(builder, storage_type, result);
          }
          Value tail_offset_i32 = mlir::arith::TruncIOp::create(
              builder, builder.getI32Type(), tail_offset);
          Value tail_offset_tuple = mlir::fly::MakeIntTupleOp::create(
              builder, scalar_offset_type,
              mlir::ValueRange{tail_offset_i32});
          Value output_pointer =
              argument_pointers[fusion.operand_count() + output_index];
          auto output_pointer_type = mlir::cast<mlir::fly::PointerType>(
              output_pointer.getType());
          auto output_memref_type = mlir::fly::MemRefType::get(
              output_pointer_type.getElemTy(),
              output_pointer_type.getAddressSpace(),
              scalar_layout_type.getAttr());
          Value advanced_output = mlir::fly::AddOffsetOp::create(
              builder, output_pointer_type, output_pointer,
              tail_offset_tuple);
          Value output_view = mlir::fly::MakeViewOp::create(
              builder, output_memref_type, advanced_output, scalar_layout);
          mlir::fly::CopyAtomCallSSA::create(
              builder, mlir::TypeRange{}, scalar_copy_atoms.at(output_type),
              result, output_view, true_predicate);
        }
        builder.setInsertionPointAfter(tail);
      }
    }
    if (uniform_output_elements_ && tail_elements != 0 &&
        element_type_ == S4) {
      auto scalar_shape_attr =
          mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
      auto scalar_shape_type = mlir::fly::IntTupleType::get(scalar_shape_attr);
      Value scalar_shape = mlir::fly::MakeIntTupleOp::create(
          builder, scalar_shape_type, mlir::ValueRange{});
      Value scalar_stride = mlir::fly::MakeIntTupleOp::create(
          builder, scalar_shape_type, mlir::ValueRange{});
      auto scalar_layout_type =
          mlir::fly::LayoutType::get(scalar_shape_attr, scalar_shape_attr);
      Value scalar_layout = mlir::fly::MakeLayoutOp::create(
          builder, scalar_layout_type, scalar_shape, scalar_stride);
      absl::flat_hash_map<PrimitiveType, Value> scalar_copy_atoms;
      for (PrimitiveType type : external_types) {
        if (type == S4) {
          continue;
        }
        const int64_t element_bits = ElementBits(type);
        auto scalar_copy_atom_type = mlir::fly::CopyAtomType::get(
            mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(
                context, element_bits, /*cacheModifier=*/0),
            element_bits);
        scalar_copy_atoms[type] = mlir::fly::MakeCopyAtomOp::create(
            builder, scalar_copy_atom_type, element_bits);
      }
      auto output_copy_atom_type = mlir::fly::CopyAtomType::get(
          mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context, /*bits=*/8,
                                                          /*cacheModifier=*/0),
          /*elementBits=*/8);
      Value output_copy_atom = mlir::fly::MakeCopyAtomOp::create(
          builder, output_copy_atom_type, /*elementBits=*/8);
      Value block_zero = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, block_i64,
          mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(), 0));
      const int64_t tail_bytes = (tail_elements + 1) / 2;
      Value tail_lane = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, thread_i64,
          mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                             tail_bytes));
      Value tail_valid =
          mlir::arith::AndIOp::create(builder, block_zero, tail_lane);
      mlir::scf::IfOp tail =
          mlir::scf::IfOp::create(builder, mlir::TypeRange{}, tail_valid,
                                  /*withElseRegion=*/false);
      {
        mlir::OpBuilder::InsertionGuard tail_guard(builder);
        builder.setInsertionPointToStart(tail.thenBlock());
        Value two = mlir::arith::ConstantIntOp::create(builder,
                                                       builder.getI64Type(), 2);
        Value logical_pair_offset = mlir::arith::AddIOp::create(
            builder,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               full_elements),
            mlir::arith::MulIOp::create(builder, thread_i64, two));
        Value second_offset =
            mlir::arith::AddIOp::create(builder, logical_pair_offset,
                                        mlir::arith::ConstantIntOp::create(
                                            builder, builder.getI64Type(), 1));
        Value second_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, second_offset,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               elements_));
        Value true_predicate =
            mlir::arith::ConstantIntOp::create(builder, builder.getI1Type(), 1);
        llvm::SmallVector<Value> results;
        results.reserve(output_count_);
        for (const HloInstruction* root : roots) {
          absl::flat_hash_map<const HloInstruction*, Value> low_cache;
          TF_ASSIGN_OR_RETURN(
              Value low,
              EmitVector(builder, argument_pointers, root, logical_pair_offset,
                         true_predicate, scalar_copy_atoms, scalar_layout,
                         /*vector_width=*/1, low_cache));
          absl::flat_hash_map<const HloInstruction*, Value> high_cache;
          TF_ASSIGN_OR_RETURN(
              Value high,
              EmitVector(builder, argument_pointers, root, second_offset,
                         second_valid, scalar_copy_atoms, scalar_layout,
                         /*vector_width=*/1, high_cache));
          auto high_type = mlir::cast<mlir::VectorType>(high.getType());
          TF_RET_CHECK(high_type.getElementType().isInteger(8));
          Value zero = mlir::arith::ConstantOp::create(
              builder, high_type,
              mlir::DenseElementsAttr::get(
                  high_type, builder.getIntegerAttr(builder.getI8Type(), 0)));
          high =
              mlir::arith::SelectOp::create(builder, second_valid, high, zero);
          llvm::SmallVector<Value> pair{low, high};
          results.push_back(
              PackS4Values(builder, AssemblePairs(builder, std::move(pair))));
        }

        Value byte_offset =
            mlir::arith::ShRUIOp::create(builder, logical_pair_offset,
                                         mlir::arith::ConstantIntOp::create(
                                             builder, builder.getI64Type(), 1));
        auto scalar_offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
            context, /*width=*/32, /*divisibility=*/1);
        auto scalar_offset_type =
            mlir::fly::IntTupleType::get(scalar_offset_attr);
        Value byte_offset_i32 = mlir::arith::TruncIOp::create(
            builder, builder.getI32Type(), byte_offset);
        Value byte_offset_tuple = mlir::fly::MakeIntTupleOp::create(
            builder, scalar_offset_type, mlir::ValueRange{byte_offset_i32});
        for (auto [output_index, result] : llvm::enumerate(results)) {
          Value output_pointer =
              argument_pointers[fusion.operand_count() + output_index];
          auto output_pointer_type =
              mlir::cast<mlir::fly::PointerType>(output_pointer.getType());
          auto output_memref_type =
              mlir::fly::MemRefType::get(output_pointer_type.getElemTy(),
                                         output_pointer_type.getAddressSpace(),
                                         scalar_layout_type.getAttr());
          Value advanced_output = mlir::fly::AddOffsetOp::create(
              builder, output_pointer_type, output_pointer, byte_offset_tuple);
          Value output_view = mlir::fly::MakeViewOp::create(
              builder, output_memref_type, advanced_output, scalar_layout);
          mlir::fly::CopyAtomCallSSA::create(builder, mlir::TypeRange{},
                                             output_copy_atom, result,
                                             output_view, true_predicate);
        }
      }
      builder.setInsertionPointAfter(tail);
    } else if (uniform_output_elements_ && tail_elements != 0) {
      auto scalar_shape_attr =
          mlir::fly::IntTupleAttr::getLeafStatic(context, 1);
      auto scalar_shape_type = mlir::fly::IntTupleType::get(scalar_shape_attr);
      Value scalar_shape = mlir::fly::MakeIntTupleOp::create(
          builder, scalar_shape_type, mlir::ValueRange{});
      Value scalar_stride = mlir::fly::MakeIntTupleOp::create(
          builder, scalar_shape_type, mlir::ValueRange{});
      auto scalar_layout_type =
          mlir::fly::LayoutType::get(scalar_shape_attr, scalar_shape_attr);
      Value scalar_layout = mlir::fly::MakeLayoutOp::create(
          builder, scalar_layout_type, scalar_shape, scalar_stride);
      absl::flat_hash_map<PrimitiveType, Value> scalar_copy_atoms;
      for (PrimitiveType type : external_types) {
        if (type == S4) {
          continue;
        }
        const int64_t element_bits = ElementBits(type);
        auto scalar_copy_atom_type = mlir::fly::CopyAtomType::get(
            mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(
                context, element_bits, /*cacheModifier=*/0),
            element_bits);
        scalar_copy_atoms[type] = mlir::fly::MakeCopyAtomOp::create(
            builder, scalar_copy_atom_type, element_bits);
      }
      Value block_zero = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, block_i64,
          mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(), 0));
      Value tail_lane = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, thread_i64,
          mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                             tail_elements));
      Value tail_valid =
          mlir::arith::AndIOp::create(builder, block_zero, tail_lane);
      mlir::scf::IfOp tail =
          mlir::scf::IfOp::create(builder, mlir::TypeRange{}, tail_valid,
                                  /*withElseRegion=*/false);
      {
        mlir::OpBuilder::InsertionGuard tail_guard(builder);
        builder.setInsertionPointToStart(tail.thenBlock());
        Value tail_offset = mlir::arith::AddIOp::create(
            builder,
            mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                               full_elements),
            thread_i64);
        Value true_predicate =
            mlir::arith::ConstantIntOp::create(builder, builder.getI1Type(), 1);
        absl::flat_hash_map<const HloInstruction*, Value> cache;
        llvm::SmallVector<Value> results;
        results.reserve(output_count_);
        for (const HloInstruction* root : roots) {
          const HloInstruction* expression =
              overwrite_row_scatter_.has_value() ? root->operand(2) : root;
          TF_ASSIGN_OR_RETURN(
              Value result,
              EmitVector(builder, argument_pointers, expression, tail_offset,
                         true_predicate, scalar_copy_atoms, scalar_layout,
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
          const PrimitiveType output_type =
              roots[output_index]->shape().element_type();
          if (overwrite_row_scatter_.has_value()) {
            TF_RET_CHECK(output_index == 0 && output_type == element_type_);
            Value scalar =
                mlir::vector::ExtractOp::create(builder, result, 0);
            RETURN_IF_ERROR(emit_scalar_scatter_store(
                scalar, tail_offset, true_predicate));
            continue;
          }
          if (output_type == PRED) {
            auto predicate_type =
                mlir::cast<mlir::VectorType>(result.getType());
            auto storage_type = mlir::VectorType::get(predicate_type.getShape(),
                                                      builder.getI8Type());
            result =
                mlir::arith::ExtUIOp::create(builder, storage_type, result);
          }
          Value output_pointer =
              argument_pointers[fusion.operand_count() + output_index];
          auto output_pointer_type =
              mlir::cast<mlir::fly::PointerType>(output_pointer.getType());
          auto output_memref_type =
              mlir::fly::MemRefType::get(output_pointer_type.getElemTy(),
                                         output_pointer_type.getAddressSpace(),
                                         scalar_layout_type.getAttr());
          Value advanced_output = mlir::fly::AddOffsetOp::create(
              builder, output_pointer_type, output_pointer, tail_offset_tuple);
          Value output_view = mlir::fly::MakeViewOp::create(
              builder, output_memref_type, advanced_output, scalar_layout);
          mlir::fly::CopyAtomCallSSA::create(
              builder, mlir::TypeRange{}, scalar_copy_atoms.at(output_type),
              result, output_view, true_predicate);
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
  std::vector<int64_t> output_elements_;
  bool uniform_output_elements_ = true;
  std::vector<int64_t> scalar_index_parameter_numbers_;
  int64_t vector_size_bits_ = 0;
  int64_t vector_width_ = 0;
  int64_t vectors_per_thread_ = 1;
  int64_t threads_ = 0;
  std::optional<StridedReductionDescriptor> cooperative_strided_reduction_;
  std::optional<OverwriteRowScatter> overwrite_row_scatter_;
  bool s4_may_start_odd_ = false;
  int32_t cache_modifier_ = 0;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

bool IsFlyXTileElementwiseFusion(const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() == 0) {
    return false;
  }
  if (analysis.fusion_root_count() == 1) {
    const HloInstruction* root =
        &analysis.fusion_root(0).instruction();
    if (GetOverwriteRowScatter(root).has_value()) {
      absl::flat_hash_set<const HloInstruction*> visited;
      if (!IsSupportedElementwiseGraph(root->operand(1),
                                       root->operand(1)->shape(), visited)) {
        return false;
      }
      visited.clear();
      return IsSupportedElementwiseGraph(root->operand(2),
                                         root->operand(2)->shape(), visited);
    }
  }
  const Shape& output_shape = analysis.fusion_root(0).shape();
  if (!output_shape.IsArray() || !output_shape.has_layout() ||
      !IsSupportedValueType(output_shape.element_type())) {
    return false;
  }
  const int64_t first_output_elements = ShapeUtil::ElementsIn(output_shape);
  if (first_output_elements == 0 ||
      ShapeUtil::ByteSizeOfElements(output_shape) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  for (int64_t root_index = 0; root_index < analysis.fusion_root_count();
       ++root_index) {
    const HloInstruction& root = analysis.fusion_root(root_index).instruction();
    if (!root.shape().IsArray() || !root.shape().has_layout() ||
        !IsSupportedValueType(root.shape().element_type()) ||
        ((root.shape().element_type() == S4) !=
         (output_shape.element_type() == S4)) ||
        ShapeUtil::ElementsIn(root.shape()) == 0 ||
        ShapeUtil::ByteSizeOfElements(root.shape()) >
            std::numeric_limits<uint32_t>::max() ||
        (output_shape.element_type() == S4 &&
         ShapeUtil::ElementsIn(root.shape()) != first_output_elements)) {
      return false;
    }
    // Every output traverses the same physical flat launch domain, but it may
    // expose that storage through a different logical bitcast view. Validate
    // each graph against its own root shape: carrying one visited set across
    // roots would incorrectly reuse a proof made under another logical view.
    absl::flat_hash_set<const HloInstruction*> visited;
    if (!IsSupportedElementwiseGraph(&root, root.shape(), visited)) {
      return false;
    }
  }
  return HasSupportedS4InputGraph(analysis);
}

bool IsFlyXTileOverwriteRowScatterFusion(
    const HloFusionAnalysis& analysis) {
  return analysis.fusion_root_count() == 1 &&
         GetOverwriteRowScatter(&analysis.fusion_root(0).instruction())
             .has_value();
}

bool IsFlyXTileAtomicOverwriteRowScatterFusion(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return false;
  }
  std::optional<OverwriteRowScatter> scatter = GetOverwriteRowScatter(
      &analysis.fusion_root(0).instruction());
  return scatter.has_value() && scatter->requires_atomic_operation;
}

int64_t GetFlyXTileElementwiseElementCount(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() == 1) {
    if (std::optional<OverwriteRowScatter> scatter = GetOverwriteRowScatter(
            &analysis.fusion_root(0).instruction())) {
      return scatter->update_rows * scatter->row_elements;
    }
  }
  int64_t elements = 0;
  for (int64_t root_index = 0; root_index < analysis.fusion_root_count();
       ++root_index) {
    elements = std::max(
        elements,
        ShapeUtil::ElementsIn(analysis.fusion_root(root_index).shape()));
  }
  return elements;
}

bool IsFlyXTileIndexedFusion(const HloFusionAnalysis& analysis) {
  if (!IsFlyXTileElementwiseFusion(analysis)) {
    return false;
  }
  if (IsFlyXTileOverwriteRowScatterFusion(analysis)) {
    return true;
  }
  absl::flat_hash_set<const HloInstruction*> visited;
  std::function<bool(const HloInstruction*)> contains_indexed_op =
      [&](const HloInstruction* instruction) {
        if (!visited.insert(instruction).second) {
          return false;
        }
        if (instruction->opcode() == HloOpcode::kConcatenate ||
            instruction->opcode() == HloOpcode::kSlice ||
            instruction->opcode() == HloOpcode::kDynamicSlice ||
            instruction->opcode() == HloOpcode::kDynamicUpdateSlice ||
            instruction->opcode() == HloOpcode::kGather ||
            (instruction->opcode() == HloOpcode::kTranspose &&
             GetDataMovingTranspose(instruction).has_value()) ||
            instruction->opcode() == HloOpcode::kReverse ||
            instruction->opcode() == HloOpcode::kPad ||
            instruction->opcode() == HloOpcode::kReduceWindow ||
            (instruction->opcode() == HloOpcode::kBroadcast &&
             !ShapeUtil::IsScalar(instruction->operand(0)->shape()))) {
          return true;
        }
        for (const HloInstruction* operand : instruction->operands()) {
          if (contains_indexed_op(operand)) {
            return true;
          }
        }
        return false;
      };
  for (int64_t root_index = 0; root_index < analysis.fusion_root_count();
       ++root_index) {
    if (contains_indexed_op(&analysis.fusion_root(root_index).instruction())) {
      return true;
    }
  }
  return false;
}

std::vector<int64_t> GetFlyXTileScalarIndexParameterNumbers(
    const HloFusionAnalysis& analysis) {
  return ScalarIndexParameterNumbers(analysis.fusion());
}

FlyXTileMemoryPolicy GetFlyXTileMemoryPolicy(
    const HloFusionAnalysis& analysis) {
  constexpr int64_t kMinimumStreamingBytes = int64_t{64} * 1024 * 1024;
  if (analysis.fusion_root_count() != 1 ||
      !IsFlyXTileElementwiseFusion(analysis) ||
      IsFlyXTileIndexedFusion(analysis)) {
    return FlyXTileMemoryPolicy::kCached;
  }

  const Shape& output_shape = analysis.fusion_root(0).shape();
  const int64_t output_elements = ShapeUtil::ElementsIn(output_shape);
  int64_t streaming_bytes = ShapeUtil::ByteSizeOfElements(output_shape);
  for (const HloInstruction* parameter : analysis.fusion().GetParameters()) {
    const Shape& shape = parameter->shape();
    if (!shape.IsArray()) {
      return FlyXTileMemoryPolicy::kCached;
    }
    if (ShapeUtil::IsScalar(shape)) {
      continue;
    }
    // Restrict bypassing to inputs consumed once in the same flat output
    // domain. Broadcasts and indexed operators can reuse cache lines and are
    // intentionally kept on the normal cache path.
    if (ShapeUtil::ElementsIn(shape) != output_elements) {
      return FlyXTileMemoryPolicy::kCached;
    }
    streaming_bytes += ShapeUtil::ByteSizeOfElements(shape);
  }
  return streaming_bytes >= kMinimumStreamingBytes
             ? FlyXTileMemoryPolicy::kNonTemporal
             : FlyXTileMemoryPolicy::kCached;
}

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileElementwiseEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileElementwiseEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
