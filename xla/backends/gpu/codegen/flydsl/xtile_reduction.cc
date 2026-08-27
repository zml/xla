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

#include "xla/backends/gpu/codegen/flydsl/xtile_reduction.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUEnums.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
#include "xla/hlo/ir/hlo_instruction.h"
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

struct RowReductionSpec {
  const HloInstruction* root;
  const HloInstruction* reduction;
  PrimitiveType output_type;
  PrimitiveType reduction_type;
  int64_t rows;
  int64_t columns;
  bool has_rowwise_output;
  std::vector<std::pair<int64_t, PrimitiveType>> input_parameters;
  int64_t max_vector_element_bits;
  HloOpcode reducer_opcode;
  std::optional<int64_t> init_parameter_number;
  double init_value;
  int64_t integer_init_value;
};

int64_t ElementBits(PrimitiveType type) {
  switch (type) {
    case PRED:
    case S8:
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

bool IsSupportedIntegerReductionType(PrimitiveType type) {
  return type == PRED || type == S8 || type == S16 || type == S32 ||
         type == S64;
}

bool IsSupportedReductionType(PrimitiveType type) {
  return type == F32 || type == F64 || type == F16 || type == BF16 ||
         IsSupportedIntegerReductionType(type);
}

bool IsSupportedReducerOpcode(PrimitiveType type, HloOpcode opcode) {
  if (opcode == HloOpcode::kAdd || opcode == HloOpcode::kMultiply ||
      opcode == HloOpcode::kMaximum || opcode == HloOpcode::kMinimum) {
    return true;
  }
  return IsSupportedIntegerReductionType(type) &&
         (opcode == HloOpcode::kAnd || opcode == HloOpcode::kOr ||
          opcode == HloOpcode::kXor);
}

mlir::Type StorageElementType(PrimitiveType type, mlir::OpBuilder& builder) {
  // XLA's device ABI stores predicates in one byte even though their logical
  // MLIR value type is i1.
  return type == PRED ? mlir::Type(builder.getI8Type())
                      : emitters::PrimitiveTypeToMlirType(type, builder);
}

std::optional<double> ScalarLiteralAsDouble(const Literal& literal,
                                            PrimitiveType type) {
  switch (type) {
    case PRED:
      return literal.GetFirstElement<bool>() ? 1.0 : 0.0;
    case S8:
      return literal.GetFirstElement<int8_t>();
    case S16:
      return literal.GetFirstElement<int16_t>();
    case S32:
      return literal.GetFirstElement<int32_t>();
    case S64:
      return static_cast<double>(literal.GetFirstElement<int64_t>());
    default:
      return literal.GetAsDouble({});
  }
}

int64_t ScalarIntegerLiteral(const Literal& literal, PrimitiveType type) {
  switch (type) {
    case PRED:
      return literal.GetFirstElement<bool>();
    case S8:
      return literal.GetFirstElement<int8_t>();
    case S16:
      return literal.GetFirstElement<int16_t>();
    case S32:
      return literal.GetFirstElement<int32_t>();
    case S64:
      return literal.GetFirstElement<int64_t>();
    default:
      LOG(FATAL) << "Unexpected Fly integer literal type.";
  }
}

bool HasSamePhysicalDimensions(const Shape& lhs, const Shape& rhs) {
  // Native row reduction addresses every input and output as a flat physical
  // buffer.  Logical singleton dimensions therefore do not change its index
  // map: [K], [1,K], and [1,1,K] are the same emitter shape when their layouts
  // make the reshape a bitcast.
  return lhs.has_layout() && rhs.has_layout() &&
         ShapeUtil::ElementsIn(lhs) == ShapeUtil::ElementsIn(rhs) &&
         ShapeUtil::ReshapeIsBitcast(lhs, rhs,
                                     /*ignore_element_type=*/true);
}

bool HasCompatibleRowShape(const Shape& input, const Shape& rows) {
  const int64_t input_rank = input.dimensions_size();
  if (input_rank < 1 || rows.dimensions_size() != input_rank - 1 ||
      !input.has_layout() || !rows.has_layout() ||
      input.layout().minor_to_major(0) != input_rank - 1) {
    return false;
  }
  for (int64_t dimension = 0; dimension < input_rank - 1; ++dimension) {
    if (input.dimensions(dimension) != rows.dimensions(dimension)) {
      return false;
    }
  }
  for (int64_t physical_dimension = 1; physical_dimension < input_rank;
       ++physical_dimension) {
    if (input.layout().minor_to_major(physical_dimension) !=
        rows.layout().minor_to_major(physical_dimension - 1)) {
      return false;
    }
  }
  return true;
}

bool HasCompatibleColumnShape(const Shape& input, const Shape& columns) {
  const int64_t input_rank = input.dimensions_size();
  return input_rank >= 1 && columns.dimensions_size() == 1 &&
         input.has_layout() && columns.has_layout() &&
         input.layout().minor_to_major(0) == input_rank - 1 &&
         columns.layout().minor_to_major(0) == 0 &&
         columns.dimensions(0) == input.dimensions(input_rank - 1);
}

bool IsColumnBroadcast(const HloInstruction& instruction,
                       const Shape& input_shape) {
  return instruction.opcode() == HloOpcode::kBroadcast &&
         instruction.operand_count() == 1 &&
         HasSamePhysicalDimensions(instruction.shape(), input_shape) &&
         HasCompatibleColumnShape(input_shape,
                                  instruction.operand(0)->shape()) &&
         instruction.dimensions().size() == 1 &&
         instruction.dimensions(0) ==
             instruction.shape().layout().minor_to_major(0);
}

bool IsRowBroadcast(const HloInstruction& instruction,
                    const Shape& input_shape) {
  if (instruction.opcode() != HloOpcode::kBroadcast ||
      instruction.operand_count() != 1 ||
      !HasSamePhysicalDimensions(instruction.shape(), input_shape) ||
      !HasCompatibleRowShape(input_shape, instruction.operand(0)->shape()) ||
      instruction.dimensions().size() !=
          instruction.operand(0)->shape().dimensions_size()) {
    return false;
  }
  for (int64_t dimension = 0; dimension < instruction.dimensions().size();
       ++dimension) {
    if (instruction.dimensions(dimension) != dimension) {
      return false;
    }
  }
  return true;
}

bool DependsOnInstruction(const HloInstruction* root,
                          const HloInstruction* target) {
  llvm::DenseSet<const HloInstruction*> visited;
  llvm::SmallVector<const HloInstruction*> worklist = {root};
  while (!worklist.empty()) {
    const HloInstruction* instruction = worklist.pop_back_val();
    if (instruction == target) {
      return true;
    }
    if (!visited.insert(instruction).second) {
      continue;
    }
    worklist.append(instruction->operands().begin(),
                    instruction->operands().end());
  }
  return false;
}

bool IsSupportedF32Unary(HloOpcode opcode) {
  return opcode == HloOpcode::kAbs || opcode == HloOpcode::kCopy ||
         opcode == HloOpcode::kExp || opcode == HloOpcode::kNegate ||
         opcode == HloOpcode::kRsqrt;
}

bool IsSupportedF32Binary(HloOpcode opcode) {
  return opcode == HloOpcode::kAdd || opcode == HloOpcode::kSubtract ||
         opcode == HloOpcode::kMultiply || opcode == HloOpcode::kDivide ||
         opcode == HloOpcode::kMaximum || opcode == HloOpcode::kMinimum;
}

bool HasDescendingLayout(const Shape& shape) {
  if (!shape.has_layout()) {
    return false;
  }
  const int64_t rank = shape.dimensions_size();
  if (shape.layout().minor_to_major_size() != rank) {
    return false;
  }
  for (int64_t physical_dimension = 0; physical_dimension < rank;
       ++physical_dimension) {
    if (shape.layout().minor_to_major(physical_dimension) !=
        rank - 1 - physical_dimension) {
      return false;
    }
  }
  return true;
}

bool IsSupportedReductionSlice(const HloInstruction& instruction,
                               const Shape& input_shape) {
  if (instruction.opcode() != HloOpcode::kSlice ||
      instruction.operand_count() != 1 ||
      !HasSamePhysicalDimensions(instruction.shape(), input_shape)) {
    return false;
  }
  const Shape& result = instruction.shape();
  const Shape& source = instruction.operand(0)->shape();
  if (result.element_type() != source.element_type() ||
      result.dimensions_size() != source.dimensions_size() ||
      !HasDescendingLayout(result) || !HasDescendingLayout(source) ||
      !LayoutUtil::Equal(result.layout(), source.layout()) ||
      !std::all_of(instruction.slice_strides().begin(),
                   instruction.slice_strides().end(),
                   [](int64_t stride) { return stride == 1; })) {
    return false;
  }
  // Vector loads must remain contiguous. Slicing outer dimensions is safe,
  // but a partial innermost physical dimension could make one load cross a
  // source-row boundary.
  const int64_t minor_dimension = result.layout().minor_to_major(0);
  return instruction.slice_starts()[minor_dimension] == 0 &&
         instruction.slice_limits()[minor_dimension] ==
             source.dimensions(minor_dimension);
}

bool IsAddReducer(const HloInstruction& reduction) {
  if (reduction.operand_count() != 2 ||
      reduction.called_computations().size() != 1) {
    return false;
  }
  const HloInstruction* init = reduction.operand(1);
  if (init->opcode() != HloOpcode::kConstant ||
      !ShapeUtil::IsScalar(init->shape()) ||
      init->shape().element_type() != F32) {
    return false;
  }
  std::optional<double> init_value = init->literal().GetAsDouble({});
  if (!init_value.has_value() || *init_value != 0.0 ||
      std::signbit(*init_value)) {
    return false;
  }
  const HloInstruction* reducer =
      reduction.called_computations().front()->root_instruction();
  return reducer->opcode() == HloOpcode::kAdd &&
         reducer->operand_count() == 2 &&
         ShapeUtil::IsScalar(reducer->shape()) &&
         reducer->shape().element_type() == F32 &&
         reducer->operand(0)->opcode() == HloOpcode::kParameter &&
         reducer->operand(1)->opcode() == HloOpcode::kParameter;
}

bool IsSupportedAuxiliaryMajorReduction(
    const HloInstruction& reduction, const Shape& target_shape,
    std::vector<std::pair<int64_t, PrimitiveType>>& input_parameters) {
  if (reduction.opcode() != HloOpcode::kReduce ||
      reduction.shape().element_type() != F32 ||
      ShapeUtil::ElementsIn(reduction.shape()) !=
          ShapeUtil::ElementsIn(target_shape) ||
      reduction.dimensions().size() != 1 || reduction.dimensions(0) != 0 ||
      !IsAddReducer(reduction) || !HasDescendingLayout(reduction.shape())) {
    return false;
  }
  const HloInstruction* input = reduction.operand(0);
  const Shape& input_shape = input->shape();
  const int64_t output_rank = reduction.shape().dimensions_size();
  if (input->opcode() != HloOpcode::kParameter ||
      input_shape.element_type() != F32 ||
      input_shape.dimensions_size() != output_rank + 1 ||
      !HasDescendingLayout(input_shape) || input_shape.dimensions(0) <= 0 ||
      input_shape.dimensions(0) > 16) {
    return false;
  }
  for (int64_t dimension = 0; dimension < output_rank; ++dimension) {
    if (input_shape.dimensions(dimension + 1) !=
        reduction.shape().dimensions(dimension)) {
      return false;
    }
  }
  input_parameters.push_back({input->parameter_number(), F32});
  return true;
}

void CollectReductions(const HloInstruction* instruction,
                       llvm::DenseSet<const HloInstruction*>& visited,
                       llvm::SmallVector<const HloInstruction*>& reductions) {
  if (!visited.insert(instruction).second) {
    return;
  }
  if (instruction->opcode() == HloOpcode::kReduce) {
    reductions.push_back(instruction);
    return;
  }
  for (const HloInstruction* operand : instruction->operands()) {
    CollectReductions(operand, visited, reductions);
  }
}

bool IsSupportedReductionInputGraph(
    const HloInstruction* instruction, const Shape& input_shape,
    llvm::DenseSet<const HloInstruction*>& visited,
    std::vector<std::pair<int64_t, PrimitiveType>>& input_parameters) {
  if (!visited.insert(instruction).second) {
    return true;
  }
  switch (instruction->opcode()) {
    case HloOpcode::kParameter: {
      const PrimitiveType type = instruction->shape().element_type();
      if ((type != F32 && type != F16 && type != BF16) ||
          !HasSamePhysicalDimensions(instruction->shape(), input_shape)) {
        return false;
      }
      input_parameters.push_back({instruction->parameter_number(), type});
      return true;
    }
    case HloOpcode::kConvert: {
      if (!HasSamePhysicalDimensions(instruction->shape(), input_shape) ||
          instruction->operand_count() != 1 ||
          !HasSamePhysicalDimensions(instruction->operand(0)->shape(),
                                     input_shape)) {
        return false;
      }
      const PrimitiveType source_type =
          instruction->operand(0)->shape().element_type();
      const PrimitiveType destination_type =
          instruction->shape().element_type();
      if (!(((source_type == F16 || source_type == BF16) &&
             destination_type == F32) ||
            (source_type == F32 &&
             (destination_type == F16 || destination_type == BF16)))) {
        return false;
      }
      return IsSupportedReductionInputGraph(
          instruction->operand(0), input_shape, visited, input_parameters);
    }
    case HloOpcode::kConstant:
      return ShapeUtil::IsScalar(instruction->shape()) &&
             instruction->shape().element_type() == F32;
    case HloOpcode::kBroadcast: {
      if (instruction->operand_count() != 1 ||
          instruction->shape().element_type() != F32 ||
          !HasSamePhysicalDimensions(instruction->shape(), input_shape)) {
        return false;
      }
      const HloInstruction* operand = instruction->operand(0);
      if (!ShapeUtil::IsEffectiveScalar(operand->shape()) &&
          !IsRowBroadcast(*instruction, input_shape) &&
          !IsColumnBroadcast(*instruction, input_shape)) {
        return false;
      }
      return IsSupportedReductionInputGraph(operand, operand->shape(), visited,
                                            input_parameters);
    }
    case HloOpcode::kBitcast: {
      if (!HasSamePhysicalDimensions(instruction->shape(), input_shape) ||
          instruction->operand_count() != 1 ||
          instruction->shape().element_type() !=
              instruction->operand(0)->shape().element_type() ||
          !ShapeUtil::ReshapeIsBitcast(instruction->operand(0)->shape(),
                                       instruction->shape())) {
        return false;
      }
      const HloInstruction* operand = instruction->operand(0);
      if (operand->opcode() == HloOpcode::kReduce) {
        return IsSupportedAuxiliaryMajorReduction(*operand, input_shape,
                                                  input_parameters);
      }
      if (operand->opcode() == HloOpcode::kConvert &&
          operand->operand_count() == 1 &&
          operand->operand(0)->opcode() == HloOpcode::kReduce) {
        const PrimitiveType source_type =
            operand->operand(0)->shape().element_type();
        const PrimitiveType destination_type = operand->shape().element_type();
        return source_type == F32 &&
               (destination_type == F16 || destination_type == BF16) &&
               IsSupportedAuxiliaryMajorReduction(
                   *operand->operand(0), input_shape, input_parameters);
      }
      // A physical bitcast only changes the logical row decomposition.  The
      // reduction emitter traverses the input as a flat buffer, so continue
      // validating the producer graph in the operand's own physical shape.
      // This covers residuals flattened by GEMM canonicalization before an
      // RMS-statistic reduction without changing any generated address.
      return IsSupportedReductionInputGraph(operand, operand->shape(), visited,
                                            input_parameters);
    }
    case HloOpcode::kSlice:
      return IsSupportedReductionSlice(*instruction, input_shape) &&
             IsSupportedReductionInputGraph(instruction->operand(0),
                                            instruction->operand(0)->shape(),
                                            visited, input_parameters);
    case HloOpcode::kReduce:
      return IsSupportedAuxiliaryMajorReduction(*instruction, input_shape,
                                                input_parameters);
    default:
      break;
  }
  if (IsSupportedF32Unary(instruction->opcode())) {
    return instruction->operand_count() == 1 &&
           instruction->shape().element_type() == F32 &&
           HasSamePhysicalDimensions(instruction->shape(), input_shape) &&
           IsSupportedReductionInputGraph(instruction->operand(0), input_shape,
                                          visited, input_parameters);
  }
  if (IsSupportedF32Binary(instruction->opcode())) {
    return instruction->operand_count() == 2 &&
           instruction->shape().element_type() == F32 &&
           HasSamePhysicalDimensions(instruction->shape(), input_shape) &&
           IsSupportedReductionInputGraph(instruction->operand(0), input_shape,
                                          visited, input_parameters) &&
           IsSupportedReductionInputGraph(instruction->operand(1), input_shape,
                                          visited, input_parameters);
  }
  return false;
}

bool IsSupportedReductionOutputGraph(
    const HloInstruction* instruction, const HloInstruction* reduction,
    const Shape& output_shape, llvm::DenseSet<const HloInstruction*>& visited,
    std::vector<std::pair<int64_t, PrimitiveType>>& input_parameters) {
  if (!visited.insert(instruction).second || instruction == reduction) {
    return true;
  }
  switch (instruction->opcode()) {
    case HloOpcode::kParameter: {
      const PrimitiveType type = instruction->shape().element_type();
      if ((type != F32 && type != F16 && type != BF16) ||
          !HasSamePhysicalDimensions(instruction->shape(), output_shape)) {
        return false;
      }
      input_parameters.push_back({instruction->parameter_number(), type});
      return true;
    }
    case HloOpcode::kConvert: {
      const PrimitiveType source_type =
          instruction->operand(0)->shape().element_type();
      const PrimitiveType destination_type =
          instruction->shape().element_type();
      const bool supported_conversion =
          ((source_type == F16 || source_type == BF16) &&
           destination_type == F32) ||
          (source_type == F32 &&
           (destination_type == F16 || destination_type == BF16));
      return instruction->operand_count() == 1 && supported_conversion &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             IsSupportedReductionOutputGraph(instruction->operand(0), reduction,
                                             output_shape, visited,
                                             input_parameters);
    }
    case HloOpcode::kBitcast:
      return instruction->operand_count() == 1 &&
             instruction->shape().element_type() ==
                 instruction->operand(0)->shape().element_type() &&
             HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
             ShapeUtil::ReshapeIsBitcast(instruction->operand(0)->shape(),
                                         instruction->shape()) &&
             IsSupportedReductionOutputGraph(instruction->operand(0), reduction,
                                             output_shape, visited,
                                             input_parameters);
    case HloOpcode::kConstant:
      return ShapeUtil::IsEffectiveScalar(instruction->shape()) &&
             instruction->shape().element_type() == F32;
    case HloOpcode::kBroadcast:
      if (instruction->operand_count() != 1 ||
          instruction->shape().element_type() != F32 ||
          !HasSamePhysicalDimensions(instruction->shape(), output_shape)) {
        return false;
      }
      return IsSupportedReductionOutputGraph(
          instruction->operand(0), reduction,
          ShapeUtil::IsEffectiveScalar(instruction->operand(0)->shape())
              ? instruction->operand(0)->shape()
              : output_shape,
          visited, input_parameters);
    default:
      break;
  }
  if (IsSupportedF32Unary(instruction->opcode())) {
    return instruction->operand_count() == 1 &&
           instruction->shape().element_type() == F32 &&
           HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
           IsSupportedReductionOutputGraph(instruction->operand(0), reduction,
                                           output_shape, visited,
                                           input_parameters);
  }
  if (IsSupportedF32Binary(instruction->opcode())) {
    return instruction->operand_count() == 2 &&
           instruction->shape().element_type() == F32 &&
           HasSamePhysicalDimensions(instruction->shape(), output_shape) &&
           IsSupportedReductionOutputGraph(instruction->operand(0), reduction,
                                           output_shape, visited,
                                           input_parameters) &&
           IsSupportedReductionOutputGraph(instruction->operand(1), reduction,
                                           output_shape, visited,
                                           input_parameters);
  }
  return false;
}

bool IsSupportedRowwiseOutputGraph(
    const HloInstruction* instruction, const HloInstruction* reduction,
    const Shape& input_shape, const Shape& row_shape,
    llvm::DenseSet<const HloInstruction*>& visited,
    std::vector<std::pair<int64_t, PrimitiveType>>& input_parameters) {
  if (!visited.insert(instruction).second || instruction == reduction) {
    return true;
  }
  const bool has_input_shape =
      HasSamePhysicalDimensions(instruction->shape(), input_shape);
  const bool has_row_shape =
      HasSamePhysicalDimensions(instruction->shape(), row_shape);
  const bool has_column_shape =
      HasCompatibleColumnShape(input_shape, instruction->shape());
  switch (instruction->opcode()) {
    case HloOpcode::kParameter: {
      const PrimitiveType type = instruction->shape().element_type();
      if ((!has_input_shape && !has_row_shape && !has_column_shape) ||
          (type != F32 && type != F16 && type != BF16)) {
        return false;
      }
      input_parameters.push_back({instruction->parameter_number(), type});
      return true;
    }
    case HloOpcode::kConvert: {
      if (instruction->operand_count() != 1 ||
          (!has_input_shape && !has_row_shape && !has_column_shape)) {
        return false;
      }
      const PrimitiveType source_type =
          instruction->operand(0)->shape().element_type();
      const PrimitiveType destination_type =
          instruction->shape().element_type();
      const bool supported_conversion =
          ((source_type == F16 || source_type == BF16) &&
           destination_type == F32) ||
          (source_type == F32 &&
           (destination_type == F16 || destination_type == BF16));
      return supported_conversion &&
             IsSupportedRowwiseOutputGraph(instruction->operand(0), reduction,
                                           input_shape, row_shape, visited,
                                           input_parameters);
    }
    case HloOpcode::kConstant:
      return ShapeUtil::IsEffectiveScalar(instruction->shape()) &&
             instruction->shape().element_type() == F32;
    case HloOpcode::kBroadcast: {
      const bool is_column_broadcast =
          IsColumnBroadcast(*instruction, input_shape);
      const PrimitiveType type = instruction->shape().element_type();
      return instruction->operand_count() == 1 &&
             (type == F32 ||
              (is_column_broadcast && (type == F16 || type == BF16))) &&
             (has_input_shape || has_row_shape) &&
             (ShapeUtil::IsEffectiveScalar(
                  instruction->operand(0)->shape()) ||
              HasSamePhysicalDimensions(instruction->operand(0)->shape(),
                                        row_shape) ||
              is_column_broadcast) &&
             IsSupportedRowwiseOutputGraph(instruction->operand(0), reduction,
                                           input_shape, row_shape, visited,
                                           input_parameters);
    }
    case HloOpcode::kBitcast: {
      if ((!has_input_shape && !has_row_shape) ||
          instruction->operand_count() != 1 ||
          instruction->shape().element_type() !=
              instruction->operand(0)->shape().element_type() ||
          !ShapeUtil::ReshapeIsBitcast(instruction->operand(0)->shape(),
                                       instruction->shape())) {
        return false;
      }
      if (has_input_shape) {
        if (DependsOnInstruction(instruction->operand(0), reduction)) {
          return IsSupportedRowwiseOutputGraph(
              instruction->operand(0), reduction, input_shape, row_shape,
              visited, input_parameters);
        }
        llvm::DenseSet<const HloInstruction*> input_visited;
        return IsSupportedReductionInputGraph(instruction, input_shape,
                                              input_visited, input_parameters);
      }
      return IsSupportedRowwiseOutputGraph(instruction->operand(0), reduction,
                                           input_shape, row_shape, visited,
                                           input_parameters);
    }
    case HloOpcode::kReduce:
      return has_input_shape && instruction != reduction &&
             IsSupportedAuxiliaryMajorReduction(*instruction, input_shape,
                                                input_parameters);
    default:
      break;
  }
  if (IsSupportedF32Unary(instruction->opcode())) {
    return instruction->operand_count() == 1 &&
           instruction->shape().element_type() == F32 &&
           (has_input_shape || has_row_shape || has_column_shape) &&
           IsSupportedRowwiseOutputGraph(instruction->operand(0), reduction,
                                         input_shape, row_shape, visited,
                                         input_parameters);
  }
  if (IsSupportedF32Binary(instruction->opcode())) {
    return instruction->operand_count() == 2 &&
           instruction->shape().element_type() == F32 &&
           (has_input_shape || has_row_shape || has_column_shape) &&
           IsSupportedRowwiseOutputGraph(instruction->operand(0), reduction,
                                         input_shape, row_shape, visited,
                                         input_parameters) &&
           IsSupportedRowwiseOutputGraph(instruction->operand(1), reduction,
                                         input_shape, row_shape, visited,
                                         input_parameters);
  }
  return false;
}

std::optional<RowReductionSpec> MatchRowReduction(const HloInstruction& root) {
  const PrimitiveType output_type = root.shape().element_type();
  if (!IsSupportedReductionType(output_type) || !root.shape().has_layout() ||
      ShapeUtil::IsZeroElementArray(root.shape())) {
    return std::nullopt;
  }
  if (output_type != F32 &&
      !((root.opcode() == HloOpcode::kConvert && root.operand_count() == 1 &&
         root.operand(0)->shape().element_type() == F32 &&
         HasSamePhysicalDimensions(root.shape(), root.operand(0)->shape())) ||
        root.opcode() == HloOpcode::kReduce)) {
    return std::nullopt;
  }

  llvm::DenseSet<const HloInstruction*> reduction_visited;
  llvm::SmallVector<const HloInstruction*> reductions;
  CollectReductions(&root, reduction_visited, reductions);
  llvm::SmallVector<const HloInstruction*> row_reductions;
  for (const HloInstruction* candidate : reductions) {
    if (candidate->operand_count() != 2 ||
        !IsSupportedReductionType(candidate->shape().element_type()) ||
        candidate->dimensions().size() != 1 ||
        candidate->called_computations().size() != 1) {
      continue;
    }
    const HloInstruction* candidate_input = candidate->operand(0);
    if (candidate_input->shape().element_type() !=
            candidate->shape().element_type() ||
        candidate_input->shape().dimensions_size() < 1 ||
        candidate->dimensions(0) !=
            candidate_input->shape().dimensions_size() - 1 ||
        !HasCompatibleRowShape(candidate_input->shape(), candidate->shape())) {
      continue;
    }
    if (HasSamePhysicalDimensions(root.shape(), candidate_input->shape()) ||
        HasSamePhysicalDimensions(root.shape(), candidate->shape())) {
      row_reductions.push_back(candidate);
    }
  }
  if (row_reductions.size() != 1) {
    return std::nullopt;
  }
  const HloInstruction* reduction = row_reductions.front();
  const PrimitiveType reduction_type = reduction->shape().element_type();

  const HloInstruction* input = reduction->operand(0);

  std::vector<std::pair<int64_t, PrimitiveType>> input_parameters;
  if (reduction_type == F32) {
    llvm::DenseSet<const HloInstruction*> input_visited;
    if (!IsSupportedReductionInputGraph(input, input->shape(), input_visited,
                                        input_parameters) ||
        input_parameters.empty()) {
      return std::nullopt;
    }
  } else {
    // XLA/Triton permit direct F16/BF16 and integral row reductions. Widen the
    // low-precision floating values to F32 and the sub-dword integral values
    // to i32 while keeping their external ABI. More elaborate non-F32 input
    // and epilogue DAGs continue to use the generic reduction emitter.
    if (input->opcode() != HloOpcode::kParameter ||
        input->shape().element_type() != reduction_type) {
      return std::nullopt;
    }
    input_parameters.push_back(
        {input->parameter_number(), input->shape().element_type()});
  }
  const bool has_rowwise_output =
      HasSamePhysicalDimensions(root.shape(), input->shape());
  if (has_rowwise_output) {
    llvm::DenseSet<const HloInstruction*> output_visited;
    if (!IsSupportedRowwiseOutputGraph(&root, reduction, input->shape(),
                                       reduction->shape(), output_visited,
                                       input_parameters)) {
      return std::nullopt;
    }
  } else {
    if (!HasSamePhysicalDimensions(root.shape(), reduction->shape())) {
      return std::nullopt;
    }
    llvm::DenseSet<const HloInstruction*> output_visited;
    if (!IsSupportedReductionOutputGraph(&root, reduction, root.shape(),
                                         output_visited, input_parameters)) {
      return std::nullopt;
    }
  }
  const HloInstruction* init = reduction->operand(1);
  if (!ShapeUtil::IsScalar(init->shape()) ||
      init->shape().element_type() != reduction_type) {
    return std::nullopt;
  }
  std::optional<int64_t> init_parameter_number;
  double init_value = 0.0;
  int64_t integer_init_value = 0;
  if (init->opcode() == HloOpcode::kConstant) {
    std::optional<double> literal_value =
        ScalarLiteralAsDouble(init->literal(), reduction_type);
    if (!literal_value.has_value()) {
      return std::nullopt;
    }
    init_value = *literal_value;
    if (IsSupportedIntegerReductionType(reduction_type)) {
      integer_init_value =
          ScalarIntegerLiteral(init->literal(), reduction_type);
    }
  } else if (init->opcode() == HloOpcode::kParameter) {
    init_parameter_number = init->parameter_number();
    input_parameters.push_back({*init_parameter_number, reduction_type});
  } else {
    return std::nullopt;
  }

  std::sort(input_parameters.begin(), input_parameters.end());
  input_parameters.erase(
      std::unique(input_parameters.begin(), input_parameters.end()),
      input_parameters.end());

  const HloComputation* reducer = reduction->called_computations().front();
  const HloInstruction* reducer_root = reducer->root_instruction();
  if (!IsSupportedReducerOpcode(reduction_type, reducer_root->opcode()) ||
      reducer_root->operand_count() != 2 ||
      !ShapeUtil::IsScalar(reducer_root->shape()) ||
      reducer_root->shape().element_type() != reduction_type ||
      reducer_root->operand(0)->opcode() != HloOpcode::kParameter ||
      reducer_root->operand(1)->opcode() != HloOpcode::kParameter) {
    return std::nullopt;
  }

  const int64_t rows = ShapeUtil::ElementsIn(reduction->shape());
  const int64_t columns =
      input->shape().dimensions(input->shape().dimensions_size() - 1);
  int64_t max_vector_element_bits =
      has_rowwise_output ? ElementBits(output_type) : 0;
  for (auto [parameter_number, input_type] : input_parameters) {
    max_vector_element_bits =
        std::max(max_vector_element_bits, ElementBits(input_type));
    if (parameter_number < 0 ||
        parameter_number >= root.parent()->num_parameters()) {
      return std::nullopt;
    }
    const HloInstruction* parameter =
        root.parent()->parameter_instruction(parameter_number);
    if (parameter == nullptr ||
        parameter->shape().element_type() != input_type) {
      return std::nullopt;
    }
    const int64_t input_elements = ShapeUtil::ElementsIn(parameter->shape());
    if (input_elements >
        std::numeric_limits<uint32_t>::max() / (ElementBits(input_type) / 8)) {
      return std::nullopt;
    }
  }
  if (rows <= 0 || columns <= 0) {
    return std::nullopt;
  }

  return RowReductionSpec{&root,
                          reduction,
                          output_type,
                          reduction_type,
                          rows,
                          columns,
                          has_rowwise_output,
                          std::move(input_parameters),
                          max_vector_element_bits,
                          reducer_root->opcode(),
                          init_parameter_number,
                          init_value,
                          integer_init_value};
}

class FlyXTileRowReductionEmitter final : public MlirKernelEmitter {
 public:
  explicit FlyXTileRowReductionEmitter(const HloFusionAnalysis& analysis) {
    std::optional<RowReductionSpec> spec =
        MatchRowReduction(analysis.fusion_root(0).instruction());
    CHECK(spec.has_value());
    rows_ = spec->rows;
    columns_ = spec->columns;
    output_type_ = spec->output_type;
    reduction_type_ = spec->reduction_type;
    has_rowwise_output_ = spec->has_rowwise_output;
    max_vector_element_bits_ = spec->max_vector_element_bits;
    reducer_opcode_ = spec->reducer_opcode;
    init_parameter_number_ = spec->init_parameter_number;
    init_value_ = spec->init_value;
    integer_init_value_ = spec->integer_init_value;

    const BlockLevelFusionConfig& config =
        analysis.fusion_backend_config().block_level_fusion_config();
    num_warps_ = config.num_warps();
    CHECK_GT(num_warps_, 0);
    CHECK_LE(num_warps_, 16);
    CHECK_EQ(config.output_tiles_size(), 1);
    CHECK_GT(config.output_tiles(0).sizes_size(), 0);
    output_partitions_ =
        has_rowwise_output_ ? config.output_tiles(0).sizes(0) : 1;
    CHECK_GT(output_partitions_, 0);
    CHECK_LE(output_partitions_, 16);
    vector_size_bits_ =
        config.vector_size_bits() == 0 ? 128 : config.vector_size_bits();
    CHECK(vector_size_bits_ == 16 || vector_size_bits_ == 32 ||
          vector_size_bits_ == 64 || vector_size_bits_ == 128);
    CHECK_GE(vector_size_bits_, max_vector_element_bits_);
    CHECK_EQ(vector_size_bits_ % max_vector_element_bits_, 0);
    vector_width_ = vector_size_bits_ / max_vector_element_bits_;
    if (output_partitions_ > 1) {
      CHECK_EQ(columns_ % (64 * vector_width_ * output_partitions_), 0);
    }
    launch_dimensions_ = LaunchDimensions(
        se::BlockDim((rows_ * output_partitions_ + num_warps_ - 1) / num_warps_,
                     1, 1),
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
      const BufferAssignment*) const override {
    context.getOrLoadDialect<mlir::amdgpu::AMDGPUDialect>();
    context.getOrLoadDialect<mlir::fly::FlyDialect>();
    context.getOrLoadDialect<mlir::fly_rocdl::FlyROCDLDialect>();
    context.getOrLoadDialect<mlir::gpu::GPUDialect>();
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
        module_builder, location, "fly_reduction_kernels");
    module_builder.setInsertionPointToStart(
        &gpu_module.getBodyRegion().front());

    mlir::fly::AddressSpaceAttr global_address =
        mlir::fly::AddressSpaceAttr::get(&context,
                                         mlir::fly::AddressSpace::Global);
    llvm::SmallVector<mlir::Type> argument_types;
    argument_types.reserve(fusion.operand_count() + 1);
    for (const HloInstruction* operand : fusion.operands()) {
      mlir::Type element_type =
          StorageElementType(operand->shape().element_type(), module_builder);
      argument_types.push_back(
          mlir::fly::PointerType::get(element_type, global_address));
    }
    mlir::Type output_type = StorageElementType(output_type_, module_builder);
    argument_types.push_back(
        mlir::fly::PointerType::get(output_type, global_address));
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
        "FlyXTileRowReductionEmitter builds a native gpu.func module.");
  }

  Value I64(mlir::ImplicitLocOpBuilder& builder, int64_t value) const {
    return mlir::arith::ConstantIntOp::create(builder, builder.getI64Type(),
                                              value);
  }

  Value FloatComputeConstant(mlir::ImplicitLocOpBuilder& builder,
                             double value) const {
    if (reduction_type_ == F64) {
      return mlir::arith::ConstantFloatOp::create(builder, builder.getF64Type(),
                                                  llvm::APFloat(value));
    }
    return mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(),
        llvm::APFloat(static_cast<float>(value)));
  }

  bool UsesIntegerCompute() const {
    return IsSupportedIntegerReductionType(reduction_type_);
  }

  mlir::Type ComputeElementType(mlir::ImplicitLocOpBuilder& builder) const {
    if (reduction_type_ == S64) {
      return builder.getI64Type();
    }
    if (reduction_type_ == F64) {
      return builder.getF64Type();
    }
    return UsesIntegerCompute() ? mlir::Type(builder.getI32Type())
                                : mlir::Type(builder.getF32Type());
  }

  int64_t IntegerNeutralValue() const {
    CHECK(UsesIntegerCompute());
    switch (reducer_opcode_) {
      case HloOpcode::kAdd:
        return 0;
      case HloOpcode::kMultiply:
        return 1;
      case HloOpcode::kAnd:
        return reduction_type_ == PRED ? 1 : -1;
      case HloOpcode::kOr:
      case HloOpcode::kXor:
        return 0;
      case HloOpcode::kMaximum:
        if (reduction_type_ == PRED) {
          return 0;
        }
        return reduction_type_ == S8    ? std::numeric_limits<int8_t>::min()
               : reduction_type_ == S16 ? std::numeric_limits<int16_t>::min()
               : reduction_type_ == S32 ? std::numeric_limits<int32_t>::min()
                                        : std::numeric_limits<int64_t>::min();
      case HloOpcode::kMinimum:
        if (reduction_type_ == PRED) {
          return 1;
        }
        return reduction_type_ == S8    ? std::numeric_limits<int8_t>::max()
               : reduction_type_ == S16 ? std::numeric_limits<int16_t>::max()
               : reduction_type_ == S32 ? std::numeric_limits<int32_t>::max()
                                        : std::numeric_limits<int64_t>::max();
      default:
        LOG(FATAL) << "Unexpected Fly integer row reduction opcode.";
    }
  }

  Value ComputeConstant(mlir::ImplicitLocOpBuilder& builder,
                        double value) const {
    CHECK(!UsesIntegerCompute());
    return FloatComputeConstant(builder, value);
  }

  Value IntegerComputeConstant(mlir::ImplicitLocOpBuilder& builder,
                               int64_t value) const {
    CHECK(UsesIntegerCompute());
    return mlir::arith::ConstantIntOp::create(
        builder, ComputeElementType(builder), value);
  }

  double NeutralValue() const {
    switch (reducer_opcode_) {
      case HloOpcode::kAdd:
        return 0.0;
      case HloOpcode::kMultiply:
        return 1.0;
      case HloOpcode::kMaximum:
        return -std::numeric_limits<double>::infinity();
      case HloOpcode::kMinimum:
        return std::numeric_limits<double>::infinity();
      default:
        LOG(FATAL) << "Unexpected Fly row reduction opcode.";
    }
  }

  bool HasNeutralInit() const {
    if (init_parameter_number_.has_value()) {
      return false;
    }
    if (UsesIntegerCompute()) {
      return integer_init_value_ == IntegerNeutralValue();
    }
    switch (reducer_opcode_) {
      case HloOpcode::kAdd:
        // Preserve an explicit -0.0 init: although it compares equal to zero,
        // it is not bitwise the additive identity for an all-zero row.
        return init_value_ == 0.0 && !std::signbit(init_value_);
      case HloOpcode::kMultiply:
        return init_value_ == 1.0;
      case HloOpcode::kMaximum:
        return init_value_ == -std::numeric_limits<double>::infinity();
      case HloOpcode::kMinimum:
        return init_value_ == std::numeric_limits<double>::infinity();
      default:
        LOG(FATAL) << "Unexpected Fly row reduction opcode.";
    }
  }

  Value Combine(mlir::ImplicitLocOpBuilder& builder, Value lhs,
                Value rhs) const {
    if (UsesIntegerCompute()) {
      switch (reducer_opcode_) {
        case HloOpcode::kAdd:
          return reduction_type_ == PRED
                     ? mlir::arith::XOrIOp::create(builder, lhs, rhs)
                           .getResult()
                     : mlir::arith::AddIOp::create(builder, lhs, rhs)
                           .getResult();
        case HloOpcode::kMultiply:
          return reduction_type_ == PRED
                     ? mlir::arith::AndIOp::create(builder, lhs, rhs)
                           .getResult()
                     : mlir::arith::MulIOp::create(builder, lhs, rhs)
                           .getResult();
        case HloOpcode::kAnd:
          return mlir::arith::AndIOp::create(builder, lhs, rhs).getResult();
        case HloOpcode::kOr:
          return mlir::arith::OrIOp::create(builder, lhs, rhs).getResult();
        case HloOpcode::kXor:
          return mlir::arith::XOrIOp::create(builder, lhs, rhs).getResult();
        case HloOpcode::kMaximum:
          return reduction_type_ == PRED
                     ? mlir::arith::OrIOp::create(builder, lhs, rhs).getResult()
                     : mlir::arith::MaxSIOp::create(builder, lhs, rhs)
                           .getResult();
        case HloOpcode::kMinimum:
          return reduction_type_ == PRED
                     ? mlir::arith::AndIOp::create(builder, lhs, rhs)
                           .getResult()
                     : mlir::arith::MinSIOp::create(builder, lhs, rhs)
                           .getResult();
        default:
          LOG(FATAL) << "Unexpected Fly integer row reduction opcode.";
      }
    }
    return EmitF32Binary(builder, reducer_opcode_, lhs, rhs);
  }

  Value EmitF32Binary(mlir::ImplicitLocOpBuilder& builder, HloOpcode opcode,
                      Value lhs, Value rhs) const {
    switch (opcode) {
      case HloOpcode::kAdd:
        // Preserve two-lane F32 vector adds through MLIR's vector lowering so
        // the AMD backend can select v_pk_add_f32. arith.addf vectors are
        // scalarized before LLVM and lose that packed instruction opportunity.
        if (mlir::isa<mlir::VectorType>(lhs.getType())) {
          return mlir::LLVM::FAddOp::create(builder, lhs, rhs);
        }
        return mlir::arith::AddFOp::create(builder, lhs, rhs);
      case HloOpcode::kSubtract:
        return mlir::arith::SubFOp::create(builder, lhs, rhs);
      case HloOpcode::kMultiply:
        return mlir::arith::MulFOp::create(builder, lhs, rhs);
      case HloOpcode::kDivide:
        return mlir::arith::DivFOp::create(builder, lhs, rhs);
      case HloOpcode::kMaximum:
        return mlir::arith::MaximumFOp::create(builder, lhs, rhs);
      case HloOpcode::kMinimum:
        return mlir::arith::MinimumFOp::create(builder, lhs, rhs);
      default:
        LOG(FATAL) << "Unexpected Fly F32 binary opcode.";
    }
  }

  Value EmitF32Unary(mlir::ImplicitLocOpBuilder& builder, HloOpcode opcode,
                     Value operand) const {
    auto emit_scalar = [&](Value scalar) -> Value {
      switch (opcode) {
        case HloOpcode::kAbs:
          return mlir::math::AbsFOp::create(builder, scalar);
        case HloOpcode::kExp:
          return mlir::math::ExpOp::create(builder, scalar);
        case HloOpcode::kNegate:
          return mlir::arith::NegFOp::create(builder, scalar);
        case HloOpcode::kRsqrt:
          return mlir::math::RsqrtOp::create(builder, scalar);
        default:
          LOG(FATAL) << "Unexpected Fly F32 unary opcode.";
      }
    };
    auto vector_type = mlir::dyn_cast<mlir::VectorType>(operand.getType());
    if (!vector_type) {
      return emit_scalar(operand);
    }
    llvm::SmallVector<Value> elements;
    elements.reserve(vector_type.getNumElements());
    for (int64_t lane = 0; lane < vector_type.getNumElements(); ++lane) {
      Value scalar = mlir::vector::ExtractOp::create(builder, operand, lane);
      elements.push_back(emit_scalar(scalar));
    }
    return mlir::vector::FromElementsOp::create(builder, vector_type, elements);
  }

  mlir::vector::CombiningKind CombiningKind() const {
    if (UsesIntegerCompute()) {
      switch (reducer_opcode_) {
        case HloOpcode::kAdd:
          return reduction_type_ == PRED ? mlir::vector::CombiningKind::XOR
                                         : mlir::vector::CombiningKind::ADD;
        case HloOpcode::kMultiply:
          return reduction_type_ == PRED ? mlir::vector::CombiningKind::AND
                                         : mlir::vector::CombiningKind::MUL;
        case HloOpcode::kAnd:
          return mlir::vector::CombiningKind::AND;
        case HloOpcode::kOr:
          return mlir::vector::CombiningKind::OR;
        case HloOpcode::kXor:
          return mlir::vector::CombiningKind::XOR;
        case HloOpcode::kMaximum:
          return reduction_type_ == PRED ? mlir::vector::CombiningKind::OR
                                         : mlir::vector::CombiningKind::MAXSI;
        case HloOpcode::kMinimum:
          return reduction_type_ == PRED ? mlir::vector::CombiningKind::AND
                                         : mlir::vector::CombiningKind::MINSI;
        default:
          LOG(FATAL) << "Unexpected Fly integer row reduction opcode.";
      }
    }
    switch (reducer_opcode_) {
      case HloOpcode::kAdd:
        return mlir::vector::CombiningKind::ADD;
      case HloOpcode::kMultiply:
        return mlir::vector::CombiningKind::MUL;
      case HloOpcode::kMaximum:
        return mlir::vector::CombiningKind::MAXIMUMF;
      case HloOpcode::kMinimum:
        return mlir::vector::CombiningKind::MINIMUMF;
      default:
        LOG(FATAL) << "Unexpected Fly row reduction opcode.";
    }
  }

  Value SplatInteger(mlir::ImplicitLocOpBuilder& builder, mlir::VectorType type,
                     int64_t value) const {
    return mlir::arith::ConstantOp::create(
        builder, type,
        mlir::DenseElementsAttr::get(
            type, builder.getIntegerAttr(type.getElementType(), value)));
  }

  Value Bf16ToF32(mlir::ImplicitLocOpBuilder& builder, Value input) const {
    auto input_type = mlir::cast<mlir::VectorType>(input.getType());
    auto bf16_pair_type = mlir::VectorType::get({2}, builder.getBF16Type());
    auto i16_pair_type = mlir::VectorType::get({2}, builder.getI16Type());
    auto i32_pair_type = mlir::VectorType::get({2}, builder.getI32Type());
    auto f32_pair_type = mlir::VectorType::get({2}, builder.getF32Type());
    llvm::SmallVector<Value> elements;
    elements.reserve(input_type.getNumElements());
    for (int64_t pair = 0; pair < input_type.getNumElements() / 2; ++pair) {
      llvm::SmallVector<int64_t, 2> mask{2 * pair, 2 * pair + 1};
      Value values = mlir::vector::ShuffleOp::create(builder, bf16_pair_type,
                                                     input, input, mask);
      values = mlir::arith::BitcastOp::create(builder, i16_pair_type, values);
      values = mlir::arith::ExtUIOp::create(builder, i32_pair_type, values);
      values = mlir::arith::ShLIOp::create(
          builder, values, SplatInteger(builder, i32_pair_type, 16));
      values = mlir::arith::BitcastOp::create(builder, f32_pair_type, values);
      elements.push_back(mlir::vector::ExtractOp::create(builder, values, 0));
      elements.push_back(mlir::vector::ExtractOp::create(builder, values, 1));
    }
    if (input_type.getNumElements() % 2 != 0) {
      Value value = mlir::vector::ExtractOp::create(
          builder, input, input_type.getNumElements() - 1);
      value =
          mlir::arith::BitcastOp::create(builder, builder.getI16Type(), value);
      value =
          mlir::arith::ExtUIOp::create(builder, builder.getI32Type(), value);
      value =
          mlir::arith::ShLIOp::create(builder, value,
                                      mlir::arith::ConstantIntOp::create(
                                          builder, builder.getI32Type(), 16));
      elements.push_back(
          mlir::arith::BitcastOp::create(builder, builder.getF32Type(), value));
    }
    auto result_type = mlir::VectorType::get({input_type.getNumElements()},
                                             builder.getF32Type());
    return mlir::vector::FromElementsOp::create(builder, result_type, elements);
  }

  Value RoundF32ToBf16(mlir::ImplicitLocOpBuilder& builder, Value input) const {
    auto input_type = mlir::cast<mlir::VectorType>(input.getType());
    auto i32_type =
        mlir::VectorType::get(input_type.getShape(), builder.getI32Type());
    auto bf16_type =
        mlir::VectorType::get(input_type.getShape(), builder.getBF16Type());

    // gfx942 has no v_cvt_pk_bf16_f32. Follow FlyDSL's normalization kernels:
    // round-to-nearest-even in the integer domain, then pack the high words.
    // This avoids LLVM's substantially longer generic fptrunc expansion.
    Value bits = mlir::arith::BitcastOp::create(builder, i32_type, input);
    Value upper = mlir::arith::ShRUIOp::create(
        builder, bits, SplatInteger(builder, i32_type, 16));
    Value lsb = mlir::arith::AndIOp::create(builder, upper,
                                            SplatInteger(builder, i32_type, 1));
    Value bias = mlir::arith::AddIOp::create(
        builder, lsb, SplatInteger(builder, i32_type, 0x7FFF));
    Value rounded = mlir::arith::AddIOp::create(builder, bits, bias);
    Value rounded_upper = mlir::arith::ShRUIOp::create(
        builder, rounded, SplatInteger(builder, i32_type, 16));

    const int64_t elements = input_type.getNumElements();
    if (elements % 2 != 0) {
      auto i16_type =
          mlir::VectorType::get(input_type.getShape(), builder.getI16Type());
      Value bf16_bits =
          mlir::arith::TruncIOp::create(builder, i16_type, rounded_upper);
      return mlir::arith::BitcastOp::create(builder, bf16_type, bf16_bits);
    }

    // Preserve FlyDSL's explicit packing as well as its rounding. Keeping each
    // adjacent BF16 pair in one i32 lets the AMD backend select one v_perm_b32
    // per pair instead of first materializing a vector<i16> conversion.
    llvm::SmallVector<int64_t> even_indices;
    llvm::SmallVector<int64_t> odd_indices;
    even_indices.reserve(elements / 2);
    odd_indices.reserve(elements / 2);
    for (int64_t element = 0; element < elements; element += 2) {
      even_indices.push_back(element);
      odd_indices.push_back(element + 1);
    }
    auto packed_type =
        mlir::VectorType::get({elements / 2}, builder.getI32Type());
    Value even = mlir::vector::ShuffleOp::create(
        builder, packed_type, rounded_upper, rounded_upper, even_indices);
    Value odd = mlir::vector::ShuffleOp::create(
        builder, packed_type, rounded_upper, rounded_upper, odd_indices);
    odd = mlir::arith::ShLIOp::create(builder, odd,
                                      SplatInteger(builder, packed_type, 16));
    Value packed = mlir::arith::OrIOp::create(builder, even, odd);
    return mlir::arith::BitcastOp::create(builder, bf16_type, packed);
  }

  Value MakeDescriptorPointer(mlir::ImplicitLocOpBuilder& builder,
                              Value pointer, PrimitiveType element_type,
                              int64_t elements) const {
    auto address =
        mlir::fly_rocdl::BufferDescAddressAttr::get(builder.getContext());
    auto pointer_type = mlir::fly::PointerType::get(
        emitters::PrimitiveTypeToMlirType(element_type, builder), address);
    Value stride =
        mlir::arith::ConstantIntOp::create(builder, builder.getI16Type(), 0);
    Value extent = mlir::arith::ConstantIntOp::create(
        builder, builder.getI64Type(),
        elements * (ElementBits(element_type) / 8));
    Value flags = mlir::arith::ConstantIntOp::create(
        builder, builder.getI32Type(), 0x27000);
    return mlir::fly::MakePtrOp::create(
        builder, pointer_type, mlir::ValueRange{pointer, stride, extent, flags},
        /*dictAttrs=*/nullptr);
  }

  struct InputState {
    PrimitiveType element_type;
    mlir::VectorType vector_type;
    mlir::Type pointer_type;
    mlir::Type memref_type;
    mlir::Type offset_type;
    Value pointer;
    Value layout;
    Value copy_atom;
    Value neutral_vector;
  };

  InputState CreateInputState(mlir::ImplicitLocOpBuilder& builder,
                              mlir::gpu::GPUFuncOp kernel,
                              int64_t parameter_number,
                              PrimitiveType element_type, int64_t elements,
                              int64_t vector_width) const {
    mlir::MLIRContext* context = builder.getContext();
    mlir::Type mlir_element_type = StorageElementType(element_type, builder);
    auto vector_type = mlir::VectorType::get({vector_width}, mlir_element_type);
    auto shape_attr =
        mlir::fly::IntTupleAttr::getLeafStatic(context, vector_width);
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

    const int64_t copy_bits = vector_width * ElementBits(element_type);
    auto copy_atom_type = mlir::fly::CopyAtomType::get(
        mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context, copy_bits,
                                                        /*cacheModifier=*/0),
        ElementBits(element_type));
    Value copy_atom = mlir::fly::MakeCopyAtomOp::create(
        builder, copy_atom_type, ElementBits(element_type));
    Value pointer = MakeDescriptorPointer(
        builder, kernel.getArgument(parameter_number), element_type, elements);
    auto pointer_type = mlir::cast<mlir::fly::PointerType>(pointer.getType());
    auto memref_type = mlir::fly::MemRefType::get(
        mlir_element_type, pointer_type.getAddressSpace(),
        layout_type.getAttr());
    auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
        context, /*width=*/32, /*divisibility=*/vector_width);
    auto offset_type = mlir::fly::IntTupleType::get(offset_attr);

    mlir::Attribute neutral =
        UsesIntegerCompute() ? mlir::Attribute(builder.getIntegerAttr(
                                   mlir_element_type, IntegerNeutralValue()))
                             : mlir::Attribute(builder.getFloatAttr(
                                   mlir_element_type, NeutralValue()));
    Value neutral_vector = mlir::arith::ConstantOp::create(
        builder, vector_type,
        mlir::DenseElementsAttr::get(vector_type, neutral));
    return InputState{element_type, vector_type, pointer_type,
                      memref_type,  offset_type, pointer,
                      layout,       copy_atom,   neutral_vector};
  }

  absl::StatusOr<Value> EmitInputLoad(
      mlir::ImplicitLocOpBuilder& builder, int64_t parameter_number,
      Value element_offset, Value predicate,
      const std::vector<std::optional<InputState>>& input_states) const {
    TF_RET_CHECK(parameter_number >= 0 &&
                 parameter_number < input_states.size() &&
                 input_states[parameter_number].has_value());
    const InputState& state = *input_states[parameter_number];
    Value offset_i32 = mlir::arith::TruncIOp::create(
        builder, builder.getI32Type(), element_offset);
    Value offset_tuple = mlir::fly::MakeIntTupleOp::create(
        builder, state.offset_type, mlir::ValueRange{offset_i32});
    Value advanced = mlir::fly::AddOffsetOp::create(
        builder, state.pointer_type, state.pointer, offset_tuple);
    Value view = mlir::fly::MakeViewOp::create(builder, state.memref_type,
                                               advanced, state.layout);
    return mlir::fly::CopyAtomCallSSA::create(
               builder, mlir::TypeRange{state.vector_type}, state.copy_atom,
               view, state.neutral_vector, predicate)
        .getResult(0);
  }

  Value EmitSliceOperandOffset(mlir::ImplicitLocOpBuilder& builder,
                               const HloInstruction& slice,
                               Value element_offset) const {
    const Shape& result = slice.shape();
    const Shape& source = slice.operand(0)->shape();
    const int64_t rank = result.dimensions_size();
    std::vector<int64_t> result_strides(rank);
    std::vector<int64_t> source_strides(rank);
    int64_t stride = 1;
    for (int64_t dimension : result.layout().minor_to_major()) {
      result_strides[dimension] = stride;
      stride *= result.dimensions(dimension);
    }
    stride = 1;
    for (int64_t dimension : source.layout().minor_to_major()) {
      source_strides[dimension] = stride;
      stride *= source.dimensions(dimension);
    }

    Value source_offset = I64(builder, 0);
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      Value coordinate = element_offset;
      if (result_strides[dimension] != 1) {
        coordinate = mlir::arith::DivUIOp::create(
            builder, coordinate, I64(builder, result_strides[dimension]));
      }
      if (result.dimensions(dimension) != 1) {
        coordinate = mlir::arith::RemUIOp::create(
            builder, coordinate, I64(builder, result.dimensions(dimension)));
      } else {
        coordinate = I64(builder, 0);
      }
      if (slice.slice_starts()[dimension] != 0) {
        coordinate = mlir::arith::AddIOp::create(
            builder, coordinate, I64(builder, slice.slice_starts()[dimension]));
      }
      if (source_strides[dimension] != 1) {
        coordinate = mlir::arith::MulIOp::create(
            builder, coordinate, I64(builder, source_strides[dimension]));
      }
      source_offset =
          mlir::arith::AddIOp::create(builder, source_offset, coordinate);
    }
    return source_offset;
  }

  absl::StatusOr<Value> EmitReductionInputGraph(
      mlir::ImplicitLocOpBuilder& builder, const HloInstruction* instruction,
      Value element_offset, Value predicate,
      const std::vector<std::optional<InputState>>& input_states,
      const std::vector<std::optional<InputState>>& scalar_input_states,
      int64_t vector_width,
      llvm::DenseMap<const HloInstruction*, Value>& cache) const {
    if (auto it = cache.find(instruction); it != cache.end()) {
      return it->second;
    }

    Value result;
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        TF_ASSIGN_OR_RETURN(
            result, EmitInputLoad(builder, instruction->parameter_number(),
                                  element_offset, predicate, input_states));
        if (reduction_type_ == BF16) {
          result = Bf16ToF32(builder, result);
        } else if (reduction_type_ == F16) {
          auto compute_type =
              mlir::VectorType::get({vector_width}, builder.getF32Type());
          result = mlir::arith::ExtFOp::create(builder, compute_type, result);
        } else if (UsesIntegerCompute()) {
          auto compute_type = mlir::VectorType::get(
              {vector_width}, ComputeElementType(builder));
          if (reduction_type_ == PRED) {
            auto storage_type =
                mlir::VectorType::get({vector_width}, builder.getI8Type());
            Value predicate = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ne, result,
                mlir::arith::ConstantOp::create(
                    builder, storage_type,
                    mlir::DenseElementsAttr::get(
                        storage_type,
                        builder.getIntegerAttr(builder.getI8Type(), 0))));
            result =
                mlir::arith::ExtUIOp::create(builder, compute_type, predicate);
          } else if (reduction_type_ != S32 && reduction_type_ != S64) {
            result =
                mlir::arith::ExtSIOp::create(builder, compute_type, result);
          }
        }
        break;
      }
      case HloOpcode::kConstant: {
        result = FloatComputeConstant(
            builder, instruction->literal().GetFirstElement<float>());
        break;
      }
      case HloOpcode::kBroadcast: {
        const HloInstruction* operand = instruction->operand(0);
        const bool is_scalar = ShapeUtil::IsEffectiveScalar(operand->shape());
        const bool is_row = IsRowBroadcast(*instruction, instruction->shape());
        const bool is_column =
            IsColumnBroadcast(*instruction, instruction->shape());
        TF_RET_CHECK(is_scalar || is_row || is_column);
        Value source_offset = element_offset;
        const std::vector<std::optional<InputState>>* source_states =
            &input_states;
        int64_t source_vector_width = vector_width;
        if (is_scalar) {
          source_offset = I64(builder, 0);
          source_states = &scalar_input_states;
          source_vector_width = 1;
        } else if (is_row) {
          source_offset = mlir::arith::DivUIOp::create(builder, element_offset,
                                                       I64(builder, columns_));
          source_states = &scalar_input_states;
          source_vector_width = 1;
        } else {
          source_offset = mlir::arith::RemUIOp::create(builder, element_offset,
                                                       I64(builder, columns_));
        }
        // A broadcasted operand has a different address map and, for scalar
        // or row broadcasts, a different vector width from the surrounding
        // reduction DAG. Keep its memo table local so a shared producer is
        // never reused with an incompatible offset or type.
        llvm::DenseMap<const HloInstruction*, Value> broadcast_cache;
        TF_ASSIGN_OR_RETURN(
            Value operand_value,
            EmitReductionInputGraph(builder, operand, source_offset, predicate,
                                    *source_states, scalar_input_states,
                                    source_vector_width, broadcast_cache));
        if (is_scalar || is_row) {
          if (mlir::isa<mlir::VectorType>(operand_value.getType())) {
            operand_value =
                mlir::vector::ExtractOp::create(builder, operand_value, 0);
          }
          auto vector_type =
              mlir::VectorType::get({vector_width}, builder.getF32Type());
          result = mlir::vector::BroadcastOp::create(builder, vector_type,
                                                     operand_value);
        } else {
          result = operand_value;
        }
        break;
      }
      case HloOpcode::kConvert: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitReductionInputGraph(builder, instruction->operand(0),
                                    element_offset, predicate, input_states,
                                    scalar_input_states, vector_width, cache));
        const PrimitiveType source_type =
            instruction->operand(0)->shape().element_type();
        const PrimitiveType destination_type =
            instruction->shape().element_type();
        if (destination_type == F32) {
          auto vector_type =
              mlir::VectorType::get({vector_width}, builder.getF32Type());
          if (source_type == BF16) {
            result = Bf16ToF32(builder, operand);
          } else {
            TF_RET_CHECK(source_type == F16);
            result = mlir::arith::ExtFOp::create(builder, vector_type, operand);
          }
        } else {
          TF_RET_CHECK(source_type == F32 &&
                       (destination_type == F16 || destination_type == BF16));
          mlir::Type destination_element_type =
              emitters::PrimitiveTypeToMlirType(destination_type, builder);
          auto vector_type =
              mlir::VectorType::get({vector_width}, destination_element_type);
          result = destination_type == BF16 ? RoundF32ToBf16(builder, operand)
                                            : mlir::arith::TruncFOp::create(
                                                  builder, vector_type, operand)
                                                  .getResult();
        }
        break;
      }
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy: {
        TF_ASSIGN_OR_RETURN(
            result,
            EmitReductionInputGraph(builder, instruction->operand(0),
                                    element_offset, predicate, input_states,
                                    scalar_input_states, vector_width, cache));
        break;
      }
      case HloOpcode::kSlice: {
        Value source_offset =
            EmitSliceOperandOffset(builder, *instruction, element_offset);
        // Distinct slices may reach the same packed parameter with different
        // source offsets (Q and K are the common case). Do not reuse a value
        // memoized while walking another slice path.
        llvm::DenseMap<const HloInstruction*, Value> slice_cache;
        TF_ASSIGN_OR_RETURN(
            result, EmitReductionInputGraph(builder, instruction->operand(0),
                                            source_offset, predicate,
                                            input_states, scalar_input_states,
                                            vector_width, slice_cache));
        break;
      }
      case HloOpcode::kReduce: {
        TF_RET_CHECK(IsAddReducer(*instruction));
        const HloInstruction* parameter = instruction->operand(0);
        TF_RET_CHECK(parameter->opcode() == HloOpcode::kParameter);
        const int64_t split_factor = parameter->shape().dimensions(0);
        const int64_t slice_elements =
            ShapeUtil::ElementsIn(instruction->shape());
        auto vector_type =
            mlir::VectorType::get({vector_width}, builder.getF32Type());
        result = mlir::arith::ConstantOp::create(
            builder, vector_type,
            mlir::DenseElementsAttr::get(vector_type,
                                         builder.getF32FloatAttr(0.0f)));
        for (int64_t split = 0; split < split_factor; ++split) {
          Value split_offset = element_offset;
          if (split != 0) {
            split_offset = mlir::arith::AddIOp::create(
                builder, element_offset, I64(builder, split * slice_elements));
          }
          TF_ASSIGN_OR_RETURN(
              Value partial,
              EmitInputLoad(builder, parameter->parameter_number(),
                            split_offset, predicate, input_states));
          result = EmitF32Binary(builder, HloOpcode::kAdd, result, partial);
        }
        break;
      }
      case HloOpcode::kAbs:
      case HloOpcode::kExp:
      case HloOpcode::kNegate:
      case HloOpcode::kRsqrt: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitReductionInputGraph(builder, instruction->operand(0),
                                    element_offset, predicate, input_states,
                                    scalar_input_states, vector_width, cache));
        result = EmitF32Unary(builder, instruction->opcode(), operand);
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
            EmitReductionInputGraph(builder, instruction->operand(0),
                                    element_offset, predicate, input_states,
                                    scalar_input_states, vector_width, cache));
        TF_ASSIGN_OR_RETURN(
            Value rhs,
            EmitReductionInputGraph(builder, instruction->operand(1),
                                    element_offset, predicate, input_states,
                                    scalar_input_states, vector_width, cache));
        result = EmitF32Binary(builder, instruction->opcode(), lhs, rhs);
        break;
      }
      default:
        return absl::InternalError(
            "Unexpected instruction in native Fly reduction input graph.");
    }
    cache[instruction] = result;
    return result;
  }

  absl::StatusOr<Value> EmitReductionOutputGraph(
      mlir::ImplicitLocOpBuilder& builder, const HloInstruction* instruction,
      const HloInstruction* reduction, Value reduced, Value row,
      Value predicate,
      const std::vector<std::optional<InputState>>& input_states,
      llvm::DenseMap<const HloInstruction*, Value>& cache) const {
    if (instruction == reduction) {
      const PrimitiveType type = reduction->shape().element_type();
      if (type == F32 || type == F64) {
        return reduced;
      }
      if (IsSupportedIntegerReductionType(type)) {
        if (type == S32 || type == S64) {
          return reduced;
        }
        if (type == PRED) {
          return mlir::arith::CmpIOp::create(
                     builder, mlir::arith::CmpIPredicate::ne, reduced,
                     mlir::arith::ConstantIntOp::create(
                         builder, builder.getI32Type(), 0))
              .getResult();
        }
        mlir::Type output_type =
            emitters::PrimitiveTypeToMlirType(type, builder);
        return mlir::arith::TruncIOp::create(builder, output_type, reduced)
            .getResult();
      }
      TF_RET_CHECK(type == F16 || type == BF16);
      mlir::Type output_type = emitters::PrimitiveTypeToMlirType(type, builder);
      return mlir::arith::TruncFOp::create(builder, output_type, reduced)
          .getResult();
    }
    if (auto it = cache.find(instruction); it != cache.end()) {
      return it->second;
    }

    Value result;
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        TF_ASSIGN_OR_RETURN(
            Value loaded,
            EmitInputLoad(builder, instruction->parameter_number(), row,
                          predicate, input_states));
        result = mlir::vector::ExtractOp::create(builder, loaded, 0);
        break;
      }
      case HloOpcode::kConstant: {
        result = FloatComputeConstant(
            builder, instruction->literal().GetFirstElement<float>());
        break;
      }
      case HloOpcode::kConvert: {
        TF_ASSIGN_OR_RETURN(Value operand,
                            EmitReductionOutputGraph(
                                builder, instruction->operand(0), reduction,
                                reduced, row, predicate, input_states, cache));
        mlir::Type output_type = emitters::PrimitiveTypeToMlirType(
            instruction->shape().element_type(), builder);
        result =
            instruction->shape().element_type() == F32
                ? mlir::arith::ExtFOp::create(builder, output_type, operand)
                      .getResult()
                : mlir::arith::TruncFOp::create(builder, output_type, operand)
                      .getResult();
        break;
      }
      case HloOpcode::kBroadcast: {
        Value operand_row =
            ShapeUtil::IsEffectiveScalar(instruction->operand(0)->shape())
                ? I64(builder, 0)
                : row;
        TF_ASSIGN_OR_RETURN(
            result, EmitReductionOutputGraph(builder, instruction->operand(0),
                                             reduction, reduced, operand_row,
                                             predicate, input_states, cache));
        break;
      }
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy: {
        TF_ASSIGN_OR_RETURN(
            result, EmitReductionOutputGraph(builder, instruction->operand(0),
                                             reduction, reduced, row, predicate,
                                             input_states, cache));
        break;
      }
      case HloOpcode::kAbs:
      case HloOpcode::kExp:
      case HloOpcode::kNegate:
      case HloOpcode::kRsqrt: {
        TF_ASSIGN_OR_RETURN(Value operand,
                            EmitReductionOutputGraph(
                                builder, instruction->operand(0), reduction,
                                reduced, row, predicate, input_states, cache));
        result = EmitF32Unary(builder, instruction->opcode(), operand);
        break;
      }
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum: {
        TF_ASSIGN_OR_RETURN(
            Value lhs, EmitReductionOutputGraph(
                           builder, instruction->operand(0), reduction, reduced,
                           row, predicate, input_states, cache));
        TF_ASSIGN_OR_RETURN(
            Value rhs, EmitReductionOutputGraph(
                           builder, instruction->operand(1), reduction, reduced,
                           row, predicate, input_states, cache));
        result = EmitF32Binary(builder, instruction->opcode(), lhs, rhs);
        break;
      }
      default:
        return absl::InternalError(
            "Unexpected instruction in native Fly reduction output graph.");
    }
    cache[instruction] = result;
    return result;
  }

  absl::StatusOr<Value> EmitRowwiseOutputGraph(
      mlir::ImplicitLocOpBuilder& builder, const HloInstruction* instruction,
      const HloInstruction* reduction, const Shape& input_shape,
      const Shape& row_shape, Value reduced, Value element_offset,
      Value predicate,
      const std::vector<std::optional<InputState>>& input_states,
      const std::vector<std::optional<InputState>>& row_input_states,
      int64_t vector_width,
      llvm::DenseMap<const HloInstruction*, Value>& vector_cache,
      llvm::DenseMap<const HloInstruction*, Value>& row_cache) const {
    const bool is_vector =
        HasSamePhysicalDimensions(instruction->shape(), input_shape);
    const bool is_column =
        HasCompatibleColumnShape(input_shape, instruction->shape());
    llvm::DenseMap<const HloInstruction*, Value>& cache =
        (is_vector || is_column) ? vector_cache : row_cache;
    if (auto it = cache.find(instruction); it != cache.end()) {
      return it->second;
    }
    if (instruction == reduction) {
      row_cache[instruction] = reduced;
      return reduced;
    }

    Value result;
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        if (is_vector) {
          TF_ASSIGN_OR_RETURN(
              result, EmitInputLoad(builder, instruction->parameter_number(),
                                    element_offset, predicate, input_states));
        } else if (is_column) {
          Value column_offset = mlir::arith::RemUIOp::create(
              builder, element_offset, I64(builder, columns_));
          TF_ASSIGN_OR_RETURN(
              result, EmitInputLoad(builder, instruction->parameter_number(),
                                    column_offset, predicate, input_states));
        } else {
          Value row_offset = mlir::arith::DivUIOp::create(
              builder, element_offset, I64(builder, columns_));
          TF_ASSIGN_OR_RETURN(
              Value loaded,
              EmitInputLoad(builder, instruction->parameter_number(),
                            row_offset, predicate, row_input_states));
          result = mlir::vector::ExtractOp::create(builder, loaded, 0);
        }
        break;
      }
      case HloOpcode::kConstant: {
        result = FloatComputeConstant(
            builder, instruction->literal().GetFirstElement<float>());
        break;
      }
      case HloOpcode::kBroadcast: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitRowwiseOutputGraph(
                builder, instruction->operand(0), reduction, input_shape,
                row_shape, reduced, element_offset, predicate, input_states,
                row_input_states, vector_width, vector_cache, row_cache));
        if (is_vector &&
            !HasSamePhysicalDimensions(instruction->operand(0)->shape(),
                                       input_shape) &&
            !HasCompatibleColumnShape(input_shape,
                                      instruction->operand(0)->shape())) {
          auto vector_type =
              mlir::VectorType::get({vector_width}, builder.getF32Type());
          result =
              mlir::vector::BroadcastOp::create(builder, vector_type, operand);
        } else {
          result = operand;
        }
        break;
      }
      case HloOpcode::kConvert: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitRowwiseOutputGraph(
                builder, instruction->operand(0), reduction, input_shape,
                row_shape, reduced, element_offset, predicate, input_states,
                row_input_states, vector_width, vector_cache, row_cache));
        const PrimitiveType source_type =
            instruction->operand(0)->shape().element_type();
        const PrimitiveType destination_type =
            instruction->shape().element_type();
        const bool is_lane_vector = is_vector || is_column;
        if (destination_type == F32) {
          mlir::Type destination_mlir_type = builder.getF32Type();
          if (is_lane_vector) {
            destination_mlir_type =
                mlir::VectorType::get({vector_width}, builder.getF32Type());
          }
          if (source_type == BF16) {
            result = Bf16ToF32(builder, operand);
          } else {
            TF_RET_CHECK(source_type == F16);
            result = mlir::arith::ExtFOp::create(builder, destination_mlir_type,
                                                 operand);
          }
        } else {
          TF_RET_CHECK(source_type == F32 &&
                       (destination_type == F16 || destination_type == BF16));
          mlir::Type destination_element_type =
              emitters::PrimitiveTypeToMlirType(destination_type, builder);
          mlir::Type destination_mlir_type = destination_element_type;
          if (is_lane_vector) {
            destination_mlir_type =
                mlir::VectorType::get({vector_width}, destination_element_type);
          }
          if (destination_type == BF16 && is_lane_vector) {
            result = RoundF32ToBf16(builder, operand);
          } else {
            result = mlir::arith::TruncFOp::create(
                builder, destination_mlir_type, operand);
          }
        }
        break;
      }
      case HloOpcode::kCopy: {
        TF_ASSIGN_OR_RETURN(
            result,
            EmitRowwiseOutputGraph(
                builder, instruction->operand(0), reduction, input_shape,
                row_shape, reduced, element_offset, predicate, input_states,
                row_input_states, vector_width, vector_cache, row_cache));
        break;
      }
      case HloOpcode::kBitcast: {
        // A rowwise epilogue can reuse an auxiliary leading-dimension
        // reduction (for example a split-K projection) through a physical
        // bitcast. The full-row path finds this value in the reduction-input
        // cache; partitioned output deliberately reloads its slice, so emit
        // that same supported input subgraph on demand.
        if (is_vector &&
            !DependsOnInstruction(instruction->operand(0), reduction)) {
          TF_ASSIGN_OR_RETURN(
              result,
              EmitReductionInputGraph(builder, instruction, element_offset,
                                      predicate, input_states, row_input_states,
                                      vector_width, vector_cache));
        } else {
          TF_ASSIGN_OR_RETURN(
              result,
              EmitRowwiseOutputGraph(
                  builder, instruction->operand(0), reduction, input_shape,
                  row_shape, reduced, element_offset, predicate, input_states,
                  row_input_states, vector_width, vector_cache, row_cache));
        }
        break;
      }
      case HloOpcode::kAbs:
      case HloOpcode::kExp:
      case HloOpcode::kNegate:
      case HloOpcode::kRsqrt: {
        TF_ASSIGN_OR_RETURN(
            Value operand,
            EmitRowwiseOutputGraph(
                builder, instruction->operand(0), reduction, input_shape,
                row_shape, reduced, element_offset, predicate, input_states,
                row_input_states, vector_width, vector_cache, row_cache));
        result = EmitF32Unary(builder, instruction->opcode(), operand);
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
            EmitRowwiseOutputGraph(
                builder, instruction->operand(0), reduction, input_shape,
                row_shape, reduced, element_offset, predicate, input_states,
                row_input_states, vector_width, vector_cache, row_cache));
        TF_ASSIGN_OR_RETURN(
            Value rhs,
            EmitRowwiseOutputGraph(
                builder, instruction->operand(1), reduction, input_shape,
                row_shape, reduced, element_offset, predicate, input_states,
                row_input_states, vector_width, vector_cache, row_cache));
        result = EmitF32Binary(builder, instruction->opcode(), lhs, rhs);
        break;
      }
      default:
        return absl::InternalError(
            "Unexpected instruction in native Fly rowwise output graph.");
    }
    cache[instruction] = result;
    return result;
  }

  struct OutputState {
    mlir::VectorType vector_type;
    mlir::Type pointer_type;
    mlir::Type memref_type;
    mlir::Type offset_type;
    Value pointer;
    Value layout;
    Value copy_atom;
  };

  OutputState CreateOutputState(mlir::ImplicitLocOpBuilder& builder,
                                mlir::gpu::GPUFuncOp kernel,
                                int64_t vector_width, int64_t elements) const {
    mlir::MLIRContext* context = builder.getContext();
    mlir::Type element_type = StorageElementType(output_type_, builder);
    auto vector_type = mlir::VectorType::get({vector_width}, element_type);
    auto shape_attr =
        mlir::fly::IntTupleAttr::getLeafStatic(context, vector_width);
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

    const int64_t output_bits = ElementBits(output_type_);
    const int64_t copy_bits = vector_width * output_bits;
    auto copy_atom_type = mlir::fly::CopyAtomType::get(
        mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context, copy_bits,
                                                        /*cacheModifier=*/0),
        output_bits);
    Value copy_atom =
        mlir::fly::MakeCopyAtomOp::create(builder, copy_atom_type, output_bits);
    Value pointer = MakeDescriptorPointer(builder, kernel.getArguments().back(),
                                          output_type_, elements);
    auto pointer_type = mlir::cast<mlir::fly::PointerType>(pointer.getType());
    auto memref_type = mlir::fly::MemRefType::get(
        element_type, pointer_type.getAddressSpace(), layout_type.getAttr());
    auto offset_attr = mlir::fly::IntTupleAttr::getLeafDynamic(
        context, /*width=*/32, /*divisibility=*/vector_width);
    auto offset_type = mlir::fly::IntTupleType::get(offset_attr);
    return OutputState{vector_type, pointer_type, memref_type, offset_type,
                       pointer,     layout,       copy_atom};
  }

  void EmitOutputStore(mlir::ImplicitLocOpBuilder& builder,
                       const OutputState& state, Value element_offset,
                       Value predicate, Value output) const {
    Value offset_i32 = mlir::arith::TruncIOp::create(
        builder, builder.getI32Type(), element_offset);
    Value offset_tuple = mlir::fly::MakeIntTupleOp::create(
        builder, state.offset_type, mlir::ValueRange{offset_i32});
    Value advanced = mlir::fly::AddOffsetOp::create(
        builder, state.pointer_type, state.pointer, offset_tuple);
    Value view = mlir::fly::MakeViewOp::create(builder, state.memref_type,
                                               advanced, state.layout);
    mlir::fly::CopyAtomCallSSA::create(
        builder, mlir::TypeRange{}, state.copy_atom, output, view, predicate);
  }

  absl::Status EmitKernel(mlir::gpu::GPUFuncOp kernel,
                          const HloFusionInstruction& fusion) const {
    TF_RET_CHECK(kernel.getNumArguments() == fusion.operand_count() + 1);
    std::optional<RowReductionSpec> spec =
        MatchRowReduction(*fusion.fused_expression_root());
    TF_RET_CHECK(spec.has_value());

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
        mlir::arith::RemUIOp::create(builder, thread, I64(builder, 64));
    Value wave =
        mlir::arith::DivUIOp::create(builder, thread, I64(builder, 64));
    Value work = mlir::arith::AddIOp::create(
        builder,
        mlir::arith::MulIOp::create(builder, block, I64(builder, num_warps_)),
        wave);
    Value row = work;
    Value output_partition = I64(builder, 0);
    if (output_partitions_ > 1) {
      row = mlir::arith::DivUIOp::create(builder, work,
                                         I64(builder, output_partitions_));
      output_partition = mlir::arith::RemUIOp::create(
          builder, work, I64(builder, output_partitions_));
    }
    Value row_valid = mlir::arith::CmpIOp::create(
        builder, mlir::arith::CmpIPredicate::ult, row, I64(builder, rows_));

    mlir::Type compute_element_type = ComputeElementType(builder);
    auto compute_vector_type =
        mlir::VectorType::get({vector_width_}, compute_element_type);
    std::vector<std::optional<InputState>> input_states(fusion.operand_count());
    std::vector<std::optional<InputState>> output_input_states(
        fusion.operand_count());
    const int64_t tail_elements = columns_ % vector_width_;
    std::vector<std::optional<InputState>> tail_input_states(
        tail_elements == 0 ? 0 : fusion.operand_count());
    for (auto [parameter_number, input_type] : spec->input_parameters) {
      TF_RET_CHECK(parameter_number >= 0 &&
                   parameter_number < fusion.operand_count());
      const int64_t input_elements =
          ShapeUtil::ElementsIn(fusion.operand(parameter_number)->shape());
      const int64_t input_vector_width =
          init_parameter_number_ == parameter_number ? 1 : vector_width_;
      input_states[parameter_number] =
          CreateInputState(builder, kernel, parameter_number, input_type,
                           input_elements, input_vector_width);
      output_input_states[parameter_number] =
          CreateInputState(builder, kernel, parameter_number, input_type,
                           input_elements, /*vector_width=*/1);
      if (tail_elements != 0) {
        tail_input_states[parameter_number] =
            CreateInputState(builder, kernel, parameter_number, input_type,
                             input_elements, /*vector_width=*/1);
      }
    }

    Value neutral = UsesIntegerCompute()
                        ? IntegerComputeConstant(builder, IntegerNeutralValue())
                        : FloatComputeConstant(builder, NeutralValue());
    mlir::Attribute neutral_attribute =
        UsesIntegerCompute() ? mlir::Attribute(builder.getIntegerAttr(
                                   compute_element_type, IntegerNeutralValue()))
                             : mlir::Attribute(builder.getFloatAttr(
                                   compute_element_type, NeutralValue()));
    Value neutral_vector = mlir::arith::ConstantOp::create(
        builder, compute_vector_type,
        mlir::DenseElementsAttr::get(compute_vector_type, neutral_attribute));
    Value local_vector = neutral_vector;
    Value row_base =
        mlir::arith::MulIOp::create(builder, row, I64(builder, columns_));
    Value lane_base =
        mlir::arith::MulIOp::create(builder, lane, I64(builder, vector_width_));
    Value base = mlir::arith::AddIOp::create(builder, row_base, lane_base);
    const int64_t elements_per_wave = 64 * vector_width_;
    const int64_t full_columns = columns_ - tail_elements;
    const int64_t vectors_per_lane =
        (full_columns + elements_per_wave - 1) / elements_per_wave;
    std::vector<llvm::DenseMap<const HloInstruction*, Value>>
        vector_input_caches(vectors_per_lane);
    for (int64_t tile = 0; tile < vectors_per_lane; ++tile) {
      Value offset = base;
      if (tile != 0) {
        offset = mlir::arith::AddIOp::create(
            builder, base, I64(builder, tile * elements_per_wave));
      }
      Value column_offset = lane_base;
      if (tile != 0) {
        column_offset = mlir::arith::AddIOp::create(
            builder, lane_base, I64(builder, tile * elements_per_wave));
      }
      Value vector_valid = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, column_offset,
          I64(builder, full_columns));
      Value load_valid =
          mlir::arith::AndIOp::create(builder, row_valid, vector_valid);
      llvm::DenseMap<const HloInstruction*, Value>& input_cache =
          vector_input_caches[tile];
      TF_ASSIGN_OR_RETURN(
          Value values,
          EmitReductionInputGraph(builder, spec->reduction->operand(0), offset,
                                  load_valid, input_states, output_input_states,
                                  vector_width_, input_cache));
      values = mlir::arith::SelectOp::create(builder, load_valid, values,
                                             neutral_vector);
      // Keep one independent accumulator chain per vector lane and perform a
      // single horizontal reduction after all global loads. This gives LLVM
      // enough instruction-level parallelism to overlap the long F32 add
      // chains; reducing every loaded vector immediately serializes the row.
      local_vector = Combine(builder, local_vector, values);
    }

    Value local = mlir::vector::ReductionOp::create(
        builder, CombiningKind(), local_vector, neutral,
        mlir::arith::FastMathFlags::reassoc);
    llvm::DenseMap<const HloInstruction*, Value> tail_input_cache;
    if (tail_elements != 0) {
      Value tail_lane_valid =
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::ult,
                                      lane, I64(builder, tail_elements));
      Value tail_valid =
          mlir::arith::AndIOp::create(builder, row_valid, tail_lane_valid);
      Value tail_offset = mlir::arith::AddIOp::create(
          builder, row_base,
          mlir::arith::AddIOp::create(builder, I64(builder, full_columns),
                                      lane));
      TF_ASSIGN_OR_RETURN(
          Value tail_values,
          EmitReductionInputGraph(builder, spec->reduction->operand(0),
                                  tail_offset, tail_valid, tail_input_states,
                                  output_input_states, /*vector_width=*/1,
                                  tail_input_cache));
      auto scalar_vector_type =
          mlir::VectorType::get({1}, compute_element_type);
      Value scalar_neutral = mlir::arith::ConstantOp::create(
          builder, scalar_vector_type,
          mlir::DenseElementsAttr::get(scalar_vector_type, neutral_attribute));
      tail_values = mlir::arith::SelectOp::create(builder, tail_valid,
                                                  tail_values, scalar_neutral);
      Value tail_local = mlir::vector::ReductionOp::create(
          builder, CombiningKind(), tail_values, neutral,
          mlir::arith::FastMathFlags::reassoc);
      local = Combine(builder, local, tail_local);
    }

    if (ElementBits(reduction_type_) == 64) {
      // CDNA3 DPP operates on one 32-bit VGPR word. Let MLIR split 64-bit
      // values across the required pair of shuffles, using a butterfly so the
      // complete result is available in every lane.
      for (int32_t distance : {32, 16, 8, 4, 2, 1}) {
        Value shuffled = mlir::gpu::ShuffleOp::create(
                             builder, local, distance,
                             /*width=*/64, mlir::gpu::ShuffleMode::XOR)
                             .getShuffleResult();
        local = Combine(builder, local, shuffled);
      }
    } else {
      // Reduce each 16-lane hardware row with DPP, then combine the four rows
      // with DPP row broadcasts. This is the all-register Wave64 topology used
      // by the fastest ROCm reduction kernels. In particular, it avoids the
      // ds_bpermute(distance=32) and ds_swizzle(distance=16) generated by a
      // generic gpu.shuffle tree.
      Value poison = mlir::ub::PoisonOp::create(builder, local.getType());
      for (int32_t distance : {8, 4, 2, 1}) {
        Value shuffled = mlir::amdgpu::DPPOp::create(
            builder, local.getType(), /*old=*/poison, /*src=*/local,
            mlir::amdgpu::DPPPerm::row_shr, builder.getI32IntegerAttr(distance),
            /*row_mask=*/0xF,
            /*bank_mask=*/0xF, /*bound_ctrl=*/true);
        local = Combine(builder, local, shuffled);
      }
      Value adjacent_row = mlir::amdgpu::DPPOp::create(
          builder, local.getType(), /*old=*/local, /*src=*/local,
          mlir::amdgpu::DPPPerm::row_bcast_15,
          /*permArgument=*/nullptr, /*row_mask=*/0xA, /*bank_mask=*/0xF,
          /*bound_ctrl=*/true);
      local = Combine(builder, local, adjacent_row);
      Value lower_half = mlir::amdgpu::DPPOp::create(
          builder, local.getType(), /*old=*/poison, /*src=*/local,
          mlir::amdgpu::DPPPerm::row_bcast_31,
          /*permArgument=*/nullptr, /*row_mask=*/0xF, /*bank_mask=*/0xF,
          /*bound_ctrl=*/true);
      local = Combine(builder, local, lower_half);
      // The full result is in lane 63. Broadcast it once so both scalar and
      // rowwise epilogues can consume the same uniform value.
      local = mlir::gpu::ShuffleOp::create(builder, local, /*offset=*/63,
                                           /*width=*/64,
                                           mlir::gpu::ShuffleMode::IDX)
                  .getShuffleResult();
    }
    if (init_parameter_number_.has_value()) {
      llvm::DenseMap<const HloInstruction*, Value> init_cache;
      TF_ASSIGN_OR_RETURN(
          Value init_vector,
          EmitReductionInputGraph(builder, spec->reduction->operand(1),
                                  I64(builder, 0), row_valid,
                                  output_input_states, output_input_states,
                                  /*vector_width=*/1, init_cache));
      Value init_scalar =
          mlir::vector::ExtractOp::create(builder, init_vector, 0);
      local = Combine(builder, local, init_scalar);
    } else if (!HasNeutralInit()) {
      local = Combine(builder, local,
                      UsesIntegerCompute()
                          ? IntegerComputeConstant(builder, integer_init_value_)
                          : ComputeConstant(builder, init_value_));
    }
    if (!has_rowwise_output_) {
      llvm::DenseMap<const HloInstruction*, Value> output_cache;
      TF_ASSIGN_OR_RETURN(
          local, EmitReductionOutputGraph(builder, spec->root, spec->reduction,
                                          local, row, row_valid,
                                          output_input_states, output_cache));
      OutputState output_state =
          CreateOutputState(builder, kernel, /*vector_width=*/1, rows_);
      Value lane_zero = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, lane, I64(builder, 0));
      Value store = mlir::arith::AndIOp::create(builder, row_valid, lane_zero);
      mlir::VectorType logical_output_type =
          output_type_ == PRED ? mlir::VectorType::get({1}, builder.getI1Type())
                               : output_state.vector_type;
      Value output_value = mlir::vector::FromElementsOp::create(
          builder, logical_output_type, mlir::ValueRange{local});
      if (output_type_ == PRED) {
        output_value = mlir::arith::ExtUIOp::create(
            builder, output_state.vector_type, output_value);
      }
      EmitOutputStore(builder, output_state, row, store, output_value);
    } else {
      const Shape& input_shape = spec->reduction->operand(0)->shape();
      const Shape& row_shape = spec->reduction->shape();
      llvm::DenseMap<const HloInstruction*, Value> row_output_cache;
      OutputState output_state =
          CreateOutputState(builder, kernel, vector_width_, rows_ * columns_);
      if (output_partitions_ == 1) {
        for (int64_t tile = 0; tile < vectors_per_lane; ++tile) {
          Value offset = base;
          Value column_offset = lane_base;
          if (tile != 0) {
            offset = mlir::arith::AddIOp::create(
                builder, base, I64(builder, tile * elements_per_wave));
            column_offset = mlir::arith::AddIOp::create(
                builder, lane_base, I64(builder, tile * elements_per_wave));
          }
          Value vector_valid = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, column_offset,
              I64(builder, full_columns));
          Value store =
              mlir::arith::AndIOp::create(builder, row_valid, vector_valid);
          llvm::DenseMap<const HloInstruction*, Value>& vector_output_cache =
              vector_input_caches[tile];
          TF_ASSIGN_OR_RETURN(
              Value output,
              EmitRowwiseOutputGraph(
                  builder, spec->root, spec->reduction, input_shape, row_shape,
                  local, offset, store, input_states, output_input_states,
                  vector_width_, vector_output_cache, row_output_cache));
          EmitOutputStore(builder, output_state, offset, store, output);
        }
      } else {
        // For short batches on a large GPU, one wave per row does not expose
        // enough parallelism. Let each partition recompute the cheap row
        // reduction, then normalize and store only its contiguous output
        // slice. Reloading that slice keeps the live register set bounded and
        // mirrors the geometry selected by XLA's Triton block emitter.
        const int64_t vectors_per_partition =
            vectors_per_lane / output_partitions_;
        Value partition_vector = mlir::arith::MulIOp::create(
            builder, output_partition, I64(builder, vectors_per_partition));
        for (int64_t tile = 0; tile < vectors_per_partition; ++tile) {
          Value vector_index = partition_vector;
          if (tile != 0) {
            vector_index = mlir::arith::AddIOp::create(
                builder, partition_vector, I64(builder, tile));
          }
          Value relative = mlir::arith::MulIOp::create(
              builder, vector_index, I64(builder, elements_per_wave));
          Value offset = mlir::arith::AddIOp::create(builder, base, relative);
          llvm::DenseMap<const HloInstruction*, Value> vector_output_cache;
          TF_ASSIGN_OR_RETURN(
              Value output,
              EmitRowwiseOutputGraph(
                  builder, spec->root, spec->reduction, input_shape, row_shape,
                  local, offset, row_valid, input_states, output_input_states,
                  vector_width_, vector_output_cache, row_output_cache));
          EmitOutputStore(builder, output_state, offset, row_valid, output);
        }
      }
      if (tail_elements != 0) {
        Value tail_lane_valid = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, lane,
            I64(builder, tail_elements));
        Value store =
            mlir::arith::AndIOp::create(builder, row_valid, tail_lane_valid);
        Value tail_offset = mlir::arith::AddIOp::create(
            builder, row_base,
            mlir::arith::AddIOp::create(builder, I64(builder, full_columns),
                                        lane));
        TF_ASSIGN_OR_RETURN(
            Value output,
            EmitRowwiseOutputGraph(builder, spec->root, spec->reduction,
                                   input_shape, row_shape, local, tail_offset,
                                   store, tail_input_states,
                                   output_input_states, /*vector_width=*/1,
                                   tail_input_cache, row_output_cache));
        OutputState tail_output_state = CreateOutputState(
            builder, kernel, /*vector_width=*/1, rows_ * columns_);
        EmitOutputStore(builder, tail_output_state, tail_offset, store, output);
      }
    }

    mlir::gpu::ReturnOp::create(builder);
    return absl::OkStatus();
  }

  int64_t rows_ = 0;
  int64_t columns_ = 0;
  PrimitiveType output_type_ = PRIMITIVE_TYPE_INVALID;
  PrimitiveType reduction_type_ = PRIMITIVE_TYPE_INVALID;
  bool has_rowwise_output_ = false;
  int64_t max_vector_element_bits_ = 0;
  HloOpcode reducer_opcode_ = HloOpcode::kAdd;
  std::optional<int64_t> init_parameter_number_;
  double init_value_ = 0.0;
  int64_t integer_init_value_ = 0;
  int64_t num_warps_ = 0;
  int64_t output_partitions_ = 1;
  int64_t vector_size_bits_ = 0;
  int64_t vector_width_ = 0;
  LaunchDimensions launch_dimensions_;
};

}  // namespace

bool IsFlyXTileRowReductionFusion(const HloFusionAnalysis& analysis) {
  return analysis.fusion_root_count() == 1 &&
         MatchRowReduction(analysis.fusion_root(0).instruction()).has_value();
}

bool IsFlyXTileRowReductionConfigSupported(const HloFusionAnalysis& analysis,
                                           int64_t vector_size_bits,
                                           int64_t output_partitions) {
  if (analysis.fusion_root_count() != 1) {
    return false;
  }
  std::optional<RowReductionSpec> spec =
      MatchRowReduction(analysis.fusion_root(0).instruction());
  if (!spec.has_value() || output_partitions <= 0 || output_partitions > 16 ||
      (vector_size_bits != 16 && vector_size_bits != 32 &&
       vector_size_bits != 64 && vector_size_bits != 128) ||
      vector_size_bits < spec->max_vector_element_bits ||
      vector_size_bits > 128 ||
      vector_size_bits % spec->max_vector_element_bits != 0) {
    return false;
  }
  if (output_partitions == 1) {
    return true;
  }
  const int64_t vector_width = vector_size_bits / spec->max_vector_element_bits;
  if (!spec->has_rowwise_output ||
      spec->columns % (64 * vector_width * output_partitions) != 0) {
    return false;
  }
  return true;
}

std::unique_ptr<MlirKernelEmitter> CreateFlyXTileRowReductionEmitter(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FlyXTileRowReductionEmitter>(analysis);
}

}  // namespace xla::gpu::flydsl
