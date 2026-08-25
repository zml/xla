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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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
#include "mlir/Dialect/Math/IR/Math.h"
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
#include "xla/shape_util.h"
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

bool IsFnuzFp8(PrimitiveType type) {
  return type == F8E4M3FNUZ || type == F8E5M2FNUZ;
}

bool IsHalfPrecisionFloat(PrimitiveType type) {
  return type == F16 || type == BF16;
}

bool IsSupportedContractionConversion(PrimitiveType source,
                                      PrimitiveType destination) {
  return (source == F32 && destination == BF16) ||
         (source == S4 && (destination == S8 || destination == BF16)) ||
         (source == S8 && destination == BF16);
}

mlir::Type ContractionElementType(mlir::ImplicitLocOpBuilder& builder,
                                  PrimitiveType type) {
  switch (type) {
    case F32:
      return builder.getF32Type();
    case F16:
      return builder.getF16Type();
    case BF16:
      return builder.getBF16Type();
    case F8E4M3FNUZ:
      return builder.getType<mlir::Float8E4M3FNUZType>();
    case F8E5M2FNUZ:
      return builder.getType<mlir::Float8E5M2FNUZType>();
    default:
      CHECK(false) << "Unsupported Fly contraction element type: " << type;
      return {};
  }
}

std::string FlyTypeName(PrimitiveType type) {
  switch (type) {
    case F32:
      return "f32";
    case F16:
      return "f16";
    case BF16:
      return "bf16";
    case F8E4M3FNUZ:
      return "f8E4M3FNUZ";
    case F8E5M2FNUZ:
      return "f8E5M2FNUZ";
    default:
      CHECK(false) << "Unsupported Fly contraction element type: " << type;
      return {};
  }
}

int64_t ContractionElementBytes(PrimitiveType type) {
  CHECK(type == F32 || IsHalfPrecisionFloat(type) || IsFnuzFp8(type));
  return type == F32 ? 4 : (IsHalfPrecisionFloat(type) ? 2 : 1);
}

Value SwizzleXor16(mlir::ImplicitLocOpBuilder& builder, Value row, Value column,
                   int64_t stage_k, int64_t element_bytes = 2) {
  // FlyDSL's swizzle_xor16 operates in bytes:
  //   col_bytes ^ ((row % (stage_bytes / 16)) * 16).
  // Express the transform in logical elements for either BF16 or FP8.
  CHECK(element_bytes == 1 || element_bytes == 2 || element_bytes == 4);
  const int64_t elements_per_16_bytes = 16 / element_bytes;
  Value swizzle =
      Mul(builder, Rem(builder, row, stage_k / elements_per_16_bytes),
          IndexConstant(builder, elements_per_16_bytes));
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

Value SwizzleTritonXf32(mlir::ImplicitLocOpBuilder& builder, Value row,
                        Value column) {
  // Triton's gfx942 XF32 tile uses a two-element vector and rotates each
  // successive row by one vector within a 16-row phase:
  //   physical_k = logical_k ^ (2 * (row % 16)).
  // Keeping this in element units makes the corresponding vector<2xf32>
  // reads combine into ds_read2st64_b64 without LDS bank conflicts.
  Value phase = Rem(builder, row, 16);
  return mlir::arith::XOrIOp::create(
      builder, column, Mul(builder, phase, IndexConstant(builder, 2)));
}

Value SwizzleTritonAmdRotatingRhs(mlir::ImplicitLocOpBuilder& builder,
                                   Value row, Value column) {
  // Triton's AMD rotating shared layout extends the ordinary
  // vec=4/perPhase=2/maxPhase=8 swizzle with a high-row block rotation:
  //   combined_phase = ((row / 2) % 8) ^ ((row / 16) % 8)
  //   physical_k = logical_k ^ (4 * combined_phase)
  // This is the gfx942 layout selected for the in-thread-transposed B tile of
  // the 128x256x32 MFMA32 kernel.
  Value phase = Rem(builder, Div(builder, row, 2), 8);
  Value block = Rem(builder, Div(builder, row, 16), 8);
  Value combined_phase =
      mlir::arith::XOrIOp::create(builder, phase, block);
  return mlir::arith::XOrIOp::create(
      builder, column,
      Mul(builder, combined_phase, IndexConstant(builder, 4)));
}

Value SwizzleTritonMfma32(mlir::ImplicitLocOpBuilder& builder, Value row,
                          Value column, int64_t stage_k = 64) {
  // Triton's gfx942 128x64x128 MFMA32 GEMM uses a four-BF16 vector and one
  // row per swizzle phase. `column` is a BF16 element offset, so each row
  // advances the xor phase by four elements and the phase repeats every 16
  // rows.
  // K32 has eight four-BF16 swizzle phases; K64 and the K128 rolling-refill
  // path use the original sixteen phases. Keeping the phase inside the
  // physical K extent is required when the square pipeline uses a K32 LDS
  // stage.
  const int64_t phase_count = std::min<int64_t>(stage_k / 4, 16);
  Value phase = Rem(builder, row, phase_count);
  return mlir::arith::XOrIOp::create(
      builder, column, Mul(builder, phase, IndexConstant(builder, 4)));
}

Value SwizzleTritonMfma32TransposedRhs(mlir::ImplicitLocOpBuilder& builder,
                                       Value row, Value column,
                                       int64_t stage_k = 64) {
  // Triton's transposed B tile folds the high 16-row group into the ordinary
  // MFMA32 phase. This extra XOR distributes the eight transpose stores from
  // each source vector across LDS banks instead of aliasing every eighth row.
  const int64_t phase_count = stage_k / 4;
  Value phase = mlir::arith::XOrIOp::create(
      builder, Rem(builder, row, phase_count),
      Rem(builder, Div(builder, row, phase_count), phase_count));
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

const HloInstruction* FindContraction(const HloInstruction& root) {
  if (root.opcode() == HloOpcode::kDot ||
      root.opcode() == HloOpcode::kScaledDot) {
    return &root;
  }
  for (const HloInstruction* operand : root.operands()) {
    if (const HloInstruction* contraction = FindContraction(*operand)) {
      return contraction;
    }
  }
  return nullptr;
}

std::vector<int64_t> PhysicalStrides(const Shape& shape);

bool IsSupportedBatchInnerTranspose(const HloInstruction& transpose) {
  return transpose.opcode() == HloOpcode::kTranspose &&
         transpose.operand_count() == 1 &&
         transpose.shape().dimensions_size() == 3 &&
         transpose.operand(0)->shape().dimensions_size() == 3 &&
         transpose.shape().element_type() ==
             transpose.operand(0)->shape().element_type() &&
         transpose.dimensions() == std::vector<int64_t>({0, 2, 1});
}

struct ContractionInput {
  std::vector<int64_t> parameter_numbers;
  PrimitiveType parameter_type;
  bool requires_linear_addressing;
  int64_t physical_offset;
  std::vector<int64_t> physical_strides;
  int64_t concat_dimension;
  int64_t concat_fragment_size;
  std::vector<int64_t> dynamic_offset_parameter_numbers;
  std::vector<int64_t> dynamic_offset_constants;
  std::vector<int64_t> dynamic_offset_limits;
};

ContractionInput FindContractionInput(const HloInstruction& input) {
  const PrimitiveType contraction_type = input.shape().element_type();
  CHECK(IsHalfPrecisionFloat(contraction_type) ||
        IsFnuzFp8(contraction_type) || contraction_type == F32);
  if (input.opcode() == HloOpcode::kConcatenate) {
    CHECK_GE(input.operand_count(), 2);
    std::vector<int64_t> parameter_numbers;
    std::optional<ContractionInput> first_fragment;
    for (const HloInstruction* operand : input.operands()) {
      ContractionInput fragment = FindContractionInput(*operand);
      CHECK_EQ(fragment.parameter_numbers.size(), 1);
      CHECK_EQ(fragment.concat_dimension, -1);
      CHECK(!fragment.requires_linear_addressing);
      if (!first_fragment.has_value()) {
        first_fragment = fragment;
      } else {
        CHECK_EQ(fragment.parameter_type, first_fragment->parameter_type);
        CHECK_EQ(fragment.physical_offset, first_fragment->physical_offset);
        CHECK_EQ(fragment.physical_strides, first_fragment->physical_strides);
      }
      parameter_numbers.push_back(fragment.parameter_numbers.front());
    }
    CHECK(first_fragment.has_value());
    const int64_t concat_dimension = input.concatenate_dimension();
    return {std::move(parameter_numbers),
            first_fragment->parameter_type,
            /*requires_linear_addressing=*/false,
            first_fragment->physical_offset,
            std::move(first_fragment->physical_strides),
            concat_dimension,
            input.operand(0)->shape().dimensions(concat_dimension),
            /*dynamic_offset_parameter_numbers=*/{},
            /*dynamic_offset_constants=*/{},
            /*dynamic_offset_limits=*/{}};
  }
  const HloInstruction* value = &input;
  bool has_conversion = false;
  bool has_input_view = false;
  bool requires_linear_addressing = false;
  bool physical_mapping_fixed = false;
  bool has_dynamic_slice = false;
  int64_t physical_offset = 0;
  std::vector<int64_t> logical_offsets(input.shape().dimensions_size(), 0);
  std::vector<int64_t> physical_strides;
  std::vector<int64_t> dynamic_offset_parameter_numbers(
      input.shape().dimensions_size(), -1);
  std::vector<int64_t> dynamic_offset_constants(input.shape().dimensions_size(),
                                                0);
  std::vector<int64_t> dynamic_offset_limits(input.shape().dimensions_size(),
                                             0);
  while (value->opcode() == HloOpcode::kBitcast ||
         value->opcode() == HloOpcode::kConvert ||
         value->opcode() == HloOpcode::kSlice ||
         value->opcode() == HloOpcode::kDynamicSlice ||
         value->opcode() == HloOpcode::kTranspose) {
    const bool is_dynamic_slice = value->opcode() == HloOpcode::kDynamicSlice;
    CHECK_EQ(value->operand_count(),
             is_dynamic_slice ? value->shape().dimensions_size() + 1 : 1);
    const HloInstruction* operand = value->operand(0);
    if (value->opcode() == HloOpcode::kTranspose) {
      CHECK(!physical_mapping_fixed);
      CHECK(IsSupportedBatchInnerTranspose(*value));
      std::vector<int64_t> operand_strides = PhysicalStrides(operand->shape());
      physical_strides.resize(value->shape().dimensions_size());
      for (int64_t dimension = 0;
           dimension < value->shape().dimensions_size(); ++dimension) {
        physical_strides[dimension] =
            operand_strides[value->dimensions(dimension)];
        physical_offset +=
            logical_offsets[dimension] * physical_strides[dimension];
      }
      physical_mapping_fixed = true;
      has_input_view = true;
      requires_linear_addressing = true;
    } else if (value->opcode() == HloOpcode::kBitcast) {
      CHECK_EQ(value->shape().element_type(), operand->shape().element_type());
      if (!physical_mapping_fixed) {
        physical_strides = PhysicalStrides(value->shape());
        for (int64_t dimension = 0; dimension < logical_offsets.size();
             ++dimension) {
          physical_offset +=
              logical_offsets[dimension] * physical_strides[dimension];
        }
        physical_mapping_fixed = true;
      }
      has_input_view = true;
      requires_linear_addressing = true;
    } else if (value->opcode() == HloOpcode::kConvert) {
      CHECK(IsSupportedContractionConversion(operand->shape().element_type(),
                                             value->shape().element_type()));
      CHECK(ShapeUtil::SameDimensions(value->shape(), operand->shape()));
      has_conversion = true;
    } else if (value->opcode() == HloOpcode::kSlice) {
      CHECK(!physical_mapping_fixed);
      CHECK_EQ(value->shape().dimensions_size(),
               operand->shape().dimensions_size());
      for (int64_t dimension = 0; dimension < logical_offsets.size();
           ++dimension) {
        CHECK_EQ(value->slice_strides(dimension), 1);
        logical_offsets[dimension] += value->slice_starts(dimension);
      }
      has_input_view = true;
      requires_linear_addressing = true;
    } else {
      CHECK(!physical_mapping_fixed);
      CHECK(!has_dynamic_slice);
      CHECK_EQ(value->shape().dimensions_size(),
               operand->shape().dimensions_size());
      for (int64_t dimension = 0; dimension < value->shape().dimensions_size();
           ++dimension) {
        const HloInstruction* start = value->operand(dimension + 1);
        CHECK(ShapeUtil::IsScalar(start->shape()));
        CHECK(start->shape().element_type() == S32 ||
              start->shape().element_type() == S64);
        const int64_t limit = operand->shape().dimensions(dimension) -
                              value->shape().dimensions(dimension);
        CHECK_GE(limit, 0);
        dynamic_offset_limits[dimension] = limit;
        if (start->opcode() == HloOpcode::kParameter) {
          dynamic_offset_parameter_numbers[dimension] =
              start->parameter_number();
        } else {
          CHECK_EQ(start->opcode(), HloOpcode::kConstant);
          const int64_t constant =
              start->shape().element_type() == S32
                  ? start->literal().GetFirstElement<int32_t>()
                  : start->literal().GetFirstElement<int64_t>();
          dynamic_offset_constants[dimension] =
              std::clamp<int64_t>(constant, 0, limit);
        }
      }
      has_input_view = true;
      has_dynamic_slice = true;
    }
    value = operand;
  }
  CHECK_EQ(value->opcode(), HloOpcode::kParameter);
  const PrimitiveType parameter_type = value->shape().element_type();
  if (parameter_type == S4) {
    CHECK_EQ(contraction_type, BF16);
    CHECK(has_conversion);
    CHECK(!has_input_view);
  } else if (parameter_type == F32) {
    CHECK((contraction_type == BF16 && has_conversion) ||
          (contraction_type == F32 && !has_conversion));
  } else {
    CHECK_EQ(parameter_type, contraction_type);
    CHECK(!has_conversion);
  }
  CHECK_EQ(value->shape().element_type(), parameter_type);
  if (!physical_mapping_fixed) {
    physical_strides = PhysicalStrides(value->shape());
    for (int64_t dimension = 0; dimension < logical_offsets.size();
         ++dimension) {
      physical_offset +=
          logical_offsets[dimension] * physical_strides[dimension];
    }
  }
  if (!has_dynamic_slice) {
    dynamic_offset_parameter_numbers.clear();
    dynamic_offset_constants.clear();
    dynamic_offset_limits.clear();
  }
  return {{value->parameter_number()},
          parameter_type,
          requires_linear_addressing,
          physical_offset,
          std::move(physical_strides),
          /*concat_dimension=*/-1,
          /*concat_fragment_size=*/0,
          std::move(dynamic_offset_parameter_numbers),
          std::move(dynamic_offset_constants),
          std::move(dynamic_offset_limits)};
}

std::vector<int64_t> PhysicalStrides(const Shape& shape) {
  CHECK(shape.has_layout());
  std::vector<int64_t> strides(shape.dimensions_size());
  int64_t stride = 1;
  for (int64_t dimension : shape.layout().minor_to_major()) {
    strides[dimension] = stride;
    stride *= shape.dimensions(dimension);
  }
  return strides;
}

std::optional<int64_t> OutputTransposeBatchInner(
    const HloInstruction& root, const HloInstruction& dot) {
  if (dot.shape().element_type() != BF16 ||
      dot.shape().dimensions_size() != 3 ||
      root.opcode() != HloOpcode::kTranspose || root.operand_count() != 1 ||
      root.shape().element_type() != BF16 ||
      root.shape().dimensions_size() != 4 ||
      root.dimensions() != std::vector<int64_t>({0, 3, 1, 2})) {
    return std::nullopt;
  }
  const HloInstruction* bitcast = root.operand(0);
  if (bitcast->opcode() != HloOpcode::kBitcast ||
      bitcast->operand_count() != 1 || bitcast->operand(0) != &dot ||
      bitcast->shape().dimensions_size() != 4) {
    return std::nullopt;
  }
  const int64_t batch_outer = bitcast->shape().dimensions(0);
  const int64_t batch_inner = bitcast->shape().dimensions(1);
  if (batch_outer <= 0 || batch_inner <= 0 ||
      batch_outer * batch_inner != dot.shape().dimensions(0) ||
      bitcast->shape().dimensions(2) != dot.shape().dimensions(1) ||
      bitcast->shape().dimensions(3) != dot.shape().dimensions(2)) {
    return std::nullopt;
  }
  return batch_inner;
}

bool ContainsInstruction(const HloInstruction& root,
                         const HloInstruction& target) {
  if (&root == &target) {
    return true;
  }
  return std::any_of(root.operands().begin(), root.operands().end(),
                     [&](const HloInstruction* operand) {
                       return ContainsInstruction(*operand, target);
                     });
}

struct EpilogueStep {
  HloOpcode opcode;
  // Unary steps do not consume a parameter. For binary steps, -1 denotes a
  // scalar broadcast and 0/1 the retained row/column dimension.
  int64_t parameter_number = -1;
  PrimitiveType input_type = F32;
  int64_t broadcast_dimension = -2;
  int64_t contraction_operand = 0;
  bool input_is_constant = false;
  double constant_value = 0.0;
  PrimitiveType round_input_type = PRIMITIVE_TYPE_INVALID;
};

std::vector<EpilogueStep> FindEpilogueSteps(const HloInstruction& root,
                                            const HloInstruction& contraction) {
  const HloInstruction* value = &root;
  if (value->opcode() == HloOpcode::kConvert) {
    value = value->operand(0);
  }
  std::vector<EpilogueStep> steps;
  while (value != &contraction) {
    CHECK_LT(steps.size(), 3);
    EpilogueStep step{.opcode = value->opcode()};
    const HloInstruction* contraction_value = nullptr;
    if (value->opcode() == HloOpcode::kNegate) {
      contraction_value = value->operand(0);
    } else {
      CHECK(value->opcode() == HloOpcode::kAdd ||
            value->opcode() == HloOpcode::kMultiply ||
            value->opcode() == HloOpcode::kSubtract ||
            value->opcode() == HloOpcode::kDivide ||
            value->opcode() == HloOpcode::kMaximum ||
            value->opcode() == HloOpcode::kMinimum);
      const bool lhs_contains =
          ContainsInstruction(*value->operand(0), contraction);
      const int64_t contraction_operand = lhs_contains ? 0 : 1;
      step.contraction_operand = contraction_operand;
      contraction_value = value->operand(contraction_operand);
      const HloInstruction* broadcast = value->operand(1 - contraction_operand);
      if (broadcast->opcode() == HloOpcode::kConvert) {
        broadcast = broadcast->operand(0);
      }
      const HloInstruction* input = broadcast->operand(0);
      step.input_type = input->shape().element_type();
      step.broadcast_dimension =
          broadcast->dimensions().empty() ? -1 : broadcast->dimensions(0);
      if (input->opcode() == HloOpcode::kConstant) {
        step.input_is_constant = true;
        step.constant_value = input->literal().GetAsDouble({}).value();
      } else {
        step.parameter_number = input->parameter_number();
      }
    }
    const PrimitiveType contraction_value_type =
        contraction_value->shape().element_type();
    if (IsHalfPrecisionFloat(contraction_value_type)) {
      step.round_input_type = contraction_value_type;
    }
    if (contraction_value->opcode() == HloOpcode::kConvert) {
      const PrimitiveType converted_input_type =
          contraction_value->operand(0)->shape().element_type();
      if (contraction_value_type == F32 &&
          IsHalfPrecisionFloat(converted_input_type)) {
        // A BF16/F16 dot followed by a widening conversion has an observable
        // rounding boundary before its FP32 epilogue (attention score scaling
        // is the common case). Preserve it while accumulating the MFMA in F32.
        step.round_input_type = converted_input_type;
      }
      contraction_value = contraction_value->operand(0);
    }
    steps.push_back(step);
    value = contraction_value;
  }
  std::reverse(steps.begin(), steps.end());
  return steps;
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
      mfma_atom_ = fusion_config.fly_gemm_config().mfma_atom();
      use_mfma_32_ = mfma_atom_ == FlyGemmConfig::FLY_MFMA_32X32X8 ||
                     mfma_atom_ == FlyGemmConfig::FLY_MFMA_32X32X16_FP8 ||
                     mfma_atom_ == FlyGemmConfig::FLY_MFMA_32X32X2_F32 ||
                     mfma_atom_ == FlyGemmConfig::FLY_MFMA_32X32X4_XF32;
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
      dequantize_block_scales_ =
          fusion_config.fly_gemm_config().dequantize_block_scales();
      workgroup_mapping_n_ =
          fusion_config.fly_gemm_config().workgroup_mapping_n();
    }
    const HloInstruction& root = analysis.fusion_root(0).instruction();
    output_physical_strides_ = PhysicalStrides(root.shape());
    direct_accumulator_store_offsets_fit_ =
        ShapeUtil::ByteSizeOf(root.shape()) <=
        std::numeric_limits<uint32_t>::max();
    const HloInstruction* dot = FindContraction(root);
    CHECK_NE(dot, nullptr);
    output_transpose_batch_inner_ = OutputTransposeBatchInner(root, *dot);
    CHECK(!output_transpose_batch_inner_.has_value() || stage_output_);
    lhs_element_type_ = dot->operand(0)->shape().element_type();
    rhs_element_type_ = dot->operand(1)->shape().element_type();
    is_fp8_ = IsFnuzFp8(lhs_element_type_);
    CHECK_EQ(is_fp8_, IsFnuzFp8(rhs_element_type_));
    const bool native_fp8_atom =
        mfma_atom_ == FlyGemmConfig::FLY_MFMA_16X16X32_FP8 ||
        mfma_atom_ == FlyGemmConfig::FLY_MFMA_32X32X16_FP8;
    CHECK_EQ(is_fp8_ && !dequantize_block_scales_, native_fp8_atom);
    const bool native_f32_atom =
        mfma_atom_ == FlyGemmConfig::FLY_MFMA_16X16X4_F32 ||
        mfma_atom_ == FlyGemmConfig::FLY_MFMA_32X32X2_F32 ||
        mfma_atom_ == FlyGemmConfig::FLY_MFMA_32X32X4_XF32;
    CHECK_EQ(lhs_element_type_ == F32 && rhs_element_type_ == F32,
             native_f32_atom);
    CHECK(!dequantize_block_scales_ || is_fp8_);
    atom_k_ = native_fp8_atom
                  ? (use_mfma_32_ ? 16 : 32)
              : mfma_atom_ == FlyGemmConfig::FLY_MFMA_32X32X4_XF32
                  ? 4
              : native_f32_atom ? (use_mfma_32_ ? 2 : 4)
                                : (use_mfma_32_ ? 8 : 16);
    const ContractionInput lhs_input = FindContractionInput(*dot->operand(0));
    lhs_parameter_numbers_ = lhs_input.parameter_numbers;
    lhs_input_type_ = lhs_input.parameter_type;
    lhs_requires_linear_addressing_ = lhs_input.requires_linear_addressing;
    lhs_physical_offset_ = lhs_input.physical_offset;
    lhs_physical_strides_ = lhs_input.physical_strides;
    lhs_dynamic_offset_parameter_numbers_ =
        lhs_input.dynamic_offset_parameter_numbers;
    lhs_dynamic_offset_constants_ = lhs_input.dynamic_offset_constants;
    lhs_dynamic_offset_limits_ = lhs_input.dynamic_offset_limits;
    lhs_concat_fragment_size_ = lhs_input.concat_fragment_size;
    const ContractionInput rhs_input = FindContractionInput(*dot->operand(1));
    rhs_parameter_numbers_ = rhs_input.parameter_numbers;
    rhs_input_type_ = rhs_input.parameter_type;
    rhs_requires_linear_addressing_ = rhs_input.requires_linear_addressing;
    rhs_physical_offset_ = rhs_input.physical_offset;
    rhs_physical_strides_ = rhs_input.physical_strides;
    rhs_dynamic_offset_parameter_numbers_ =
        rhs_input.dynamic_offset_parameter_numbers;
    rhs_dynamic_offset_constants_ = rhs_input.dynamic_offset_constants;
    rhs_dynamic_offset_limits_ = rhs_input.dynamic_offset_limits;
    rhs_concat_fragment_size_ = rhs_input.concat_fragment_size;
    epilogue_steps_ = output_transpose_batch_inner_.has_value()
                          ? std::vector<EpilogueStep>{}
                          : FindEpilogueSteps(root, *dot);
    epilogue_row_dimension_ = dot->shape().dimensions_size() - 2;
    epilogue_column_dimension_ = dot->shape().dimensions_size() - 1;
    has_vector_epilogue_ = std::any_of(
        epilogue_steps_.begin(), epilogue_steps_.end(),
        [](const EpilogueStep& step) { return step.broadcast_dimension >= 0; });
    has_row_epilogue_ = std::any_of(
        epilogue_steps_.begin(), epilogue_steps_.end(),
        [&](const EpilogueStep& step) {
          return step.broadcast_dimension == epilogue_row_dimension_;
        });
    fold_final_negate_ =
        epilogue_steps_.size() == 3 &&
        epilogue_steps_[0].opcode == HloOpcode::kMultiply &&
        epilogue_steps_[1].opcode == HloOpcode::kAdd &&
        epilogue_steps_[2].opcode == HloOpcode::kNegate &&
        epilogue_steps_[0].input_type == F32 &&
        epilogue_steps_[1].input_type == F32 &&
        std::none_of(
            epilogue_steps_.begin(), epilogue_steps_.end(),
            [](const EpilogueStep& step) {
              return step.round_input_type != PRIMITIVE_TYPE_INVALID;
            });
    fusion_parameter_count_ = root.parent()->num_parameters();
    is_scaled_dot_ = dot->opcode() == HloOpcode::kScaledDot;
    if (is_scaled_dot_) {
      CHECK_EQ(dot->operand(2)->opcode(), HloOpcode::kParameter);
      CHECK_EQ(dot->operand(3)->opcode(), HloOpcode::kParameter);
      lhs_scale_parameter_number_ = dot->operand(2)->parameter_number();
      rhs_scale_parameter_number_ = dot->operand(3)->parameter_number();
      lhs_scale_element_type_ = dot->operand(2)->shape().element_type();
      rhs_scale_element_type_ = dot->operand(3)->shape().element_type();
      lhs_scale_dimensions_.assign(
          dot->operand(2)->shape().dimensions().begin(),
          dot->operand(2)->shape().dimensions().end());
      rhs_scale_dimensions_.assign(
          dot->operand(3)->shape().dimensions().begin(),
          dot->operand(3)->shape().dimensions().end());
      block_scaled_dot_ =
          ShapeUtil::ElementsIn(dot->operand(2)->shape()) != 1 ||
          ShapeUtil::ElementsIn(dot->operand(3)->shape()) != 1;
    }
    CHECK(!dequantize_block_scales_ || block_scaled_dot_);
    const DotDimensionNumbers& dot_dims = dot->dot_dimension_numbers();
    const bool rank_three = dot->shape().dimensions_size() == 3;
    global_split_k_ = rank_three && dot_dims.lhs_batch_dimensions_size() == 1 &&
                      dot_dims.lhs_batch_dimensions(0) == 1;
    batched_gemm_ = rank_three && dot_dims.lhs_batch_dimensions_size() == 1 &&
                    dot_dims.lhs_batch_dimensions(0) == 0;
    CHECK(!rank_three || global_split_k_ || batched_gemm_);
    batch_count_ = rank_three ? dot->shape().dimensions(0) : 1;
    split_k_batches_ = global_split_k_ ? batch_count_ : 1;
    m_ = dot->shape().dimensions(rank_three ? 1 : 0);
    n_ = dot->shape().dimensions(rank_three ? 2 : 1);
    k_ = dot->operand(0)->shape().dimensions(
        dot->dot_dimension_numbers().lhs_contracting_dimensions(0));
    lhs_contracting_dimension_ =
        dot->dot_dimension_numbers().lhs_contracting_dimensions(0);
    rhs_contracting_dimension_ =
        dot->dot_dimension_numbers().rhs_contracting_dimensions(0);
    if (global_split_k_) {
      lhs_batch_dimension_ = dot_dims.lhs_batch_dimensions(0);
      rhs_batch_dimension_ = dot_dims.rhs_batch_dimensions(0);
      for (int64_t dimension = 0; dimension < 3; ++dimension) {
        if (dimension != lhs_batch_dimension_ &&
            dimension != lhs_contracting_dimension_) {
          lhs_noncontracting_dimension_ = dimension;
        }
        if (dimension != rhs_batch_dimension_ &&
            dimension != rhs_contracting_dimension_) {
          rhs_noncontracting_dimension_ = dimension;
        }
      }
    }
    CHECK(lhs_parameter_numbers_.size() == 1 ||
          lhs_input.concat_dimension == 0);
    CHECK(rhs_parameter_numbers_.size() == 1 ||
          rhs_input.concat_dimension == 1 - rhs_contracting_dimension_);
    lhs_k_contiguous_ =
        lhs_physical_strides_[lhs_contracting_dimension_] == 1;
    rhs_k_contiguous_ =
        rhs_physical_strides_[rhs_contracting_dimension_] == 1;
    stage_k_ = is_fp8_ ? 32 : (k_ % 32 == 0 ? 32 : 16);
    absl::StatusOr<Tile> contraction_tile = dot->backend_config<Tile>();
    const bool supports_masked_k_tail =
        !is_fp8_ && !block_scaled_dot_ && !global_split_k_ && !async_lhs_ &&
        !direct_to_vgpr_ && !rolling_refill_ && !local_split_k_;
    if (contraction_tile.ok() && contraction_tile->sizes_size() == 1 &&
        contraction_tile->sizes(0) >= atom_k_ &&
        contraction_tile->sizes(0) % atom_k_ == 0 &&
        (k_ % contraction_tile->sizes(0) == 0 ||
         supports_masked_k_tail)) {
      stage_k_ = contraction_tile->sizes(0);
    }
    const int64_t blocks = batch_count_ * ((m_ + block_m_ - 1) / block_m_) *
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
    TF_RET_CHECK(entry_function.getNumArguments() ==
                 fusion_parameter_count_ + 1);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());

    CHECK(!lhs_parameter_numbers_.empty());
    CHECK(!rhs_parameter_numbers_.empty());
    Value lhs = entry_function.getArgument(lhs_parameter_numbers_.front());
    Value rhs = entry_function.getArgument(rhs_parameter_numbers_.front());
    Value output =
        entry_function.getArgument(entry_function.getNumArguments() - 1);
    Value output_scale;
    if (is_scaled_dot_ && !block_scaled_dot_) {
      Value zero = IndexConstant(builder, 0);
      auto load_scale = [&](Value tensor) {
        auto tensor_type = mlir::cast<mlir::RankedTensorType>(tensor.getType());
        llvm::SmallVector<Value> indices(tensor_type.getRank(), zero);
        Value scale = mlir::tensor::ExtractOp::create(
            builder, tensor, mlir::ValueRange{indices});
        mlir::Type type = scale.getType();
        if (type.isF32()) {
          return scale;
        }
        if (type.isInteger(8)) {
          return mlir::arith::SIToFPOp::create(builder, builder.getF32Type(),
                                               scale)
              .getResult();
        }
        return mlir::arith::ExtFOp::create(builder, builder.getF32Type(), scale)
            .getResult();
      };
      output_scale = mlir::arith::MulFOp::create(
          builder,
          load_scale(entry_function.getArgument(lhs_scale_parameter_number_)),
          load_scale(entry_function.getArgument(rhs_scale_parameter_number_)));
    }
    llvm::SmallVector<Value> epilogue_inputs(epilogue_steps_.size());
    llvm::SmallVector<Value> scalar_epilogue_values(epilogue_steps_.size());
    for (int64_t index = 0; index < epilogue_steps_.size(); ++index) {
      const EpilogueStep& step = epilogue_steps_[index];
      if (step.opcode == HloOpcode::kNegate) {
        continue;
      }
      if (step.input_is_constant) {
        TF_RET_CHECK(step.broadcast_dimension == -1);
        Value scalar = mlir::arith::ConstantFloatOp::create(
            builder, builder.getF32Type(),
            llvm::APFloat(static_cast<float>(step.constant_value)));
        if (fold_final_negate_ && index < 2) {
          scalar = mlir::arith::NegFOp::create(builder, scalar);
        }
        scalar_epilogue_values[index] = scalar;
        continue;
      }
      epilogue_inputs[index] =
          entry_function.getArgument(step.parameter_number);
      if (step.broadcast_dimension == -1) {
        Value scalar = mlir::tensor::ExtractOp::create(
            builder, epilogue_inputs[index], mlir::ValueRange{});
        if (IsHalfPrecisionFloat(step.input_type)) {
          scalar = mlir::arith::ExtFOp::create(builder, builder.getF32Type(),
                                               scalar);
        } else {
          TF_RET_CHECK(step.input_type == F32);
        }
        if (fold_final_negate_ && index < 2) {
          scalar = mlir::arith::NegFOp::create(builder, scalar);
        }
        scalar_epilogue_values[index] = scalar;
      }
    }
    auto emit_dynamic_offsets = [&](llvm::ArrayRef<int64_t> parameter_numbers,
                                    llvm::ArrayRef<int64_t> constants,
                                    llvm::ArrayRef<int64_t> limits) {
      CHECK_EQ(parameter_numbers.size(), constants.size());
      CHECK_EQ(parameter_numbers.size(), limits.size());
      llvm::SmallVector<Value, 3> offsets;
      offsets.reserve(parameter_numbers.size());
      for (int64_t dimension = 0; dimension < parameter_numbers.size();
           ++dimension) {
        if (parameter_numbers[dimension] < 0) {
          offsets.push_back(IndexConstant(builder, constants[dimension]));
          continue;
        }
        Value scalar = mlir::tensor::ExtractOp::create(
            builder, entry_function.getArgument(parameter_numbers[dimension]),
            mlir::ValueRange{});
        Value offset = mlir::arith::IndexCastOp::create(
            builder, builder.getIndexType(), scalar);
        Value zero = IndexConstant(builder, 0);
        Value limit = IndexConstant(builder, limits[dimension]);
        Value below_zero = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::slt, offset, zero);
        offset =
            mlir::arith::SelectOp::create(builder, below_zero, zero, offset);
        Value above_limit = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::sgt, offset, limit);
        offsets.push_back(
            mlir::arith::SelectOp::create(builder, above_limit, limit, offset));
      }
      return offsets;
    };
    llvm::SmallVector<Value, 3> lhs_dynamic_offsets = emit_dynamic_offsets(
        lhs_dynamic_offset_parameter_numbers_, lhs_dynamic_offset_constants_,
        lhs_dynamic_offset_limits_);
    llvm::SmallVector<Value, 3> rhs_dynamic_offsets = emit_dynamic_offsets(
        rhs_dynamic_offset_parameter_numbers_, rhs_dynamic_offset_constants_,
        rhs_dynamic_offset_limits_);
    auto add_dynamic_offsets = [&](llvm::SmallVectorImpl<Value>& indices,
                                   mlir::ValueRange offsets) {
      CHECK(offsets.empty() || indices.size() == offsets.size());
      for (int64_t dimension = 0; dimension < offsets.size(); ++dimension) {
        indices[dimension] =
            Add(builder, indices[dimension], offsets[dimension]);
      }
    };
    Value batch_id;
    auto lhs_indices = [&](Value m, Value k) {
      llvm::SmallVector<Value, 3> indices;
      if (lhs_parameter_numbers_.size() > 1) {
        m = Rem(builder, m, lhs_concat_fragment_size_);
      }
      if (global_split_k_) {
        indices.resize(3);
        indices[lhs_noncontracting_dimension_] = m;
        indices[lhs_batch_dimension_] = batch_id;
        indices[lhs_contracting_dimension_] = k;
      } else if (batched_gemm_) {
        indices.assign({batch_id, m, k});
      } else {
        indices.assign({m, k});
      }
      add_dynamic_offsets(indices, lhs_dynamic_offsets);
      return indices;
    };
    auto rhs_indices = [&](Value k, Value n) {
      llvm::SmallVector<Value, 3> indices;
      if (rhs_parameter_numbers_.size() > 1) {
        n = Rem(builder, n, rhs_concat_fragment_size_);
      }
      if (global_split_k_) {
        indices.resize(3);
        indices[rhs_noncontracting_dimension_] = n;
        indices[rhs_batch_dimension_] = batch_id;
        indices[rhs_contracting_dimension_] = k;
      } else if (batched_gemm_) {
        if (rhs_contracting_dimension_ == 1) {
          indices.assign({batch_id, k, n});
        } else {
          indices.assign({batch_id, n, k});
        }
      } else if (rhs_contracting_dimension_ == 0) {
        indices.assign({k, n});
      } else {
        indices.assign({n, k});
      }
      add_dynamic_offsets(indices, rhs_dynamic_offsets);
      return indices;
    };
    auto physical_linear_index = [&](mlir::ValueRange indices,
                                     llvm::ArrayRef<int64_t> strides,
                                     int64_t offset) {
      CHECK_EQ(indices.size(), strides.size());
      Value linear = IndexConstant(builder, offset);
      for (int64_t dimension = 0; dimension < strides.size(); ++dimension) {
        linear = Add(builder, linear,
                     Mul(builder, indices[dimension],
                         IndexConstant(builder, strides[dimension])));
      }
      return linear;
    };
    auto lhs_physical_linear_index = [&](Value m, Value k) {
      llvm::SmallVector<Value, 3> indices = lhs_indices(m, k);
      return physical_linear_index(indices, lhs_physical_strides_,
                                   lhs_physical_offset_);
    };
    auto rhs_physical_linear_index = [&](Value k, Value n) {
      llvm::SmallVector<Value, 3> indices = rhs_indices(k, n);
      return physical_linear_index(indices, rhs_physical_strides_,
                                   rhs_physical_offset_);
    };
    // A row-vector GEMV has only one 16x16 MFMA output atom. Two or four waves
    // can nevertheless cooperate on it by accumulating disjoint K partitions
    // of a staged K256 tile and reducing their vector<4xf32> results through
    // LDS.
    // Keep this distinct from the large FlyDSL LocalSplitU schedule below:
    // that schedule assumes four waves and sixteen accumulators per wave.
    const bool tiny_gemv_local_split =
        local_split_k_ && stage_rhs_ && preload_lds_fragments_ &&
        !single_buffer_lds_ && !use_mfma_32_ && !is_fp8_ &&
        !global_split_k_ && m_ == 1 && block_m_ == 16 && block_n_ == 16 &&
        stage_k_ == 256 && (num_warps_ == 2 || num_warps_ == 4) &&
        lhs_element_type_ == rhs_element_type_ &&
        IsHalfPrecisionFloat(lhs_element_type_);
    if (stage_rhs_) {
      const int64_t copy_elements =
          lhs_element_type_ == F32 ? 4 : (is_fp8_ ? 8 : 2);
      const int64_t block_threads = num_warps_ * 64;
      const bool row_major_rhs_square_double_buffer =
          !single_buffer_lds_ && use_mfma_32_ && num_warps_ == 4 &&
          block_m_ == 128 &&
          ((block_n_ == 128 && (stage_k_ == 32 || stage_k_ == 64)) ||
           (block_n_ == 256 && stage_k_ == 32)) &&
          !rhs_k_contiguous_;
      TF_RET_CHECK(stage_k_ >= 32 && stage_k_ % 32 == 0 && block_m_ % 16 == 0 &&
                   block_n_ % 16 == 0 &&
                   (!use_mfma_32_ || lhs_element_type_ == F32 ||
                    single_buffer_lds_ ||
                    row_major_rhs_square_double_buffer));
      TF_RET_CHECK((block_m_ * stage_k_ / copy_elements) % block_threads == 0 &&
                   (block_n_ * stage_k_ / copy_elements) % block_threads == 0);
      const int64_t lds_stages =
          (single_buffer_lds_ || row_major_rhs_square_double_buffer) ? 1 : 2;
      const int64_t tensile_stage_k = is_fp8_ ? 128 : 64;
      const bool tensile_double_buffer =
          !single_buffer_lds_ && direct_to_vgpr_ && !use_mfma_32_ &&
          num_warps_ == 4 && stage_k_ == tensile_stage_k &&
          ((block_m_ == 256 && block_n_ == 224) ||
           (block_m_ == 224 && block_n_ == 256));
      // Tensile stages only the non-DirectToVgpr operand. Its 224-row payload
      // uses seven padded 32-row blocks (31.5 KiB) and a 32 KiB ping-pong
      // stride, for an exact 65,024-byte allocation for either BF16xK64 or
      // FP8xK128.
      const int64_t tensile_element_bytes =
          ContractionElementBytes(lhs_element_type_);
      const int64_t kTensileStageStrideElements =
          32 * 1024 / tensile_element_bytes;
      const int64_t kTensilePhysicalStageElements =
          7 * 4608 / tensile_element_bytes;
      const int64_t staged_lds_bytes =
          tensile_double_buffer
              ? (kTensileStageStrideElements +
                 kTensilePhysicalStageElements) *
                    tensile_element_bytes
              : lds_stages * stage_k_ *
                    (block_m_ * ContractionElementBytes(lhs_element_type_) +
                     block_n_ * ContractionElementBytes(rhs_element_type_));
      TF_RET_CHECK(staged_lds_bytes <= 64 * 1024);
      // XLA's global split-K and Fly's wave-local split-K are composable: the
      // former selects an input/output batch while the latter reduces two
      // wave partitions into that batch's FP32 partial. Staged-output and
      // async-LHS variants still have no rank-3 epilogue.
      TF_RET_CHECK(!global_split_k_ || (!stage_output_ && !async_lhs_));
      const bool tensile_edge_tile =
          (single_buffer_lds_ || tensile_double_buffer) && !use_mfma_32_ &&
          num_warps_ == 4 && stage_k_ == tensile_stage_k &&
          ((block_m_ == 256 && block_n_ == 224) ||
           (block_m_ == 224 && block_n_ == 256)) &&
          m_ >= block_m_ && n_ >= block_n_;
      const bool shifted_staged_edge_tile =
          !global_split_k_ && m_ >= block_m_ && n_ >= block_n_;
      TF_RET_CHECK((m_ % block_m_ == 0 && n_ % block_n_ == 0) ||
                   (global_split_k_ && m_ % block_m_ == 0 && n_ % 16 == 0) ||
                   tensile_edge_tile || shifted_staged_edge_tile ||
                   (tiny_gemv_local_split && n_ % block_n_ == 0));
    }
    if (preload_lds_fragments_) {
      const bool native_double_buffer =
          !single_buffer_lds_ && !use_mfma_32_ &&
          ((stage_k_ == 64 && block_m_ == block_n_ &&
            (block_m_ == 64 || block_m_ == 128)) ||
           (is_fp8_ && stage_k_ == 128 && block_m_ == 128 &&
            block_n_ == 128));
      const bool triton_single_buffer =
          single_buffer_lds_ && use_mfma_32_ && num_warps_ == 4 &&
          ((stage_k_ == 32 && ((block_m_ == 128 && block_n_ == 256) ||
                               (block_m_ == 256 && block_n_ == 128))) ||
           (stage_k_ == 64 && block_m_ == 128 && block_n_ == 128));
      const bool triton_eight_wave_single_buffer =
          single_buffer_lds_ && use_mfma_32_ && num_warps_ == 8 &&
          block_m_ == 256 && block_n_ == 128 &&
          (stage_k_ == 64 ||
           (lhs_element_type_ == F32 && stage_k_ == 32));
      const int64_t tensile_stage_k = is_fp8_ ? 128 : 64;
      const bool tensile_single_buffer =
          single_buffer_lds_ && !use_mfma_32_ && num_warps_ == 4 &&
          stage_k_ == tensile_stage_k &&
          ((block_m_ == 256 && block_n_ == 224) ||
           (block_m_ == 224 && block_n_ == 256));
      const bool tensile_double_buffer =
          !single_buffer_lds_ && direct_to_vgpr_ && !use_mfma_32_ &&
          num_warps_ == 4 && stage_k_ == tensile_stage_k &&
          ((block_m_ == 256 && block_n_ == 224) ||
           (block_m_ == 224 && block_n_ == 256));
      const bool small_grid_single_buffer =
          single_buffer_lds_ && !use_mfma_32_ && num_warps_ == 4 &&
          stage_k_ == 128 && block_m_ == 128 &&
          (block_n_ == 64 || block_n_ == 96);
      const bool row_major_rhs_single_buffer =
          single_buffer_lds_ && use_mfma_32_ && num_warps_ == 8 &&
          stage_k_ == 128 && block_m_ == 128 && block_n_ == 64 &&
          !rhs_k_contiguous_;
      const bool row_major_rhs_square_double_buffer =
          !single_buffer_lds_ && use_mfma_32_ && num_warps_ == 4 &&
          block_m_ == 128 &&
          ((block_n_ == 128 && (stage_k_ == 32 || stage_k_ == 64)) ||
           (block_n_ == 256 && stage_k_ == 32)) &&
          !rhs_k_contiguous_;
      TF_RET_CHECK(stage_rhs_ && !async_lhs_ &&
                   (native_double_buffer || triton_single_buffer ||
                    triton_eight_wave_single_buffer ||
                    tensile_single_buffer || tensile_double_buffer ||
                    small_grid_single_buffer || row_major_rhs_single_buffer ||
                    row_major_rhs_square_double_buffer ||
                    tiny_gemv_local_split));
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
      // gfx942 buffer-to-LDS copies move one dword per lane. Express the
      // uniform wave base in destination elements: 64 lanes * 4 bytes.
      // BF16 therefore advances by 128 elements and FP8 by 256 elements.
      constexpr int64_t kDirectCopyBytesPerLane = 4;
      direct_copy_wave_offset = ReadFirstLaneIndex(
          builder,
          Mul(builder, wave_id,
              IndexConstant(builder,
                            64 * kDirectCopyBytesPerLane /
                                ContractionElementBytes(lhs_element_type_))));
    }
    const bool use_mfma_32 = use_mfma_32_;
    const bool paired_mfma = stage_rhs_ || async_lhs_;
    const bool triton_vec4_lds = single_buffer_lds_ && use_mfma_32 &&
                                 stage_k_ == 32 && block_m_ == 128 &&
                                 block_n_ == 256 && num_warps_ == 4;
    const int64_t tensile_stage_k = is_fp8_ ? 128 : 64;
    const bool tensile_wide_tile = !use_mfma_32 &&
                                   stage_k_ == tensile_stage_k &&
                                   ((block_m_ == 256 && block_n_ == 224) ||
                                    (block_m_ == 224 && block_n_ == 256)) &&
                                   num_warps_ == 4;
    const bool row_major_rhs_single_buffer =
        single_buffer_lds_ && use_mfma_32 && num_warps_ == 8 &&
        stage_k_ == 128 && block_m_ == 128 && block_n_ == 64 &&
        !rhs_k_contiguous_;
    const bool row_major_rhs_square_double_buffer =
        stage_rhs_ && preload_lds_fragments_ && !single_buffer_lds_ &&
        use_mfma_32 && num_warps_ == 4 && block_m_ == 128 &&
        ((block_n_ == 128 && (stage_k_ == 32 || stage_k_ == 64)) ||
         (block_n_ == 256 && stage_k_ == 32)) &&
        !rhs_k_contiguous_;
    const bool row_major_rhs_wide_source_swap =
        row_major_rhs_square_double_buffer && block_n_ == 256 &&
        !stage_output_;
    const bool small_grid_single_buffer =
        (single_buffer_lds_ && !use_mfma_32 && num_warps_ == 4 &&
         stage_k_ == 128 && block_m_ == 128 &&
         (block_n_ == 64 || block_n_ == 96)) ||
        row_major_rhs_single_buffer;
    const bool small_grid_rolling_refill =
        small_grid_single_buffer && rolling_refill_;
    const bool grouped_tile = workgroup_mapping_n_ > 0;
    const bool tensile_transposed_tile =
        tensile_wide_tile && block_m_ == 224 && block_n_ == 256;
    const bool source_swap_mfma = tensile_wide_tile || local_split_k_;
    TF_RET_CHECK(!block_scaled_dot_ ||
                 (!prefetch_rhs_ && !stage_output_ && !stage_rhs_ &&
                  !async_lhs_ && !preload_lds_fragments_ &&
                  !single_buffer_lds_ && !direct_to_vgpr_ &&
                  !rolling_refill_ && !local_split_k_));
    if (block_scaled_dot_) {
      TF_RET_CHECK(lhs_scale_dimensions_.size() == 2 &&
                   rhs_scale_dimensions_.size() == 2);
      const int64_t lhs_scale_k = lhs_scale_dimensions_[1];
      const int64_t rhs_scale_k =
          rhs_scale_dimensions_[rhs_contracting_dimension_];
      const int64_t lhs_block = k_ / lhs_scale_k;
      const int64_t rhs_block = k_ / rhs_scale_k;
      const int64_t scale_block =
          lhs_scale_k == 1 ? rhs_block : lhs_block;
      TF_RET_CHECK((rhs_scale_k == 1 || lhs_scale_k == 1 ||
                    lhs_block == rhs_block) &&
                   stage_k_ == scale_block);
    }
    TF_RET_CHECK(!rolling_refill_ || small_grid_single_buffer);
    TF_RET_CHECK(!local_split_k_ ||
                 ((small_grid_single_buffer || tiny_gemv_local_split) &&
                  stage_rhs_ &&
                  preload_lds_fragments_ && !stage_output_ &&
                  stage_k_ % 2 == 0 && k_ % stage_k_ == 0 &&
                  k_ >= 2 * stage_k_));
    // hipBLASLt's matching wide-tile solutions use DirectToVgprA. Keep the
    // wave-local 256-wide operand in registers in either output orientation.
    const bool tensile_direct_lhs =
        direct_to_vgpr_ && tensile_wide_tile && !tensile_transposed_tile;
    const bool tensile_direct_rhs =
        direct_to_vgpr_ && tensile_wide_tile && tensile_transposed_tile;
    const bool tensile_direct_operand =
        tensile_direct_lhs || tensile_direct_rhs;
    const bool tensile_double_buffer =
        tensile_direct_operand && !single_buffer_lds_;
    // Tensile's ShiftPtr edge mode moves the final wide macro-tile back so
    // every MFMA fragment and output store is in bounds.  The preceding tile
    // and final tile deliberately overlap, but compute identical values.  In
    // addition to removing per-fragment masks this is essential for keeping
    // the 224 FP32 accumulators in AGPRs across the epilogue.
    const bool shift_n_edge = stage_rhs_ && !global_split_k_ &&
                              n_ >= block_n_ && n_ % block_n_ != 0;
    const bool shift_m_edge = stage_rhs_ && !global_split_k_ &&
                              m_ >= block_m_ && m_ % block_m_ != 0;
    TF_RET_CHECK(!is_fp8_ || !tensile_wide_tile ||
                 lhs_element_type_ == rhs_element_type_);
    const int64_t atom_m = use_mfma_32 ? 32 : 16;
    const int64_t atom_k = atom_k_;
    const bool xf32_single_buffer =
        single_buffer_lds_ && use_mfma_32 &&
        mfma_atom_ == FlyGemmConfig::FLY_MFMA_32X32X4_XF32 &&
        lhs_element_type_ == F32 && rhs_element_type_ == F32;
    const bool xf32_wide_single_buffer =
        xf32_single_buffer && block_m_ == 256 && block_n_ == 128 &&
        stage_k_ == 32 && num_warps_ == 8;
    const bool triton_xf32_paired_lds = false;
    const int64_t accumulator_elements = use_mfma_32 ? 16 : 4;
    Value lane_axis = Rem(builder, lane_id, atom_m);
    Value lane_group = Div(builder, lane_id, atom_m);
    const int64_t local_split_factor =
        tiny_gemv_local_split ? num_warps_ : (local_split_k_ ? 2 : 1);
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
    batch_id = (global_split_k_ || batched_gemm_)
                   ? Div(builder, block_id, output_grid)
                   : IndexConstant(builder, 0);
    Value output_block_id = (global_split_k_ || batched_gemm_)
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
          mlir::arith::CmpIOp::create(builder, mlir::arith::CmpIPredicate::ult,
                                      xcc_group_offset, mapped_range),
          Add(builder, xcc_group_base, mapped_offset), output_block_id);
    }
    Value block_m;
    Value block_n;
    if (row_major_rhs_square_double_buffer) {
      // Triton's grouped program-id mapping visits all M tiles for one N tile
      // before advancing N.  This shape has only four M tiles but 86 N tiles,
      // so the M-fast order lets neighboring workgroups reuse the same 1 MiB
      // RHS macro-tile instead of sweeping the 90 MiB RHS once per M tile.
      block_m = Rem(builder, output_block_id, grid_m);
      block_n = Div(builder, output_block_id, grid_m);
    } else if (tensile_direct_rhs) {
      // The matching hipBLASLt WGM4 tile is transposed relative to this one:
      // its 256-wide DirectToVgprA dimension is XLA's N dimension. Apply the
      // corresponding grouping to four neighboring N tiles here. The final
      // group may contain fewer than four N tiles.
      constexpr int64_t kWorkgroupMappingN = 4;
      Value group_id =
          Div(builder, output_block_id, kWorkgroupMappingN * grid_m);
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
          Rem(builder, output_block_id, kWorkgroupMappingN * grid_m);
      block_m =
          mlir::arith::DivUIOp::create(builder, block_in_group, group_n);
      block_n = Add(builder, first_block_n,
                    mlir::arith::RemUIOp::create(builder, block_in_group,
                                                group_n));
    } else if (tensile_direct_lhs) {
      // Apply the autotuned positive WGM after WGMXCC has remapped the
      // serialized dispatch id.  Legacy/preconfigured HLOs use the original
      // WGM=8 default when the field is absent.
      const int64_t workgroup_mapping_n =
          workgroup_mapping_n_ > 0 ? workgroup_mapping_n_ : 8;
      Value group_id =
          Div(builder, output_block_id, grid_m * workgroup_mapping_n);
      Value first_block_n =
          Mul(builder, group_id,
              IndexConstant(builder, workgroup_mapping_n));
      Value remaining_n =
          Sub(builder, IndexConstant(builder, grid_n), first_block_n);
      Value group_n = mlir::arith::SelectOp::create(
          builder,
          mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, remaining_n,
              IndexConstant(builder, workgroup_mapping_n)),
          remaining_n, IndexConstant(builder, workgroup_mapping_n));
      Value block_in_group =
          Rem(builder, output_block_id, grid_m * workgroup_mapping_n);
      block_m = mlir::arith::DivUIOp::create(builder, block_in_group, group_n);
      block_n = Add(builder, first_block_n,
                    mlir::arith::RemUIOp::create(builder, block_in_group,
                                                group_n));
    } else if (grouped_tile) {
      // Balance reuse of both operands on launch-scale GEMMs. Grouping
      // neighboring N tiles keeps their B payload hot while sweeping M,
      // instead of exhausting all N tiles for one A row before revisiting B.
      const int64_t workgroup_mapping_n = workgroup_mapping_n_;
      Value group_id =
          Div(builder, output_block_id, grid_m * workgroup_mapping_n);
      Value first_block_n =
          Mul(builder, group_id, IndexConstant(builder, workgroup_mapping_n));
      Value remaining_n =
          Sub(builder, IndexConstant(builder, grid_n), first_block_n);
      Value group_n = mlir::arith::SelectOp::create(
          builder,
          mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, remaining_n,
              IndexConstant(builder, workgroup_mapping_n)),
          remaining_n, IndexConstant(builder, workgroup_mapping_n));
      Value block_in_group =
          Rem(builder, output_block_id, grid_m * workgroup_mapping_n);
      block_m = mlir::arith::DivUIOp::create(builder, block_in_group, group_n);
      block_n =
          Add(builder, first_block_n,
              mlir::arith::RemUIOp::create(builder, block_in_group, group_n));
    } else {
      block_m = Div(builder, output_block_id, grid_n);
      block_n = Rem(builder, output_block_id, grid_n);
    }
    Value block_m_base =
        Mul(builder, block_m, IndexConstant(builder, block_m_));
    Value block_n_base =
        Mul(builder, block_n, IndexConstant(builder, block_n_));
    if (shift_m_edge) {
      Value shifted_edge = IndexConstant(builder, m_ - block_m_);
      block_m_base = mlir::arith::SelectOp::create(
          builder,
          mlir::arith::CmpIOp::create(builder,
                                      mlir::arith::CmpIPredicate::ult,
                                      block_m_base, shifted_edge),
          block_m_base, shifted_edge);
    }
    if (shift_n_edge) {
      Value shifted_edge = IndexConstant(builder, n_ - block_n_);
      block_n_base = mlir::arith::SelectOp::create(
          builder,
          mlir::arith::CmpIOp::create(builder,
                                      mlir::arith::CmpIPredicate::ult,
                                      block_n_base, shifted_edge),
          block_n_base, shifted_edge);
    }
    Value lhs_concat_fragment;
    if (lhs_parameter_numbers_.size() > 1) {
      lhs_concat_fragment =
          Div(builder, block_m_base, lhs_concat_fragment_size_);
    }
    Value rhs_concat_fragment;
    if (rhs_parameter_numbers_.size() > 1) {
      rhs_concat_fragment =
          Div(builder, block_n_base, rhs_concat_fragment_size_);
    }
    auto output_indices = [&](Value row, Value column) {
      llvm::SmallVector<Value> indices;
      if (output_transpose_batch_inner_.has_value()) {
        Value batch_outer =
            Div(builder, batch_id, *output_transpose_batch_inner_);
        Value batch_inner =
            Rem(builder, batch_id, *output_transpose_batch_inner_);
        indices.assign({batch_outer, column, batch_inner, row});
        return indices;
      }
      if (global_split_k_ || batched_gemm_) {
        indices.assign({batch_id, row, column});
      } else {
        indices.assign({row, column});
      }
      return indices;
    };

    const PrimitiveType mma_lhs_element_type =
        dequantize_block_scales_ ? BF16 : lhs_element_type_;
    const PrimitiveType mma_rhs_element_type =
        dequantize_block_scales_ ? BF16 : rhs_element_type_;
    const std::string mma_atom_assembly =
        "!fly.mma_atom<!fly_rocdl.cdna3.mfma<" + std::to_string(atom_m) + "x" +
        std::to_string(atom_m) + "x" + std::to_string(atom_k) + ", (" +
        FlyTypeName(mma_lhs_element_type) + "," +
        FlyTypeName(mma_rhs_element_type) + ")->f32>>";
    mlir::Type mma_atom_type =
        mlir::parseType(mma_atom_assembly, entry_function.getContext());
    TF_RET_CHECK(mma_atom_type != nullptr);
    mlir::OperationState atom_state(entry_function.getLoc(),
                                    "fly.make_mma_atom");
    atom_state.addTypes(mma_atom_type);
    Value mma_atom = builder.create(atom_state)->getResult(0);

    mlir::Type lhs_element_type =
        ContractionElementType(builder, lhs_element_type_);
    mlir::Type rhs_element_type =
        ContractionElementType(builder, rhs_element_type_);
    const int64_t input_fragment_elements = atom_m * atom_k / 64;
    mlir::VectorType input_vector_type =
        mlir::VectorType::get({input_fragment_elements}, lhs_element_type);
    mlir::VectorType rhs_input_vector_type =
        mlir::VectorType::get({input_fragment_elements}, rhs_element_type);
    mlir::VectorType staged_input_vector_type =
        mlir::VectorType::get({2 * input_fragment_elements}, lhs_element_type);
    mlir::VectorType accumulator_type =
        mlir::VectorType::get({accumulator_elements}, builder.getF32Type());
    auto apply_epilogue_step = [&](Value value, int64_t step_index,
                                   Value operand) -> Value {
      const EpilogueStep& step = epilogue_steps_[step_index];
      mlir::Type compute_type = value.getType();
      mlir::Type rounded_type =
          step.round_input_type == F16 ? builder.getF16Type()
                                       : builder.getBF16Type();
      if (auto vector_type =
              mlir::dyn_cast<mlir::VectorType>(value.getType())) {
        if (step.opcode != HloOpcode::kNegate &&
            !mlir::isa<mlir::VectorType>(operand.getType())) {
          operand =
              mlir::vector::BroadcastOp::create(builder, vector_type, operand);
        }
        rounded_type = mlir::VectorType::get(
            vector_type.getShape(),
            step.round_input_type == F16 ? builder.getF16Type()
                                         : builder.getBF16Type());
      }
      if (step.round_input_type != PRIMITIVE_TYPE_INVALID) {
        value = mlir::arith::TruncFOp::create(builder, rounded_type, value);
        value = mlir::arith::ExtFOp::create(builder, compute_type, value);
      }
      if (step.opcode == HloOpcode::kNegate) {
        return mlir::arith::NegFOp::create(builder, value);
      }
      if (step.opcode == HloOpcode::kAdd) {
        return mlir::arith::AddFOp::create(builder, value, operand);
      }
      if (step.opcode == HloOpcode::kMultiply) {
        return mlir::arith::MulFOp::create(builder, value, operand);
      }
      Value lhs = step.contraction_operand == 0 ? value : operand;
      Value rhs = step.contraction_operand == 0 ? operand : value;
      if (step.opcode == HloOpcode::kSubtract) {
        return mlir::arith::SubFOp::create(builder, lhs, rhs);
      }
      if (step.opcode == HloOpcode::kDivide) {
        return mlir::arith::DivFOp::create(builder, lhs, rhs);
      }
      if (step.opcode == HloOpcode::kMaximum) {
        return mlir::arith::MaximumFOp::create(builder, lhs, rhs);
      }
      CHECK_EQ(step.opcode, HloOpcode::kMinimum);
      return mlir::arith::MinimumFOp::create(builder, lhs, rhs);
    };
    auto load_epilogue_element = [&](int64_t step_index, Value row,
                                     Value column) -> Value {
      const EpilogueStep& step = epilogue_steps_[step_index];
      CHECK_NE(step.opcode, HloOpcode::kNegate);
      if (step.broadcast_dimension == -1) {
        return scalar_epilogue_values[step_index];
      }
      Value index;
      int64_t dimension_size;
      if (step.broadcast_dimension == epilogue_row_dimension_) {
        index = row;
        dimension_size = m_;
      } else if (step.broadcast_dimension == epilogue_column_dimension_) {
        index = column;
        dimension_size = n_;
      } else {
        CHECK(batched_gemm_ && step.broadcast_dimension == 0);
        index = batch_id;
        dimension_size = batch_count_;
      }
      Value in_bounds = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, index,
          IndexConstant(builder, dimension_size));
      Value safe_index = mlir::arith::SelectOp::create(
          builder, in_bounds, index, IndexConstant(builder, 0));
      Value element = mlir::tensor::ExtractOp::create(
          builder, epilogue_inputs[step_index], mlir::ValueRange{safe_index});
      if (IsHalfPrecisionFloat(step.input_type)) {
        element =
            mlir::arith::ExtFOp::create(builder, builder.getF32Type(), element);
      }
      if (fold_final_negate_ && step_index < 2) {
        element = mlir::arith::NegFOp::create(builder, element);
      }
      return element;
    };
    auto apply_output_epilogue = [&](Value value, Value row,
                                     Value column) -> Value {
      const int64_t step_count =
          epilogue_steps_.size() - (fold_final_negate_ ? 1 : 0);
      for (int64_t step_index = 0; step_index < step_count; ++step_index) {
        Value operand;
        if (epilogue_steps_[step_index].opcode != HloOpcode::kNegate) {
          operand = load_epilogue_element(step_index, row, column);
        }
        value = apply_epilogue_step(value, step_index, operand);
      }
      return value;
    };
    auto apply_scalar_epilogue = [&](Value value) -> Value {
      CHECK(!has_vector_epilogue_);
      const int64_t step_count =
          epilogue_steps_.size() - (fold_final_negate_ ? 1 : 0);
      for (int64_t step_index = 0; step_index < step_count; ++step_index) {
        value = apply_epilogue_step(value, step_index,
                                    scalar_epilogue_values[step_index]);
      }
      return value;
    };
    auto emit_buffer_load = [&](Value source, Value source_linear_index,
                                mlir::Type result_type) {
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
      store_state.addOperands({value, destination, destination_linear_index});
      store_state.addTypes(destination.getType());
      store_state.addAttribute("cache_policy", builder.getI32IntegerAttr(0));
      return builder.create(store_state)->getResult(0);
    };
    auto emit_accumulator_store = [&](Value value, Value destination,
                                      Value destination_linear_index) {
      mlir::OperationState store_state(entry_function.getLoc(),
                                       "xla_gpu.accumulator_store");
      store_state.addOperands({value, destination, destination_linear_index});
      store_state.addTypes(destination.getType());
      return builder.create(store_state)->getResult(0);
    };
    auto emit_split_buffer_load = [&](Value source, Value vector_linear_index,
                                      Value scalar_linear_index,
                                      mlir::Type result_type) {
      mlir::OperationState load_state(entry_function.getLoc(),
                                      "xla_gpu.split_buffer_load");
      load_state.addOperands(
          {source, vector_linear_index, scalar_linear_index});
      load_state.addTypes(result_type);
      return builder.create(load_state)->getResult(0);
    };
    auto source_vector_type = [&](mlir::VectorType contraction_type,
                                  PrimitiveType source_type) {
      if (source_type == S4) {
        return mlir::VectorType::get(contraction_type.getShape(),
                                     builder.getIntegerType(4));
      }
      return source_type == F32
                 ? mlir::VectorType::get(contraction_type.getShape(),
                                         builder.getF32Type())
                 : contraction_type;
    };
    auto convert_input_vector = [&](Value value,
                                    mlir::VectorType contraction_type,
                                    PrimitiveType source_type) {
      if (source_type == F32) {
        if (contraction_type.getElementType().isF32()) {
          return value;
        }
        return mlir::arith::TruncFOp::create(builder, contraction_type, value)
            .getResult();
      }
      if (source_type == S4) {
        auto widened_type = mlir::VectorType::get(contraction_type.getShape(),
                                                  builder.getI8Type());
        Value widened =
            mlir::arith::ExtSIOp::create(builder, widened_type, value);
        return mlir::arith::SIToFPOp::create(builder, contraction_type, widened)
            .getResult();
      }
      CHECK(IsHalfPrecisionFloat(source_type) || IsFnuzFp8(source_type));
      return value;
    };
    auto unpack_s4_vector = [&](Value packed,
                                mlir::VectorType result_type) -> Value {
      const int64_t packed_elements =
          mlir::cast<mlir::VectorType>(packed.getType()).getNumElements();
      CHECK_EQ(result_type.getNumElements(), 2 * packed_elements);
      Value shift = mlir::arith::ConstantOp::create(
          builder, packed.getType(),
          mlir::DenseElementsAttr::get(
              mlir::cast<mlir::ShapedType>(packed.getType()),
              builder.getI8IntegerAttr(4)));
      Value low = mlir::arith::ShLIOp::create(builder, packed, shift);
      low = mlir::arith::ShRSIOp::create(builder, low, shift);
      Value high = mlir::arith::ShRSIOp::create(builder, packed, shift);
      llvm::SmallVector<int64_t> interleave;
      interleave.reserve(2 * packed_elements);
      for (int64_t element = 0; element < packed_elements; ++element) {
        interleave.push_back(element);
        interleave.push_back(packed_elements + element);
      }
      auto unpacked_type =
          mlir::VectorType::get(result_type.getShape(), builder.getI8Type());
      Value unpacked = mlir::vector::ShuffleOp::create(builder, unpacked_type,
                                                       low, high, interleave);
      auto f32_type =
          mlir::VectorType::get(result_type.getShape(), builder.getF32Type());
      Value converted =
          mlir::arith::SIToFPOp::create(builder, f32_type, unpacked);
      auto i32_type =
          mlir::VectorType::get(result_type.getShape(), builder.getI32Type());
      Value bits = mlir::arith::BitcastOp::create(builder, i32_type, converted);
      Value high_half_shift = mlir::arith::ConstantOp::create(
          builder, i32_type,
          mlir::DenseElementsAttr::get(i32_type,
                                       builder.getI32IntegerAttr(16)));
      bits = mlir::arith::ShRUIOp::create(builder, bits, high_half_shift);
      auto i16_type =
          mlir::VectorType::get(result_type.getShape(), builder.getI16Type());
      Value bf16_bits = mlir::arith::TruncIOp::create(builder, i16_type, bits);
      // Every signed nibble is exactly representable in BF16, so the low 16
      // FP32 mantissa bits are zero. Extracting the high half avoids LLVM's
      // generic BF16 rounding expansion on gfx942 (roughly five instructions
      // per element) while preserving the exact conversion semantics.
      return mlir::arith::BitcastOp::create(builder, result_type, bf16_bits);
    };
    auto emit_selected_input = [&](Value source, mlir::Type result_type,
                                   auto emit) -> Value {
      const std::vector<int64_t>* parameters = nullptr;
      Value fragment;
      if (source == lhs && lhs_parameter_numbers_.size() > 1) {
        parameters = &lhs_parameter_numbers_;
        fragment = lhs_concat_fragment;
      } else if (source == rhs && rhs_parameter_numbers_.size() > 1) {
        parameters = &rhs_parameter_numbers_;
        fragment = rhs_concat_fragment;
      }
      if (parameters == nullptr) {
        return emit(source);
      }

      // Branch around the actual load. Selecting a tensor argument directly
      // is not supported by XLA's tensor-flattening pass.
      std::function<Value(int64_t)> emit_branch = [&](int64_t index) -> Value {
        Value candidate = entry_function.getArgument((*parameters)[index]);
        CHECK_EQ(candidate.getType(), source.getType());
        if (index + 1 == parameters->size()) {
          return emit(candidate);
        }
        Value is_fragment = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::eq, fragment,
            IndexConstant(builder, index));
        mlir::scf::IfOp select = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{result_type}, is_fragment,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(select.thenBlock());
          mlir::scf::YieldOp::create(builder, emit(candidate));
          builder.setInsertionPointToStart(select.elseBlock());
          mlir::scf::YieldOp::create(builder, emit_branch(index + 1));
        }
        builder.setInsertionPointAfter(select);
        return select.getResult(0);
      };
      return emit_branch(0);
    };
    auto emit_input_buffer_load = [&](Value source, PrimitiveType source_type,
                                      Value source_linear_index,
                                      mlir::VectorType result_type) -> Value {
      return emit_selected_input(
          source, result_type, [&](Value selected_source) -> Value {
            if (source_type == S4) {
              CHECK_EQ(result_type.getNumElements() % 2, 0);
              const int64_t packed_elements = result_type.getNumElements() / 2;
              CHECK_LE(packed_elements, 4);
              auto packed_word_type = builder.getI32Type();
              auto load_type = mlir::VectorType::get({4}, builder.getI8Type());
              Value load_index = source_linear_index;
              Value use_high_half;
              if (packed_elements == 2) {
                const int64_t source_elements =
                    mlir::cast<mlir::RankedTensorType>(
                        selected_source.getType())
                        .getNumElements();
                Value last_full_load =
                    IndexConstant(builder, source_elements - 8);
                use_high_half = mlir::arith::CmpIOp::create(
                    builder, mlir::arith::CmpIPredicate::ugt,
                    source_linear_index, last_full_load);
                load_index = mlir::arith::SelectOp::create(
                    builder, use_high_half, last_full_load,
                    source_linear_index);
              }
              Value packed_word = emit_buffer_load(selected_source, load_index,
                                                   packed_word_type);
              Value loaded = mlir::arith::BitcastOp::create(builder, load_type,
                                                            packed_word);
              Value packed = loaded;
              if (packed_elements == 2) {
                auto packed_type =
                    mlir::VectorType::get({2}, builder.getI8Type());
                Value low = mlir::vector::ShuffleOp::create(
                    builder, packed_type, loaded, loaded,
                    llvm::ArrayRef<int64_t>{0, 1});
                Value high = mlir::vector::ShuffleOp::create(
                    builder, packed_type, loaded, loaded,
                    llvm::ArrayRef<int64_t>{2, 3});
                Value vector_condition = mlir::vector::BroadcastOp::create(
                    builder, mlir::VectorType::get({2}, builder.getI1Type()),
                    use_high_half);
                packed = mlir::arith::SelectOp::create(
                    builder, vector_condition, high, low);
              }
              return unpack_s4_vector(packed, result_type);
            }
            if (source_type == F32 && result_type.getNumElements() == 8) {
              auto source_half_type =
                  mlir::VectorType::get({4}, builder.getF32Type());
              auto result_half_type =
                  mlir::VectorType::get({4}, builder.getBF16Type());
              Value low = convert_input_vector(
                  emit_buffer_load(selected_source, source_linear_index,
                                   source_half_type),
                  result_half_type, source_type);
              Value high = convert_input_vector(
                  emit_buffer_load(selected_source,
                                   Add(builder, source_linear_index,
                                       IndexConstant(builder, 4)),
                                   source_half_type),
                  result_half_type, source_type);
              return mlir::vector::ShuffleOp::create(
                         builder, result_type, low, high,
                         llvm::ArrayRef<int64_t>{0, 1, 2, 3, 4, 5, 6, 7})
                  .getResult();
            }
            if (IsFnuzFp8(source_type)) {
              // LLVM has no native FNUZ FP8 type. Keep the overloaded ROCm
              // raw-buffer intrinsic dword-typed and reinterpret its payload
              // only after the load. This is also required for 16-byte
              // vector loads: asking ROCDL to translate vector<16xf8> makes
              // the LLVM intrinsic type translator construct an invalid
              // vector element type, while vector<16xi8> cannot be legalized
              // by AMDGPU SelectionDAG. The native load types are v2i32 and
              // v4i32 for eight and sixteen FP8 values, respectively.
              CHECK_EQ(result_type.getNumElements() % 4, 0);
              auto packed_type = mlir::VectorType::get(
                  {result_type.getNumElements() / 4}, builder.getI32Type());
              Value packed = emit_buffer_load(
                  selected_source, source_linear_index, packed_type);
              return mlir::arith::BitcastOp::create(builder, result_type,
                                                    packed)
                  .getResult();
            }
            Value loaded =
                emit_buffer_load(selected_source, source_linear_index,
                                 source_vector_type(result_type, source_type));
            return convert_input_vector(loaded, result_type, source_type);
          });
    };
    auto emit_input_split_buffer_load =
        [&](Value source, PrimitiveType source_type, Value vector_linear_index,
            Value scalar_linear_index, mlir::VectorType result_type) -> Value {
      return emit_selected_input(
          source, result_type, [&](Value selected_source) -> Value {
            if (source_type == S4) {
              CHECK_EQ(result_type.getNumElements(), 8);
              Value packed_word = emit_split_buffer_load(
                  selected_source, vector_linear_index, scalar_linear_index,
                  builder.getI32Type());
              auto packed_type =
                  mlir::VectorType::get({4}, builder.getI8Type());
              Value packed = mlir::arith::BitcastOp::create(
                  builder, packed_type, packed_word);
              return unpack_s4_vector(packed, result_type);
            }
            if (source_type == F32 && result_type.getNumElements() == 8) {
              auto source_half_type =
                  mlir::VectorType::get({4}, builder.getF32Type());
              auto result_half_type =
                  mlir::VectorType::get({4}, builder.getBF16Type());
              Value low = convert_input_vector(
                  emit_split_buffer_load(selected_source, vector_linear_index,
                                         scalar_linear_index, source_half_type),
                  result_half_type, source_type);
              Value high = convert_input_vector(
                  emit_split_buffer_load(selected_source,
                                         Add(builder, vector_linear_index,
                                             IndexConstant(builder, 4)),
                                         scalar_linear_index, source_half_type),
                  result_half_type, source_type);
              return mlir::vector::ShuffleOp::create(
                         builder, result_type, low, high,
                         llvm::ArrayRef<int64_t>{0, 1, 2, 3, 4, 5, 6, 7})
                  .getResult();
            }
            if (IsFnuzFp8(source_type)) {
              CHECK_EQ(result_type.getNumElements() % 4, 0);
              auto packed_type = mlir::VectorType::get(
                  {result_type.getNumElements() / 4}, builder.getI32Type());
              Value packed = emit_split_buffer_load(
                  selected_source, vector_linear_index, scalar_linear_index,
                  packed_type);
              return mlir::arith::BitcastOp::create(builder, result_type,
                                                    packed)
                  .getResult();
            }
            Value loaded = emit_split_buffer_load(
                selected_source, vector_linear_index, scalar_linear_index,
                source_vector_type(result_type, source_type));
            return convert_input_vector(loaded, result_type, source_type);
          });
    };
    auto emit_input_transfer_read =
        [&](Value source, PrimitiveType source_type,
            mlir::VectorType result_type, mlir::ValueRange indices,
            bool requires_linear_addressing, int64_t physical_offset,
            llvm::ArrayRef<int64_t> strides) -> Value {
      if (requires_linear_addressing || source_type == S4) {
        return emit_input_buffer_load(
            source, source_type,
            physical_linear_index(indices, strides, physical_offset),
            result_type);
      }
      return emit_selected_input(
          source, result_type, [&](Value selected_source) -> Value {
            Value loaded = mlir::vector::TransferReadOp::create(
                builder, source_vector_type(result_type, source_type),
                selected_source, indices, /*padding=*/std::nullopt,
                llvm::ArrayRef<bool>{true});
            return convert_input_vector(loaded, result_type, source_type);
          });
    };
    auto emit_input_mapped_transfer_read =
        [&](Value source, PrimitiveType source_type,
            mlir::VectorType result_type, mlir::ValueRange indices,
            mlir::AffineMap permutation_map, bool requires_linear_addressing,
            int64_t physical_offset, llvm::ArrayRef<int64_t> strides) -> Value {
      if (requires_linear_addressing || source_type == S4) {
        return emit_input_buffer_load(
            source, source_type,
            physical_linear_index(indices, strides, physical_offset),
            result_type);
      }
      return emit_selected_input(
          source, result_type, [&](Value selected_source) -> Value {
            Value loaded = mlir::vector::TransferReadOp::create(
                builder, source_vector_type(result_type, source_type),
                selected_source, indices, /*padding=*/std::nullopt,
                permutation_map, llvm::ArrayRef<bool>{true});
            return convert_input_vector(loaded, result_type, source_type);
          });
    };
    auto extract_input_element =
        [&](Value source, PrimitiveType source_type, mlir::ValueRange indices,
            bool requires_linear_addressing, int64_t physical_offset,
            llvm::ArrayRef<int64_t> strides) -> Value {
      const PrimitiveType contraction_type =
          source_type == F32
              ? (source == lhs ? lhs_element_type_ : rhs_element_type_)
              : (source_type == S4 ? BF16 : source_type);
      mlir::Type element_type =
          ContractionElementType(builder, contraction_type);
      if (requires_linear_addressing) {
        auto result_type = mlir::VectorType::get({1}, element_type);
        Value loaded = emit_input_buffer_load(
            source, source_type,
            physical_linear_index(indices, strides, physical_offset),
            result_type);
        return mlir::vector::ExtractOp::create(builder, loaded,
                                               llvm::SmallVector<int64_t>{0})
            .getResult();
      }
      return emit_selected_input(
          source, element_type, [&](Value selected_source) -> Value {
            Value element = mlir::tensor::ExtractOp::create(
                builder, selected_source, indices);
            if (source_type == F32) {
              if (contraction_type == F32) {
                return element;
              }
              return mlir::arith::TruncFOp::create(builder, element_type,
                                                   element)
                  .getResult();
            }
            if (source_type == S4) {
              Value widened = mlir::arith::ExtSIOp::create(
                  builder, builder.getI8Type(), element);
              return mlir::arith::SIToFPOp::create(builder, element_type,
                                                   widened)
                  .getResult();
            }
            CHECK(IsHalfPrecisionFloat(source_type) ||
                  IsFnuzFp8(source_type));
            return element;
          });
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
    const int64_t pipeline_stages =
        (single_buffer_lds_ || row_major_rhs_square_double_buffer) ? 1 : 2;
    const int64_t lhs_stage_elements =
        tiny_gemv_local_split
            ? stage_k_
            : block_m_ * stage_k_ +
                  (local_split_k_
                       ? block_m_ / kLocalSplitRowsPerPad *
                             kLocalSplitPadElements
                       : 0);
    const int64_t tensile_element_bytes =
        ContractionElementBytes(lhs_element_type_);
    const int64_t kTensilePadElements = 32 / tensile_element_bytes;
    const int64_t kTensileStageStrideElements =
        32 * 1024 / tensile_element_bytes;
    const int64_t kTensilePhysicalBlockElements =
        4608 / tensile_element_bytes;
    const int64_t kTensilePhysicalStageElements =
        7 * kTensilePhysicalBlockElements;
    const int64_t rhs_stage_elements =
        (tensile_double_buffer ? kTensileStageStrideElements
                               : block_n_ * stage_k_) +
        (local_split_k_ && !tiny_gemv_local_split
             ? block_n_ / kLocalSplitRowsPerPad * kLocalSplitPadElements
             : 0);
    const int64_t lhs_shared_elements =
        tensile_direct_lhs
            ? 0
            : (tensile_double_buffer && tensile_direct_rhs
                   ? kTensileStageStrideElements + kTensilePhysicalStageElements
                   : pipeline_stages * lhs_stage_elements);
    const int64_t rhs_shared_elements =
        stage_rhs_ && !tensile_direct_rhs
            ? (tensile_double_buffer
                   ? kTensileStageStrideElements + kTensilePhysicalStageElements
                   : pipeline_stages * rhs_stage_elements)
            : 0;
    // The attention-value output transform makes the MFMA row fragments
    // contiguous in the final tensor, so it can store packed rows directly.
    // Other staged epilogues still reuse the operand LDS allocation for C.
    const bool stage_output_via_lds =
        stage_output_ && !output_transpose_batch_inner_.has_value();
    const int64_t output_shared_elements =
        stage_output_via_lds ? block_m_ * block_n_ : 0;
    const int64_t local_split_shared_elements =
        local_split_k_ ? 2 * block_m_ * block_n_ : 0;
    auto lhs_shared_type = mlir::RankedTensorType::get(
        {std::max({lhs_shared_elements + rhs_shared_elements,
                   output_shared_elements, local_split_shared_elements})},
        lhs_element_type);
    Value lhs_shared = AllocateSharedOp::create(builder, lhs_shared_type);

    const int64_t tile_rows = block_m_ / atom_m;
    const int64_t tile_columns = block_n_ / atom_m;
    TF_RET_CHECK(tile_rows * tile_columns >= output_waves);
    int64_t wave_grid_rows = 1;
    int64_t wave_grid_columns = output_waves;
    int64_t best_fragment_loads = std::numeric_limits<int64_t>::max();
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
    if (row_major_rhs_square_double_buffer && block_n_ == 256) {
      // Match Triton's winning transposed-MFMA layout. Four waves split N and
      // each cover a 128x64 output slice: four M atoms by two N atoms. This
      // retains eight independent accumulators per wave while sharing each B
      // fragment across twice as many M atoms.
      wave_grid_rows = 1;
      wave_grid_columns = 4;
    } else if (tensile_wide_tile) {
      wave_grid_rows = tensile_transposed_tile ? 1 : 4;
      wave_grid_columns = tensile_transposed_tile ? 4 : 1;
    } else {
      TF_RET_CHECK(best_fragment_loads !=
                   std::numeric_limits<int64_t>::max());
    }
    TF_RET_CHECK(tile_rows % wave_grid_rows == 0);
    TF_RET_CHECK(tile_columns % wave_grid_columns == 0);

    const int64_t wave_tile_rows = tile_rows / wave_grid_rows;
    const int64_t wave_tile_columns = tile_columns / wave_grid_columns;
    const int64_t accumulators_per_wave = wave_tile_rows * wave_tile_columns;
    // Triton's wide XF32 kernel interleaves the output atoms owned by each
    // wave.  The 4x2 wave grid starts consecutive waves one MFMA atom apart,
    // while a wave's two local M/N atoms are separated by the full wave-grid
    // extent (128 rows and 64 columns respectively).  Keep these strides
    // explicit because they must agree in the operand reads and the output
    // epilogue; using the ordinary contiguous wave tile in either place
    // silently permutes the result.
    const int64_t tile_row_stride =
        atom_m * (triton_xf32_paired_lds ? wave_grid_rows : 1);
    const int64_t tile_column_stride =
        atom_m * (triton_xf32_paired_lds ? wave_grid_columns : 1);
    auto get_accumulator_index = [&](int64_t tile_row, int64_t tile_column) {
      // DirectToVgprA holds one staged-B fragment while visiting four A rows.
      // Keep those four accumulators adjacent, as Tensile does, instead of
      // separating them by the full 14-column wave tile in the AGPR file.
      return tensile_direct_lhs ? tile_column * wave_tile_rows + tile_row
                                : tile_row * wave_tile_columns + tile_column;
    };
    Value wave_row = Div(builder, output_wave_id, wave_grid_columns);
    Value wave_column = Rem(builder, output_wave_id, wave_grid_columns);
    const int64_t wave_row_stride =
        triton_xf32_paired_lds ? atom_m : wave_tile_rows * atom_m;
    const int64_t wave_column_stride =
        triton_xf32_paired_lds ? atom_m : wave_tile_columns * atom_m;
    Value wave_row_offset =
        Mul(builder, wave_row, IndexConstant(builder, wave_row_stride));
    Value wave_column_offset =
        Mul(builder, wave_column, IndexConstant(builder, wave_column_stride));
    Value wave_row_base = Add(builder, block_m_base, wave_row_offset);
    Value wave_column_base = Add(builder, block_n_base, wave_column_offset);

    llvm::SmallVector<Value> initial_accumulators(accumulators_per_wave,
                                                  zero_accumulator);
    auto scale_value_to_f32 = [&](Value value,
                                  PrimitiveType element_type) -> Value {
      if (element_type == F32) {
        return value;
      }
      if (element_type == S8) {
        return mlir::arith::SIToFPOp::create(
            builder, builder.getF32Type(), value);
      }
      return mlir::arith::ExtFOp::create(builder, builder.getF32Type(), value);
    };
    auto bounded_scale_index = [&](Value logical_index, int64_t logical_size,
                                   int64_t scale_size) -> Value {
      if (scale_size == 1) {
        return IndexConstant(builder, 0);
      }
      CHECK_EQ(scale_size, logical_size);
      Value in_bounds = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, logical_index,
          IndexConstant(builder, logical_size));
      return mlir::arith::SelectOp::create(
          builder, in_bounds, logical_index,
          IndexConstant(builder, logical_size - 1));
    };
    auto load_block_scale =
        [&](int64_t parameter_number, PrimitiveType element_type,
            llvm::ArrayRef<int64_t> dimensions,
            int64_t contracting_dimension, Value noncontracting_index,
            int64_t noncontracting_size, Value k_base) -> Value {
      CHECK_EQ(dimensions.size(), 2);
      const int64_t noncontracting_dimension = 1 - contracting_dimension;
      llvm::SmallVector<Value, 2> indices(2, IndexConstant(builder, 0));
      indices[noncontracting_dimension] = bounded_scale_index(
          noncontracting_index, noncontracting_size,
          dimensions[noncontracting_dimension]);
      indices[contracting_dimension] =
          Div(builder, k_base,
              k_ / dimensions[contracting_dimension]);
      Value value = mlir::tensor::ExtractOp::create(
          builder, entry_function.getArgument(parameter_number), indices);
      return scale_value_to_f32(value, element_type);
    };
    auto accumulate_block_scaled =
        [&](llvm::ArrayRef<Value> running,
            llvm::ArrayRef<Value> partial,
            Value k_base) -> llvm::SmallVector<Value> {
      CHECK(block_scaled_dot_);
      CHECK_EQ(running.size(), accumulators_per_wave);
      CHECK_EQ(partial.size(), accumulators_per_wave);
      llvm::SmallVector<Value> lhs_scales;
      lhs_scales.reserve(wave_tile_rows);
      for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
        llvm::SmallVector<Value> elements;
        elements.reserve(accumulator_elements);
        for (int64_t element = 0; element < accumulator_elements; ++element) {
          const int64_t element_row =
              use_mfma_32 ? (element / 4) * 8 + element % 4 : element;
          Value output_row = Add(
              builder, wave_row_base,
              Add(builder,
                  Mul(builder, lane_group, IndexConstant(builder, 4)),
                  IndexConstant(builder, tile_row * tile_row_stride + element_row)));
          elements.push_back(load_block_scale(
              lhs_scale_parameter_number_, lhs_scale_element_type_,
              lhs_scale_dimensions_, /*contracting_dimension=*/1, output_row,
              m_, k_base));
        }
        lhs_scales.push_back(mlir::vector::FromElementsOp::create(
            builder, accumulator_type, elements));
      }
      llvm::SmallVector<Value> rhs_scales;
      rhs_scales.reserve(wave_tile_columns);
      for (int64_t tile_column = 0; tile_column < wave_tile_columns;
           ++tile_column) {
        Value output_column =
            Add(builder, wave_column_base,
                Add(builder, lane_axis,
                          IndexConstant(builder, tile_column * tile_column_stride)));
        rhs_scales.push_back(load_block_scale(
            rhs_scale_parameter_number_, rhs_scale_element_type_,
            rhs_scale_dimensions_, rhs_contracting_dimension_, output_column,
            n_, k_base));
      }
      llvm::SmallVector<Value> result(running);
      for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             ++tile_column) {
          const int64_t accumulator_index =
              get_accumulator_index(tile_row, tile_column);
          Value rhs_vector = mlir::vector::BroadcastOp::create(
              builder, accumulator_type, rhs_scales[tile_column]);
          Value scale_vector = mlir::arith::MulFOp::create(
              builder, lhs_scales[tile_row], rhs_vector);
          result[accumulator_index] = mlir::math::FmaOp::create(
              builder, partial[accumulator_index], scale_vector,
              running[accumulator_index]);
        }
      }
      return result;
    };
    auto dequantize_scaled_fragment = [&](Value fragment,
                                          Value scale) -> Value {
      CHECK(dequantize_block_scales_);
      llvm::SmallVector<Value> elements;
      elements.reserve(input_fragment_elements);
      for (int64_t element = 0; element < input_fragment_elements; ++element) {
        Value fp8 = mlir::vector::ExtractOp::create(
            builder, fragment, llvm::SmallVector<int64_t>{element});
        Value fp32 = mlir::arith::ExtFOp::create(
            builder, builder.getF32Type(), fp8);
        Value scaled = mlir::arith::MulFOp::create(builder, fp32, scale);
        elements.push_back(mlir::arith::TruncFOp::create(
            builder, builder.getBF16Type(), scaled));
      }
      auto dequantized_type = mlir::VectorType::get(
          {input_fragment_elements}, builder.getBF16Type());
      return mlir::vector::FromElementsOp::create(builder, dequantized_type,
                                                   elements);
    };
    Value lower_bound = IndexConstant(builder, 0);
    Value upper_bound = IndexConstant(builder, k_);
    const int64_t padded_k =
        ((k_ + stage_k_ - 1) / stage_k_) * stage_k_;
    Value compute_upper_bound = IndexConstant(builder, padded_k);
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
      // For a batched contraction, advancing past one matrix can still alias
      // the following batch.  Move past the complete allocation so the buffer
      // descriptor suppresses speculative modulo-pipeline loads for every
      // batch, including batch zero.
      const int64_t allocation_batches = batched_gemm_ ? batch_count_ : 1;
      Value out_of_bounds = IndexConstant(
          builder, allocation_batches * std::max(m_, n_) * k_);
      return mlir::arith::SelectOp::create(builder, in_bounds, logical_k,
                                           out_of_bounds)
          .getResult();
    };
    // Keep each wide-tile VMEM operation at 16 bytes: eight BF16 values for
    // K64 or sixteen FP8 values for K128. The resulting copy counts and
    // modulo-pipeline schedule are identical.
    const int64_t load_vector_width =
        row_major_rhs_square_double_buffer && stage_k_ == 32 && block_n_ == 128
            ? 4
            : (lhs_element_type_ == F32
                   ? 4
                   : (tensile_wide_tile && is_fp8_ ? 16 : 8));
    auto load_vector_type =
        mlir::VectorType::get({load_vector_width}, lhs_element_type);
    auto lds_vector_type = mlir::VectorType::get({4}, builder.getBF16Type());
    Value zero_load_vector = mlir::arith::ConstantOp::create(
        builder, load_vector_type, builder.getZeroAttr(load_vector_type));
    Value zero_rhs_input_vector = mlir::arith::ConstantOp::create(
        builder, rhs_input_vector_type,
        builder.getZeroAttr(rhs_input_vector_type));
    Value zero_staged_input_vector = mlir::arith::ConstantOp::create(
        builder, staged_input_vector_type,
        builder.getZeroAttr(staged_input_vector_type));
    Value zero_lhs_input_element = mlir::arith::ConstantOp::create(
        builder, lhs_element_type, builder.getZeroAttr(lhs_element_type));
    Value zero_rhs_input_element = mlir::arith::ConstantOp::create(
        builder, rhs_element_type, builder.getZeroAttr(rhs_element_type));
    const bool masked_k_tail = k_ % stage_k_ != 0;
    Value load_step = IndexConstant(builder, num_warps_ * 64);
    const int64_t lhs_vectors_per_row = stage_k_ / load_vector_width;
    auto emit_lhs_register_vector_with_mask =
        [&](Value global_k, int64_t copy, bool mask_k_tail) {
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
            Add(builder, Mul(builder, tile, IndexConstant(builder, 16)), lane));
        shared_k = Mul(builder, Rem(builder, thread_id, lhs_vectors_per_row),
                       IndexConstant(builder, load_vector_width));
      }
      Value logical_row =
          local_split_k_ ? local_split_logical_row(shared_row) : shared_row;
      Value logical_k =
          (triton_vec4_lds || triton_xf32_paired_lds ||
           row_major_rhs_single_buffer ||
           row_major_rhs_square_double_buffer || local_split_k_ ||
           tensile_double_buffer)
              ? shared_k
              : SwizzleXor16(builder, shared_row, shared_k, stage_k_,
                             ContractionElementBytes(lhs_element_type_));
      if (tensile_direct_rhs || local_split_k_) {
        Value vector_linear_index = lhs_physical_linear_index(
            Add(builder, block_m_base, logical_row), logical_k);
        Value scalar_linear_index =
            Mul(builder, ReadFirstLaneIndex(builder, global_k),
                IndexConstant(builder, lhs_physical_strides_.back()));
        return emit_input_split_buffer_load(
            lhs, lhs_input_type_, vector_linear_index, scalar_linear_index,
            load_vector_type);
      }
      Value global_row = Add(builder, block_m_base, shared_row);
      Value first_k = Add(builder, global_k, logical_k);
      auto emit_unchecked_vector = [&]() -> Value {
        Value source_linear_index =
            lhs_physical_linear_index(global_row, first_k);
        return emit_input_buffer_load(lhs, lhs_input_type_,
                                      source_linear_index, load_vector_type);
      };
      if (!mask_k_tail) {
        return emit_unchecked_vector();
      }
      Value complete_vector = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ule,
          Add(builder, first_k, IndexConstant(builder, load_vector_width)),
          upper_bound);
      mlir::scf::IfOp load = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{load_vector_type}, complete_vector,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard if_guard(builder);
        builder.setInsertionPointToStart(load.thenBlock());
        mlir::scf::YieldOp::create(builder, emit_unchecked_vector());

        builder.setInsertionPointToStart(load.elseBlock());
        llvm::SmallVector<Value, 8> elements;
        elements.reserve(load_vector_width);
        for (int64_t element = 0; element < load_vector_width; ++element) {
          Value element_k =
              Add(builder, first_k, IndexConstant(builder, element));
          Value in_bounds = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, element_k,
              upper_bound);
          Value safe_k = mlir::arith::SelectOp::create(
              builder, in_bounds, element_k, lower_bound);
          llvm::SmallVector<Value, 3> indices =
              lhs_indices(global_row, safe_k);
          Value element_value = extract_input_element(
              lhs, lhs_input_type_, indices, lhs_requires_linear_addressing_,
              lhs_physical_offset_, lhs_physical_strides_);
          elements.push_back(mlir::arith::SelectOp::create(
              builder, in_bounds, element_value, zero_lhs_input_element));
        }
        Value padded = mlir::vector::FromElementsOp::create(
            builder, load_vector_type, elements);
        mlir::scf::YieldOp::create(builder, padded);
      }
      builder.setInsertionPointAfter(load);
      return load.getResult(0);
    };
    auto emit_lhs_register_vector = [&](Value global_k, int64_t copy) {
      return emit_lhs_register_vector_with_mask(global_k, copy,
                                                masked_k_tail);
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
    auto emit_rhs_register_vector_with_mask =
        [&](Value global_k, int64_t copy, bool mask_k_tail) {
      const int64_t block_threads = num_warps_ * 64;
      if (row_major_rhs_single_buffer || row_major_rhs_square_double_buffer) {
        const int64_t n_vectors = block_n_ / load_vector_width;
        Value source_k;
        if (row_major_rhs_square_double_buffer) {
          const int64_t copies_per_thread =
              block_n_ * stage_k_ / load_vector_width / block_threads;
          source_k = Add(builder,
                         Mul(builder, Div(builder, thread_id, n_vectors),
                             IndexConstant(builder, copies_per_thread)),
                         IndexConstant(builder, copy));
        } else {
          const int64_t k_pairs_per_copy = block_threads / n_vectors;
          const int64_t pair = copy / 2;
          const int64_t half = copy % 2;
          source_k =
              Add(builder,
                  Mul(builder,
                      Add(builder, Div(builder, thread_id, n_vectors),
                          IndexConstant(builder, pair * k_pairs_per_copy)),
                      IndexConstant(builder, 2)),
                  IndexConstant(builder, half));
        }
        Value source_n = Mul(builder, Rem(builder, thread_id, n_vectors),
                             IndexConstant(builder, load_vector_width));
        Value logical_k = Add(builder, global_k, source_k);
        Value in_bounds;
        if (mask_k_tail) {
          in_bounds = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, logical_k,
              upper_bound);
          logical_k = mlir::arith::SelectOp::create(builder, in_bounds,
                                                    logical_k, lower_bound);
        }
        Value source_linear_index = rhs_physical_linear_index(
            logical_k, Add(builder, block_n_base, source_n));
        Value loaded = emit_input_buffer_load(
            rhs, rhs_input_type_, source_linear_index, load_vector_type);
        if (mask_k_tail) {
          loaded = mlir::arith::SelectOp::create(builder, in_bounds, loaded,
                                                 zero_load_vector);
        }
        return loaded;
      }
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
        shared_row =
            Add(builder, IndexConstant(builder, copy * 32), physical_row);
        shared_k = Mul(builder, Rem(builder, thread_id, rhs_vectors_per_row),
                       IndexConstant(builder, load_vector_width));
      }
      Value logical_row =
          local_split_k_ ? local_split_logical_row(shared_row) : shared_row;
      Value logical_k =
          (triton_vec4_lds || triton_xf32_paired_lds || local_split_k_ ||
           tensile_double_buffer)
              ? shared_k
              : SwizzleXor16(builder, shared_row, shared_k, stage_k_,
                             ContractionElementBytes(rhs_element_type_));
      if (tensile_direct_lhs || local_split_k_) {
        Value vector_linear_index = rhs_physical_linear_index(
            logical_k, Add(builder, block_n_base, logical_row));
        Value scalar_linear_index = Mul(
            builder, ReadFirstLaneIndex(builder, tensile_physical_k(global_k)),
            IndexConstant(builder,
                          rhs_physical_strides_[rhs_contracting_dimension_]));
        if (tensile_direct_lhs && rhs_input_type_ == S4) {
          // Keep the modulo-loop payload packed. Seven i32 values replace
          // seven vector<8xbf16> values in the loop state; each dword is
          // unpacked only when its scheduled LDS write is issued.
          return emit_selected_input(
              rhs, builder.getI32Type(), [&](Value selected_source) -> Value {
                return emit_split_buffer_load(
                    selected_source, vector_linear_index, scalar_linear_index,
                    builder.getI32Type());
              });
        }
        return emit_input_split_buffer_load(
            rhs, rhs_input_type_, vector_linear_index, scalar_linear_index,
            load_vector_type);
      }
      Value global_column = Add(builder, block_n_base, shared_row);
      Value first_k = Add(builder, global_k, logical_k);
      auto emit_unchecked_vector = [&]() -> Value {
        Value source_linear_index =
            rhs_physical_linear_index(first_k, global_column);
        return emit_input_buffer_load(rhs, rhs_input_type_,
                                      source_linear_index, load_vector_type);
      };
      if (!mask_k_tail) {
        return emit_unchecked_vector();
      }
      Value complete_vector = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ule,
          Add(builder, first_k, IndexConstant(builder, load_vector_width)),
          upper_bound);
      mlir::scf::IfOp load = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{load_vector_type}, complete_vector,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard if_guard(builder);
        builder.setInsertionPointToStart(load.thenBlock());
        mlir::scf::YieldOp::create(builder, emit_unchecked_vector());

        builder.setInsertionPointToStart(load.elseBlock());
        llvm::SmallVector<Value, 8> elements;
        elements.reserve(load_vector_width);
        for (int64_t element = 0; element < load_vector_width; ++element) {
          Value element_k =
              Add(builder, first_k, IndexConstant(builder, element));
          Value in_bounds = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ult, element_k,
              upper_bound);
          Value safe_k = mlir::arith::SelectOp::create(
              builder, in_bounds, element_k, lower_bound);
          llvm::SmallVector<Value, 2> indices =
              rhs_indices(safe_k, global_column);
          Value element_value = extract_input_element(
              rhs, rhs_input_type_, indices, rhs_requires_linear_addressing_,
              rhs_physical_offset_, rhs_physical_strides_);
          elements.push_back(mlir::arith::SelectOp::create(
              builder, in_bounds, element_value, zero_rhs_input_element));
        }
        Value padded = mlir::vector::FromElementsOp::create(
            builder, load_vector_type, elements);
        mlir::scf::YieldOp::create(builder, padded);
      }
      builder.setInsertionPointAfter(load);
      return load.getResult(0);
    };
    auto emit_rhs_register_vector = [&](Value global_k, int64_t copy) {
      return emit_rhs_register_vector_with_mask(global_k, copy,
                                                masked_k_tail);
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
      if (row_major_rhs_single_buffer) {
        CHECK_GE(copy_begin, 0);
        CHECK_LE(copy_end, vectors.size());
        constexpr int64_t kLdsVectorWidth = 4;
        for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
          Value vector_index = Add(
              builder, thread_id, IndexConstant(builder, copy * block_threads));
          Value shared_row = Div(builder, vector_index, lhs_vectors_per_row);
          Value logical_k =
              Mul(builder, Rem(builder, vector_index, lhs_vectors_per_row),
                  IndexConstant(builder, load_vector_width));
          for (int64_t half = 0; half < 2; ++half) {
            llvm::SmallVector<int64_t, 4> half_mask = {
                half * kLdsVectorWidth + 0, half * kLdsVectorWidth + 1,
                half * kLdsVectorWidth + 2, half * kLdsVectorWidth + 3};
            Value fragment = mlir::vector::ShuffleOp::create(
                builder, lds_vector_type, vectors[copy], vectors[copy],
                half_mask);
            Value physical_k = SwizzleTritonMfma32(
                builder, shared_row,
                Add(builder, logical_k,
                    IndexConstant(builder, half * kLdsVectorWidth)));
            Value shared_index =
                Add(builder,
                    Mul(builder, shared_row, IndexConstant(builder, stage_k_)),
                    physical_k);
            stored =
                mlir::vector::TransferWriteOp::create(
                    builder, fragment, stored, mlir::ValueRange{shared_index},
                    llvm::ArrayRef<bool>{true})
                    .getResult();
          }
        }
        return stored;
      }
      if (triton_xf32_paired_lds) {
        CHECK_GE(copy_begin, 0);
        CHECK_LE(copy_end, vectors.size());
        CHECK_EQ(load_vector_width, 4);
        // Keep equal-K fragments from rows separated by 64 together.  Their
        // byte addresses differ by an ST64-encodable offset, so LLVM combines
        // each pair into ds_write2st64_b64 exactly as in Triton's refill.
        for (int64_t half = 0; half < 2; ++half) {
          for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
            Value vector_index = Add(
                builder, thread_id,
                IndexConstant(builder, copy * block_threads));
            Value shared_row = Div(builder, vector_index, lhs_vectors_per_row);
            Value logical_k =
                Mul(builder, Rem(builder, vector_index, lhs_vectors_per_row),
                    IndexConstant(builder, load_vector_width));
            llvm::SmallVector<int64_t, 2> half_mask = {half * 2,
                                                       half * 2 + 1};
            Value fragment = mlir::vector::ShuffleOp::create(
                builder, input_vector_type, vectors[copy], vectors[copy],
                half_mask);
            Value physical_k = SwizzleTritonXf32(
                builder, shared_row,
                Add(builder, logical_k, IndexConstant(builder, half * 2)));
            Value shared_index = Add(
                builder,
                Mul(builder, shared_row, IndexConstant(builder, stage_k_)),
                physical_k);
            stored =
                mlir::vector::TransferWriteOp::create(
                    builder, fragment, stored, mlir::ValueRange{shared_index},
                    llvm::ArrayRef<bool>{true})
                    .getResult();
          }
        }
        return stored;
      }
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
          if (lhs_element_type_ == F32 && stage_rhs_ &&
              !row_major_rhs_square_double_buffer) {
            CHECK_GE(copy_begin, 0);
            CHECK_LE(copy_end, vectors.size());
            const int64_t block_threads = num_warps_ * 64;
            Value stage_base = Mul(builder, shared_stage,
                                   IndexConstant(builder, lhs_stage_elements));
            Value stored = destination;
            if (triton_xf32_paired_lds) {
              CHECK_EQ(load_vector_width, 4);
              for (int64_t half = 0; half < 2; ++half) {
                for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
                  Value vector_index =
                      Add(builder, thread_id,
                          IndexConstant(builder, copy * block_threads));
                  Value shared_row =
                      Div(builder, vector_index, lhs_vectors_per_row);
                  Value logical_k = Mul(
                      builder, Rem(builder, vector_index, lhs_vectors_per_row),
                      IndexConstant(builder, load_vector_width));
                  llvm::SmallVector<int64_t, 2> half_mask = {half * 2,
                                                             half * 2 + 1};
                  Value fragment = mlir::vector::ShuffleOp::create(
                      builder, input_vector_type, vectors[copy], vectors[copy],
                      half_mask);
                  Value physical_k = SwizzleTritonXf32(
                      builder, shared_row,
                      Add(builder, logical_k,
                          IndexConstant(builder, half * 2)));
                  Value shared_index = Add(
                      builder, stage_base,
                      Add(builder,
                          Mul(builder, shared_row,
                              IndexConstant(builder, stage_k_)),
                          physical_k));
                  stored =
                      mlir::vector::TransferWriteOp::create(
                          builder, fragment, stored,
                          mlir::ValueRange{shared_index},
                          llvm::ArrayRef<bool>{true})
                          .getResult();
                }
              }
              return stored;
            }
            for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
              Value vector_index = Add(
                  builder, thread_id,
                  IndexConstant(builder, copy * block_threads));
              Value physical_linear =
                  Mul(builder, vector_index,
                      IndexConstant(builder, load_vector_width));
              stored = mlir::vector::TransferWriteOp::create(
                           builder, vectors[copy], stored,
                           mlir::ValueRange{
                               Add(builder, stage_base, physical_linear)},
                           llvm::ArrayRef<bool>{true})
                           .getResult();
            }
            return stored;
          }
          if (row_major_rhs_square_double_buffer) {
            CHECK_GE(copy_begin, 0);
            CHECK_LE(copy_end, vectors.size());
            constexpr int64_t kLdsVectorWidth = 4;
            const int64_t lds_vectors_per_load =
                load_vector_width / kLdsVectorWidth;
            const int64_t block_threads = num_warps_ * 64;
            Value stage_base = Mul(builder, shared_stage,
                                   IndexConstant(builder, lhs_stage_elements));
            if (row_major_rhs_wide_source_swap) {
              CHECK_EQ(copy_begin, 0);
              CHECK_EQ(copy_end, 2);
              CHECK_EQ(vectors.size(), 2);
              CHECK_EQ(load_vector_width, 8);
              auto packed_i64_type =
                  mlir::VectorType::get({1}, builder.getI64Type());
              for (int64_t half = 0; half < lds_vectors_per_load; ++half) {
                llvm::SmallVector<int64_t, 4> half_mask = {
                    half * kLdsVectorWidth + 0,
                    half * kLdsVectorWidth + 1,
                    half * kLdsVectorWidth + 2,
                    half * kLdsVectorWidth + 3};
                llvm::SmallVector<Value, 2> packed_values;
                for (int64_t copy = 0; copy < 2; ++copy) {
                  Value fragment = mlir::vector::ShuffleOp::create(
                      builder, lds_vector_type, vectors[copy], vectors[copy],
                      half_mask);
                  Value packed_vector = mlir::vector::BitCastOp::create(
                      builder, packed_i64_type, fragment);
                  packed_values.push_back(mlir::vector::ExtractOp::create(
                      builder, packed_vector, llvm::SmallVector<int64_t>{0}));
                }
                Value vector_index = thread_id;
                Value shared_row =
                    Div(builder, vector_index, lhs_vectors_per_row);
                Value logical_k = Mul(
                    builder, Rem(builder, vector_index, lhs_vectors_per_row),
                    IndexConstant(builder, load_vector_width));
                Value physical_k = SwizzleTritonVec4(
                    builder, shared_row,
                    Add(builder, logical_k,
                        IndexConstant(builder, half * kLdsVectorWidth)));
                Value shared_index =
                    Add(builder, stage_base,
                        Add(builder,
                            Mul(builder, shared_row,
                                IndexConstant(builder, stage_k_)),
                            physical_k));
                Value byte_index = mlir::arith::IndexCastOp::create(
                    builder, builder.getI32Type(),
                    Mul(builder, shared_index, IndexConstant(builder, 2)));
                mlir::LLVM::InlineAsmOp::create(
                    builder, mlir::TypeRange{},
                    mlir::ValueRange{byte_index, packed_values[0],
                                     packed_values[1]},
                    "ds_write2st64_b64 $0, $1, $2 offset1:8;\n",
                    "v,v,v,~{memory}",
                    /*has_side_effects=*/true,
                    /*is_align_stack=*/false,
                    mlir::LLVM::TailCallKind::None,
                    mlir::LLVM::AsmDialectAttr::get(
                        builder.getContext(),
                        mlir::LLVM::AsmDialect::AD_ATT),
                    /*operand_attrs=*/mlir::ArrayAttr());
              }
              return destination;
            }
            Value stored = destination;
            for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
              Value vector_index =
                  Add(builder, thread_id,
                      IndexConstant(builder, copy * block_threads));
              Value shared_row =
                  Div(builder, vector_index, lhs_vectors_per_row);
              Value logical_k =
                  Mul(builder, Rem(builder, vector_index, lhs_vectors_per_row),
                      IndexConstant(builder, load_vector_width));
              for (int64_t half = 0; half < lds_vectors_per_load; ++half) {
                llvm::SmallVector<int64_t, 4> half_mask = {
                    half * kLdsVectorWidth + 0, half * kLdsVectorWidth + 1,
                    half * kLdsVectorWidth + 2, half * kLdsVectorWidth + 3};
                Value fragment = mlir::vector::ShuffleOp::create(
                    builder, lds_vector_type, vectors[copy], vectors[copy],
                    half_mask);
                Value physical_k = SwizzleTritonMfma32(
                    builder, shared_row,
                    Add(builder, logical_k,
                        IndexConstant(builder, half * kLdsVectorWidth)),
                    stage_k_);
                Value shared_index =
                    Add(builder, stage_base,
                        Add(builder,
                            Mul(builder, shared_row,
                                IndexConstant(builder, stage_k_)),
                            physical_k));
                stored = mlir::vector::TransferWriteOp::create(
                             builder, fragment, stored,
                             mlir::ValueRange{shared_index},
                             llvm::ArrayRef<bool>{true})
                             .getResult();
              }
            }
            return stored;
          }
          if (!tensile_double_buffer) {
            return emit_lhs_register_stores(destination, vectors, copy_begin,
                                            copy_end);
          }
          CHECK_GE(copy_begin, 0);
          CHECK_LE(copy_end, vectors.size());
          Value stored = destination;
          Value stage_base =
              Mul(builder, shared_stage,
                  IndexConstant(builder, kTensileStageStrideElements));
          Value thread_offset = Add(
              builder,
              Mul(builder, thread_id,
                  IndexConstant(builder, load_vector_width)),
              Mul(builder, Div(builder, thread_id, 16),
                  IndexConstant(builder, kTensilePadElements)));
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
                  IndexConstant(builder, kTensilePadElements)));
          const int64_t word_elements =
              4 / ContractionElementBytes(lhs_element_type_);
          Value shared_index = Add(
              builder, stage_base,
              Add(builder, thread_offset,
                  IndexConstant(builder, copy * kTensilePhysicalBlockElements +
                                             word * word_elements)));
          auto word_type =
              mlir::VectorType::get({word_elements}, lhs_element_type);
          llvm::SmallVector<int64_t, 4> word_mask;
          word_mask.reserve(word_elements);
          for (int64_t element = 0; element < word_elements; ++element) {
            word_mask.push_back(word * word_elements + element);
          }
          Value word_value = mlir::vector::ShuffleOp::create(
              builder, word_type, vector, vector, word_mask);
          return mlir::vector::TransferWriteOp::create(
                     builder, word_value, destination,
                     mlir::ValueRange{shared_index}, llvm::ArrayRef<bool>{true})
              .getResult();
        };
    auto emit_rhs_register_stores = [&](Value destination,
                                        llvm::ArrayRef<Value> vectors,
                                        int64_t copy_begin, int64_t copy_end) {
      const int64_t block_threads = num_warps_ * 64;
      const int64_t rhs_shared_base = lhs_shared_elements;
      Value stored = destination;
      if (row_major_rhs_single_buffer) {
        auto pair_type = mlir::VectorType::get({2}, builder.getBF16Type());
        const int64_t n_vectors = block_n_ / load_vector_width;
        const int64_t k_pairs_per_copy = block_threads / n_vectors;
        Value source_n_base = Mul(builder, Rem(builder, thread_id, n_vectors),
                                  IndexConstant(builder, load_vector_width));
        for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
          if (copy % 2 == 0) {
            continue;
          }
          const int64_t pair = copy / 2;
          Value destination_k =
              Mul(builder,
                  Add(builder, Div(builder, thread_id, n_vectors),
                      IndexConstant(builder, pair * k_pairs_per_copy)),
                  IndexConstant(builder, 2));
          for (int64_t element = 0; element < load_vector_width; ++element) {
            Value destination_n =
                Add(builder, source_n_base, IndexConstant(builder, element));
            Value physical_k =
                row_major_rhs_single_buffer
                    ? SwizzleTritonMfma32(builder, destination_n, destination_k)
                    : SwizzleXor16(builder, destination_n, destination_k,
                                   stage_k_);
            Value shared_index =
                Add(builder, IndexConstant(builder, rhs_shared_base),
                    Add(builder,
                        Mul(builder, destination_n,
                            IndexConstant(builder, stage_k_)),
                        physical_k));
            Value low = mlir::vector::ExtractOp::create(
                builder, vectors[copy - 1],
                llvm::SmallVector<int64_t>{element});
            Value high = mlir::vector::ExtractOp::create(
                builder, vectors[copy], llvm::SmallVector<int64_t>{element});
            Value packed = mlir::vector::FromElementsOp::create(
                builder, pair_type, mlir::ValueRange{low, high});
            stored =
                mlir::vector::TransferWriteOp::create(
                    builder, packed, stored, mlir::ValueRange{shared_index},
                    llvm::ArrayRef<bool>{true})
                    .getResult();
          }
        }
        return stored;
      }
      if (triton_xf32_paired_lds) {
        CHECK_GE(copy_begin, 0);
        CHECK_LE(copy_end, vectors.size());
        CHECK_EQ(load_vector_width, 4);
        const int64_t rhs_vectors_per_row = stage_k_ / load_vector_width;
        for (int64_t half = 0; half < 2; ++half) {
          for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
            Value vector_index = Add(
                builder, thread_id,
                IndexConstant(builder, copy * block_threads));
            Value shared_row = Div(builder, vector_index, rhs_vectors_per_row);
            Value logical_k =
                Mul(builder, Rem(builder, vector_index, rhs_vectors_per_row),
                    IndexConstant(builder, load_vector_width));
            llvm::SmallVector<int64_t, 2> half_mask = {half * 2,
                                                       half * 2 + 1};
            Value fragment = mlir::vector::ShuffleOp::create(
                builder, input_vector_type, vectors[copy], vectors[copy],
                half_mask);
            Value physical_k = SwizzleTritonXf32(
                builder, shared_row,
                Add(builder, logical_k, IndexConstant(builder, half * 2)));
            Value shared_index = Add(
                builder, IndexConstant(builder, rhs_shared_base),
                Add(builder,
                    Mul(builder, shared_row, IndexConstant(builder, stage_k_)),
                    physical_k));
            stored =
                mlir::vector::TransferWriteOp::create(
                    builder, fragment, stored, mlir::ValueRange{shared_index},
                    llvm::ArrayRef<bool>{true})
                    .getResult();
          }
        }
        return stored;
      }
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
          if (rhs_element_type_ == F32 && stage_rhs_ &&
              !row_major_rhs_square_double_buffer) {
            CHECK_GE(copy_begin, 0);
            CHECK_LE(copy_end, vectors.size());
            const int64_t block_threads = num_warps_ * 64;
            const int64_t rhs_shared_base = lhs_shared_elements;
            Value stage_base = Mul(builder, shared_stage,
                                   IndexConstant(builder, rhs_stage_elements));
            Value stored = destination;
            if (triton_xf32_paired_lds) {
              CHECK_EQ(load_vector_width, 4);
              const int64_t rhs_vectors_per_row =
                  stage_k_ / load_vector_width;
              for (int64_t half = 0; half < 2; ++half) {
                for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
                  Value vector_index =
                      Add(builder, thread_id,
                          IndexConstant(builder, copy * block_threads));
                  Value shared_row =
                      Div(builder, vector_index, rhs_vectors_per_row);
                  Value logical_k = Mul(
                      builder, Rem(builder, vector_index, rhs_vectors_per_row),
                      IndexConstant(builder, load_vector_width));
                  llvm::SmallVector<int64_t, 2> half_mask = {half * 2,
                                                             half * 2 + 1};
                  Value fragment = mlir::vector::ShuffleOp::create(
                      builder, input_vector_type, vectors[copy], vectors[copy],
                      half_mask);
                  Value physical_k = SwizzleTritonXf32(
                      builder, shared_row,
                      Add(builder, logical_k,
                          IndexConstant(builder, half * 2)));
                  Value shared_index = Add(
                      builder, IndexConstant(builder, rhs_shared_base),
                      Add(builder, stage_base,
                          Add(builder,
                              Mul(builder, shared_row,
                                  IndexConstant(builder, stage_k_)),
                              physical_k)));
                  stored =
                      mlir::vector::TransferWriteOp::create(
                          builder, fragment, stored,
                          mlir::ValueRange{shared_index},
                          llvm::ArrayRef<bool>{true})
                          .getResult();
                }
              }
              return stored;
            }
            for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
              Value vector_index = Add(
                  builder, thread_id,
                  IndexConstant(builder, copy * block_threads));
              Value physical_linear =
                  Mul(builder, vector_index,
                      IndexConstant(builder, load_vector_width));
              stored = mlir::vector::TransferWriteOp::create(
                           builder, vectors[copy], stored,
                           mlir::ValueRange{Add(
                               builder, IndexConstant(builder, rhs_shared_base),
                               Add(builder, stage_base, physical_linear))},
                           llvm::ArrayRef<bool>{true})
                           .getResult();
            }
            return stored;
          }
          if (row_major_rhs_square_double_buffer) {
            CHECK_EQ(copy_begin, 0);
            const int64_t block_threads = num_warps_ * 64;
            const int64_t copies_per_thread =
                block_n_ * stage_k_ / load_vector_width / block_threads;
            CHECK_EQ(copy_end, copies_per_thread);
            CHECK_EQ(vectors.size(), copies_per_thread);
            CHECK_EQ(copies_per_thread % input_fragment_elements, 0);
            const int64_t n_vector_count = block_n_ / load_vector_width;
            const int64_t rhs_shared_base = lhs_shared_elements;
            Value stage_base = Mul(builder, shared_stage,
                                   IndexConstant(builder, rhs_stage_elements));
            Value source_n_base =
                Mul(builder, Rem(builder, thread_id, n_vector_count),
                    IndexConstant(builder, load_vector_width));
            Value stored = destination;
            for (int64_t group = 0;
                 group < copies_per_thread / input_fragment_elements;
                 ++group) {
              Value destination_k = Add(
                  builder,
                  Mul(builder, Div(builder, thread_id, n_vector_count),
                      IndexConstant(builder, copies_per_thread)),
                  IndexConstant(builder, group * input_fragment_elements));
              for (int64_t element = 0; element < load_vector_width;
                   ++element) {
                Value destination_n = Add(
                    builder, source_n_base, IndexConstant(builder, element));
                Value physical_k =
                    row_major_rhs_wide_source_swap
                        ? SwizzleTritonAmdRotatingRhs(
                              builder, destination_n, destination_k)
                        : SwizzleTritonMfma32TransposedRhs(
                              builder, destination_n, destination_k,
                              stage_k_);
                Value shared_index =
                    Add(builder, IndexConstant(builder, rhs_shared_base),
                        Add(builder, stage_base,
                            Add(builder,
                                Mul(builder, destination_n,
                                    IndexConstant(builder, stage_k_)),
                                physical_k)));
                llvm::SmallVector<Value, 4> elements;
                elements.reserve(input_fragment_elements);
                for (int64_t member = 0; member < input_fragment_elements;
                     ++member) {
                  Value vector =
                      vectors[group * input_fragment_elements + member];
                  elements.push_back(mlir::vector::ExtractOp::create(
                      builder, vector, llvm::SmallVector<int64_t>{element}));
                }
                Value packed = mlir::vector::FromElementsOp::create(
                    builder, input_vector_type, elements);
                stored =
                    mlir::vector::TransferWriteOp::create(
                        builder, packed, stored,
                        mlir::ValueRange{shared_index},
                        llvm::ArrayRef<bool>{true})
                        .getResult();
              }
            }
            return stored;
          }
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
                  IndexConstant(builder, kTensilePadElements)));
          for (int64_t copy = copy_begin; copy < copy_end; ++copy) {
            Value vector = vectors[copy];
            if (vector.getType().isInteger(32)) {
              auto packed_type =
                  mlir::VectorType::get({4}, builder.getI8Type());
              Value packed =
                  mlir::arith::BitcastOp::create(builder, packed_type, vector);
              vector = unpack_s4_vector(packed, load_vector_type);
            }
            Value shared_index =
                Add(builder, IndexConstant(builder, rhs_shared_base),
                    Add(builder, stage_base,
                        Add(builder, thread_offset,
                            IndexConstant(
                                builder,
                                copy * kTensilePhysicalBlockElements))));
            stored =
                mlir::vector::TransferWriteOp::create(
                    builder, vector, stored, mlir::ValueRange{shared_index},
                    llvm::ArrayRef<bool>{true})
                    .getResult();
          }
          return stored;
        };
    auto emit_rhs_register_store_word_at_stage = [&](Value destination,
                                                     Value vector, int64_t copy,
                                                     int64_t word,
                                                     Value shared_stage) {
      CHECK(tensile_double_buffer);
      CHECK_GE(copy, 0);
      CHECK_LT(copy, 7);
      CHECK_GE(word, 0);
      CHECK_LT(word, 4);
      const int64_t rhs_shared_base = lhs_shared_elements;
      Value stage_base =
          Mul(builder, shared_stage,
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
        store_thread_id = Add(builder,
                              Mul(builder, interleaved_row,
                                  IndexConstant(builder, rhs_vectors_per_row)),
                              Rem(builder, thread_id, rhs_vectors_per_row));
      }
      Value thread_offset = Add(builder,
                                Mul(builder, store_thread_id,
                                    IndexConstant(builder, load_vector_width)),
                                Mul(builder, Div(builder, store_thread_id, 16),
                                    IndexConstant(
                                        builder, kTensilePadElements)));
      const int64_t word_elements =
          4 / ContractionElementBytes(rhs_element_type_);
      Value shared_index = Add(
          builder, IndexConstant(builder, rhs_shared_base),
          Add(builder, stage_base,
              Add(builder, thread_offset,
                  IndexConstant(builder, copy * kTensilePhysicalBlockElements +
                                             word * word_elements))));
      auto word_type =
          mlir::VectorType::get({word_elements}, rhs_element_type);
      Value word_value;
      if (vector.getType().isInteger(32)) {
        Value packed_word = vector;
        if (word != 0) {
          packed_word = mlir::arith::ShRUIOp::create(
              builder, packed_word,
              mlir::arith::ConstantIntOp::create(builder, builder.getI32Type(),
                                                 word * 8));
        }
        Value packed_byte = mlir::arith::TruncIOp::create(
            builder, builder.getI8Type(), packed_word);
        auto packed_byte_type = mlir::VectorType::get({1}, builder.getI8Type());
        Value packed = mlir::arith::BitcastOp::create(builder, packed_byte_type,
                                                      packed_byte);
        word_value = unpack_s4_vector(packed, word_type);
      } else {
        llvm::SmallVector<int64_t, 4> word_mask;
        word_mask.reserve(word_elements);
        for (int64_t element = 0; element < word_elements; ++element) {
          word_mask.push_back(word * word_elements + element);
        }
        word_value = mlir::vector::ShuffleOp::create(builder, word_type, vector,
                                                     vector, word_mask);
      }
      return mlir::vector::TransferWriteOp::create(
                 builder, word_value, destination,
                 mlir::ValueRange{shared_index}, llvm::ArrayRef<bool>{true})
          .getResult();
    };
    auto emit_lhs_register_stage = [&](Value destination, Value global_k,
                                       Value shared_stage) {
      if (!lhs_k_contiguous_) {
        // Attention Q arrives as a fused [B,K,M] -> [B,M,K] transpose.  Load
        // the physically contiguous M vectors and scatter-transpose them into
        // the ordinary [M,K] XOR-swizzled LDS tile.  Scalar K gathers are
        // correct but waste nearly every cache-line transaction for this
        // layout.
        CHECK_EQ(block_m_ % load_vector_width, 0);
        const int64_t vectors_per_k = block_m_ / load_vector_width;
        const int64_t lhs_vectors = stage_k_ * vectors_per_k;
        mlir::scf::ForOp load_loop = mlir::scf::ForOp::create(
            builder, thread_id, IndexConstant(builder, lhs_vectors), load_step,
            mlir::ValueRange{destination},
            [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
        {
          mlir::OpBuilder::InsertionGuard load_guard(builder);
          builder.setInsertionPointToStart(load_loop.getBody());
          Value vector_index = load_loop.getInductionVar();
          Value shared_k = Div(builder, vector_index, vectors_per_k);
          Value shared_row =
              Mul(builder, Rem(builder, vector_index, vectors_per_k),
                  IndexConstant(builder, load_vector_width));
          Value global_row = Add(builder, block_m_base, shared_row);
          Value logical_k = Add(builder, global_k, shared_k);
          llvm::SmallVector<Value, 3> indices =
              lhs_indices(global_row, logical_k);
          const int64_t lhs_noncontracting_dimension = batched_gemm_ ? 1 : 0;
          mlir::AffineMap m_vector_map = mlir::AffineMap::get(
              /*dimCount=*/indices.size(), /*symbolCount=*/0,
              builder.getAffineDimExpr(lhs_noncontracting_dimension),
              builder.getContext());
          Value loaded;
          if (m_ % block_m_ == 0 || shift_m_edge) {
            loaded = emit_input_mapped_transfer_read(
                lhs, lhs_input_type_, load_vector_type, indices, m_vector_map,
                lhs_requires_linear_addressing_, lhs_physical_offset_,
                lhs_physical_strides_);
          } else {
            llvm::SmallVector<Value, 8> elements;
            elements.reserve(load_vector_width);
            for (int64_t element = 0; element < load_vector_width; ++element) {
              Value element_row =
                  Add(builder, global_row, IndexConstant(builder, element));
              Value in_bounds = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::ult, element_row,
                  IndexConstant(builder, m_));
              Value safe_row = mlir::arith::SelectOp::create(
                  builder, in_bounds, element_row,
                  IndexConstant(builder, 0));
              llvm::SmallVector<Value, 3> element_indices =
                  lhs_indices(safe_row, logical_k);
              Value element_value = extract_input_element(
                  lhs, lhs_input_type_, element_indices,
                  lhs_requires_linear_addressing_, lhs_physical_offset_,
                  lhs_physical_strides_);
              elements.push_back(mlir::arith::SelectOp::create(
                  builder, in_bounds, element_value,
                  zero_lhs_input_element));
            }
            loaded = mlir::vector::FromElementsOp::create(
                builder, load_vector_type, elements);
          }
          Value written = load_loop.getRegionIterArg(0);
          for (int64_t element = 0; element < load_vector_width; ++element) {
            Value scalar = mlir::vector::ExtractOp::create(
                builder, loaded, llvm::SmallVector<int64_t>{element});
            Value element_row =
                Add(builder, shared_row, IndexConstant(builder, element));
            Value swizzled_k = SwizzleXor16(
                builder, element_row, shared_k, stage_k_,
                ContractionElementBytes(lhs_element_type_));
            Value stage_base =
                Mul(builder, shared_stage,
                    IndexConstant(builder, block_m_ * stage_k_));
            Value row_base =
                Mul(builder, element_row, IndexConstant(builder, stage_k_));
            written = mlir::tensor::InsertOp::create(
                builder, scalar, written,
                Add(builder, stage_base, Add(builder, row_base, swizzled_k)));
          }
          mlir::scf::YieldOp::create(builder, written);
        }
        builder.setInsertionPointAfter(load_loop);
        return load_loop.getResult(0);
      }
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
        Value logical_k =
            SwizzleXor16(builder, shared_row, shared_k, stage_k_,
                         ContractionElementBytes(lhs_element_type_));
        Value global_row = Add(builder, block_m_base, shared_row);
        auto emit_unchecked_lhs_vector = [&]() -> Value {
          if (async_lhs_) {
            Value source_linear_index = lhs_physical_linear_index(
                global_row, Add(builder, global_k, logical_k));
            return emit_input_buffer_load(lhs, lhs_input_type_,
                                          source_linear_index,
                                          load_vector_type);
          }
          llvm::SmallVector<Value, 3> indices =
              lhs_indices(global_row, Add(builder, global_k, logical_k));
          return emit_input_transfer_read(
              lhs, lhs_input_type_, load_vector_type, indices,
              lhs_requires_linear_addressing_, lhs_physical_offset_,
              lhs_physical_strides_);
        };
        auto emit_lhs_vector = [&]() -> Value {
          if (!masked_k_tail) {
            return emit_unchecked_lhs_vector();
          }
          Value first_k = Add(builder, global_k, logical_k);
          Value complete_vector = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule,
              Add(builder, first_k, IndexConstant(builder, load_vector_width)),
              upper_bound);
          mlir::scf::IfOp load = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{load_vector_type}, complete_vector,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard if_guard(builder);
            builder.setInsertionPointToStart(load.thenBlock());
            mlir::scf::YieldOp::create(builder,
                                       emit_unchecked_lhs_vector());

            builder.setInsertionPointToStart(load.elseBlock());
            llvm::SmallVector<Value, 8> elements;
            elements.reserve(load_vector_width);
            for (int64_t element = 0; element < load_vector_width; ++element) {
              Value element_k = Add(builder, first_k,
                                    IndexConstant(builder, element));
              Value in_bounds = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::ult, element_k,
                  upper_bound);
              Value safe_k = mlir::arith::SelectOp::create(
                  builder, in_bounds, element_k, lower_bound);
              llvm::SmallVector<Value, 3> indices =
                  lhs_indices(global_row, safe_k);
              Value loaded = extract_input_element(
                  lhs, lhs_input_type_, indices,
                  lhs_requires_linear_addressing_, lhs_physical_offset_,
                  lhs_physical_strides_);
              elements.push_back(mlir::arith::SelectOp::create(
                  builder, in_bounds, loaded, zero_lhs_input_element));
            }
            Value padded = mlir::vector::FromElementsOp::create(
                builder, load_vector_type, elements);
            mlir::scf::YieldOp::create(builder, padded);
          }
          builder.setInsertionPointAfter(load);
          return load.getResult(0);
        };
        Value loaded;
        if (m_ % block_m_ == 0 || shift_m_edge) {
          loaded = emit_lhs_vector();
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
            mlir::scf::YieldOp::create(builder, emit_lhs_vector());

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
      constexpr int64_t kCopyBytes = 4;
      const int64_t kCopyElements =
          kCopyBytes / ContractionElementBytes(lhs_element_type_);
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
        Value logical_k = SwizzleXor16(
            builder, shared_row, shared_k, stage_k_,
            ContractionElementBytes(lhs_element_type_));
        Value destination_wave_base = Add(
            builder,
            Mul(builder, shared_stage,
                IndexConstant(builder, block_m_ * stage_k_)),
            Add(builder, direct_copy_wave_offset,
                IndexConstant(builder, copy * block_threads * kCopyElements)));
        Value source_linear_index =
            lhs_physical_linear_index(Add(builder, block_m_base, shared_row),
                                      Add(builder, global_k, logical_k));
        copied = emit_selected_input(
            lhs, destination.getType(), [&](Value selected_source) -> Value {
              mlir::OperationState copy_state(
                  entry_function.getLoc(),
                  "xla_gpu.async_copy_global_to_shared");
              copy_state.addOperands({selected_source, source_linear_index,
                                      copied, destination_wave_base});
              copy_state.addTypes(destination.getType());
              copy_state.addAttribute("copy_bytes",
                                      builder.getI32IntegerAttr(kCopyBytes));
              return builder.create(copy_state)->getResult(0);
            });
      }
      return copied;
    };

    auto emit_lhs_stage = [&](Value destination, Value global_k,
                              Value shared_stage) {
      if (lhs_element_type_ == F32 && stage_rhs_ && !masked_k_tail) {
        llvm::SmallVector<Value> vectors =
            emit_lhs_register_vectors(global_k);
        return emit_lhs_register_stores_at_stage(destination, vectors, 0,
                                                 vectors.size(), shared_stage);
      }
      if (row_major_rhs_square_double_buffer) {
        llvm::SmallVector<Value> vectors = emit_lhs_register_vectors(global_k);
        return emit_lhs_register_stores_at_stage(destination, vectors, 0,
                                                 vectors.size(), shared_stage);
      }
      const bool can_copy_direct =
          (stage_rhs_ || async_lhs_) &&
          lhs_k_contiguous_ &&
          (m_ % block_m_ == 0 || shift_m_edge) &&
          (IsHalfPrecisionFloat(lhs_input_type_) ||
           (IsFnuzFp8(lhs_input_type_) &&
            lhs_input_type_ == rhs_input_type_));
      if (can_copy_direct) {
        if (!masked_k_tail) {
          return emit_lhs_direct_stage(destination, global_k, shared_stage);
        }
        Value complete_stage = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule,
            Add(builder, global_k, IndexConstant(builder, stage_k_)),
            upper_bound);
        mlir::scf::IfOp stage = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{destination.getType()}, complete_stage,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard stage_guard(builder);
          builder.setInsertionPointToStart(stage.thenBlock());
          mlir::scf::YieldOp::create(
              builder,
              emit_lhs_direct_stage(destination, global_k, shared_stage));

          builder.setInsertionPointToStart(stage.elseBlock());
          mlir::scf::YieldOp::create(
              builder,
              emit_lhs_register_stage(destination, global_k, shared_stage));
        }
        builder.setInsertionPointAfter(stage);
        return stage.getResult(0);
      }
      return emit_lhs_register_stage(destination, global_k, shared_stage);
    };

    auto emit_rhs_lds_stage = [&](Value destination, Value global_k,
                                  Value shared_stage) {
      const int64_t rhs_shared_base = lhs_shared_elements;
      if (rhs_element_type_ == F32 && stage_rhs_ && !masked_k_tail) {
        llvm::SmallVector<Value> vectors =
            emit_rhs_register_vectors(global_k);
        return emit_rhs_register_stores_at_stage(destination, vectors, 0,
                                                 vectors.size(), shared_stage);
      }
      if (row_major_rhs_square_double_buffer) {
        llvm::SmallVector<Value> vectors = emit_rhs_register_vectors(global_k);
        return emit_rhs_register_stores_at_stage(destination, vectors, 0,
                                                 vectors.size(), shared_stage);
      }
      const bool can_copy_direct =
          rhs_k_contiguous_ &&
          (IsHalfPrecisionFloat(rhs_input_type_) ||
           (IsFnuzFp8(rhs_input_type_) &&
            lhs_input_type_ == rhs_input_type_));
      if (can_copy_direct && !masked_k_tail) {
        constexpr int64_t kCopyBytes = 4;
        const int64_t kCopyElements =
            kCopyBytes / ContractionElementBytes(rhs_element_type_);
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
          Value logical_k = SwizzleXor16(
              builder, shared_n, shared_k, stage_k_,
              ContractionElementBytes(rhs_element_type_));
          Value destination_wave_base =
              Add(builder, IndexConstant(builder, rhs_shared_base),
                  Add(builder,
                      Mul(builder, shared_stage,
                          IndexConstant(builder, block_n_ * stage_k_)),
                      Add(builder, direct_copy_wave_offset,
                          IndexConstant(
                              builder, copy * block_threads * kCopyElements))));
          Value source_linear_index =
              rhs_physical_linear_index(Add(builder, global_k, logical_k),
                                        Add(builder, block_n_base, shared_n));
          copied = emit_selected_input(
              rhs, destination.getType(), [&](Value selected_source) -> Value {
                mlir::OperationState copy_state(
                    entry_function.getLoc(),
                    "xla_gpu.async_copy_global_to_shared");
                copy_state.addOperands({selected_source, source_linear_index,
                                        copied, destination_wave_base});
                copy_state.addTypes(destination.getType());
                copy_state.addAttribute("copy_bytes",
                                        builder.getI32IntegerAttr(kCopyBytes));
                return builder.create(copy_state)->getResult(0);
              });
        }
        return copied;
      }
      if (can_copy_direct) {
        Value complete_stage = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ule,
            Add(builder, global_k, IndexConstant(builder, stage_k_)),
            upper_bound);
        mlir::scf::IfOp stage = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{destination.getType()}, complete_stage,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard stage_guard(builder);
          builder.setInsertionPointToStart(stage.thenBlock());
          constexpr int64_t kCopyBytes = 4;
          const int64_t kCopyElements =
              kCopyBytes / ContractionElementBytes(rhs_element_type_);
          const int64_t copies = block_n_ * stage_k_ / kCopyElements;
          const int64_t block_threads = num_warps_ * 64;
          const int64_t copies_per_thread = copies / block_threads;
          Value copied = destination;
          for (int64_t copy = 0; copy < copies_per_thread; ++copy) {
            Value copy_index = Add(
                builder, thread_id,
                IndexConstant(builder, copy * block_threads));
            Value physical_linear = Mul(
                builder, copy_index, IndexConstant(builder, kCopyElements));
            Value shared_n = Div(builder, physical_linear, stage_k_);
            Value shared_k = Rem(builder, physical_linear, stage_k_);
            Value logical_k = SwizzleXor16(
                builder, shared_n, shared_k, stage_k_,
                ContractionElementBytes(rhs_element_type_));
            Value destination_wave_base =
                Add(builder, IndexConstant(builder, rhs_shared_base),
                    Add(builder,
                        Mul(builder, shared_stage,
                            IndexConstant(builder, block_n_ * stage_k_)),
                        Add(builder, direct_copy_wave_offset,
                            IndexConstant(builder, copy * block_threads *
                                                       kCopyElements))));
            Value source_linear_index = rhs_physical_linear_index(
                Add(builder, global_k, logical_k),
                Add(builder, block_n_base, shared_n));
            copied = emit_selected_input(
                rhs, destination.getType(),
                [&](Value selected_source) -> Value {
                  mlir::OperationState copy_state(
                      entry_function.getLoc(),
                      "xla_gpu.async_copy_global_to_shared");
                  copy_state.addOperands({selected_source, source_linear_index,
                                          copied, destination_wave_base});
                  copy_state.addTypes(destination.getType());
                  copy_state.addAttribute(
                      "copy_bytes", builder.getI32IntegerAttr(kCopyBytes));
                  return builder.create(copy_state)->getResult(0);
                });
          }
          mlir::scf::YieldOp::create(builder, copied);

          builder.setInsertionPointToStart(stage.elseBlock());
          // Fall through to the bounded register path below. Build it in a
          // nested helper so the complete-stage branch keeps direct-to-LDS
          // copies in every hot-loop iteration except the final K tile.
          Value tail_destination = destination;
          const int64_t rhs_vectors =
              block_n_ * stage_k_ / load_vector_width;
          mlir::scf::ForOp load_loop = mlir::scf::ForOp::create(
              builder, thread_id, IndexConstant(builder, rhs_vectors),
              load_step, mlir::ValueRange{tail_destination},
              [](mlir::OpBuilder&, mlir::Location, Value,
                 mlir::ValueRange) {});
          {
            mlir::OpBuilder::InsertionGuard load_guard(builder);
            builder.setInsertionPointToStart(load_loop.getBody());
            Value vector_index = load_loop.getInductionVar();
            const int64_t vectors_per_n = stage_k_ / load_vector_width;
            Value shared_n = Div(builder, vector_index, vectors_per_n);
            Value shared_k =
                Mul(builder, Rem(builder, vector_index, vectors_per_n),
                    IndexConstant(builder, load_vector_width));
            Value first_k = Add(builder, global_k, shared_k);
            llvm::SmallVector<Value, 8> elements;
            elements.reserve(load_vector_width);
            for (int64_t element = 0; element < load_vector_width; ++element) {
              Value element_k =
                  Add(builder, first_k, IndexConstant(builder, element));
              Value in_bounds = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::ult, element_k,
                  upper_bound);
              Value safe_k = mlir::arith::SelectOp::create(
                  builder, in_bounds, element_k, lower_bound);
              llvm::SmallVector<Value, 2> indices = rhs_indices(
                  safe_k, Add(builder, block_n_base, shared_n));
              Value element_value = extract_input_element(
                  rhs, rhs_input_type_, indices,
                  rhs_requires_linear_addressing_, rhs_physical_offset_,
                  rhs_physical_strides_);
              elements.push_back(mlir::arith::SelectOp::create(
                  builder, in_bounds, element_value,
                  zero_rhs_input_element));
            }
            Value loaded = mlir::vector::FromElementsOp::create(
                builder, load_vector_type, elements);
            Value written = load_loop.getRegionIterArg(0);
            for (int64_t element = 0; element < load_vector_width; ++element) {
              Value scalar = mlir::vector::ExtractOp::create(
                  builder, loaded, llvm::SmallVector<int64_t>{element});
              Value element_k =
                  Add(builder, shared_k, IndexConstant(builder, element));
              Value swizzled_k = SwizzleXor16(
                  builder, shared_n, element_k, stage_k_,
                  ContractionElementBytes(rhs_element_type_));
              Value stage_base =
                  Mul(builder, shared_stage,
                      IndexConstant(builder, block_n_ * stage_k_));
              Value row_base =
                  Mul(builder, shared_n, IndexConstant(builder, stage_k_));
              written = mlir::tensor::InsertOp::create(
                  builder, scalar, written,
                  Add(builder, IndexConstant(builder, rhs_shared_base),
                      Add(builder, stage_base,
                          Add(builder, row_base, swizzled_k))));
            }
            mlir::scf::YieldOp::create(builder, written);
          }
          builder.setInsertionPointAfter(load_loop);
          mlir::scf::YieldOp::create(builder, load_loop.getResult(0));
        }
        builder.setInsertionPointAfter(stage);
        return stage.getResult(0);
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
          Value first_k = Add(builder, global_k, shared_k);
          auto emit_unchecked_vector = [&]() -> Value {
            llvm::SmallVector<Value, 2> indices = rhs_indices(
                first_k, Add(builder, block_n_base, shared_n));
            return emit_input_mapped_transfer_read(
                rhs, rhs_input_type_, load_vector_type, indices, k_vector_map,
                rhs_requires_linear_addressing_, rhs_physical_offset_,
                rhs_physical_strides_);
          };
          if (!masked_k_tail) {
            loaded = emit_unchecked_vector();
          } else {
            Value complete_vector = mlir::arith::CmpIOp::create(
                builder, mlir::arith::CmpIPredicate::ule,
                Add(builder, first_k,
                    IndexConstant(builder, load_vector_width)),
                upper_bound);
            mlir::scf::IfOp load = mlir::scf::IfOp::create(
                builder, mlir::TypeRange{load_vector_type}, complete_vector,
                /*withElseRegion=*/true);
            {
              mlir::OpBuilder::InsertionGuard if_guard(builder);
              builder.setInsertionPointToStart(load.thenBlock());
              mlir::scf::YieldOp::create(builder, emit_unchecked_vector());

              builder.setInsertionPointToStart(load.elseBlock());
              llvm::SmallVector<Value, 8> elements;
              elements.reserve(load_vector_width);
              for (int64_t element = 0; element < load_vector_width;
                   ++element) {
                Value element_k = Add(builder, first_k,
                                      IndexConstant(builder, element));
                Value in_bounds = mlir::arith::CmpIOp::create(
                    builder, mlir::arith::CmpIPredicate::ult, element_k,
                    upper_bound);
                Value safe_k = mlir::arith::SelectOp::create(
                    builder, in_bounds, element_k, lower_bound);
                llvm::SmallVector<Value, 2> indices = rhs_indices(
                    safe_k, Add(builder, block_n_base, shared_n));
                Value element_value = extract_input_element(
                    rhs, rhs_input_type_, indices,
                    rhs_requires_linear_addressing_, rhs_physical_offset_,
                    rhs_physical_strides_);
                elements.push_back(mlir::arith::SelectOp::create(
                    builder, in_bounds, element_value,
                    zero_rhs_input_element));
              }
              Value padded = mlir::vector::FromElementsOp::create(
                  builder, load_vector_type, elements);
              mlir::scf::YieldOp::create(builder, padded);
            }
            builder.setInsertionPointAfter(load);
            loaded = load.getResult(0);
          }
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
          llvm::SmallVector<Value, 2> indices =
              rhs_indices(Add(builder, global_k, shared_k),
                          Add(builder, block_n_base, shared_n));
          loaded = emit_input_mapped_transfer_read(
              rhs, rhs_input_type_, load_vector_type, indices, n_vector_map,
              rhs_requires_linear_addressing_, rhs_physical_offset_,
              rhs_physical_strides_);
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
              SwizzleXor16(builder, element_n, element_k, stage_k_,
                           ContractionElementBytes(rhs_element_type_));
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
          Mul(builder, lane_group,
              IndexConstant(builder, async_lhs_ ? 2 * input_fragment_elements
                                                : input_fragment_elements)),
          IndexConstant(builder, k_offset));
      mlir::VectorType rhs_vector_type =
          async_lhs_ ? staged_input_vector_type : rhs_input_vector_type;
      for (int64_t tile_column = 0; tile_column < wave_tile_columns;
           ++tile_column) {
        Value column = Add(builder, wave_column_offset,
                           Add(builder, lane_axis,
                          IndexConstant(builder, tile_column * tile_column_stride)));
        Value global_column = Add(builder, block_n_base, column);
        auto emit_unchecked_rhs_vector = [&]() -> Value {
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
              Value source_linear_index = rhs_physical_linear_index(
                  Add(builder, global_k, group_k), global_column);
              return emit_input_buffer_load(
                  rhs, rhs_input_type_, source_linear_index, rhs_vector_type);
            }
            llvm::SmallVector<Value, 2> indices =
                rhs_indices(Add(builder, global_k, group_k), global_column);
            return emit_input_mapped_transfer_read(
                rhs, rhs_input_type_, rhs_vector_type, indices, k_vector_map,
                rhs_requires_linear_addressing_, rhs_physical_offset_,
                rhs_physical_strides_);
          }
          llvm::SmallVector<Value, 8> elements;
          const int64_t vector_elements = async_lhs_
                                              ? 2 * input_fragment_elements
                                              : input_fragment_elements;
          for (int64_t element = 0; element < vector_elements; ++element) {
            Value k_index =
                Add(builder, group_k, IndexConstant(builder, element));
            llvm::SmallVector<Value, 2> indices =
                rhs_indices(Add(builder, global_k, k_index), global_column);
            elements.push_back(extract_input_element(
                rhs, rhs_input_type_, indices, rhs_requires_linear_addressing_,
                rhs_physical_offset_, rhs_physical_strides_));
          }
          return mlir::vector::FromElementsOp::create(builder, rhs_vector_type,
                                                      elements);
        };
        auto emit_rhs_vector = [&]() -> Value {
          if (!masked_k_tail) {
            return emit_unchecked_rhs_vector();
          }
          const int64_t vector_elements = async_lhs_
                                              ? 2 * input_fragment_elements
                                              : input_fragment_elements;
          Value first_k = Add(builder, global_k, group_k);
          Value complete_vector = mlir::arith::CmpIOp::create(
              builder, mlir::arith::CmpIPredicate::ule,
              Add(builder, first_k, IndexConstant(builder, vector_elements)),
              upper_bound);
          mlir::scf::IfOp load = mlir::scf::IfOp::create(
              builder, mlir::TypeRange{rhs_vector_type}, complete_vector,
              /*withElseRegion=*/true);
          {
            mlir::OpBuilder::InsertionGuard if_guard(builder);
            builder.setInsertionPointToStart(load.thenBlock());
            mlir::scf::YieldOp::create(builder,
                                       emit_unchecked_rhs_vector());

            builder.setInsertionPointToStart(load.elseBlock());
            llvm::SmallVector<Value, 8> elements;
            elements.reserve(vector_elements);
            for (int64_t element = 0; element < vector_elements; ++element) {
              Value element_k = Add(builder, first_k,
                                    IndexConstant(builder, element));
              Value in_bounds = mlir::arith::CmpIOp::create(
                  builder, mlir::arith::CmpIPredicate::ult, element_k,
                  upper_bound);
              Value safe_k = mlir::arith::SelectOp::create(
                  builder, in_bounds, element_k, lower_bound);
              llvm::SmallVector<Value, 2> indices =
                  rhs_indices(safe_k, global_column);
              Value loaded = extract_input_element(
                  rhs, rhs_input_type_, indices,
                  rhs_requires_linear_addressing_, rhs_physical_offset_,
                  rhs_physical_strides_);
              elements.push_back(mlir::arith::SelectOp::create(
                  builder, in_bounds, loaded, zero_rhs_input_element));
            }
            Value padded = mlir::vector::FromElementsOp::create(
                builder, rhs_vector_type, elements);
            mlir::scf::YieldOp::create(builder, padded);
          }
          builder.setInsertionPointAfter(load);
          return load.getResult(0);
        };
        if (n_ % block_n_ == 0 || shift_n_edge) {
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
                                                    : zero_rhs_input_vector);
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
          Add(builder,
              Mul(builder, lane_group,
                  IndexConstant(builder, 2 * input_fragment_elements)),
              Add(builder, local_split_k_offset,
                  IndexConstant(builder, k_offset)));
      if (tensile_double_buffer) {
        const int64_t kPaddedRowStride =
            2 * stage_k_ + kTensilePadElements;
        Value dynamic_offset =
            Add(builder,
                Mul(builder, lane_axis,
                    IndexConstant(builder, kPaddedRowStride)),
                group_k);
        const int64_t static_offset =
            (tile_column / 2) * kTensilePhysicalBlockElements +
            (tile_column % 2) * stage_k_;
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
                          IndexConstant(builder, tile_column * tile_column_stride)));
      Value physical_row =
          local_split_k_ ? local_split_physical_row(row) : row;
      Value physical_k = local_split_k_
                             ? group_k
                             : SwizzleXor16(
                                   builder, row, group_k, stage_k_,
                                   ContractionElementBytes(rhs_element_type_));
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
          Add(builder,
              Mul(builder, lane_group,
                  IndexConstant(builder, 2 * input_fragment_elements)),
              Add(builder, local_split_k_offset,
                  IndexConstant(builder, k_offset)));
      if (triton_xf32_paired_lds) {
        CHECK_EQ(wave_tile_columns, 2);
        fragments.resize(2 * wave_tile_columns);
        // Match Triton's native MFMA32 XF32 K order.  A lane group owns two
        // adjacent F32 values in each K4 MFMA, while consecutive MFMAs select
        // the low and high K4 halves of this K8 group.  Keeping this order in
        // the address expressions (rather than relying on an equivalent K
        // permutation) also exposes the same paired LDS-read pattern to LLVM.
        Value paired_group_k =
            Add(builder,
                Mul(builder, lane_group,
                    IndexConstant(builder, input_fragment_elements)),
                Add(builder, local_split_k_offset,
                    IndexConstant(builder, k_offset)));
        Value row = Add(builder, wave_column_offset, lane_axis);
        Value stage_base = Mul(builder, shared_stage,
                               IndexConstant(builder, block_n_ * stage_k_));
        for (int64_t mfma = 0; mfma < 2; ++mfma) {
          Value physical_k = SwizzleTritonXf32(
              builder, row,
              Add(builder, paired_group_k,
                  IndexConstant(builder, mfma * atom_k)));
          Value first_index = Add(
              builder, IndexConstant(builder, rhs_shared_base),
              Add(builder, stage_base,
                  Add(builder,
                      Mul(builder, row, IndexConstant(builder, stage_k_)),
                      physical_k)));
          Value second_index =
              Add(builder, first_index,
                  IndexConstant(builder, tile_column_stride * stage_k_));
          fragments[mfma * wave_tile_columns] =
              mlir::vector::TransferReadOp::create(
                  builder, input_vector_type, shared,
                  mlir::ValueRange{first_index}, /*padding=*/std::nullopt,
                  llvm::ArrayRef<bool>{true});
          fragments[mfma * wave_tile_columns + 1] =
              mlir::vector::TransferReadOp::create(
                  builder, input_vector_type, shared,
                  mlir::ValueRange{second_index}, /*padding=*/std::nullopt,
                  llvm::ArrayRef<bool>{true});
        }
        return fragments;
      }
      if (triton_vec4_lds) {
        CHECK_EQ(wave_tile_columns % 2, 0);
        llvm::SmallVector<Value> low_fragments(wave_tile_columns);
        llvm::SmallVector<Value> high_fragments(wave_tile_columns);
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             tile_column += 2) {
          Value row = Add(builder, wave_column_offset,
                          Add(builder, lane_axis,
                          IndexConstant(builder, tile_column * tile_column_stride)));
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
          Add(builder,
              Mul(builder, lane_group,
                  IndexConstant(builder, 2 * input_fragment_elements)),
              Add(builder, local_split_k_offset,
                  IndexConstant(builder, k_offset)));
      if (tensile_double_buffer && tensile_direct_rhs) {
        const int64_t kPaddedRowStride =
            2 * stage_k_ + kTensilePadElements;
        Value dynamic_offset =
            Add(builder,
                Mul(builder, lane_axis,
                    IndexConstant(builder, kPaddedRowStride)),
                group_k);
        const int64_t static_offset =
            (tile_row / 2) * kTensilePhysicalBlockElements +
            (tile_row % 2) * stage_k_;
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
          Add(builder, lane_axis,
              IndexConstant(builder, tile_row * tile_row_stride)));
      Value physical_row =
          local_split_k_ ? local_split_physical_row(row) : row;
      Value physical_k = local_split_k_
                             ? group_k
                             : SwizzleXor16(
                                   builder, row, group_k, stage_k_,
                                   ContractionElementBytes(lhs_element_type_));
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
          Add(builder,
              Mul(builder, lane_group,
                  IndexConstant(builder, 2 * input_fragment_elements)),
              Add(builder, local_split_k_offset,
                  IndexConstant(builder, k_offset)));
      if (triton_xf32_paired_lds) {
        CHECK_EQ(wave_tile_rows, 2);
        fragments.resize(2 * wave_tile_rows);
        Value paired_group_k =
            Add(builder,
                Mul(builder, lane_group,
                    IndexConstant(builder, input_fragment_elements)),
                Add(builder, local_split_k_offset,
                    IndexConstant(builder, k_offset)));
        Value row = Add(builder, wave_row_offset, lane_axis);
        Value stage_base = Mul(builder, shared_stage,
                               IndexConstant(builder, block_m_ * stage_k_));
        for (int64_t mfma = 0; mfma < 2; ++mfma) {
          Value physical_k = SwizzleTritonXf32(
              builder, row,
              Add(builder, paired_group_k,
                  IndexConstant(builder, mfma * atom_k)));
          Value first_index = Add(
              builder, stage_base,
              Add(builder,
                  Mul(builder, row, IndexConstant(builder, stage_k_)),
                  physical_k));
          Value second_index =
              Add(builder, first_index,
                  IndexConstant(builder, tile_row_stride * stage_k_));
          fragments[mfma * wave_tile_rows] =
              mlir::vector::TransferReadOp::create(
                  builder, input_vector_type, shared,
                  mlir::ValueRange{first_index}, /*padding=*/std::nullopt,
                  llvm::ArrayRef<bool>{true});
          fragments[mfma * wave_tile_rows + 1] =
              mlir::vector::TransferReadOp::create(
                  builder, input_vector_type, shared,
                  mlir::ValueRange{second_index}, /*padding=*/std::nullopt,
                  llvm::ArrayRef<bool>{true});
        }
        return fragments;
      }
      if (triton_vec4_lds) {
        CHECK_EQ(wave_tile_rows % 2, 0);
        llvm::SmallVector<Value> low_fragments(wave_tile_rows);
        llvm::SmallVector<Value> high_fragments(wave_tile_rows);
        for (int64_t tile_row = 0; tile_row < wave_tile_rows; tile_row += 2) {
          Value row = Add(builder, wave_row_offset,
                          Add(builder, lane_axis,
                              IndexConstant(builder,
                                            tile_row * tile_row_stride)));
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
    auto emit_row_major_lds_pair =
        [&](Value shared, Value shared_stage, Value row, int64_t shared_base,
            int64_t stage_elements,
            int64_t group) -> llvm::SmallVector<Value, 2> {
      Value logical_k =
          Add(builder, Mul(builder, lane_group, IndexConstant(builder, 4)),
              IndexConstant(builder, group * atom_k));
      Value physical_k = SwizzleTritonMfma32(builder, row, logical_k);
      Value stage_base =
          Mul(builder, shared_stage, IndexConstant(builder, stage_elements));
      Value first_index = Add(
          builder, IndexConstant(builder, shared_base),
          Add(builder, stage_base,
              Add(builder, Mul(builder, row, IndexConstant(builder, stage_k_)),
                  physical_k)));
      Value second_index = Add(builder, first_index,
                               IndexConstant(builder, compute_stage_k / 2));
      Value first = mlir::vector::TransferReadOp::create(
          builder, input_vector_type, shared, mlir::ValueRange{first_index},
          /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
      Value second = mlir::vector::TransferReadOp::create(
          builder, input_vector_type, shared, mlir::ValueRange{second_index},
          /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
      return {first, second};
    };
    auto emit_lhs_lds_fragments = [&](Value shared, Value shared_stage) {
      llvm::SmallVector<Value> fragments;
      if (row_major_rhs_single_buffer) {
        CHECK_EQ(wave_tile_rows, 1);
        const int64_t mfmas = compute_stage_k / atom_k;
        CHECK_EQ(mfmas % 2, 0);
        fragments.resize(mfmas);
        Value row = Add(builder, wave_row_offset, lane_axis);
        // Emit the K and K+64 reads next to each other so LLVM combines them
        // into one ds_read2_b64, while retaining their contiguous-K MFMA
        // consumption order in the result vector.
        for (int64_t group = 0; group < mfmas / 2; ++group) {
          llvm::SmallVector<Value, 2> pair = emit_row_major_lds_pair(
              shared, shared_stage, row, /*shared_base=*/0, lhs_stage_elements,
              group);
          for (int64_t half = 0; half < 2; ++half) {
            const int64_t mfma = group + half * mfmas / 2;
            fragments[mfma] = pair[half];
          }
        }
        return fragments;
      }
      const int64_t k_groups = compute_stage_k / (2 * atom_k);
      fragments.reserve(k_groups * wave_tile_rows *
                        (triton_xf32_paired_lds ? 2 : 1));
      for (int64_t group = 0; group < k_groups; ++group) {
        llvm::SmallVector<Value> values =
            emit_lhs_lds_group(shared, shared_stage, group * 2 * atom_k);
        fragments.append(values);
      }
      return fragments;
    };
    auto emit_rhs_lds_fragments = [&](Value shared, Value shared_stage) {
      llvm::SmallVector<Value> fragments;
      if (row_major_rhs_single_buffer) {
        CHECK_EQ(wave_tile_columns, 1);
        const int64_t mfmas = compute_stage_k / atom_k;
        CHECK_EQ(mfmas % 2, 0);
        fragments.resize(mfmas);
        Value row = Add(builder, wave_column_offset, lane_axis);
        for (int64_t group = 0; group < mfmas / 2; ++group) {
          llvm::SmallVector<Value, 2> pair = emit_row_major_lds_pair(
              shared, shared_stage, row, lhs_shared_elements,
              rhs_stage_elements, group);
          for (int64_t half = 0; half < 2; ++half) {
            const int64_t mfma = group + half * mfmas / 2;
            fragments[mfma] = pair[half];
          }
        }
        return fragments;
      }
      const int64_t k_groups = compute_stage_k / (2 * atom_k);
      fragments.reserve(k_groups * wave_tile_columns *
                        (triton_xf32_paired_lds ? 2 : 1));
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
                          IndexConstant(builder, tile_column * tile_column_stride)));
      Value vector_linear_index = rhs_physical_linear_index(
          Mul(builder, lane_group,
              IndexConstant(builder, 2 * input_fragment_elements)),
          Add(builder, block_n_base, row));
      Value scalar_linear_index =
          Mul(builder,
              ReadFirstLaneIndex(
                  builder, Add(builder, global_k,
                               IndexConstant(builder, group * 2 * atom_k))),
              IndexConstant(builder,
                            rhs_physical_strides_[rhs_contracting_dimension_]));
      if (rhs_input_type_ == S4) {
        // Carry the two K32 groups as eight packed dwords. Expanding only the
        // current group's four fragments below keeps the other group from
        // consuming sixteen VGPRs across an entire MFMA traversal.
        return emit_selected_input(
            rhs, builder.getI32Type(), [&](Value selected_source) -> Value {
              return emit_split_buffer_load(
                  selected_source, vector_linear_index, scalar_linear_index,
                  builder.getI32Type());
            });
      }
      return emit_input_split_buffer_load(
          rhs, rhs_input_type_, vector_linear_index, scalar_linear_index,
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
              Mul(builder, lane_axis, IndexConstant(builder, wave_tile_rows)),
              IndexConstant(builder, tile_row)));
      Value vector_linear_index = lhs_physical_linear_index(
          Add(builder, block_m_base, row),
          Mul(builder, lane_group,
              IndexConstant(builder, 2 * input_fragment_elements)));
      Value scalar_linear_index =
          Mul(builder,
              ReadFirstLaneIndex(
                  builder, Add(builder, tensile_physical_k(global_k),
                               IndexConstant(builder, group * 2 * atom_k))),
              IndexConstant(builder, lhs_physical_strides_.back()));
      return emit_input_split_buffer_load(
          lhs, lhs_input_type_, vector_linear_index, scalar_linear_index,
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

    if (tiny_gemv_local_split) {
      TF_RET_CHECK(wave_tile_rows == 1 && wave_tile_columns == 1 &&
                   accumulators_per_wave == 1 && accumulator_elements == 4 &&
                   atom_k == 16 &&
                   (compute_stage_k == 64 || compute_stage_k == 128) &&
                   n_ % block_n_ == 0 && k_ % stage_k_ == 0 &&
                   k_ >= 2 * stage_k_ && !block_scaled_dot_);

      // The ordinary staged loaders already provide bounded M-tail handling
      // and the same XOR-swizzled LDS representation used by Fly GEMM.  Read
      // one K partition from each K256 stage per wave. The source swap makes
      // each surviving lane own four adjacent N columns, so the final GEMV
      // store is one dwordx2 rather than four scattered scalar stores.
      const int64_t fragment_groups = compute_stage_k / (2 * atom_k);
      auto emit_tiny_fragments = [&](Value shared, Value shared_stage,
                                     bool lhs_operand) {
        llvm::SmallVector<Value> fragments;
        fragments.reserve(fragment_groups);
        for (int64_t group = 0; group < fragment_groups; ++group) {
          Value row = lhs_operand ? IndexConstant(builder, 0) : lane_axis;
          Value logical_k =
              Add(builder, local_split_k_offset,
                  Add(builder,
                      Mul(builder, lane_group,
                          IndexConstant(builder,
                                        2 * input_fragment_elements)),
                      IndexConstant(builder, group * 2 * atom_k)));
          const int64_t operand_stage_elements =
              lhs_operand ? lhs_stage_elements : rhs_stage_elements;
          const int64_t operand_shared_base =
              lhs_operand ? 0 : lhs_shared_elements;
          Value physical_k = SwizzleXor16(
              builder, row, logical_k, stage_k_,
              ContractionElementBytes(lhs_operand ? lhs_element_type_
                                                  : rhs_element_type_));
          Value shared_index = Add(
              builder, IndexConstant(builder, operand_shared_base),
              Add(builder,
                  Mul(builder, shared_stage,
                      IndexConstant(builder, operand_stage_elements)),
                  Add(builder,
                      Mul(builder, row, IndexConstant(builder, stage_k_)),
                      physical_k)));
          fragments.push_back(mlir::vector::TransferReadOp::create(
              builder, staged_input_vector_type, shared,
              mlir::ValueRange{shared_index}, /*padding=*/std::nullopt,
              llvm::ArrayRef<bool>{true}));
        }
        return fragments;
      };
      auto emit_tiny_compute = [&](llvm::ArrayRef<Value> lhs_fragments,
                                   llvm::ArrayRef<Value> rhs_fragments,
                                   Value accumulator) -> Value {
        CHECK_EQ(lhs_fragments.size(), fragment_groups);
        CHECK_EQ(rhs_fragments.size(), fragment_groups);
        for (int64_t group = 0; group < fragment_groups; ++group) {
          for (int64_t mfma = 0; mfma < 2; ++mfma) {
            llvm::SmallVector<int64_t, 4> fragment_mask;
            fragment_mask.reserve(input_fragment_elements);
            for (int64_t element = 0; element < input_fragment_elements;
                 ++element) {
              fragment_mask.push_back(mfma * input_fragment_elements +
                                      element);
            }
            Value lhs_mfma = mlir::vector::ShuffleOp::create(
                builder, input_vector_type, lhs_fragments[group],
                lhs_fragments[group], fragment_mask);
            Value rhs_mfma = mlir::vector::ShuffleOp::create(
                builder, input_vector_type, rhs_fragments[group],
                rhs_fragments[group], fragment_mask);
            mlir::OperationState mma_state(entry_function.getLoc(),
                                           "fly.mma_atom_call_ssa");
            mma_state.addOperands(
                {mma_atom, rhs_mfma, lhs_mfma, accumulator});
            mma_state.addTypes(accumulator_type);
            accumulator = builder.create(mma_state)->getResult(0);
          }
        }
        return accumulator;
      };
      // The N-contiguous source arrives as adjacent 16-byte rows of a Kx8
      // tile. Keep the VMEM read and the register-transpose/LDS-write
      // phases separate so the hot loop can hide those reads under current
      // tile MFMAs, matching FlyDSL's two-stage pipeline.
      constexpr int64_t kNVector = 8;
      const int64_t kKVector = 8 / num_warps_;
      constexpr int64_t kNTiles = 2;
      auto rhs_stage_vector_type =
          mlir::VectorType::get({kKVector}, rhs_element_type);
      auto emit_tiny_rhs_loads = [&](Value global_k) {
        CHECK_EQ(block_n_, kNVector * kNTiles);
        CHECK_EQ(stage_k_ / kKVector * kNTiles, num_warps_ * 64);
        Value n_tile = Rem(builder, thread_id, kNTiles);
        Value shared_k =
            Mul(builder, Div(builder, thread_id, kNTiles),
                IndexConstant(builder, kKVector));
        Value source_n =
            Add(builder, block_n_base,
                Mul(builder, n_tile, IndexConstant(builder, kNVector)));
        llvm::SmallVector<Value> source_rows;
        source_rows.reserve(kKVector);
        for (int64_t k_element = 0; k_element < kKVector; ++k_element) {
          Value source_index = rhs_physical_linear_index(
              Add(builder, global_k,
                  Add(builder, shared_k,
                      IndexConstant(builder, k_element))),
              source_n);
          source_rows.push_back(emit_input_buffer_load(
              rhs, rhs_input_type_, source_index, load_vector_type));
        }
        return source_rows;
      };
      auto emit_tiny_rhs_stores =
          [&](Value destination, llvm::ArrayRef<Value> source_rows,
              Value shared_stage) -> Value {
        CHECK_EQ(source_rows.size(), kKVector);
        Value n_tile = Rem(builder, thread_id, kNTiles);
        Value shared_k =
            Mul(builder, Div(builder, thread_id, kNTiles),
                IndexConstant(builder, kKVector));
        Value stored = destination;
        const int64_t rhs_shared_base = lhs_shared_elements;
        Value stage_base =
            Mul(builder, shared_stage,
                IndexConstant(builder, rhs_stage_elements));
        for (int64_t n_element = 0; n_element < kNVector; ++n_element) {
          llvm::SmallVector<Value> elements;
          elements.reserve(kKVector);
          for (int64_t k_element = 0; k_element < kKVector; ++k_element) {
            elements.push_back(mlir::vector::ExtractOp::create(
                builder, source_rows[k_element],
                llvm::SmallVector<int64_t>{n_element}));
          }
          Value packed = mlir::vector::FromElementsOp::create(
              builder, rhs_stage_vector_type, elements);
          Value shared_n =
              Add(builder,
                  Mul(builder, n_tile, IndexConstant(builder, kNVector)),
                  IndexConstant(builder, n_element));
          Value physical_k = SwizzleXor16(
              builder, shared_n, shared_k, stage_k_,
              ContractionElementBytes(rhs_element_type_));
          Value shared_index = Add(
              builder, IndexConstant(builder, rhs_shared_base),
              Add(builder, stage_base,
                  Add(builder,
                      Mul(builder, shared_n,
                          IndexConstant(builder, stage_k_)),
                      physical_k)));
          stored = mlir::vector::TransferWriteOp::create(
                       builder, packed, stored,
                       mlir::ValueRange{shared_index},
                       llvm::ArrayRef<bool>{true})
                       .getResult();
        }
        return stored;
      };
      auto emit_tiny_rhs_stage = [&](Value destination, Value global_k,
                                     Value shared_stage) -> Value {
        if (rhs_k_contiguous_) {
          return emit_rhs_lds_stage(destination, global_k, shared_stage);
        }
        llvm::SmallVector<Value> loaded = emit_tiny_rhs_loads(global_k);
        return emit_tiny_rhs_stores(destination, loaded, shared_stage);
      };
      auto emit_tiny_lhs_load = [&](Value global_k) -> Value {
        constexpr int64_t kActiveThreads = 256 / 8;
        Value is_active = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, thread_id,
            IndexConstant(builder, kActiveThreads));
        mlir::scf::IfOp load = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{load_vector_type}, is_active,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard load_guard(builder);
          builder.setInsertionPointToStart(load.thenBlock());
          Value shared_k =
              Mul(builder, thread_id, IndexConstant(builder, 8));
          Value source_index = lhs_physical_linear_index(
              block_m_base, Add(builder, global_k, shared_k));
          Value loaded = emit_input_buffer_load(
              lhs, lhs_input_type_, source_index, load_vector_type);
          mlir::scf::YieldOp::create(builder, loaded);
          builder.setInsertionPointToStart(load.elseBlock());
          mlir::scf::YieldOp::create(builder, zero_load_vector);
        }
        builder.setInsertionPointAfter(load);
        return load.getResult(0);
      };
      auto emit_tiny_lhs_store = [&](Value destination, Value loaded,
                                     Value shared_stage) -> Value {
        constexpr int64_t kActiveThreads = 256 / 8;
        Value is_active = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, thread_id,
            IndexConstant(builder, kActiveThreads));
        mlir::scf::IfOp store = mlir::scf::IfOp::create(
            builder, mlir::TypeRange{lhs_shared_type}, is_active,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard store_guard(builder);
          builder.setInsertionPointToStart(store.thenBlock());
          Value shared_k =
              Mul(builder, thread_id, IndexConstant(builder, 8));
          Value shared_index = Add(
              builder,
              Mul(builder, shared_stage,
                  IndexConstant(builder, lhs_stage_elements)),
              shared_k);
          Value stored = mlir::vector::TransferWriteOp::create(
                             builder, loaded, destination,
                             mlir::ValueRange{shared_index},
                             llvm::ArrayRef<bool>{true})
                             .getResult();
          mlir::scf::YieldOp::create(builder, stored);
          builder.setInsertionPointToStart(store.elseBlock());
          mlir::scf::YieldOp::create(builder, destination);
        }
        builder.setInsertionPointAfter(store);
        return store.getResult(0);
      };
      auto emit_tiny_lhs_stage = [&](Value destination, Value global_k,
                                     Value shared_stage) -> Value {
        Value loaded = emit_tiny_lhs_load(global_k);
        return emit_tiny_lhs_store(destination, loaded, shared_stage);
      };

      Value initial_shared = lhs_shared;
      initial_shared =
          emit_tiny_rhs_stage(initial_shared, lower_bound, lower_bound);
      initial_shared =
          emit_tiny_lhs_stage(initial_shared, lower_bound, lower_bound);
      emit_wait_vmcnt(0);
      initial_shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                initial_shared)
              .getResult(0);

      mlir::scf::ForOp gemv_loop = mlir::scf::ForOp::create(
          builder, lower_bound, IndexConstant(builder, k_ - stage_k_), step,
          mlir::ValueRange{zero_accumulator, initial_shared},
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard loop_guard(builder);
        builder.setInsertionPointToStart(gemv_loop.getBody());
        Value k_base = gemv_loop.getInductionVar();
        Value current_stage =
            Rem(builder, Div(builder, k_base, stage_k_), 2);
        Value next_stage = Rem(
            builder, Add(builder, current_stage, IndexConstant(builder, 1)),
            2);
        Value staged_shared = gemv_loop.getRegionIterArg(1);
        llvm::SmallVector<Value> lhs_fragments =
            emit_tiny_fragments(staged_shared, current_stage,
                                /*lhs_operand=*/true);
        llvm::SmallVector<Value> rhs_fragments =
            emit_tiny_fragments(staged_shared, current_stage,
                                /*lhs_operand=*/false);

        Value next_k = Add(builder, k_base, step);
        Value refilled_shared = staged_shared;
        llvm::SmallVector<Value> next_rhs;
        if (rhs_k_contiguous_) {
          refilled_shared =
              emit_rhs_lds_stage(refilled_shared, next_k, next_stage);
        } else {
          next_rhs = emit_tiny_rhs_loads(next_k);
        }
        Value next_lhs = emit_tiny_lhs_load(next_k);
        Value next_accumulator = emit_tiny_compute(
            lhs_fragments, rhs_fragments, gemv_loop.getRegionIterArg(0));
        if (!rhs_k_contiguous_) {
          refilled_shared =
              emit_tiny_rhs_stores(refilled_shared, next_rhs, next_stage);
        }
        refilled_shared =
            emit_tiny_lhs_store(refilled_shared, next_lhs, next_stage);
        emit_wait_vmcnt(0);
        refilled_shared =
            SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                  refilled_shared)
                .getResult(0);
        mlir::scf::YieldOp::create(
            builder, mlir::ValueRange{next_accumulator, refilled_shared});
      }
      builder.setInsertionPointAfter(gemv_loop);

      Value tail_stage = IndexConstant(
          builder, (k_ / stage_k_ - 1) % 2);
      llvm::SmallVector<Value> tail_lhs = emit_tiny_fragments(
          gemv_loop.getResult(1), tail_stage, /*lhs_operand=*/true);
      llvm::SmallVector<Value> tail_rhs = emit_tiny_fragments(
          gemv_loop.getResult(1), tail_stage, /*lhs_operand=*/false);
      Value final_accumulator = emit_tiny_compute(
          tail_lhs, tail_rhs, gemv_loop.getResult(0));

      // Every upper wave publishes its K partial. The matching lane in wave
      // zero adds all partners, and only the four lane-axis-zero lanes write
      // the valid M=1 output row (four adjacent columns each).
      Value reduction_shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                gemv_loop.getResult(1))
              .getResult(0);
      auto scratch_vector_type = mlir::VectorType::get(
          {2 * accumulator_elements}, lhs_element_type);
      Value scratch_index =
          Mul(builder,
              Add(builder,
                  Mul(builder, wave_id, IndexConstant(builder, 64)), lane_id),
              IndexConstant(builder, 2 * accumulator_elements));
      Value is_upper_wave = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ugt, wave_id,
          IndexConstant(builder, 0));
      Value is_output_lane = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, lane_axis,
          IndexConstant(builder, 0));
      Value should_spill = mlir::arith::AndIOp::create(
          builder, is_upper_wave, is_output_lane);
      mlir::scf::IfOp spill = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{lhs_shared_type}, should_spill,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard spill_guard(builder);
        builder.setInsertionPointToStart(spill.thenBlock());
        Value partial_bits = mlir::vector::BitCastOp::create(
            builder, scratch_vector_type, final_accumulator);
        Value spilled = mlir::vector::TransferWriteOp::create(
                            builder, partial_bits, reduction_shared,
                            mlir::ValueRange{scratch_index},
                            llvm::ArrayRef<bool>{true})
                            .getResult();
        mlir::scf::YieldOp::create(builder, spilled);
        builder.setInsertionPointToStart(spill.elseBlock());
        mlir::scf::YieldOp::create(builder, reduction_shared);
      }
      builder.setInsertionPointAfter(spill);
      reduction_shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                spill.getResult(0))
              .getResult(0);

      Value is_lower_wave = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, wave_id,
          IndexConstant(builder, 0));
      Value should_store = mlir::arith::AndIOp::create(
          builder, is_lower_wave, is_output_lane);
      mlir::scf::IfOp store = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{output.getType()}, should_store,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard store_guard(builder);
        builder.setInsertionPointToStart(store.thenBlock());
        Value reduced = final_accumulator;
        for (int64_t partner_wave = 1; partner_wave < num_warps_;
             ++partner_wave) {
          Value partner_index = Mul(
              builder,
              Add(builder, IndexConstant(builder, partner_wave * 64), lane_id),
              IndexConstant(builder, 2 * accumulator_elements));
          Value partner_bits = mlir::vector::TransferReadOp::create(
              builder, scratch_vector_type, reduction_shared,
              mlir::ValueRange{partner_index}, /*padding=*/std::nullopt,
              llvm::ArrayRef<bool>{true});
          Value partner = mlir::vector::BitCastOp::create(
              builder, accumulator_type, partner_bits);
          reduced = mlir::arith::AddFOp::create(builder, reduced, partner);
        }
        if (is_scaled_dot_ && !block_scaled_dot_) {
          Value scale_vector = mlir::vector::BroadcastOp::create(
              builder, accumulator_type, output_scale);
          reduced = mlir::arith::MulFOp::create(builder, reduced,
                                                scale_vector);
        }
        Value output_row = wave_row_base;
        Value output_column = Add(
            builder, wave_column_base,
            Mul(builder, lane_group,
                IndexConstant(builder, accumulator_elements)));
        for (int64_t step_index = 0; step_index < epilogue_steps_.size();
             ++step_index) {
          Value operand;
          const EpilogueStep& epilogue_step = epilogue_steps_[step_index];
          if (epilogue_step.opcode != HloOpcode::kNegate) {
            if (epilogue_step.broadcast_dimension ==
                epilogue_column_dimension_) {
              llvm::SmallVector<Value, 4> elements;
              elements.reserve(accumulator_elements);
              for (int64_t element = 0; element < accumulator_elements;
                   ++element) {
                elements.push_back(load_epilogue_element(
                    step_index, output_row,
                    Add(builder, output_column,
                        IndexConstant(builder, element))));
              }
              operand = mlir::vector::FromElementsOp::create(
                  builder, accumulator_type, elements);
            } else {
              operand = load_epilogue_element(step_index, output_row,
                                               output_column);
            }
          }
          reduced = apply_epilogue_step(reduced, step_index, operand);
        }
        mlir::Type output_type =
            mlir::cast<mlir::RankedTensorType>(output.getType())
                .getElementType();
        TF_RET_CHECK(output_type.isBF16() || output_type.isF16() ||
                     output_type.isF32());
        auto packed_output_type = mlir::VectorType::get(
            {accumulator_elements}, output_type);
        Value packed = output_type.isF32()
                           ? reduced
                           : mlir::arith::TruncFOp::create(
                                 builder, packed_output_type, reduced)
                                 .getResult();
        Value stored_output = mlir::vector::TransferWriteOp::create(
                                  builder, packed, output,
                                  output_indices(output_row, output_column),
                                  llvm::ArrayRef<bool>{true})
                                  .getResult();
        mlir::scf::YieldOp::create(builder, stored_output);
        builder.setInsertionPointToStart(store.elseBlock());
        mlir::scf::YieldOp::create(builder, output);
      }
      builder.setInsertionPointAfter(store);
      mlir::func::ReturnOp::create(builder, store.getResult(0));
      return absl::OkStatus();
    }

    // Match FlyDSL's two-stage HGEMM pipeline: retain the current RHS tile in
    // registers while issuing the next tile's VMEM loads. Larger K tiles would
    // double too many live fragments, so leave those to the streaming path.
    const bool prefetch_rhs = prefetch_rhs_ && !stage_rhs_;
    const bool load_rhs_on_demand =
        (single_buffer_lds_ && block_m_ == 128 && block_n_ == 128) ||
        row_major_rhs_single_buffer || row_major_rhs_square_double_buffer ||
        (xf32_wide_single_buffer && schedule_instructions_);
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
    // LLVM's generic SCF backedge coalescer is unable to assign every wide
    // kernel VGPR bank in place. Carry the direct operand, which removes the
    // most expensive boundary reload and remains spill-free with the 224-AGPR
    // accumulator cap. Recreate the staged payload and LDS fragment group at
    // each two-tile boundary; carrying either one makes large-K kernels spill.
    constexpr bool kCarryWideDirectAcrossLoop = true;
    constexpr bool kCarryWideStagedPayloadAcrossLoop = false;
    constexpr bool kCarryWidePreloadedGroupAcrossLoop = false;
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
    // The row-major-RHS loop issues the next A loads before its opening
    // barrier, exactly as Triton does. That barrier also closes the initial
    // LDS fill, so emitting another one here only serializes the launch.
    if (!row_major_rhs_single_buffer) {
      initial_shared =
          SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                initial_shared)
              .getResult(0);
    }
    if (preload_lds_fragments_ && !load_rhs_on_demand &&
        (!tensile_direct_operand || kCarryWideDirectAcrossLoop)) {
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
    if ((tensile_wide_tile && kCarryWideStagedPayloadAcrossLoop) ||
        local_split_k_) {
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
    // Do not carry the fourteen group-0 LDS fragments through the outer SCF
    // backedge. LLVM cannot currently coalesce those 56 VGPRs with the
    // progressively consumed/reloaded fragment bank and spills the full hot
    // loop for large K. The two unrolled halves still pipeline this bank
    // between them; only the outer-loop boundary reloads it from LDS.
    if (tensile_direct_operand && kCarryWidePreloadedGroupAcrossLoop) {
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
      llvm::SmallVector<Value> lhs_dequant_scales;
      llvm::SmallVector<Value> rhs_dequant_scales;
      if (dequantize_block_scales_) {
        lhs_dequant_scales.reserve(wave_tile_rows);
        for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
          Value output_row = Add(
              builder, wave_row_base,
              Add(builder, lane_axis,
                  IndexConstant(builder, tile_row * tile_row_stride)));
          lhs_dequant_scales.push_back(load_block_scale(
              lhs_scale_parameter_number_, lhs_scale_element_type_,
              lhs_scale_dimensions_, /*contracting_dimension=*/1, output_row,
              m_, k_base));
        }
        rhs_dequant_scales.reserve(wave_tile_columns);
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             ++tile_column) {
          Value output_column = Add(
              builder, wave_column_base,
              Add(builder, lane_axis,
                          IndexConstant(builder, tile_column * tile_column_stride)));
          rhs_dequant_scales.push_back(load_block_scale(
              rhs_scale_parameter_number_, rhs_scale_element_type_,
              rhs_scale_dimensions_, rhs_contracting_dimension_,
              output_column, n_, k_base));
        }
      }
      const int64_t compute_k_step = paired_mfma ? 2 * atom_k : atom_k;
      for (int64_t k_offset = 0; k_offset < stage_k_;
           k_offset += compute_k_step) {
        Value group_k = Add(
            builder,
            Mul(builder, lane_group,
                IndexConstant(builder, paired_mfma ? 2 * input_fragment_elements
                                                   : input_fragment_elements)),
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
                                  IndexConstant(builder,
                                                tile_row * tile_row_stride)));
              Value swizzled_group_k =
                  SwizzleXor16(builder, row, group_k, stage_k_,
                               ContractionElementBytes(lhs_element_type_));
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

        if (dequantize_block_scales_) {
          for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
            lhs_vectors[tile_row] = dequantize_scaled_fragment(
                lhs_vectors[tile_row], lhs_dequant_scales[tile_row]);
          }
          for (int64_t tile_column = 0; tile_column < wave_tile_columns;
               ++tile_column) {
            rhs_vectors[tile_column] = dequantize_scaled_fragment(
                rhs_vectors[tile_column], rhs_dequant_scales[tile_column]);
          }
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
                const int64_t fragment_base =
                    mfma * input_fragment_elements;
                llvm::SmallVector<int64_t> fragment_mask;
                fragment_mask.reserve(input_fragment_elements);
                for (int64_t element = 0; element < input_fragment_elements;
                     ++element) {
                  fragment_mask.push_back(fragment_base + element);
                }
                lhs_fragment = mlir::vector::ShuffleOp::create(
                    builder, input_vector_type, lhs_fragment, lhs_fragment,
                    fragment_mask);
                rhs_fragment = mlir::vector::ShuffleOp::create(
                    builder, input_vector_type, rhs_fragment, rhs_fragment,
                    fragment_mask);
              }
              if (is_fp8_ && !dequantize_block_scales_) {
                auto packed_type = mlir::VectorType::get(
                    {input_fragment_elements}, builder.getI8Type());
                lhs_fragment = mlir::vector::BitCastOp::create(
                    builder, packed_type, lhs_fragment);
                rhs_fragment = mlir::vector::BitCastOp::create(
                    builder, packed_type, rhs_fragment);
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
      if (row_major_rhs_single_buffer) {
        CHECK_EQ(wave_tile_rows, 1);
        CHECK_EQ(wave_tile_columns, 1);
        CHECK_EQ(lhs_fragments.size(), compute_stage_k / atom_k);
        CHECK_EQ(rhs_fragments.size(), compute_stage_k / atom_k);
        for (int64_t mfma = 0; mfma < lhs_fragments.size(); ++mfma) {
          before_mfma(mfma_index++);
          mlir::OperationState mma_state(entry_function.getLoc(),
                                         "fly.mma_atom_call_ssa");
          mma_state.addOperands({mma_atom, lhs_fragments[mfma],
                                 rhs_fragments[mfma], next_accumulators[0]});
          mma_state.addTypes(accumulator_type);
          next_accumulators[0] = builder.create(mma_state)->getResult(0);
        }
        return next_accumulators;
      }
      if (xf32_single_buffer &&
          (schedule_instructions_ || triton_xf32_paired_lds)) {
        // Keep four independent MFMA32 accumulator chains in flight before
        // returning to the second K half of a packed LDS fragment. Triton's
        // winning gfx942 schedule uses the same round-robin traversal under
        // elevated wave priority. Compiler-only barriers make that order
        // survive LLVM's generic machine scheduler.
        if (schedule_instructions_) {
          mlir::ROCDL::SetPrioOp::create(builder, /*priority=*/1);
        }
        constexpr int64_t kMfmasPerFragment = 2;
        for (int64_t group = 0; group < k_groups; ++group) {
          for (int64_t mfma = 0; mfma < kMfmasPerFragment; ++mfma) {
            for (int64_t tile_column = 0; tile_column < wave_tile_columns;
                 ++tile_column) {
              Value rhs_fragment =
                  triton_xf32_paired_lds
                      ? rhs_fragments[(group * kMfmasPerFragment + mfma) *
                                          wave_tile_columns +
                                      tile_column]
                      : rhs_fragments[group * wave_tile_columns + tile_column];
              for (int64_t tile_row = 0; tile_row < wave_tile_rows;
                   ++tile_row) {
                Value lhs_fragment =
                    triton_xf32_paired_lds
                        ? lhs_fragments[(group * kMfmasPerFragment + mfma) *
                                            wave_tile_rows +
                                        tile_row]
                        : lhs_fragments[group * wave_tile_rows + tile_row];
                const int64_t accumulator_index =
                    get_accumulator_index(tile_row, tile_column);
                before_mfma(mfma_index++);
                Value lhs_mfma = lhs_fragment;
                Value rhs_mfma = rhs_fragment;
                if (!triton_xf32_paired_lds) {
                  const int64_t fragment_base =
                      mfma * input_fragment_elements;
                  llvm::SmallVector<int64_t> fragment_mask;
                  fragment_mask.reserve(input_fragment_elements);
                  for (int64_t element = 0;
                       element < input_fragment_elements; ++element) {
                    fragment_mask.push_back(fragment_base + element);
                  }
                  lhs_mfma = mlir::vector::ShuffleOp::create(
                      builder, input_vector_type, lhs_fragment, lhs_fragment,
                      fragment_mask);
                  rhs_mfma = mlir::vector::ShuffleOp::create(
                      builder, input_vector_type, rhs_fragment, rhs_fragment,
                      fragment_mask);
                }
                if (schedule_instructions_ && !triton_xf32_paired_lds) {
                  mlir::ROCDL::SchedBarrier::create(
                      builder, mlir::ROCDL::SchedGroupMask::none);
                }
                mlir::OperationState mma_state(entry_function.getLoc(),
                                               "fly.mma_atom_call_ssa");
                // Triton's wide XF32 layout feeds B as the first MFMA source
                // and A as the second.  The resulting native accumulator view
                // has four adjacent N columns in each consecutive group of
                // four elements, which the direct epilogue stores as v4f32.
                mma_state.addOperands(
                    triton_xf32_paired_lds
                        ? mlir::ValueRange{
                              mma_atom, rhs_mfma, lhs_mfma,
                              next_accumulators[accumulator_index]}
                        : mlir::ValueRange{
                              mma_atom, lhs_mfma, rhs_mfma,
                              next_accumulators[accumulator_index]});
                mma_state.addTypes(accumulator_type);
                next_accumulators[accumulator_index] =
                    builder.create(mma_state)->getResult(0);
              }
            }
          }
        }
        if (schedule_instructions_) {
          mlir::ROCDL::SchedBarrier::create(
              builder, mlir::ROCDL::SchedGroupMask::none);
          mlir::ROCDL::SetPrioOp::create(builder, /*priority=*/0);
        }
        return next_accumulators;
      }
      constexpr int64_t kMfmasPerFragment = 2;
      for (int64_t group = 0; group < k_groups; ++group) {
        for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
          Value lhs_fragment =
              lhs_fragments[group * wave_tile_rows + tile_row];
          for (int64_t tile_column = 0; tile_column < wave_tile_columns;
               ++tile_column) {
            Value rhs_fragment =
                rhs_fragments[group * wave_tile_columns + tile_column];
            const int64_t accumulator_index =
                get_accumulator_index(tile_row, tile_column);
            for (int64_t mfma = 0; mfma < kMfmasPerFragment; ++mfma) {
              before_mfma(mfma_index++);
              const int64_t fragment_base =
                  mfma * input_fragment_elements;
              llvm::SmallVector<int64_t> fragment_mask;
              fragment_mask.reserve(input_fragment_elements);
              for (int64_t element = 0; element < input_fragment_elements;
                   ++element) {
                fragment_mask.push_back(fragment_base + element);
              }
              Value lhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, lhs_fragment, lhs_fragment,
                  fragment_mask);
              Value rhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, rhs_fragment, rhs_fragment,
                  fragment_mask);
              if (is_fp8_) {
                auto packed_type = mlir::VectorType::get(
                    {input_fragment_elements}, builder.getI8Type());
                lhs_mfma = mlir::vector::BitCastOp::create(
                    builder, packed_type, lhs_mfma);
                rhs_mfma = mlir::vector::BitCastOp::create(
                    builder, packed_type, rhs_mfma);
              }
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

    auto emit_compute_row_major_streamed =
        [&](Value shared, Value shared_stage,
            llvm::ArrayRef<Value> accumulators,
            auto&& before_mfma) -> llvm::SmallVector<Value> {
      CHECK(row_major_rhs_single_buffer);
      CHECK_EQ(wave_tile_rows, 1);
      CHECK_EQ(wave_tile_columns, 1);
      CHECK_EQ(accumulators.size(), 1);
      const int64_t mfmas = compute_stage_k / atom_k;
      CHECK_EQ(mfmas, 16);
      const int64_t groups = mfmas / 2;
      llvm::SmallVector<Value> low_lhs(groups);
      llvm::SmallVector<Value> high_lhs(groups);
      llvm::SmallVector<Value> low_rhs(groups);
      llvm::SmallVector<Value> high_rhs(groups);
      Value lhs_row = Add(builder, wave_row_offset, lane_axis);
      Value rhs_row = Add(builder, wave_column_offset, lane_axis);
      int64_t loaded_groups = 0;
      auto load_through = [&](int64_t last_group) {
        mlir::ROCDL::SchedBarrier::create(builder,
                                          mlir::ROCDL::SchedGroupMask::none);
        for (; loaded_groups <= last_group; ++loaded_groups) {
          llvm::SmallVector<Value, 2> lhs_pair = emit_row_major_lds_pair(
              shared, shared_stage, lhs_row, /*shared_base=*/0,
              lhs_stage_elements, loaded_groups);
          llvm::SmallVector<Value, 2> rhs_pair = emit_row_major_lds_pair(
              shared, shared_stage, rhs_row, lhs_shared_elements,
              rhs_stage_elements, loaded_groups);
          low_lhs[loaded_groups] = lhs_pair[0];
          high_lhs[loaded_groups] = lhs_pair[1];
          low_rhs[loaded_groups] = rhs_pair[0];
          high_rhs[loaded_groups] = rhs_pair[1];
        }
        mlir::ROCDL::SchedBarrier::create(builder,
                                          mlir::ROCDL::SchedGroupMask::none);
      };
      Value accumulator = accumulators.front();
      int64_t mfma_index = 0;
      auto emit_mfma = [&](Value lhs, Value rhs) {
        before_mfma(mfma_index++);
        mlir::OperationState mma_state(entry_function.getLoc(),
                                       "fly.mma_atom_call_ssa");
        mma_state.addOperands({mma_atom, lhs, rhs, accumulator});
        mma_state.addTypes(accumulator_type);
        accumulator = builder.create(mma_state)->getResult(0);
      };
      // Match Triton's gfx942 modulo schedule: start MFMA 0 after the first
      // A/B pair and bring the remaining K groups in under it. Each read
      // returns both output-tile rows, so both M/N fragments are resident when
      // the refill window opens.
      constexpr std::array<int64_t, 4> kReadBatchEnds = {0, 3, 5, 7};
      for (int64_t group = 0; group < groups; ++group) {
        if (group < kReadBatchEnds.size()) {
          load_through(kReadBatchEnds[group]);
        }
        emit_mfma(low_lhs[group], low_rhs[group]);
      }
      for (int64_t group = 0; group < groups; ++group) {
        emit_mfma(high_lhs[group], high_rhs[group]);
      }
      return {accumulator};
    };

    auto emit_compute_row_major_square_streamed =
        [&](Value shared, Value shared_stage,
            llvm::ArrayRef<Value> accumulators, auto&& after_first_read_batch,
            auto&& after_second_read_batch,
            auto&& before_mfma) -> llvm::SmallVector<Value> {
      CHECK(row_major_rhs_square_double_buffer);
      CHECK(wave_tile_rows == 2 || wave_tile_rows == 4);
      CHECK(wave_tile_columns == 2 || wave_tile_columns == 4);
      CHECK_EQ(accumulators.size(), wave_tile_rows * wave_tile_columns);
      const int64_t kGroups = stage_k_ / atom_k;
      llvm::SmallVector<Value> lhs_fragments(kGroups * wave_tile_rows);
      llvm::SmallVector<Value> rhs_fragments(kGroups * wave_tile_columns);
      auto emit_fragment_pair = [&](Value row, int64_t shared_base,
                                    int64_t stage_elements, int64_t group,
                                    bool transposed_rhs) {
        auto logical_k_for_group = [&](int64_t k_group) {
          return Add(builder,
                     Mul(builder, lane_group, IndexConstant(builder, 4)),
                     IndexConstant(builder, k_group * atom_k));
        };
        Value first_logical_k = logical_k_for_group(group);
        // Advancing the transposed RHS by 32 rows changes the high-row
        // swizzle phase by eight BF16 elements. The caller assigns that second
        // fragment to the adjacent K group, so both reads retain the same
        // physical column and LLVM can combine them.
        Value stage_base =
            Mul(builder, shared_stage, IndexConstant(builder, stage_elements));
        auto emit_fragment_index = [&](Value logical_row, Value logical_k) {
          Value physical_k =
              shared_base == 0
                  ? (row_major_rhs_wide_source_swap
                         ? SwizzleTritonVec4(builder, logical_row, logical_k)
                         : SwizzleTritonMfma32(builder, logical_row, logical_k,
                                              stage_k_))
                  : (row_major_rhs_wide_source_swap
                         ? SwizzleTritonAmdRotatingRhs(
                               builder, logical_row, logical_k)
                         : SwizzleTritonMfma32TransposedRhs(
                               builder, logical_row, logical_k, stage_k_));
          return Add(builder, IndexConstant(builder, shared_base),
                     Add(builder, stage_base,
                         Add(builder,
                             Mul(builder, logical_row,
                                 IndexConstant(builder, stage_k_)),
                             physical_k)));
        };
        Value first_index = emit_fragment_index(row, first_logical_k);
        const int64_t second_tile_stride = atom_m;
        Value second_index =
            Add(builder, first_index,
                IndexConstant(builder, second_tile_stride * stage_k_));
        Value first = mlir::vector::TransferReadOp::create(
            builder, input_vector_type, shared, mlir::ValueRange{first_index},
            /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
        Value second = mlir::vector::TransferReadOp::create(
            builder, input_vector_type, shared, mlir::ValueRange{second_index},
            /*padding=*/std::nullopt, llvm::ArrayRef<bool>{true});
        return llvm::SmallVector<Value, 2>{first, second};
      };
      auto load_operand = [&](bool transposed_rhs) {
        mlir::ROCDL::SchedBarrier::create(builder,
                                          mlir::ROCDL::SchedGroupMask::none);
        for (int64_t group = 0; group < kGroups; ++group) {
          if (!transposed_rhs) {
            for (int64_t pair = 0; pair < wave_tile_rows / 2; ++pair) {
              Value lhs_row = Add(
                  builder, wave_row_offset,
                  Add(builder, lane_axis,
                      IndexConstant(builder, pair * 2 * atom_m)));
              llvm::SmallVector<Value, 2> lhs_pair = emit_fragment_pair(
                  lhs_row, /*shared_base=*/0, lhs_stage_elements, group,
                  /*transposed_rhs=*/false);
              lhs_fragments[group * wave_tile_rows + pair * 2] = lhs_pair[0];
              lhs_fragments[group * wave_tile_rows + pair * 2 + 1] =
                  lhs_pair[1];
            }
            continue;
          }
          for (int64_t pair = 0; pair < wave_tile_columns / 2; ++pair) {
            Value rhs_row = Add(
                builder, wave_column_offset,
                Add(builder, lane_axis,
                    IndexConstant(builder, pair * 2 * atom_m)));
            llvm::SmallVector<Value, 2> rhs_pair = emit_fragment_pair(
                rhs_row, lhs_shared_elements, rhs_stage_elements, group,
                /*transposed_rhs=*/true);
            rhs_fragments[group * wave_tile_columns + pair * 2] = rhs_pair[0];
            // Moving to the adjacent 32-row output tile changes the
            // transposed swizzle phase by one K group for K64 and two groups
            // for K32.
            const int64_t second_rhs_group =
                row_major_rhs_wide_source_swap
                    ? group ^ 1
                    : group ^ (64 / stage_k_);
            rhs_fragments[second_rhs_group * wave_tile_columns + pair * 2 + 1] =
                rhs_pair[1];
          }
        }
        mlir::ROCDL::SchedBarrier::create(builder,
                                          mlir::ROCDL::SchedGroupMask::none);
      };
      llvm::SmallVector<Value> next_accumulators(accumulators);
      int64_t mfma_index = 0;
      // Triton batches LDS by operand: all A fragments, four next-tile B
      // loads, all B fragments, then four next-tile A loads. This lets the
      // first MFMA begin once nine of sixteen reads retire instead of waiting
      // for an adjacent A/B pair from each K group.
      load_operand(/*transposed_rhs=*/false);
      mlir::ROCDL::SetPrioOp::create(builder, /*priority=*/1);
      after_first_read_batch();
      mlir::ROCDL::SchedBarrier::create(builder,
                                        mlir::ROCDL::SchedGroupMask::none);
      load_operand(/*transposed_rhs=*/true);
      mlir::ROCDL::SetPrioOp::create(builder, /*priority=*/0);
      after_second_read_batch();
      mlir::ROCDL::SchedBarrier::create(builder,
                                        mlir::ROCDL::SchedGroupMask::none);
      auto emit_mfma = [&](int64_t group, int64_t tile_row,
                           int64_t tile_column) {
        before_mfma(mfma_index++);
        const int64_t accumulator_index =
            get_accumulator_index(tile_row, tile_column);
        mlir::OperationState mma_state(entry_function.getLoc(),
                                       "fly.mma_atom_call_ssa");
        Value lhs_fragment =
            lhs_fragments[group * wave_tile_rows + tile_row];
        Value rhs_fragment =
            rhs_fragments[group * wave_tile_columns + tile_column];
        mma_state.addOperands(
            row_major_rhs_wide_source_swap
                ? mlir::ValueRange{mma_atom, rhs_fragment, lhs_fragment,
                                   next_accumulators[accumulator_index]}
                : mlir::ValueRange{mma_atom, lhs_fragment, rhs_fragment,
                                   next_accumulators[accumulator_index]});
        mma_state.addTypes(accumulator_type);
        next_accumulators[accumulator_index] =
            builder.create(mma_state)->getResult(0);
        if (mfma_index == 1) {
          mlir::ROCDL::SetPrioOp::create(builder, /*priority=*/1);
        }
      };
      for (int64_t group = 0; group < kGroups; ++group) {
        for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
          for (int64_t tile_column = 0; tile_column < wave_tile_columns;
               ++tile_column) {
            emit_mfma(group, tile_row, tile_column);
          }
        }
      }
      mlir::ROCDL::SetPrioOp::create(builder, /*priority=*/0);
      return next_accumulators;
    };

    auto emit_compute_local_split =
        [&](Value shared, Value shared_stage,
            llvm::ArrayRef<Value> first_lhs_fragments,
            llvm::ArrayRef<Value> first_rhs_fragments,
            llvm::ArrayRef<Value> accumulators, auto&& before_mfma,
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
          if (triton_xf32_paired_lds && schedule_instructions_) {
            // The XF32 path emits Triton's four LDS/VMEM/MFMA regions and
            // their compiler barriers explicitly. Applying the generic
            // homogeneous scheduling groups here collects its split reads and
            // loads back together and defeats the modulo schedule.
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
            return;
          }
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
          if (row_major_rhs_square_double_buffer) {
            // The square path emits Triton's two LDS/VMEM read batches
            // explicitly. Do not let the generic scheduler collect those
            // batches back into homogeneous memory runs.
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
                bool preload_next_staged_group,
                bool tie_mfma_accumulators) -> DirectHalfState {
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
            // hipBLASLt wide-tile kernel. The eight next-tile direct vectors
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
          // LLVM naturally coalesces the loop-carried accumulator chains, but
          // assigns fresh AGPR tuples to many MFMAs in the statically peeled
          // F16 tail. The x4 accumulator is exactly 128 bits, so AMDGPU has no
          // tied `_mac` machine form for this atom. Batch independent MFMAs
          // with matching inline-assembly constraints in the tail only; this
          // removes the otherwise required accvgpr read/write repair copies
          // without constraining register allocation in the recurring loop.
          llvm::SmallVector<Value> tied_mfma_a;
          llvm::SmallVector<Value> tied_mfma_b;
          llvm::SmallVector<int64_t> tied_mfma_accumulator_indices;
          auto flush_tied_mfma_batch = [&] {
            const int64_t batch_size = tied_mfma_a.size();
            if (batch_size == 0) {
              return;
            }
            CHECK(tie_mfma_accumulators);
            CHECK_EQ(lhs_element_type_, F16);
            llvm::SmallVector<mlir::Type> output_types(batch_size,
                                                       accumulator_type);
            mlir::Type output_type = accumulator_type;
            if (batch_size != 1) {
              output_type = mlir::LLVM::LLVMStructType::getLiteral(
                  builder.getContext(), output_types);
            }
            llvm::SmallVector<Value> operands;
            operands.reserve(3 * batch_size);
            for (int64_t index = 0; index < batch_size; ++index) {
              operands.push_back(tied_mfma_a[index]);
              operands.push_back(tied_mfma_b[index]);
            }
            for (int64_t accumulator_index :
                 tied_mfma_accumulator_indices) {
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
              assembly += "v_mfma_f32_16x16x16_f16 $" +
                          std::to_string(index) + ", $" +
                          std::to_string(batch_size + 2 * index) + ", $" +
                          std::to_string(batch_size + 2 * index + 1) +
                          ", $" + std::to_string(3 * batch_size + index) +
                          ";\n";
            }
            Value result =
                mlir::LLVM::InlineAsmOp::create(
                    builder, output_type, operands, assembly, constraints,
                    /*has_side_effects=*/false,
                    /*is_align_stack=*/false, mlir::LLVM::TailCallKind::None,
                    mlir::LLVM::AsmDialectAttr::get(
                        builder.getContext(), mlir::LLVM::AsmDialect::AD_ATT),
                    /*operand_attrs=*/mlir::ArrayAttr())
                    .getResult(0);
            for (int64_t index = 0; index < batch_size; ++index) {
              next_accumulators[tied_mfma_accumulator_indices[index]] =
                  batch_size == 1
                      ? result
                      : mlir::LLVM::ExtractValueOp::create(builder, result,
                                                           index);
            }
            tied_mfma_a.clear();
            tied_mfma_b.clear();
            tied_mfma_accumulator_indices.clear();
          };
          auto is_tied_mfma_schedule_boundary = [](int64_t mfma_index) {
            constexpr std::array<int64_t, 44> kBoundaries = {
                1,   2,   3,   4,   7,   8,   9,   10,  13,  14,  15,
                16,  19,  20,  35,  57,  79,  101, 112, 122, 131, 134,
                143, 145, 154, 156, 165, 167, 176, 178, 179, 180, 181,
                182, 187, 189, 191, 192, 193, 194, 195, 196, 199, 200};
            constexpr std::array<int64_t, 8> kLateBoundaries = {
                201, 202, 205, 206, 207, 208, 211, 212};
            return std::find(kBoundaries.begin(), kBoundaries.end(),
                             mfma_index) != kBoundaries.end() ||
                   std::find(kLateBoundaries.begin(), kLateBoundaries.end(),
                             mfma_index) != kLateBoundaries.end();
          };
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
              if (rhs_input_type_ == S4) {
                for (Value& fragment : rhs_group) {
                  CHECK(fragment.getType().isInteger(32));
                  auto packed_type =
                      mlir::VectorType::get({4}, builder.getI8Type());
                  Value packed = mlir::arith::BitcastOp::create(
                      builder, packed_type, fragment);
                  fragment = unpack_s4_vector(packed, staged_input_vector_type);
                }
              }
            } else if (group == 0 && !half_preloaded_staged_group.empty()) {
              rhs_group = llvm::SmallVector<Value>(half_preloaded_staged_group);
            } else if (group == 1 && !pipelined_staged_group.empty()) {
              CHECK_EQ(pipelined_staged_group.size(), wave_tile_columns);
              rhs_group = std::move(pipelined_staged_group);
            } else {
              rhs_group = emit_rhs_lds_group(half_shared, half_stage,
                                             group * 2 * atom_k);
            }
            auto append_mfma = [&](int64_t tile_row, int64_t tile_column,
                                   int64_t mfma) {
              if (tie_mfma_accumulators &&
                  is_tied_mfma_schedule_boundary(mfma_index)) {
                flush_tied_mfma_batch();
              }
              before_mfma(mfma_index++);
              const int64_t fragment_base =
                  mfma * input_fragment_elements;
              llvm::SmallVector<int64_t> fragment_mask;
              fragment_mask.reserve(input_fragment_elements);
              for (int64_t element = 0; element < input_fragment_elements;
                   ++element) {
                fragment_mask.push_back(fragment_base + element);
              }
              Value lhs_fragment = lhs_group[tile_row];
              Value rhs_fragment = rhs_group[tile_column];
              Value lhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, lhs_fragment, lhs_fragment,
                  fragment_mask);
              Value rhs_mfma = mlir::vector::ShuffleOp::create(
                  builder, input_vector_type, rhs_fragment, rhs_fragment,
                  fragment_mask);
              if (is_fp8_) {
                auto packed_type = mlir::VectorType::get(
                    {input_fragment_elements}, builder.getI8Type());
                lhs_mfma = mlir::vector::BitCastOp::create(
                    builder, packed_type, lhs_mfma);
                rhs_mfma = mlir::vector::BitCastOp::create(
                    builder, packed_type, rhs_mfma);
              }
              const int64_t accumulator_index =
                  get_accumulator_index(tile_row, tile_column);
              if (tie_mfma_accumulators) {
                CHECK(std::find(tied_mfma_accumulator_indices.begin(),
                                tied_mfma_accumulator_indices.end(),
                                accumulator_index) ==
                      tied_mfma_accumulator_indices.end());
                tied_mfma_a.push_back(source_swap_mfma ? rhs_mfma
                                                       : lhs_mfma);
                tied_mfma_b.push_back(source_swap_mfma ? lhs_mfma
                                                       : rhs_mfma);
                tied_mfma_accumulator_indices.push_back(accumulator_index);
                if (tied_mfma_a.size() == 4) {
                  flush_tied_mfma_batch();
                }
                return;
              }
              mlir::OperationState mma_state(entry_function.getLoc(),
                                             "fly.mma_atom_call_ssa");
              mma_state.addOperands(
                  source_swap_mfma
                      ? mlir::ValueRange{
                            mma_atom, rhs_mfma, lhs_mfma,
                            next_accumulators[accumulator_index]}
                      : mlir::ValueRange{
                            mma_atom, lhs_mfma, rhs_mfma,
                            next_accumulators[accumulator_index]});
              mma_state.addTypes(accumulator_type);
              next_accumulators[accumulator_index] =
                  builder.create(mma_state)->getResult(0);
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
                }
              }
            } else {
              for (int64_t tile_row = 0; tile_row < wave_tile_rows;
                   ++tile_row) {
                for (int64_t mfma = 0; mfma < 2; ++mfma) {
                  for (int64_t tile_column = 0;
                       tile_column < wave_tile_columns; ++tile_column) {
                    append_mfma(tile_row, tile_column, mfma);
                  }
                }
              }
            }
          }
          flush_tied_mfma_batch();
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
    const bool peel_row_major_square_masked_tail =
        row_major_rhs_square_double_buffer && masked_k_tail &&
        padded_k >= 2 * stage_k_;
    Value loop_upper_bound =
        tensile_direct_operand
            ? IndexConstant(
                  builder,
                  padded_k - (((padded_k / stage_k_) % 2)
                                  ? stage_k_
                                  : 2 * stage_k_))
            : (faithful_fly_pipeline
                   ? IndexConstant(
                         builder,
                         padded_k -
                             (peel_row_major_square_masked_tail ? 2 : 1) *
                                 stage_k_)
                   : compute_upper_bound);
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
          faithful_fly_pipeline ? next_k : Rem(builder, next_k, padded_k);
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
        if (current_direct.empty()) {
          current_direct =
              tensile_direct_lhs
                  ? emit_tensile_direct_lhs_fragments(k_base)
                  : emit_tensile_direct_rhs_fragments(k_base);
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
        if (next_staged_registers.empty()) {
          next_staged_registers =
              tensile_direct_lhs
                  ? emit_rhs_register_vectors(Add(builder, k_base, step))
                  : emit_lhs_register_vectors(Add(builder, k_base, step));
        }
        llvm::SmallVector<Value> current_preloaded_staged_group;
        for (int64_t index = 0; index < preloaded_staged_group_count; ++index) {
          current_preloaded_staged_group.push_back(loop.getRegionIterArg(
              preloaded_staged_group_loop_index + index));
        }
        if (current_preloaded_staged_group.empty()) {
          current_preloaded_staged_group =
              tensile_direct_lhs
                  ? emit_rhs_lds_group(staged_shared, current_stage, 0)
                  : emit_lhs_lds_group(staged_shared, current_stage, 0);
        }
        DirectHalfState first = emit_direct_half(
            k_base, staged_shared, current_stage, next_stage,
            current_accumulators, current_direct, next_staged_registers,
            current_preloaded_staged_group,
            /*prefetch_next_direct=*/true,
            /*refill_next_staged=*/true,
            /*load_future_staged=*/true,
            /*direct_has_trailing_staged_loads=*/true,
            /*preload_next_staged_group=*/true,
            /*tie_mfma_accumulators=*/false);
        DirectHalfState second = emit_direct_half(
            Add(builder, k_base, step), first.shared, next_stage, current_stage,
            first.accumulators, first.direct, first.staged_registers,
            first.preloaded_staged_group,
            /*prefetch_next_direct=*/kCarryWideDirectAcrossLoop,
            /*refill_next_staged=*/true,
            /*load_future_staged=*/kCarryWideStagedPayloadAcrossLoop,
            /*direct_has_trailing_staged_loads=*/true,
            /*preload_next_staged_group=*/false,
            /*tie_mfma_accumulators=*/false);

        llvm::SmallVector<Value> next_loop_values =
            std::move(second.accumulators);
        if (kCarryWideDirectAcrossLoop) {
          next_loop_values.append(second.direct);
        }
        if (kCarryWideStagedPayloadAcrossLoop) {
          next_loop_values.append(second.staged_registers);
        }
        if (kCarryWidePreloadedGroupAcrossLoop) {
          next_loop_values.append(second.preloaded_staged_group);
        }
        next_loop_values.push_back(second.shared);
        mlir::scf::YieldOp::create(builder, next_loop_values);
      } else {
        llvm::SmallVector<Value> current_lhs;
        if (preload_lds_fragments_ && !row_major_rhs_single_buffer &&
            !row_major_rhs_square_double_buffer) {
          if (local_split_k_) {
            const int64_t lhs_begin = accumulators_per_wave + wave_tile_columns;
            for (int64_t index = 0; index < wave_tile_rows; ++index) {
              current_lhs.push_back(loop.getRegionIterArg(lhs_begin + index));
            }
          } else {
            current_lhs = emit_lhs_lds_fragments(staged_shared, current_stage);
          }
        }
        llvm::SmallVector<Value> next_lhs_registers;
        llvm::SmallVector<Value> next_rhs_registers;
        if (single_buffer_lds_ || row_major_rhs_square_double_buffer) {
          if (row_major_rhs_square_double_buffer) {
            // The MFMA32 compute path issues these payloads between its two
            // LDS read batches, matching Triton's stage-2 modulo pipeline.
            const int64_t block_threads = num_warps_ * 64;
            next_rhs_registers.resize(block_n_ * stage_k_ /
                                      load_vector_width / block_threads);
            next_lhs_registers.resize(block_m_ * stage_k_ /
                                      load_vector_width / block_threads);
          } else if (tensile_wide_tile || local_split_k_) {
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
            if (row_major_rhs_single_buffer) {
              // Keep the four coalesced A loads at the front of the VMEM
              // queue. The two B loads are issued after the first-half MFMAs,
              // matching Triton's modulo schedule.
              for (int64_t copy = 0; copy < lhs_copies; ++copy) {
                next_lhs_registers[copy] =
                    emit_lhs_register_vector(prefetch_k, copy);
              }
            } else {
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
            }
          } else {
            next_rhs_registers = emit_rhs_register_vectors(prefetch_k);
            next_lhs_registers = emit_lhs_register_vectors(prefetch_k);
          }
        } else if (stage_rhs_) {
          staged_shared =
              emit_rhs_lds_stage(staged_shared, prefetch_k, next_stage);
        }
        if (!single_buffer_lds_ && !row_major_rhs_square_double_buffer) {
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
        } else if (load_rhs_on_demand && !row_major_rhs_single_buffer &&
                   !row_major_rhs_square_double_buffer) {
          current_rhs = emit_rhs_lds_fragments(staged_shared, current_stage);
        } else {
          current_rhs.reserve(fragment_state_end - accumulators_per_wave);
          for (int64_t index = accumulators_per_wave;
               index < fragment_state_end; ++index) {
            current_rhs.push_back(loop.getRegionIterArg(index));
          }
        }
        if (row_major_rhs_single_buffer) {
          // Close the previous iteration's refill after issuing the next A
          // loads, then stream both operands from the newly completed tile.
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
          staged_shared =
              SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                    staged_shared)
                  .getResult(0);
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
        } else if (small_grid_single_buffer && !local_split_k_) {
          // The complete current A/B tile is now resident in VGPRs. Make that
          // point uniform across the workgroup before any wave starts
          // overwriting the single LDS allocation with the next tile.
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
          staged_shared =
              SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                    staged_shared)
                  .getResult(0);
          mlir::ROCDL::SchedBarrier::create(builder,
                                            mlir::ROCDL::SchedGroupMask::none);
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
          if (row_major_rhs_single_buffer) {
            if (mfma_index == 9) {
              // Triton's winning modulo schedule issues the two adjacent-K B
              // payloads after MFMA 8, then opens LDS and retires all four A
              // payloads before MFMA 9.
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              for (int64_t copy = 0; copy < next_rhs_registers.size(); ++copy) {
                next_rhs_registers[copy] =
                    emit_rhs_register_vector(prefetch_k, copy);
              }
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              staged_shared =
                  SyncThreadsOp::create(
                      builder, mlir::TypeRange{lhs_shared_type}, staged_shared)
                      .getResult(0);
              emit_wait_vmcnt(4);
              staged_shared = emit_lhs_register_stores(
                  staged_shared, next_lhs_registers, 0, 2);
              emit_wait_vmcnt(2);
              staged_shared = emit_lhs_register_stores(
                  staged_shared, next_lhs_registers, 2, 4);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }
            if (mfma_index == 10) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              emit_wait_vmcnt(0);
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              return;
            }
            if (mfma_index == 12) {
              mlir::ROCDL::SchedBarrier::create(
                  builder, mlir::ROCDL::SchedGroupMask::none);
              staged_shared =
                  emit_rhs_register_stores(staged_shared, next_rhs_registers, 0,
                                           next_rhs_registers.size());
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
              emit_wait_vmcnt(std::min(kPrefetchDepth, refill_copies - copy) -
                              1);
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
        const bool accumulate_scaled_partials =
            block_scaled_dot_ && !dequantize_block_scales_;
        llvm::ArrayRef<Value> compute_accumulators =
            accumulate_scaled_partials
                ? llvm::ArrayRef<Value>(initial_accumulators)
                : llvm::ArrayRef<Value>(current_accumulators);
        llvm::SmallVector<Value> next_accumulators =
            local_split_k_
                ? emit_compute_local_split(
                      staged_shared, current_stage, current_lhs, current_rhs,
                      compute_accumulators, before_mfma, after_mfma)
            : row_major_rhs_single_buffer
                ? emit_compute_row_major_streamed(staged_shared, current_stage,
                                                  compute_accumulators,
                                                  before_mfma)
            : row_major_rhs_square_double_buffer
                ? emit_compute_row_major_square_streamed(
                      staged_shared, current_stage, compute_accumulators,
                      [&] {
                        for (int64_t copy = 0; copy < next_rhs_registers.size();
                             ++copy) {
                          next_rhs_registers[copy] =
                              emit_rhs_register_vector_with_mask(
                                  prefetch_k, copy,
                                  /*mask_k_tail=*/
                                      !peel_row_major_square_masked_tail);
                        }
                      },
                      [&] {
                        for (int64_t copy = 0; copy < next_lhs_registers.size();
                             ++copy) {
                          next_lhs_registers[copy] =
                              emit_lhs_register_vector_with_mask(
                                  prefetch_k, copy,
                                  /*mask_k_tail=*/
                                      !peel_row_major_square_masked_tail);
                        }
                      },
                      before_mfma)
            : preload_lds_fragments_
                ? emit_compute_preloaded(current_lhs, current_rhs,
                                         compute_accumulators, before_mfma)
                : emit_compute_tile(staged_shared, current_stage, k_base,
                                    compute_accumulators, current_rhs);
        if (accumulate_scaled_partials) {
          next_accumulators = accumulate_block_scaled(
              current_accumulators, next_accumulators, k_base);
        }
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
        if (row_major_rhs_square_double_buffer) {
          // The opening barrier retires every current-tile LDS read before the
          // one physical stage is overwritten. LLVM inserts progressive VMEM
          // waits for the dependent vector stores; the closing barrier
          // publishes the refilled stage to every wave.
          synchronized_lhs =
              SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                    synchronized_lhs)
                  .getResult(0);
          synchronized_lhs = emit_rhs_register_stores_at_stage(
              synchronized_lhs, next_rhs_registers, 0,
              next_rhs_registers.size(), next_stage);
          synchronized_lhs = emit_lhs_register_stores_at_stage(
              synchronized_lhs, next_lhs_registers, 0,
              next_lhs_registers.size(), next_stage);
          synchronized_lhs =
              SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                    synchronized_lhs)
                  .getResult(0);
        } else if (!tensile_wide_tile && !small_grid_single_buffer) {
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
      if ((padded_k / stage_k_) % 2 == 0) {
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
        const int64_t first_tail_tile = padded_k / stage_k_ - 2;
        Value first_tail_k =
            IndexConstant(builder, padded_k - 2 * stage_k_);
        Value first_tail_stage =
            IndexConstant(builder, first_tail_tile % pipeline_stages);
        Value second_tail_stage = IndexConstant(
            builder, (first_tail_tile + 1) % pipeline_stages);
        if (tail_direct.empty()) {
          tail_direct =
              tensile_direct_lhs
                  ? emit_tensile_direct_lhs_fragments(first_tail_k)
                  : emit_tensile_direct_rhs_fragments(first_tail_k);
        }
        if (tail_staged_registers.empty()) {
          tail_staged_registers =
              tensile_direct_lhs
                  ? emit_rhs_register_vectors(
                        IndexConstant(builder, padded_k - stage_k_))
                  : emit_lhs_register_vectors(
                        IndexConstant(builder, padded_k - stage_k_));
        }
        if (tail_preloaded_staged_group.empty()) {
          tail_preloaded_staged_group =
              tensile_direct_lhs
                  ? emit_rhs_lds_group(tail_shared, first_tail_stage, 0)
                  : emit_lhs_lds_group(tail_shared, first_tail_stage, 0);
        }
        DirectHalfState first = emit_direct_half(
            first_tail_k, tail_shared, first_tail_stage, second_tail_stage,
            loop_accumulators, tail_direct, tail_staged_registers,
            tail_preloaded_staged_group,
            /*prefetch_next_direct=*/true,
            /*refill_next_staged=*/true,
            /*load_future_staged=*/false,
            /*direct_has_trailing_staged_loads=*/true,
            /*preload_next_staged_group=*/true,
            // A zero-trip recurring loop leaves constant-zero accumulators.
            // Let the ordinary MFMA lowering materialize those in AGPRs before
            // applying the tied inline-assembly form in the second half.
            // Rank-3 kernels also keep batch addressing live through the tail;
            // LLVM does not allocate the tied inline-assembly accumulators
            // reliably under that pressure, so retain its native MFMA form.
            /*tie_mfma_accumulators=*/
                lhs_element_type_ == F16 && !batched_gemm_ &&
                padded_k > 2 * stage_k_);
        DirectHalfState second = emit_direct_half(
            IndexConstant(builder, padded_k - stage_k_), first.shared,
            second_tail_stage, first_tail_stage, first.accumulators,
            first.direct, first.staged_registers,
            first.preloaded_staged_group,
            /*prefetch_next_direct=*/false,
            /*refill_next_staged=*/false,
            /*load_future_staged=*/false,
            /*direct_has_trailing_staged_loads=*/false,
            /*preload_next_staged_group=*/false,
            /*tie_mfma_accumulators=*/
                lhs_element_type_ == F16 && !batched_gemm_);
        final_accumulators = std::move(second.accumulators);
      } else {
        llvm::SmallVector<Value> tail_direct;
        for (int64_t index = accumulators_per_wave;
             index < preloaded_fragment_end; ++index) {
          tail_direct.push_back(loop.getResult(index));
        }
        Value tail_shared = loop.getResult(shared_loop_index);
        if (tail_direct.empty()) {
          Value tail_k = IndexConstant(builder, padded_k - stage_k_);
          tail_direct =
              tensile_direct_lhs
                  ? emit_tensile_direct_lhs_fragments(tail_k)
                  : emit_tensile_direct_rhs_fragments(tail_k);
        }
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
      if (peel_row_major_square_masked_tail) {
        // The recurring loop stops after refilling the penultimate complete
        // stage. Compute that stage while loading the one partial stage, then
        // refill and compute the tail explicitly. This keeps every K-bound
        // branch out of the hot loop: only these four A and four B tail loads
        // need masking for a K32/K64 square tile.
        Value tail_shared = loop.getResult(shared_loop_index);
        Value physical_stage = lower_bound;
        Value tail_k = IndexConstant(builder, padded_k - stage_k_);
        llvm::SmallVector<Value> tail_rhs_registers(4);
        llvm::SmallVector<Value> tail_lhs_registers(4);
        llvm::SmallVector<Value> penultimate_accumulators =
            emit_compute_row_major_square_streamed(
                tail_shared, physical_stage, loop_accumulators,
                [&] {
                  for (int64_t copy = 0; copy < tail_rhs_registers.size();
                       ++copy) {
                    tail_rhs_registers[copy] =
                        emit_rhs_register_vector_with_mask(
                            tail_k, copy, /*mask_k_tail=*/true);
                  }
                },
                [&] {
                  for (int64_t copy = 0; copy < tail_lhs_registers.size();
                       ++copy) {
                    tail_lhs_registers[copy] =
                        emit_lhs_register_vector_with_mask(
                            tail_k, copy, /*mask_k_tail=*/true);
                  }
                },
                [](int64_t) {});
        emit_hot_loop_schedule();
        tail_shared =
            SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                  tail_shared)
                .getResult(0);
        tail_shared = emit_rhs_register_stores_at_stage(
            tail_shared, tail_rhs_registers, 0, tail_rhs_registers.size(),
            physical_stage);
        tail_shared = emit_lhs_register_stores_at_stage(
            tail_shared, tail_lhs_registers, 0, tail_lhs_registers.size(),
            physical_stage);
        tail_shared =
            SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                  tail_shared)
                .getResult(0);
        final_accumulators = emit_compute_row_major_square_streamed(
            tail_shared, physical_stage, penultimate_accumulators, [] {},
            [] {}, [](int64_t) {});
      } else if (local_split_k_) {
        llvm::SmallVector<Value> tail_rhs;
        llvm::SmallVector<Value> tail_lhs;
        for (int64_t index = 0; index < wave_tile_columns; ++index) {
          tail_rhs.push_back(loop.getResult(accumulators_per_wave + index));
        }
        for (int64_t index = 0; index < wave_tile_rows; ++index) {
          tail_lhs.push_back(loop.getResult(accumulators_per_wave +
                                            wave_tile_columns + index));
        }
        final_accumulators = emit_compute_local_split(
            loop.getResult(shared_loop_index), lower_bound, tail_lhs, tail_rhs,
            loop_accumulators, [](int64_t) {}, [](int64_t) {});
      } else {
        llvm::SmallVector<Value> tail_rhs;
        const int64_t fragment_state_end = preloaded_fragment_end;
        for (int64_t index = accumulators_per_wave; index < fragment_state_end;
             ++index) {
          tail_rhs.push_back(loop.getResult(index));
        }
        Value tail_stage =
            IndexConstant(builder,
                          (padded_k / stage_k_ - 1) % pipeline_stages);
        if (preload_lds_fragments_) {
          Value tail_shared = loop.getResult(shared_loop_index);
          if (row_major_rhs_single_buffer) {
            // The main loop deliberately leaves the final refill open so the
            // next iteration can issue VMEM before closing it. The peeled
            // tail has no next iteration, so close it here before LDS reads.
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
            tail_shared =
                SyncThreadsOp::create(builder, mlir::TypeRange{lhs_shared_type},
                                      tail_shared)
                    .getResult(0);
            mlir::ROCDL::SchedBarrier::create(
                builder, mlir::ROCDL::SchedGroupMask::none);
          }
          if (row_major_rhs_single_buffer) {
            final_accumulators = emit_compute_row_major_streamed(
                tail_shared, tail_stage, loop_accumulators, [](int64_t) {});
          } else if (row_major_rhs_square_double_buffer) {
            final_accumulators = emit_compute_row_major_square_streamed(
                tail_shared, tail_stage, loop_accumulators, [] {}, [] {},
                [](int64_t) {});
          } else {
            llvm::SmallVector<Value> tail_lhs =
                emit_lhs_lds_fragments(tail_shared, tail_stage);
            if (load_rhs_on_demand) {
              tail_rhs = emit_rhs_lds_fragments(tail_shared, tail_stage);
            }
            final_accumulators = emit_compute_preloaded(
                tail_lhs, tail_rhs, loop_accumulators, [](int64_t) {});
          }
        } else {
          final_accumulators = emit_compute_tile(
              lhs_shared, tail_stage,
              IndexConstant(builder, padded_k - stage_k_),
              loop_accumulators, tail_rhs);
        }
      }
    } else {
      for (int64_t index = 0; index < accumulators_per_wave; ++index) {
        final_accumulators.push_back(loop.getResult(index));
      }
    }
    // Source-swapped MFMAs keep their accumulators in AGPRs through the
    // Tensile-style epilogue. AMD vector ALU instructions cannot consume an
    // AGPR directly, so scale those values after the explicit AGPR-to-VGPR
    // row read below.
    if (is_scaled_dot_ && !block_scaled_dot_ && !source_swap_mfma) {
      Value scale_vector = mlir::vector::BroadcastOp::create(
          builder, accumulator_type, output_scale);
      for (Value& accumulator : final_accumulators) {
        accumulator =
            mlir::arith::MulFOp::create(builder, accumulator, scale_vector);
      }
    }
    if (!epilogue_steps_.empty() && !has_vector_epilogue_ &&
        !source_swap_mfma) {
      for (Value& accumulator : final_accumulators) {
        accumulator = apply_scalar_epilogue(accumulator);
      }
    }
    mlir::Type output_type =
        mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType();
    const bool output_is_16_bit_float =
        output_type.isBF16() || output_type.isF16();
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

      auto scratch_vector_type = mlir::VectorType::get(
          {2 * accumulator_elements}, builder.getBF16Type());
      constexpr int64_t kEpilogueAccumulatorsPerWave = 8;
      auto scratch_index = [&](Value source_wave, int64_t slot) {
        // Keep lanes contiguous within one accumulator slot. The previous
        // lane-major layout advanced adjacent lanes by 128 bytes, mapping all
        // dwordx4 operations onto the same LDS bank group.
        Value wave_slot =
            Add(builder,
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
          stored =
              mlir::vector::TransferWriteOp::create(
                  builder, partial_bits, stored,
                  mlir::ValueRange{partial_index}, llvm::ArrayRef<bool>{true})
                  .getResult();
        }
        mlir::scf::YieldOp::create(builder, stored);
      };
      {
        mlir::OpBuilder::InsertionGuard spill_guard(builder);
        builder.setInsertionPointToStart(spill.thenBlock());
        // The lower K partition writes the upper half of its tile.
        spill_accumulators(reduction_shared, kEpilogueAccumulatorsPerWave);
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
      llvm::SmallVector<llvm::SmallVector<Value>>
          local_split_column_epilogue_operands(epilogue_steps_.size());
      for (int64_t step_index = 0; step_index < epilogue_steps_.size();
           ++step_index) {
        if (epilogue_steps_[step_index].broadcast_dimension !=
            epilogue_column_dimension_) {
          continue;
        }
        local_split_column_epilogue_operands[step_index].reserve(
            wave_tile_columns);
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             ++tile_column) {
          Value output_column = Add(
              builder, wave_column_base,
              Add(builder, Mul(builder, lane_group, IndexConstant(builder, 4)),
                          IndexConstant(builder, tile_column * tile_column_stride)));
          llvm::SmallVector<Value, 4> elements;
          for (int64_t element = 0; element < accumulator_elements; ++element) {
            elements.push_back(load_epilogue_element(
                step_index, wave_row_base,
                Add(builder, output_column, IndexConstant(builder, element))));
          }
          local_split_column_epilogue_operands[step_index].push_back(
              mlir::vector::FromElementsOp::create(builder, accumulator_type,
                                                   elements));
        }
      }
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
          if (is_scaled_dot_ && !block_scaled_dot_) {
            Value scale_vector = mlir::vector::BroadcastOp::create(
                builder, accumulator_type, output_scale);
            reduced =
                mlir::arith::MulFOp::create(builder, reduced, scale_vector);
          }
          const int64_t tile_row = accumulator_index / wave_tile_columns;
          const int64_t tile_column = accumulator_index % wave_tile_columns;
          Value output_row =
              Add(builder, wave_row_base,
                  Add(builder, lane_axis,
                      IndexConstant(builder, tile_row * tile_row_stride)));
          Value output_column = Add(
              builder, wave_column_base,
              Add(builder, Mul(builder, lane_group, IndexConstant(builder, 4)),
                          IndexConstant(builder, tile_column * tile_column_stride)));
          if (!epilogue_steps_.empty()) {
            if (!has_vector_epilogue_) {
              reduced = apply_scalar_epilogue(reduced);
            } else {
              const int64_t step_count =
                  epilogue_steps_.size() - (fold_final_negate_ ? 1 : 0);
              for (int64_t step_index = 0; step_index < step_count;
                   ++step_index) {
                Value operand;
                const EpilogueStep& step = epilogue_steps_[step_index];
                if (step.opcode != HloOpcode::kNegate) {
                  operand =
                      step.broadcast_dimension == epilogue_column_dimension_
                          ? local_split_column_epilogue_operands[step_index]
                                                                  [tile_column]
                          : load_epilogue_element(step_index, output_row,
                                                  output_column);
                }
                reduced =
                    apply_epilogue_step(reduced, step_index, operand);
              }
            }
          }
          Value packed = output_type.isBF16()
                             ? mlir::arith::TruncFOp::create(
                                   builder, packed_output_type, reduced)
                                   .getResult()
                             : reduced;
          llvm::SmallVector<Value, 3> indices =
              output_indices(output_row, output_column);
          stored_output = mlir::vector::TransferWriteOp::create(
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
    if (stage_output_via_lds) {
      TF_RET_CHECK(output_type.isBF16());
      TF_RET_CHECK((m_ % block_m_ == 0 || shift_m_edge) &&
                   (n_ % block_n_ == 0 || shift_n_edge));
      TF_RET_CHECK(block_m_ * block_n_ * sizeof(uint16_t) <= 64 * 1024);
    }
    Value staged_output = lhs_shared;
    if (stage_output_via_lds) {
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
    if (output_transpose_batch_inner_.has_value()) {
      // In the final [B,N,H,M] tensor, M is contiguous. A 16x16 BF16 MFMA
      // gives each lane four adjacent M values for one N column (and a 32x32
      // MFMA gives four such groups). Store those fragments directly instead
      // of transposing the complete macro-tile through LDS. The wave's four
      // lane groups collectively cover 16 contiguous M values per N column.
      TF_RET_CHECK(stage_output_ && output_type.isBF16() &&
                   epilogue_steps_.empty() && m_ % block_m_ == 0 &&
                   n_ % block_n_ == 0);
      constexpr int64_t kPackedRows = 4;
      auto packed_f32_type =
          mlir::VectorType::get({kPackedRows}, builder.getF32Type());
      auto packed_output_type =
          mlir::VectorType::get({kPackedRows}, output_type);
      for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             ++tile_column) {
          Value accumulator = final_accumulators[
              get_accumulator_index(tile_row, tile_column)];
          Value output_column = Add(
              builder, wave_column_base,
              Add(builder, lane_axis,
                  IndexConstant(builder,
                                tile_column * tile_column_stride)));
          for (int64_t group = 0;
               group < accumulator_elements / kPackedRows; ++group) {
            llvm::SmallVector<int64_t, kPackedRows> mask;
            for (int64_t element = 0; element < kPackedRows; ++element) {
              mask.push_back(group * kPackedRows + element);
            }
            Value values = mlir::vector::ShuffleOp::create(
                builder, packed_f32_type, accumulator, accumulator, mask);
            Value packed = mlir::arith::TruncFOp::create(
                builder, packed_output_type, values);
            Value output_row = Add(
                builder, wave_row_base,
                Add(builder,
                    Mul(builder, lane_group,
                        IndexConstant(builder, kPackedRows)),
                    IndexConstant(builder,
                                  tile_row * tile_row_stride +
                                      (use_mfma_32_ ? group * 8 : 0))));
            output = mlir::vector::TransferWriteOp::create(
                         builder, packed, output,
                         output_indices(output_row, output_column),
                         llvm::ArrayRef<bool>{true})
                         .getResult();
          }
        }
      }
    } else if (triton_xf32_paired_lds) {
      // Source-swapping the wide XF32 MFMA matches Triton's native output
      // view: each consecutive group of four accumulator elements is four
      // adjacent N columns.  Keep that view through the epilogue so LLVM can
      // select one global_store_dwordx4 instead of four scalar stores.
      TF_RET_CHECK(output_type.isF32() && m_ % block_m_ == 0 &&
                   n_ % block_n_ == 0 &&
                   output_physical_strides_.back() == 1 &&
                   !is_scaled_dot_ && epilogue_steps_.empty());
      constexpr int64_t kPackedColumns = 4;
      auto packed_output_type =
          mlir::VectorType::get({kPackedColumns}, output_type);
      for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
        Value output_row =
            Add(builder, wave_row_base,
                Add(builder, lane_axis,
                    IndexConstant(builder, tile_row * tile_row_stride)));
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             ++tile_column) {
          Value accumulator = final_accumulators[
              get_accumulator_index(tile_row, tile_column)];
          for (int64_t group = 0;
               group < accumulator_elements / kPackedColumns; ++group) {
            llvm::SmallVector<int64_t, kPackedColumns> mask;
            for (int64_t element = 0; element < kPackedColumns; ++element) {
              mask.push_back(group * kPackedColumns + element);
            }
            Value packed = mlir::vector::ShuffleOp::create(
                builder, packed_output_type, accumulator, accumulator, mask);
            Value output_column = Add(
                builder, wave_column_base,
                Add(builder,
                    Mul(builder, lane_group,
                        IndexConstant(builder, kPackedColumns)),
                    IndexConstant(builder, tile_column * tile_column_stride +
                                               group * 8)));
            output = mlir::vector::TransferWriteOp::create(
                         builder, packed, output,
                         output_indices(output_row, output_column),
                         llvm::ArrayRef<bool>{true})
                         .getResult();
          }
        }
      }
    } else if (row_major_rhs_wide_source_swap) {
      // Source-swapping the 32x32 MFMA computes the transposed fragment view.
      // After transposing that view back into XLA's row-major output, every
      // consecutive group of four accumulator elements is four adjacent N
      // columns. Store those groups directly instead of scalarizing the
      // native accumulator layout or round-tripping C through LDS.
      TF_RET_CHECK(output_type.isBF16() && m_ % block_m_ == 0 &&
                   n_ % block_n_ == 0 &&
                   output_physical_strides_.back() == 1 &&
                   !is_scaled_dot_ && epilogue_steps_.empty());
      constexpr int64_t kPackedColumns = 4;
      auto packed_f32_type =
          mlir::VectorType::get({kPackedColumns}, builder.getF32Type());
      auto packed_output_type =
          mlir::VectorType::get({kPackedColumns}, output_type);
      for (int64_t tile_row = 0; tile_row < wave_tile_rows; ++tile_row) {
        Value output_row =
            Add(builder, wave_row_base,
                Add(builder, lane_axis,
                    IndexConstant(builder, tile_row * tile_row_stride)));
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             ++tile_column) {
          Value accumulator = final_accumulators[
              get_accumulator_index(tile_row, tile_column)];
          for (int64_t group = 0;
               group < accumulator_elements / kPackedColumns; ++group) {
            llvm::SmallVector<int64_t, kPackedColumns> mask;
            for (int64_t element = 0; element < kPackedColumns; ++element) {
              mask.push_back(group * kPackedColumns + element);
            }
            Value values = mlir::vector::ShuffleOp::create(
                builder, packed_f32_type, accumulator, accumulator, mask);
            Value packed = mlir::arith::TruncFOp::create(
                builder, packed_output_type, values);
            Value output_column = Add(
                builder, wave_column_base,
                Add(builder,
                    Mul(builder, lane_group,
                        IndexConstant(builder, kPackedColumns)),
                    IndexConstant(builder, tile_column * tile_column_stride +
                                               group * 8)));
            output = mlir::vector::TransferWriteOp::create(
                         builder, packed, output,
                         output_indices(output_row, output_column),
                         llvm::ArrayRef<bool>{true})
                         .getResult();
          }
        }
      }
    } else if (source_swap_mfma) {
      TF_RET_CHECK(!stage_output_ && (n_ % 16 == 0 || shift_n_edge) &&
                   (m_ % 16 == 0 || shift_m_edge));
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
      auto pack_accumulator_f16 = [&](Value accumulator) -> Value {
        mlir::Type i32_type = builder.getI32Type();
        auto packed_i32_type = mlir::VectorType::get({2}, i32_type);
        mlir::Type asm_result_type = mlir::LLVM::LLVMStructType::getLiteral(
            builder.getContext(), {i32_type, i32_type});
        llvm::SmallVector<Value, 4> operands;
        operands.reserve(accumulator_elements);
        for (int64_t element = 0; element < accumulator_elements; ++element) {
          operands.push_back(mlir::vector::ExtractOp::create(
              builder, accumulator, llvm::SmallVector<int64_t>{element}));
        }
        // Narrow one accumulator at a time. Expressing vector<4xf32> ->
        // vector<4xf16> through arith.truncf made LLVM keep a complete
        // 56-value output row live. Tie each packed result to its first source
        // so an accumulator already resident in VGPRs is converted in place;
        // forcing an AGPR operand here inserted an avoidable write/read pair.
        Value packed =
            mlir::LLVM::InlineAsmOp::create(
                builder, asm_result_type, operands,
                "v_cvt_pkrtz_f16_f32 $0, $0, $3;\n"
                "v_cvt_pkrtz_f16_f32 $1, $1, $5;\n",
                "=v,=v,0,v,1,v",
                /*has_side_effects=*/false,
                /*is_align_stack=*/false, mlir::LLVM::TailCallKind::None,
                mlir::LLVM::AsmDialectAttr::get(
                    builder.getContext(), mlir::LLVM::AsmDialect::AD_ATT),
                /*operand_attrs=*/mlir::ArrayAttr())
                .getResult(0);
        Value packed_words = mlir::vector::FromElementsOp::create(
            builder, packed_i32_type,
            mlir::ValueRange{
                mlir::LLVM::ExtractValueOp::create(builder, packed, 0),
                mlir::LLVM::ExtractValueOp::create(builder, packed, 1)});
        return mlir::vector::BitCastOp::create(builder, packed_output_type,
                                               packed_words);
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
                    builder, accumulator, llvm::SmallVector<int64_t>{element}));
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
                          ", $" + std::to_string(value_count + index) + ";\n";
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
              values.push_back(
                  mlir::LLVM::ExtractValueOp::create(builder, result, index));
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
          Value bits = mlir::arith::BitcastOp::create(builder, i32_type, value);
          Value retained_lsb = mlir::arith::AndIOp::create(
              builder, mlir::arith::ShRUIOp::create(builder, bits, shift_16),
              one);
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
        Value permute_mask =
            mlir::arith::ConstantIntOp::create(builder, i32_type, 0x07060302);
        for (int64_t element = 0; element < accumulator_elements;
             element += 2) {
          packed_words.push_back(
              mlir::LLVM::InlineAsmOp::create(
                  builder, i32_type,
                  mlir::ValueRange{rounded_bits[element + 1],
                                   rounded_bits[element], permute_mask},
                  "v_perm_b32 $0, $1, $2, $3;\n", "=v,v,v,s",
                  /*has_side_effects=*/false,
                  /*is_align_stack=*/false, mlir::LLVM::TailCallKind::None,
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
      auto pack_vgpr_16_bit_float =
          [&](llvm::ArrayRef<Value> values) -> Value {
        CHECK_EQ(values.size(), accumulator_elements);
        if (output_type.isBF16()) {
          return pack_vgpr_bf16(values);
        }
        CHECK(output_type.isF16());
        Value values_vector = mlir::vector::FromElementsOp::create(
            builder, accumulator_type, values);
        return mlir::arith::TruncFOp::create(builder, packed_output_type,
                                             values_vector);
      };
      llvm::SmallVector<llvm::SmallVector<Value>> column_epilogue_operands(
          epilogue_steps_.size());
      for (int64_t step_index = 0; step_index < epilogue_steps_.size();
           ++step_index) {
        if (epilogue_steps_[step_index].broadcast_dimension !=
            epilogue_column_dimension_) {
          continue;
        }
        column_epilogue_operands[step_index].reserve(wave_tile_columns);
        for (int64_t tile_column = 0; tile_column < wave_tile_columns;
             ++tile_column) {
          Value output_column = Add(
              builder, wave_column_base,
              Add(builder, Mul(builder, lane_group, IndexConstant(builder, 4)),
                          IndexConstant(builder, tile_column * tile_column_stride)));
          llvm::SmallVector<Value, 4> elements;
          for (int64_t element = 0; element < accumulator_elements; ++element) {
            elements.push_back(load_epilogue_element(
                step_index, wave_row_base,
                Add(builder, output_column, IndexConstant(builder, element))));
          }
          column_epilogue_operands[step_index].push_back(
              mlir::vector::FromElementsOp::create(builder, accumulator_type,
                                                   elements));
        }
      }
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
                          IndexConstant(builder, tile_row * tile_row_stride)));
        llvm::SmallVector<Value> row_accumulator_values;
        if (((tensile_direct_lhs && output_type.isBF16()) ||
             (is_scaled_dot_ && !block_scaled_dot_) ||
             !epilogue_steps_.empty()) &&
            output_is_16_bit_float) {
          llvm::SmallVector<Value> row_accumulators;
          row_accumulators.reserve(wave_tile_columns);
          for (int64_t tile_column = 0; tile_column < wave_tile_columns;
               ++tile_column) {
            row_accumulators.push_back(final_accumulators[get_accumulator_index(
                tile_row, tile_column)]);
          }
          row_accumulator_values = read_accumulators_to_vgprs(row_accumulators);
          if (is_scaled_dot_ && !block_scaled_dot_) {
            for (Value& value : row_accumulator_values) {
              value = mlir::arith::MulFOp::create(builder, value, output_scale);
            }
          }
          if (!epilogue_steps_.empty() && !has_vector_epilogue_) {
            for (Value& value : row_accumulator_values) {
              value = apply_scalar_epilogue(value);
            }
          }
        }
        auto emit_row_stores = [&](Value destination) {
          Value updated = destination;
          for (int64_t tile_column = 0; tile_column < wave_tile_columns;
               ++tile_column) {
            const int64_t accumulator_index =
                get_accumulator_index(tile_row, tile_column);
            Value output_column =
                Add(builder, wave_column_base,
                    Add(builder,
                        Mul(builder, lane_group, IndexConstant(builder, 4)),
                          IndexConstant(builder, tile_column * tile_column_stride)));
            Value packed_result = final_accumulators[accumulator_index];
            const bool direct_accumulator_store =
                tensile_direct_lhs && output_type.isF32() &&
                !is_scaled_dot_ && epilogue_steps_.empty() &&
                output_physical_strides_.back() == 1 &&
                direct_accumulator_store_offsets_fit_;
            if (output_type.isF32() && !direct_accumulator_store) {
              // Source-swapped MFMAs leave their results in AGPRs. Materialize
              // one four-value accumulator at a time because vector stores
              // require VGPR sources. This is explicit for the tied FP8 MFMA;
              // otherwise LLVM can drop an AGPR-valued store during selection.
              // The same narrow lifetime also keeps scaled F32 output from
              // making the complete 4x14 tile simultaneously live in VGPRs.
              llvm::SmallVector<Value> values = read_accumulators_to_vgprs(
                  llvm::ArrayRef<Value>(
                      final_accumulators[accumulator_index]));
              if (is_scaled_dot_ && !block_scaled_dot_) {
                for (Value& value : values) {
                  value =
                      mlir::arith::MulFOp::create(builder, value, output_scale);
                }
              }
              packed_result = mlir::vector::FromElementsOp::create(
                  builder, accumulator_type, values);
            }
            if (output_is_16_bit_float) {
              if (row_accumulator_values.empty()) {
                packed_result = output_type.isBF16()
                                    ? pack_accumulator_bf16(packed_result)
                                    : pack_accumulator_f16(packed_result);
              } else {
                llvm::SmallVector<Value, 4> values(
                    llvm::ArrayRef<Value>(row_accumulator_values)
                        .slice(tile_column * accumulator_elements,
                               accumulator_elements));
                if (has_vector_epilogue_) {
                  Value epilogue_value = mlir::vector::FromElementsOp::create(
                      builder, accumulator_type, values);
                  const int64_t step_count =
                      epilogue_steps_.size() - (fold_final_negate_ ? 1 : 0);
                  for (int64_t step_index = 0; step_index < step_count;
                       ++step_index) {
                    Value operand;
                    const EpilogueStep& step = epilogue_steps_[step_index];
                    if (step.opcode != HloOpcode::kNegate) {
                      operand =
                          step.broadcast_dimension == epilogue_column_dimension_
                              ? column_epilogue_operands[step_index]
                                                        [tile_column]
                              : load_epilogue_element(step_index, output_row,
                                                      output_column);
                    }
                    epilogue_value = apply_epilogue_step(epilogue_value,
                                                         step_index, operand);
                  }
                  for (int64_t element = 0; element < accumulator_elements;
                       ++element) {
                    values[element] = mlir::vector::ExtractOp::create(
                        builder, epilogue_value,
                        llvm::SmallVector<int64_t>{element});
                  }
                }
                packed_result = pack_vgpr_16_bit_float(values);
              }
            }
            if (direct_accumulator_store) {
              Value output_linear_index = physical_linear_index(
                  output_indices(output_row, output_column),
                  output_physical_strides_, /*offset=*/0);
              updated = emit_accumulator_store(
                  packed_result, updated, output_linear_index);
            } else if (tensile_direct_lhs && !shift_n_edge &&
                n_ % block_n_ != 0 &&
                output_physical_strides_.back() == 1) {
              Value output_linear_index = physical_linear_index(
                  output_indices(output_row, output_column),
                  output_physical_strides_, /*offset=*/0);
              // N is a multiple of the 16-column MFMA atom. The prefix that
              // also exists in the final partial macro-tile is therefore
              // always safe; only the remaining column fragments need an
              // explicit row-bound mask. Allocation bounds alone are not
              // sufficient because an invalid column can alias the next row.
              if (tile_column * tile_column_stride >= n_ % block_n_) {
                Value column_in_bounds = mlir::arith::CmpIOp::create(
                    builder, mlir::arith::CmpIPredicate::ult, output_column,
                    IndexConstant(builder, n_));
                output_linear_index = mlir::arith::SelectOp::create(
                    builder, column_in_bounds, output_linear_index,
                    IndexConstant(builder, 1LL << 30));
              }
              Value stored_result = packed_result;
              if (output_is_16_bit_float) {
                stored_result = mlir::vector::BitCastOp::create(
                    builder, mlir::VectorType::get({2}, builder.getI32Type()),
                    stored_result);
              }
              updated = emit_buffer_store(stored_result, updated,
                                          output_linear_index);
            } else {
              updated = mlir::vector::TransferWriteOp::create(
                            builder, packed_result, updated,
                            output_indices(output_row, output_column),
                            llvm::ArrayRef<bool>{true})
                            .getResult();
            }
          }
          return updated;
        };
        if (m_ % block_m_ == 0 || shift_m_edge) {
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
          Value output_column =
              Add(builder, wave_column_base,
                  Add(builder, lane_axis,
                          IndexConstant(builder, tile_column * tile_column_stride)));
          const bool vectorized_epilogue =
              has_vector_epilogue_ && !has_row_epilogue_;
          if (vectorized_epilogue) {
            // Every value in a generic MFMA accumulator shares its output
            // column. Load a column scale/bias once and broadcast it across
            // the accumulator instead of issuing one load per output row.
            accumulator = apply_output_epilogue(accumulator, wave_row_base,
                                                output_column);
          }
          for (int64_t element = 0; element < accumulator_elements; ++element) {
            Value result = mlir::vector::ExtractOp::create(
                builder, accumulator, llvm::SmallVector<int64_t>{element});
            const int64_t element_row =
                use_mfma_32 ? (element / 4) * 8 + element % 4 : element;
            Value output_row = Add(
                builder, wave_row_base,
                Add(builder,
                    Mul(builder, lane_group, IndexConstant(builder, 4)),
                  IndexConstant(builder, tile_row * tile_row_stride + element_row)));
            if (has_vector_epilogue_ && !vectorized_epilogue) {
              result = apply_output_epilogue(result, output_row, output_column);
            }
            if (output_type.isBF16() || output_type.isF16()) {
              result =
                  mlir::arith::TruncFOp::create(builder, output_type, result);
            }
            llvm::SmallVector<Value, 3> indices =
                output_indices(output_row, output_column);
            if (stage_output_via_lds) {
              Value local_row = Add(
                  builder, wave_row_offset,
                  Add(builder,
                      Mul(builder, lane_group, IndexConstant(builder, 4)),
                  IndexConstant(builder, tile_row * tile_row_stride + element_row)));
              Value local_column =
                  Add(builder, wave_column_offset,
                      Add(builder, lane_axis,
                          IndexConstant(builder, tile_column * tile_column_stride)));
              Value shared_index =
                  Add(builder,
                      Mul(builder, local_row, IndexConstant(builder, block_n_)),
                      local_column);
              staged_output = mlir::tensor::InsertOp::create(
                  builder, result, staged_output, shared_index);
            } else if ((m_ % block_m_ == 0 || shift_m_edge) &&
                       (n_ % block_n_ == 0 || shift_n_edge)) {
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

    if (stage_output_via_lds) {
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
                output_indices(Add(builder, block_m_base, local_row),
                               Add(builder, block_n_base, local_column)),
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
  int64_t atom_k_ = 16;
  FlyGemmConfig::MfmaAtom mfma_atom_ = FlyGemmConfig::FLY_MFMA_16X16X16;
  bool use_mfma_32_ = false;
  bool is_fp8_ = false;
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
  bool dequantize_block_scales_ = false;
  int32_t workgroup_mapping_n_ = 0;
  bool global_split_k_ = false;
  bool batched_gemm_ = false;
  bool is_scaled_dot_ = false;
  bool block_scaled_dot_ = false;
  bool has_vector_epilogue_ = false;
  bool has_row_epilogue_ = false;
  int64_t epilogue_row_dimension_ = 0;
  int64_t epilogue_column_dimension_ = 1;
  bool fold_final_negate_ = false;
  // global_store's vector offset is 32 bits. Larger tensors use the regular
  // store path, which computes a full-width address after reading the AGPRs.
  bool direct_accumulator_store_offsets_fit_ = false;
  int64_t fusion_parameter_count_ = 0;
  std::vector<int64_t> lhs_parameter_numbers_ = {0};
  std::vector<int64_t> rhs_parameter_numbers_ = {1};
  int64_t lhs_scale_parameter_number_ = 2;
  int64_t rhs_scale_parameter_number_ = 3;
  PrimitiveType lhs_scale_element_type_ = BF16;
  PrimitiveType rhs_scale_element_type_ = BF16;
  std::vector<int64_t> lhs_scale_dimensions_;
  std::vector<int64_t> rhs_scale_dimensions_;
  PrimitiveType lhs_element_type_ = BF16;
  PrimitiveType rhs_element_type_ = BF16;
  PrimitiveType lhs_input_type_ = BF16;
  PrimitiveType rhs_input_type_ = BF16;
  bool lhs_requires_linear_addressing_ = false;
  bool rhs_requires_linear_addressing_ = false;
  int64_t lhs_physical_offset_ = 0;
  int64_t rhs_physical_offset_ = 0;
  std::vector<int64_t> lhs_physical_strides_;
  std::vector<int64_t> rhs_physical_strides_;
  std::vector<int64_t> lhs_dynamic_offset_parameter_numbers_;
  std::vector<int64_t> lhs_dynamic_offset_constants_;
  std::vector<int64_t> lhs_dynamic_offset_limits_;
  std::vector<int64_t> rhs_dynamic_offset_parameter_numbers_;
  std::vector<int64_t> rhs_dynamic_offset_constants_;
  std::vector<int64_t> rhs_dynamic_offset_limits_;
  std::vector<int64_t> output_physical_strides_;
  std::optional<int64_t> output_transpose_batch_inner_;
  int64_t lhs_concat_fragment_size_ = 0;
  int64_t rhs_concat_fragment_size_ = 0;
  std::vector<EpilogueStep> epilogue_steps_;
  int64_t split_k_batches_ = 1;
  int64_t batch_count_ = 1;
  int64_t lhs_batch_dimension_ = 0;
  int64_t rhs_batch_dimension_ = 0;
  int64_t lhs_contracting_dimension_ = 0;
  int64_t rhs_contracting_dimension_ = 0;
  int64_t lhs_noncontracting_dimension_ = 0;
  int64_t rhs_noncontracting_dimension_ = 0;
  bool lhs_k_contiguous_ = false;
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
        (config.output_tiles(0).sizes_size() == 2 ||
         config.output_tiles(0).sizes_size() == 3)) {
      const int64_t tile_rank = config.output_tiles(0).sizes_size();
      block_m_ = config.output_tiles(0).sizes(tile_rank - 2);
      block_n_ = config.output_tiles(0).sizes(tile_rank - 1);
      num_warps_ = config.num_warps();
    }
    if (analysis.fusion_backend_config().has_fly_gemm_config()) {
      const FlyGemmConfig& fly_config =
          analysis.fusion_backend_config().fly_gemm_config();
      outputs_per_wave_ =
          std::clamp<int64_t>(fly_config.gemv_outputs_per_wave(), 1, 8);
      k_vector_width_ =
          std::clamp<int64_t>(fly_config.gemv_k_vector_width(), 1, 4);
      split_k_ = fly_config.gemv_split_k();
    }
    const HloInstruction& root = analysis.fusion_root(0).instruction();
    fusion_parameter_count_ = root.parent()->num_parameters();
    const HloInstruction* dot = FindContraction(root);
    CHECK_NE(dot, nullptr);
    epilogue_steps_ = FindEpilogueSteps(root, *dot);
    epilogue_row_dimension_ = dot->shape().dimensions_size() - 2;
    epilogue_column_dimension_ = dot->shape().dimensions_size() - 1;
    is_scaled_dot_ = dot->opcode() == HloOpcode::kScaledDot;
    if (is_scaled_dot_) {
      CHECK_EQ(dot->operand(2)->opcode(), HloOpcode::kParameter);
      CHECK_EQ(dot->operand(3)->opcode(), HloOpcode::kParameter);
      CHECK_EQ(ShapeUtil::ElementsIn(dot->operand(2)->shape()), 1);
      CHECK_EQ(ShapeUtil::ElementsIn(dot->operand(3)->shape()), 1);
      lhs_scale_parameter_number_ = dot->operand(2)->parameter_number();
      rhs_scale_parameter_number_ = dot->operand(3)->parameter_number();
    }
    const ContractionInput lhs_input = FindContractionInput(*dot->operand(0));
    CHECK_EQ(lhs_input.parameter_numbers.size(), 1);
    lhs_parameter_number_ = lhs_input.parameter_numbers.front();
    lhs_input_type_ = lhs_input.parameter_type;
    lhs_requires_linear_addressing_ = lhs_input.requires_linear_addressing;
    lhs_physical_offset_ = lhs_input.physical_offset;
    lhs_physical_strides_ = lhs_input.physical_strides;
    lhs_dynamic_offset_parameter_numbers_ =
        lhs_input.dynamic_offset_parameter_numbers;
    lhs_dynamic_offset_constants_ = lhs_input.dynamic_offset_constants;
    lhs_dynamic_offset_limits_ = lhs_input.dynamic_offset_limits;
    const ContractionInput rhs_input = FindContractionInput(*dot->operand(1));
    CHECK_EQ(rhs_input.parameter_numbers.size(), 1);
    rhs_parameter_number_ = rhs_input.parameter_numbers.front();
    rhs_input_type_ = rhs_input.parameter_type;
    rhs_requires_linear_addressing_ = rhs_input.requires_linear_addressing;
    rhs_physical_offset_ = rhs_input.physical_offset;
    rhs_physical_strides_ = rhs_input.physical_strides;
    rhs_dynamic_offset_parameter_numbers_ =
        rhs_input.dynamic_offset_parameter_numbers;
    rhs_dynamic_offset_constants_ = rhs_input.dynamic_offset_constants;
    rhs_dynamic_offset_limits_ = rhs_input.dynamic_offset_limits;
    const DotDimensionNumbers& dot_dims = dot->dot_dimension_numbers();
    rank_three_ = dot->shape().dimensions_size() == 3;
    global_split_k_ = rank_three_ &&
                      dot_dims.lhs_batch_dimensions_size() == 1 &&
                      dot_dims.lhs_batch_dimensions(0) == 1;
    batched_gemv_ = rank_three_ && dot_dims.lhs_batch_dimensions_size() == 1 &&
                    dot_dims.lhs_batch_dimensions(0) == 0;
    CHECK(!rank_three_ || global_split_k_ || batched_gemv_);
    batch_count_ = rank_three_ ? dot->shape().dimensions(0) : 1;
    m_ = dot->shape().dimensions(rank_three_ ? 1 : 0);
    n_ = dot->shape().dimensions(rank_three_ ? 2 : 1);
    k_ = dot->operand(0)->shape().dimensions(
        dot_dims.lhs_contracting_dimensions(0));
    rhs_contracting_dimension_ = dot_dims.rhs_contracting_dimensions(0);
    const int64_t rhs_column_dimension =
        global_split_k_
            ? 0
            : (batched_gemv_ ? (rhs_contracting_dimension_ == 1 ? 2 : 1)
                             : 1 - rhs_contracting_dimension_);
    rhs_column_contiguous_ = rhs_physical_strides_[rhs_column_dimension] == 1;
    rhs_k_contiguous_gemv_ =
        rhs_physical_strides_[rhs_contracting_dimension_] == 1;
    blocks_per_batch_ = m_ == 1 ? (n_ + block_n_ - 1) / block_n_
                                : (m_ + block_m_ - 1) / block_m_;
    launch_dimensions_ =
        LaunchDimensions(se::BlockDim(batch_count_ * blocks_per_batch_, 1, 1),
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
    TF_RET_CHECK(entry_function.getNumArguments() ==
                 fusion_parameter_count_ + 1);
    TF_RET_CHECK(m_ == 1 || n_ == 1);

    mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
    builder.setInsertionPointToStart(entry_function.addEntryBlock());
    Value lhs = entry_function.getArgument(lhs_parameter_number_);
    Value rhs = entry_function.getArgument(rhs_parameter_number_);
    Value output = entry_function.getArgument(fusion_parameter_count_);
    Value thread_id = EmitThreadId(builder, 0);
    Value block_id = EmitBlockId(builder, 0);
    Value batch_id = rank_three_ ? Div(builder, block_id, blocks_per_batch_)
                                 : IndexConstant(builder, 0);
    Value output_block_id =
        rank_three_ ? Rem(builder, block_id, blocks_per_batch_) : block_id;
    Value wave_id = Div(builder, thread_id, 64);
    Value lane_id = Rem(builder, thread_id, 64);
    Value zero = mlir::arith::ConstantFloatOp::create(
        builder, builder.getF32Type(), llvm::APFloat(0.0f));
    Value output_scale;
    if (is_scaled_dot_) {
      auto load_scale = [&](int64_t parameter_number) {
        Value tensor = entry_function.getArgument(parameter_number);
        auto tensor_type = mlir::cast<mlir::RankedTensorType>(tensor.getType());
        llvm::SmallVector<Value> indices(tensor_type.getRank(),
                                         IndexConstant(builder, 0));
        Value scale = mlir::tensor::ExtractOp::create(
            builder, tensor, mlir::ValueRange{indices});
        mlir::Type type = scale.getType();
        if (type.isF32()) {
          return scale;
        }
        if (type.isInteger(8)) {
          return mlir::arith::SIToFPOp::create(builder, builder.getF32Type(),
                                               scale)
              .getResult();
        }
        return mlir::arith::ExtFOp::create(builder, builder.getF32Type(), scale)
            .getResult();
      };
      output_scale = mlir::arith::MulFOp::create(
          builder, load_scale(lhs_scale_parameter_number_),
          load_scale(rhs_scale_parameter_number_));
    }
    llvm::SmallVector<Value> epilogue_inputs(epilogue_steps_.size());
    llvm::SmallVector<Value> scalar_epilogue_values(epilogue_steps_.size());
    for (int64_t index = 0; index < epilogue_steps_.size(); ++index) {
      const EpilogueStep& step = epilogue_steps_[index];
      if (step.opcode == HloOpcode::kNegate) {
        continue;
      }
      if (step.input_is_constant) {
        TF_RET_CHECK(step.broadcast_dimension == -1);
        scalar_epilogue_values[index] =
            mlir::arith::ConstantFloatOp::create(
                builder, builder.getF32Type(),
                llvm::APFloat(static_cast<float>(step.constant_value)));
        continue;
      }
      epilogue_inputs[index] =
          entry_function.getArgument(step.parameter_number);
      if (step.broadcast_dimension == -1) {
        Value scalar = mlir::tensor::ExtractOp::create(
            builder, epilogue_inputs[index], mlir::ValueRange{});
        if (IsHalfPrecisionFloat(step.input_type)) {
          scalar = mlir::arith::ExtFOp::create(builder, builder.getF32Type(),
                                               scalar);
        } else {
          TF_RET_CHECK(step.input_type == F32);
        }
        scalar_epilogue_values[index] = scalar;
      }
    }

    auto physical_linear_index = [&](mlir::ValueRange indices,
                                     llvm::ArrayRef<int64_t> strides,
                                     int64_t offset) {
      CHECK_EQ(indices.size(), strides.size());
      Value linear = IndexConstant(builder, offset);
      for (int64_t dimension = 0; dimension < strides.size(); ++dimension) {
        linear = Add(builder, linear,
                     Mul(builder, indices[dimension],
                         IndexConstant(builder, strides[dimension])));
      }
      return linear;
    };
    auto emit_dynamic_offsets = [&](llvm::ArrayRef<int64_t> parameter_numbers,
                                    llvm::ArrayRef<int64_t> constants,
                                    llvm::ArrayRef<int64_t> limits) {
      CHECK_EQ(parameter_numbers.size(), constants.size());
      CHECK_EQ(parameter_numbers.size(), limits.size());
      llvm::SmallVector<Value, 3> offsets;
      offsets.reserve(parameter_numbers.size());
      for (int64_t dimension = 0; dimension < parameter_numbers.size();
           ++dimension) {
        if (parameter_numbers[dimension] < 0) {
          offsets.push_back(IndexConstant(builder, constants[dimension]));
          continue;
        }
        Value scalar = mlir::tensor::ExtractOp::create(
            builder, entry_function.getArgument(parameter_numbers[dimension]),
            mlir::ValueRange{});
        Value offset = mlir::arith::IndexCastOp::create(
            builder, builder.getIndexType(), scalar);
        Value zero_index = IndexConstant(builder, 0);
        Value limit = IndexConstant(builder, limits[dimension]);
        Value below_zero = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::slt, offset, zero_index);
        offset = mlir::arith::SelectOp::create(builder, below_zero, zero_index,
                                               offset);
        Value above_limit = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::sgt, offset, limit);
        offsets.push_back(
            mlir::arith::SelectOp::create(builder, above_limit, limit, offset));
      }
      return offsets;
    };
    llvm::SmallVector<Value, 3> lhs_dynamic_offsets = emit_dynamic_offsets(
        lhs_dynamic_offset_parameter_numbers_, lhs_dynamic_offset_constants_,
        lhs_dynamic_offset_limits_);
    llvm::SmallVector<Value, 3> rhs_dynamic_offsets = emit_dynamic_offsets(
        rhs_dynamic_offset_parameter_numbers_, rhs_dynamic_offset_constants_,
        rhs_dynamic_offset_limits_);
    auto add_dynamic_offsets = [&](llvm::SmallVectorImpl<Value>& indices,
                                   mlir::ValueRange offsets) {
      CHECK(offsets.empty() || indices.size() == offsets.size());
      for (int64_t dimension = 0; dimension < offsets.size(); ++dimension) {
        indices[dimension] =
            Add(builder, indices[dimension], offsets[dimension]);
      }
    };
    auto lhs_indices = [&](Value row, Value k) {
      llvm::SmallVector<Value, 3> indices;
      if (global_split_k_) {
        indices.assign({row, batch_id, k});
      } else if (batched_gemv_) {
        indices.assign({batch_id, row, k});
      } else {
        indices.assign({row, k});
      }
      add_dynamic_offsets(indices, lhs_dynamic_offsets);
      return indices;
    };
    auto rhs_indices = [&](Value k, Value column) {
      llvm::SmallVector<Value, 3> indices;
      if (global_split_k_) {
        indices.assign({column, batch_id, k});
      } else if (batched_gemv_) {
        if (rhs_contracting_dimension_ == 1) {
          indices.assign({batch_id, k, column});
        } else {
          indices.assign({batch_id, column, k});
        }
      } else if (rhs_contracting_dimension_ == 0) {
        indices.assign({k, column});
      } else {
        indices.assign({column, k});
      }
      add_dynamic_offsets(indices, rhs_dynamic_offsets);
      return indices;
    };
    auto output_indices = [&](Value row, Value column) {
      llvm::SmallVector<Value, 3> indices;
      if (rank_three_) {
        indices.assign({batch_id, row, column});
      } else {
        indices.assign({row, column});
      }
      return indices;
    };
    const bool has_column_output_tail = m_ == 1 && n_ % block_n_ != 0;
    const bool has_row_output_tail = n_ == 1 && m_ % block_m_ != 0;
    auto safe_output_column = [&](Value column) -> Value {
      if (!has_column_output_tail) {
        return column;
      }
      Value in_bounds = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, column,
          IndexConstant(builder, n_));
      return mlir::arith::SelectOp::create(
                 builder, in_bounds, column, IndexConstant(builder, 0))
          .getResult();
    };
    auto safe_output_row = [&](Value row) -> Value {
      if (!has_row_output_tail) {
        return row;
      }
      Value in_bounds = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, row,
          IndexConstant(builder, m_));
      return mlir::arith::SelectOp::create(
                 builder, in_bounds, row, IndexConstant(builder, 0))
          .getResult();
    };
    auto apply_epilogue_step = [&](Value value, int64_t step_index,
                                   Value operand) -> Value {
      const EpilogueStep& step = epilogue_steps_[step_index];
      if (step.round_input_type != PRIMITIVE_TYPE_INVALID) {
        mlir::Type rounded_type = step.round_input_type == F16
                                      ? builder.getF16Type()
                                      : builder.getBF16Type();
        value = mlir::arith::TruncFOp::create(builder, rounded_type, value);
        value = mlir::arith::ExtFOp::create(builder, builder.getF32Type(),
                                            value);
      }
      if (step.opcode == HloOpcode::kNegate) {
        return mlir::arith::NegFOp::create(builder, value);
      }
      if (step.opcode == HloOpcode::kAdd) {
        return mlir::arith::AddFOp::create(builder, value, operand);
      }
      if (step.opcode == HloOpcode::kMultiply) {
        return mlir::arith::MulFOp::create(builder, value, operand);
      }
      Value lhs = step.contraction_operand == 0 ? value : operand;
      Value rhs = step.contraction_operand == 0 ? operand : value;
      if (step.opcode == HloOpcode::kSubtract) {
        return mlir::arith::SubFOp::create(builder, lhs, rhs);
      }
      if (step.opcode == HloOpcode::kDivide) {
        return mlir::arith::DivFOp::create(builder, lhs, rhs);
      }
      if (step.opcode == HloOpcode::kMaximum) {
        return mlir::arith::MaximumFOp::create(builder, lhs, rhs);
      }
      CHECK_EQ(step.opcode, HloOpcode::kMinimum);
      return mlir::arith::MinimumFOp::create(builder, lhs, rhs);
    };
    auto load_epilogue_element = [&](int64_t step_index, Value row,
                                     Value column) -> Value {
      const EpilogueStep& step = epilogue_steps_[step_index];
      CHECK_NE(step.opcode, HloOpcode::kNegate);
      if (step.broadcast_dimension == -1) {
        return scalar_epilogue_values[step_index];
      }
      Value index;
      int64_t dimension_size;
      if (step.broadcast_dimension == epilogue_row_dimension_) {
        index = row;
        dimension_size = m_;
      } else if (step.broadcast_dimension == epilogue_column_dimension_) {
        index = column;
        dimension_size = n_;
      } else {
        CHECK(batched_gemv_ && step.broadcast_dimension == 0);
        index = batch_id;
        dimension_size = batch_count_;
      }
      Value in_bounds = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::ult, index,
          IndexConstant(builder, dimension_size));
      Value safe_index = mlir::arith::SelectOp::create(
          builder, in_bounds, index, IndexConstant(builder, 0));
      Value element = mlir::tensor::ExtractOp::create(
          builder, epilogue_inputs[step_index], mlir::ValueRange{safe_index});
      if (IsHalfPrecisionFloat(step.input_type)) {
        element =
            mlir::arith::ExtFOp::create(builder, builder.getF32Type(), element);
      } else {
        CHECK_EQ(step.input_type, F32);
      }
      return element;
    };
    auto apply_output_epilogue = [&](Value value, Value row,
                                     Value column) -> Value {
      for (int64_t step_index = 0; step_index < epilogue_steps_.size();
           ++step_index) {
        Value operand;
        if (epilogue_steps_[step_index].opcode != HloOpcode::kNegate) {
          operand = load_epilogue_element(step_index, row, column);
        }
        value = apply_epilogue_step(value, step_index, operand);
      }
      return value;
    };
    auto extract_input = [&](Value source, PrimitiveType source_type,
                             bool requires_linear_addressing,
                             int64_t physical_offset,
                             llvm::ArrayRef<int64_t> strides,
                             mlir::ValueRange indices) {
      Value element;
      if (requires_linear_addressing) {
        mlir::Type element_type = source_type == F32
                                      ? builder.getF32Type()
                                      : ContractionElementType(builder,
                                                               source_type);
        // AMD's raw-buffer lowering cannot scalarize a one-element vector on
        // this path. Load the smallest ordinary vector and consume its first
        // element; descriptor bounds make the unused lane safe at an edge.
        auto vector_type = mlir::VectorType::get({2}, element_type);
        mlir::OperationState load_state(entry_function.getLoc(),
                                        "xla_gpu.buffer_load");
        load_state.addOperands(
            {source, physical_linear_index(indices, strides, physical_offset)});
        load_state.addTypes(vector_type);
        Value loaded = builder.create(load_state)->getResult(0);
        element = mlir::vector::ExtractOp::create(
            builder, loaded, llvm::SmallVector<int64_t>{0});
      } else {
        element = mlir::tensor::ExtractOp::create(builder, source, indices);
      }
      if (source_type == F32) {
        element = mlir::arith::TruncFOp::create(builder, builder.getBF16Type(),
                                                element);
      } else {
        CHECK(IsHalfPrecisionFloat(source_type) || IsFnuzFp8(source_type));
      }
      return mlir::arith::ExtFOp::create(builder, builder.getF32Type(), element)
          .getResult();
    };
    auto extract_input_vector = [&](Value source, PrimitiveType source_type,
                                    int64_t physical_offset,
                                    llvm::ArrayRef<int64_t> strides,
                                    mlir::ValueRange indices,
                                    int64_t element_count) -> Value {
      mlir::Type source_element_type = source_type == F32
                                           ? builder.getF32Type()
                                           : ContractionElementType(builder,
                                                                    source_type);
      mlir::Type source_vector_type =
          IsFnuzFp8(source_type)
              ? static_cast<mlir::Type>(
                    builder.getIntegerType(element_count * 8))
              : static_cast<mlir::Type>(
                    mlir::VectorType::get({element_count},
                                          source_element_type));
      mlir::OperationState load_state(entry_function.getLoc(),
                                      "xla_gpu.buffer_load");
      load_state.addOperands(
          {source, physical_linear_index(indices, strides, physical_offset)});
      load_state.addTypes(source_vector_type);
      Value loaded = builder.create(load_state)->getResult(0);
      auto bf16_vector_type =
          mlir::VectorType::get({element_count}, builder.getBF16Type());
      if (source_type == F32) {
        loaded =
            mlir::arith::TruncFOp::create(builder, bf16_vector_type, loaded);
      } else {
        CHECK(IsHalfPrecisionFloat(source_type) || IsFnuzFp8(source_type));
      }
      auto f32_vector_type =
          mlir::VectorType::get({element_count}, builder.getF32Type());
      if (IsFnuzFp8(source_type)) {
        // AMD's arith-to-amdgpu conversion supports scalar FNUZ extension but
        // does not legalize arith.extf directly on a vector of FP8 elements.
        // Keep the packed global transaction and scalarize only the register
        // conversion that follows it.
        llvm::SmallVector<Value> elements;
        elements.reserve(element_count);
        for (int64_t element = 0; element < element_count; ++element) {
          Value bits = loaded;
          if (element > 0) {
            bits = mlir::arith::ShRUIOp::create(
                builder, bits,
                mlir::arith::ConstantIntOp::create(
                    builder, source_vector_type, element * 8));
          }
          bits = mlir::arith::TruncIOp::create(builder, builder.getI8Type(),
                                               bits);
          Value fp8 = mlir::arith::BitcastOp::create(
              builder, source_element_type, bits);
          elements.push_back(mlir::arith::ExtFOp::create(
              builder, builder.getF32Type(), fp8));
        }
        return mlir::vector::FromElementsOp::create(builder, f32_vector_type,
                                                     elements)
            .getResult();
      }
      return mlir::arith::ExtFOp::create(builder, f32_vector_type, loaded)
          .getResult();
    };

    auto emit_vector_matrix_products = [&](Value column, int64_t output_count,
                                           Value k_start, int64_t k_step) {
      llvm::SmallVector<Value> initial_values(output_count, zero);
      mlir::scf::ForOp reduction = mlir::scf::ForOp::create(
          builder, k_start, IndexConstant(builder, k_),
          IndexConstant(builder, k_step), initial_values,
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(reduction.getBody());
        Value k_index = reduction.getInductionVar();
        Value a =
            extract_input(lhs, lhs_input_type_, lhs_requires_linear_addressing_,
                          lhs_physical_offset_, lhs_physical_strides_,
                          lhs_indices(IndexConstant(builder, 0), k_index));
        Value rhs_values;
        if (output_count > 1) {
          rhs_values = extract_input_vector(
              rhs, rhs_input_type_, rhs_physical_offset_, rhs_physical_strides_,
              rhs_indices(k_index, column), output_count);
        }
        llvm::SmallVector<Value> sums;
        sums.reserve(output_count);
        for (int64_t output_offset = 0; output_offset < output_count;
             ++output_offset) {
          Value output_column =
              Add(builder, column, IndexConstant(builder, output_offset));
          Value b =
              output_count > 1
                  ? mlir::vector::ExtractOp::create(
                        builder, rhs_values,
                        llvm::SmallVector<int64_t>{output_offset})
                        .getResult()
                  : extract_input(rhs, rhs_input_type_,
                                  rhs_requires_linear_addressing_,
                                  rhs_physical_offset_, rhs_physical_strides_,
                                  rhs_indices(
                                      k_index,
                                      safe_output_column(output_column)));
          Value product = mlir::arith::MulFOp::create(builder, a, b);
          sums.push_back(mlir::arith::AddFOp::create(
              builder, reduction.getRegionIterArg(output_offset), product));
        }
        mlir::scf::YieldOp::create(builder, sums);
      }
      builder.setInsertionPointAfter(reduction);
      llvm::SmallVector<Value> sums;
      sums.reserve(output_count);
      for (int64_t output_offset = 0; output_offset < output_count;
           ++output_offset) {
        sums.push_back(reduction.getResult(output_offset));
      }
      return sums;
    };

    auto emit_matrix_vector_products = [&](Value row, int64_t output_count) {
      const bool use_vector_k =
          k_vector_width_ > 1 && k_ % (64 * k_vector_width_) == 0;
      auto f32_vector_type = mlir::VectorType::get(
          {k_vector_width_}, builder.getF32Type());
      Value zero_vector = mlir::arith::ConstantOp::create(
          builder, f32_vector_type, builder.getZeroAttr(f32_vector_type));
      llvm::SmallVector<Value> initial_values(
          output_count, use_vector_k ? zero_vector : zero);
      Value k_start =
          use_vector_k
              ? Mul(builder, lane_id, IndexConstant(builder, k_vector_width_))
              : lane_id;
      const int64_t k_step = 64 * (use_vector_k ? k_vector_width_ : 1);
      mlir::scf::ForOp reduction = mlir::scf::ForOp::create(
          builder, k_start, IndexConstant(builder, k_),
          IndexConstant(builder, k_step), initial_values,
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(reduction.getBody());
        Value k_index = reduction.getInductionVar();
        Value b =
            use_vector_k
                ? extract_input_vector(
                      rhs, rhs_input_type_, rhs_physical_offset_,
                      rhs_physical_strides_,
                      rhs_indices(k_index, IndexConstant(builder, 0)),
                      k_vector_width_)
                : extract_input(
                      rhs, rhs_input_type_, rhs_requires_linear_addressing_,
                      rhs_physical_offset_, rhs_physical_strides_,
                      rhs_indices(k_index, IndexConstant(builder, 0)));
        llvm::SmallVector<Value> sums;
        sums.reserve(output_count);
        for (int64_t output_offset = 0; output_offset < output_count;
             ++output_offset) {
          Value output_row =
              Add(builder, row, IndexConstant(builder, output_offset));
          Value input_row = safe_output_row(output_row);
          Value a =
              use_vector_k
                  ? extract_input_vector(
                        lhs, lhs_input_type_, lhs_physical_offset_,
                        lhs_physical_strides_, lhs_indices(input_row, k_index),
                        k_vector_width_)
                  : extract_input(
                        lhs, lhs_input_type_, lhs_requires_linear_addressing_,
                        lhs_physical_offset_, lhs_physical_strides_,
                        lhs_indices(input_row, k_index));
          Value product = mlir::arith::MulFOp::create(builder, a, b);
          sums.push_back(mlir::arith::AddFOp::create(
              builder, reduction.getRegionIterArg(output_offset), product));
        }
        mlir::scf::YieldOp::create(builder, sums);
      }
      builder.setInsertionPointAfter(reduction);
      llvm::SmallVector<Value> sums;
      sums.reserve(output_count);
      for (int64_t output_offset = 0; output_offset < output_count;
           ++output_offset) {
        Value sum = reduction.getResult(output_offset);
        if (use_vector_k) {
          Value horizontal_sum = mlir::vector::ExtractOp::create(
              builder, sum, llvm::SmallVector<int64_t>{0});
          for (int64_t element = 1; element < k_vector_width_; ++element) {
            Value next = mlir::vector::ExtractOp::create(
                builder, sum, llvm::SmallVector<int64_t>{element});
            horizontal_sum =
                mlir::arith::AddFOp::create(builder, horizontal_sum, next);
          }
          sum = horizontal_sum;
        }
        sums.push_back(sum);
      }
      return sums;
    };

    auto emit_k_contiguous_vector_matrix_products =
        [&](Value column, int64_t output_count) {
          const bool use_vector_k =
              k_vector_width_ > 1 && k_ % (64 * k_vector_width_) == 0;
          auto f32_vector_type = mlir::VectorType::get(
              {k_vector_width_}, builder.getF32Type());
          Value zero_vector = mlir::arith::ConstantOp::create(
              builder, f32_vector_type, builder.getZeroAttr(f32_vector_type));
          llvm::SmallVector<Value> initial_values(
              output_count, use_vector_k ? zero_vector : zero);
          Value k_start =
              use_vector_k
                  ? Mul(builder, lane_id,
                        IndexConstant(builder, k_vector_width_))
                  : lane_id;
          const int64_t k_step =
              64 * (use_vector_k ? k_vector_width_ : 1);
          mlir::scf::ForOp reduction = mlir::scf::ForOp::create(
              builder, k_start, IndexConstant(builder, k_),
              IndexConstant(builder, k_step), initial_values,
              [](mlir::OpBuilder&, mlir::Location, Value,
                 mlir::ValueRange) {});
          {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(reduction.getBody());
            Value k_index = reduction.getInductionVar();
            Value a =
                use_vector_k
                    ? extract_input_vector(
                          lhs, lhs_input_type_, lhs_physical_offset_,
                          lhs_physical_strides_,
                          lhs_indices(IndexConstant(builder, 0), k_index),
                          k_vector_width_)
                    : extract_input(
                          lhs, lhs_input_type_,
                          lhs_requires_linear_addressing_,
                          lhs_physical_offset_, lhs_physical_strides_,
                          lhs_indices(IndexConstant(builder, 0), k_index));
            llvm::SmallVector<Value> sums;
            sums.reserve(output_count);
            for (int64_t output_offset = 0; output_offset < output_count;
                 ++output_offset) {
              Value output_column = Add(
                  builder, column, IndexConstant(builder, output_offset));
              Value input_column = safe_output_column(output_column);
              Value b =
                  use_vector_k
                      ? extract_input_vector(
                            rhs, rhs_input_type_, rhs_physical_offset_,
                            rhs_physical_strides_,
                            rhs_indices(k_index, input_column),
                            k_vector_width_)
                      : extract_input(
                            rhs, rhs_input_type_,
                            rhs_requires_linear_addressing_,
                            rhs_physical_offset_, rhs_physical_strides_,
                            rhs_indices(k_index, input_column));
              Value product = mlir::arith::MulFOp::create(builder, a, b);
              sums.push_back(mlir::arith::AddFOp::create(
                  builder, reduction.getRegionIterArg(output_offset), product));
            }
            mlir::scf::YieldOp::create(builder, sums);
          }
          builder.setInsertionPointAfter(reduction);
          llvm::SmallVector<Value> sums;
          sums.reserve(output_count);
          for (int64_t output_offset = 0; output_offset < output_count;
               ++output_offset) {
            Value sum = reduction.getResult(output_offset);
            if (use_vector_k) {
              Value horizontal_sum = mlir::vector::ExtractOp::create(
                  builder, sum, llvm::SmallVector<int64_t>{0});
              for (int64_t element = 1; element < k_vector_width_; ++element) {
                Value next = mlir::vector::ExtractOp::create(
                    builder, sum, llvm::SmallVector<int64_t>{element});
                horizontal_sum =
                    mlir::arith::AddFOp::create(builder, horizontal_sum, next);
              }
              sum = horizontal_sum;
            }
            sums.push_back(sum);
          }
          return sums;
        };

    auto convert_output = [&](Value value, Value row, Value column) {
      if (is_scaled_dot_) {
        value = mlir::arith::MulFOp::create(builder, value, output_scale);
      }
      value = apply_output_epilogue(value, row, column);
      mlir::Type output_type =
          mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType();
      if (output_type.isBF16() || output_type.isF16()) {
        return mlir::arith::TruncFOp::create(builder, output_type, value)
            .getResult();
      }
      return value;
    };
    auto insert_output = [&](Value destination, Value value, Value row,
                             Value column) -> Value {
      if (!has_row_output_tail && !has_column_output_tail) {
        return mlir::tensor::InsertOp::create(
                   builder, value, destination, output_indices(row, column))
            .getResult();
      }
      Value in_bounds;
      if (has_row_output_tail) {
        in_bounds = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, row,
            IndexConstant(builder, m_));
      } else {
        in_bounds = mlir::arith::CmpIOp::create(
            builder, mlir::arith::CmpIPredicate::ult, column,
            IndexConstant(builder, n_));
      }
      mlir::scf::IfOp store = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{destination.getType()}, in_bounds,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard if_guard(builder);
        builder.setInsertionPointToStart(store.thenBlock());
        mlir::scf::YieldOp::create(
            builder, mlir::tensor::InsertOp::create(
                         builder, value, destination,
                         output_indices(row, column))
                         .getResult());
        builder.setInsertionPointToStart(store.elseBlock());
        mlir::scf::YieldOp::create(builder, destination);
      }
      builder.setInsertionPointAfter(store);
      return store.getResult(0);
    };

    if (m_ == 1 && rhs_k_contiguous_gemv_) {
      Value column_base =
          Mul(builder, output_block_id, IndexConstant(builder, block_n_));
      mlir::scf::ForOp columns = mlir::scf::ForOp::create(
          builder,
          Mul(builder, wave_id, IndexConstant(builder, outputs_per_wave_)),
          IndexConstant(builder, block_n_),
          IndexConstant(builder, num_warps_ * outputs_per_wave_),
          mlir::ValueRange{output},
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(columns.getBody());
        Value column = Add(builder, column_base, columns.getInductionVar());
        llvm::SmallVector<Value> sums =
            emit_k_contiguous_vector_matrix_products(column,
                                                      outputs_per_wave_);
        for (Value& sum : sums) {
          for (int32_t distance : {32, 16, 8, 4, 2, 1}) {
            Value shuffled = mlir::gpu::ShuffleOp::create(
                                 builder, sum, distance, /*width=*/64,
                                 mlir::gpu::ShuffleMode::DOWN)
                                 .getShuffleResult();
            sum = mlir::arith::AddFOp::create(builder, sum, shuffled);
          }
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
          Value updated = columns.getRegionIterArg(0);
          for (int64_t output_offset = 0;
               output_offset < outputs_per_wave_; ++output_offset) {
            Value output_column =
                Add(builder, column, IndexConstant(builder, output_offset));
            Value result = convert_output(
                sums[output_offset], IndexConstant(builder, 0), output_column);
            updated = insert_output(updated, result,
                                    IndexConstant(builder, 0), output_column);
          }
          mlir::scf::YieldOp::create(builder, updated);
          builder.setInsertionPointToStart(store.elseBlock());
          mlir::scf::YieldOp::create(builder, columns.getRegionIterArg(0));
        }
        builder.setInsertionPointAfter(store);
        mlir::scf::YieldOp::create(builder, store.getResult(0));
      }
      builder.setInsertionPointAfter(columns);
      mlir::func::ReturnOp::create(builder, columns.getResult(0));
      return absl::OkStatus();
    }

    if (m_ == 1 && split_k_) {
      TF_RET_CHECK(rhs_column_contiguous_ && block_n_ % 64 == 0 &&
                   num_warps_ > 1);
      const int64_t outputs_per_lane = block_n_ / 64;
      auto partials_type = mlir::RankedTensorType::get(
          {num_warps_ * block_n_}, builder.getF32Type());
      Value partials = AllocateSharedOp::create(builder, partials_type);
      Value column_base =
          Mul(builder, output_block_id, IndexConstant(builder, block_n_));
      Value lane_column =
          Add(builder, column_base,
              Mul(builder, lane_id, IndexConstant(builder, outputs_per_lane)));
      llvm::SmallVector<Value> sums = emit_vector_matrix_products(
          lane_column, outputs_per_lane, wave_id, num_warps_);
      Value partial_base =
          Add(builder, Mul(builder, wave_id, IndexConstant(builder, block_n_)),
              Mul(builder, lane_id,
                  IndexConstant(builder, outputs_per_lane)));
      for (int64_t output_offset = 0; output_offset < outputs_per_lane;
           ++output_offset) {
        partials = mlir::tensor::InsertOp::create(
            builder, sums[output_offset], partials,
            mlir::ValueRange{Add(builder, partial_base,
                                 IndexConstant(builder, output_offset))});
      }
      partials = SyncThreadsOp::create(builder, mlir::TypeRange{partials_type},
                                       partials)
                     .getResult(0);

      Value is_wave_zero = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, wave_id,
          IndexConstant(builder, 0));
      mlir::scf::IfOp reduce = mlir::scf::IfOp::create(
          builder, mlir::TypeRange{output.getType()}, is_wave_zero,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard reduce_guard(builder);
        builder.setInsertionPointToStart(reduce.thenBlock());
        Value updated = output;
        Value lane_offset =
            Mul(builder, lane_id, IndexConstant(builder, outputs_per_lane));
        for (int64_t output_offset = 0; output_offset < outputs_per_lane;
             ++output_offset) {
          Value local_column = Add(builder, lane_offset,
                                   IndexConstant(builder, output_offset));
          Value sum = mlir::tensor::ExtractOp::create(
              builder, partials, mlir::ValueRange{local_column});
          for (int64_t source_wave = 1; source_wave < num_warps_;
               ++source_wave) {
            Value partial_index =
                Add(builder, local_column,
                    IndexConstant(builder, source_wave * block_n_));
            Value next = mlir::tensor::ExtractOp::create(
                builder, partials, mlir::ValueRange{partial_index});
            sum = mlir::arith::AddFOp::create(builder, sum, next);
          }
          Value output_column = Add(builder, column_base, local_column);
          updated = insert_output(
              updated,
              convert_output(sum, IndexConstant(builder, 0), output_column),
              IndexConstant(builder, 0), output_column);
        }
        mlir::scf::YieldOp::create(builder, updated);
        builder.setInsertionPointToStart(reduce.elseBlock());
        mlir::scf::YieldOp::create(builder, output);
      }
      builder.setInsertionPointAfter(reduce);
      mlir::func::ReturnOp::create(builder, reduce.getResult(0));
      return absl::OkStatus();
    }

    if (m_ == 1) {
      const int64_t threads = num_warps_ * 64;
      const int64_t outputs_per_thread =
          rhs_column_contiguous_ ? std::max<int64_t>(1, block_n_ / threads) : 1;
      Value column_base =
          Mul(builder, output_block_id, IndexConstant(builder, block_n_));
      mlir::scf::ForOp columns = mlir::scf::ForOp::create(
          builder, thread_id,
          IndexConstant(builder, block_n_ / outputs_per_thread),
          IndexConstant(builder, threads), mlir::ValueRange{output},
          [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(columns.getBody());
        Value column = Add(builder, column_base,
                           Mul(builder, columns.getInductionVar(),
                               IndexConstant(builder, outputs_per_thread)));
        llvm::SmallVector<Value> sums =
            emit_vector_matrix_products(column, outputs_per_thread,
                                        IndexConstant(builder, 0),
                                        /*k_step=*/1);
        Value updated = columns.getRegionIterArg(0);
        for (int64_t output_offset = 0; output_offset < outputs_per_thread;
             ++output_offset) {
          Value output_column =
              Add(builder, column, IndexConstant(builder, output_offset));
          Value result = convert_output(
              sums[output_offset], IndexConstant(builder, 0), output_column);
          updated = insert_output(
              updated, result, IndexConstant(builder, 0), output_column);
        }
        mlir::scf::YieldOp::create(builder, updated);
      }
      builder.setInsertionPointAfter(columns);
      mlir::func::ReturnOp::create(builder, columns.getResult(0));
      return absl::OkStatus();
    }

    TF_RET_CHECK(block_m_ % outputs_per_wave_ == 0 &&
                 num_warps_ * outputs_per_wave_ <= block_m_);
    Value row_base =
        Mul(builder, output_block_id, IndexConstant(builder, block_m_));
    mlir::scf::ForOp rows = mlir::scf::ForOp::create(
        builder,
        Mul(builder, wave_id, IndexConstant(builder, outputs_per_wave_)),
        IndexConstant(builder, block_m_),
        IndexConstant(builder, num_warps_ * outputs_per_wave_),
        mlir::ValueRange{output},
        [](mlir::OpBuilder&, mlir::Location, Value, mlir::ValueRange) {});
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(rows.getBody());
      Value row = Add(builder, row_base, rows.getInductionVar());
      llvm::SmallVector<Value> sums =
          emit_matrix_vector_products(row, outputs_per_wave_);
      for (Value& sum : sums) {
        for (int32_t distance : {32, 16, 8, 4, 2, 1}) {
          Value shuffled = mlir::gpu::ShuffleOp::create(
                               builder, sum, distance, /*width=*/64,
                               mlir::gpu::ShuffleMode::DOWN)
                               .getShuffleResult();
          sum = mlir::arith::AddFOp::create(builder, sum, shuffled);
        }
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
        Value updated = rows.getRegionIterArg(0);
        for (int64_t output_offset = 0;
             output_offset < outputs_per_wave_; ++output_offset) {
          Value output_row =
              Add(builder, row, IndexConstant(builder, output_offset));
          Value result = convert_output(
              sums[output_offset], output_row, IndexConstant(builder, 0));
          updated = insert_output(
              updated, result, output_row, IndexConstant(builder, 0));
        }
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
  bool rank_three_ = false;
  bool global_split_k_ = false;
  bool batched_gemv_ = false;
  bool rhs_column_contiguous_ = false;
  bool rhs_k_contiguous_gemv_ = false;
  int64_t outputs_per_wave_ = 1;
  int64_t k_vector_width_ = 1;
  bool split_k_ = false;
  int64_t batch_count_ = 1;
  int64_t blocks_per_batch_ = 1;
  int64_t fusion_parameter_count_ = 0;
  int64_t lhs_parameter_number_ = 0;
  int64_t rhs_parameter_number_ = 1;
  bool is_scaled_dot_ = false;
  int64_t lhs_scale_parameter_number_ = 2;
  int64_t rhs_scale_parameter_number_ = 3;
  int64_t epilogue_row_dimension_ = 0;
  int64_t epilogue_column_dimension_ = 1;
  std::vector<EpilogueStep> epilogue_steps_;
  PrimitiveType lhs_input_type_ = BF16;
  PrimitiveType rhs_input_type_ = BF16;
  bool lhs_requires_linear_addressing_ = false;
  bool rhs_requires_linear_addressing_ = false;
  int64_t lhs_physical_offset_ = 0;
  int64_t rhs_physical_offset_ = 0;
  std::vector<int64_t> lhs_physical_strides_;
  std::vector<int64_t> rhs_physical_strides_;
  std::vector<int64_t> lhs_dynamic_offset_parameter_numbers_;
  std::vector<int64_t> lhs_dynamic_offset_constants_;
  std::vector<int64_t> lhs_dynamic_offset_limits_;
  std::vector<int64_t> rhs_dynamic_offset_parameter_numbers_;
  std::vector<int64_t> rhs_dynamic_offset_constants_;
  std::vector<int64_t> rhs_dynamic_offset_limits_;
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
